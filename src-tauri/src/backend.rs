//! Python backend lifecycle: spawn, health check, shutdown, HTTP client.
//!
//! Production sidecar spawn flows through the Tauri shell plugin
//! (`app.shell().command(...)`), which validates the path + args against
//! the named-binary capability scope in `capabilities/default.json`. The
//! shell plugin's `CommandChild` does not expose `try_wait()`, so liveness
//! is tracked via a background task that drains `Receiver<CommandEvent>`
//! and flips an `AtomicBool` on `Terminated` (see `BackendProcess`).
//!
//! Dev mode (no frozen sidecar at the resolved `resource_dir()` path)
//! falls back to `std::process::Command` spawning the system Python
//! interpreter against `backend/app.main:app`. This branch is intentionally
//! NOT routed through the shell plugin — it has no fixed scope path and
//! is debug-only.

use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

#[cfg(windows)]
use std::os::windows::process::CommandExt;

use tauri::{AppHandle, Manager};
use tauri_plugin_shell::process::{CommandChild, CommandEvent};
use tauri_plugin_shell::ShellExt;

/// Windows CREATE_NO_WINDOW flag (0x08000000) — used only by the dev-mode
/// fallback (std::process::Command). The shell plugin's `Command` builder
/// does NOT expose `creation_flags`; for the frozen-sidecar production
/// path the console window is suppressed by building the sidecar with
/// `console=False` (see `backend/build_sidecar.spec`) plus the Python
/// entrypoint redirecting stdout/stderr to `~/.justsay/logs/sidecar.log`.
#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x08000000;

pub const PORT: u16 = 9377;

const REQUEST_TIMEOUT: Duration = Duration::from_secs(120);
const HEALTH_POLL_INTERVAL: Duration = Duration::from_millis(300);
const HEALTH_POLL_MAX_ATTEMPTS: u32 = 100; // 30 seconds

/// Production sidecar handle: shell-plugin child + liveness flag updated
/// by a background event-drain task. `try_wait()` is not available on
/// `CommandChild`, so this struct is the substitute.
struct ShellBackend {
    child: CommandChild,
    alive: Arc<AtomicBool>,
}

/// Either the shell-plugin-spawned sidecar (production) or a raw child
/// process from `std::process::Command` (dev mode). Stored in a `Mutex`
/// so spawn/shutdown can mutate it from different threads safely.
enum BackendProcess {
    Sidecar(ShellBackend),
    Dev(Child),
}

static BACKEND_PROCESS: Mutex<Option<BackendProcess>> = Mutex::new(None);

/// Append a stderr line from the production sidecar to
/// `~/.justsay/logs/sidecar.log`. Failure to open the log file is silent
/// to avoid spamming on shutdown when the FS is racing.
fn append_sidecar_log(line: &[u8]) {
    let home = match std::env::var("USERPROFILE").or_else(|_| std::env::var("HOME")) {
        Ok(h) => h,
        Err(_) => return,
    };
    let log_dir = PathBuf::from(home).join(".justsay").join("logs");
    if std::fs::create_dir_all(&log_dir).is_err() {
        return;
    }
    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_dir.join("sidecar.log"))
    {
        use std::io::Write;
        let _ = file.write_all(line);
        if !line.ends_with(b"\n") {
            let _ = file.write_all(b"\n");
        }
    }
}

/// Find Python executable on the system. Kept for the dev-mode fallback;
/// the production path resolves the sidecar via `app.path().resource_dir()`.
fn find_python() -> Result<String, String> {
    for candidate in ["python", "python3"] {
        let result = Command::new(candidate)
            .args(["--version"])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
        if let Ok(status) = result {
            if status.success() {
                return Ok(candidate.to_string());
            }
        }
    }
    Err("Python not found. Install Python 3.10+ and ensure it's in PATH.".to_string())
}

/// Resolve the backend directory path (dev mode only).
fn find_backend_dir() -> Result<PathBuf, String> {
    let candidates: Vec<PathBuf> = vec![
        std::env::current_dir().map(|p| p.join("backend")).unwrap_or_default(),
        std::env::current_dir().map(|p| p.join("..").join("backend")).unwrap_or_default(),
        std::env::current_exe()
            .ok()
            .and_then(|e| e.parent().map(|p| p.join("backend")))
            .unwrap_or_default(),
    ];

    for candidate in &candidates {
        if candidate.join("app").join("main.py").exists() {
            return candidate.canonicalize().map_err(|e| e.to_string());
        }
    }

    Err("Backend directory not found. Expected 'backend/app/main.py'.".to_string())
}

/// Resolve the production sidecar path inside the installed resource dir.
/// Returns `None` if no frozen sidecar is present (developer setup).
///
/// The resolved path is the same one the capability scope's `$RESOURCE`
/// token expands to: on Windows NSIS it is `<install>/resources/`, on
/// macOS bundles it is `Contents/Resources/`. Tauri's path resolver is the
/// single source of truth for both production runtime and capability
/// scope, so they agree by construction.
fn resolve_sidecar(app: &AppHandle) -> Option<PathBuf> {
    let resource_dir = app.path().resource_dir().ok()?;
    let name = if cfg!(windows) {
        "justsay-backend.exe"
    } else {
        "justsay-backend"
    };
    let candidate = resource_dir.join("justsay-backend").join(name);
    if candidate.exists() {
        Some(candidate)
    } else {
        None
    }
}

/// Check if the port is available before spawning.
fn check_port_available() -> Result<(), String> {
    match std::net::TcpListener::bind(format!("127.0.0.1:{}", PORT)) {
        Ok(_listener) => Ok(()),
        Err(_) => Err(format!(
            "Port {} is already in use. Another JustSay instance may be running.",
            PORT
        )),
    }
}

/// Spawn the Python FastAPI backend as a child process.
///
/// Preference order:
///   1. Production sidecar via shell plugin (capability-scoped).
///   2. System Python + the backend source tree (developer setup).
pub fn spawn(app: AppHandle) -> Result<(), String> {
    check_port_available()?;

    let backend = if let Some(sidecar) = resolve_sidecar(&app) {
        log::info!("Starting backend sidecar via shell plugin: {:?}", sidecar);
        let sidecar_str = sidecar.to_string_lossy().to_string();
        let port_str = PORT.to_string();
        let (mut rx, child) = app
            .shell()
            .command(sidecar_str)
            .args(["--host", "127.0.0.1", "--port", &port_str])
            .spawn()
            .map_err(|e| format!("Failed to start backend sidecar: {}", e))?;

        let alive = Arc::new(AtomicBool::new(true));
        let alive_clone = alive.clone();

        // Drain the event stream: forward stderr to the log file and flip
        // `alive` to false when the process terminates (or the channel
        // closes, which means the same thing in practice).
        tauri::async_runtime::spawn(async move {
            while let Some(event) = rx.recv().await {
                match event {
                    CommandEvent::Stderr(bytes) => append_sidecar_log(&bytes),
                    CommandEvent::Stdout(bytes) => append_sidecar_log(&bytes),
                    CommandEvent::Error(msg) => {
                        append_sidecar_log(format!("[shell error] {}", msg).as_bytes());
                        alive_clone.store(false, Ordering::Release);
                        break;
                    }
                    CommandEvent::Terminated(payload) => {
                        let line = format!(
                            "[terminated] code={:?} signal={:?}",
                            payload.code, payload.signal
                        );
                        append_sidecar_log(line.as_bytes());
                        alive_clone.store(false, Ordering::Release);
                        break;
                    }
                    _ => {}
                }
            }
            // Channel closed without explicit Terminated — treat as dead.
            alive_clone.store(false, Ordering::Release);
        });

        BackendProcess::Sidecar(ShellBackend { child, alive })
    } else {
        let python = find_python()?;
        let backend_dir = find_backend_dir()?;
        log::info!(
            "Starting backend (dev mode): {} -m uvicorn (dir: {:?})",
            python,
            backend_dir
        );
        let mut cmd = Command::new(&python);
        cmd.args([
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            &PORT.to_string(),
            "--log-level",
            "warning",
        ])
        .current_dir(&backend_dir)
        .stdout(Stdio::null())
        .stderr(Stdio::null());
        #[cfg(windows)]
        cmd.creation_flags(CREATE_NO_WINDOW);
        let child = cmd
            .spawn()
            .map_err(|e| format!("Failed to start backend: {}", e))?;
        BackendProcess::Dev(child)
    };

    let mut guard = BACKEND_PROCESS.lock().map_err(|e| e.to_string())?;
    *guard = Some(backend);

    Ok(())
}

/// Check if the child process has exited unexpectedly.
fn is_process_alive() -> bool {
    let mut guard = match BACKEND_PROCESS.lock() {
        Ok(g) => g,
        Err(_) => return false,
    };
    match guard.as_mut() {
        Some(BackendProcess::Sidecar(s)) => s.alive.load(Ordering::Acquire),
        Some(BackendProcess::Dev(child)) => match child.try_wait() {
            Ok(None) => true,
            Ok(Some(status)) => {
                log::error!("Backend (dev) exited with: {}", status);
                false
            }
            Err(e) => {
                log::warn!("try_wait error (assuming alive): {}", e);
                true
            }
        },
        None => false,
    }
}

/// Poll /health until the backend responds or timeout.
pub async fn wait_for_ready() -> Result<(), String> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_millis(500))
        .build()
        .map_err(|e| e.to_string())?;

    let url = format!("http://127.0.0.1:{}/health", PORT);

    for attempt in 1..=HEALTH_POLL_MAX_ATTEMPTS {
        if !is_process_alive() {
            return Err(
                "Backend process exited unexpectedly. Check Python dependencies.".to_string(),
            );
        }

        match client.get(&url).send().await {
            Ok(resp) if resp.status().is_success() => {
                log::info!("Backend ready (attempt {})", attempt);
                return Ok(());
            }
            _ => {
                tokio::time::sleep(HEALTH_POLL_INTERVAL).await;
            }
        }
    }

    Err(
        "Backend failed to start within 30 seconds. Check Python installation and dependencies."
            .to_string(),
    )
}

/// Kill the backend process on shutdown.
pub fn shutdown() {
    let mut guard = match BACKEND_PROCESS.lock() {
        Ok(g) => g,
        Err(_) => return,
    };

    if let Some(backend) = guard.take() {
        match backend {
            BackendProcess::Sidecar(s) => {
                let pid = s.child.pid();
                log::info!("Shutting down backend sidecar (PID: {})", pid);
                // CommandChild::kill() consumes self; the event-drain
                // task will see the channel close and flip `alive`.
                let _ = s.child.kill();
            }
            BackendProcess::Dev(mut child) => {
                let pid = child.id();
                log::info!("Shutting down backend (dev, PID: {})", pid);

                #[cfg(target_os = "windows")]
                {
                    let _ = Command::new("taskkill")
                        .args(["/T", "/F", "/PID", &pid.to_string()])
                        .stdout(Stdio::null())
                        .stderr(Stdio::null())
                        .status();
                }

                #[cfg(not(target_os = "windows"))]
                {
                    let _ = child.kill();
                }

                let _ = child.wait();
            }
        }
    }
}

/// HTTP request to the backend.
pub async fn request(
    method: &str,
    path: &str,
    body: Option<&str>,
) -> Result<String, Box<dyn std::error::Error>> {
    let url = format!("http://127.0.0.1:{}{}", PORT, path);

    let client = reqwest::Client::builder()
        .timeout(REQUEST_TIMEOUT)
        .build()?;

    let response = match method.to_uppercase().as_str() {
        "GET" => client.get(&url).send().await?,
        "POST" => {
            let mut req = client.post(&url);
            if let Some(b) = body {
                req = req
                    .header("Content-Type", "application/json")
                    .body(b.to_string());
            }
            req.send().await?
        }
        "PUT" => {
            let mut req = client.put(&url);
            if let Some(b) = body {
                req = req
                    .header("Content-Type", "application/json")
                    .body(b.to_string());
            }
            req.send().await?
        }
        _ => return Err(format!("Unsupported method: {}", method).into()),
    };

    let status = response.status();
    let text = response.text().await?;

    if !status.is_success() {
        return Err(format!("HTTP {}: {}", status.as_u16(), text).into());
    }

    Ok(text)
}
