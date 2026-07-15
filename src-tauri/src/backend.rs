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
use std::sync::{Arc, Mutex, OnceLock};
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

/// Shared `reqwest::Client` for all backend HTTP calls (IPC-triggered
/// requests + the startup readiness poll). A `Client` owns a connection
/// pool; building a fresh one per call (the previous behavior) discarded
/// that pool on every single frontend→backend round trip.
///
/// Stores the build `Result` itself (rather than the `Client` directly)
/// because `OnceLock::get_or_try_init` is not stable at this crate's MSRV
/// (`rust-version = "1.77.2"` in Cargo.toml) — it is still gated behind the
/// nightly-only `once_cell_try` feature. This gives the same "build once,
/// cache the outcome" behavior via the stable `get_or_init`.
static HTTP_CLIENT: OnceLock<Result<reqwest::Client, String>> = OnceLock::new();

/// Lazily build (once) and return the shared HTTP client, using
/// `REQUEST_TIMEOUT` as its default per-request timeout. Callers that need
/// a different timeout (e.g. `wait_for_ready()`'s faster health poll)
/// override it per-request via `RequestBuilder::timeout(...)` rather than
/// building a second client.
///
/// Returns `Err` instead of panicking if the client fails to build (e.g. a
/// broken TLS backend) — matches the pre-refactor behavior where `request()`
/// and `wait_for_ready()` each built their own `Client` and propagated
/// build failure as a normal `Result` error.
fn http_client() -> Result<&'static reqwest::Client, String> {
    HTTP_CLIENT
        .get_or_init(|| {
            reqwest::Client::builder()
                .timeout(REQUEST_TIMEOUT)
                .build()
                .map_err(|e| format!("failed to build shared reqwest client: {}", e))
        })
        .as_ref()
        .map_err(|e| e.clone())
}

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

/// Parse `python --version` stdout (e.g. `"Python 3.11.4\n"`) into
/// `(major, minor)`. Returns `None` for non-standard output (some Windows
/// Store aliases, custom builds) rather than panicking — callers treat
/// that the same as "candidate not usable".
fn parse_python_version(output: &str) -> Option<(u32, u32)> {
    let rest = output.trim().strip_prefix("Python ")?;
    let mut parts = rest.split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next()?.parse().ok()?;
    Some((major, minor))
}

/// Find a Python executable on the system that satisfies JustSay's minimum
/// supported version (3.10+). Kept for the dev-mode fallback; the
/// production path resolves the sidecar via `app.path().resource_dir()`.
fn find_python() -> Result<String, String> {
    let mut too_old: Option<(String, u32, u32)> = None;

    for candidate in ["python", "python3"] {
        let mut cmd = Command::new(candidate);
        cmd.args(["--version"]).stdout(Stdio::piped()).stderr(Stdio::piped());
        #[cfg(windows)]
        cmd.creation_flags(CREATE_NO_WINDOW);
        let result = cmd.output();

        if let Ok(output) = result {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                if let Some((major, minor)) = parse_python_version(&stdout) {
                    if (major, minor) >= (3, 10) {
                        return Ok(candidate.to_string());
                    }
                    if too_old.is_none() {
                        too_old = Some((candidate.to_string(), major, minor));
                    }
                }
            }
        }
    }

    if let Some((candidate, major, minor)) = too_old {
        return Err(format!(
            "Found {} {}.{}, but JustSay requires Python 3.10+. Install a newer Python and ensure it's first in PATH.",
            candidate, major, minor
        ));
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
///
/// On Windows, the previous `tauri dev` / installed-app session may have
/// orphaned its sidecar — `tauri-plugin-shell` does not create a Job
/// Object, so a parent crash or a failed in-process `CommandChild::kill()`
/// can leave `justsay-backend.exe` running and squatting on 9377. If we
/// detect an orphan that's clearly ours, reap it and retry instead of
/// punishing the user with a startup error.
fn check_port_available() -> Result<(), String> {
    if try_bind_port() {
        return Ok(());
    }

    #[cfg(target_os = "windows")]
    {
        if reap_orphan_sidecar() {
            // Windows takes a moment to release a closed socket; poll
            // briefly rather than rely on a fixed sleep.
            for _ in 0..20 {
                std::thread::sleep(Duration::from_millis(100));
                if try_bind_port() {
                    log::info!("Reaped orphan sidecar; port {} freed", PORT);
                    return Ok(());
                }
            }
        }
    }

    Err(format!(
        "Port {} is already in use. Another JustSay instance may be running.",
        PORT
    ))
}

fn try_bind_port() -> bool {
    std::net::TcpListener::bind(format!("127.0.0.1:{}", PORT)).is_ok()
}

/// On Windows: if `justsay-backend.exe` (our PyInstaller sidecar) is
/// running, kill it with `taskkill /F /T`. Returns true when at least one
/// such process was found and asked to terminate. Conservative — only
/// matches by the exact image name we ship, never anything else.
#[cfg(target_os = "windows")]
fn reap_orphan_sidecar() -> bool {
    use std::os::windows::process::CommandExt;
    let listing = Command::new("tasklist")
        .args(["/FI", "IMAGENAME eq justsay-backend.exe", "/NH"])
        .creation_flags(CREATE_NO_WINDOW)
        .output();
    let stdout = match listing {
        Ok(o) => o.stdout,
        Err(_) => return false,
    };
    // `tasklist` with no match prints a localised "INFO:" line on stderr
    // and an empty (or "No tasks") stdout — checking for our image name
    // in stdout is the only locale-independent signal.
    let listed = String::from_utf8_lossy(&stdout);
    if !listed.to_lowercase().contains("justsay-backend.exe") {
        return false;
    }
    log::warn!("Found orphan justsay-backend.exe — reaping before spawn");
    let _ = Command::new("taskkill")
        .args(["/F", "/T", "/IM", "justsay-backend.exe"])
        .creation_flags(CREATE_NO_WINDOW)
        .status();
    true
}

/// Spawn the Python FastAPI backend as a child process.
///
/// Preference order:
///   1. Production sidecar via shell plugin (capability-scoped).
///   2. System Python + the backend source tree (developer setup).
///
/// Debug builds (`tauri dev`) skip the frozen sidecar entirely so the
/// developer always sees their current Python source — no PyInstaller
/// rebuild needed between edits. To force the frozen path in dev (e.g.,
/// to smoke-test the bundled binary end-to-end), set
/// `JUSTSAY_USE_FROZEN_SIDECAR=1` before launching.
pub fn spawn(app: AppHandle) -> Result<(), String> {
    check_port_available()?;

    let prefer_python_source =
        cfg!(debug_assertions) && std::env::var("JUSTSAY_USE_FROZEN_SIDECAR").is_err();

    let resolved_sidecar = if prefer_python_source {
        None
    } else {
        resolve_sidecar(&app)
    };

    let backend = if let Some(sidecar) = resolved_sidecar {
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
    let url = format!("http://127.0.0.1:{}/health", PORT);
    let client = http_client()?;

    for attempt in 1..=HEALTH_POLL_MAX_ATTEMPTS {
        if !is_process_alive() {
            return Err(
                "Backend process exited unexpectedly. Check Python dependencies.".to_string(),
            );
        }

        match client
            .get(&url)
            .timeout(Duration::from_millis(500))
            .send()
            .await
        {
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
                // Windows: tree-kill via taskkill. `CommandChild::kill()`
                // only sends `TerminateProcess` to the direct child, so a
                // sidecar that spawned subprocesses (e.g. FFmpeg) would
                // orphan them. `/T` walks the descendant tree, `/F`
                // forces termination — same path the dev branch uses.
                #[cfg(target_os = "windows")]
                {
                    use std::os::windows::process::CommandExt;
                    let taskkill_status = Command::new("taskkill")
                        .args(["/T", "/F", "/PID", &pid.to_string()])
                        .stdout(Stdio::null())
                        .stderr(Stdio::null())
                        .creation_flags(CREATE_NO_WINDOW)
                        .status();
                    let taskkilled =
                        matches!(&taskkill_status, Ok(st) if st.success());
                    if taskkilled {
                        log::info!("Sidecar PID {} terminated via taskkill", pid);
                    } else {
                        log::warn!(
                            "taskkill failed for PID {} (status: {:?}); \
                             falling back to CommandChild::kill()",
                            pid,
                            taskkill_status
                        );
                        if let Err(e) = s.child.kill() {
                            log::warn!("Fallback kill also failed: {}", e);
                        }
                    }
                }
                #[cfg(not(target_os = "windows"))]
                {
                    if let Err(e) = s.child.kill() {
                        log::warn!("Sidecar kill failed for PID {}: {}", pid, e);
                    } else {
                        log::info!("Sidecar PID {} terminated", pid);
                    }
                }
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

    let client = http_client()?;

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
