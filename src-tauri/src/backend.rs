//! Python backend lifecycle: spawn, health check, shutdown, HTTP client.

use std::net::TcpListener;
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::time::Duration;

#[cfg(windows)]
use std::os::windows::process::CommandExt;

/// Windows CREATE_NO_WINDOW flag (0x08000000) — suppresses the console window
/// that would otherwise flash when spawning a CONSOLE-subsystem binary.
/// The frozen sidecar keeps its console handle so manual launches show stdout,
/// but Tauri must not surface that window to end users.
#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x08000000;

pub const PORT: u16 = 9377;

const REQUEST_TIMEOUT: Duration = Duration::from_secs(120);
const HEALTH_POLL_INTERVAL: Duration = Duration::from_millis(300);
const HEALTH_POLL_MAX_ATTEMPTS: u32 = 100; // 30 seconds — covers --onefile extraction overhead

static BACKEND_PROCESS: Mutex<Option<Child>> = Mutex::new(None);

/// Check if the port is available before spawning.
fn check_port_available() -> Result<(), String> {
    match TcpListener::bind(format!("127.0.0.1:{}", PORT)) {
        Ok(_listener) => Ok(()), // port is free, listener drops and releases it
        Err(_) => Err(format!(
            "Port {} is already in use. Another JustSay instance may be running.",
            PORT
        )),
    }
}

/// Find Python executable on the system.
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

/// Look for a PyInstaller-frozen sidecar binary next to the Tauri executable.
///
/// The CI release workflow (Plan 008) ships a `justsay-backend(.exe)` next to the
/// app binary. When that binary is present we prefer it — the user does not need
/// a system Python install. When it is absent we fall back to spawning system
/// Python (existing dev-mode behaviour).
fn find_sidecar() -> Option<std::path::PathBuf> {
    let exe_dir = std::env::current_exe().ok()?.parent()?.to_path_buf();
    let name = if cfg!(windows) {
        "justsay-backend.exe"
    } else {
        "justsay-backend"
    };

    // Tauri externalBin places the sidecar alongside the main exe on both platforms:
    // Windows: <install_dir>/justsay-backend.exe
    // macOS:   JustSay.app/Contents/MacOS/justsay-backend
    let candidates = [
        exe_dir.join(name),
        exe_dir.join("..").join("Resources").join(name),
    ];
    candidates.into_iter().find(|p| p.exists())
}

/// Resolve the backend directory path.
/// Searches: CWD/backend, CWD/../backend (for src-tauri/), next to exe.
fn find_backend_dir() -> Result<std::path::PathBuf, String> {
    let candidates: Vec<std::path::PathBuf> = vec![
        // CWD/backend (running from project root)
        std::env::current_dir().map(|p| p.join("backend")).unwrap_or_default(),
        // CWD/../backend (running from src-tauri/ during cargo run)
        std::env::current_dir().map(|p| p.join("..").join("backend")).unwrap_or_default(),
        // Next to executable (production)
        std::env::current_exe()
            .ok()
            .and_then(|e| e.parent().map(|p| p.join("backend")))
            .unwrap_or_default(),
    ];

    for candidate in &candidates {
        if candidate.join("app").join("main.py").exists() {
            // Canonicalize to resolve ".." in path
            return candidate.canonicalize().map_err(|e| e.to_string());
        }
    }

    Err("Backend directory not found. Expected 'backend/app/main.py'.".to_string())
}

/// Spawn the Python FastAPI backend as a child process.
///
/// Preference order:
///   1. Frozen PyInstaller sidecar next to the Tauri executable (production).
///   2. System Python + the backend source tree (developer setup).
pub fn spawn() -> Result<(), String> {
    check_port_available()?;

    let child = if let Some(sidecar) = find_sidecar() {
        log::info!("Starting backend sidecar: {:?}", sidecar);
        let mut cmd = Command::new(&sidecar);
        cmd.args([
            "--host",
            "127.0.0.1",
            "--port",
            &PORT.to_string(),
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::null());
        #[cfg(windows)]
        cmd.creation_flags(CREATE_NO_WINDOW);
        cmd.spawn()
            .map_err(|e| format!("Failed to start backend sidecar: {}", e))?
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
        cmd.spawn()
            .map_err(|e| format!("Failed to start backend: {}", e))?
    };

    let mut guard = BACKEND_PROCESS.lock().map_err(|e| e.to_string())?;
    *guard = Some(child);

    Ok(())
}

/// Check if the child process has exited unexpectedly.
fn is_process_alive() -> bool {
    if let Ok(mut guard) = BACKEND_PROCESS.lock() {
        if let Some(ref mut child) = *guard {
            // try_wait returns Ok(Some(status)) if exited, Ok(None) if still running
            match child.try_wait() {
                Ok(None) => return true,  // still running
                Ok(Some(status)) => {
                    log::error!("Backend process exited with: {}", status);
                    return false;
                }
                Err(e) => {
                    log::warn!("try_wait error (assuming alive): {}", e);
                    return true;
                }
            }
        }
    }
    false
}

/// Poll /health until the backend responds or timeout.
pub async fn wait_for_ready() -> Result<(), String> {
    // Short per-request timeout so Connection-refused returns immediately;
    // the 30-second budget (100 × 300ms) stays accurate.
    let client = reqwest::Client::builder()
        .timeout(Duration::from_millis(500))
        .build()
        .map_err(|e| e.to_string())?;

    let url = format!("http://127.0.0.1:{}/health", PORT);

    for attempt in 1..=HEALTH_POLL_MAX_ATTEMPTS {
        // Check if process died before polling
        if !is_process_alive() {
            return Err("Backend process exited unexpectedly. Check Python dependencies.".to_string());
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

/// Kill the backend process tree on shutdown.
pub fn shutdown() {
    if let Ok(mut guard) = BACKEND_PROCESS.lock() {
        if let Some(ref mut child) = *guard {
            let pid = child.id();
            log::info!("Shutting down backend (PID: {})", pid);

            // On Windows, kill the entire process tree to avoid orphan uvicorn workers
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
        *guard = None;
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
