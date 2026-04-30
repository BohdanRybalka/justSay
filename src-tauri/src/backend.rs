//! Python backend lifecycle: spawn, health check, shutdown, HTTP client.

use std::net::TcpListener;
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::time::Duration;

pub const PORT: u16 = 9377;

const REQUEST_TIMEOUT: Duration = Duration::from_secs(120);
const HEALTH_POLL_INTERVAL: Duration = Duration::from_millis(300);
const HEALTH_POLL_MAX_ATTEMPTS: u32 = 30; // 9 seconds total

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
pub fn spawn() -> Result<(), String> {
    check_port_available()?;

    let python = find_python()?;
    let backend_dir = find_backend_dir()?;

    log::info!(
        "Starting backend: {} -m uvicorn (dir: {:?})",
        python,
        backend_dir
    );

    let child = Command::new(&python)
        .args([
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
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| format!("Failed to start backend: {}", e))?;

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
                    log::error!("Failed to check backend process: {}", e);
                    return false;
                }
            }
        }
    }
    false
}

/// Poll /health until the backend responds or timeout.
pub async fn wait_for_ready() -> Result<(), String> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
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
        "Backend failed to start within 9 seconds. Check Python installation and dependencies."
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
