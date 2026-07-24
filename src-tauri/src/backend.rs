//! Python backend lifecycle: spawn, health check, shutdown, HTTP client,
//! crash-respawn watchdog.
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
//! NOT routed through the shell plugin — it has no fixed scope path. It is
//! the debug default, but is also reached in a release build whenever
//! `resolve_sidecar()` finds no frozen sidecar.
//!
//! `spawn_watchdog()` polls `is_process_alive()` and respawns the backend
//! on an unexpected crash, bounded by `MAX_RESPAWN_ATTEMPTS` with
//! exponential backoff — see `docs/adr/006-backend-watchdog-respawn-on-crash.md`
//! for the `SHUTDOWN_REQUESTED` race-closure rationale.

use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

#[cfg(windows)]
use std::os::windows::io::AsRawHandle;
#[cfg(windows)]
use std::os::windows::process::CommandExt;

#[cfg(windows)]
use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
#[cfg(windows)]
use windows_sys::Win32::System::JobObjects::{
    AssignProcessToJobObject, CreateJobObjectW, SetInformationJobObject,
    JobObjectExtendedLimitInformation, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
};
#[cfg(windows)]
use windows_sys::Win32::System::Threading::{OpenProcess, PROCESS_SET_QUOTA, PROCESS_TERMINATE};

use tauri::{AppHandle, Manager};
use tauri_plugin_shell::process::{CommandChild, CommandEvent};
use tauri_plugin_shell::ShellExt;

/// Windows CREATE_NO_WINDOW flag (0x08000000) — suppresses a console-window
/// flash for short-lived helper spawns via `std::process::Command`: the
/// Python version probe (`find_python`), the orphan-sidecar
/// `tasklist`/`taskkill` (`reap_orphan_sidecar`), the production
/// sidecar's `taskkill` graceful-stop fallback, and the `spawn_throwaway_child`
/// test helper. On the dev backend spawn it is set conditionally: omitted in a
/// debug build so the child inherits the dev parent's shared console and a later
/// `GenerateConsoleCtrlEvent(CTRL_BREAK_EVENT, pid)` reaches it, but kept on the
/// release fallback (the Dev branch is reachable in a release build with no
/// frozen sidecar, where the windows-subsystem parent owns no console to inherit)
/// so the child gets no fresh visible window — see the dev spawn site below and
/// `docs/adr/024-dev-backend-never-receives-ctrl-break.md`.
/// The shell plugin's `Command` builder does NOT expose `creation_flags`;
/// for the frozen-sidecar production path the console window is suppressed
/// by building the sidecar with `console=False` (see
/// `backend/build_sidecar.spec`) plus the Python entrypoint redirecting
/// stdout/stderr to `~/<data_dir_name>/logs/sidecar.log`, where
/// `data_dir_name` is `.justsay` or `.justsay-dev` depending on `spawn()`'s
/// `force_dev_data_dir` flag — see `append_sidecar_log()` below and
/// `docs/adr/012-dev-mode-data-directory-isolation.md`.
#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x08000000;

/// Windows CREATE_NEW_PROCESS_GROUP flag (0x00000200) — dev-mode only.
/// Required so a later `GenerateConsoleCtrlEvent(CTRL_BREAK_EVENT, pid)`
/// targets only the child's own process group, not the calling Tauri
/// process's group (which would also receive the event and tear itself
/// down). The shell plugin's `Command` builder exposes no way to set this
/// flag, so the production `Sidecar` path cannot use it — see
/// `docs/adr/004-windows-graceful-backend-stop.md`.
#[cfg(windows)]
const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;

#[cfg(windows)]
const CTRL_BREAK_EVENT: u32 = 1;

/// Console control event codes delivered to `SetConsoleCtrlHandler` callbacks
/// for a raw Ctrl+C keypress and a console-window close, respectively — the
/// two events `install_ctrl_handler()` intercepts below.
#[cfg(windows)]
const CTRL_C_EVENT: u32 = 0;
#[cfg(windows)]
const CTRL_CLOSE_EVENT: u32 = 2;

#[cfg(windows)]
extern "system" {
    fn GenerateConsoleCtrlEvent(dwCtrlEvent: u32, dwProcessGroupId: u32) -> i32;
    fn SetConsoleCtrlHandler(
        handler_routine: Option<unsafe extern "system" fn(u32) -> i32>,
        add: i32,
    ) -> i32;
}

/// Console control handler registered on the Tauri parent process.
///
/// `spawn()`'s Dev branch sets `CREATE_NEW_PROCESS_GROUP` on the child so
/// `terminate_gracefully()` can target it alone with `CTRL_BREAK_EVENT` (see
/// that constant's doc). A side effect of that flag: the child no longer
/// receives a broadcast `CTRL_C_EVENT` from the parent's console the way it
/// did before this existed — a raw Ctrl+C in the dev terminal only reaches
/// the parent. The OS's *default* handling of an unhandled `CTRL_C_EVENT` /
/// `CTRL_CLOSE_EVENT` is to terminate the parent immediately, which does
/// NOT emit Tauri's `RunEvent::Exit` (that's a normal-exit path, not a
/// console-event path) and does NOT trigger the panic hook (no Rust panic
/// occurred) — so without this handler, the backend child would be orphaned
/// instead of cleaned up.
///
/// Registered unconditionally (both Dev and release builds): `shutdown()`
/// is already idempotent and branch-agnostic (it no-ops if nothing is
/// running, and handles both `Dev` and `Sidecar` correctly), so calling it
/// here is never wrong — on a release build with no attached console (the
/// common case, since `windows_subsystem = "windows"` suppresses one), the
/// registration succeeds but the events are simply never generated.
#[cfg(windows)]
unsafe extern "system" fn console_ctrl_handler(ctrl_type: u32) -> i32 {
    if ctrl_type == CTRL_C_EVENT || ctrl_type == CTRL_CLOSE_EVENT {
        wait_for_inflight_shutdown_then_exit();
    }
    0
}

/// Bounded wait for any already-in-progress `shutdown()` call (e.g.
/// `RunEvent::Exit`'s `terminate_gracefully()`, mid-poll on the main thread
/// for up to ~3s while holding `BACKEND_PROCESS`) to finish, before running
/// our own (by then idempotent, likely-no-op) `shutdown()` and exiting.
///
/// `shutdown()`'s `try_lock()` was only ever designed to make *same-thread*
/// reentrancy (this handler's own thread, or the panic hook, calling
/// `shutdown()` while already inside it) a safe no-op — it has no way to
/// tell "a *different* thread already holds the lock and is mid-cleanup"
/// from "nothing to clean up." The original console handler called
/// `shutdown()` then unconditionally `process::exit(0)`; if a Ctrl+C landed
/// while another thread already held `BACKEND_PROCESS`, `shutdown()` would
/// silently no-op on contention and the immediate `exit(0)` would kill the
/// whole process out from under that other thread's in-flight graceful
/// sequence — potentially before it reached `force_kill()` — re-orphaning
/// the child via a narrower trigger (Ctrl+C during an already-in-progress
/// shutdown) than the one this handler originally fixed.
///
/// Polling for the lock to free up first closes that window: this handler
/// runs on its own dedicated OS thread (never the main thread or the
/// panic-hook's thread — see `install_ctrl_handler`'s doc), so waiting here
/// cannot reintroduce the same-thread reentrancy deadlock ADR 002 fixed.
/// The wait is bounded (not indefinite) purely as a fail-safe in case the
/// in-flight shutdown somehow runs long — same 100ms-poll shape already
/// used by `terminate_gracefully()` above.
#[cfg(windows)]
fn wait_for_inflight_shutdown_then_exit() -> ! {
    const POLL_INTERVAL: Duration = Duration::from_millis(100);
    const MAX_ATTEMPTS: u32 = 60;

    for _ in 0..MAX_ATTEMPTS {
        if BACKEND_PROCESS.try_lock().is_ok() {
            break;
        }
        std::thread::sleep(POLL_INTERVAL);
    }
    shutdown();
    std::process::exit(0);
}

/// Install the console control handler (see `console_ctrl_handler`'s doc).
/// Call once, early in `main()`, before the backend is spawned.
#[cfg(windows)]
pub fn install_ctrl_handler() {
    unsafe {
        SetConsoleCtrlHandler(Some(console_ctrl_handler), 1);
    }
}

pub const PORT: u16 = 9377;

/// Per-launch shared secret for the loopback API. Generated once (UUIDv4) and
/// stable for the process lifetime. Handed to the sidecar via the
/// `JUSTSAY_API_TOKEN` env var (see `spawn()`) and to the WebView via the
/// `get_backend_token` command (see `lib.rs`), which sends it back as the
/// `X-JustSay-Token` header. See
/// `docs/adr/026-loopback-api-request-authentication.md`.
pub fn api_token() -> &'static str {
    static API_TOKEN: OnceLock<String> = OnceLock::new();
    API_TOKEN.get_or_init(|| uuid::Uuid::new_v4().to_string())
}

const REQUEST_TIMEOUT: Duration = Duration::from_secs(120);
const HEALTH_POLL_INTERVAL: Duration = Duration::from_millis(300);
const HEALTH_POLL_MAX_ATTEMPTS: u32 = 100;

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

/// Set by `shutdown()` as its literal first statement, before anything else
/// runs. Polled by `spawn()`'s post-store recheck and by `spawn_watchdog()`'s
/// loop to distinguish an intentional stop from a crash — see
/// `docs/adr/006-backend-watchdog-respawn-on-crash.md` for the race this
/// closes.
///
/// **One-way latch: never reset back to `false`.** Correctness depends on
/// every `shutdown()` call site leading to the whole process exiting shortly
/// after — true today for all of them (`RunEvent::Exit`, the main-thread
/// panic hook, the Windows console Ctrl+C/close handler). A future
/// `shutdown()` call site that does NOT lead to process exit would
/// permanently and silently disable the watchdog for the rest of the
/// session, since `is_shutdown_requested()` would never report `false`
/// again.
static SHUTDOWN_REQUESTED: AtomicBool = AtomicBool::new(false);

fn is_shutdown_requested() -> bool {
    SHUTDOWN_REQUESTED.load(Ordering::Acquire)
}

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
/// `~/<data_dir_name>/logs/sidecar.log`. `data_dir_name` is `.justsay` or
/// `.justsay-dev`, matching `spawn()`'s `force_dev_data_dir` flag — see
/// `docs/adr/012-dev-mode-data-directory-isolation.md` — so a
/// `tauri:dev:frozen` smoke-test run's captured sidecar output lands under
/// the same dev directory as the sidecar's own history.db/settings.json.
/// Failure to open the log file is silent to avoid spamming on shutdown
/// when the FS is racing.
fn append_sidecar_log(line: &[u8], data_dir_name: &str) {
    let home = match std::env::var("USERPROFILE").or_else(|_| std::env::var("HOME")) {
        Ok(h) => h,
        Err(_) => return,
    };
    let log_dir = PathBuf::from(home).join(data_dir_name).join("logs");
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

/// Process-lifetime Windows Job Object carrying `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`,
/// created lazily on first `spawn()` (mirrors the `HTTP_CLIENT` `OnceLock`
/// pattern above). The backend child is assigned to it right after spawn (see
/// `assign_child_to_job` / `assign_pid_to_job`); the Tauri parent holds the sole
/// job handle and never closes it. When the parent dies by ANY means —
/// `taskkill /F`, Task Manager "End Task", a hard crash, `TerminateProcess` —
/// the OS closes that last handle and the kernel terminates every process still
/// in the job, freeing port 9377 for the next launch. This is the one backstop
/// that survives a kill running no in-process code (the panic hook, console
/// handler, `RunEvent::Exit` and the next-launch reap are all in-process) — see
/// `docs/adr/023-force-kill-orphaned-backend.md`.
///
/// `Option`: a failed create/configure stores `None` once, logs once, and every
/// caller degrades to the existing `reap_orphan_sidecar()` fallback rather than
/// failing startup. The raw `HANDLE` (`*mut c_void`, not `Send`/`Sync`) is
/// stored as `isize` and cast back on use so the static stays thread-safe.
#[cfg(windows)]
static BACKEND_JOB: OnceLock<Option<isize>> = OnceLock::new();

/// Create (once) the kill-on-close job and return its handle as `isize`, or
/// `None` if creation/configuration failed (logged once; never fatal).
///
/// `CreateJobObjectW(NULL, NULL)` returns a NON-inheritable handle by default —
/// required, so no spawned child can inherit a handle to the job and keep it
/// open past the parent's death, which would defeat KILL_ON_JOB_CLOSE.
#[cfg(windows)]
fn backend_job() -> Option<isize> {
    *BACKEND_JOB.get_or_init(|| {
        unsafe {
            let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
            if job.is_null() {
                log::warn!(
                    "CreateJobObjectW failed; force-kill orphan protection unavailable this session"
                );
                return None;
            }
            let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            let configured = SetInformationJobObject(
                job,
                JobObjectExtendedLimitInformation,
                &info as *const _ as *const core::ffi::c_void,
                std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            );
            if configured == 0 {
                log::warn!(
                    "SetInformationJobObject failed; force-kill orphan protection unavailable this session"
                );
                CloseHandle(job);
                return None;
            }
            Some(job as isize)
        }
    })
}

/// Assign the Dev-branch backend child to `job` via its raw OS process handle.
/// Returns `false` on failure so the caller can log-and-continue. `job` is a
/// handle from `backend_job()` (or a test-local job).
#[cfg(windows)]
fn assign_child_to_job(job: isize, child: &Child) -> bool {
    unsafe { AssignProcessToJobObject(job as HANDLE, child.as_raw_handle() as HANDLE) != 0 }
}

/// Assign the Sidecar-branch backend to `job` by pid — the shell plugin's
/// `CommandChild` exposes only `pid()`, no raw handle. Opens the minimum rights
/// `AssignProcessToJobObject` needs, assigns, and closes the OPENED handle
/// (never the job). Returns `false` on any failure — a non-existent pid, a
/// denied `OpenProcess`, or a refused assignment — so `spawn()` can
/// log-and-continue.
#[cfg(windows)]
fn assign_pid_to_job(job: isize, pid: u32) -> bool {
    unsafe {
        let process = OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, 0, pid);
        if process.is_null() {
            return false;
        }
        let assigned = AssignProcessToJobObject(job as HANDLE, process) != 0;
        CloseHandle(process);
        assigned
    }
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

    let force_dev_data_dir = cfg!(debug_assertions);
    let data_dir_name: &'static str = if force_dev_data_dir { ".justsay-dev" } else { ".justsay" };

    let resolved_sidecar = if prefer_python_source {
        None
    } else {
        resolve_sidecar(&app)
    };

    let backend = if let Some(sidecar) = resolved_sidecar {
        log::info!("Starting backend sidecar via shell plugin: {:?}", sidecar);
        let sidecar_str = sidecar.to_string_lossy().to_string();
        let port_str = PORT.to_string();
        let mut shell_cmd = app
            .shell()
            .command(sidecar_str)
            .args(["--host", "127.0.0.1", "--port", &port_str])
            .env("JUSTSAY_API_TOKEN", api_token());
        if force_dev_data_dir {
            shell_cmd = shell_cmd.env("JUSTSAY_FORCE_DEV_DATA_DIR", "1");
        }
        let (mut rx, child) = shell_cmd
            .spawn()
            .map_err(|e| format!("Failed to start backend sidecar: {}", e))?;

        #[cfg(windows)]
        if let Some(job) = backend_job() {
            let pid = child.pid();
            if !assign_pid_to_job(job, pid) {
                log::warn!(
                    "Failed to assign backend sidecar (PID {}) to the kill-on-close job; \
                     a force-kill of the app may orphan it (reap fallback still applies)",
                    pid
                );
            }
        }

        let alive = Arc::new(AtomicBool::new(true));
        let alive_clone = alive.clone();

        tauri::async_runtime::spawn(async move {
            while let Some(event) = rx.recv().await {
                match event {
                    CommandEvent::Stderr(bytes) => append_sidecar_log(&bytes, data_dir_name),
                    CommandEvent::Stdout(bytes) => append_sidecar_log(&bytes, data_dir_name),
                    CommandEvent::Error(msg) => {
                        append_sidecar_log(format!("[shell error] {}", msg).as_bytes(), data_dir_name);
                        alive_clone.store(false, Ordering::Release);
                        break;
                    }
                    CommandEvent::Terminated(payload) => {
                        let line = format!(
                            "[terminated] code={:?} signal={:?}",
                            payload.code, payload.signal
                        );
                        append_sidecar_log(line.as_bytes(), data_dir_name);
                        alive_clone.store(false, Ordering::Release);
                        break;
                    }
                    _ => {}
                }
            }
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
        .stderr(Stdio::null())
        .env("JUSTSAY_API_TOKEN", api_token());
        if force_dev_data_dir {
            cmd.env("JUSTSAY_FORCE_DEV_DATA_DIR", "1");
        }
        #[cfg(windows)]
        {
            let mut flags = CREATE_NEW_PROCESS_GROUP;
            if !cfg!(debug_assertions) {
                flags |= CREATE_NO_WINDOW;
            }
            cmd.creation_flags(flags);
        }
        let child = cmd
            .spawn()
            .map_err(|e| format!("Failed to start backend: {}", e))?;

        #[cfg(windows)]
        if let Some(job) = backend_job() {
            if !assign_child_to_job(job, &child) {
                log::warn!(
                    "Failed to assign dev backend (PID {}) to the kill-on-close job; \
                     a force-kill of the app may orphan it",
                    child.id()
                );
            }
        }

        BackendProcess::Dev(child)
    };

    {
        let mut guard = BACKEND_PROCESS.lock().map_err(|e| e.to_string())?;
        *guard = Some(backend);
    }

    if is_shutdown_requested() {
        shutdown();
    }

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

/// Attempt a graceful stop before falling back to a forced kill.
///
/// Non-Windows: sends `SIGTERM` via the `kill` command, then polls
/// `is_alive` every 100ms for up to 3s; if the process is still alive after
/// that window, calls `force_kill`.
///
/// Windows has no POSIX `SIGTERM`. When `windows_ctrl_break` is `true` (the
/// `Dev` call site only — the child must have been spawned with
/// `CREATE_NEW_PROCESS_GROUP` for this to target only itself), this sends
/// `CTRL_BREAK_EVENT` and polls the same way as the non-Windows branch
/// before falling back to a forced kill. When `windows_ctrl_break` is
/// `false` (the production `Sidecar` call site), behavior is unchanged from
/// before this parameter existed: an immediate forced `taskkill` — the
/// shell plugin's `Command` builder exposes no way to add
/// `CREATE_NEW_PROCESS_GROUP`, so `CTRL_BREAK_EVENT` cannot safely target
/// only that child (see `docs/adr/004-windows-graceful-backend-stop.md`).
#[cfg_attr(not(target_os = "windows"), allow(unused_variables))]
fn terminate_gracefully(
    pid: u32,
    mut is_alive: impl FnMut() -> bool,
    force_kill: impl FnOnce(),
    windows_ctrl_break: bool,
) {
    #[cfg(not(target_os = "windows"))]
    {
        let _ = Command::new("kill").args(["-TERM", &pid.to_string()]).status();
        for _ in 0..30 {
            if !is_alive() {
                log::info!("PID {} exited gracefully after SIGTERM", pid);
                return;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        log::warn!("PID {} still alive 3s after SIGTERM grace period; force-killing", pid);
    }
    #[cfg(target_os = "windows")]
    {
        if windows_ctrl_break {
            let sent = unsafe { GenerateConsoleCtrlEvent(CTRL_BREAK_EVENT, pid) } != 0;
            if sent {
                for _ in 0..30 {
                    if !is_alive() {
                        log::info!("PID {} exited gracefully after CTRL_BREAK_EVENT", pid);
                        return;
                    }
                    std::thread::sleep(Duration::from_millis(100));
                }
                log::warn!("PID {} still alive 3s after CTRL_BREAK_EVENT; force-killing", pid);
            } else {
                log::warn!("GenerateConsoleCtrlEvent failed for PID {}; force-killing", pid);
            }
        }
    }
    force_kill();
}

/// Kill the backend process on shutdown.
pub fn shutdown() {
    SHUTDOWN_REQUESTED.store(true, Ordering::Release);
    kill_current_process();
}

/// Terminate whatever process `BACKEND_PROCESS` currently tracks, if any —
/// extracted verbatim from `shutdown()`'s former body, no behavioral change.
///
/// Deliberately NOT folded into `shutdown()` as a single function: the
/// watchdog (`spawn_watchdog()`) needs this exact termination mechanics to
/// kill a hung-but-never-confirmed-ready previous instance before a retry,
/// without going through `shutdown()` itself — `shutdown()` also flips
/// `SHUTDOWN_REQUESTED`, which is a one-way latch that is never reset back
/// to `false` (see that static's doc comment). Calling `shutdown()` from a
/// watchdog retry would permanently poison `is_shutdown_requested()` after
/// the very first retry, defeating the watchdog for the rest of the session.
fn kill_current_process() {
    let mut guard = match BACKEND_PROCESS.try_lock() {
        Ok(g) => g,
        Err(_) => return,
    };

    if let Some(backend) = guard.take() {
        match backend {
            BackendProcess::Sidecar(s) => {
                let pid = s.child.pid();
                log::info!("Shutting down backend sidecar (PID: {})", pid);
                let alive = s.alive.clone();
                terminate_gracefully(
                    pid,
                    move || alive.load(Ordering::Acquire),
                    move || {
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
                    },
                    false,
                );
            }
            BackendProcess::Dev(child) => {
                let pid = child.id();
                log::info!("Shutting down backend (dev, PID: {})", pid);
                let child_cell = std::cell::RefCell::new(child);
                terminate_gracefully(
                    pid,
                    || {
                        child_cell
                            .borrow_mut()
                            .try_wait()
                            .map(|s| s.is_none())
                            .unwrap_or(true)
                    },
                    || {
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
                            let _ = child_cell.borrow_mut().kill();
                        }
                    },
                    true,
                );

                let _ = child_cell.borrow_mut().wait();
            }
        }
    }
}

const WATCHDOG_POLL_INTERVAL: Duration = Duration::from_secs(2);
const MAX_RESPAWN_ATTEMPTS: u32 = 3;

fn respawn_backoff(attempt: u32) -> Duration {
    Duration::from_secs(2u64.pow(attempt + 1))
}

/// Background task: detect an unexpected backend crash (or hang) and
/// respawn, bounded by `MAX_RESPAWN_ATTEMPTS` consecutive failures with
/// exponential backoff. Started once from `lib.rs`'s `setup()` right after
/// the initial `backend::spawn()` call — see
/// `docs/adr/006-backend-watchdog-respawn-on-crash.md` for the full
/// race-closure rationale behind the `SHUTDOWN_REQUESTED` checks below,
/// which are a noise/latency optimization only; `spawn()`'s own post-store
/// recheck is what actually guarantees correctness.
///
/// Health is tracked via a local `confirmed_healthy` flag, set only by
/// `wait_for_ready()` succeeding — never by `is_process_alive()` alone. A
/// hung child (e.g. OOM mid model-load, corrupted local model cache) stays
/// "alive" per `is_process_alive()` forever without ever answering
/// `/health`; conflating the two previously caused a hung respawn to
/// silently reset the retry counter every poll tick, so the watchdog never
/// gave up and never logged anything. Before each retry's `spawn()` call, a
/// still-alive-but-never-confirmed instance is killed via
/// `kill_current_process()` so the fresh spawn doesn't collide with it on
/// the port.
pub fn spawn_watchdog(app: AppHandle) {
    tauri::async_runtime::spawn(async move {
        let mut attempt: u32 = 0;
        let mut confirmed_healthy = match wait_for_ready().await {
            Ok(()) => true,
            Err(e) => {
                log::error!("Backend watchdog: initial backend not ready: {}", e);
                false
            }
        };

        loop {
            tokio::time::sleep(WATCHDOG_POLL_INTERVAL).await;
            if is_shutdown_requested() {
                return;
            }

            if confirmed_healthy {
                if is_process_alive() {
                    continue;
                }
                confirmed_healthy = false;
                attempt = 0;
            }

            if attempt >= MAX_RESPAWN_ATTEMPTS {
                log::error!(
                    "Backend watchdog: giving up after {} consecutive failed respawns",
                    MAX_RESPAWN_ATTEMPTS
                );
                return;
            }

            let backoff = respawn_backoff(attempt);
            log::warn!(
                "Backend watchdog: respawning in {:?} (attempt {}/{})",
                backoff,
                attempt + 1,
                MAX_RESPAWN_ATTEMPTS
            );
            tokio::time::sleep(backoff).await;
            if is_shutdown_requested() {
                return;
            }

            if is_process_alive() {
                log::warn!(
                    "Backend watchdog: previous backend never became ready; terminating before respawn"
                );
                kill_current_process();
            }

            match spawn(app.clone()) {
                Ok(()) => match wait_for_ready().await {
                    Ok(()) => {
                        log::info!("Backend watchdog: respawn succeeded");
                        confirmed_healthy = true;
                        attempt = 0;
                    }
                    Err(e) => {
                        log::error!("Backend watchdog: respawn did not become ready: {}", e);
                        attempt += 1;
                    }
                },
                Err(e) => {
                    log::error!("Backend watchdog: respawn failed: {}", e);
                    attempt += 1;
                }
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn respawn_backoff_follows_2_4_8_second_sequence() {
        assert_eq!(respawn_backoff(0), Duration::from_secs(2));
        assert_eq!(respawn_backoff(1), Duration::from_secs(4));
        assert_eq!(respawn_backoff(2), Duration::from_secs(8));
    }

    #[test]
    fn api_token_is_non_empty_and_stable_across_calls() {
        let first = api_token();
        assert!(!first.is_empty(), "api_token must not be empty");
        assert_eq!(first, api_token(), "api_token must be stable across calls");
    }

    /// Create a throwaway kill-on-close job for a single test. Never the shared
    /// `BACKEND_JOB` static — a test must be able to `CloseHandle` it (the kill
    /// trigger) and must not leak process-lifetime state into the test binary.
    #[cfg(windows)]
    fn create_test_job() -> isize {
        unsafe {
            let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
            assert!(!job.is_null(), "CreateJobObjectW failed in test");
            let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            let configured = SetInformationJobObject(
                job,
                JobObjectExtendedLimitInformation,
                &info as *const _ as *const core::ffi::c_void,
                std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            );
            assert_ne!(configured, 0, "SetInformationJobObject failed in test");
            job as isize
        }
    }

    /// Spawn a short-lived throwaway child that stays alive long enough to
    /// prove the job kills it (`ping -n 30` ~ 29s), with no console window.
    #[cfg(windows)]
    fn spawn_throwaway_child() -> Child {
        Command::new("cmd")
            .args(["/c", "ping", "-n", "30", "127.0.0.1"])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .creation_flags(CREATE_NO_WINDOW)
            .spawn()
            .expect("failed to spawn throwaway test child")
    }

    /// Bounded poll (~5s) for the child to exit; force-kills and returns false
    /// on timeout so a bug can never leave the child running or hang the suite.
    #[cfg(windows)]
    fn wait_for_exit(child: &mut Child) -> bool {
        for _ in 0..50 {
            match child.try_wait() {
                Ok(Some(_)) => return true,
                _ => std::thread::sleep(Duration::from_millis(100)),
            }
        }
        let _ = child.kill();
        let _ = child.wait();
        false
    }

    #[cfg(windows)]
    #[test]
    fn kill_on_job_close_terminates_child_assigned_via_raw_handle() {
        let job = create_test_job();
        let mut child = spawn_throwaway_child();

        assert!(
            assign_child_to_job(job, &child),
            "raw-handle assignment should succeed for a live owned child"
        );
        assert!(
            matches!(child.try_wait(), Ok(None)),
            "child must be alive before the job handle is closed"
        );

        unsafe {
            CloseHandle(job as HANDLE);
        }

        assert!(
            wait_for_exit(&mut child),
            "child must exit after the job's last handle is closed"
        );
    }

    #[cfg(windows)]
    #[test]
    fn kill_on_job_close_terminates_child_assigned_via_pid() {
        let job = create_test_job();
        let mut child = spawn_throwaway_child();

        assert!(
            assign_pid_to_job(job, child.id()),
            "pid/OpenProcess assignment should succeed for a live process"
        );
        assert!(
            matches!(child.try_wait(), Ok(None)),
            "child must be alive before the job handle is closed"
        );

        unsafe {
            CloseHandle(job as HANDLE);
        }

        assert!(
            wait_for_exit(&mut child),
            "child must exit after the job's last handle is closed"
        );
    }

    #[cfg(windows)]
    #[test]
    fn assign_pid_to_job_is_non_fatal_for_a_nonexistent_pid() {
        let job = create_test_job();

        let assigned = assign_pid_to_job(job, u32::MAX);
        assert!(
            !assigned,
            "assigning a non-existent pid must return false, not panic"
        );

        unsafe {
            CloseHandle(job as HANDLE);
        }
    }
}
