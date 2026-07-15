// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
  // Panic-safe backend cleanup — see docs/adr/002-backend-process-panic-safe-shutdown.md.
  // `static` destructors never run on any exit path (normal return,
  // `std::process::exit`, or an unwinding panic), so `BACKEND_PROCESS`
  // cannot be cleaned up via `Drop`; a panic hook is the reliable hook point.
  let default_hook = std::panic::take_hook();
  std::panic::set_hook(Box::new(move |info| {
    // Only the main thread's panic actually terminates the whole desktop
    // app; background tokio/tauri worker panics are contained to that task
    // and do not orphan the backend, so gating here avoids killing a
    // healthy backend for a recoverable bug.
    if std::thread::current().name() == Some("main") {
      app_lib::shutdown_backend();
    }
    default_hook(info);
  }));

  app_lib::run();
}
