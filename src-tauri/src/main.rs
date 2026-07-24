#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
  let default_hook = std::panic::take_hook();
  std::panic::set_hook(Box::new(move |info| {
    if std::thread::current().name() == Some("main") {
      app_lib::shutdown_backend();
    }
    default_hook(info);
  }));

  #[cfg(windows)]
  app_lib::install_console_ctrl_handler();

  app_lib::run();
}
