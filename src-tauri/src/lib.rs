use tauri::{
    image::Image,
    menu::{Menu, MenuItem},
    tray::TrayIconBuilder,
    webview::WebviewWindowBuilder,
    AppHandle, Manager, RunEvent, WebviewUrl, WindowEvent,
};

mod backend;

/// Kill the backend child process if one is running. Safe to call even if
/// nothing is running (no-op). Exposed narrowly for `main.rs`'s panic hook —
/// see docs/adr/002-backend-process-panic-safe-shutdown.md.
///
/// Calls the non-waiting `shutdown_without_waiting()`, never `shutdown()`:
/// this runs from the main-thread panic hook, which can re-enter on the same
/// thread `RunEvent::Exit` → `shutdown()` is already mid-run on, and
/// `std::sync::Mutex` is not reentrant — a blocking or waiting acquire here
/// would hang the app instead of letting it crash. See
/// docs/adr/032-production-quit-runs-backend-teardown.md.
pub fn shutdown_backend() {
    backend::shutdown_without_waiting();
}

/// Install a Windows console-control handler so a raw Ctrl+C or console
/// close in the dev terminal still shuts the backend down gracefully
/// instead of orphaning it — see docs/adr/004-windows-graceful-backend-stop.md
/// and `backend::console_ctrl_handler`'s doc for why this is needed only on
/// Windows (`CREATE_NEW_PROCESS_GROUP` on the Dev child, added by this same
/// spec, isolates it from the parent's console broadcast Ctrl+C).
#[cfg(windows)]
pub fn install_console_ctrl_handler() {
    backend::install_ctrl_handler();
}

/// Expose the per-launch API token to the WebView, which sends it back as the
/// `X-JustSay-Token` header on every backend request. See
/// docs/adr/026-loopback-api-request-authentication.md.
#[tauri::command]
fn get_backend_token() -> String {
    backend::api_token().to_string()
}


#[tauri::command]
fn widget_ready(app: AppHandle) {
    if let Some(widget) = app.get_webview_window("widget") {
        if let Ok(Some(monitor)) = widget.current_monitor() {
            let screen = monitor.size();
            let scale = monitor.scale_factor();
            let w = 240.0;
            let h = 48.0;
            let x = (screen.width as f64 / scale - w) / 2.0;
            let y = screen.height as f64 / scale - h - 220.0;
            let _ = widget.set_position(tauri::PhysicalPosition::new(
                (x * scale) as i32,
                (y * scale) as i32,
            ));
        }

        let _ = widget.show();
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_global_shortcut::Builder::new().build())
        .plugin(tauri_plugin_notification::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .setup(|app| {
            app.handle().plugin(
                tauri_plugin_log::Builder::default()
                    .level(if cfg!(debug_assertions) {
                        log::LevelFilter::Debug
                    } else {
                        log::LevelFilter::Info
                    })
                    .max_file_size(1_000_000)
                    .build(),
            )?;

            log::info!(
                "JustSay {} starting up (backend port {})",
                app.package_info().version,
                backend::PORT
            );

            if let Err(e) = backend::spawn(app.handle().clone()) {
                log::error!("Backend spawn failed: {}", e);
            }

            backend::spawn_watchdog(app.handle().clone());

            let _widget = WebviewWindowBuilder::new(
                app,
                "widget",
                WebviewUrl::App("/widget.html".into()),
            )
            .title("")
            .inner_size(160.0, 40.0)
            .resizable(false)
            .visible(false)
            .decorations(false)
            .transparent(true)
            .always_on_top(true)
            .skip_taskbar(true)
            .shadow(false)
            .build()?;

            let settings_item =
                MenuItem::with_id(app, "settings", "Settings", true, None::<&str>)?;
            let quit = MenuItem::with_id(app, "quit", "Quit JustSay", true, None::<&str>)?;
            let menu = Menu::with_items(app, &[&settings_item, &quit])?;

            let icon = Image::from_bytes(include_bytes!("../icons/32x32.png"))?;

            let app_handle = app.handle().clone();
            TrayIconBuilder::new()
                .icon(icon)
                .menu(&menu)
                .tooltip("JustSay — Voice to Text")
                .on_menu_event(move |_tray, event| match event.id.as_ref() {
                    "quit" => {
                        app_handle.exit(0);
                    }
                    "settings" => {
                        if let Some(window) = app_handle.get_webview_window("settings") {
                            let _ = window.show();
                            let _ = window.set_focus();
                        }
                    }
                    _ => {}
                })
                .build(app)?;

            let settings_handle = app.handle().clone();
            if let Some(settings) = app.get_webview_window("settings") {
                settings.on_window_event(move |event| {
                    if let WindowEvent::CloseRequested { api, .. } = event {
                        api.prevent_close();
                        if let Some(win) = settings_handle.get_webview_window("settings") {
                            let _ = win.hide();
                        }
                    }
                });
            }

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            widget_ready,
            get_backend_token
        ])
        .build(tauri::generate_context!())
        .expect("error while building tauri application");

    app.run(|_app, event| {
        if let RunEvent::Exit = event {
            backend::shutdown();
        }
    });
}
