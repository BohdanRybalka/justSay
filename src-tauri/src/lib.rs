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
pub fn shutdown_backend() {
    backend::shutdown();
}

#[tauri::command]
async fn backend_request(
    method: String,
    path: String,
    body: Option<String>,
) -> Result<String, String> {
    backend::request(&method, &path, body.as_deref())
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
fn get_backend_url() -> String {
    format!("http://127.0.0.1:{}", backend::PORT)
}


#[tauri::command]
fn widget_ready(app: AppHandle) {
    if let Some(widget) = app.get_webview_window("widget") {
        // Position at bottom-center of screen
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

        // Pill shape and translucency are handled uniformly in CSS on a
        // transparent window, so no platform-specific window shaping here.
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
            if cfg!(debug_assertions) {
                app.handle().plugin(
                    tauri_plugin_log::Builder::default()
                        .level(log::LevelFilter::Info)
                        .build(),
                )?;
            }

            // Spawn Python backend (production = shell-plugin spawn, dev = system Python)
            if let Err(e) = backend::spawn(app.handle().clone()) {
                log::error!("Backend spawn failed: {}", e);
            }

            // Create widget window — transparent so the CSS-drawn rounded pill
            // renders identically on macOS and Windows (corners stay see-through).
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

            // System tray
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

            // Settings window: hide on close instead of quitting
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

            // Wait for backend readiness
            tauri::async_runtime::spawn(async move {
                match backend::wait_for_ready().await {
                    Ok(()) => log::info!("Backend is ready"),
                    Err(e) => log::error!("Backend not ready: {}", e),
                }
            });

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![backend_request, get_backend_url, widget_ready])
        .build(tauri::generate_context!())
        .expect("error while building tauri application");

    app.run(|_app, event| {
        if let RunEvent::Exit = event {
            backend::shutdown();
        }
    });
}
