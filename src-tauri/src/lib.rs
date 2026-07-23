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
            // Logging is registered in release builds too. A packaged launch
            // that fails (backend spawn, or the WebView never getting its
            // token) leaves no other trace on a machine we cannot attach a
            // debugger to — see docs/adr/028-csp-must-enumerate-every-tauri-bridge-source.md.
            // The plugin's 40 KB default with KeepOne can rotate the startup
            // lines away mid-session, which is exactly what must survive, so
            // the ceiling is raised to ~1 MB per file (~2 MB retained).
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

            // Spawn Python backend (production = shell-plugin spawn, dev = system Python)
            if let Err(e) = backend::spawn(app.handle().clone()) {
                log::error!("Backend spawn failed: {}", e);
            }

            // Watchdog: detect a crashed/never-came-up backend and respawn
            // it (bounded retries with backoff) — same retry path handles
            // both cases, deliberately not distinguished. See
            // docs/adr/006-backend-watchdog-respawn-on-crash.md.
            backend::spawn_watchdog(app.handle().clone());

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

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            backend_request,
            get_backend_url,
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
