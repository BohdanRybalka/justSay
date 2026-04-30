use tauri::{
    image::Image,
    menu::{Menu, MenuItem},
    tray::TrayIconBuilder,
    webview::WebviewWindowBuilder,
    AppHandle, Manager, RunEvent, WebviewUrl, WindowEvent,
};

mod backend;

/// Clip the widget window to a pill shape.
#[cfg(target_os = "windows")]
fn apply_pill_region(widget: &tauri::WebviewWindow) {
    use windows::Win32::Graphics::Gdi::{CreateRoundRectRgn, SetWindowRgn};

    let (hwnd, size) = match (widget.hwnd(), widget.inner_size()) {
        (Ok(h), Ok(s)) => (h, s),
        _ => return,
    };

    unsafe {
        let w = size.width as i32;
        let h = size.height as i32;
        let rgn = CreateRoundRectRgn(0, 0, w + 1, h + 1, h, h);
        if !rgn.is_invalid() {
            let _ = SetWindowRgn(hwnd, Some(rgn), true);
        }
    }
}

/// Apply uniform transparency to the widget (call AFTER show).
#[cfg(target_os = "windows")]
fn apply_widget_alpha(widget: &tauri::WebviewWindow, alpha: u8) {
    use windows::Win32::UI::WindowsAndMessaging::*;

    let hwnd = match widget.hwnd() {
        Ok(h) => h,
        Err(_) => return,
    };

    unsafe {
        let ex_style = GetWindowLongW(hwnd, GWL_EXSTYLE);
        SetWindowLongW(hwnd, GWL_EXSTYLE, ex_style | WS_EX_LAYERED.0 as i32);
        let _ = SetLayeredWindowAttributes(
            hwnd,
            windows::Win32::Foundation::COLORREF(0),
            alpha,
            LWA_ALPHA,
        );
        log::info!("Widget alpha set to {}", alpha);
    }
}

#[cfg(not(target_os = "windows"))]
fn apply_pill_region(_widget: &tauri::WebviewWindow) {}
#[cfg(not(target_os = "windows"))]
fn apply_widget_alpha(_widget: &tauri::WebviewWindow, _alpha: u8) {}

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
        // Pill shape BEFORE show
        apply_pill_region(&widget);

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

        let _ = widget.show();

        // Alpha AFTER show (WS_EX_LAYERED must be set on visible window)
        apply_widget_alpha(&widget, 160); // ~63% opacity
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_global_shortcut::Builder::new().build())
        .plugin(tauri_plugin_notification::init())
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            if cfg!(debug_assertions) {
                app.handle().plugin(
                    tauri_plugin_log::Builder::default()
                        .level(log::LevelFilter::Info)
                        .build(),
                )?;
            }

            // Spawn Python backend
            if let Err(e) = backend::spawn() {
                log::error!("Backend spawn failed: {}", e);
            }

            // Create widget window — shaped as pill via SetWindowRgn, no transparency needed
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
