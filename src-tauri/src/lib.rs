use std::sync::Mutex;

use tauri::{
    image::Image,
    menu::{Menu, MenuItem},
    tray::TrayIconBuilder,
    webview::WebviewWindowBuilder,
    AppHandle, Emitter, Manager, RunEvent, WebviewUrl, WindowEvent, Wry,
};

mod backend;

/// The widget window's logical width, as built and as placed.
const WIDGET_WIDTH: f64 = 160.0;

/// The widget window's logical height, as built and as placed.
const WIDGET_HEIGHT: f64 = 40.0;

/// Logical gap between the bottom of the monitor and the widget's bottom edge.
const WIDGET_BOTTOM_GAP: f64 = 220.0;

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


/// The tray's meeting-recording item, kept so its label can follow the actual
/// recording state. Held in Tauri's managed state rather than a static: the
/// item is created inside `setup`, and the command that relabels it runs later
/// on whichever thread the WebView's IPC lands on.
struct MeetingMenuItem(MenuItem<Wry>);

/// Relabel the tray item after the widget has started or stopped a meeting
/// recording. HTTP stays in TypeScript and Rust stays a system-events layer,
/// which is why the widget calls the backend and then tells the shell.
#[tauri::command]
fn set_meeting_recording(app: AppHandle, active: bool) {
    if let Some(item) = app.try_state::<MeetingMenuItem>() {
        let label = if active {
            "Stop meeting recording"
        } else {
            "Start meeting recording"
        };
        let _ = item.0.set_text(label);
    }
}

/// The on-screen answer most recently announced for the settings window, so a
/// run of window events does not send the page one event per tick. `None`
/// until the first announcement.
static ANNOUNCED_SETTINGS_ON_SCREEN: Mutex<Option<bool>> = Mutex::new(None);

/// Whether the settings surface is on screen: the window is visible and is not
/// minimised. Nothing else takes part — a visible, unminimised window the user
/// has clicked away from is still on screen, so a lost focus is a reason to
/// read this again and never an answer to it (ADR 089).
fn settings_is_on_screen(visible: bool, minimized: bool) -> bool {
    visible && !minimized
}

/// The on-screen value that must be announced, or `None` when it is the value
/// already announced. Remembering the last one is what keeps a stream of
/// window events from spraying the page with news it has already acted on.
fn settings_visibility_to_announce(announced: Option<bool>, on_screen: bool) -> Option<bool> {
    if announced == Some(on_screen) {
        None
    } else {
        Some(on_screen)
    }
}

/// The on-screen answer a window event carries by itself, or `None` when the
/// event is only a reason to read the window again. Taking focus answers it:
/// `tao` moves focus onto a window only while it is visible and not minimised,
/// on both shipping platforms, and a macOS restore from the Dock arrives as
/// that focus gain and as nothing else (ADR 089).
fn settings_on_screen_from_window_event(event: &WindowEvent) -> Option<bool> {
    match event {
        WindowEvent::Focused(true) => Some(true),
        _ => None,
    }
}

/// Announce a change in the settings window's on-screen state, so the tab on
/// it takes back or lets go of what it holds. Every path that shows it, hides
/// it or is told it moved calls this one, which is what makes both
/// announcements complete (ADR 089). `known_on_screen` is the answer the
/// caller's event already carries; `None` reads the window, and a state that
/// cannot be read announces nothing.
fn announce_settings_visibility(app: &AppHandle, known_on_screen: Option<bool>) {
    let Some(window) = app.get_webview_window("settings") else {
        return;
    };
    let on_screen = match known_on_screen {
        Some(answer) => answer,
        None => {
            let (Ok(visible), Ok(minimized)) = (window.is_visible(), window.is_minimized()) else {
                log::warn!("Reading the settings window's state failed, so nothing is announced");
                return;
            };
            settings_is_on_screen(visible, minimized)
        }
    };

    let announced = {
        let mut last = ANNOUNCED_SETTINGS_ON_SCREEN
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        match settings_visibility_to_announce(*last, on_screen) {
            None => return,
            Some(value) => {
                *last = Some(value);
                value
            }
        }
    };

    let _ = if announced {
        app.emit("settings-shown", ())
    } else {
        app.emit("settings-hidden", ())
    };
}

/// Show the settings window and hand the outcome to the announcer, so a tab
/// that released what it held on the dismissal can take it back. Every path
/// that shows that window calls this one (ADR 089).
///
/// A failed `show()` announces nothing: the page would otherwise resume its
/// polling into a window the user cannot see.
fn show_settings(app: &AppHandle) {
    if let Some(window) = app.get_webview_window("settings") {
        if let Err(e) = window.show() {
            log::warn!("Showing the settings window failed, so nothing is announced: {}", e);
            return;
        }
        if let Err(e) = window.unminimize() {
            log::warn!("Restoring the minimised settings window failed: {}", e);
        }
        let _ = window.set_focus();
        announce_settings_visibility(app, None);
    }
}

/// Hide the settings window and hand the outcome to the announcer, so the tab
/// on it lets go of what it is holding. Every path that hides that window
/// calls this one (ADR 089).
///
/// A failed `hide()` announces nothing: the page would otherwise stop polling
/// a window that is still on screen.
fn hide_settings(app: &AppHandle) {
    if let Some(window) = app.get_webview_window("settings") {
        if let Err(e) = window.hide() {
            log::warn!("Hiding the settings window failed, so nothing is announced: {}", e);
            return;
        }
        announce_settings_visibility(app, None);
    }
}

/// Bring the settings window up on the meeting disclosure. Called when the
/// backend refuses to start a recording because it has not been acknowledged
/// (docs/adr/040-recording-other-people-is-not-covered-by-zero-leak.md).
#[tauri::command]
fn show_settings_window(app: AppHandle) {
    show_settings(&app);
}

#[tauri::command]
fn widget_ready(app: AppHandle) {
    if let Some(widget) = app.get_webview_window("widget") {
        match widget.current_monitor() {
            Ok(Some(monitor)) => {
                let screen = monitor.size();
                let origin = monitor.position();
                let scale = monitor.scale_factor();
                let x = origin.x as f64 + (screen.width as f64 - WIDGET_WIDTH * scale) / 2.0;
                let y = origin.y as f64 + screen.height as f64
                    - (WIDGET_HEIGHT + WIDGET_BOTTOM_GAP) * scale;
                if let Err(e) = widget
                    .set_position(tauri::PhysicalPosition::new(x.round() as i32, y.round() as i32))
                {
                    log::warn!("Placing the widget failed, so it keeps where the OS put it: {}", e);
                }
            }
            Ok(None) => log::warn!("The widget reports no monitor, so it keeps where the OS put it"),
            Err(e) => log::warn!("Reading the widget's monitor failed: {}", e),
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
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_process::init())
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
                backend::report_backend_failure(&format!("Backend spawn failed: {}", e));
            }

            backend::spawn_watchdog(app.handle().clone());

            let _widget = WebviewWindowBuilder::new(
                app,
                "widget",
                WebviewUrl::App("/widget.html".into()),
            )
            .title("")
            .inner_size(WIDGET_WIDTH, WIDGET_HEIGHT)
            .resizable(false)
            .visible(false)
            .decorations(false)
            .transparent(true)
            .always_on_top(true)
            .skip_taskbar(true)
            .shadow(false)
            .build()?;

            let meeting_item = MenuItem::with_id(
                app,
                "meeting",
                "Start meeting recording",
                true,
                None::<&str>,
            )?;
            let settings_item =
                MenuItem::with_id(app, "settings", "Settings", true, None::<&str>)?;
            let quit = MenuItem::with_id(app, "quit", "Quit JustSay", true, None::<&str>)?;
            let menu = Menu::with_items(app, &[&meeting_item, &settings_item, &quit])?;
            app.manage(MeetingMenuItem(meeting_item));

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
                        show_settings(&app_handle);
                    }
                    "meeting" => {
                        let _ = app_handle.emit("meeting-toggle", ());
                    }
                    _ => {}
                })
                .build(app)?;

            let settings_handle = app.handle().clone();
            if let Some(settings) = app.get_webview_window("settings") {
                settings.on_window_event(move |event| match event {
                    WindowEvent::CloseRequested { api, .. } => {
                        api.prevent_close();
                        hide_settings(&settings_handle);
                    }
                    WindowEvent::Resized(_) | WindowEvent::Focused(_) => {
                        announce_settings_visibility(
                            &settings_handle,
                            settings_on_screen_from_window_event(event),
                        );
                    }
                    _ => {}
                });
            }

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            widget_ready,
            get_backend_token,
            set_meeting_recording,
            show_settings_window
        ])
        .build(tauri::generate_context!())
        .expect("error while building tauri application");

    app.run(|_app, event| {
        if let RunEvent::Exit = event {
            backend::shutdown();
        }
    });
}

#[cfg(test)]
mod tests {
    use super::{
        settings_is_on_screen, settings_on_screen_from_window_event,
        settings_visibility_to_announce,
    };
    use tauri::{PhysicalSize, WindowEvent};

    #[test]
    fn the_settings_surface_is_on_screen_only_while_visible_and_not_minimised() {
        assert!(settings_is_on_screen(true, false));
        assert!(!settings_is_on_screen(true, true));
        assert!(!settings_is_on_screen(false, false));
        assert!(!settings_is_on_screen(false, true));
    }

    #[test]
    fn an_unchanged_on_screen_value_is_announced_once() {
        assert_eq!(settings_visibility_to_announce(None, true), Some(true));
        assert_eq!(settings_visibility_to_announce(Some(true), true), None);
        assert_eq!(settings_visibility_to_announce(Some(true), false), Some(false));
        assert_eq!(settings_visibility_to_announce(Some(false), false), None);
        assert_eq!(settings_visibility_to_announce(None, false), Some(false));
    }

    #[test]
    fn only_a_gained_focus_answers_the_on_screen_question_by_itself() {
        assert_eq!(
            settings_on_screen_from_window_event(&WindowEvent::Focused(true)),
            Some(true)
        );
        assert_eq!(
            settings_on_screen_from_window_event(&WindowEvent::Focused(false)),
            None
        );
        assert_eq!(
            settings_on_screen_from_window_event(&WindowEvent::Resized(PhysicalSize::new(900, 700))),
            None
        );
    }

    #[test]
    fn a_restore_is_announced_even_while_the_window_still_reads_minimised() {
        let announced = Some(false);
        let stale_read = settings_is_on_screen(true, true);
        assert_eq!(
            settings_visibility_to_announce(announced, stale_read),
            None,
            "the read alone announces nothing, and no further event follows a restore"
        );

        let from_event = settings_on_screen_from_window_event(&WindowEvent::Focused(true))
            .expect("a restore arrives as a gained focus and must answer by itself");
        assert_eq!(
            settings_visibility_to_announce(announced, from_event),
            Some(true)
        );
    }
}
