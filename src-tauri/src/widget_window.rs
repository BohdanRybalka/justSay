//! The widget window: it never takes focus from the app being typed into, and
//! it takes clicks only on the pill the page draws inside it. Tauri cannot make
//! part of a window click-through, so a cursor loop flips the whole window's
//! cursor events as the pointer crosses the pill, and drives the hover look
//! from a slightly larger zone around it.

use std::sync::{
    atomic::{AtomicBool, Ordering},
    Mutex,
};
use std::time::Duration;

use serde::Serialize;
use tauri::{
    webview::WebviewWindowBuilder, AppHandle, Emitter, Manager, PhysicalPosition, PhysicalRect,
    State, WebviewUrl,
};

#[cfg(target_os = "macos")]
use tauri_nspanel::tauri_panel;

/// The window's logical width: the widest pill plus the hover zone around it.
const WINDOW_WIDTH: f64 = 240.0;

/// The window's logical height: the pill plus the hover zone around it.
const WINDOW_HEIGHT: f64 = 64.0;

/// Logical distance from the bottom of the work area up to the pill's centre.
/// With Windows' 48px taskbar it puts the pill where it has always been.
const PILL_CENTRE_ABOVE_WORK_AREA_BOTTOM: f64 = 192.0;

/// How far the hover zone reaches past the pill on the left and right, logical.
const HOVER_MARGIN_X: f64 = 24.0;

/// How far the hover zone reaches past the pill above and below, logical.
const HOVER_MARGIN_Y: f64 = 16.0;

/// The cursor loop's period, about 30 times a second.
const CURSOR_POLL: Duration = Duration::from_millis(33);

/// Set by the first `widget_ready`, so a reloaded page never starts a second loop.
static CURSOR_LOOP_STARTED: AtomicBool = AtomicBool::new(false);

#[cfg(target_os = "macos")]
tauri_panel! {
    panel!(WidgetPanel {
        config: {
            can_become_key_window: false,
            can_become_main_window: false,
            is_floating_panel: true,
            hides_on_deactivate: false
        }
    })
}

/// Where the page drew the pill, in physical pixels from the window's client
/// origin.
#[derive(Clone, Copy, Debug, PartialEq)]
struct PillRect {
    x: f64,
    y: f64,
    width: f64,
    height: f64,
}

/// The pill as the page last reported it. `None` until the first report, which
/// leaves the whole window click-through.
#[derive(Default)]
pub struct WidgetPill(Mutex<Option<PillRect>>);

/// Where the pointer stands: on the pill, the only place that takes clicks, and
/// inside the hover zone around it, which only drives the hover look.
#[derive(Clone, Copy, Debug, PartialEq)]
struct PointerOver {
    on_pill: bool,
    in_hover_zone: bool,
}

/// What the cursor loop last applied to the window and announced to the page.
#[derive(Default)]
struct Applied {
    ignoring: Option<bool>,
    hovering: Option<bool>,
}

/// What one tick must apply; `None` is an answer that is already applied.
#[derive(Debug, PartialEq)]
struct PointerChanges {
    ignore: Option<bool>,
    hover: Option<bool>,
}

/// Payload of the event that tells the page the pointer entered or left the
/// hover zone.
#[derive(Clone, Serialize)]
struct WidgetHover {
    inside: bool,
}

/// Build the hidden widget window and the state its cursor loop reads. On
/// macOS it becomes a non-activating panel: a plain window there activates the
/// app on every click, even one that can never take the keyboard.
pub fn build(app: &tauri::App) -> tauri::Result<()> {
    app.manage(WidgetPill::default());

    #[cfg_attr(not(target_os = "macos"), allow(unused_variables))]
    let widget = WebviewWindowBuilder::new(app, "widget", WebviewUrl::App("/widget.html".into()))
        .title("")
        .inner_size(WINDOW_WIDTH, WINDOW_HEIGHT)
        .resizable(false)
        .visible(false)
        .decorations(false)
        .transparent(true)
        .always_on_top(true)
        .skip_taskbar(true)
        .shadow(false)
        .focusable(false)
        .focused(false)
        .accept_first_mouse(true)
        .build()?;

    #[cfg(target_os = "macos")]
    {
        use tauri_nspanel::{StyleMask, WebviewWindowExt};

        let panel = widget.to_panel::<WidgetPanel>()?;
        panel
            .add_style_mask(StyleMask::empty().nonactivating_panel().into())
            .map_err(|e| tauri::Error::Anyhow(e.into()))?;
    }

    Ok(())
}

/// Place the widget from its monitor's work area, show it without taking
/// focus, and start the cursor loop. The page calls this once it has drawn.
#[tauri::command]
pub fn widget_ready(app: AppHandle) {
    let Some(widget) = app.get_webview_window("widget") else {
        return;
    };
    match widget.current_monitor() {
        Ok(Some(monitor)) => {
            let origin = window_origin(monitor.work_area(), monitor.scale_factor());
            if let Err(e) = widget.set_position(origin) {
                log::warn!("Placing the widget failed, so it keeps where the OS put it: {}", e);
            }
        }
        Ok(None) => log::warn!("The widget reports no monitor, so it keeps where the OS put it"),
        Err(e) => log::warn!("Reading the widget's monitor failed: {}", e),
    }

    if let Err(e) = widget.show() {
        log::warn!("Showing the widget failed: {}", e);
    }
    follow_the_cursor(&app);
}

/// Record where the page drew the pill, in physical pixels from the window's
/// client origin, so the cursor loop tests the pointer against the real shape.
#[tauri::command]
pub fn set_widget_pill_rect(
    pill: State<'_, WidgetPill>,
    x: f64,
    y: f64,
    width: f64,
    height: f64,
) {
    *pill.0.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) =
        Some(PillRect { x, y, width, height });
}

/// Start the cursor loop once. It runs from the widget's first show for the
/// life of the app, because nothing hides the widget again.
fn follow_the_cursor(app: &AppHandle) {
    if CURSOR_LOOP_STARTED.swap(true, Ordering::SeqCst) {
        return;
    }
    let app = app.clone();
    std::thread::spawn(move || {
        let mut applied = Applied::default();
        while let Some(widget) = app.get_webview_window("widget") {
            std::thread::sleep(CURSOR_POLL);
            let (Ok(cursor), Ok(origin), Ok(scale)) = (
                widget.cursor_position(),
                widget.inner_position(),
                widget.scale_factor(),
            ) else {
                continue;
            };
            let cursor_scale = if cfg!(target_os = "macos") {
                app.primary_monitor().ok().flatten().map_or(scale, |primary| primary.scale_factor())
            } else {
                scale
            };
            let cursor = cursor_in_window_pixels(cursor, cursor_scale, scale);
            let pill = *app
                .state::<WidgetPill>()
                .0
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let changes = pointer_changes(&applied, pointer_over(cursor, origin, pill, scale));

            if let Some(ignore) = changes.ignore {
                if let Err(e) = widget.set_ignore_cursor_events(ignore) {
                    log::warn!("Switching the widget's click-through failed: {}", e);
                }
                applied.ignoring = Some(ignore);
            }
            if let Some(inside) = changes.hover {
                if let Err(e) = app.emit_to("widget", "widget-hover", WidgetHover { inside }) {
                    log::warn!("Telling the widget about the hover failed: {}", e);
                }
                applied.hovering = Some(inside);
            }
        }
    });
}

/// Where the window's top-left corner goes so the pill, centred in the window,
/// sits centred across the work area and a fixed distance above its bottom.
fn window_origin(work_area: &PhysicalRect<i32, u32>, scale: f64) -> PhysicalPosition<i32> {
    let centre_x = f64::from(work_area.position.x) + f64::from(work_area.size.width) / 2.0;
    let centre_y = f64::from(work_area.position.y) + f64::from(work_area.size.height)
        - PILL_CENTRE_ABOVE_WORK_AREA_BOTTOM * scale;
    PhysicalPosition::new(
        (centre_x - WINDOW_WIDTH * scale / 2.0).round() as i32,
        (centre_y - WINDOW_HEIGHT * scale / 2.0).round() as i32,
    )
}

/// The cursor rescaled from `cursor_scale` into the window's own pixels. tao on
/// macOS converts the cursor with the primary display's factor and the window
/// with its display's, so on mixed displays the two disagree; Windows reports
/// both in true pixels, and there the two factors are the same.
fn cursor_in_window_pixels(
    cursor: PhysicalPosition<f64>,
    cursor_scale: f64,
    window_scale: f64,
) -> PhysicalPosition<f64> {
    let ratio = window_scale / cursor_scale;
    PhysicalPosition::new(cursor.x * ratio, cursor.y * ratio)
}

/// Where the pointer stands relative to the pill. Positions are physical;
/// `scale` turns the logical hover margins into physical pixels.
fn pointer_over(
    cursor: PhysicalPosition<f64>,
    origin: PhysicalPosition<i32>,
    pill: Option<PillRect>,
    scale: f64,
) -> PointerOver {
    let Some(pill) = pill else {
        return PointerOver { on_pill: false, in_hover_zone: false };
    };
    let x = cursor.x - f64::from(origin.x);
    let y = cursor.y - f64::from(origin.y);
    let within = |margin_x: f64, margin_y: f64| {
        x >= pill.x - margin_x
            && x < pill.x + pill.width + margin_x
            && y >= pill.y - margin_y
            && y < pill.y + pill.height + margin_y
    };
    PointerOver {
        on_pill: within(0.0, 0.0),
        in_hover_zone: within(HOVER_MARGIN_X * scale, HOVER_MARGIN_Y * scale),
    }
}

/// The window ignores the cursor everywhere off the pill. Each answer is acted
/// on only when it differs from the one already applied.
fn pointer_changes(applied: &Applied, over: PointerOver) -> PointerChanges {
    let ignore = !over.on_pill;
    PointerChanges {
        ignore: (applied.ignoring != Some(ignore)).then_some(ignore),
        hover: (applied.hovering != Some(over.in_hover_zone)).then_some(over.in_hover_zone),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        cursor_in_window_pixels, pointer_changes, pointer_over, window_origin, Applied, PillRect,
        PointerChanges, PointerOver,
    };
    use tauri::{PhysicalPosition, PhysicalRect, PhysicalSize};

    const PILL: PillRect = PillRect { x: 80.0, y: 24.0, width: 160.0, height: 40.0 };

    fn over(cursor_x: f64, cursor_y: f64, scale: f64) -> PointerOver {
        pointer_over(
            PhysicalPosition::new(cursor_x, cursor_y),
            PhysicalPosition::new(1000, 500),
            Some(PILL),
            scale,
        )
    }

    #[test]
    fn only_the_pill_itself_takes_clicks() {
        assert!(over(1080.0, 524.0, 2.0).on_pill, "the pill's top-left corner");
        assert!(over(1239.0, 563.0, 2.0).on_pill, "the pill's last pixel");
        assert!(!over(1240.0, 540.0, 2.0).on_pill, "one pixel past its right edge");
        assert!(!over(1079.0, 540.0, 2.0).on_pill, "one pixel before its left edge");
        assert!(!over(1150.0, 564.0, 2.0).on_pill, "one pixel below it");
    }

    #[test]
    fn the_hover_zone_reaches_the_margin_scaled_to_physical_pixels() {
        assert!(over(1080.0 - 48.0, 524.0 - 32.0, 2.0).in_hover_zone);
        assert!(!over(1080.0 - 49.0, 540.0, 2.0).in_hover_zone);
        assert!(!over(1150.0, 524.0 - 33.0, 2.0).in_hover_zone);
        assert!(over(1240.0 + 23.0, 540.0, 1.0).in_hover_zone);
        assert!(!over(1240.0 + 24.0, 540.0, 1.0).in_hover_zone);
        assert!(!over(1240.0 + 24.0, 540.0, 1.0).on_pill);
    }

    #[test]
    fn before_the_page_reports_the_pill_the_window_takes_no_click() {
        let before_report = pointer_over(
            PhysicalPosition::new(1100.0, 540.0),
            PhysicalPosition::new(1000, 500),
            None,
            1.0,
        );
        assert_eq!(before_report, PointerOver { on_pill: false, in_hover_zone: false });
        assert_eq!(pointer_changes(&Applied::default(), before_report).ignore, Some(true));
    }

    #[test]
    fn each_answer_is_applied_only_when_it_changes() {
        let on_pill = PointerOver { on_pill: true, in_hover_zone: true };
        let near_pill = PointerOver { on_pill: false, in_hover_zone: true };

        assert_eq!(
            pointer_changes(&Applied::default(), on_pill),
            PointerChanges { ignore: Some(false), hover: Some(true) }
        );
        let applied = Applied { ignoring: Some(false), hovering: Some(true) };
        assert_eq!(
            pointer_changes(&applied, on_pill),
            PointerChanges { ignore: None, hover: None }
        );
        assert_eq!(
            pointer_changes(&applied, near_pill),
            PointerChanges { ignore: Some(true), hover: None }
        );
    }

    #[test]
    fn a_cursor_scaled_by_another_display_lands_on_the_pill_it_is_over() {
        let cursor_from_a_retina_primary = PhysicalPosition::new(4200.0, 1100.0);
        let cursor = cursor_in_window_pixels(cursor_from_a_retina_primary, 2.0, 1.0);
        assert_eq!(cursor, PhysicalPosition::new(2100.0, 550.0));

        let on_a_plain_second_display = pointer_over(
            cursor,
            PhysicalPosition::new(2000, 500),
            Some(PillRect { x: 40.0, y: 12.0, width: 160.0, height: 40.0 }),
            1.0,
        );
        assert!(on_a_plain_second_display.on_pill);
        assert_eq!(
            cursor_in_window_pixels(PhysicalPosition::new(1024.0, 912.0), 1.5, 1.5),
            PhysicalPosition::new(1024.0, 912.0)
        );
    }

    #[test]
    fn the_pill_centre_sits_centred_and_above_the_work_area_bottom() {
        let work_area = PhysicalRect {
            position: PhysicalPosition::new(0, 0),
            size: PhysicalSize::new(3840, 2064),
        };
        let origin = window_origin(&work_area, 1.5);
        assert_eq!(origin, PhysicalPosition::new(1920 - 180, 2064 - 288 - 48));

        let second_monitor = PhysicalRect {
            position: PhysicalPosition::new(-1920, 100),
            size: PhysicalSize::new(1920, 1032),
        };
        assert_eq!(
            window_origin(&second_monitor, 1.0),
            PhysicalPosition::new(-1920 + 960 - 120, 100 + 1032 - 192 - 32)
        );
    }
}
