//! The ring window: the widget's actions in a slim column that opens in the
//! widget's place on the screen edge, from a right-click on the widget. It is
//! built once and kept hidden, and like the widget it never takes focus. While
//! it is open a loop sends the page the pointer's offset from the column's
//! anchor, which the page turns into the hovered button, and closes the column
//! on a mouse press outside it. Escape is a global shortcut held only while the
//! column is open, so the key never reaches the user's app.

use std::sync::{
    atomic::{AtomicU64, Ordering},
    Mutex,
};
use std::time::Duration;

use serde::Serialize;
use tauri::{webview::WebviewWindowBuilder, AppHandle, Emitter, Manager, WebviewUrl};
use tauri_plugin_global_shortcut::{Code, GlobalShortcutExt, Shortcut, ShortcutState};

#[cfg(target_os = "macos")]
use tauri_nspanel::tauri_panel;

const RING_LABEL: &str = "ring";

const WIDGET_LABEL: &str = "widget";

/// The window's logical width: the column and the longest label left of it.
const WINDOW_WIDTH: f64 = 280.0;

/// The window's logical height: the column and room for its shadow.
const WINDOW_HEIGHT: f64 = 180.0;

/// The column's logical width, measured from the edge it docks on.
const COLUMN_WIDTH: f64 = 37.0;

/// How far the column reaches above and below the widget's centre line, logical.
const COLUMN_HALF_HEIGHT: f64 = 77.0;

/// The pointer loop's period, about 60 times a second.
const POINTER_POLL: Duration = Duration::from_millis(16);

static NEXT_GENERATION: AtomicU64 = AtomicU64::new(1);

/// The column on screen, if any. `generation` tells one opening from the next,
/// so a loop or a close meant for an earlier opening leaves a newer one alone.
static OPEN_RING: Mutex<Option<OpenRing>> = Mutex::new(None);

#[cfg(target_os = "macos")]
tauri_panel! {
    panel!(RingPanel {
        config: {
            can_become_key_window: false,
            can_become_main_window: false,
            is_floating_panel: true,
            hides_on_deactivate: false
        }
    })
}

/// A point in screen units: physical pixels on Windows, points on macOS.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Point {
    x: f64,
    y: f64,
}

/// A window's client area in the same screen units as `Point`.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Area {
    x: f64,
    y: f64,
    width: f64,
    height: f64,
}

/// Where the ring window goes: its top-left corner and its size, in screen units.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Frame {
    x: f64,
    y: f64,
    width: f64,
    height: f64,
}

/// The widget's client area, with `scale` screen units per logical pixel.
struct Widget {
    area: Area,
    scale: f64,
}

#[derive(Clone, Copy)]
struct OpenRing {
    generation: u64,
    anchor: Point,
    scale: f64,
}

/// Payload of the event that tells the page where the pointer is, in logical
/// pixels from the column's anchor.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
struct RingPointer {
    x: f64,
    y: f64,
}

/// Build the hidden ring window. On macOS it becomes a non-activating panel
/// that never becomes key, the same as the widget.
pub fn build(app: &tauri::App) -> tauri::Result<()> {
    #[cfg_attr(not(target_os = "macos"), allow(unused_variables))]
    let ring = WebviewWindowBuilder::new(app, "ring", WebviewUrl::App("/ring.html".into()))
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

        let panel = ring.to_panel::<RingPanel>()?;
        panel
            .add_style_mask(StyleMask::empty().nonactivating_panel().into())
            .map_err(|e| tauri::Error::Anyhow(e.into()))?;
    }

    Ok(())
}

/// Open the column in the widget's place: its right side on the widget's,
/// centred on the widget's height. Opening it while it is open places it again.
#[tauri::command]
pub fn open_ring(app: AppHandle) {
    let Some(ring) = app.get_webview_window(RING_LABEL) else {
        return;
    };
    let Some(widget) = platform::widget(&app) else {
        log::warn!("Reading where the widget is failed, so the actions stay closed");
        return;
    };
    let anchor = column_anchor(widget.area);
    if let Err(e) = platform::show_at(&app, &ring, ring_frame(anchor, widget.scale)) {
        log::warn!("Showing the actions failed: {}", e);
        return;
    }

    let generation = NEXT_GENERATION.fetch_add(1, Ordering::SeqCst);
    *open_ring_state() = Some(OpenRing { generation, anchor, scale: widget.scale });
    hold_escape(&app);
    if let Err(e) = app.emit("ring-opened", ()) {
        log::warn!("Telling the windows the actions opened failed: {}", e);
    }
    follow_the_pointer(app, generation);
}

/// Close the column; the page calls this after a button is picked and on a
/// right click on the mic.
#[tauri::command]
pub fn close_ring(app: AppHandle) {
    close(&app, None);
}

fn open_ring_state() -> std::sync::MutexGuard<'static, Option<OpenRing>> {
    OPEN_RING.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Close the column on the main thread, where the window and the shortcut live.
/// `generation` limits it to that opening; `None` closes whatever is open.
fn close(app: &AppHandle, generation: Option<u64>) {
    let handle = app.clone();
    let result = app.run_on_main_thread(move || {
        {
            let mut open = open_ring_state();
            match *open {
                Some(ring) if generation.map_or(true, |g| g == ring.generation) => *open = None,
                _ => return,
            }
        }
        if let Some(ring) = handle.get_webview_window(RING_LABEL) {
            if let Err(e) = platform::hide(&handle, &ring) {
                log::warn!("Hiding the actions failed: {}", e);
            }
        }
        release_escape(&handle);
        if let Err(e) = handle.emit("ring-closed", ()) {
            log::warn!("Telling the windows the actions closed failed: {}", e);
        }
    });
    if let Err(e) = result {
        log::warn!("Closing the actions failed: {}", e);
    }
}

fn escape() -> Shortcut {
    Shortcut::new(None, Code::Escape)
}

/// Take Escape for as long as the column is open. The plugin runs the handler
/// under its own lock, which releasing the shortcut needs, so the close is
/// handed to another thread.
fn hold_escape(app: &AppHandle) {
    let shortcuts = app.global_shortcut();
    if shortcuts.is_registered(escape()) {
        return;
    }
    let registered = shortcuts.on_shortcut(escape(), |app, _shortcut, event| {
        if event.state == ShortcutState::Pressed {
            let app = app.clone();
            std::thread::spawn(move || close(&app, None));
        }
    });
    if let Err(e) = registered {
        log::warn!("Escape could not be taken, so only a click closes the actions: {}", e);
    }
}

fn release_escape(app: &AppHandle) {
    let shortcuts = app.global_shortcut();
    if !shortcuts.is_registered(escape()) {
        return;
    }
    if let Err(e) = shortcuts.unregister(escape()) {
        log::warn!("Releasing Escape after the actions closed failed: {}", e);
    }
}

/// While this opening lasts, send the page the pointer whenever it moves, and
/// close the column when a mouse button goes down outside it.
fn follow_the_pointer(app: AppHandle, generation: u64) {
    std::thread::spawn(move || {
        let mut was_down = platform::mouse_button_down();
        let mut sent: Option<RingPointer> = None;
        loop {
            std::thread::sleep(POINTER_POLL);
            let Some(ring) = *open_ring_state() else {
                return;
            };
            if ring.generation != generation {
                return;
            }
            let Some(cursor) = platform::cursor(&app) else {
                continue;
            };
            let pointer = pointer_offset(cursor, ring.anchor, ring.scale);
            let down = platform::mouse_button_down();
            if pressed_outside_column(was_down, down, pointer) {
                close(&app, Some(generation));
                return;
            }
            was_down = down;
            if sent != Some(pointer) {
                if let Err(e) = app.emit_to("ring", "ring-pointer", pointer) {
                    log::warn!("Telling the actions where the pointer is failed: {}", e);
                }
                sent = Some(pointer);
            }
        }
    });
}

/// Where the column docks: the widget's right side, on its centre line, where
/// the page draws the pill.
fn column_anchor(widget: Area) -> Point {
    Point { x: widget.x + widget.width, y: widget.y + widget.height / 2.0 }
}

/// The window left of the anchor and centred on it: the page draws the column
/// flush with the window's right side, in the middle of its height.
fn ring_frame(anchor: Point, scale: f64) -> Frame {
    let (width, height) = (WINDOW_WIDTH * scale, WINDOW_HEIGHT * scale);
    Frame { x: anchor.x - width, y: anchor.y - height / 2.0, width, height }
}

/// The pointer's offset from the anchor in whole logical pixels, so sub-pixel
/// jitter sends the page nothing.
fn pointer_offset(cursor: Point, anchor: Point, scale: f64) -> RingPointer {
    RingPointer {
        x: ((cursor.x - anchor.x) / scale).round(),
        y: ((cursor.y - anchor.y) / scale).round(),
    }
}

fn over_column(pointer: RingPointer) -> bool {
    (-COLUMN_WIDTH..=0.0).contains(&pointer.x) && pointer.y.abs() <= COLUMN_HALF_HEIGHT
}

/// A button that went down on this tick, outside the column. One already held
/// when the column opened, the right-click that opened it, does not count.
fn pressed_outside_column(was_down: bool, down: bool, pointer: RingPointer) -> bool {
    down && !was_down && !over_column(pointer)
}

/// Bring the ring's panel to the front without making it key, or take it away.
#[cfg(target_os = "macos")]
fn order_panel(app: &AppHandle, front: bool) -> Result<(), String> {
    use tauri_nspanel::ManagerExt;

    let panel = app.get_webview_panel(RING_LABEL).map_err(|e| format!("{:?}", e))?;
    if front {
        panel.show();
    } else {
        panel.hide();
    }
    Ok(())
}

#[cfg(windows)]
mod platform {
    use super::{Area, Frame, Point, Widget, WIDGET_LABEL};
    use tauri::{AppHandle, Manager, WebviewWindow};
    use windows_sys::Win32::UI::Input::KeyboardAndMouse::{
        GetAsyncKeyState, VK_LBUTTON, VK_MBUTTON, VK_RBUTTON,
    };
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        SetWindowPos, HWND_TOPMOST, SET_WINDOW_POS_FLAGS, SWP_HIDEWINDOW, SWP_NOACTIVATE,
        SWP_NOMOVE, SWP_NOSIZE, SWP_NOZORDER, SWP_SHOWWINDOW,
    };

    const KEY_DOWN: u16 = 0x8000;

    pub(super) fn cursor(app: &AppHandle) -> Option<Point> {
        let cursor = app.cursor_position().ok()?;
        Some(Point { x: cursor.x, y: cursor.y })
    }

    /// The widget's client area in physical pixels, at its monitor's scale.
    pub(super) fn widget(app: &AppHandle) -> Option<Widget> {
        let widget = app.get_webview_window(WIDGET_LABEL)?;
        let origin = widget.inner_position().ok()?;
        let size = widget.inner_size().ok()?;
        Some(Widget {
            area: Area {
                x: f64::from(origin.x),
                y: f64::from(origin.y),
                width: f64::from(size.width),
                height: f64::from(size.height),
            },
            scale: widget.scale_factor().ok()?,
        })
    }

    /// Placed, sized and shown in one `SetWindowPos` in physical pixels, then
    /// placed again: tao answers the move to a monitor of another scale by
    /// moving the window itself. Neither call activates it.
    pub(super) fn show_at(_app: &AppHandle, ring: &WebviewWindow, frame: Frame) -> Result<(), String> {
        let place = (
            frame.x.round() as i32,
            frame.y.round() as i32,
            frame.width.round() as i32,
            frame.height.round() as i32,
        );
        set_window_pos(ring, place, SWP_NOACTIVATE | SWP_SHOWWINDOW)?;
        set_window_pos(ring, place, SWP_NOACTIVATE)
    }

    pub(super) fn hide(_app: &AppHandle, ring: &WebviewWindow) -> Result<(), String> {
        set_window_pos(
            ring,
            (0, 0, 0, 0),
            SWP_NOACTIVATE | SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_HIDEWINDOW,
        )
    }

    fn set_window_pos(
        ring: &WebviewWindow,
        (x, y, width, height): (i32, i32, i32, i32),
        flags: SET_WINDOW_POS_FLAGS,
    ) -> Result<(), String> {
        let hwnd = ring.hwnd().map_err(|e| e.to_string())?;
        let placed = unsafe { SetWindowPos(hwnd.0 as _, HWND_TOPMOST, x, y, width, height, flags) };
        if placed == 0 {
            return Err(std::io::Error::last_os_error().to_string());
        }
        Ok(())
    }

    pub(super) fn mouse_button_down() -> bool {
        [VK_LBUTTON, VK_RBUTTON, VK_MBUTTON]
            .iter()
            .any(|&vk| unsafe { GetAsyncKeyState(i32::from(vk)) } as u16 & KEY_DOWN != 0)
    }
}

#[cfg(target_os = "macos")]
mod platform {
    use super::{Area, Frame, Point, Widget, WIDGET_LABEL};
    use objc2_app_kit::NSEvent;
    use tauri::{AppHandle, LogicalPosition, Manager, WebviewWindow};

    /// The cursor in points. tao reports it scaled by the primary display's
    /// factor, whatever display it is on.
    pub(super) fn cursor(app: &AppHandle) -> Option<Point> {
        let cursor = app.cursor_position().ok()?;
        let scale = app.primary_monitor().ok().flatten()?.scale_factor();
        Some(Point { x: cursor.x / scale, y: cursor.y / scale })
    }

    /// The widget's client area in points, where one logical pixel is one unit
    /// on every display; tao reports a window scaled by its own display's factor.
    pub(super) fn widget(app: &AppHandle) -> Option<Widget> {
        let widget = app.get_webview_window(WIDGET_LABEL)?;
        let scale = widget.scale_factor().ok()?;
        let origin = widget.inner_position().ok()?;
        let size = widget.inner_size().ok()?;
        Some(Widget {
            area: Area {
                x: f64::from(origin.x) / scale,
                y: f64::from(origin.y) / scale,
                width: f64::from(size.width) / scale,
                height: f64::from(size.height) / scale,
            },
            scale: 1.0,
        })
    }

    pub(super) fn show_at(app: &AppHandle, ring: &WebviewWindow, frame: Frame) -> Result<(), String> {
        ring.set_position(LogicalPosition::new(frame.x, frame.y))
            .map_err(|e| e.to_string())?;
        super::order_panel(app, true)
    }

    pub(super) fn hide(app: &AppHandle, _ring: &WebviewWindow) -> Result<(), String> {
        super::order_panel(app, false)
    }

    pub(super) fn mouse_button_down() -> bool {
        NSEvent::pressedMouseButtons() != 0
    }
}

#[cfg(test)]
mod tests {
    use super::{
        column_anchor, pointer_offset, pressed_outside_column, ring_frame, Area, Frame, Point,
        RingPointer,
    };

    #[test]
    fn the_column_docks_on_the_widgets_right_side_at_its_centre_line() {
        let widget = Area { x: 1680.0, y: 587.0, width: 240.0, height: 64.0 };
        assert_eq!(column_anchor(widget), Point { x: 1920.0, y: 619.0 });
    }

    #[test]
    fn the_window_hangs_left_of_the_anchor_centred_on_it() {
        assert_eq!(
            ring_frame(Point { x: 1920.0, y: 619.0 }, 1.0),
            Frame { x: 1640.0, y: 529.0, width: 280.0, height: 180.0 }
        );
    }

    #[test]
    fn the_window_scales_to_physical_pixels() {
        let widget = Area { x: 3240.0, y: 1190.0, width: 360.0, height: 96.0 };
        assert_eq!(
            ring_frame(column_anchor(widget), 1.5),
            Frame { x: 3180.0, y: 1103.0, width: 420.0, height: 270.0 }
        );
    }

    #[test]
    fn a_widget_on_a_monitor_left_of_the_primary_keeps_the_column_beside_it() {
        let widget = Area { x: -240.0, y: 700.0, width: 240.0, height: 64.0 };
        assert_eq!(
            ring_frame(column_anchor(widget), 1.0),
            Frame { x: -280.0, y: 642.0, width: 280.0, height: 180.0 }
        );
    }

    #[test]
    fn the_pointer_is_sent_in_whole_logical_pixels_from_the_anchor() {
        let anchor = Point { x: 2880.0, y: 900.0 };
        assert_eq!(
            pointer_offset(Point { x: 2851.5, y: 855.0 }, anchor, 1.5),
            RingPointer { x: -19.0, y: -30.0 }
        );
        assert_eq!(
            pointer_offset(Point { x: 2879.6, y: 900.2 }, anchor, 1.0),
            RingPointer { x: -0.0, y: 0.0 }
        );
    }

    #[test]
    fn only_a_fresh_press_outside_the_column_closes_it() {
        let on_the_mic = RingPointer { x: -19.0, y: 0.0 };
        let on_the_top_button = RingPointer { x: -19.0, y: -60.0 };
        let column_corners = [
            RingPointer { x: -37.0, y: -77.0 },
            RingPointer { x: 0.0, y: 77.0 },
        ];
        let beside_on_the_labels = RingPointer { x: -38.0, y: 0.0 };
        let above = RingPointer { x: -19.0, y: -78.0 };
        let below = RingPointer { x: -19.0, y: 78.0 };
        let past_the_edge = RingPointer { x: 1.0, y: 0.0 };

        for inside in [on_the_mic, on_the_top_button, column_corners[0], column_corners[1]] {
            assert!(!pressed_outside_column(false, true, inside), "{inside:?} is on the column");
        }
        for outside in [beside_on_the_labels, above, below, past_the_edge] {
            assert!(pressed_outside_column(false, true, outside), "{outside:?} is off the column");
        }
        assert!(
            !pressed_outside_column(true, true, beside_on_the_labels),
            "the right-click still held from opening"
        );
        assert!(!pressed_outside_column(false, false, beside_on_the_labels));
    }
}
