"""Tests for keyboard shortcuts not covered by test_controls.py (H, +, -, 0, Escape deselect)."""

from __future__ import annotations

import re

from playwright.sync_api import Page, expect


def _focus_viewer(page: Page) -> None:
    """Click the canvas area to ensure the component has keyboard focus."""
    page.locator(".canvas-wrapper").click()


def _focus_without_canvas_click(page: Page) -> None:
    """Focus the component element for keyboard events without triggering canvas mouse handlers."""
    page.evaluate("() => document.querySelector('.pose-viewer-container').parentElement.focus()")


# ── H key: hide selected annotation ──


def test_key_h_hides_selected(detection_app: Page):
    """Pressing 'h' should hide the selected annotation."""
    # Select the first annotation row
    row = detection_app.locator(".annotation-row").first
    row.click()
    expect(row).to_have_class(re.compile("selected"))

    # Focus the component without clicking the canvas (which would deselect)
    _focus_without_canvas_click(detection_app)
    detection_app.keyboard.press("h")

    # The first checkbox should now be unchecked
    checkbox = detection_app.locator(".ann-checkbox").first
    expect(checkbox).not_to_be_checked()

    # No row should be selected after hiding
    rows = detection_app.locator(".annotation-row.selected")
    assert rows.count() == 0


def test_key_h_no_effect_without_selection(detection_app: Page):
    """Pressing 'h' without a selected annotation should not change any checkbox."""
    _focus_without_canvas_click(detection_app)

    # All checkboxes should be checked initially
    checkboxes = detection_app.locator(".ann-checkbox")
    for i in range(checkboxes.count()):
        expect(checkboxes.nth(i)).to_be_checked()

    detection_app.keyboard.press("h")

    # All should still be checked
    checkboxes = detection_app.locator(".ann-checkbox")
    for i in range(checkboxes.count()):
        expect(checkboxes.nth(i)).to_be_checked()


# ── Escape: deselect annotation ──


def test_escape_deselects_annotation(detection_app: Page):
    """Pressing Escape should deselect the currently selected annotation."""
    # Select the first annotation row
    row = detection_app.locator(".annotation-row").first
    row.click()
    expect(row).to_have_class(re.compile("selected"))

    _focus_without_canvas_click(detection_app)
    detection_app.keyboard.press("Escape")

    rows = detection_app.locator(".annotation-row.selected")
    assert rows.count() == 0


# ── Zoom keyboard shortcuts ──
# We verify zoom by moving the mouse to an empty canvas area (bottom-right corner,
# where no bboxes exist) and checking the cursor: "grab" when zoomed, "default" at 1x.


def _get_cursor_at_empty_area(page: Page) -> str:
    """Move mouse to the bottom-right corner of canvas (no annotations there) and return cursor."""
    canvas = page.locator("canvas")
    box = canvas.bounding_box()
    # Bottom-right of the canvas, outside all detection_app bboxes
    page.mouse.move(box["x"] + box["width"] - 5, box["y"] + box["height"] - 5)
    page.wait_for_timeout(50)
    return page.evaluate("() => document.querySelector('canvas').style.cursor")


def test_key_plus_zooms_in(detection_app: Page):
    """Pressing '+' should increase zoom level (cursor becomes 'grab' in empty area)."""
    _focus_viewer(detection_app)
    detection_app.keyboard.press("+")
    detection_app.wait_for_timeout(100)

    cursor = _get_cursor_at_empty_area(detection_app)
    assert cursor == "grab"


def test_key_minus_does_not_go_below_min(detection_app: Page):
    """Pressing '-' at minimum zoom should keep zoom at 1 (cursor stays 'default')."""
    _focus_viewer(detection_app)
    detection_app.keyboard.press("-")
    detection_app.wait_for_timeout(100)

    cursor = _get_cursor_at_empty_area(detection_app)
    assert cursor == "default"


def test_key_0_resets_zoom(detection_app: Page):
    """Pressing '0' should reset zoom to 1 after zooming in."""
    _focus_viewer(detection_app)

    # Zoom in first
    detection_app.keyboard.press("+")
    detection_app.keyboard.press("+")
    detection_app.wait_for_timeout(100)

    cursor_zoomed = _get_cursor_at_empty_area(detection_app)
    assert cursor_zoomed == "grab"

    # Reset zoom
    _focus_viewer(detection_app)
    detection_app.keyboard.press("0")
    detection_app.wait_for_timeout(100)

    cursor_reset = _get_cursor_at_empty_area(detection_app)
    assert cursor_reset == "default"


def test_key_plus_minus_round_trip(detection_app: Page):
    """Zooming in then resetting with '0' should return to normal state."""
    _focus_viewer(detection_app)

    detection_app.keyboard.press("+")
    detection_app.keyboard.press("+")
    detection_app.keyboard.press("+")
    detection_app.wait_for_timeout(100)

    cursor_zoomed = _get_cursor_at_empty_area(detection_app)
    assert cursor_zoomed == "grab"

    _focus_viewer(detection_app)
    detection_app.keyboard.press("0")
    detection_app.wait_for_timeout(100)

    cursor_reset = _get_cursor_at_empty_area(detection_app)
    assert cursor_reset == "default"
