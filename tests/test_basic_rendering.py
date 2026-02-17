"""Tests for basic component rendering and placeholder states."""

from __future__ import annotations

from playwright.sync_api import Page, expect


def test_placeholder_when_no_data(empty_app: Page):
    """Component should show placeholder text when no data is provided."""
    placeholder = empty_app.locator(".placeholder")
    expect(placeholder).to_be_visible()
    expect(placeholder).to_have_text("No data")


def test_canvas_hidden_when_no_data(empty_app: Page):
    """Canvas should not be visible when no data is provided."""
    canvas = empty_app.locator(".canvas-wrapper")
    expect(canvas).to_be_hidden()


def test_control_panel_hidden_when_no_data(empty_app: Page):
    """Control panel should not be visible when no data is provided."""
    cp = empty_app.locator(".control-panel")
    expect(cp).to_be_hidden()


def test_canvas_visible_with_data(detection_app: Page):
    """Canvas should be visible when data is provided."""
    canvas = detection_app.locator(".canvas-wrapper")
    expect(canvas).to_be_visible()


def test_control_panel_visible_with_data(detection_app: Page):
    """Control panel should be visible when data is provided."""
    cp = detection_app.locator(".control-panel")
    expect(cp).to_be_visible()


def test_placeholder_hidden_with_data(detection_app: Page):
    """Placeholder should be hidden when data is provided."""
    placeholder = detection_app.locator(".placeholder")
    expect(placeholder).to_have_class("placeholder hidden")


def test_annotation_row_count(detection_app: Page):
    """Number of annotation rows should match number of annotations."""
    rows = detection_app.locator(".annotation-row")
    expect(rows).to_have_count(4)


def test_annotation_labels_displayed(detection_app: Page):
    """Each annotation label should be displayed."""
    labels = detection_app.locator(".ann-label")
    texts = labels.all_text_contents()
    # Sorted by score desc by default: person(0.95), dog(0.88), chair(0.72), bench(0.65)
    assert "person" in texts
    assert "dog" in texts
    assert "chair" in texts
    assert "bench" in texts


def test_header_shows_count(detection_app: Page):
    """Header should display annotation count."""
    count = detection_app.locator(".control-panel-count")
    text = count.text_content()
    assert "4" in text


def test_panel_title_default(detection_app: Page):
    """Default panel title should be 'Detections'."""
    title = detection_app.locator(".control-panel-title")
    expect(title).to_have_text("Detections")
