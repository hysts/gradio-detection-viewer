"""Tests for edge cases and custom constructor parameters."""

from __future__ import annotations

import re

import gradio as gr
from _helpers import (
    COCO_SKELETON,
    GradioApp,
    make_grid_image,
    make_person_keypoints,
)
from playwright.sync_api import Browser, Page, expect

from detection_viewer import DetectionViewer

# ── Empty annotations list ──


def test_empty_annotations_shows_content(browser: Browser):
    """An empty annotations list should still show the canvas but no rows."""
    value = (make_grid_image(), [])
    with gr.Blocks() as demo:
        DetectionViewer(value=value, label="Viewer")
    with GradioApp(demo, browser) as page:
        canvas = page.locator(".canvas-wrapper")
        expect(canvas).to_be_visible()

        # No annotation rows
        rows = page.locator(".annotation-row")
        assert rows.count() == 0


def test_empty_annotations_hides_control_panel(browser: Browser):
    """An empty annotations list should not show the control panel."""
    value = (make_grid_image(), [])
    with gr.Blocks() as demo:
        DetectionViewer(value=value, label="Viewer")
    with GradioApp(demo, browser) as page:
        cp = page.locator(".control-panel")
        expect(cp).not_to_have_class(re.compile("visible"))


# ── Annotations without scores ──


def test_annotations_without_score(browser: Browser):
    """Annotations without score field should still render without errors."""
    annotations = [
        {"bbox": {"x": 10, "y": 10, "width": 100, "height": 100}, "label": "obj1"},
        {"bbox": {"x": 200, "y": 200, "width": 80, "height": 80}, "label": "obj2"},
    ]
    value = (make_grid_image(), annotations)

    with gr.Blocks() as demo:
        DetectionViewer(value=value, label="Viewer")
    with GradioApp(demo, browser) as page:
        rows = page.locator(".annotation-row")
        expect(rows).to_have_count(2)

        # No score in header stats (just count)
        count_text = page.locator(".control-panel-count").text_content()
        assert "2" in count_text


def test_no_sort_controls_without_both_score_and_bbox(browser: Browser):
    """Sort controls should not appear when there is only one sort dimension."""
    annotations = [
        {"keypoints": make_person_keypoints(200, 200), "connections": COCO_SKELETON, "label": "p1"},
        {"keypoints": make_person_keypoints(400, 200), "connections": COCO_SKELETON, "label": "p2"},
    ]
    value = (make_grid_image(), annotations)

    with gr.Blocks() as demo:
        DetectionViewer(value=value, label="Viewer")
    with GradioApp(demo, browser) as page:
        sort = page.locator(".sort-controls")
        assert sort.count() == 0


# ── Annotations without bbox (keypoints only) ──


def test_keypoints_only_no_bbox(browser: Browser):
    """Annotations with only keypoints and no bbox should render."""
    annotations = [
        {
            "keypoints": make_person_keypoints(200, 200),
            "connections": COCO_SKELETON,
            "label": "person",
        },
    ]
    value = (make_grid_image(), annotations)

    with gr.Blocks() as demo:
        DetectionViewer(value=value, label="Viewer")
    with GradioApp(demo, browser) as page:
        rows = page.locator(".annotation-row")
        expect(rows).to_have_count(1)

        labels = page.locator(".ann-label")
        expect(labels.first).to_have_text("person")


# ── Custom panel_title ──


def test_custom_panel_title(browser: Browser):
    """Custom panel_title should be displayed in the header."""
    annotations = [
        {"bbox": {"x": 10, "y": 10, "width": 50, "height": 50}, "label": "item"},
    ]
    value = (make_grid_image(), annotations)

    with gr.Blocks() as demo:
        DetectionViewer(value=value, label="Viewer", panel_title="My Custom Title")
    with GradioApp(demo, browser) as page:
        title = page.locator(".control-panel-title")
        expect(title).to_have_text("My Custom Title")


# ── Custom keypoint_threshold ──


def test_custom_keypoint_threshold(browser: Browser):
    """Custom keypoint_threshold should affect the initial slider value."""
    annotations = [
        {
            "keypoints": make_person_keypoints(200, 200),
            "connections": COCO_SKELETON,
            "label": "person",
        },
    ]
    value = (make_grid_image(), annotations)

    with gr.Blocks() as demo:
        DetectionViewer(value=value, label="Viewer", keypoint_threshold=0.5)
    with GradioApp(demo, browser) as page:
        kp_label = page.locator(".keypoint-threshold-value")
        expect(kp_label).to_have_text("50%")


# ── Custom keypoint_radius ──


def test_custom_keypoint_radius(browser: Browser):
    """Custom keypoint_radius should set the initial slider value."""
    annotations = [
        {
            "keypoints": make_person_keypoints(200, 200),
            "connections": COCO_SKELETON,
            "label": "person",
        },
    ]
    value = (make_grid_image(), annotations)

    with gr.Blocks() as demo:
        DetectionViewer(value=value, label="Viewer", keypoint_radius=8)
    with GradioApp(demo, browser) as page:
        page.locator(".draw-options-toggle").click()
        kr_label = page.locator(".keypoint-radius-value")
        expect(kr_label).to_have_text("8")


# ── Shift+Click to hide annotation ──


def test_shift_click_hides_annotation(browser: Browser):
    """Shift+clicking on an annotation in the canvas should hide it."""
    annotations = [
        {"bbox": {"x": 50, "y": 50, "width": 200, "height": 200}, "score": 0.95, "label": "big"},
    ]
    value = (make_grid_image(), annotations)

    with gr.Blocks() as demo:
        DetectionViewer(value=value, label="Viewer")
    with GradioApp(demo, browser) as page:
        checkbox = page.locator(".ann-checkbox").first
        expect(checkbox).to_be_checked()

        # Shift+click on the canvas where the bbox is
        canvas = page.locator("canvas")
        box = canvas.bounding_box()
        # Click roughly in the center of the bbox area
        click_x = box["x"] + box["width"] * 0.25
        click_y = box["y"] + box["height"] * 0.35
        page.keyboard.down("Shift")
        page.mouse.click(click_x, click_y)
        page.keyboard.up("Shift")
        page.wait_for_timeout(300)

        checkbox = page.locator(".ann-checkbox").first
        expect(checkbox).not_to_be_checked()


# ── Single annotation ──


def test_single_annotation(browser: Browser):
    """A single annotation should render correctly."""
    annotations = [
        {"bbox": {"x": 50, "y": 50, "width": 100, "height": 100}, "score": 0.99, "label": "sole"},
    ]
    value = (make_grid_image(), annotations)

    with gr.Blocks() as demo:
        DetectionViewer(value=value, label="Viewer")
    with GradioApp(demo, browser) as page:
        rows = page.locator(".annotation-row")
        expect(rows).to_have_count(1)

        count = page.locator(".control-panel-count")
        assert "1" in count.text_content()


# ── Header stats show score range ──


def test_header_stats_score_range(detection_app: Page):
    """Header stats should display the score range."""
    count_text = detection_app.locator(".control-panel-count").text_content()
    # detection_app has scores: 0.95, 0.88, 0.72, 0.65
    # Expected format: "4 / 4 . 65-95%"
    assert "4 / 4" in count_text
    assert "65" in count_text
    assert "95" in count_text
