"""Tests for UI elements that appear conditionally based on annotation types."""

from __future__ import annotations

import re

from playwright.sync_api import Page, expect


# ── Layer toggles ──


def test_no_layer_toggles_for_bbox_only(detection_app: Page):
    """Bbox-only annotations should not show layer toggles (only 1 type)."""
    assert detection_app.locator(".layer-toggles").count() == 0


def test_layer_toggles_for_pose(pose_app: Page):
    """Pose annotations (bbox + skeleton + keypoints) should show layer toggles."""
    toggles = pose_app.locator(".layer-toggles")
    expect(toggles).to_be_visible()

    btns = pose_app.locator(".layer-btn")
    assert btns.count() >= 2


def test_layer_toggle_click(pose_app: Page):
    """Clicking a layer toggle should deactivate it."""
    btn = pose_app.locator(".layer-btn").first
    expect(btn).to_have_class(re.compile("active"))

    btn.click()
    expect(btn).not_to_have_class(re.compile("active"))

    btn.click()
    expect(btn).to_have_class(re.compile("active"))


def test_layer_toggles_for_segmentation(segmentation_app: Page):
    """Segmentation (bbox + mask) should show layer toggles."""
    toggles = segmentation_app.locator(".layer-toggles")
    expect(toggles).to_be_visible()


# ── Sort controls ──


def test_sort_controls_for_detections_with_bbox(detection_app: Page):
    """Detections with scores and bboxes should show sort controls."""
    sort = detection_app.locator(".sort-controls")
    expect(sort).to_be_visible()

    btns = detection_app.locator(".sort-btn")
    assert btns.count() == 2  # score, size


def test_sort_by_score_toggles(detection_app: Page):
    """Clicking sort-by-score button should toggle asc/desc."""
    btn = detection_app.locator(".sort-btn[data-sort-key='score']")
    expect(btn).to_have_class(re.compile("active"))

    # Second click should toggle direction
    btn.click()
    text = btn.text_content()
    assert "\u25b2" in text  # ascending arrow


def test_sort_by_size(detection_app: Page):
    """Clicking sort-by-size button should activate it."""
    btn = detection_app.locator(".sort-btn[data-sort-key='size']")
    btn.click()
    expect(btn).to_have_class(re.compile("active"))


# ── Conditional draw options ──


def test_mask_opacity_slider_for_segmentation(segmentation_app: Page):
    """Segmentation annotations should show mask opacity slider in draw options."""
    segmentation_app.locator(".draw-options-toggle").click()
    slider = segmentation_app.locator(".mask-alpha-slider")
    expect(slider).to_be_visible()


def test_no_mask_slider_for_bbox_only(detection_app: Page):
    """Bbox-only annotations should not have mask opacity slider."""
    detection_app.locator(".draw-options-toggle").click()
    assert detection_app.locator(".mask-alpha-slider").count() == 0


def test_keypoint_controls_for_pose(pose_app: Page):
    """Pose annotations should show keypoint size and line width sliders."""
    pose_app.locator(".draw-options-toggle").click()
    expect(pose_app.locator(".keypoint-radius-slider")).to_be_visible()
    expect(pose_app.locator(".connection-width-slider")).to_be_visible()


def test_no_keypoint_controls_for_bbox_only(detection_app: Page):
    """Bbox-only annotations should not have keypoint/connection sliders."""
    detection_app.locator(".draw-options-toggle").click()
    assert detection_app.locator(".keypoint-radius-slider").count() == 0
    assert detection_app.locator(".connection-width-slider").count() == 0


def test_bbox_line_width_slider_present(detection_app: Page):
    """Bbox annotations should show bbox line width slider in draw options."""
    detection_app.locator(".draw-options-toggle").click()
    expect(detection_app.locator(".bbox-line-width-slider")).to_be_visible()
