"""Tests for dynamic value updates via Gradio interactions."""

from __future__ import annotations

import re

import gradio as gr
from _helpers import (
    GradioApp,
    make_detections,
    make_grid_image,
    make_pose_annotations,
)
from playwright.sync_api import Browser, expect

from detection_viewer import DetectionViewer


def test_update_from_detections_to_pose(browser: Browser):
    """Updating value should re-render with new annotation type."""
    det_value = (make_grid_image(), make_detections(4))
    pose_value = (make_grid_image(), make_pose_annotations())

    with gr.Blocks() as demo:
        viewer = DetectionViewer(value=det_value, label="Viewer")
        btn = gr.Button("Switch to pose")
        btn.click(fn=lambda: pose_value, outputs=viewer)

    with GradioApp(demo, browser) as page:
        # Initially: 4 detection rows, no layer toggles
        rows = page.locator(".annotation-row")
        expect(rows).to_have_count(4)
        expect(page.locator(".layer-toggles")).to_have_count(0)

        # Click the button to switch
        page.get_by_role("button", name="Switch to pose").click()

        # After update: 2 pose rows, layer toggles visible
        rows = page.locator(".annotation-row")
        expect(rows).to_have_count(2)
        expect(page.locator(".layer-toggles")).to_be_visible()


def test_update_to_none_shows_placeholder(browser: Browser):
    """Updating value to None should show placeholder."""
    value = (make_grid_image(), make_detections(4))

    with gr.Blocks() as demo:
        viewer = DetectionViewer(value=value, label="Viewer")
        btn = gr.Button("Clear")
        btn.click(fn=lambda: None, outputs=viewer)

    with GradioApp(demo, browser) as page:
        expect(page.locator(".control-panel")).to_be_visible()

        page.get_by_role("button", name="Clear").click()

        placeholder = page.locator(".placeholder")
        expect(placeholder).to_be_visible()
        expect(page.locator(".control-panel")).to_be_hidden()


def test_update_to_none_exits_maximize(browser: Browser):
    """Clearing the value while maximized should restore normal page state."""
    value = (make_grid_image(), make_detections(4))

    with gr.Blocks() as demo:
        viewer = DetectionViewer(value=value, label="Viewer")
        btn = gr.Button("Clear")
        btn.click(fn=lambda: None, outputs=viewer)

    with GradioApp(demo, browser) as page:
        wrapper = page.locator(".pose-viewer-container").first.locator("xpath=..")
        page.locator(".maximize-btn").click()
        expect(wrapper).to_have_class(re.compile("maximized"))

        # Use dispatch_event because the maximized overlay (z-index: 9999)
        # covers the button, blocking normal .click() pointer-event checks.
        page.get_by_role("button", name="Clear").dispatch_event("click")

        expect(wrapper).not_to_have_class(re.compile("maximized"))
        expect(page.locator(".placeholder")).to_be_visible()
        assert page.evaluate("() => document.body.style.overflow") == ""
