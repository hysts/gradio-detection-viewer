"""Shared fixtures for Playwright UI tests."""

from __future__ import annotations

import gradio as gr
import pytest
from _helpers import (
    GradioApp,
    make_detections,
    make_grid_image,
    make_pose_annotations,
    make_segmentation_annotations,
)
from playwright.sync_api import Browser, sync_playwright

from detection_viewer import DetectionViewer


@pytest.fixture(scope="session")
def browser():
    with sync_playwright() as p:
        b = p.chromium.launch()
        yield b
        b.close()


@pytest.fixture
def detection_app(browser: Browser):
    """App with 4 bbox-only detections (person, dog, chair, bench)."""
    value = (
        make_grid_image(),
        [
            {"bbox": {"x": 50, "y": 30, "width": 200, "height": 280}, "score": 0.95, "label": "person"},
            {"bbox": {"x": 300, "y": 100, "width": 150, "height": 150}, "score": 0.88, "label": "dog"},
            {"bbox": {"x": 480, "y": 50, "width": 120, "height": 200}, "score": 0.72, "label": "chair"},
            {"bbox": {"x": 100, "y": 350, "width": 250, "height": 100}, "score": 0.65, "label": "bench"},
        ],
    )
    with gr.Blocks() as demo:
        DetectionViewer(value=value, label="Viewer")
    with GradioApp(demo, browser) as page:
        yield page


@pytest.fixture
def pose_app(browser: Browser):
    """App with 2 pose annotations (bbox + keypoints + skeleton)."""
    value = (make_grid_image(), make_pose_annotations())
    with gr.Blocks() as demo:
        DetectionViewer(value=value, label="Viewer")
    with GradioApp(demo, browser) as page:
        yield page


@pytest.fixture
def segmentation_app(browser: Browser):
    """App with 2 segmentation annotations (bbox + mask)."""
    value = (make_grid_image(), make_segmentation_annotations())
    with gr.Blocks() as demo:
        DetectionViewer(value=value, label="Viewer")
    with GradioApp(demo, browser) as page:
        yield page


@pytest.fixture
def many_detections_app(browser: Browser):
    """App with 20 bbox detections across 8 labels."""
    value = (make_grid_image(), make_detections(20))
    with gr.Blocks() as demo:
        DetectionViewer(value=value, label="Viewer")
    with GradioApp(demo, browser) as page:
        yield page


@pytest.fixture
def empty_app(browser: Browser):
    """App with no initial value."""
    with gr.Blocks() as demo:
        DetectionViewer(value=None, label="Viewer")
    with GradioApp(demo, browser) as page:
        yield page
