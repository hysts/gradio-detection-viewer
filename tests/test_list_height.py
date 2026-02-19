"""Tests for list_height parameter controlling annotation rows max-height."""

from __future__ import annotations

import gradio as gr
from conftest import GradioApp, make_detections, make_grid_image
from playwright.sync_api import Browser

from detection_viewer import DetectionViewer


def _measure(browser: Browser, list_height: int, n_detections: int) -> dict:
    value = (make_grid_image(), make_detections(n_detections))
    with gr.Blocks() as demo:
        DetectionViewer(value=value, label="Viewer", list_height=list_height)
    with GradioApp(demo, browser) as page:
        return page.evaluate("""() => {
            const rows = document.querySelector('.annotation-rows');
            const annList = document.querySelector('.annotation-list');
            if (!rows) return { error: '.annotation-rows not found' };
            const cs = getComputedStyle(rows);
            return {
                cssMaxHeight: cs.maxHeight,
                rowsHeight: rows.getBoundingClientRect().height,
                rowsScrollHeight: rows.scrollHeight,
                annListHeight: annList.getBoundingClientRect().height,
            };
        }""")


def test_max_height_applied(browser: Browser):
    """list_height=300 should set max-height on .annotation-rows."""
    r = _measure(browser, list_height=300, n_detections=20)
    assert r["cssMaxHeight"] == "300px"


def test_rows_capped_at_max_height(browser: Browser):
    """With many items, rows should be capped and scrollable."""
    r = _measure(browser, list_height=300, n_detections=20)
    assert r["rowsHeight"] == 300
    assert r["rowsScrollHeight"] > 300


def test_controls_outside_scroll_area(browser: Browser):
    """Controls (filters, sliders) should be outside the scrollable rows area."""
    r = _measure(browser, list_height=300, n_detections=20)
    assert r["annListHeight"] > r["rowsHeight"]


def test_custom_height_changes_rows(browser: Browser):
    """Different list_height values should produce different row area sizes."""
    r300 = _measure(browser, list_height=300, n_detections=20)
    r500 = _measure(browser, list_height=500, n_detections=20)
    assert r500["rowsHeight"] > r300["rowsHeight"]


def test_few_items_shrinks(browser: Browser):
    """With few items, rows should shrink below max-height."""
    r = _measure(browser, list_height=300, n_detections=2)
    assert r["rowsHeight"] < 300
