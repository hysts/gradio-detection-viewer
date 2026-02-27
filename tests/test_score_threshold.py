"""Tests for score threshold slider and config override."""

from __future__ import annotations

import gradio as gr
from _helpers import GradioApp, make_grid_image
from playwright.sync_api import Browser, Page, expect

from detection_viewer import DetectionViewer

# ── Score threshold slider ──


def test_threshold_slider_present(detection_app: Page):
    """Score threshold sliders should be present."""
    slider_min = detection_app.locator(".threshold-slider-min")
    slider_max = detection_app.locator(".threshold-slider-max")
    expect(slider_min).to_be_attached()
    expect(slider_max).to_be_attached()


def test_threshold_label_shows_range(detection_app: Page):
    """Threshold label should display the current range."""
    label = detection_app.locator(".threshold-value")
    text = label.text_content()
    # Default range is 0%-100%
    assert "0%" in text
    assert "100%" in text


def test_raise_min_threshold_dims_rows(browser: Browser):
    """Raising the minimum threshold should dim low-score annotations."""
    annotations = [
        {"bbox": {"x": 0, "y": 0, "width": 50, "height": 50}, "score": 0.3, "label": "low"},
        {"bbox": {"x": 100, "y": 100, "width": 50, "height": 50}, "score": 0.9, "label": "high"},
    ]
    value = (make_grid_image(), annotations)

    with gr.Blocks() as demo:
        DetectionViewer(value=value, label="Viewer")
    with GradioApp(demo, browser) as page:
        # Initially no rows should be below threshold
        assert page.locator(".annotation-row.below-threshold").count() == 0

        # Set min threshold to 50% via JS
        page.evaluate("""() => {
            const slider = document.querySelector('.threshold-slider-min');
            const nativeInputValueSetter = Object.getOwnPropertyDescriptor(
                window.HTMLInputElement.prototype, 'value'
            ).set;
            nativeInputValueSetter.call(slider, '50');
            slider.dispatchEvent(new Event('input', { bubbles: true }));
        }""")
        page.wait_for_timeout(200)

        # The low-score row should now be dimmed
        dimmed = page.locator(".annotation-row.below-threshold")
        assert dimmed.count() == 1


def test_lower_max_threshold_dims_rows(browser: Browser):
    """Lowering the maximum threshold should dim high-score annotations."""
    annotations = [
        {"bbox": {"x": 0, "y": 0, "width": 50, "height": 50}, "score": 0.3, "label": "low"},
        {"bbox": {"x": 100, "y": 100, "width": 50, "height": 50}, "score": 0.9, "label": "high"},
    ]
    value = (make_grid_image(), annotations)

    with gr.Blocks() as demo:
        DetectionViewer(value=value, label="Viewer")
    with GradioApp(demo, browser) as page:
        # Set max threshold to 50% via JS
        page.evaluate("""() => {
            const slider = document.querySelector('.threshold-slider-max');
            const nativeInputValueSetter = Object.getOwnPropertyDescriptor(
                window.HTMLInputElement.prototype, 'value'
            ).set;
            nativeInputValueSetter.call(slider, '50');
            slider.dispatchEvent(new Event('input', { bubbles: true }));
        }""")
        page.wait_for_timeout(200)

        # The high-score row should now be dimmed
        dimmed = page.locator(".annotation-row.below-threshold")
        assert dimmed.count() == 1


# ── Custom score_threshold parameter ──


def test_custom_score_threshold_parameter(browser: Browser):
    """Custom score_threshold should set the initial slider values."""
    annotations = [
        {"bbox": {"x": 0, "y": 0, "width": 50, "height": 50}, "score": 0.2, "label": "low"},
        {"bbox": {"x": 100, "y": 100, "width": 50, "height": 50}, "score": 0.5, "label": "mid"},
        {"bbox": {"x": 200, "y": 200, "width": 50, "height": 50}, "score": 0.9, "label": "high"},
    ]
    value = (make_grid_image(), annotations)

    with gr.Blocks() as demo:
        DetectionViewer(value=value, label="Viewer", score_threshold=(0.3, 0.7))
    with GradioApp(demo, browser) as page:
        label = page.locator(".threshold-value")
        text = label.text_content()
        assert "30%" in text
        assert "70%" in text

        # Rows outside the range should be dimmed
        dimmed = page.locator(".annotation-row.below-threshold")
        assert dimmed.count() == 2  # low (0.2 < 0.3) and high (0.9 > 0.7)


# ── 3-tuple config override ──


def test_config_overrides_threshold(browser: Browser):
    """Updating value with 3-tuple config should override score thresholds."""
    annotations = [
        {"bbox": {"x": 0, "y": 0, "width": 50, "height": 50}, "score": 0.2, "label": "low"},
        {"bbox": {"x": 100, "y": 100, "width": 50, "height": 50}, "score": 0.8, "label": "high"},
    ]
    initial = (make_grid_image(), annotations)
    config_value = (make_grid_image(), annotations, {"score_threshold": (0.5, 1.0)})

    with gr.Blocks() as demo:
        viewer = DetectionViewer(value=initial, label="Viewer")
        btn = gr.Button("Apply config")
        btn.click(fn=lambda: config_value, outputs=viewer)
    with GradioApp(demo, browser) as page:
        # Initially no filtering
        assert page.locator(".annotation-row.below-threshold").count() == 0

        page.get_by_role("button", name="Apply config").click()
        page.wait_for_timeout(1000)

        label = page.locator(".threshold-value")
        text = label.text_content()
        assert "50%" in text

        dimmed = page.locator(".annotation-row.below-threshold")
        assert dimmed.count() == 1  # low score (0.2) is below 0.5
