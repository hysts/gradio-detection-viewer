"""Shared test helpers (constants, factory functions, GradioApp context manager).

Fixtures live in conftest.py; this module holds plain helpers that test files
can import directly with ``from _helpers import ...``.
"""

from __future__ import annotations

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw
from playwright.sync_api import Browser, Page

# ── Focus helpers ──


def focus_viewer(page: Page) -> None:
    """Click the canvas area to ensure the component has keyboard focus."""
    page.locator(".canvas-wrapper").click()


def focus_without_canvas_click(page: Page) -> None:
    """Focus the component element for keyboard events without triggering canvas mouse handlers."""
    page.evaluate("() => document.querySelector('.pose-viewer-container').parentElement.focus()")


COCO_SKELETON = [
    [0, 1],
    [0, 2],
    [1, 3],
    [2, 4],
    [5, 6],
    [5, 7],
    [7, 9],
    [6, 8],
    [8, 10],
    [5, 11],
    [6, 12],
    [11, 12],
    [11, 13],
    [13, 15],
    [12, 14],
    [14, 16],
]

COCO_KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


def make_grid_image(width: int = 640, height: int = 480) -> Image.Image:
    img = Image.new("RGB", (width, height), "#f0f0f0")
    draw = ImageDraw.Draw(img)
    for x in range(0, width, 40):
        draw.line([(x, 0), (x, height)], fill="#ddd", width=1)
    for y in range(0, height, 40):
        draw.line([(0, y), (width, y)], fill="#ddd", width=1)
    return img


def make_detections(n: int) -> list[dict]:
    rng = np.random.default_rng(seed=123)
    labels = ["person", "car", "dog", "cat", "bicycle", "chair", "bottle", "phone"]
    detections = []
    for i in range(n):
        x = float(rng.integers(0, 500))
        y = float(rng.integers(0, 350))
        w = float(rng.integers(40, 150))
        h = float(rng.integers(40, 150))
        score = round(float(rng.uniform(0.3, 0.99)), 2)
        label = labels[i % len(labels)]
        detections.append({"bbox": {"x": x, "y": y, "width": w, "height": h}, "score": score, "label": label})
    return detections


def make_person_keypoints(cx: float, cy: float, scale: float = 1.0) -> list[dict]:
    offsets = [
        (0, -100),
        (-15, -115),
        (15, -115),
        (-30, -105),
        (30, -105),
        (-50, -60),
        (50, -60),
        (-80, -10),
        (80, -10),
        (-100, 40),
        (100, 40),
        (-35, 30),
        (35, 30),
        (-40, 100),
        (40, 100),
        (-45, 170),
        (45, 170),
    ]
    keypoints = []
    for i, (dx, dy) in enumerate(offsets):
        rng = np.random.default_rng(seed=42 + i)
        conf = round(float(rng.uniform(0.7, 0.99)), 2)
        keypoints.append(
            {
                "x": cx + dx * scale,
                "y": cy + dy * scale,
                "name": COCO_KEYPOINT_NAMES[i],
                "confidence": conf,
            }
        )
    return keypoints


def make_rle_mask(x: int, y: int, w: int, h: int, img_h: int = 480, img_w: int = 640) -> dict:
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    mask[y : y + h, x : x + w] = 1
    flat = mask.flatten(order="F")
    counts: list[int] = []
    val = 0
    count = 0
    for pixel in flat:
        if pixel == val:
            count += 1
        else:
            counts.append(count)
            count = 1
            val = pixel
    counts.append(count)
    return {"counts": counts, "size": [img_h, img_w]}


def make_pose_annotations() -> list[dict]:
    return [
        {
            "bbox": {"x": 100, "y": 80, "width": 200, "height": 350},
            "score": 0.96,
            "label": "person",
            "keypoints": make_person_keypoints(200, 250),
            "connections": COCO_SKELETON,
        },
        {
            "bbox": {"x": 350, "y": 100, "width": 180, "height": 330},
            "score": 0.89,
            "label": "person",
            "keypoints": make_person_keypoints(440, 260, scale=0.9),
            "connections": COCO_SKELETON,
        },
    ]


def make_segmentation_annotations() -> list[dict]:
    return [
        {
            "bbox": {"x": 50, "y": 30, "width": 200, "height": 280},
            "mask": make_rle_mask(50, 30, 200, 280),
            "score": 0.95,
            "label": "person",
        },
        {
            "bbox": {"x": 300, "y": 100, "width": 150, "height": 150},
            "mask": make_rle_mask(300, 100, 150, 150),
            "score": 0.88,
            "label": "dog",
        },
    ]


def set_range_value(page: Page, selector: str, value: int) -> None:
    """Set a range input's value using the native setter to trigger an 'input' event.

    Playwright cannot ``fill()`` range inputs, so this uses ``evaluate()``
    with the native ``HTMLInputElement.value`` setter.
    """
    page.evaluate(
        """([sel, val]) => {
            const slider = document.querySelector(sel);
            const nativeInputValueSetter = Object.getOwnPropertyDescriptor(
                window.HTMLInputElement.prototype, 'value'
            ).set;
            nativeInputValueSetter.call(slider, String(val));
            slider.dispatchEvent(new Event('input', { bubbles: true }));
        }""",
        [selector, value],
    )


class GradioApp:
    """Context manager that launches a Gradio demo and opens a Playwright page."""

    def __init__(self, demo: gr.Blocks, browser: Browser) -> None:
        self._demo = demo
        self._browser = browser
        self.page: Page | None = None
        self._url: str = ""

    def __enter__(self) -> Page:
        _, self._url, _ = self._demo.launch(prevent_thread_lock=True)
        self.page = self._browser.new_page(viewport={"width": 1280, "height": 900})
        self.page.set_default_timeout(5000)
        self.page.goto(self._url)
        self.page.wait_for_timeout(500)
        return self.page

    def __exit__(self, *_: object) -> None:
        if self.page:
            self.page.close()
        self._demo.close()
