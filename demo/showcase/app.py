from collections.abc import Callable

import gradio as gr
import numpy as np
from detection_viewer import DetectionViewer
from PIL import Image, ImageDraw

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


def make_person_keypoints(
    cx: float,
    cy: float,
    scale: float = 1.0,
    *,
    missing_index: int | None = None,
) -> list[dict]:
    offsets = [
        (0, -100),  # nose
        (-15, -115),  # left_eye
        (15, -115),  # right_eye
        (-30, -105),  # left_ear
        (30, -105),  # right_ear
        (-50, -60),  # left_shoulder
        (50, -60),  # right_shoulder
        (-80, -10),  # left_elbow
        (80, -10),  # right_elbow
        (-100, 40),  # left_wrist
        (100, 40),  # right_wrist
        (-35, 30),  # left_hip
        (35, 30),  # right_hip
        (-40, 100),  # left_knee
        (40, 100),  # right_knee
        (-45, 170),  # left_ankle
        (45, 170),  # right_ankle
    ]
    keypoints = []
    for i, (dx, dy) in enumerate(offsets):
        if i == missing_index:
            keypoints.append({"x": None, "y": None, "name": COCO_KEYPOINT_NAMES[i], "confidence": 0.0})
        else:
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


def _make_two_people() -> list[dict]:
    return [
        {
            "keypoints": make_person_keypoints(200, 250, scale=1.0),
            "connections": COCO_SKELETON,
        },
        {
            "keypoints": make_person_keypoints(450, 260, scale=0.9, missing_index=10),
            "connections": COCO_SKELETON,
        },
    ]


def _make_single_person() -> list[dict]:
    return [
        {
            "keypoints": make_person_keypoints(320, 250, scale=1.1),
            "connections": COCO_SKELETON,
            "color": "#E91E63",
            "label": "Detected Person",
        },
    ]


def _make_object_detection() -> list[dict]:
    return [
        {"bbox": {"x": 50, "y": 30, "width": 200, "height": 280}, "score": 0.95, "label": "person"},
        {"bbox": {"x": 300, "y": 100, "width": 150, "height": 150}, "score": 0.88, "label": "dog"},
        {"bbox": {"x": 480, "y": 50, "width": 120, "height": 200}, "score": 0.72, "label": "chair"},
        {"bbox": {"x": 100, "y": 350, "width": 250, "height": 100}, "score": 0.65, "label": "bench"},
    ]


def _make_detection_pose() -> list[dict]:
    return [
        {
            "bbox": {"x": 100, "y": 80, "width": 200, "height": 350},
            "score": 0.96,
            "label": "person",
            "keypoints": make_person_keypoints(200, 250, scale=1.0),
            "connections": COCO_SKELETON,
        },
        {
            "bbox": {"x": 350, "y": 100, "width": 180, "height": 330},
            "score": 0.89,
            "label": "person",
            "keypoints": make_person_keypoints(440, 260, scale=0.9, missing_index=10),
            "connections": COCO_SKELETON,
        },
    ]


def _make_many_detections() -> list[dict]:
    rng = np.random.default_rng(seed=123)
    labels = ["person", "car", "dog", "cat", "bicycle", "chair", "bottle", "phone"]
    detections = []
    for i in range(20):
        x = float(rng.integers(0, 500))
        y = float(rng.integers(0, 350))
        w = float(rng.integers(40, 150))
        h = float(rng.integers(40, 150))
        score = round(float(rng.uniform(0.3, 0.99)), 2)
        label = labels[i % len(labels)]
        detections.append({"bbox": {"x": x, "y": y, "width": w, "height": h}, "score": score, "label": label})
    return detections


def _make_rle_mask(x: int, y: int, w: int, h: int, img_h: int, img_w: int) -> dict:
    """Create an uncompressed COCO RLE mask for a rectangular region."""
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    mask[y : y + h, x : x + w] = 1
    # Column-major (Fortran order) run-length encoding
    flat = mask.flatten(order="F")
    counts = []
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


def _make_segmentation() -> list[dict]:
    img_h, img_w = 480, 640
    return [
        {
            "bbox": {"x": 50, "y": 30, "width": 200, "height": 280},
            "mask": _make_rle_mask(50, 30, 200, 280, img_h, img_w),
            "score": 0.95,
            "label": "person",
        },
        {
            "bbox": {"x": 300, "y": 100, "width": 150, "height": 150},
            "mask": _make_rle_mask(300, 100, 150, 150, img_h, img_w),
            "score": 0.88,
            "label": "dog",
        },
        {
            "bbox": {"x": 480, "y": 50, "width": 120, "height": 200},
            "mask": _make_rle_mask(480, 50, 120, 200, img_h, img_w),
            "score": 0.72,
            "label": "chair",
        },
        {
            "bbox": {"x": 100, "y": 350, "width": 250, "height": 100},
            "mask": _make_rle_mask(100, 350, 250, 100, img_h, img_w),
            "score": 0.65,
            "label": "bench",
        },
    ]


_EXAMPLE_BUILDERS: dict[str, Callable[[], list[dict]]] = {
    "Two people": _make_two_people,
    "Single person": _make_single_person,
    "Object detection": _make_object_detection,
    "Detection + Pose": _make_detection_pose,
    "Many detections": _make_many_detections,
    "Segmentation": _make_segmentation,
}


def get_example(choice: str) -> tuple[Image.Image, list[dict], dict] | None:
    builder = _EXAMPLE_BUILDERS.get(choice)
    if builder is None:
        return None
    return make_grid_image(), builder(), {"score_threshold": (0.0, 1.0)}


initial_value = get_example("Two people")

with gr.Blocks(title="Detection Viewer") as demo:
    gr.Markdown("# Detection Viewer")
    gr.Markdown("Select an example to display bounding boxes, keypoints, and skeleton connections.")

    viewer = DetectionViewer(value=initial_value, label="Detection Viewer", keypoint_threshold=0.3)

    selector = gr.Radio(
        choices=[
            "Two people",
            "Single person",
            "Object detection",
            "Detection + Pose",
            "Many detections",
            "Segmentation",
            "Clear",
        ],
        value="Two people",
        label="Example",
    )

    selector.change(fn=get_example, inputs=selector, outputs=viewer)

if __name__ == "__main__":
    demo.launch()
