# Detection Viewer

[![Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue)](https://huggingface.co/spaces/hysts-gradio-custom-html/detection-viewer-demo)

A Gradio custom HTML component for visualizing object detection results — bounding boxes, segmentation masks, human pose keypoints, skeleton connections, and confidence scores.

Built on top of `gr.HTML` using HTML/CSS/JavaScript for rendering, with no frontend build step required.

## Features

- **Bounding boxes** with labels and confidence scores
- **Segmentation masks** with adjustable opacity (COCO RLE format)
- **Keypoints** (e.g., COCO 17-joint format) with per-point confidence
- **Skeleton connections** between keypoints
- **Interactive control panel**
  - Toggle visibility of individual annotations
  - Dual-thumb range slider for confidence threshold filtering
  - Per-keypoint confidence threshold
  - Toggle layers (Boxes / Skeleton / Keypoints / Masks / Image)
  - Mask opacity slider
  - Click to select an annotation and view details (bbox coordinates, keypoint values)
  - Sort annotations by score or bounding box size
  - Filter by label (double-click to solo a label)
  - Expandable annotation detail panel
  - Adjustable annotation list height
- **Keyboard shortcuts**

  | Key | Action |
  |-----|--------|
  | `Escape` | Deselect annotation |
  | `A` | Toggle all annotations |
  | `H` | Hide selected annotation |
  | `I` | Toggle image layer |
  | `F` | Maximize / fullscreen |
  | `R` | Reset view |
  | `+` / `=` | Zoom in |
  | `-` / `_` | Zoom out |
  | `0` | Reset zoom |
  | `?` | Show help overlay |

## Installation

Requires Python >= 3.12.

```bash
pip install .
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv pip install .
```

For development (editable install):

```bash
uv pip install -e .
```

To build a wheel/sdist:

```bash
uv build
```

## Usage

```python
import gradio as gr
from detection_viewer import DetectionViewer

annotations = [
    {
        "bbox": {"x": 50, "y": 30, "width": 200, "height": 280},
        "keypoints": [
            {"x": 150, "y": 100, "name": "nose", "confidence": 0.95},
            {"x": 135, "y": 85, "name": "left_eye", "confidence": 0.90},
            # ...
        ],
        "connections": [[0, 1], [0, 2]],
        "score": 0.95,
        "label": "person",
    },
]

with gr.Blocks() as demo:
    viewer = DetectionViewer(value=("image.jpg", annotations), label="Results")

demo.launch()
```

### Constructor parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `value` | `tuple` or `None` | `None` | `(image, annotations)` or `(image, annotations, config)` |
| `label` | `str` or `None` | `None` | Component label |
| `panel_title` | `str` | `"Detections"` | Title shown in the control panel |
| `list_height` | `int` | `300` | Height in pixels for the annotation list |
| `score_threshold` | `tuple[float, float]` | `(0.0, 1.0)` | Initial min/max range for the score slider |
| `keypoint_threshold` | `float` | `0.0` | Minimum confidence for keypoint visibility |
| `keypoint_radius` | `int` | `3` | Default keypoint circle radius in pixels |

### Annotation format

Each annotation is a dict with the following optional fields:

| Field | Type | Description |
|---|---|---|
| `bbox` | `{"x", "y", "width", "height"}` | Bounding box |
| `keypoints` | `list[{"x", "y", "name", "confidence"}]` | Keypoint coordinates |
| `connections` | `list[[int, int]]` | Index pairs into `keypoints` for skeleton edges |
| `mask` | `{"counts": list[int], "size": [H, W]}` | Segmentation mask in uncompressed COCO RLE format |
| `score` | `float` | Overall confidence score (0–1) |
| `label` | `str` | Display label (defaults to "Person N" or "Detection N") |
| `color` | `str` | Hex color (defaults to a built-in palette) |

### Dynamic config

Pass a 3-tuple `(image, annotations, config)` as the value to dynamically control settings:

```python
viewer.value = (image, annotations, {"score_threshold": (0.3, 0.9)})
```

| Config key | Type | Description |
|------------|------|-------------|
| `score_threshold` | `tuple[float, float]` | Override the score range slider bounds |

### Image input

The image can be a file path (`str` / `Path`), a `PIL.Image.Image`, or a `numpy.ndarray`.

## Demo

```bash
uv run python demo/showcase/app.py
```

The showcase demo provides preset examples: pose estimation, object detection, combined detection + pose, segmentation, and stress tests.

Model-specific demos are also available (require `transformers` and `torch`):

```bash
uv run python demo/vitpose/app.py          # ViTPose pose estimation
uv run python demo/rtdetr/app.py            # RT-DETR object detection
uv run python demo/grounding-dino/app.py    # Grounding DINO zero-shot detection
uv run python demo/mask2former/app.py       # Mask2Former instance segmentation
```

## Testing

The project includes a Playwright-based UI test suite with 60 tests.

```bash
uv run pytest tests/
```
