"""Unit tests for Python backend logic (_load_image, _process, postprocess, api_info)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from detection_viewer import DetectionViewer, _load_image

# ── _load_image ──


class TestLoadImage:
    def test_from_pil_image(self):
        img = Image.new("RGB", (100, 100), "red")
        result = _load_image(img)
        assert isinstance(result, Image.Image)
        assert result is img

    @pytest.mark.parametrize(
        ("shape", "expected_size"),
        [((100, 100, 3), (100, 100)), ((80, 120, 4), (120, 80))],
        ids=["rgb", "rgba"],
    )
    def test_from_numpy_array(self, shape: tuple, expected_size: tuple):
        arr = np.zeros(shape, dtype=np.uint8)
        result = _load_image(arr)
        assert isinstance(result, Image.Image)
        assert result.size == expected_size

    @pytest.mark.parametrize("as_str", [True, False], ids=["str_path", "path_object"])
    def test_from_filesystem_path(self, tmp_path: Path, *, as_str: bool):
        img_path = tmp_path / "test.png"
        Image.new("RGB", (50, 50), "blue").save(img_path)
        result = _load_image(str(img_path) if as_str else img_path)
        assert isinstance(result, Image.Image)
        assert result.size == (50, 50)

    def test_from_invalid_path_raises(self):
        with pytest.raises(FileNotFoundError):
            _load_image("/nonexistent/path/image.png")


# ── _process ──


def _make_viewer(**kwargs: object) -> DetectionViewer:
    """Create a DetectionViewer instance with defaults for testing."""
    import gradio as gr

    with gr.Blocks():
        return DetectionViewer(value=None, **kwargs)


class TestProcess:
    def test_none_returns_none(self):
        viewer = _make_viewer()
        assert viewer._process(None) is None

    def test_2_tuple_basic(self):
        img = Image.new("RGB", (100, 100), "red")
        annotations = [
            {"bbox": {"x": 10, "y": 20, "width": 30, "height": 40}, "score": 0.9, "label": "cat"},
        ]
        viewer = _make_viewer()
        result = viewer._process((img, annotations))
        assert result is not None
        data = json.loads(result)
        assert "image" in data
        assert "annotations" in data
        assert len(data["annotations"]) == 1
        ann = data["annotations"][0]
        assert ann["label"] == "cat"
        assert ann["score"] == 0.9
        assert ann["bbox"] == {"x": 10, "y": 20, "width": 30, "height": 40}

    def test_3_tuple_with_config(self):
        img = Image.new("RGB", (100, 100), "red")
        annotations = [{"bbox": {"x": 0, "y": 0, "width": 10, "height": 10}, "score": 0.5}]
        config = {"score_threshold": (0.3, 0.8)}
        viewer = _make_viewer()
        result = viewer._process((img, annotations, config))
        data = json.loads(result)
        assert data["scoreThresholdMin"] == 0.3
        assert data["scoreThresholdMax"] == 0.8

    def test_config_without_score_threshold(self):
        img = Image.new("RGB", (100, 100), "red")
        annotations = [{"bbox": {"x": 0, "y": 0, "width": 10, "height": 10}}]
        config = {"other_key": "value"}
        viewer = _make_viewer()
        result = viewer._process((img, annotations, config))
        data = json.loads(result)
        assert "scoreThresholdMin" not in data
        assert "scoreThresholdMax" not in data

    def test_empty_annotations(self):
        img = Image.new("RGB", (100, 100), "red")
        viewer = _make_viewer()
        result = viewer._process((img, []))
        data = json.loads(result)
        assert data["annotations"] == []
        assert "image" in data

    def test_default_label_detection(self):
        img = Image.new("RGB", (100, 100), "red")
        annotations = [
            {"bbox": {"x": 0, "y": 0, "width": 10, "height": 10}},
            {"bbox": {"x": 20, "y": 20, "width": 10, "height": 10}},
        ]
        viewer = _make_viewer()
        result = viewer._process((img, annotations))
        data = json.loads(result)
        assert data["annotations"][0]["label"] == "Detection 1"
        assert data["annotations"][1]["label"] == "Detection 2"

    def test_default_label_person_with_keypoints(self):
        img = Image.new("RGB", (100, 100), "red")
        annotations = [
            {"keypoints": [{"x": 50, "y": 50, "name": "nose", "confidence": 0.9}]},
            {"keypoints": [{"x": 60, "y": 60, "name": "nose", "confidence": 0.8}]},
        ]
        viewer = _make_viewer()
        result = viewer._process((img, annotations))
        data = json.loads(result)
        assert data["annotations"][0]["label"] == "Person 1"
        assert data["annotations"][1]["label"] == "Person 2"

    def test_explicit_label_overrides_default(self):
        img = Image.new("RGB", (100, 100), "red")
        annotations = [
            {"keypoints": [{"x": 50, "y": 50, "name": "nose"}], "label": "athlete"},
        ]
        viewer = _make_viewer()
        result = viewer._process((img, annotations))
        data = json.loads(result)
        assert data["annotations"][0]["label"] == "athlete"

    def test_color_palette_cycling(self):
        img = Image.new("RGB", (100, 100), "red")
        annotations = [{"bbox": {"x": 0, "y": 0, "width": 10, "height": 10}} for _ in range(10)]
        viewer = _make_viewer()
        result = viewer._process((img, annotations))
        data = json.loads(result)
        # 8 colors in palette, so index 8 should wrap to index 0
        assert data["annotations"][0]["color"] == data["annotations"][8]["color"]
        assert data["annotations"][1]["color"] == data["annotations"][9]["color"]
        # Adjacent colors should differ
        assert data["annotations"][0]["color"] != data["annotations"][1]["color"]

    def test_custom_color_preserved(self):
        img = Image.new("RGB", (100, 100), "red")
        annotations = [{"bbox": {"x": 0, "y": 0, "width": 10, "height": 10}, "color": "#ABCDEF"}]
        viewer = _make_viewer()
        result = viewer._process((img, annotations))
        data = json.loads(result)
        assert data["annotations"][0]["color"] == "#ABCDEF"

    def test_optional_fields_omitted_when_absent(self):
        img = Image.new("RGB", (100, 100), "red")
        annotations = [{"label": "minimal"}]
        viewer = _make_viewer()
        result = viewer._process((img, annotations))
        data = json.loads(result)
        ann = data["annotations"][0]
        assert "bbox" not in ann
        assert "score" not in ann
        assert "mask" not in ann
        assert ann["keypoints"] == []
        assert ann["connections"] == []

    def test_mask_included(self):
        img = Image.new("RGB", (100, 100), "red")
        mask = {"counts": [100, 50, 100], "size": [100, 100]}
        annotations = [{"mask": mask, "label": "obj"}]
        viewer = _make_viewer()
        result = viewer._process((img, annotations))
        data = json.loads(result)
        assert data["annotations"][0]["mask"] == mask

    def test_connections_included(self):
        img = Image.new("RGB", (100, 100), "red")
        annotations = [
            {
                "keypoints": [{"x": 0, "y": 0, "name": "a"}, {"x": 10, "y": 10, "name": "b"}],
                "connections": [[0, 1]],
            },
        ]
        viewer = _make_viewer()
        result = viewer._process((img, annotations))
        data = json.loads(result)
        assert data["annotations"][0]["connections"] == [[0, 1]]


# ── postprocess ──


class TestPostprocess:
    def test_string_passthrough(self):
        viewer = _make_viewer()
        assert viewer.postprocess('{"test": true}') == '{"test": true}'

    def test_none_value(self):
        viewer = _make_viewer()
        assert viewer.postprocess(None) is None

    def test_tuple_calls_process(self):
        viewer = _make_viewer()
        img = Image.new("RGB", (100, 100), "red")
        annotations = [{"bbox": {"x": 0, "y": 0, "width": 10, "height": 10}, "label": "obj"}]
        result = viewer.postprocess((img, annotations))
        assert result is not None
        data = json.loads(result)
        assert data["annotations"][0]["label"] == "obj"


# ── api_info ──


class TestApiInfo:
    def test_returns_dict(self):
        viewer = _make_viewer()
        info = viewer.api_info()
        assert isinstance(info, dict)

    def test_type_is_string(self):
        viewer = _make_viewer()
        info = viewer.api_info()
        assert info["type"] == "string"

    def test_has_description(self):
        viewer = _make_viewer()
        info = viewer.api_info()
        assert "description" in info
        assert "JSON" in info["description"]
