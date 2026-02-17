import gradio as gr
import numpy as np
import torch
from detection_viewer import DetectionViewer
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

MODEL_ID = "facebook/mask2former-swin-tiny-coco-instance"

processor = AutoImageProcessor.from_pretrained(MODEL_ID)
model = Mask2FormerForUniversalSegmentation.from_pretrained(MODEL_ID)
model.eval()


def _mask_to_rle(mask: np.ndarray) -> dict:
    """Convert a binary mask to uncompressed RLE (column-major / COCO format)."""
    h, w = mask.shape
    flat = mask.ravel(order="F")
    changes = np.diff(flat)
    change_idx = np.flatnonzero(changes)
    runs = np.diff(np.concatenate([[-1], change_idx, [len(flat) - 1]]))
    counts = runs.tolist()
    if flat[0] == 1:
        counts = [0, *counts]
    return {"counts": counts, "size": [h, w]}


def _mask_to_bbox(mask: np.ndarray) -> dict:
    """Compute bounding box from a binary mask."""
    ys, xs = np.where(mask)
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    return {"x": x_min, "y": y_min, "width": x_max - x_min + 1, "height": y_max - y_min + 1}


@torch.inference_mode()
def detect(image: Image.Image, threshold: float) -> tuple[Image.Image, list[dict]] | None:
    if image is None:
        return None

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    results = processor.post_process_instance_segmentation(
        outputs,
        target_sizes=[(image.height, image.width)],
        threshold=threshold,
    )[0]

    segmentation = results["segmentation"].numpy().astype(np.uint8)
    annotations = []
    for segment in results["segments_info"]:
        binary_mask = (segmentation == segment["id"]).astype(np.uint8)
        if binary_mask.sum() == 0:
            continue

        annotations.append(
            {
                "bbox": _mask_to_bbox(binary_mask),
                "mask": _mask_to_rle(binary_mask),
                "score": round(float(segment["score"]), 3),
                "label": model.config.id2label[int(segment["label_id"])],
            }
        )

    return image, annotations, {"score_threshold": (threshold, 1.0)}


with gr.Blocks(title="Mask2Former Detection Demo") as demo:
    gr.Markdown("# Mask2Former Instance Segmentation Demo")
    gr.Markdown(f"Model: `{MODEL_ID}`")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Input Image", type="pil")
            threshold = gr.Slider(label="Confidence Threshold", minimum=0.0, maximum=1.0, step=0.05, value=0.5)
            run_btn = gr.Button("Detect", variant="primary")
        with gr.Column():
            viewer = DetectionViewer(label="Detection Results")

    run_btn.click(fn=detect, inputs=[input_image, threshold], outputs=viewer)

if __name__ == "__main__":
    demo.launch()
