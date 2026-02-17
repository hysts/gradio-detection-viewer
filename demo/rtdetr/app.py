import gradio as gr
import torch
from detection_viewer import DetectionViewer
from PIL import Image
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

MODEL_ID = "PekingU/rtdetr_r18vd"

processor = RTDetrImageProcessor.from_pretrained(MODEL_ID)
model = RTDetrForObjectDetection.from_pretrained(MODEL_ID)
model.eval()


@torch.inference_mode()
def detect(image: Image.Image, threshold: float) -> tuple[Image.Image, list[dict]] | None:
    if image is None:
        return None

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    results = processor.post_process_object_detection(
        outputs,
        target_sizes=torch.tensor([(image.height, image.width)]),
        threshold=threshold,
    )[0]

    annotations = []
    for score, label_id, box in zip(results["scores"], results["labels"], results["boxes"], strict=True):
        x_min, y_min, x_max, y_max = box.tolist()
        annotations.append(
            {
                "bbox": {
                    "x": x_min,
                    "y": y_min,
                    "width": x_max - x_min,
                    "height": y_max - y_min,
                },
                "score": round(score.item(), 3),
                "label": model.config.id2label[label_id.item()],
            }
        )

    return image, annotations, {"score_threshold": (threshold, 1.0)}


with gr.Blocks(title="RT-DETR Detection Demo") as demo:
    gr.Markdown("# RT-DETR Detection Demo")
    gr.Markdown(f"Model: `{MODEL_ID}`")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Input Image", type="pil")
            threshold = gr.Slider(label="Confidence Threshold", minimum=0.0, maximum=1.0, step=0.05, value=0.25)
            run_btn = gr.Button("Detect", variant="primary")
        with gr.Column():
            viewer = DetectionViewer(label="Detection Results")

    run_btn.click(fn=detect, inputs=[input_image, threshold], outputs=viewer)

if __name__ == "__main__":
    demo.launch()
