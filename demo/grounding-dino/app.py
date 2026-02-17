import gradio as gr
import torch
from detection_viewer import DetectionViewer
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

MODEL_ID = "IDEA-Research/grounding-dino-tiny"

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID)
model.eval()


@torch.inference_mode()
def detect(image: Image.Image, labels: str, threshold: float) -> tuple[Image.Image, list[dict]] | None:
    if image is None or not labels.strip():
        return None

    text = labels.strip().rstrip(".")
    candidate_labels = [part.strip() for part in text.split(",") if part.strip()]
    # Grounding DINO expects a single string with labels separated by ". " and ending with "."
    text_prompt = ". ".join(candidate_labels) + "."

    inputs = processor(images=image, text=text_prompt, return_tensors="pt")
    outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        input_ids=inputs["input_ids"],
        target_sizes=[(image.height, image.width)],
        threshold=threshold,
        text_threshold=threshold,
    )[0]

    annotations = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"], strict=True):
        x_min, y_min, x_max, y_max = box.tolist()
        annotations.append(
            {
                "bbox": {
                    "x": x_min,
                    "y": y_min,
                    "width": x_max - x_min,
                    "height": y_max - y_min,
                },
                "score": round(float(score), 3),
                "label": label,
            }
        )

    return image, annotations, {"score_threshold": (threshold, 1.0)}


with gr.Blocks(title="Grounding DINO Demo") as demo:
    gr.Markdown("# Grounding DINO Zero-Shot Detection Demo")
    gr.Markdown(f"Model: `{MODEL_ID}`")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Input Image", type="pil")
            labels_input = gr.Textbox(
                label="Labels (comma-separated)",
                placeholder="person, dog, car, chair",
                value="person, dog, cat, car",
            )
            threshold = gr.Slider(label="Confidence Threshold", minimum=0.0, maximum=1.0, step=0.05, value=0.3)
            run_btn = gr.Button("Detect", variant="primary")
        with gr.Column():
            viewer = DetectionViewer(label="Detection Results")

    run_btn.click(fn=detect, inputs=[input_image, labels_input, threshold], outputs=viewer)

if __name__ == "__main__":
    demo.launch()
