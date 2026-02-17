import gradio as gr
import torch
from detection_viewer import DetectionViewer
from PIL import Image
from transformers import AutoProcessor, RTDetrForObjectDetection, VitPoseForPoseEstimation

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

DETECTOR_MODEL_ID = "PekingU/rtdetr_r18vd"
POSE_MODEL_ID = "usyd-community/vitpose-base-simple"

# Person detector (RT-DETR)
det_processor = AutoProcessor.from_pretrained(DETECTOR_MODEL_ID)
det_model = RTDetrForObjectDetection.from_pretrained(DETECTOR_MODEL_ID)
det_model.eval()

# Find the label ID for "person"
PERSON_LABEL_ID = next(k for k, v in det_model.config.id2label.items() if v == "person")

# Pose estimator (ViTPose)
pose_processor = AutoProcessor.from_pretrained(POSE_MODEL_ID)
pose_model = VitPoseForPoseEstimation.from_pretrained(POSE_MODEL_ID)
pose_model.eval()

# Skeleton connections from ViTPose config
SKELETON = pose_model.config.edges


@torch.inference_mode()
def detect(image: Image.Image, det_threshold: float) -> tuple[Image.Image, list[dict]] | None:
    if image is None:
        return None

    # Step 1: Detect persons with RT-DETR
    det_inputs = det_processor(images=image, return_tensors="pt")
    det_outputs = det_model(**det_inputs)

    det_results = det_processor.post_process_object_detection(
        det_outputs,
        target_sizes=torch.tensor([(image.height, image.width)]),
        threshold=det_threshold,
    )[0]

    person_mask = det_results["labels"] == PERSON_LABEL_ID
    person_boxes_voc = det_results["boxes"][person_mask].cpu().numpy()
    person_scores = det_results["scores"][person_mask].cpu().numpy()

    if len(person_boxes_voc) == 0:
        return image, []

    # Convert VOC (x1, y1, x2, y2) to COCO (x1, y1, w, h) for ViTPose
    person_boxes_coco = person_boxes_voc.copy()
    person_boxes_coco[:, 2] = person_boxes_voc[:, 2] - person_boxes_voc[:, 0]
    person_boxes_coco[:, 3] = person_boxes_voc[:, 3] - person_boxes_voc[:, 1]

    # Step 2: Estimate keypoints with ViTPose
    pose_inputs = pose_processor(image, boxes=[person_boxes_coco], return_tensors="pt")
    pose_outputs = pose_model(**pose_inputs)

    pose_results = pose_processor.post_process_pose_estimation(pose_outputs, boxes=[person_boxes_coco])

    # Step 3: Build annotations
    annotations = []
    for i, pose_result in enumerate(pose_results[0]):
        keypoints_xy = pose_result["keypoints"].cpu().numpy()
        keypoints_scores = pose_result["scores"].cpu().numpy()

        x1, y1, x2, y2 = person_boxes_voc[i]
        keypoints = [
            {
                "x": float(keypoints_xy[j][0]),
                "y": float(keypoints_xy[j][1]),
                "name": COCO_KEYPOINT_NAMES[j],
                "confidence": round(float(keypoints_scores[j]), 3),
            }
            for j in range(len(keypoints_xy))
        ]

        annotations.append(
            {
                "bbox": {
                    "x": float(x1),
                    "y": float(y1),
                    "width": float(x2 - x1),
                    "height": float(y2 - y1),
                },
                "score": round(float(person_scores[i]), 3),
                "label": "person",
                "keypoints": keypoints,
                "connections": SKELETON,
            }
        )

    return image, annotations, {"score_threshold": (det_threshold, 1.0)}


with gr.Blocks(title="ViTPose Demo") as demo:
    gr.Markdown("# ViTPose Pose Estimation Demo")
    gr.Markdown(f"Detector: `{DETECTOR_MODEL_ID}` / Pose: `{POSE_MODEL_ID}`")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Input Image", type="pil")
            det_threshold = gr.Slider(
                label="Person Detection Threshold", minimum=0.0, maximum=1.0, step=0.05, value=0.3
            )
            run_btn = gr.Button("Detect", variant="primary")
        with gr.Column():
            viewer = DetectionViewer(
                label="Pose Estimation Results",
                keypoint_threshold=0.3,
            )

    run_btn.click(fn=detect, inputs=[input_image, det_threshold], outputs=viewer)

if __name__ == "__main__":
    demo.launch()
