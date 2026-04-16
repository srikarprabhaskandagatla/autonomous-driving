import time
import numpy as np
import torch
import cv2
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor


CLASS_NAMES = [
    'pedestrian', 'rider', 'car', 'truck', 'bus',
    'train', 'motorcycle', 'bicycle', 'traffic light', 'traffic sign'
]


class PerceptionPipeline:

    def __init__(
        self,
        yolo_weights: str,
        sam_checkpoint: str,
        sam_model_type: str = "vit_h",
        device: str = "cuda",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("Pipeline requires CUDA.")

        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        self.yolo = YOLO(yolo_weights)
        self.yolo.to(device)

        sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
        sam.to(device)
        self.predictor = SamPredictor(sam)

        self._warmup()

    def _warmup(self, n: int = 5):
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        for _ in range(n):
            self.yolo(dummy, verbose=False)
            self.predictor.set_image(dummy)

    def infer(self, image: np.ndarray) -> dict:
        t_start = time.perf_counter()

        yolo_out = self.yolo(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=640,
            verbose=False,
        )[0]

        boxes = yolo_out.boxes
        if len(boxes) == 0:
            return {
                'detections': [],
                'latency_ms': (time.perf_counter() - t_start) * 1000
            }

        self.predictor.set_image(image)

        input_boxes_np = boxes.xyxy.cpu().numpy()

        input_boxes_t = self.predictor.transform.apply_boxes_torch(
            torch.tensor(input_boxes_np, dtype=torch.float32, device=self.device),
            image.shape[:2]
        )

        masks, scores, _ = self.predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=input_boxes_t,
            multimask_output=False,
        )

        latency_ms = (time.perf_counter() - t_start) * 1000

        detections = []
        for i in range(len(boxes)):
            detections.append({
                'bbox': input_boxes_np[i].tolist(),
                'class_id': int(boxes.cls[i].item()),
                'class_name': CLASS_NAMES[int(boxes.cls[i].item())],
                'confidence': float(boxes.conf[i].item()),
                'mask': masks[i, 0].cpu().numpy(),
                'mask_score': float(scores[i, 0].item()),
            })

        return {
            'detections': detections,
            'latency_ms': latency_ms,
        }
