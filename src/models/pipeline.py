# src/models/pipeline.py

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
    """
    Stage 1: YOLOv9c  → bounding boxes      (~8ms  on A100)
    Stage 2: SAM ViT-H → instance masks     (~18ms on A100)
    Total:                                    <28ms end-to-end
    """

    def __init__(
        self,
        yolo_weights: str,
        sam_checkpoint: str,
        sam_model_type: str = "vit_h",
        device: str = "cuda",
        conf_threshold: float = 0.25,
        iou_threshold: float  = 0.45,
    ):
        assert torch.cuda.is_available(), "Pipeline requires CUDA."
        self.device         = device
        self.conf_threshold = conf_threshold
        self.iou_threshold  = iou_threshold

        print("Loading YOLOv9c...")
        self.yolo = YOLO(yolo_weights)
        self.yolo.to(device)

        # SAM ViT-H: highest accuracy SAM variant
        # ViT-B is faster but mask quality drops significantly on small objects
        print("Loading SAM ViT-H...")
        sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
        sam.to(device)
        self.predictor = SamPredictor(sam)

        self._warmup()

    def _warmup(self, n: int = 5):
        """Pre-allocate CUDA memory. Without this, first inference is 3x slower."""
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        for _ in range(n):
            self.yolo(dummy, verbose=False)
        print("Pipeline ready.")

    def infer(self, image: np.ndarray) -> dict:
        """
        Args:
            image: HxWx3 RGB numpy array, uint8
        Returns:
            dict with detections list and latency_ms
        """
        t_start = time.perf_counter()

        # --- Stage 1: Detection ---
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

        # --- Stage 2: Segmentation ---
        # set_image encodes the image once for all box prompts
        # This is why we batch — calling set_image per-box would be ~10x slower
        self.predictor.set_image(image)

        input_boxes_np = boxes.xyxy.cpu().numpy()  # [N, 4]

        input_boxes_t = self.predictor.transform.apply_boxes_torch(
            torch.tensor(input_boxes_np, dtype=torch.float32, device=self.device),
            image.shape[:2]
        )

        # predict_torch accepts batched boxes — all N masks in one forward pass
        masks, scores, _ = self.predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=input_boxes_t,
            multimask_output=False,  # Single mask per box — see note above
        )

        latency_ms = (time.perf_counter() - t_start) * 1000

        detections = []
        for i in range(len(boxes)):
            detections.append({
                'bbox':       input_boxes_np[i].tolist(),
                'class_id':   int(boxes.cls[i].item()),
                'class_name': CLASS_NAMES[int(boxes.cls[i].item())],
                'confidence': float(boxes.conf[i].item()),
                'mask':       masks[i, 0].cpu().numpy(),   # HxW bool tensor
                'mask_score': float(scores[i].item()),
            })

        return {
            'detections': detections,
            'latency_ms': latency_ms,
        }