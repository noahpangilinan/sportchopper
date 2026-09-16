"""Thin wrapper around a pretrained YOLOv8 model for person/ball detection."""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .config import COCO_PERSON_CLASS, COCO_SPORTS_BALL_CLASS


@dataclass
class Detection:
    cls: int
    conf: float
    cx: float
    cy: float
    w: float
    h: float


class Detector:
    """Loads YOLOv8n once (auto-downloading COCO weights on first use) and
    yields person/ball detections per frame. No training required for v1."""

    def __init__(self, conf_threshold: float = 0.3, weights: str = "yolov8n.pt"):
        from ultralytics import YOLO  # imported lazily: heavy import, downloads weights

        self._model = YOLO(weights)
        self._conf_threshold = conf_threshold

    def detect(self, frame: np.ndarray) -> Tuple[List[Detection], List[Detection]]:
        """Returns (people, balls) detected in this frame."""
        results = self._model.predict(
            frame,
            classes=[COCO_PERSON_CLASS, COCO_SPORTS_BALL_CLASS],
            conf=self._conf_threshold,
            verbose=False,
        )[0]

        people: List[Detection] = []
        balls: List[Detection] = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            det = Detection(
                cls=cls_id,
                conf=conf,
                cx=(x1 + x2) / 2.0,
                cy=(y1 + y2) / 2.0,
                w=x2 - x1,
                h=y2 - y1,
            )
            if cls_id == COCO_PERSON_CLASS:
                people.append(det)
            elif cls_id == COCO_SPORTS_BALL_CLASS:
                balls.append(det)
        return people, balls
