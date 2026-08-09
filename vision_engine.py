"""YOLO-based object detection."""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from ultralytics import YOLO

import config

logger = logging.getLogger(__name__)


def _select_device() -> str:
    """Pick the fastest available device: CUDA > MPS (Apple Silicon) > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class VisionEngine:
    """Wraps an Ultralytics YOLO model for real-time object detection."""

    def __init__(
        self,
        model_path: str = config.YOLO_MODEL_PATH,
        confidence_threshold: float = config.YOLO_CONFIDENCE_THRESHOLD,
    ):
        self.device = _select_device()
        logger.info("Loading YOLO model on device: %s", self.device)

        self.model = YOLO(model_path)
        self.model.to(self.device)

        self.confidence_threshold = confidence_threshold
        logger.info("YOLO model loaded.")

    def detect_objects(
        self, frame: np.ndarray
    ) -> Tuple[List[Dict[str, Any]], Any]:
        """Run detection on a single frame.

        Returns:
            objects: list of {"name": str, "confidence": float}
            result: the raw Ultralytics result (for drawing boxes, etc.)
        """
        results = self.model(frame, device=self.device, verbose=False)
        result = results[0]

        objects: List[Dict[str, Any]] = []

        for box in result.boxes:
            confidence = float(box.conf[0])

            if confidence < self.confidence_threshold:
                continue

            class_id = int(box.cls[0])
            class_name = result.names[class_id]

            objects.append({"name": class_name, "confidence": confidence})

        return objects, result
