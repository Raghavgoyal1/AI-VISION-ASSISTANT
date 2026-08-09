"""Webcam capture wrapper around OpenCV's VideoCapture."""

import logging
from typing import Optional

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)


class CameraEngine:
    """Thin, safe wrapper around cv2.VideoCapture.

    Supports use as a context manager:

        with CameraEngine() as camera:
            frame = camera.read()
    """

    def __init__(
        self,
        camera_index: int = config.CAMERA_INDEX,
        frame_width: int = config.FRAME_WIDTH,
        frame_height: int = config.FRAME_HEIGHT,
    ):
        self.camera_index = camera_index
        self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open webcam at index {self.camera_index}. "
                "Check that a camera is connected and not in use by "
                "another application."
            )

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

        logger.info("Camera %s opened successfully.", self.camera_index)

    def read(self) -> Optional[np.ndarray]:
        """Grab a single frame. Returns None if the read failed."""
        ret, frame = self.cap.read()

        if not ret:
            logger.warning("Failed to read frame from camera.")
            return None

        return frame

    def release(self) -> None:
        """Release the underlying camera device."""
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            logger.info("Camera released.")

    def __enter__(self) -> "CameraEngine":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()
