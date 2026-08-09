"""BLIP-based image captioning."""

import logging

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

import config

logger = logging.getLogger(__name__)


def _select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class CaptionEngine:
    """Wraps a BLIP model to generate natural-language captions for frames."""

    def __init__(self, model_name: str = config.BLIP_MODEL_NAME):
        self.device = _select_device()
        logger.info("Loading BLIP model '%s' on device: %s", model_name, self.device)

        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)

        logger.info("BLIP model loaded.")

    def generate_caption(
        self, frame: np.ndarray, max_new_tokens: int = config.CAPTION_MAX_NEW_TOKENS
    ) -> str:
        """Generate a caption describing the given BGR frame."""
        if frame is None:
            raise ValueError("generate_caption() received an empty frame.")

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_frame)

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        caption = self.processor.decode(output[0], skip_special_tokens=True)
        return caption
