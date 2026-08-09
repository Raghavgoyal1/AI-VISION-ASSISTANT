"""
Central configuration for AI Vision Assistant.

Keeping these values in one place means camera, model, and UI
behaviour can be tuned without touching the engine logic.
"""

# --- Camera ---------------------------------------------------------------
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# --- YOLO (object detection) ----------------------------------------------
YOLO_MODEL_PATH = "yolo11n.pt"          # auto-downloaded by ultralytics if missing
YOLO_CONFIDENCE_THRESHOLD = 0.50

# --- BLIP (image captioning) -----------------------------------------------
BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-base"
CAPTION_MAX_NEW_TOKENS = 30

# --- UI ---------------------------------------------------------------------
WINDOW_NAME = "AI Vision Assistant"
FONT = "FONT_HERSHEY_SIMPLEX"
TEXT_COLOR = (0, 255, 0)
