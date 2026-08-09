"""
AI Vision Assistant
--------------------
Real-time webcam pipeline that combines:
  - YOLO11        -> object detection
  - BLIP          -> image captioning
  - SceneAnalyzer -> fuses both into one natural-language description

Controls:
  SPACE -> analyze the current frame (caption + scene description)
  Q     -> quit
"""

import argparse
import logging

import cv2

import config
from camera_engine import CameraEngine
from caption_engine import CaptionEngine
from scene_analyzer import SceneAnalyzer
from vision_engine import VisionEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI Vision Assistant")
    parser.add_argument(
        "--camera",
        type=int,
        default=config.CAMERA_INDEX,
        help="Camera index to use (default: %(default)s)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=config.YOLO_CONFIDENCE_THRESHOLD,
        help="YOLO confidence threshold, 0-1 (default: %(default)s)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=config.YOLO_MODEL_PATH,
        help="Path or name of the YOLO model to load (default: %(default)s)",
    )
    return parser.parse_args()


def format_object_text(objects) -> str:
    if not objects:
        return "Detected: None"

    parts = [f"{obj['name']} ({obj['confidence']:.0%})" for obj in objects]
    return "Detected: " + ", ".join(parts)


def main() -> None:
    args = parse_args()

    logger.info("Starting AI Vision Assistant...")

    camera = None

    try:
        camera = CameraEngine(camera_index=args.camera)
        vision = VisionEngine(
            model_path=args.model, confidence_threshold=args.confidence
        )
        caption_engine = CaptionEngine()
        scene_analyzer = SceneAnalyzer()
    except Exception:
        logger.exception("Failed to initialize AI Vision Assistant.")
        if camera is not None:
            camera.release()
        return

    logger.info("System ready! SPACE = analyze scene, Q = quit.")

    caption = "Press SPACE to analyze"
    scene_description = ""

    try:
        while True:
            frame = camera.read()

            if frame is None:
                logger.warning("Could not read from webcam. Stopping.")
                break

            objects, result = vision.detect_objects(frame)
            display_frame = result.plot()

            object_text = format_object_text(objects)

            key = cv2.waitKey(1) & 0xFF

            if key == ord(" "):
                logger.info("Analyzing scene...")

                try:
                    caption = caption_engine.generate_caption(frame)
                    scene_description = scene_analyzer.create_scene_description(
                        objects, caption
                    )

                    logger.info("YOLO detections:")
                    for obj in objects:
                        logger.info(
                            "  %s -> %.2f%%", obj["name"], obj["confidence"] * 100
                        )

                    logger.info("BLIP caption: %s", caption)
                    logger.info("Final scene description: %s", scene_description)
                except Exception:
                    logger.exception("Scene analysis failed; keeping previous result.")

            cv2.putText(
                display_frame,
                object_text,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                config.TEXT_COLOR,
                2,
            )

            cv2.putText(
                display_frame,
                "BLIP: " + caption,
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                config.TEXT_COLOR,
                2,
            )

            if scene_description:
                cv2.putText(
                    display_frame,
                    "Scene: " + scene_description,
                    (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    config.TEXT_COLOR,
                    2,
                )

            cv2.imshow(config.WINDOW_NAME, display_frame)

            if key == ord("q"):
                break

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        camera.release()
        cv2.destroyAllWindows()
        logger.info("AI Vision Assistant stopped.")


if __name__ == "__main__":
    main()
