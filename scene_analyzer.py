"""Combines YOLO detections and a BLIP caption into one readable description."""

from typing import Any, Dict, List


class SceneAnalyzer:
    """Fuses object-detection results with a scene caption."""

    def create_scene_description(
        self,
        objects: List[Dict[str, Any]],
        caption: str,
    ) -> str:
        """Build a human-readable scene description.

        Args:
            objects: list of {"name": str, "confidence": float} from YOLO.
            caption: free-text caption from BLIP.

        Returns:
            A single descriptive sentence (or two).
        """
        if not objects:
            return f"Scene description: {caption}"

        # Deduplicate while preserving detection order.
        object_names: List[str] = []
        for obj in objects:
            name = obj["name"]
            if name not in object_names:
                object_names.append(name)

        object_text = ", ".join(object_names)

        return f"I can see {object_text}. {caption.capitalize()}."
