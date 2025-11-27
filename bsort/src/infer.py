"""Inference module for bsort."""

from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
from ultralytics import YOLO

from src.config import Config


def run_inference(config: Config, image_path: str) -> Dict[str, Any]:
    """Run inference on an image using trained model.

    Args:
        config: Configuration object
        image_path: Path to input image

    Returns:
        Dictionary containing inference results

    Raises:
        FileNotFoundError: If no trained model found
    """
    # Load model (prioritize ONNX for inference)
    model_path = Config.infer_model_path

    if model_path is None:
        raise FileNotFoundError("No trained model found for inference")

    model = YOLO(model_path)

    # Run inference
    results = model.predict(image_path, conf=0.5)

    # Process results
    detections = []
    for i, r in enumerate(results):
        image_detections = {
            "image_index": i,
            "num_detections": len(r.boxes),
            "detections": [],
        }

        if len(r.boxes) > 0:
            for j, box in enumerate(r.boxes):
                detection = {
                    "object_id": j + 1,
                    "class": int(box.cls[0]),
                    "confidence": float(box.conf[0]),
                    "bbox": box.xyxy[0].tolist() if hasattr(box.xyxy[0], "tolist") else box.xyxy[0],
                }
                image_detections["detections"].append(detection)

        detections.append(image_detections)

    return {
        "model_path": model_path,
        "image_path": image_path,
        "detections": detections,
        "num_images": len(results),
    }


def visualize_inference(image_path: str, detections: List[Dict], output_path: str) -> None:
    """Visualize inference results on image.

    Args:
        image_path: Path to input image
        detections: List of detection dictionaries
        output_path: Path to save output image
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for detection in detections:
        bbox = detection["bbox"]
        class_id = detection["class"]
        confidence = detection["confidence"]

        x1, y1, x2, y2 = map(int, bbox)

        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Draw label
        label = f"Class {class_id}: {confidence:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]

        cv2.rectangle(
            image,
            (x1, y1 - label_size[1] - 5),
            (x1 + label_size[0], y1),
            (255, 0, 0),
            -1,
        )
        cv2.putText(
            image,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
