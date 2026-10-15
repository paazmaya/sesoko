"""Optional detectors returning boxes in original-image coordinates."""

import importlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from PIL import Image


@dataclass(frozen=True)
class Detection:
    box: tuple[float, float, float, float]
    confidence: float
    label: str = "person"


class Detector(Protocol):
    def detect(self, image: Image.Image) -> list[Detection]: ...

    def get_available_classes(self) -> list[str]: ...


class YOLODetector:
    def __init__(self, model_path: str = "yolo11n-seg.pt") -> None:
        try:
            runtime: Any = importlib.import_module("ultralytics")
        except ImportError as exc:
            raise RuntimeError("YOLO requires: uv sync --extra detector-yolo") from exc
        self.model: Any = runtime.YOLO(model_path)

    def get_available_classes(self) -> list[str]:
        return list(self.model.names.values())

    def detect(self, image: Image.Image) -> list[Detection]:
        detections: list[Detection] = []
        for result in self.model(image, verbose=False):
            for box in result.boxes:
                left, top, right, bottom = (float(value) for value in box.xyxy[0].tolist())
                detections.append(
                    Detection(
                        (left, top, right, bottom),
                        float(box.conf[0]),
                        self.model.names[int(box.cls[0])],
                    )
                )
        return detections


class DEIMDetector:
    """Run the standard small HGNetv2 DEIMv2 COCO deployment export on CPU."""

    def __init__(self, model_path: str | Path, *, session: Any = None) -> None:
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("DEIMv2 requires: uv sync --extra detector-onnx") from exc
        self.np = np
        if session is None:
            if not Path(model_path).is_file():
                raise FileNotFoundError(f"DEIMv2 ONNX model not found: {model_path}")
            try:
                runtime: Any = importlib.import_module("onnxruntime")
            except ImportError as exc:
                raise RuntimeError("DEIMv2 requires: uv sync --extra detector-onnx") from exc
            session = runtime.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.session: Any = session
        inputs = {item.name: item for item in session.get_inputs()}
        if set(inputs) != {"images", "orig_target_sizes"}:
            raise ValueError("Expected DEIMv2 deployment inputs: images, orig_target_sizes")
        shape = inputs["images"].shape
        if (
            len(shape) != 4
            or shape[1] != 3
            or not all(isinstance(value, int) and value > 0 for value in shape[2:])
            or shape[2] != shape[3]
        ):
            raise ValueError("DEIMv2 requires a fixed square NCHW RGB input export")
        if inputs["images"].type != "tensor(float)":
            raise ValueError("DEIMv2 adapter expects an FP32 input export")
        if inputs["orig_target_sizes"].type != "tensor(int64)":
            raise ValueError("DEIMv2 adapter expects int64 orig_target_sizes")
        if not {"labels", "boxes", "scores"}.issubset(
            {item.name for item in session.get_outputs()}
        ):
            raise ValueError("Expected DEIMv2 deployment outputs: labels, boxes, scores")
        self.size = int(shape[2])

    def get_available_classes(self) -> list[str]:
        return ["person"]

    def detect(self, image: Image.Image) -> list[Detection]:
        np = self.np
        ratio = min(self.size / image.width, self.size / image.height)
        resized_size = (max(1, int(image.width * ratio)), max(1, int(image.height * ratio)))
        padding = ((self.size - resized_size[0]) // 2, (self.size - resized_size[1]) // 2)
        canvas = Image.new("RGB", (self.size, self.size))
        canvas.paste(image.convert("RGB").resize(resized_size, Image.Resampling.BILINEAR), padding)
        pixels = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
        labels, boxes, scores = self.session.run(
            ["labels", "boxes", "scores"],
            {
                "images": np.ascontiguousarray(pixels),
                "orig_target_sizes": np.array([[self.size, self.size]], dtype=np.int64),
            },
        )
        detections: list[Detection] = []
        for label, box, score in zip(labels[0], boxes[0], scores[0], strict=True):
            if int(label) != 0 or not math.isfinite(float(score)):
                continue
            if not np.isfinite(box).all():
                continue
            left, top, right, bottom = (
                (float(value) - padding[index % 2]) / ratio for index, value in enumerate(box)
            )
            coordinates = (
                max(0.0, left),
                max(0.0, top),
                min(float(image.width), right),
                min(float(image.height), bottom),
            )
            if coordinates[2] > coordinates[0] and coordinates[3] > coordinates[1]:
                detections.append(Detection(coordinates, float(score)))
        return detections
