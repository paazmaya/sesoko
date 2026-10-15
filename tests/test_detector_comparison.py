"""Opt-in comparison of local DEIMv2 and YOLO checkpoints; no weight downloads.

Enable with SESOKO_COMPARE_DETECTORS=1. SESOKO_COMPARE_IMAGES accepts an image
or directory; SESOKO_COMPARE_OUTPUT selects the report directory. Both default
to local project paths. Timings compare CPU FP32 inference, not captioning.
"""

import hashlib
import json
import math
import os
import platform
import statistics
import time
from dataclasses import asdict, dataclass
from functools import partial
from importlib.metadata import version
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from crop_yolo import YOLOCropper
from lib.image_utils import get_image_files, open_image
from lib.person_detection import DEIMDetector, Detection, Detector, YOLODetector

PROJECT_ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(
    os.environ.get("SESOKO_COMPARE_DETECTORS") != "1",
    reason="Real-model comparison is opt-in: set SESOKO_COMPARE_DETECTORS=1",
)


@dataclass
class DetectionSnapshot:
    detections: list[Detection]

    def detect(self, image: Image.Image) -> list[Detection]:
        del image
        return self.detections

    def get_available_classes(self) -> list[str]:
        return ["person"]


def box_iou(first: Detection, second: Detection) -> float:
    left = max(first.box[0], second.box[0])
    top = max(first.box[1], second.box[1])
    right = min(first.box[2], second.box[2])
    bottom = min(first.box[3], second.box[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = (first.box[2] - first.box[0]) * (first.box[3] - first.box[1])
    second_area = (second.box[2] - second.box[0]) * (second.box[3] - second.box[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def largest_person(detections: list[Detection]) -> Detection | None:
    return max(
        detections,
        key=lambda detection: (
            (detection.box[2] - detection.box[0]) * (detection.box[3] - detection.box[1]),
            detection.confidence,
            detection.box,
        ),
        default=None,
    )


def save_visuals(
    image: Image.Image, people: list[Detection], output: Path, model_name: str
) -> str | None:
    annotated = image.copy()
    drawing = ImageDraw.Draw(annotated)
    for person in people:
        drawing.rectangle(person.box, outline="red", width=3)
        drawing.text(
            (person.box[0], person.box[1]),
            f"person {person.confidence:.3f}",
            fill="white",
            stroke_width=1,
            stroke_fill="black",
        )
    annotated.save(output / f"{model_name}_detections.png")
    cropper = YOLOCropper(
        crop_focus="person", detector=DetectionSnapshot(people), resolution=512, margin=0.2
    )
    cropped = cropper.process_image(image)
    if cropped is None:
        return None
    assert cropped.size == (512, 512)
    path = output / f"{model_name}_crop.png"
    cropped.save(path)
    return str(path)


def measure_detector(
    detector: Detector, image: Image.Image, repeats: int, confidence: float
) -> tuple[list[Detection], dict[str, float]]:
    detector.detect(image)
    samples: list[float] = []
    detections: list[Detection] = []
    for _ in range(repeats):
        started = time.perf_counter()
        detections = detector.detect(image)
        samples.append((time.perf_counter() - started) * 1000)
    assert all(math.isfinite(sample) and sample >= 0 for sample in samples)
    people = [
        detection
        for detection in detections
        if detection.label == "person" and detection.confidence >= confidence
    ]
    for person in people:
        left, top, right, bottom = person.box
        assert all(math.isfinite(value) for value in person.box)
        assert 0 <= left < right <= image.width
        assert 0 <= top < bottom <= image.height
        assert confidence <= person.confidence <= 1
    return people, {
        "median_ms": statistics.median(samples),
        "p95_ms": sorted(samples)[math.ceil(0.95 * len(samples)) - 1],
        "min_ms": min(samples),
    }


def test_compare_local_person_detectors() -> None:
    pytest.importorskip("numpy")
    ort = pytest.importorskip("onnxruntime")
    torch = pytest.importorskip("torch")
    pytest.importorskip("ultralytics")
    models = {
        "deimv2-pico": PROJECT_ROOT / "deimv2_hgnetv2_pico_coco.onnx",
        "yolo11n-seg": PROJECT_ROOT / "yolo11n-seg.pt",
        "yolov8n-seg": PROJECT_ROOT / "yolov8n-seg.pt",
    }
    for path in models.values():
        assert path.is_file(), f"Local checkpoint required; nothing is downloaded: {path}"

    source = (
        Path(
            os.environ.get(
                "SESOKO_COMPARE_IMAGES", str(PROJECT_ROOT / "tests" / "yolo_person_detected.png")
            )
        )
        .expanduser()
        .resolve()
    )
    assert source.exists(), f"Image input does not exist: {source}"
    images = get_image_files(source) if source.is_dir() else [source]
    assert images, f"No supported images found in {source}"
    output = (
        Path(
            os.environ.get(
                "SESOKO_COMPARE_OUTPUT", str(PROJECT_ROOT / "target" / "detector-comparison")
            )
        )
        .expanduser()
        .resolve()
    )
    output.mkdir(parents=True, exist_ok=True)
    repeats = int(os.environ.get("SESOKO_COMPARE_REPEATS", "5"))
    assert repeats >= 1, "SESOKO_COMPARE_REPEATS must be positive"
    confidence = 0.4
    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    report: dict[str, object] = {
        "device": "cpu",
        "precision": "float32",
        "threads": 1,
        "input_size": 640,
        "confidence": confidence,
        "warmups_per_image": 1,
        "measured_repeats": repeats,
        "platform": platform.platform(),
        "versions": {
            name: version(name)
            for name in ("onnxruntime", "torch", "ultralytics", "numpy", "pillow")
        },
        "notes": [
            "No ground-truth labels: IoU measures agreement, not accuracy.",
            "Confidence scores from different models are not calibrated against each other.",
            "Timings include adapter preprocessing/postprocessing, not image decoding or crops.",
            "YOLO segmentation checkpoints compute masks even though only person boxes are used.",
            "Default input is an existing annotated sample, not a representative evaluation set.",
            "CPU only: this does not verify NVIDIA execution or caption quality.",
        ],
    }
    results: dict[str, list[dict[str, object]]] = {}
    selected: dict[str, list[Detection | None]] = {}
    try:
        for model_name, model_path in models.items():
            started = time.perf_counter()
            detector: Detector
            if model_name == "deimv2-pico":
                options = ort.SessionOptions()
                options.intra_op_num_threads = 1
                options.inter_op_num_threads = 1
                session = ort.InferenceSession(
                    str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
                )
                detector = DEIMDetector(model_path, session=session)
            else:
                yolo = YOLODetector(str(model_path))
                yolo.model.predict = partial(
                    yolo.model.predict,
                    device="cpu",
                    imgsz=640,
                    rect=False,
                    conf=confidence,
                    half=False,
                )
                detector = yolo
            load_seconds = time.perf_counter() - started
            with model_path.open("rb") as checkpoint:
                digest = hashlib.file_digest(checkpoint, "sha256").hexdigest()
            rows: list[dict[str, object]] = []
            selected[model_name] = []
            for index, image_path in enumerate(images):
                image = open_image(image_path)
                people, timings = measure_detector(detector, image, repeats, confidence)
                if isinstance(detector, YOLODetector):
                    assert detector.model.predictor.args.rect is False
                    assert detector.model.predictor.args.device == "cpu"
                if "SESOKO_COMPARE_IMAGES" not in os.environ:
                    assert people, f"{model_name} missed the person in the default positive sample"
                image_output = output / f"{index:04d}_{image_path.stem}"
                image_output.mkdir(parents=True, exist_ok=True)
                crop_path = save_visuals(image, people, image_output, model_name)
                selected[model_name].append(largest_person(people))
                rows.append(
                    {
                        "image": str(image_path),
                        "image_size": list(image.size),
                        "checkpoint": str(model_path),
                        "checkpoint_sha256": digest,
                        "checkpoint_bytes": model_path.stat().st_size,
                        "load_seconds": load_seconds,
                        "person_count": len(people),
                        "detections": [asdict(person) for person in people],
                        "crop": crop_path,
                        **timings,
                    }
                )
                print(
                    f"{model_name:14s} {image_path.name}: {len(people)} person(s), "
                    f"median={timings['median_ms']:.1f} ms, p95={timings['p95_ms']:.1f} ms"
                )
            results[model_name] = rows
            del detector
        comparisons: list[dict[str, object]] = []
        for index, image_path in enumerate(images):
            deim = selected["deimv2-pico"][index]
            for model_name in ("yolo11n-seg", "yolov8n-seg"):
                yolo_person = selected[model_name][index]
                comparisons.append(
                    {
                        "image": str(image_path),
                        "models": ["deimv2-pico", model_name],
                        "largest_person_iou": box_iou(deim, yolo_person)
                        if deim is not None and yolo_person is not None
                        else None,
                    }
                )
        report["results"] = results
        report["largest_person_agreement"] = comparisons
        report_path = output / "comparison.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Comparison report and visual outputs: {output}")
    finally:
        torch.set_num_threads(original_threads)
