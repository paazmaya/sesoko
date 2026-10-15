#!/usr/bin/env python3
"""Exact-size image preparation with optional DEIMv2 or YOLO person detection."""

import json
import math
from pathlib import Path
from typing import Any, Optional, TypedDict, Union

import click
from PIL import Image

from lib.image_utils import (
    crop_to_square,
    get_image_files,
    open_image,
    prepare_crop,
    save_image_optimized,
)
from lib.person_detection import DEIMDetector, Detector, YOLODetector


class CropStats(TypedDict):
    base_folder: str
    output_folder: str
    crop_focus: str
    processed: list[str]
    skipped: list[str]
    failed: list[str]


class YOLOCropper:
    """Prepare crops through interchangeable, lazily loaded detectors."""

    def __init__(
        self,
        crop_focus: Optional[str] = None,
        resolution: int = 512,
        *,
        detector: Detector | None = None,
        backend: str = "yolo",
        model_path: str | None = None,
        confidence: float = 0.4,
        margin: float = 0.2,
        width: int | None = None,
        height: int | None = None,
        output_format: str = "jpeg",
        quality: int = 85,
    ) -> None:
        """
        Initialize the YOLO cropper.

        Args:
            crop_focus: Object class to focus on (e.g., 'person' or YOLO's 'dog').
            resolution: Default output width and height, independently overridable.
        """
        self.crop_focus = crop_focus
        self.resolution = resolution
        self.output_size = (width or resolution, height or resolution)
        if resolution <= 0 or any(value is not None and value <= 0 for value in (width, height)):
            raise ValueError("Output dimensions must be positive")
        if not 0 <= confidence <= 1 or not math.isfinite(margin) or margin < 0:
            raise ValueError("Confidence must be in [0, 1] and margin finite and nonnegative")
        if backend not in {"yolo", "deim"}:
            raise ValueError("Detector must be yolo or deim")
        if backend == "deim" and crop_focus and crop_focus.lower() != "person":
            raise ValueError("The DEIMv2 adapter supports --crop-focus person only")
        if output_format not in {"jpeg", "png", "webp"} or not 1 <= quality <= 100:
            raise ValueError("Expected jpeg/png/webp format and quality in [1, 100]")
        self.backend = backend
        self.model_path = model_path
        self.confidence = confidence
        self.margin = margin
        self.output_format = output_format
        self.quality = quality
        self._detector = detector

    @property
    def model(self) -> Any:
        return getattr(self._detector, "model", None)

    def _get_detector(self) -> Detector:
        if self._detector is None:
            if self.backend == "deim":
                if not self.model_path:
                    raise ValueError("DEIMv2 requires --model pointing to an HGNetv2 ONNX export")
                self._detector = DEIMDetector(self.model_path)
            else:
                self._detector = YOLODetector(self.model_path or "yolo11n-seg.pt")
        return self._detector

    def get_available_classes(self) -> list[str]:
        """Get list of available object classes YOLO can detect."""
        return self._get_detector().get_available_classes()

    def process_image(
        self, image: Image.Image, skip_if_not_found: bool = True
    ) -> Optional[Image.Image]:
        """
        Prepare an exact-size center crop or retain the largest detected subject.

        Args:
            image: PIL Image to process
            skip_if_not_found: If True, return None if crop_focus object not found

        Returns:
            Processed image or None if object not found and skip_if_not_found=True.
        """
        if self.crop_focus:
            cropped = self._content_aware_crop(image)
            if cropped is not None or skip_if_not_found:
                return cropped
        return prepare_crop(image, self.output_size)

    def _has_target_object(self, image: Image.Image) -> bool:
        """Check if image contains the target object."""
        return bool(self.crop_focus and self._select_target(image))

    def _select_target(
        self, image: Image.Image
    ) -> tuple[tuple[float, float, float, float], float] | None:
        if self.crop_focus is None:
            return None
        targets: list[tuple[tuple[float, float, float, float], float]] = []
        for detection in self._get_detector().detect(image):
            if detection.label.lower() != self.crop_focus.lower():
                continue
            if not self.confidence <= detection.confidence <= 1:
                continue
            if not all(math.isfinite(value) for value in detection.box):
                continue
            left, top, right, bottom = detection.box
            box = (max(0, left), max(0, top), min(image.width, right), min(image.height, bottom))
            if box[2] > box[0] and box[3] > box[1]:
                targets.append((box, detection.confidence))
        return max(
            targets,
            key=lambda target: (
                (target[0][2] - target[0][0]) * (target[0][3] - target[0][1]),
                target[1],
                target[0],
            ),
            default=None,
        )

    def _content_aware_crop(self, image: Image.Image) -> Optional[Image.Image]:
        """Crop around the largest target without clipping its detected bounds."""
        if not self.crop_focus:
            return None

        target = self._select_target(image)
        if target is None:
            return None
        return prepare_crop(image, self.output_size, target[0], self.margin)

    def _center_crop(self, image: Image.Image) -> Image.Image:
        """Crop image to square using the smaller dimension (center-aligned, no padding)."""
        return crop_to_square(image)

    def _square_pad(self, image: Image.Image) -> Image.Image:
        """Crop an image to a square using the smaller dimension.

        The output square will be (min_dimension x min_dimension), ensuring
        no upscaling or black bars - the larger dimension is cropped.
        """
        return crop_to_square(image)

    def _output_path(self, file_path: Path, output_path: Path, stem_counts: dict[str, int]) -> Path:
        stem = file_path.name if stem_counts[file_path.stem.casefold()] > 1 else file_path.stem
        extension = "jpg" if self.output_format == "jpeg" else self.output_format
        candidate = output_path / f"{stem}.{extension}"
        sequence = 2
        while candidate.exists():
            candidate = output_path / f"{stem}_{sequence}.{extension}"
            sequence += 1
        return candidate

    def process_folder(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        skip_if_not_found: bool = True,
    ) -> CropStats:
        """
        Process top-level images and save new files in the configured format.

        Args:
            input_dir: Path to input images
            output_dir: Path to save cropped images
            skip_if_not_found: Skip images where crop_focus object not found

        Returns:
            Stats about processed, skipped, and failed files
        """
        input_path = Path(input_dir).resolve()
        output_path = Path(output_dir).resolve()
        if input_path == output_path:
            raise ValueError(
                "Input and output directories must differ; originals are never overwritten"
            )
        output_path.mkdir(parents=True, exist_ok=True)
        if self.crop_focus:
            self._get_detector()

        stats: CropStats = {
            "base_folder": str(input_path),
            "output_folder": str(output_path),
            "crop_focus": self.crop_focus or "center",
            "processed": [],
            "skipped": [],
            "failed": [],
        }

        image_files = get_image_files(input_path, recursive=False)
        stem_counts: dict[str, int] = {}
        for file_path in image_files:
            stem = file_path.stem.casefold()
            stem_counts[stem] = stem_counts.get(stem, 0) + 1

        for file_path in image_files:
            try:
                img = open_image(file_path)
                processed = self.process_image(img, skip_if_not_found=skip_if_not_found)

                if processed is None:
                    stats["skipped"].append(str(file_path))
                    continue

                output_file = self._output_path(file_path, output_path, stem_counts)
                save_image_optimized(processed, output_file, self.output_format, self.quality)
                stats["processed"].append(str(file_path))

            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                stats["failed"].append(str(file_path))

        return stats


@click.command()
@click.option(
    "--input-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=False,
    help="Input folder with images",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True),
    required=False,
    help="Output folder for cropped images",
)
@click.option(
    "--crop-focus",
    type=str,
    default=None,
    help="Object class to focus on (e.g., person). Leave empty for center crop.",
)
@click.option(
    "--resolution",
    type=click.IntRange(min=1),
    default=512,
    help="Output image resolution (width and height)",
)
@click.option("--width", type=click.IntRange(min=1), help="Override output width")
@click.option("--height", type=click.IntRange(min=1), help="Override output height")
@click.option("--detector", "backend", type=click.Choice(["yolo", "deim"]), default="yolo")
@click.option("--model", "model_path", type=click.Path(dir_okay=False), default=None)
@click.option("--confidence", type=click.FloatRange(0, 1), default=0.4, show_default=True)
@click.option("--margin", type=click.FloatRange(min=0), default=0.2, show_default=True)
@click.option(
    "--format", "output_format", type=click.Choice(["jpeg", "png", "webp"]), default="jpeg"
)
@click.option("--quality", type=click.IntRange(1, 100), default=85, show_default=True)
@click.option("--fallback-center", is_flag=True, help="Center crop when no target is found")
@click.option(
    "--list-classes",
    is_flag=True,
    help="List all available YOLO object classes and exit",
)
@click.option(
    "--stats",
    type=click.Path(file_okay=True, dir_okay=False),
    default=None,
    help="Save processing statistics to JSON file",
)
def main(
    input_dir: str,
    output_dir: str,
    crop_focus: Optional[str],
    resolution: int,
    list_classes: bool,
    stats: Optional[str],
    **options: Any,
) -> None:
    """
    Prepare exact-size images with optional DEIMv2 or YOLO detection.

    Examples:

    \b
    # Center crop all images to 512x512
    uv run python src/crop_yolo.py --input-dir images/ --output-dir output/

    \b
    # Crop focusing on 'person' objects
    uv run python src/crop_yolo.py --input-dir images/ --output-dir output/ --crop-focus person

    \b
    # List all available object classes
    uv run python src/crop_yolo.py --list-classes

    \b
    # Save processing statistics
    uv run python src/crop_yolo.py --input-dir images/ --output-dir output/ --crop-focus person --stats stats.json
    """
    fallback_center = options.pop("fallback_center", False)
    try:
        cropper = YOLOCropper(
            crop_focus=crop_focus,
            resolution=resolution,
            **options,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    if list_classes:
        try:
            classes = cropper.get_available_classes()
        except (ValueError, RuntimeError, FileNotFoundError) as exc:
            raise click.ClickException(str(exc)) from exc
        click.echo("Available YOLO object classes:")
        for cls in sorted(classes):
            click.echo(f"  - {cls}")
        return

    if not input_dir or not output_dir:
        click.echo(click.get_current_context().get_help())
        raise click.UsageError(
            "--input-dir and --output-dir are required unless using --list-classes"
        )

    click.echo(f"Processing images from: {input_dir}")
    if crop_focus:
        click.echo(f"Crop focus: {crop_focus}")
    else:
        click.echo("Crop focus: center crop")
    click.echo(f"Output resolution: {cropper.output_size[0]}x{cropper.output_size[1]}")
    click.echo(f"Output format: {cropper.output_format.upper()} (quality {cropper.quality})")
    click.echo(f"Output folder: {output_dir}")
    click.echo()

    try:
        result_stats = cropper.process_folder(input_dir, output_dir, not fallback_center)
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(f"Processed: {len(result_stats['processed'])} images")
    click.echo(f"Skipped: {len(result_stats['skipped'])} images")
    click.echo(f"Failed: {len(result_stats['failed'])} images")

    if stats:
        stats_path = Path(stats)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_path, "w") as f:
            json.dump(result_stats, f, indent=2)
        click.echo(f"Statistics saved to: {stats}")
    if result_stats["failed"]:
        raise click.ClickException(f"Failed to process {len(result_stats['failed'])} image(s)")


if __name__ == "__main__":
    main()
