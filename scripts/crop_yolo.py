#!/usr/bin/env python3
"""
YOLO-based Image Cropper - Detect and crop objects in images using YOLO11.

Supports object detection and cropping with smart padding and resizing.
"""

import json
from pathlib import Path
from typing import Optional, Union

import click
from PIL import Image
from ultralytics import YOLO  # type: ignore


class YOLOCropper:
    """Crop images based on YOLO object detection."""

    def __init__(self, crop_focus: Optional[str] = None, resolution: int = 512):
        """
        Initialize the YOLO cropper.

        Args:
            crop_focus: Object class to focus on (e.g., 'person', 'face', 'dog')
            resolution: Output resolution for cropped images
        """
        self.crop_focus = crop_focus
        self.resolution = resolution
        self.model = YOLO("yolo11n-seg.pt")

    def get_available_classes(self) -> list[str]:
        """Get list of available object classes YOLO can detect."""
        return list(self.model.names.values())

    def process_image(
        self, image: Image.Image, skip_if_not_found: bool = True
    ) -> Optional[Image.Image]:
        """
        Crop to square and resize a single image based on YOLO detection.

        Always produces a square image by cropping to the smallest dimension,
        then resizing to the target resolution.

        Args:
            image: PIL Image to process
            skip_if_not_found: If True, return None if crop_focus object not found

        Returns:
            Processed square image or None if object not found and skip_if_not_found=True
        """
        if self.crop_focus:
            cropped = self._content_aware_crop(image)
            if cropped is None and skip_if_not_found:
                return None
            if cropped is None:
                # No target found, use center crop as fallback
                image = self._center_crop(image)
            else:
                image = cropped
        else:
            image = self._center_crop(image)

        image = image.resize((self.resolution, self.resolution), Image.Resampling.LANCZOS)
        return image

    def _has_target_object(self, image: Image.Image) -> bool:
        """Check if image contains the target object."""
        if not self.crop_focus:
            return False

        results = self.model(image, verbose=False)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                cls_name = self.model.names[cls_id]

                if cls_name.lower() == self.crop_focus.lower():
                    return True

        return False

    def _content_aware_crop(self, image: Image.Image) -> Optional[Image.Image]:
        """
        Use YOLO11 to detect the focus object and crop around it to a square.

        Detects the target object, creates a square crop around it with padding,
        then converts to square using the smaller dimension.

        Returns None if target not found.
        """
        if not self.crop_focus:
            return None

        results = self.model(image, verbose=False)
        target_box = None

        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                cls_name = self.model.names[cls_id]

                if cls_name.lower() == self.crop_focus.lower():
                    xyxy = box.xyxy[0].cpu().numpy()
                    target_box = xyxy
                    break
            if target_box is not None:
                break

        if target_box is None:
            return None

        # Calculate crop with padding
        x1, y1, x2, y2 = target_box
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2
        cy = y1 + h / 2

        size = max(w, h) * 1.2  # 20% padding

        half_size = size / 2
        crop_x1 = max(0, cx - half_size)
        crop_y1 = max(0, cy - half_size)
        crop_x2 = min(image.width, cx + half_size)
        crop_y2 = min(image.height, cy + half_size)

        cropped = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        return self._square_pad(cropped)

    def _center_crop(self, image: Image.Image) -> Image.Image:
        """Crop image to square using the smaller dimension (center-aligned, no padding)."""
        width, height = image.size
        new_size = min(width, height)

        left = (width - new_size) / 2
        top = (height - new_size) / 2
        right = (width + new_size) / 2
        bottom = (height + new_size) / 2

        return image.crop((left, top, right, bottom))

    def _square_pad(self, image: Image.Image) -> Image.Image:
        """Crop an image to a square using the smaller dimension.

        The output square will be (min_dimension x min_dimension), ensuring
        no upscaling or black bars - the larger dimension is cropped.
        """
        width, height = image.size
        if width == height:
            return image

        new_size = min(width, height)
        left = (width - new_size) / 2
        top = (height - new_size) / 2
        right = left + new_size
        bottom = top + new_size

        return image.crop((left, top, right, bottom))

    def process_folder(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        skip_if_not_found: bool = True,
    ) -> dict:
        """
        Process all images in input_dir and save to output_dir as optimized JPEG files.

        Args:
            input_dir: Path to input images
            output_dir: Path to save cropped images
            skip_if_not_found: Skip images where crop_focus object not found

        Returns:
            Stats about processed, skipped, and failed files
        """
        input_path = Path(input_dir).resolve()
        output_path = Path(output_dir).resolve()
        output_path.mkdir(parents=True, exist_ok=True)

        stats = {
            "base_folder": str(input_path),
            "output_folder": str(output_path),
            "crop_focus": self.crop_focus or "center",
            "processed": [],
            "skipped": [],
            "failed": [],
        }

        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        for file_path in sorted(input_path.iterdir()):
            if file_path.suffix.lower() in valid_extensions:
                try:
                    img = Image.open(file_path).convert("RGB")
                    processed = self.process_image(img, skip_if_not_found=skip_if_not_found)

                    if processed is None:
                        stats["skipped"].append(str(file_path))
                        continue

                    output_file = output_path / f"{file_path.stem}.jpg"
                    processed.save(output_file, "JPEG", quality=85, optimize=True)
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
    help="Object class to focus crop on (e.g., person, face, dog). Leave empty for center crop.",
)
@click.option(
    "--resolution",
    type=int,
    default=512,
    help="Output image resolution (width and height)",
)
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
):
    """
    Crop images using YOLO11 object detection.

    Examples:

    \b
    # Center crop all images to 512x512
    python scripts/crop_yolo.py --input-dir images/ --output-dir output/

    \b
    # Crop focusing on 'person' objects
    python scripts/crop_yolo.py --input-dir images/ --output-dir output/ --crop-focus person

    \b
    # List all available object classes
    python scripts/crop_yolo.py --list-classes

    \b
    # Save processing statistics
    python scripts/crop_yolo.py --input-dir images/ --output-dir output/ --crop-focus face --stats stats.json
    """
    cropper = YOLOCropper(crop_focus=crop_focus, resolution=resolution)

    if list_classes:
        classes = cropper.get_available_classes()
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
    click.echo(f"Output resolution: {resolution}x{resolution}")
    click.echo("Output format: JPEG (quality 85, optimized)")
    click.echo(f"Output folder: {output_dir}")
    click.echo()

    result_stats = cropper.process_folder(input_dir, output_dir)

    click.echo(f"Processed: {len(result_stats['processed'])} images")
    click.echo(f"Skipped: {len(result_stats['skipped'])} images")
    click.echo(f"Failed: {len(result_stats['failed'])} images")

    if stats:
        stats_path = Path(stats)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_path, "w") as f:
            json.dump(result_stats, f, indent=2)
        click.echo(f"Statistics saved to: {stats}")


if __name__ == "__main__":
    main()
