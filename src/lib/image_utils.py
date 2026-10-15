"""Shared image handling utilities for caption and cropping scripts."""

import importlib
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

# Support all common image formats
SUPPORTED_IMAGE_EXTENSIONS = {
    # Standard formats
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".webp",
    ".bmp",
    ".tiff",
    ".tif",
    # Apple formats
    ".heic",
    ".heif",
    # Modern formats
    ".avif",
    ".jxl",  # JPEG XL
    # Lossless formats
    ".psd",
    ".ico",
    ".cur",
}

# Register additional image handlers if available
try:
    heif: Any = importlib.import_module("pillow_heif")
    heif.register_heif_opener()
except ImportError:
    pass

try:
    from PIL import ImageFile

    ImageFile.LOAD_TRUNCATED_IMAGES = True
except ImportError:
    pass


def get_image_files(folder_path: Path, recursive: bool = True) -> list[Path]:
    """Get all image files from the given folder.

    Args:
        folder_path: Path to the folder containing images
        recursive: If True, search recursively; if False, only search top level

    Returns:
        List of paths to image files, sorted
    """
    image_files: list[Path] = []
    iterator = folder_path.rglob("*") if recursive else folder_path.iterdir()

    for file in iterator:
        if file.is_file() and file.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            image_files.append(file)

    return sorted(image_files)


def open_image(image_path: Path) -> Image.Image:
    """Safely open an image file and convert to RGB.

    Args:
        image_path: Path to the image file

    Returns:
        PIL Image in RGB mode, or None if unable to open

    Raises:
        Exception: Re-raises any exception that occurs during opening
    """
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source)
        if "A" in image.getbands() or "transparency" in image.info:
            rgba = image.convert("RGBA")
            background = Image.new("RGBA", rgba.size, "white")
            return Image.alpha_composite(background, rgba).convert("RGB")
        return image.convert("RGB")


def resize_image_aspect_ratio(
    image: Image.Image, target_size: int, *, allow_upscale: bool = True
) -> Image.Image:
    """Resize image to fit within target_size maintaining aspect ratio.

    Args:
        image: PIL Image object
        target_size: Target size in pixels (longest side)

    Returns:
        Resized PIL Image object
    """
    if target_size <= 0:
        raise ValueError("Target size must be positive")
    if not allow_upscale and max(image.size) <= target_size:
        return image
    scale = target_size / max(image.size)
    size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def crop_to_square(image: Image.Image) -> Image.Image:
    """Crop image to square using the smaller dimension (center-aligned, no padding).

    Args:
        image: PIL Image object

    Returns:
        Cropped square PIL Image object
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


def prepare_crop(
    image: Image.Image,
    output_size: tuple[int, int],
    subject_box: tuple[float, float, float, float] | None = None,
    margin: float = 0.2,
    background: str = "white",
) -> Image.Image:
    """Fit a subject into an exact-size crop without clipping or stretching it."""
    width, height = output_size
    if width <= 0 or height <= 0:
        raise ValueError("Output dimensions must be positive")
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("Margin must be finite and nonnegative")
    if subject_box is None:
        return ImageOps.fit(image, output_size, method=Image.Resampling.LANCZOS)

    if not all(math.isfinite(value) for value in subject_box):
        raise ValueError("Subject coordinates must be finite")
    left, top, right, bottom = subject_box
    left, top = max(0, left), max(0, top)
    right, bottom = min(image.width, right), min(image.height, bottom)
    if right <= left or bottom <= top:
        raise ValueError("Subject box must overlap the image and have positive area")

    ratio = width / height
    crop_width = max(right - left, (bottom - top) * ratio)
    crop_height = crop_width / ratio
    available_scale = min(image.width / crop_width, image.height / crop_height)
    scale = min(1 + margin, max(1, available_scale))
    crop_width = math.ceil(crop_width * scale)
    crop_height = math.ceil(crop_height * scale)
    crop_left = round((left + right - crop_width) / 2)
    crop_top = round((top + bottom - crop_height) / 2)
    if crop_width <= image.width:
        crop_left = min(max(0, crop_left), image.width - crop_width)
    else:
        crop_left = min(max(crop_left, math.ceil(right - crop_width)), math.floor(left))
    if crop_height <= image.height:
        crop_top = min(max(0, crop_top), image.height - crop_height)
    else:
        crop_top = min(max(crop_top, math.ceil(bottom - crop_height)), math.floor(top))

    region = image.crop(
        (
            max(0, crop_left),
            max(0, crop_top),
            min(image.width, crop_left + crop_width),
            min(image.height, crop_top + crop_height),
        )
    )
    canvas = Image.new("RGB", (crop_width, crop_height), background)
    canvas.paste(region, (max(0, -crop_left), max(0, -crop_top)))
    return ImageOps.pad(canvas, output_size, method=Image.Resampling.LANCZOS, color=background)


def save_image_optimized(
    image: Image.Image, output_path: Path, output_format: str = "jpeg", quality: int = 85
) -> None:
    """Save image as optimized JPEG.

    Args:
        image: PIL Image object to save
        output_path: Path where to save the image
    """
    if output_format not in {"jpeg", "png", "webp"} or not 1 <= quality <= 100:
        raise ValueError("Expected jpeg/png/webp format and quality in [1, 100]")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    options = {"optimize": True} if output_format == "png" else {"quality": quality}
    if output_format == "jpeg":
        options["optimize"] = True
    image.save(output_path, output_format.upper(), **options)
