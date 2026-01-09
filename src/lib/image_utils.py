"""Shared image handling utilities for caption and cropping scripts."""

from pathlib import Path
from typing import Optional

from PIL import Image

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
    import pillow_heif

    pillow_heif.register_heif_opener()
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
    image_files = []
    iterator = folder_path.rglob("*") if recursive else folder_path.iterdir()

    for file in iterator:
        if file.is_file() and file.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            image_files.append(file)

    return sorted(image_files)


def open_image(image_path: Path) -> Optional[Image.Image]:
    """Safely open an image file and convert to RGB.

    Args:
        image_path: Path to the image file

    Returns:
        PIL Image in RGB mode, or None if unable to open

    Raises:
        Exception: Re-raises any exception that occurs during opening
    """
    try:
        img = Image.open(image_path)
        # Convert to RGB to ensure compatibility
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    except Exception:
        raise


def resize_image_aspect_ratio(image: Image.Image, target_size: int) -> Image.Image:
    """Resize image to fit within target_size maintaining aspect ratio.

    Args:
        image: PIL Image object
        target_size: Target size in pixels (longest side)

    Returns:
        Resized PIL Image object
    """
    aspect_ratio = image.width / image.height

    if aspect_ratio > 1:
        # Width is larger
        new_width = target_size
        new_height = int(target_size / aspect_ratio)
    else:
        # Height is larger or equal
        new_height = target_size
        new_width = int(target_size * aspect_ratio)

    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


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


def save_image_optimized(image: Image.Image, output_path: Path) -> None:
    """Save image as optimized JPEG.

    Args:
        image: PIL Image object to save
        output_path: Path where to save the image
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, "JPEG", quality=85, optimize=True)
