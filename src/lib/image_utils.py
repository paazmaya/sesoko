"""Shared image handling and atomic file utilities for caption and cropping scripts."""

import importlib
import math
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image, ImageFile, ImageOps

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

ImageFile.LOAD_TRUNCATED_IMAGES = False


def atomic_write(path: Path, content: bytes) -> None:
    """Atomically write bytes to ``path`` using a temporary file in the same directory.

    Args:
        path: Destination file path.
        content: Raw bytes to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(content)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def get_image_files(folder_path: Path, recursive: bool = True) -> list[Path]:
    """Get all supported image files from the given folder.

    Args:
        folder_path: Path to the folder containing images.
        recursive: If True, search recursively; if False, only search top level.

    Returns:
        Sorted list of paths to supported image files.
    """
    image_files: list[Path] = []
    iterator = folder_path.rglob("*") if recursive else folder_path.iterdir()

    for file in iterator:
        if file.is_file() and file.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            image_files.append(file)

    return sorted(image_files)


def _to_rgb(image: Image.Image, background: str = "white") -> Image.Image:
    """Convert any PIL image to RGB, compositing transparency onto ``background``."""
    if "A" in image.getbands() or "transparency" in image.info:
        rgba = image.convert("RGBA")
        canvas = Image.new("RGBA", rgba.size, background)
        return Image.alpha_composite(canvas, rgba).convert("RGB")
    return image.convert("RGB")


def open_image(image_path: Path) -> Image.Image:
    """Open, validate, EXIF-transpose, and convert an image file to RGB.

    Args:
        image_path: Path to the image file.

    Returns:
        Decoded PIL Image in RGB mode with transparency composited on white.

    Raises:
        OSError: If the file cannot be opened or is truncated/corrupted.
    """
    with Image.open(image_path) as source:
        source.load()
        image = ImageOps.exif_transpose(source)
        return _to_rgb(image)


def resize_image_aspect_ratio(
    image: Image.Image, target_size: int, *, allow_upscale: bool = True
) -> Image.Image:
    """Resize image so its longest side matches ``target_size`` while preserving aspect ratio.

    Args:
        image: PIL Image object.
        target_size: Target size in pixels for the longest edge.
        allow_upscale: When False, return the original image if its longest edge is
            already at or below ``target_size``.

    Returns:
        Resized PIL Image object.

    Raises:
        ValueError: If ``target_size`` is not positive.
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
    """Fit a subject into an exact-size crop without clipping or stretching it.

    Args:
        image: Source PIL Image.
        output_size: Target ``(width, height)`` in pixels.
        subject_box: Optional ``(left, top, right, bottom)`` bounding box in image
            coordinates. When ``None``, performs a center-aligned aspect-fill crop.
        margin: Fractional extra context margin around ``subject_box`` (e.g., ``0.2``
            adds 20% width and height when space permits before padding).
        background: Fill color used only when padding is required to preserve aspect ratio.

    Returns:
        Exact-size PIL Image of dimensions ``output_size``.

    Raises:
        ValueError: If ``output_size``, ``margin``, or ``subject_box`` is invalid.
    """
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
    """Save an image in JPEG, PNG, or WebP format with optimization enabled.

    Non-RGB images (such as RGBA or palette images) are converted to RGB with
    white alpha compositing when saving as JPEG.

    Args:
        image: PIL Image object to save.
        output_path: Destination file path.
        output_format: Target image format (``"jpeg"``, ``"png"``, or ``"webp"``).
        quality: Compression quality in ``[1, 100]`` for JPEG and WebP outputs.

    Raises:
        ValueError: If ``output_format`` or ``quality`` is out of range.
    """
    if output_format not in {"jpeg", "png", "webp"} or not 1 <= quality <= 100:
        raise ValueError("Expected jpeg/png/webp format and quality in [1, 100]")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "jpeg" and image.mode != "RGB":
        image = _to_rgb(image)
    options: dict[str, Any] = {"optimize": True} if output_format == "png" else {"quality": quality}
    if output_format == "jpeg":
        options["optimize"] = True
    image.save(output_path, output_format.upper(), **options)
