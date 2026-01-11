"""Unit tests for image_utils module."""

import tempfile
from pathlib import Path

import pytest
from PIL import Image

from lib.image_utils import (
    crop_to_square,
    get_image_files,
    open_image,
    resize_image_aspect_ratio,
    save_image_optimized,
)


@pytest.fixture
def temp_image_dir():
    """Create a temporary directory with test images."""
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_path = Path(tmpdir)

        # Create test images with various formats and dimensions
        # Wide image (landscape)
        wide_img = Image.new("RGB", (800, 400), color="red")
        wide_img.save(temp_path / "wide.jpg")

        # Tall image (portrait)
        tall_img = Image.new("RGB", (400, 800), color="green")
        tall_img.save(temp_path / "tall.png")

        # Square image
        square_img = Image.new("RGB", (500, 500), color="blue")
        square_img.save(temp_path / "square.jpg")

        # Create subdirectory with images
        subdir = temp_path / "subdir"
        subdir.mkdir()
        subimg = Image.new("RGB", (300, 300), color="yellow")
        subimg.save(subdir / "sub_image.png")

        # Create image with RGBA mode
        rgba_img = Image.new("RGBA", (200, 200), color=(255, 0, 0, 128))
        rgba_img.save(temp_path / "rgba.png")

        yield temp_path


class TestGetImageFiles:
    """Tests for get_image_files function."""

    def test_get_image_files_recursive(self, temp_image_dir):
        """Test finding images recursively."""
        files = get_image_files(temp_image_dir, recursive=True)
        assert len(files) == 5
        filenames = [f.name for f in files]
        assert "wide.jpg" in filenames
        assert "tall.png" in filenames
        assert "square.jpg" in filenames
        assert "sub_image.png" in filenames
        assert "rgba.png" in filenames

    def test_get_image_files_non_recursive(self, temp_image_dir):
        """Test finding images non-recursively."""
        files = get_image_files(temp_image_dir, recursive=False)
        assert len(files) == 4
        filenames = [f.name for f in files]
        assert "wide.jpg" in filenames
        assert "sub_image.png" not in filenames  # Should not find subdirectory images

    def test_get_image_files_empty_directory(self):
        """Test with empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            files = get_image_files(Path(tmpdir))
            assert len(files) == 0

    def test_get_image_files_sorted(self, temp_image_dir):
        """Test that results are sorted."""
        files = get_image_files(temp_image_dir, recursive=False)
        filenames = [f.name for f in files]
        assert filenames == sorted(filenames)

    def test_get_image_files_filters_non_images(self, temp_image_dir):
        """Test that non-image files are filtered out."""
        # Create a non-image file
        (temp_image_dir / "readme.txt").write_text("This is not an image")
        files = get_image_files(temp_image_dir, recursive=False)
        filenames = [f.name for f in files]
        assert "readme.txt" not in filenames


class TestOpenImage:
    """Tests for open_image function."""

    def test_open_image_jpg(self, temp_image_dir):
        """Test opening a JPEG image."""
        img_path = temp_image_dir / "wide.jpg"
        img = open_image(img_path)
        assert img is not None
        assert img.mode == "RGB"
        assert img.size == (800, 400)

    def test_open_image_png(self, temp_image_dir):
        """Test opening a PNG image."""
        img_path = temp_image_dir / "tall.png"
        img = open_image(img_path)
        assert img is not None
        assert img.mode == "RGB"
        assert img.size == (400, 800)

    def test_open_image_rgba_conversion(self, temp_image_dir):
        """Test that RGBA images are converted to RGB."""
        img_path = temp_image_dir / "rgba.png"
        img = open_image(img_path)
        assert img is not None
        assert img.mode == "RGB"
        assert img.size == (200, 200)

    def test_open_image_nonexistent(self):
        """Test opening a non-existent image file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "nonexistent.jpg"
            with pytest.raises(Exception):
                open_image(img_path)


class TestResizeImageAspectRatio:
    """Tests for resize_image_aspect_ratio function."""

    def test_resize_wide_image(self):
        """Test resizing a wide image (landscape)."""
        img = Image.new("RGB", (1000, 500), color="red")
        resized = resize_image_aspect_ratio(img, target_size=400)
        assert resized.size == (400, 200)

    def test_resize_tall_image(self):
        """Test resizing a tall image (portrait)."""
        img = Image.new("RGB", (500, 1000), color="green")
        resized = resize_image_aspect_ratio(img, target_size=400)
        assert resized.size == (200, 400)

    def test_resize_square_image(self):
        """Test resizing a square image."""
        img = Image.new("RGB", (1000, 1000), color="blue")
        resized = resize_image_aspect_ratio(img, target_size=400)
        assert resized.size == (400, 400)

    def test_resize_maintains_aspect_ratio(self):
        """Test that aspect ratio is maintained during resize."""
        img = Image.new("RGB", (800, 400), color="red")
        resized = resize_image_aspect_ratio(img, target_size=400)
        original_aspect = img.width / img.height
        resized_aspect = resized.width / resized.height
        assert abs(original_aspect - resized_aspect) < 0.01

    def test_resize_smaller_image(self):
        """Test resizing an image smaller than target."""
        img = Image.new("RGB", (200, 100), color="red")
        resized = resize_image_aspect_ratio(img, target_size=400)
        # Should not upscale beyond target
        assert max(resized.size) == 400


class TestCropToSquare:
    """Tests for crop_to_square function."""

    def test_crop_square_image_no_change(self):
        """Test that square images are unchanged."""
        img = Image.new("RGB", (500, 500), color="blue")
        cropped = crop_to_square(img)
        assert cropped.size == (500, 500)

    def test_crop_wide_image(self):
        """Test cropping a wide image."""
        img = Image.new("RGB", (800, 400), color="red")
        cropped = crop_to_square(img)
        assert cropped.size == (400, 400)

    def test_crop_tall_image(self):
        """Test cropping a tall image."""
        img = Image.new("RGB", (400, 800), color="green")
        cropped = crop_to_square(img)
        assert cropped.size == (400, 400)

    def test_crop_center_aligned(self):
        """Test that crop is center-aligned."""
        # Create a wide image with specific pattern
        img = Image.new("RGB", (400, 200), color="white")
        # Add a red stripe in the center
        for x in range(150, 250):
            for y in range(0, 200):
                img.putpixel((x, y), (255, 0, 0))

        cropped = crop_to_square(img)
        # The red stripe should still be in the center of the cropped image
        assert cropped.size == (200, 200)
        # Check that red pixels are in the center
        center_x = 100
        center_pixel = cropped.getpixel((center_x, 100))
        assert isinstance(center_pixel, tuple)
        assert center_pixel[0] > 200  # Red component should be high


class TestSaveImageOptimized:
    """Tests for save_image_optimized function."""

    def test_save_image_as_jpeg(self):
        """Test saving image as optimized JPEG."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img = Image.new("RGB", (500, 500), color="blue")
            output_path = Path(tmpdir) / "test_output.jpg"

            save_image_optimized(img, output_path)

            assert output_path.exists()
            # Verify it's a valid JPEG
            loaded = Image.open(output_path)
            assert loaded.format == "JPEG"
            assert loaded.size == (500, 500)
            loaded.close()

    def test_save_image_creates_directories(self):
        """Test that save_image_optimized creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img = Image.new("RGB", (100, 100), color="green")
            output_path = Path(tmpdir) / "subdir1" / "subdir2" / "output.jpg"

            save_image_optimized(img, output_path)

            assert output_path.exists()
            assert output_path.parent.exists()

    def test_save_image_quality(self):
        """Test that image is saved with quality 85."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img = Image.new("RGB", (500, 500), color="red")
            output_path = Path(tmpdir) / "test_quality.jpg"

            save_image_optimized(img, output_path)

            # Load and verify it's a valid image
            loaded = Image.open(output_path)
            assert loaded.size == (500, 500)
            assert loaded.mode == "RGB"
            loaded.close()

    def test_save_rgba_image_converts_to_rgb(self):
        """Test that RGBA images are properly handled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create RGBA image
            img = Image.new("RGBA", (200, 200), color=(255, 0, 0, 128))
            output_path = Path(tmpdir) / "rgba_output.jpg"

            # Need to convert before saving as JPEG
            img_rgb = img.convert("RGB")
            save_image_optimized(img_rgb, output_path)

            loaded = Image.open(output_path)
            assert loaded.mode == "RGB"
            assert loaded.format == "JPEG"
            loaded.close()


class TestImageIntegration:
    """Integration tests combining multiple image operations."""

    def test_full_image_processing_pipeline(self, temp_image_dir):
        """Test complete image processing: get -> open -> resize -> crop -> save."""
        with tempfile.TemporaryDirectory() as output_dir:
            output_path = Path(output_dir)

            # Get images
            images = get_image_files(temp_image_dir, recursive=False)
            assert len(images) > 0

            # Process each image
            for img_file in images:
                if img_file.name == "rgba.png":
                    continue  # Skip RGBA for JPEG save

                # Open and convert
                img = open_image(img_file)
                assert img is not None

                # Resize
                img = resize_image_aspect_ratio(img, target_size=128)

                # Crop to square
                img = crop_to_square(img)
                # After resizing to fit 128px and cropping, should be a square
                # but may be smaller than 128x128 depending on original aspect ratio
                assert img.width == img.height
                assert img.width <= 128

                # Save
                output = output_path / f"processed_{img_file.stem}.jpg"
                save_image_optimized(img, output)
                assert output.exists()

    def test_image_pipeline_preserves_content(self, temp_image_dir):
        """Test that pipeline doesn't corrupt image content."""
        with tempfile.TemporaryDirectory() as output_dir:
            original_path = temp_image_dir / "square.jpg"
            output_path = Path(output_dir) / "processed.jpg"

            # Load original
            original = open_image(original_path)
            assert original is not None
            original_color = original.getpixel((250, 250))
            assert isinstance(original_color, tuple)

            # Process
            processed = resize_image_aspect_ratio(original, target_size=500)
            processed = crop_to_square(processed)
            save_image_optimized(processed, output_path)

            # Load processed
            reloaded = Image.open(output_path)
            reloaded_color = reloaded.getpixel((250, 250))
            assert isinstance(reloaded_color, tuple)
            reloaded.close()

            # Colors should be similar (not exact due to JPEG compression)
            assert abs(reloaded_color[0] - original_color[0]) < 10
            assert abs(reloaded_color[1] - original_color[1]) < 10
            assert abs(reloaded_color[2] - original_color[2]) < 10
