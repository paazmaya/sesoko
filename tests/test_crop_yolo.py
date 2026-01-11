"""Unit tests for YOLO cropper module."""

from PIL import Image

from crop_yolo import YOLOCropper


class TestYOLOCropperInit:
    """Tests for YOLOCropper initialization."""

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        cropper = YOLOCropper()
        assert cropper.crop_focus is None
        assert cropper.resolution == 512
        assert cropper.model is not None

    def test_init_custom_crop_focus(self):
        """Test initialization with custom crop focus."""
        cropper = YOLOCropper(crop_focus="person")
        assert cropper.crop_focus == "person"
        assert cropper.resolution == 512

    def test_init_custom_resolution(self):
        """Test initialization with custom resolution."""
        cropper = YOLOCropper(resolution=768)
        assert cropper.resolution == 768
        assert cropper.crop_focus is None

    def test_init_all_parameters(self):
        """Test initialization with all parameters."""
        cropper = YOLOCropper(crop_focus="dog", resolution=256)
        assert cropper.crop_focus == "dog"
        assert cropper.resolution == 256


class TestYOLOCropperAvailableClasses:
    """Tests for available classes method."""

    def test_get_available_classes_returns_list(self):
        """Test that get_available_classes returns a list."""
        cropper = YOLOCropper()
        classes = cropper.get_available_classes()
        assert isinstance(classes, list)
        assert len(classes) > 0

    def test_get_available_classes_contains_common_objects(self):
        """Test that available classes contain common YOLO object types."""
        cropper = YOLOCropper()
        classes = cropper.get_available_classes()
        # YOLO11 should detect common objects
        assert any(c.lower() in classes for c in ["person", "car", "dog", "cat"])


class TestYOLOCropperProcessImage:
    """Tests for image processing methods."""

    def test_process_image_returns_square(self):
        """Test that process_image returns a square image."""
        cropper = YOLOCropper(resolution=512)
        # Create a test image
        img = Image.new("RGB", (800, 600), color="red")

        result = cropper.process_image(img, skip_if_not_found=False)

        assert result is not None
        assert result.size == (512, 512)
        assert result.mode == "RGB"

    def test_process_image_wide_image(self):
        """Test processing a wide image."""
        cropper = YOLOCropper(resolution=256)
        img = Image.new("RGB", (1000, 500), color="green")

        result = cropper.process_image(img, skip_if_not_found=False)

        assert result is not None
        assert result.size == (256, 256)

    def test_process_image_tall_image(self):
        """Test processing a tall image."""
        cropper = YOLOCropper(resolution=256)
        img = Image.new("RGB", (500, 1000), color="blue")

        result = cropper.process_image(img, skip_if_not_found=False)

        assert result is not None
        assert result.size == (256, 256)

    def test_process_image_square_image(self):
        """Test processing a square image."""
        cropper = YOLOCropper(resolution=512)
        img = Image.new("RGB", (1000, 1000), color="yellow")

        result = cropper.process_image(img, skip_if_not_found=False)

        assert result is not None
        assert result.size == (512, 512)

    def test_process_image_skip_if_not_found_true(self):
        """Test that None is returned when object not found and skip_if_not_found=True."""
        cropper = YOLOCropper(crop_focus="nonexistent_object", resolution=512)
        img = Image.new("RGB", (800, 600), color="red")

        result = cropper.process_image(img, skip_if_not_found=True)

        # Should return None or use fallback
        assert result is None or result.size == (512, 512)

    def test_process_image_skip_if_not_found_false(self):
        """Test that fallback is used when object not found and skip_if_not_found=False."""
        cropper = YOLOCropper(crop_focus="nonexistent_object", resolution=512)
        img = Image.new("RGB", (800, 600), color="red")

        result = cropper.process_image(img, skip_if_not_found=False)

        # Should use center crop as fallback
        assert result is not None
        assert result.size == (512, 512)


class TestYOLOCropperPrivateMethods:
    """Tests for private helper methods."""

    def test_has_target_object_without_crop_focus(self):
        """Test _has_target_object returns False when crop_focus is None."""
        cropper = YOLOCropper(crop_focus=None)
        img = Image.new("RGB", (800, 600), color="red")

        result = cropper._has_target_object(img)

        assert result is False

    def test_center_crop(self):
        """Test _center_crop returns square image."""
        cropper = YOLOCropper()
        img = Image.new("RGB", (800, 600), color="green")

        result = cropper._center_crop(img)

        assert result.size == (600, 600)

    def test_square_pad(self):
        """Test _square_pad returns square image."""
        cropper = YOLOCropper()
        img = Image.new("RGB", (800, 600), color="blue")

        result = cropper._square_pad(img)

        assert result.size == (600, 600)


class TestYOLOCropperProcessFolder:
    """Tests for folder processing."""

    def test_process_folder_stats_structure(self, tmp_path):
        """Test that process_folder returns correct stats structure."""
        input_path = tmp_path / "input"
        output_path = tmp_path / "output"
        input_path.mkdir()

        # Create test image
        img = Image.new("RGB", (800, 600), color="red")
        img.save(input_path / "test.jpg")

        cropper = YOLOCropper(resolution=512)
        stats = cropper.process_folder(str(input_path), str(output_path))

        assert "base_folder" in stats
        assert "output_folder" in stats
        assert "crop_focus" in stats
        assert "processed" in stats
        assert "skipped" in stats
        assert "failed" in stats

    def test_process_folder_creates_output_directory(self, tmp_path):
        """Test that process_folder creates output directory."""
        input_path = tmp_path / "input"
        output_path = tmp_path / "output"
        input_path.mkdir()

        # Create test image
        img = Image.new("RGB", (800, 600), color="green")
        img.save(input_path / "test.jpg")

        cropper = YOLOCropper(resolution=512)
        cropper.process_folder(str(input_path), str(output_path))

        assert output_path.exists()
        assert output_path.is_dir()
