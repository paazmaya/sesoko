"""Unit tests for YOLO cropper module."""

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Never

import pytest
from PIL import Image, ImageDraw

from crop_yolo import YOLOCropper
from lib.image_utils import prepare_crop
from lib.person_detection import Detection


class FakeDetector:
    def __init__(self, detections: Iterable[Detection] = ()) -> None:
        self.detections = list(detections)

    def detect(self, image: Image.Image) -> list[Detection]:
        return self.detections

    def get_available_classes(self) -> list[str]:
        return ["person", "car", "dog", "cat"]


@pytest.fixture(autouse=True)
def no_model_downloads(monkeypatch: pytest.MonkeyPatch) -> None:
    def factory(_model_path: str) -> FakeDetector:
        return FakeDetector()

    monkeypatch.setattr("crop_yolo.YOLODetector", factory)


class TestYOLOCropperInit:
    """Tests for YOLOCropper initialization."""

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        cropper = YOLOCropper()
        assert cropper.crop_focus is None
        assert cropper.resolution == 512
        assert cropper.model is None

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

        assert result is None

    def test_process_image_skip_if_not_found_false(self):
        """Test that fallback is used when object not found and skip_if_not_found=False."""
        cropper = YOLOCropper(crop_focus="nonexistent_object", resolution=512)
        img = Image.new("RGB", (800, 600), color="red")

        result = cropper.process_image(img, skip_if_not_found=False)

        # Should use center crop as fallback
        assert result is not None
        assert result.size == (512, 512)


class TestYOLOCropperCenterCrops:
    """Test center-crop behavior through the public processing API."""

    def test_without_crop_focus(self):
        """No focus retains center cropping without a target requirement."""
        cropper = YOLOCropper(crop_focus=None)
        img = Image.new("RGB", (800, 600), color="red")

        result = cropper.process_image(img)
        assert result is not None
        assert result.size == (512, 512)

    def test_center_crop(self):
        """Center cropping preserves a square at the requested resolution."""
        cropper = YOLOCropper(resolution=600)
        img = Image.new("RGB", (800, 600), color="green")

        result = cropper.process_image(img)
        assert result is not None
        assert result.size == (600, 600)

    def test_portrait_center_crop(self):
        """Portrait images also produce the requested square."""
        cropper = YOLOCropper(resolution=600)
        img = Image.new("RGB", (600, 800), color="blue")

        result = cropper.process_image(img)
        assert result is not None
        assert result.size == (600, 600)


class TestYOLOCropperProcessFolder:
    """Tests for folder processing."""

    def test_process_folder_stats_structure(self, tmp_path: Path):
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

    def test_process_folder_creates_output_directory(self, tmp_path: Path):
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

    def test_same_stem_and_existing_outputs_are_not_overwritten(self, tmp_path: Path):
        input_path = tmp_path / "input"
        input_path.mkdir()
        for name in ("same.jpg", "same.png", "same.jpg.png"):
            Image.new("RGB", (10, 10), "red").save(input_path / name)
        cropper = YOLOCropper(resolution=16)
        output_path = tmp_path / "output"
        first = cropper.process_folder(input_path, output_path)
        second = cropper.process_folder(input_path, output_path)
        assert not first["failed"]
        assert not second["failed"]
        assert len(list(output_path.iterdir())) == 6

    def test_original_directory_rejected(self, tmp_path: Path):
        cropper = YOLOCropper()
        with pytest.raises(ValueError, match="must differ"):
            cropper.process_folder(tmp_path, tmp_path)


class TestPersonSelection:
    def test_largest_confident_person_selected_not_first_box(self):
        detector = FakeDetector(
            [
                Detection((0, 0, 10, 10), 0.99),
                Detection((20, 10, 80, 90), 0.8),
                Detection((0, 0, 100, 100), 0.1),
                Detection((0, 0, 100, 100), 0.99, "dog"),
                Detection((0, 0, float("nan"), 100), 0.99),
            ]
        )
        cropper = YOLOCropper(crop_focus="person", detector=detector)
        image = Image.new("RGB", (100, 100), "blue")
        ImageDraw.Draw(image).rectangle((20, 10, 79, 89), fill="red")
        result = cropper.process_image(image)
        assert result is not None
        expected = prepare_crop(image, (512, 512), (20, 10, 80, 90))
        assert result.tobytes() == expected.tobytes()

    def test_person_crop_uses_requested_rectangular_size(self):
        cropper = YOLOCropper(
            crop_focus="person",
            detector=FakeDetector([Detection((0, 0, 20, 80), 0.9)]),
            width=160,
            height=80,
            margin=0,
        )
        result = cropper.process_image(Image.new("RGB", (20, 80), "red"))
        assert result is not None
        assert result.size == (160, 80)
        assert result.getpixel((0, 40)) == (255, 255, 255)

    def test_center_crop_does_not_load_detector(self, monkeypatch: pytest.MonkeyPatch):
        def fail(*_args: object) -> Never:
            raise AssertionError("Detector must not be loaded")

        monkeypatch.setattr("crop_yolo.YOLODetector", fail)
        result = YOLOCropper().process_image(Image.new("RGB", (20, 80)))
        assert result is not None
        assert result.size == (512, 512)

    def test_deim_requires_explicit_model(self):
        cropper = YOLOCropper(crop_focus="person", backend="deim")
        image = Image.new("RGB", (10, 10))
        with pytest.raises(ValueError, match="requires --model"):
            cropper.process_image(image)


class TestDEIMDetector:
    def test_preprocessing_and_original_coordinate_mapping(self):
        from types import SimpleNamespace

        np = pytest.importorskip("numpy")
        from lib.person_detection import DEIMDetector

        class Session:
            def get_inputs(self) -> list[SimpleNamespace]:
                return [
                    SimpleNamespace(name="images", shape=["N", 3, 100, 100], type="tensor(float)"),
                    SimpleNamespace(name="orig_target_sizes", type="tensor(int64)"),
                ]

            def get_outputs(self) -> list[SimpleNamespace]:
                return [SimpleNamespace(name=name) for name in ("scores", "boxes", "labels")]

            def run(self, names: list[str], inputs: dict[str, Any]) -> tuple[Any, Any, Any]:
                assert names == ["labels", "boxes", "scores"]
                pixels = inputs["images"]
                assert pixels.shape == (1, 3, 100, 100)
                assert pixels.dtype == np.float32
                assert pixels.flags.c_contiguous
                assert pixels[0, 0, 50, 50] == 1
                assert pixels[0, 1, 50, 50] == 0
                assert pixels[0, 0, 0, 50] == 0
                assert inputs["orig_target_sizes"].dtype == np.int64
                assert inputs["orig_target_sizes"].tolist() == [[100, 100]]
                return (
                    np.array([[0, 1, 0, 0]]),
                    np.array(
                        [
                            [
                                [10, 30, 90, 70],
                                [0, 25, 100, 75],
                                [0, 0, 0, 0],
                                [0, 0, float("nan"), 10],
                            ]
                        ]
                    ),
                    np.array([[0.9, 0.99, 0.8, 0.7]]),
                )

        detector = DEIMDetector("unused.onnx", session=Session())
        detections = detector.detect(Image.new("RGB", (200, 100), "red"))
        assert len(detections) == 1
        assert detections[0].box == (20, 10, 180, 90)
        assert detections[0].label == "person"

    def test_model_contract_rejected(self):
        from types import SimpleNamespace

        pytest.importorskip("numpy")
        from lib.person_detection import DEIMDetector

        session = SimpleNamespace(get_inputs=lambda: [SimpleNamespace(name="wrong")])
        with pytest.raises(ValueError, match="deployment inputs"):
            DEIMDetector("unused.onnx", session=session)

    def test_missing_model_is_not_downloaded(self, tmp_path: Path):
        pytest.importorskip("numpy")
        from lib.person_detection import DEIMDetector

        with pytest.raises(FileNotFoundError, match="not found"):
            DEIMDetector(tmp_path / "missing.onnx")

    def test_real_onnx_runtime_contract(self, tmp_path: Path):
        np = pytest.importorskip("numpy")
        onnx = pytest.importorskip("onnx")
        pytest.importorskip("onnxruntime")
        tensor_proto = onnx.TensorProto
        helper = onnx.helper
        numpy_helper = onnx.numpy_helper

        from lib.person_detection import DEIMDetector

        inputs = [
            helper.make_tensor_value_info("images", tensor_proto.FLOAT, [1, 3, 100, 100]),
            helper.make_tensor_value_info("orig_target_sizes", tensor_proto.INT64, [1, 2]),
        ]
        values = {
            "labels": np.array([[0]], dtype=np.int64),
            "boxes": np.array([[[10, 30, 90, 70]]], dtype=np.float32),
            "scores": np.array([[0.9]], dtype=np.float32),
        }
        nodes = [
            helper.make_node("Constant", [], [name], value=numpy_helper.from_array(value))
            for name, value in values.items()
        ]
        outputs = [
            helper.make_tensor_value_info(
                name,
                tensor_proto.INT64 if name == "labels" else tensor_proto.FLOAT,
                list(value.shape),
            )
            for name, value in values.items()
        ]
        graph = helper.make_graph(nodes, "detector-contract", inputs, outputs)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
        path = tmp_path / "contract.onnx"
        onnx.save(model, path)
        result = DEIMDetector(path).detect(Image.new("RGB", (200, 100), "red"))
        assert len(result) == 1
        assert result[0].box == (20, 10, 180, 90)


def test_crop_cli_rectangular_png_without_ml(tmp_path: Path):
    from click.testing import CliRunner

    from crop_yolo import main

    input_path = tmp_path / "input"
    input_path.mkdir()
    Image.new("RGB", (80, 100), "red").save(input_path / "photo.jpg")
    output_path = tmp_path / "output"
    result = CliRunner().invoke(
        main,
        [
            "--input-dir",
            str(input_path),
            "--output-dir",
            str(output_path),
            "--width",
            "64",
            "--height",
            "96",
            "--format",
            "png",
        ],
    )
    assert result.exit_code == 0, result.output
    with Image.open(output_path / "photo.png") as image:
        assert image.size == (64, 96)
        assert image.format == "PNG"
