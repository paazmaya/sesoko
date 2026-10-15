"""Caption workflow tests without loading ML runtimes or weights."""

import json
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace
from typing import Never, Self

import pytest
from PIL import Image

import caption_images


def test_decode_only_generated_tokens_and_preserve_assistant_word():
    def decode(outputs: list[list[int]], skip_special_tokens: bool) -> list[str]:
        assert outputs == [[4, 5]]
        assert skip_special_tokens
        return [" A practitioner and an assistant hold a staff. "]

    processor = SimpleNamespace(batch_decode=decode)
    result = caption_images.decode_completion(processor, [[1, 2, 3, 4, 5]], 3)
    assert result == "A practitioner and an assistant hold a staff."


def test_pixel_budget_handles_processor_area_fields():
    processor = SimpleNamespace(
        image_processor=SimpleNamespace(
            min_pixels=65536,
            max_pixels=16777216,
            size={"shortest_edge": 65536, "longest_edge": 16777216},
        )
    )
    caption_images.configure_pixel_budget(processor, 512 * 512)
    assert processor.image_processor.max_pixels == 512 * 512
    assert processor.image_processor.size["longest_edge"] == 512 * 512
    assert processor.image_processor.size["shortest_edge"] == 65536


def test_atomic_write_preserves_previous_file_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "captions.toml"
    path.write_bytes(b"original")

    def fail(*_args: object) -> Never:
        raise PermissionError("blocked")

    monkeypatch.setattr("pathlib.Path.replace", fail)
    with pytest.raises(PermissionError):
        caption_images.atomic_write(path, b"replacement")
    assert path.read_bytes() == b"original"
    assert list(tmp_path.iterdir()) == [path]


def test_caption_resume_and_settings_invalidation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("tomli_w")
    folder = tmp_path / "images"
    folder.mkdir()
    Image.new("RGB", (100, 50), "red").save(folder / "photo.png")
    output = tmp_path / "nested" / "captions.toml"
    calls: list[tuple[tuple[int, int], int]] = []
    loads: list[dict[str, object]] = []

    def load(*_args: object, **kwargs: object) -> tuple[object, object, str]:
        loads.append(kwargs)
        return object(), object(), "cpu"

    def generate(
        _model: object,
        _processor: object,
        image: Image.Image,
        _device: str,
        _model_key: str,
        max_new_tokens: int,
    ) -> str:
        calls.append((image.size, max_new_tokens))
        return "A practitioner in a red gi."

    monkeypatch.setattr(caption_images, "load_model_and_processor", load)
    monkeypatch.setattr(caption_images, "generate_caption", generate)
    argv = ["caption_images.py", str(folder), "-o", str(output), "--backend", "transformers"]
    monkeypatch.setattr(sys, "argv", argv)
    caption_images.main()
    assert calls == [((100, 50), 64)]
    assert len(loads) == 1
    data = tomllib.loads(output.read_text())
    assert data[str(folder)]["photo.png"] == "A practitioner in a red gi."
    sidecar = folder / "photo.png.txt"
    sidecar.unlink()
    caption_images.main()
    assert len(calls) == 1
    assert len(loads) == 1
    assert sidecar.read_text() == "A practitioner in a red gi."
    monkeypatch.setattr(sys, "argv", argv + ["--image-size", "512"])
    caption_images.main()
    assert len(calls) == 2
    Image.new("RGB", (120, 60), "blue").save(folder / "photo.png")
    caption_images.main()
    assert len(calls) == 3
    assert json.loads(output.with_suffix(".toml.meta.json").read_text())


def test_new_qwen_model_keys_and_local_path(tmp_path: Path):
    assert "qwen35-0.8b" in caption_images.MODEL_CHOICES
    assert "qwen35-2b" in caption_images.MODEL_CHOICES
    assert (
        caption_images.resolve_model_path("qwen35-0.8b", "transformers", str(tmp_path)) == tmp_path
    )


def test_missing_model_not_downloaded(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Local model not found"):
        caption_images.load_model_and_processor(model_path=str(tmp_path / "missing"))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_explicit_device_overrides_automatic_mlx_backend(device: str):
    assert caption_images.resolve_backend("auto", "qwen3vl", device) == "transformers"


@pytest.mark.parametrize(
    "cuda,mps,expected", [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")]
)
def test_device_selection(cuda: bool, mps: bool, expected: str):
    torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
    )
    assert caption_images.select_torch_device(torch, "auto") == expected


def test_quantization_rejected_off_cuda():
    with pytest.raises(ValueError, match="CUDA"):
        caption_images.quantization_options("qwen3vl", "mps", "int4", None)


def test_transformers_loader_is_offline_and_uses_cpu_float32(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    (tmp_path / "config.json").write_text('{"model_type": "qwen3_5"}')
    calls: dict[str, str | bool] = {}
    model_options: dict[str, object] = {}
    model = SimpleNamespace()

    def move_model(device: str) -> SimpleNamespace:
        calls["device"] = device
        return model

    model.to = move_model
    model.eval = lambda: calls.update(eval=True)
    processor = SimpleNamespace()

    def from_pretrained(path: str, **options: object) -> SimpleNamespace:
        model_options.update(options)
        assert path == str(tmp_path)
        return model

    def load_processor(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return processor

    runtime = SimpleNamespace(
        AutoModelForImageTextToText=SimpleNamespace(from_pretrained=from_pretrained),
        AutoProcessor=SimpleNamespace(from_pretrained=load_processor),
    )
    torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
        float32="float32",
    )

    def load_module(name: str, _extra: str) -> SimpleNamespace:
        return torch if name == "torch" else runtime

    monkeypatch.setattr(caption_images, "load_optional_module", load_module)
    result = caption_images.load_model_and_processor(
        "qwen35-0.8b",
        backend="transformers",
        model_path=str(tmp_path),
    )
    assert result == (model, processor, "cpu")
    assert calls["device"] == "cpu"
    assert calls["eval"] is True
    assert model_options["dtype"] == "float32"
    assert model_options["local_files_only"] is True
    assert model_options["trust_remote_code"] is False
    assert "device_map" not in model_options


def test_qwen_inference_preserves_integer_ids_and_casts_pixels(monkeypatch: pytest.MonkeyPatch):
    from contextlib import nullcontext

    moves: dict[str, tuple[tuple[object, ...], dict[str, object]]] = {}

    class Tensor:
        shape = (1, 3)

        def __init__(self, name: str, floating: bool = False) -> None:
            self.name = name
            self.floating = floating

        def is_floating_point(self) -> bool:
            return self.floating

        def to(self, *args: object, **kwargs: object) -> Self:
            moves[self.name] = (args, kwargs)
            return self

    ids = Tensor("ids")
    pixels = Tensor("pixels", floating=True)

    class Processor:
        def apply_chat_template(self, _messages: object, **kwargs: object) -> str:
            assert kwargs["enable_thinking"] is False
            return "prompt"

        def __call__(self, **_kwargs: object) -> dict[str, Tensor]:
            return {"input_ids": ids, "pixel_values": pixels, "token_type_ids": Tensor("types")}

        def batch_decode(self, outputs: list[list[int]], **_kwargs: object) -> list[str]:
            assert outputs == [[4, 5]]
            return ["A practitioner and an assistant hold a staff."]

    def generate(**kwargs: object) -> list[list[int]]:
        assert kwargs["input_ids"] is ids
        assert "token_type_ids" not in kwargs
        assert kwargs["max_new_tokens"] == 64
        assert kwargs["do_sample"] is False
        return [[1, 2, 3, 4, 5]]

    def load_module(*_args: object) -> SimpleNamespace:
        return SimpleNamespace(inference_mode=nullcontext)

    monkeypatch.setattr(caption_images, "load_optional_module", load_module)
    result = caption_images.generate_caption(
        SimpleNamespace(dtype="float16", generate=generate),
        Processor(),
        Image.new("RGB", (10, 20)),
        "cuda",
    )
    assert result == "A practitioner and an assistant hold a staff."
    assert moves["ids"] == (("cuda",), {})
    assert moves["pixels"] == ((), {"device": "cuda", "dtype": "float16"})
