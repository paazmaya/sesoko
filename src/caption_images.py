#!/usr/bin/env python3
"""Local martial-arts image captions with optional Transformers or MLX backends."""

import hashlib
import importlib.util
import json
import platform
import sys
import time
import tomllib
from pathlib import Path
from typing import Any, cast

import click
from PIL import Image

from lib.image_utils import (
    atomic_write,
    get_image_files,
    open_image,
    resize_image_aspect_ratio,
)

try:
    import tomli_w
except ImportError:
    tomli_w = None

# Paths to locally stored models, relative to this file's location
_MODELS_DIR = Path(__file__).parent.parent / "models"

MODEL_CHOICES: dict[str, Path] = {
    # https://huggingface.co/Amirhossein75/Image-Captioning-Blip
    "blip": _MODELS_DIR / "Image-Captioning-Blip",
    # https://huggingface.co/internlm/CapRL-Qwen3VL-2B
    "caprl": _MODELS_DIR / "CapRL-Qwen3VL-2B",
    # https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct
    "qwen3vl": _MODELS_DIR / "Qwen3-VL-2B-Instruct",
    "qwen35-0.8b": _MODELS_DIR / "Qwen3.5-0.8B",
    "qwen35-2b": _MODELS_DIR / "Qwen3.5-2B",
}

MODEL_FINGERPRINT_SUFFIXES = {
    ".bin",
    ".jinja",
    ".json",
    ".model",
    ".safetensors",
    ".tiktoken",
    ".txt",
}

DEFAULT_MODEL = "qwen3vl"
CAPTION_PROMPT = (
    "Describe the visible people, martial-arts clothing and belt colors, weapons, and body "
    "positions in one plain-text sentence under 255 characters. Use precise martial-arts "
    "terms only when clearly identifiable. Do not infer rank, a kata name, or unseen details."
)


def load_optional_module(name: str, extra: str) -> Any:
    """Import an optional dependency module or raise an actionable installation error."""
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise RuntimeError(f"{name} requires: uv sync --extra {extra}") from exc


def resolve_backend(backend: str, model_key: str, device: str = "auto") -> str:
    """Resolve ``'auto'`` backend to ``'mlx'`` on Apple Silicon when available, else ``'transformers'``."""
    if backend != "auto":
        return backend
    if device in {"cpu", "cuda"}:
        return "transformers"
    if (
        model_key != "blip"
        and sys.platform == "darwin"
        and platform.machine() == "arm64"
        and importlib.util.find_spec("mlx_vlm") is not None
    ):
        return "mlx"
    return "transformers"


def resolve_model_path(model_key: str, backend: str, model_path: str | None) -> Path:
    """Resolve the local directory for ``model_key`` and ``backend`` without downloading."""
    if model_path:
        return Path(model_path).expanduser().resolve()
    mlx_path = _MODELS_DIR / "Qwen3-VL-2B-Instruct-4bit"
    if backend == "mlx" and model_key == "qwen3vl" and mlx_path.is_dir():
        return mlx_path
    return MODEL_CHOICES[model_key]


def configure_pixel_budget(processor: Any, max_pixels: int) -> None:
    """Constrain a vision processor's maximum pixel budget in place when supported."""
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        return
    if hasattr(image_processor, "max_pixels"):
        image_processor.max_pixels = max_pixels
        image_processor.min_pixels = min(getattr(image_processor, "min_pixels", 65536), max_pixels)
    size = getattr(image_processor, "size", None)
    if isinstance(size, dict) and "longest_edge" in size and "shortest_edge" in size:
        size = cast(dict[str, Any], size)
        image_processor.size = {
            **size,
            "longest_edge": max_pixels,
            "shortest_edge": min(size["shortest_edge"], max_pixels),
        }


def load_model_and_processor(
    model_key: str = DEFAULT_MODEL,
    quantize: str = "none",
    *,
    backend: str = "auto",
    model_path: str | None = None,
    device: str = "auto",
    max_pixels: int = 640 * 640,
) -> tuple[Any, Any, str]:
    """Load local model weights and processor; optional runtimes are imported on demand.

    Args:
        model_key: Model identifier from ``MODEL_CHOICES``.
        quantize: Optional CUDA bitsandbytes quantization mode (``'none'``, ``'int8'``, ``'int4'``).
        backend: Execution backend (``'auto'``, ``'transformers'``, or ``'mlx'``).
        model_path: Optional explicit path to a local model checkpoint directory.
        device: Target compute device (``'auto'``, ``'cpu'``, ``'cuda'``, or ``'mps'``).
        max_pixels: Maximum pixel budget configured on dynamic-resolution image processors.

    Returns:
        Tuple of ``(model, processor, effective_device)``.
    """
    backend = resolve_backend(backend, model_key, device)
    path = resolve_model_path(model_key, backend, model_path)
    if not path.is_dir():
        raise FileNotFoundError(f"Local model not found: {path}. Supply --model-path.")
    print(f"Loading '{model_key}' with {backend} from {path}...")
    if backend == "mlx":
        if sys.platform != "darwin" or platform.machine() != "arm64":
            raise ValueError("The MLX backend requires an Apple Silicon Mac")
        if model_key == "blip" or device not in {"auto", "mps"} or quantize != "none":
            raise ValueError("MLX uses its checkpoint precision; omit --quantize and --device")
        runtime = load_optional_module("mlx_vlm", "caption-mlx")
        model, processor = runtime.load(str(path), trust_remote_code=False)
        configure_pixel_budget(processor, max_pixels)
        print("Using MLX/Metal; precision is defined by the local checkpoint")
        return model, processor, "mlx"
    return _load_transformers(model_key, path, device, quantize, max_pixels)


def select_torch_device(torch: Any, device: str) -> str:
    """Select and validate the PyTorch execution device (``'cuda'``, ``'mps'``, or ``'cpu'``)."""
    if device == "auto":
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS requested but unavailable")
    return device


def torch_dtype(torch: Any, device: str) -> Any:
    """Return the preferred PyTorch floating-point dtype for ``device``."""
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float16 if device == "mps" else torch.float32


def quantization_options(model_key: str, device: str, quantize: str, dtype: Any) -> dict[str, Any]:
    """Build Transformers ``BitsAndBytesConfig`` kwargs when CUDA quantization is requested."""
    if quantize == "none":
        return {}
    if device != "cuda" or model_key == "blip":
        raise ValueError("--quantize requires a Qwen model on CUDA")
    if importlib.util.find_spec("bitsandbytes") is None:
        raise RuntimeError(
            "Quantization requires: uv sync --extra caption-transformers --extra cuda-quantization"
        )
    runtime = load_optional_module("transformers", "caption-transformers")

    return {
        "quantization_config": runtime.BitsAndBytesConfig(
            load_in_4bit=quantize == "int4",
            load_in_8bit=quantize == "int8",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ),
        "device_map": {"": device},
    }


def _load_transformers(
    model_key: str, path: Path, device: str, quantize: str, max_pixels: int
) -> tuple[Any, Any, str]:
    torch = load_optional_module("torch", "caption-transformers")
    runtime = load_optional_module("transformers", "caption-transformers")
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    if "quantization" in config:
        raise ValueError("MLX quantized weights require --backend mlx, not Transformers")
    device = select_torch_device(torch, device)
    dtype = torch_dtype(torch, device)
    options: dict[str, Any] = {"dtype": dtype, "local_files_only": True, "trust_remote_code": False}
    options.update(quantization_options(model_key, device, quantize, dtype))
    if model_key == "blip":
        model_class = runtime.BlipForConditionalGeneration
    else:
        model_class = runtime.AutoModelForImageTextToText
    model = model_class.from_pretrained(str(path), **options)
    if quantize == "none":
        model = model.to(device)
    model.eval()
    processor = runtime.AutoProcessor.from_pretrained(
        str(path), local_files_only=True, trust_remote_code=False
    )
    if model_key != "blip":
        configure_pixel_budget(processor, max_pixels)
    print(f"Using {device} | dtype={dtype} | quantize={quantize}")
    return model, processor, device


def generate_caption(
    model: Any,
    processor: Any,
    image: Image.Image,
    device: str,
    model_key: str = DEFAULT_MODEL,
    max_new_tokens: int = 64,
) -> str:
    """Generate a single-sentence caption for ``image``.

    Dispatches to the appropriate backend and model family.

    Args:
        model: Loaded MLX or Transformers vision-language model.
        processor: Matching image/text processor.
        image: RGB PIL Image object.
        device: Effective execution target (``"cuda"``, ``"mps"``, ``"cpu"``, or ``"mlx"``).
        model_key: Key from ``MODEL_CHOICES`` used to select prompt strategy.
        max_new_tokens: Maximum number of tokens to generate.

    Returns:
        Stripped caption string.
    """
    if device == "mlx":
        runtime = load_optional_module("mlx_vlm", "caption-mlx")
        templates = load_optional_module("mlx_vlm.prompt_utils", "caption-mlx")
        prompt = templates.apply_chat_template(
            processor, model.config, CAPTION_PROMPT, num_images=1, enable_thinking=False
        )
        result = runtime.generate(
            model,
            processor,
            prompt,
            image=[image],
            max_tokens=max_new_tokens,
            temperature=0.0,
            verbose=False,
        )
        return result.text.strip()
    if model_key == "blip":
        return _generate_caption_blip(model, processor, image, device, max_new_tokens)
    return _generate_caption_qwen3vl(model, processor, image, device, model_key, max_new_tokens)


def _generate_caption_blip(
    model: Any,
    processor: Any,
    image: Image.Image,
    device: str,
    max_new_tokens: int = 64,
) -> str:
    """Generate a concise, keyword-rich caption using BLIP."""
    torch = load_optional_module("torch", "caption-transformers")

    inputs = processor(images=image, return_tensors="pt").to(device, model.dtype)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, num_beams=1, do_sample=False
        )
    caption: str = processor.decode(outputs[0], skip_special_tokens=True)
    return caption.strip()


def _generate_caption_qwen3vl(
    model: Any,
    processor: Any,
    image: Image.Image,
    device: str,
    model_key: str = "qwen3vl",
    max_new_tokens: int = 64,
) -> str:
    """Generate only the completion using a short, visible-attributes prompt."""
    torch = load_optional_module("torch", "caption-transformers")

    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": CAPTION_PROMPT},
            ],
        }
    ]

    text: str = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs: dict[str, Any] = processor(  # type: ignore
        text=[text],
        images=[image],
        return_tensors="pt",
        padding=True,
    )
    inputs.pop("token_type_ids", None)
    inputs = {
        key: value.to(device=device, dtype=model.dtype)
        if value.is_floating_point()
        else value.to(device)
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    return decode_completion(processor, outputs, inputs["input_ids"].shape[1])


def decode_completion(processor: Any, outputs: Any, input_length: int) -> str:
    """Decode only the newly generated token slice beyond ``input_length``."""
    generated = [output[input_length:] for output in outputs]
    return processor.batch_decode(generated, skip_special_tokens=True)[0].strip()


def image_fingerprint(image_path: Path, settings: dict[str, Any]) -> str:
    """Compute a SHA-256 cache fingerprint from image stat metadata and active settings."""
    stat = image_path.stat()
    payload: dict[str, Any] = {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "settings": settings,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def model_fingerprint(path: Path) -> list[list[str | int]]:
    """Record filenames, byte sizes, and modification times for local model artifacts."""
    return (
        [
            [item.name, item.stat().st_size, item.stat().st_mtime_ns]
            for item in sorted(path.iterdir())
            if item.is_file() and item.suffix in MODEL_FINGERPRINT_SUFFIXES
        ]
        if path.is_dir()
        else []
    )


def sidecar_path_for(
    image_path: Path,
    folder_path: Path,
    *,
    no_sidecar: bool = False,
    sidecar_dir: str | Path | None = None,
) -> Path | None:
    """Return the target `.txt` sidecar path for ``image_path``, or ``None`` when disabled."""
    if no_sidecar:
        return None
    if sidecar_dir:
        relative = image_path.relative_to(folder_path)
        return Path(sidecar_dir) / relative.with_suffix(relative.suffix + ".txt")
    return image_path.with_suffix(image_path.suffix + ".txt")


@click.command()
@click.argument("folder", type=str)
@click.option(
    "-o",
    "--output",
    type=str,
    default="captions.toml",
    show_default=True,
    help="Output TOML file path",
)
@click.option(
    "--model",
    "model_key",
    type=click.Choice(list(MODEL_CHOICES.keys())),
    default=DEFAULT_MODEL,
    show_default=True,
    help="Vision-language model to use for captioning (local weights)",
)
@click.option(
    "--quantize",
    type=click.Choice(["none", "int8", "int4"]),
    default="none",
    show_default=True,
    help="Optional CUDA bitsandbytes quantization (not used with MLX)",
)
@click.option(
    "--no-sidecar",
    is_flag=True,
    help="Disable creation of sidecar .txt files with captions",
)
@click.option(
    "--sidecar-dir",
    type=str,
    default=None,
    help="Directory to write sidecar .txt files (preserves folder structure)",
)
@click.option(
    "--backend",
    type=click.Choice(["auto", "transformers", "mlx"]),
    default="auto",
    show_default=True,
)
@click.option(
    "--model-path",
    type=str,
    default=None,
    help="Local compatible model directory; no implicit downloads",
)
@click.option(
    "--device",
    type=click.Choice(["auto", "cpu", "cuda", "mps"]),
    default="auto",
    show_default=True,
)
@click.option(
    "--image-size",
    type=int,
    default=640,
    show_default=True,
    help="Caption preview longest edge",
)
@click.option(
    "--max-pixels",
    type=int,
    default=None,
    help="Processor pixel budget (default: image-size squared)",
)
@click.option("--max-new-tokens", type=int, default=64, show_default=True)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Regenerate even unchanged captions",
)
def main(
    folder: str,
    output: str,
    model_key: str,
    quantize: str,
    no_sidecar: bool,
    sidecar_dir: str | None,
    backend: str,
    model_path: str | None,
    device: str,
    image_size: int,
    max_pixels: int | None,
    max_new_tokens: int,
    overwrite: bool,
) -> None:
    """Caption images recursively using a local vision-language model."""
    if image_size < 32 or max_new_tokens < 1:
        raise click.UsageError(
            "--image-size must be at least 32; --max-new-tokens must be positive"
        )
    effective_max_pixels = max_pixels if max_pixels is not None else image_size**2
    if effective_max_pixels < 32 * 32:
        raise click.UsageError("--max-pixels must be at least 1024")
    if tomli_w is None:
        raise click.UsageError("TOML output requires tomli-w; install a caption extra")
    resolved_backend = resolve_backend(backend, model_key, device)
    resolved_model_path = resolve_model_path(model_key, resolved_backend, model_path)

    folder_path = Path(folder)
    if not folder_path.is_dir():
        raise click.ClickException(f"{folder} is not a valid directory")

    image_files = get_image_files(folder_path)
    if not image_files:
        raise click.ClickException(f"No image files found in {folder}")

    print(f"Found {len(image_files)} image(s) to process")

    output_path = Path(output)
    if output_path.suffix.lower() != ".toml":
        output_path = output_path.with_suffix(".toml")

    print(f"Captions will be saved to {output_path}")
    if not no_sidecar:
        if sidecar_dir:
            print(f"Sidecar .txt files will be created in {sidecar_dir}")
        else:
            print("Sidecar .txt files will also be created alongside images")

    absolute_folder_path = folder_path.resolve().as_posix()

    if output_path.exists():
        try:
            with open(output_path, "rb") as f:
                captions: dict[str, Any] = tomllib.load(f)
        except (OSError, ValueError) as exc:
            raise click.UsageError(
                f"Cannot read existing captions; refusing to overwrite: {exc}"
            ) from exc
    else:
        captions = {}

    if absolute_folder_path in captions and not isinstance(captions[absolute_folder_path], dict):
        raise click.UsageError("Existing caption folder section must be a TOML table")
    if absolute_folder_path not in captions:
        captions[absolute_folder_path] = {}

    metadata_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    metadata: dict[str, Any] = {}
    try:
        metadata_object: object = json.loads(metadata_path.read_text(encoding="utf-8"))
        if isinstance(metadata_object, dict):
            metadata = cast(dict[str, Any], metadata_object)
    except (OSError, ValueError):
        metadata = {}
    if not isinstance(metadata.get(absolute_folder_path), dict):
        metadata[absolute_folder_path] = {}
    settings: dict[str, Any] = {
        "model_key": model_key,
        "model": str(resolved_model_path.resolve()),
        "model_files": model_fingerprint(resolved_model_path),
        "backend": resolved_backend,
        "device": device,
        "quantize": quantize,
        "image_size": image_size,
        "max_pixels": effective_max_pixels,
        "max_new_tokens": max_new_tokens,
        "prompt": CAPTION_PROMPT,
        "pipeline_version": 1,
    }
    pending: list[tuple[Path, str]] = []
    for image_path in image_files:
        relative_path = image_path.relative_to(folder_path).as_posix()
        fingerprint = image_fingerprint(image_path, settings)
        caption = captions[absolute_folder_path].get(relative_path)
        if (
            not overwrite
            and isinstance(caption, str)
            and caption.strip()
            and metadata[absolute_folder_path].get(relative_path) == fingerprint
        ):
            sidecar = sidecar_path_for(
                image_path, folder_path, no_sidecar=no_sidecar, sidecar_dir=sidecar_dir
            )
            if sidecar is not None and not sidecar.exists():
                atomic_write(sidecar, caption.encode("utf-8"))
            continue
        pending.append((image_path, fingerprint))
    print(f"Skipping {len(image_files) - len(pending)} unchanged image(s)")
    if not pending:
        print("Captions are up to date; no model loaded")
        return
    try:
        loaded_model, processor, active_device = load_model_and_processor(
            model_key,
            quantize,
            backend=resolved_backend,
            model_path=str(resolved_model_path),
            device=device,
            max_pixels=effective_max_pixels,
        )
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        raise click.ClickException(str(exc)) from exc

    if not no_sidecar and sidecar_dir:
        Path(sidecar_dir).mkdir(parents=True, exist_ok=True)

    failures = 0
    for idx, (image_path, fingerprint) in enumerate(pending, 1):
        try:
            started = time.perf_counter()
            print(
                f"[{idx}/{len(pending)}] Processing {image_path.name}...",
                end=" ",
                flush=True,
            )

            image = open_image(image_path)
            image = resize_image_aspect_ratio(image, target_size=image_size, allow_upscale=False)

            caption = generate_caption(
                loaded_model, processor, image, active_device, model_key, max_new_tokens
            )
            if not caption.strip():
                raise ValueError("Model returned an empty caption")

            relative_path = image_path.relative_to(folder_path).as_posix()
            captions[absolute_folder_path][relative_path] = caption

            print(f"✓ {caption} ({time.perf_counter() - started:.2f}s)")

            atomic_write(output_path, tomli_w.dumps(captions).encode("utf-8"))

            sidecar_path = sidecar_path_for(
                image_path, folder_path, no_sidecar=no_sidecar, sidecar_dir=sidecar_dir
            )
            if sidecar_path is not None:
                try:
                    atomic_write(sidecar_path, caption.encode("utf-8"))
                except PermissionError:
                    print(
                        f"✗ Permission denied writing sidecar file: {sidecar_path}", file=sys.stderr
                    )
                    print(
                        "Stopping processing - cannot write sidecar files with current permissions.",
                        file=sys.stderr,
                    )
                    print(
                        "Use --no-sidecar flag to process without sidecar files.", file=sys.stderr
                    )
                    if not sidecar_dir:
                        print(
                            "Or use --sidecar-dir to write sidecar files to a different location.",
                            file=sys.stderr,
                        )
                    sys.exit(1)

            metadata[absolute_folder_path][relative_path] = fingerprint
            atomic_write(metadata_path, json.dumps(metadata, indent=2).encode("utf-8"))

        except Exception as e:
            import traceback

            print(f"✗ Error: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            failures += 1
            continue

    print(
        f"\nSuccessfully captioned {len(captions[absolute_folder_path])}/{len(image_files)} images"
    )
    print(f"Captions saved to {output_path}")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
