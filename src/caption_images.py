#!/usr/bin/env python3
"""Local martial-arts image captions with optional Transformers or MLX backends."""

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
import tempfile
import time
import tomllib
from pathlib import Path
from typing import Any, cast

from PIL import Image

from lib.image_utils import (
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

# Models that share the Qwen3VL architecture and inference path
QWEN_MODELS = {"caprl", "qwen3vl", "qwen35-0.8b", "qwen35-2b"}

DEFAULT_MODEL = "qwen3vl"
CAPTION_PROMPT = (
    "Describe the visible people, martial-arts clothing and belt colors, weapons, and body "
    "positions in one plain-text sentence under 255 characters. Use precise martial-arts "
    "terms only when clearly identifiable. Do not infer rank, a kata name, or unseen details."
)


def load_optional_module(name: str, extra: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise RuntimeError(f"{name} requires: uv sync --extra {extra}") from exc


def resolve_backend(backend: str, model_key: str, device: str = "auto") -> str:
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
    if model_path:
        return Path(model_path).expanduser().resolve()
    mlx_path = _MODELS_DIR / "Qwen3-VL-2B-Instruct-4bit"
    if backend == "mlx" and model_key == "qwen3vl" and mlx_path.is_dir():
        return mlx_path
    return MODEL_CHOICES[model_key]


def configure_pixel_budget(processor: Any, max_pixels: int) -> None:
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
    """Load local weights only; optional runtimes are imported on demand."""
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
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float16 if device == "mps" else torch.float32


def quantization_options(model_key: str, device: str, quantize: str, dtype: Any) -> dict[str, Any]:
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
    """Generate caption for an image.

    Dispatches to the appropriate captioning function based on model type.

    Args:
        model: Loaded model (BlipForConditionalGeneration or Qwen3VLForConditionalGeneration)
        processor: Matching processor
        image: PIL Image object
        device: Device to use ("cuda" or "cpu")
        model_key: Key from MODEL_CHOICES used to select prompt strategy.

    Returns:
        Generated caption string
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
    generated = [output[input_length:] for output in outputs]
    return processor.batch_decode(generated, skip_special_tokens=True)[0].strip()


def atomic_write(path: Path, content: bytes) -> None:
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


def image_fingerprint(image_path: Path, settings: dict[str, Any]) -> str:
    stat = image_path.stat()
    payload: dict[str, Any] = {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "settings": settings,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def model_fingerprint(path: Path) -> list[list[str | int]]:
    return (
        [
            [item.name, item.stat().st_size, item.stat().st_mtime_ns]
            for item in sorted(path.iterdir())
            if item.is_file() and item.suffix in {".json", ".safetensors", ".jinja", ".txt"}
        ]
        if path.is_dir()
        else []
    )


def sidecar_path_for(image_path: Path, folder_path: Path, args: Any) -> Path | None:
    if args.no_sidecar:
        return None
    if args.sidecar_dir:
        relative = image_path.relative_to(folder_path)
        return Path(args.sidecar_dir) / relative.with_suffix(relative.suffix + ".txt")
    return image_path.with_suffix(image_path.suffix + ".txt")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Caption images using a selectable vision-language model"
    )
    parser.add_argument("folder", type=str, help="Path to the folder containing images")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="captions.toml",
        help="Output TOML file path (default: captions.toml)",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=list(MODEL_CHOICES.keys()),
        default=DEFAULT_MODEL,
        help=(
            f"Vision-language model to use for captioning (default: {DEFAULT_MODEL}). "
            "blip, caprl, qwen3vl, qwen35-0.8b, or qwen35-2b (local weights)"
        ),
    )
    parser.add_argument(
        "--quantize",
        type=str,
        choices=["none", "int8", "int4"],
        default="none",
        help=(
            "Optional CUDA bitsandbytes quantization (default: none). "
            "Reduces weight memory; speed depends on hardware. Not used with MLX."
        ),
    )
    parser.add_argument(
        "--no-sidecar",
        action="store_true",
        help="Disable creation of sidecar .txt files with captions",
    )
    parser.add_argument(
        "--sidecar-dir",
        type=str,
        default=None,
        help="Directory to write sidecar .txt files (preserves folder structure)",
    )
    parser.add_argument("--backend", choices=["auto", "transformers", "mlx"], default="auto")
    parser.add_argument(
        "--model-path", help="Local compatible model directory; no implicit downloads"
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument(
        "--image-size", type=int, default=640, help="Caption preview longest edge (default: 640)"
    )
    parser.add_argument(
        "--max-pixels", type=int, help="Processor pixel budget (default: image-size squared)"
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--overwrite", action="store_true", help="Regenerate even unchanged captions"
    )

    args = parser.parse_args()
    if args.image_size < 32 or args.max_new_tokens < 1:
        parser.error("--image-size must be at least 32; --max-new-tokens must be positive")
    args.max_pixels = args.max_pixels if args.max_pixels is not None else args.image_size**2
    if args.max_pixels < 32 * 32:
        parser.error("--max-pixels must be at least 1024")
    if tomli_w is None:
        parser.error("TOML output requires tomli-w; install a caption extra")
    args.backend = resolve_backend(args.backend, args.model, args.device)
    model_path = resolve_model_path(args.model, args.backend, args.model_path)

    # Validate folder
    folder_path = Path(args.folder)
    if not folder_path.is_dir():
        print(f"Error: {args.folder} is not a valid directory", file=sys.stderr)
        sys.exit(1)

    # Get image files
    image_files = get_image_files(folder_path)
    if not image_files:
        print(f"No image files found in {args.folder}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(image_files)} image(s) to process")

    # Determine output path and format
    output_path = Path(args.output)
    if output_path.suffix.lower() != ".toml":
        output_path = output_path.with_suffix(".toml")

    print(f"Captions will be saved to {output_path}")
    if not args.no_sidecar:
        if args.sidecar_dir:
            print(f"Sidecar .txt files will be created in {args.sidecar_dir}")
        else:
            print("Sidecar .txt files will also be created alongside images")

    # Process images and generate captions with streaming writes
    # Get absolute path of the input folder to use as section name
    absolute_folder_path = folder_path.resolve().as_posix()

    # Load existing captions if the file exists
    if output_path.exists():
        try:
            with open(output_path, "rb") as f:
                captions: dict[str, Any] = tomllib.load(f)
        except (OSError, ValueError) as exc:
            parser.error(f"Cannot read existing captions; refusing to overwrite: {exc}")
    else:
        captions = {}

    # Only create the section if it doesn't already exist
    if absolute_folder_path in captions and not isinstance(captions[absolute_folder_path], dict):
        parser.error("Existing caption folder section must be a TOML table")
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
        "model": str(model_path.resolve()),
        "model_files": model_fingerprint(model_path),
        "backend": args.backend,
        "device": args.device,
        "quantize": args.quantize,
        "image_size": args.image_size,
        "max_pixels": args.max_pixels,
        "max_new_tokens": args.max_new_tokens,
        "prompt": CAPTION_PROMPT,
        "pipeline_version": 1,
    }
    pending: list[tuple[Path, str]] = []
    for image_path in image_files:
        relative_path = image_path.relative_to(folder_path).as_posix()
        fingerprint = image_fingerprint(image_path, settings)
        caption = captions[absolute_folder_path].get(relative_path)
        if (
            not args.overwrite
            and isinstance(caption, str)
            and caption.strip()
            and metadata[absolute_folder_path].get(relative_path) == fingerprint
        ):
            sidecar = sidecar_path_for(image_path, folder_path, args)
            if sidecar is not None and not sidecar.exists():
                atomic_write(sidecar, caption.encode("utf-8"))
            continue
        pending.append((image_path, fingerprint))
    print(f"Skipping {len(image_files) - len(pending)} unchanged image(s)")
    if not pending:
        print("Captions are up to date; no model loaded")
        return
    model, processor, device = load_model_and_processor(
        args.model,
        args.quantize,
        backend=args.backend,
        model_path=str(model_path),
        device=args.device,
        max_pixels=args.max_pixels,
    )

    # Prepare sidecar directory if specified
    sidecar_base_dir = None
    if not args.no_sidecar and args.sidecar_dir:
        sidecar_base_dir = Path(args.sidecar_dir)
        sidecar_base_dir.mkdir(parents=True, exist_ok=True)

    failures = 0
    for idx, (image_path, fingerprint) in enumerate(pending, 1):
        try:
            started = time.perf_counter()
            print(
                f"[{idx}/{len(pending)}] Processing {image_path.name}...",
                end=" ",
                flush=True,
            )

            # Load and resize image
            image = open_image(image_path)
            image = resize_image_aspect_ratio(
                image, target_size=args.image_size, allow_upscale=False
            )

            # Generate caption
            caption = generate_caption(
                model, processor, image, device, args.model, args.max_new_tokens
            )
            if not caption.strip():
                raise ValueError("Model returned an empty caption")

            # Store with relative path as key under the folder section
            relative_path = image_path.relative_to(folder_path).as_posix()
            captions[absolute_folder_path][relative_path] = caption

            # Show caption
            print(f"✓ {caption} ({time.perf_counter() - started:.2f}s)")

            # Stream write to TOML file immediately
            atomic_write(output_path, tomli_w.dumps(captions).encode("utf-8"))

            # Write sidecar .txt file if not disabled
            if not args.no_sidecar:
                if sidecar_base_dir:
                    # Write to sidecar directory, preserving folder structure
                    relative_path_obj = image_path.relative_to(folder_path)
                    sidecar_path = sidecar_base_dir / relative_path_obj.with_suffix(
                        relative_path_obj.suffix + ".txt"
                    )
                    # Create subdirectories if needed
                    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
                else:
                    # Write alongside image
                    sidecar_path = image_path.with_suffix(image_path.suffix + ".txt")

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
                    if not args.sidecar_dir:
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
