#!/usr/bin/env python3
"""Image captioning script using Qwen3-VL-2B-Instruct.

This script processes all images in a given folder and generates captions
using the Qwen3-VL-2B-Instruct model, storing the results in a TOML file.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

try:
    import tomli_w
except ImportError:
    tomli_w = None

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:
    pass


def get_image_files(folder_path: Path) -> list[Path]:
    """Get all image files from the given folder.

    Args:
        folder_path: Path to the folder containing images

    Returns:
        List of paths to image files
    """
    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".webp",
        ".bmp",
        ".heic",
        ".heif",
        ".avif",
    }
    image_files = []

    for file in folder_path.rglob("*"):
        if file.is_file() and file.suffix.lower() in image_extensions:
            image_files.append(file)

    return sorted(image_files)


def resize_image(image: Image.Image, target_size: int = 896) -> Image.Image:
    """Resize image to fit within target_size x target_size.

    Args:
        image: PIL Image object
        target_size: Target size in pixels (default 896x896)

    Returns:
        Resized PIL Image object
    """
    # Calculate aspect ratio
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


def load_model_and_processor():
    """Load Qwen3-VL-4B-Instruct model and processor."""
    # https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
    # model_id = "H:\\vision-models\\Qwen_Qwen3-VL-4B-Instruct"
    # https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct
    model_id = "H:\\vision-models\\Qwen_Qwen3-VL-2B-Instruct"

    print(f"Loading model {model_id}...")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )

    processor = AutoProcessor.from_pretrained(model_id)

    # Device is handled automatically by device_map="auto"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    return model, processor, device


def generate_caption(
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    image: Image.Image,
    device: str,
) -> str:
    """Generate caption for an image using Qwen3-VL model.

    Args:
        model: Qwen3-VL model
        processor: Model processor
        image: PIL Image object
        device: Device to use ("cuda" or "cpu")

    Returns:
        Generated caption string
    """
    # Detailed prompt for martial arts - instruct model to respond in plain text
    prompt = "Describe the image, it contains Japanese martial arts. What martial art is shown? Describe clothing, belt, technique, the surrounding area, what kind of emotions the person might show? Respond with plain text only, no formatting or markdown. Be firm, no guessing."

    # Prepare inputs using chat template (following official example)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }
    ]

    # Prepare inputs - simpler approach
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, images=[image], return_tensors="pt").to(device)

    # Generate without trimming complexity
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
        )

    # Simply decode the entire output
    caption = processor.decode(outputs[0], skip_special_tokens=True)

    # Remove the prompt part if it's in there
    if "assistant" in caption:
        caption = caption.split("assistant")[-1].strip()

    return caption


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Caption images using Qwen3-VL-2B-Instruct model")
    parser.add_argument("folder", type=str, help="Path to the folder containing images")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="captions.toml",
        help="Output TOML file path (default: captions.toml)",
    )

    args = parser.parse_args()

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

    # Load model and processor
    model, processor, device = load_model_and_processor()

    # Determine output path and format
    output_path = Path(args.output)
    if output_path.suffix.lower() != ".toml":
        output_path = output_path.with_suffix(".toml")

    print(f"Captions will be saved to {output_path}")

    # Process images and generate captions with streaming writes
    # Get absolute path of the input folder to use as section name
    absolute_folder_path = folder_path.resolve().as_posix()
    captions: Dict[str, Dict[str, str]] = {absolute_folder_path: {}}

    for idx, image_path in enumerate(image_files, 1):
        try:
            print(
                f"[{idx}/{len(image_files)}] Processing {image_path.name}...",
                end=" ",
                flush=True,
            )

            # Load and resize image
            image = Image.open(image_path).convert("RGB")
            image = resize_image(image, target_size=896)

            # Generate caption
            caption = generate_caption(model, processor, image, device)

            # Store with relative path as key under the folder section
            relative_path = image_path.relative_to(folder_path).as_posix()
            captions[absolute_folder_path][relative_path] = caption

            # Show caption
            print(f"✓ {caption}")

            # Stream write to TOML file immediately
            if tomli_w:
                with open(output_path, "wb") as f:
                    tomli_w.dump(captions, f)
            else:
                # Fallback to JSON if tomli_w not available
                with open(output_path.with_suffix(".json"), "w", encoding="utf-8") as f:
                    json.dump(captions, f, indent=2, ensure_ascii=False)

            # Clear GPU cache periodically to prevent memory fragmentation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            import traceback

            print(f"✗ Error: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            continue

    print(f"\nSuccessfully captioned {len(captions)}/{len(image_files)} images")
    print(f"Captions saved to {output_path}")


if __name__ == "__main__":
    main()
