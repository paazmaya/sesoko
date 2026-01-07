# Image Captioning for Martial Arts Using Qwen3-VL

Fast, efficient image captioning for martial arts imagery using **Qwen3-VL-2B-Instruct**.

## Quick Start

```bash
# Install dependencies
uv sync

# Run captioning on a folder
uv run python caption_images.py "path/to/images"

# Output will be saved to captions.toml in the current folder
# and for each image as sidecar text files under "path/to/images"
```

## Current Model

**Qwen3-VL-2B-Instruct** - Optimized for speed and quality:
- 🚀 **Ultra-fast inference** with greedy decoding
- 💾 **Minimal VRAM** (~4-6 GB) - fits on any modern GPU
- ✨ **High quality** captions for martial arts images
- 🔄 **Streaming writes** - recovers from interruptions
- 📝 **Plain text output** - no markdown formatting

[View on Hugging Face](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)

## Features

- ✅ Streams captions to TOML file in real-time
- ✅ Creates sidecar `.txt` files alongside images (by default)
- ✅ Aspect-ratio preserving image resizing (896x896 max)
- ✅ Plain English descriptions without formatting
- ✅ Graceful error handling with per-image try-catch
- ✅ Progress tracking with image count

## Alternative Models

For higher quality at the cost of slower inference, use the 4B version:

Edit `caption_images.py` line ~86:
```python
model_id = "Qwen/Qwen3-VL-4B-Instruct"
```

**4B Model Specs:**
- Parameters: 4B
- Speed: ~2x slower than 2B
- Quality: Higher detail in captions
- VRAM: ~6-8 GB

## Configuration

### Model Loading

The model is loaded with optimizations in `load_model_and_processor()`:
- `dtype=torch.bfloat16` - Reduced precision for faster inference
- `low_cpu_mem_usage=True` - Memory efficient loading
- `device_map="auto"` - Automatic device placement

### Generation Parameters

Captions are generated with these parameters:
- `max_new_tokens=128` - Maximum caption length
- `do_sample=False` - Greedy decoding (fastest)

### Output Format

**TOML File** - Captions are saved to TOML format, organized by folder absolute path:
```toml
["/absolute/path/to/images/folder"]
"image1.jpg" = "Caption text here..."
"image2.jpg" = "Another caption..."

["/another/absolute/path"]
"photo.jpg" = "Different folder caption..."
```

**Sidecar Files** - By default, `.txt` files are created alongside each image:
```
images/
  photo1.jpg
  photo1.jpg.txt     # Contains the caption for photo1.jpg
  photo2.jpg
  photo2.jpg.txt     # Contains the caption for photo2.jpg
```

Streaming writes mean captions are saved after each image is processed. The absolute path of the input folder is used as the section name in the TOML file, allowing you to organize captions from multiple folders in one file.

## Supported Image Formats

- `.jpg`, `.jpeg`
- `.png`
- `.gif`
- `.webp`
- `.bmp`
- `.heic`, `.heif` (with pillow-heif)
- `.avif`

## Requirements

- Python 3.11+
- CUDA-capable GPU (Nvidia that is)
- ~4-6 GB VRAM for 2B model
- ~6-8 GB VRAM for 4B model

## Advanced Usage

### Custom Output Path

```bash
uv run python caption_images.py "path/to/images" -o my_captions.toml
```

### Disable Sidecar Files

By default, sidecar `.txt` files are created alongside images. To disable this:

```bash
uv run python caption_images.py "path/to/images" --no-sidecar
```

This will only create the TOML file without generating individual caption files.

### Write Sidecar Files to a Different Directory

If you have permission issues writing to the image directory (common with Dropbox/cloud storage), write sidecar files to a different location:

```bash
uv run python caption_images.py "Dropbox\Karatejukka 2023" --sidecar-dir "captions"
```

This preserves the folder structure relative to the source images. For example:
```
Source images:
  Dropbox\Karatejukka 2023\folder\image.jpg

Sidecar files written to:
  captions\folder\image.jpg.txt
```

### Processing Large Batches

The script processes images sequentially with streaming saves. For very large batches:
1. Resume interrupted runs - partial results are saved
2. Monitor GPU memory with `nvidia-smi`
3. Use 2B model for fastest processing

## Troubleshooting

**OOM (Out of Memory):**
- Use the 2B model instead of 4B
- Reduce image size in `resize_image()` (currently 896x896)

**Permission denied writing sidecar files:**
- Dropbox and other cloud storage services can lock files during sync
- Use `--no-sidecar` to skip sidecar file creation
- Or use `--sidecar-dir` to write sidecar files to a local directory outside Dropbox
- Example: `uv run python caption_images.py "Dropbox\images" --sidecar-dir "captions"`

**Slow Inference:**
- Ensure Flash Attention 2 is installed
- Check GPU isn't throttled: `nvidia-smi dmon`
- Verify CUDA is properly detected

**Missing Files:**
- Install dependencies: `uv sync`
- Check image folder path exists

## Notes

- The model focuses on detailed caption generation
- Prompt explicitly requests plain text (no markdown)
- GPU cache is cleared after each image to prevent fragmentation
- Results are appended to the output file (streaming)

---

## Research Context (Legacy)

This project was developed from comprehensive research into vision-language models for martial arts captioning. Original research evaluated:
- PaliGemma 2
- LLaVA-7B
- BLIP-2
- Flamingo
- And other VLMs

The decision to use Qwen3-VL-2B-Instruct is based on superior speed/quality tradeoff for batch processing scenarios.

## License

MIT