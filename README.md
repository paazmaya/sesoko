# sesoko (瀬底)

> Prepare images for training machine learning models

Prepare exact-size images with optional person-centered crops and local martial-arts captions.
Python 3.13+ is required. Basic preparation needs only Pillow and Click; no GPU or ML runtime.

## Installation

```bash
uv sync
uv run python src/crop_yolo.py --input-dir images --output-dir prepared
```

Install only the extras needed for the current workflow:

| Extra                  | Purpose                                                    |
| ---------------------- | ---------------------------------------------------------- |
| `detector-onnx`        | DEIMv2 CPU detection on macOS, Windows, and Linux          |
| `detector-yolo`        | Optional Ultralytics compatibility backend                 |
| `caption-mlx`          | MLX-VLM on Apple Silicon; use an MLX-compatible checkpoint |
| `caption-transformers` | PyTorch/Transformers on NVIDIA, MPS, or CPU                |
| `cuda-quantization`    | Optional bitsandbytes weight quantization on CUDA          |
| `codecs`               | Additional HEIC/HEIF decoding                              |
| `training`             | Separate training dependencies, not needed for preparation |

Include extras on `uv run` commands to retain their dependencies during synchronization.
Multiple extras can be combined. `uv run --no-sync` uses the already installed environment.
The project no longer forces CUDA 13 wheels on every platform. NVIDIA users should install
a PyTorch build compatible with their GPU and driver using the official
[PyTorch installation selector](https://pytorch.org/get-started/locally/).
MLX and Transformers extras are substantial; neither is needed for cropping.

## Person-Centered Preparation

The existing `src/crop_yolo.py` entry point supports both detector backends.
Without `--crop-focus`, it center-crops without loading a detector.

Before running the DEIMv2 examples, download
[deimv2_hgnetv2_pico_coco.onnx](https://huggingface.co/tatsuyaaaaaaa/deimv2-onnx/resolve/e1514db73b6a64566560cc54a2c28f3bf25a31da/deimv2_hgnetv2_pico_coco.onnx)
(approximately 6.35 MB) into `models/`. This command downloads the tested revision:

```bash
uvx --from huggingface-hub hf download tatsuyaaaaaaa/deimv2-onnx \
  deimv2_hgnetv2_pico_coco.onnx \
  --revision e1514db73b6a64566560cc54a2c28f3bf25a31da --local-dir models
```

```bash
# DEIMv2: select the largest confident person and produce exactly 512x512 JPEG
uv run --extra detector-onnx python src/crop_yolo.py \
  --input-dir images --output-dir prepared \
  --crop-focus person --detector deim --model models/deimv2_hgnetv2_pico_coco.onnx

# Rectangular output with extra space for extended arms or weapons
uv run --extra detector-onnx python src/crop_yolo.py \
  --input-dir images --output-dir prepared \
  --crop-focus person --detector deim --model models/deimv2_hgnetv2_pico_coco.onnx \
  --width 768 --height 1024 --margin 0.3 --format png --stats crops.json
```

The optional YOLO backend uses
[yolo11n-seg.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt).
Place it in the project root; Ultralytics can also download it automatically if absent.

```bash
# Optional YOLO compatibility path; --detector yolo remains the default
uv run --extra detector-yolo python src/crop_yolo.py \
  --input-dir images --output-dir prepared --crop-focus person
```

- Default output is 512x512 optimized JPEG at quality 85. `--resolution` sets both dimensions;
  `--width` and `--height` independently override them. JPEG, PNG, and WebP are supported.
- Person detections below `--confidence 0.4` are discarded. The largest remaining box wins,
  with confidence and coordinates breaking ties deterministically.
- `--margin 0.2` requests 20% extra width and height. The crop expands to the output aspect
  ratio, shifts inward at image edges, and reduces optional margin before adding padding.
  White padding is used only when the detected subject cannot fit that ratio inside the source.
- Images are resized once to the exact requested dimensions, including upscaling if necessary.
  People are not stretched. Padding cannot restore body parts missing from the source photo.
- Missing targets are skipped; `--fallback-center` explicitly enables center-crop fallback.
- Processing is nonrecursive. Originals are not overwritten; input and output folders must
  differ. Duplicate stems and existing outputs receive distinct names; reruns create new outputs.
- EXIF orientation is applied before detection. Transparency is composited on white.
- This is object detection, not identity recognition or background segmentation. Person boxes
  can omit a sword, staff, or practice partner; adjust the margin and inspect the results.

### Obtaining A DEIMv2 Export

No checkpoint is downloaded automatically. Start with **Pico** (1.5M parameters), then compare
**N** (3.6M) if difficult poses are missed. Atto (0.5M) is a smaller, lower-accuracy option.
Overall COCO AP and upstream latency numbers are not person-crop or local latency guarantees.

#### Ready-Made ONNX Exports

For the simplest setup, use
[tatsuyaaaaaaa/deimv2-onnx](https://huggingface.co/tatsuyaaaaaaa/deimv2-onnx), specifically
**`deimv2_hgnetv2_pico_coco.onnx`** (approximately 6.35 MB). It works directly with the current
adapter; no conversion, Torch installation, or upstream training environment is required.

The [first person-cropping example](#person-centered-preparation) includes the pinned download
and usage commands.

SHA-256 of the tested Pico artifact:

```text
507850498155f279ca7fa194cf0b95ebf1068ed9420b1bb6479acc4a66a37f8d
```

An alternative is [wrekd/deimv2-pico-onnx](https://huggingface.co/wrekd/deimv2-pico-onnx), file
`onnx/model.onnx` (approximately 6.38 MB), tested at revision
`50623ab09d0890146c3df3f67823a3d406a2d896`. Both exports loaded successfully with ONNX Runtime CPU
and the current adapter. On one existing sample image, they produced effectively identical person
boxes with confidence approximately 0.953; inference took 38-41 ms, excluding download and session
loading. This is a compatibility smoke test, not a representative accuracy or performance benchmark.

Choose **HGNetv2 Pico**, not the DINOv3-S file in the same repository. These are community exports;
verify upstream licensing and provenance for the exact artifact before redistribution or commercial use.

#### Exporting From Original Weights

The original Hugging Face checkpoint is
[Intellindust/DEIMv2_HGNetv2_PICO_COCO](https://huggingface.co/Intellindust/DEIMv2_HGNetv2_PICO_COCO).
Its pretrained weights are in
[model.safetensors](https://huggingface.co/Intellindust/DEIMv2_HGNetv2_PICO_COCO/blob/main/model.safetensors),
with a matching `config.json` (1.56M parameters, 640x640 input). That repository does not provide
an ONNX export. Neither its Safetensors file nor a renamed copy can be passed directly to Sesoko's
`--model` option or to the upstream export script's `.pth` checkpoint loader. Use the matching
upstream Pico `.pth` checkpoint for the export command below, or load the Hugging Face weights
with the matching upstream model code before exporting. DINOv3 and EdgeCrafter checkpoints are
not compatible with this adapter's preprocessing contract.

In a separate [DEIMv2 checkout/environment](https://github.com/Intellindust-AI-Lab/DEIMv2),
download the pretrained Pico `.pth` checkpoint linked in its model zoo, then export:

```bash
python tools/deployment/export_onnx.py \
  -c configs/deimv2/deimv2_hgnetv2_pico_coco.yml \
  -r /path/to/pico.pth --check
```

Supply the resulting ONNX path with `--model`. Record the upstream revision, checkpoint/config
and SHA-256 alongside the artifact; retain its license and notices.

The adapter supports **standard HGNetv2 Atto/Femto/Pico/N COCO deployment exports**, not DINOv3
S/M/L/X exports or arbitrary ONNX detectors. It uses the graph's fixed square input size,
FP32 RGB NCHW values in [0, 1], black letterboxing, and int64 `orig_target_sizes`.
Named outputs must be `labels`, `boxes`, and `scores`; deployed person label is **0**.
Boxes are mapped back from the padded preview to the oriented original before cropping.
The runtime is ONNX Runtime CPU; Torch and the upstream training code are not needed after export.

## Local Captioning

Caption the **prepared folder** so descriptions match the content kept in the training images.
Caption previews stay in memory and do not change saved output dimensions or encoding.

Download a complete model directory before running captioning; tokenizer and processor files
are needed alongside the weights. The links below open each model's downloadable files:

| Model Key           | Model Download                                                                                | Default Local Directory        |
| ------------------- | --------------------------------------------------------------------------------------------- | ------------------------------ |
| `qwen3vl` (default) | [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/tree/main)            | `models/Qwen3-VL-2B-Instruct`  |
| `qwen35-0.8b`       | [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B/tree/main)                            | `models/Qwen3.5-0.8B`          |
| `qwen35-2b`         | [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B/tree/main)                                | `models/Qwen3.5-2B`            |
| `caprl`             | [CapRL-Qwen3VL-2B](https://huggingface.co/internlm/CapRL-Qwen3VL-2B/tree/main)                | `models/CapRL-Qwen3VL-2B`      |
| `blip`              | [Image-Captioning-Blip](https://huggingface.co/Amirhossein75/Image-Captioning-Blip/tree/main) | `models/Image-Captioning-Blip` |

For Apple Silicon, use the separate
[MLX 4-bit Qwen3-VL-2B-Instruct download](https://huggingface.co/mlx-community/Qwen3-VL-2B-Instruct-4bit/tree/main)
in `models/Qwen3-VL-2B-Instruct-4bit`. It is not interchangeable with the Transformers weights.

```bash
# Apple Silicon: download the MLX checkpoint before its first use
uvx --from huggingface-hub hf download mlx-community/Qwen3-VL-2B-Instruct-4bit \
  --local-dir models/Qwen3-VL-2B-Instruct-4bit
uv run --extra caption-mlx python src/caption_images.py prepared --backend mlx

# NVIDIA: download the Transformers checkpoint before its first use
uvx --from huggingface-hub hf download Qwen/Qwen3-VL-2B-Instruct \
  --local-dir models/Qwen3-VL-2B-Instruct
uv run --extra caption-transformers --extra cuda-quantization python src/caption_images.py \
  prepared --backend transformers --device cuda --quantize int4

# Compare a smaller preview without modifying the prepared images
uv run --extra caption-mlx python src/caption_images.py prepared \
  --image-size 512 --max-new-tokens 64 -o captions-512.toml
```

`--model-path` overrides the local directory. New weights are never downloaded implicitly.
MLX automatically uses `models/Qwen3-VL-2B-Instruct-4bit` for the default model when available.
MLX quantization comes from the checkpoint; do not pass CUDA `--quantize` flags to it.
GGUF files are not supported by these backends.

`--backend auto` prefers installed MLX on Apple Silicon for Qwen models; otherwise it selects
Transformers. Explicit `--device cpu` or `--device cuda` selects Transformers under auto mode.
Transformers selects CUDA, then MPS, then CPU; precision is BF16 on compatible CUDA hardware,
FP16 on other CUDA/MPS devices, and FP32 on CPU. Device and effective precision are printed.
Quantization defaults to **none**; lower weight memory does not guarantee faster inference.

Defaults are a **640px longest edge**, no preview upscaling, a **409,600-pixel processor budget**,
and **64 generated tokens**, with thinking disabled. Override with `--image-size`, `--max-pixels`,
and `--max-new-tokens`. Processors may resize small images to their minimum area or patch grid;
fixed-resolution models such as BLIP retain their processor's resolution. A token budget is an
upper limit, not a guarantee of a complete sentence. The 255-character instruction is prompt-based.

The short prompt asks for visible clothing, belts, weapons and poses without guessing rank or
kata. Only generated tokens are decoded. BLIP uses greedy decoding rather than five-beam search.
The model stays loaded across images and CUDA cache is not emptied after every caption.

### Output And Resume

Captions are atomically checkpointed after each image to `captions.toml`, grouped by absolute
folder path. Sidecars default to `photo.jpg.txt`; use `--no-sidecar` or `--sidecar-dir` to change this.
`captions.toml.meta.json` records fingerprints using input size/mtime, local model-file metadata,
prompt, backend and generation settings. Keep it alongside the TOML file for resume.

Unchanged images with valid captions are skipped, and missing sidecars are restored from cached
captions. If everything is cached, no model is loaded. Changed settings or source metadata invalidate
the cache; `--overwrite` forces regeneration. Legacy outputs without metadata are regenerated once.
Per-image failures are reported and return a nonzero exit status while retaining completed captions.

```toml
["/absolute/path/to/prepared"]
"photo.jpg" = "A practitioner in a white gi holds a wooden sword."
```

### Quality And Performance Checks

Keep Qwen3-VL-2B as the baseline. Compare 896/640/512px previews on representative martial-arts
photos before changing the default model. Smaller previews reduce vision work but can lose thin
weapons, belt details, or distant people. Compare Qwen3.5-0.8B and 2B using compatible recent
runtime versions and matching local checkpoints; their domain quality is not assumed.

Measure cold loading separately from warm per-image median/p95 latency, peak memory, body/weapon
retention, and invented or missing caption details. The script prints per-image elapsed time.
No full dataset benchmark or NVIDIA execution has been performed as part of this implementation.

## Formats And Development

Common Pillow-readable formats include JPEG, PNG, WebP, GIF, TIFF, and AVIF. Optional `codecs`
adds HEIF support. Extension discovery is not a guarantee of decoding support: JPEG XL and other
formats require appropriate installed Pillow plugins. Animated/multipage inputs use the first frame.

```bash
uv run ruff format
uv run ruff check src tests
uv run pytest

# Optional detector tests; the synthetic ONNX graph test also needs the onnx package
uv run --extra detector-onnx --with onnx pytest tests/test_crop_yolo.py

# Caption workflow tests use fakes and only need the small TOML writer, not a model
uv run --with tomli-w pytest tests/test_caption_images.py
```

Unit tests use injected detectors and fake caption models, not downloaded checkpoints.

### Comparing Local Detectors

`tests/test_detector_comparison.py` is an opt-in real-model test. It compares the project-root
`deimv2_hgnetv2_pico_coco.onnx`, `yolo11n-seg.pt`, and `yolov8n-seg.pt` without downloading weights.
For the YOLOv8 checkpoint, use
[yolov8n-seg.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-seg.pt)
if it is not already present. All three checkpoints must exist before running the test.

```bash
SESOKO_COMPARE_DETECTORS=1 uv run --extra detector-onnx --extra detector-yolo \
  pytest tests/test_detector_comparison.py -s -o addopts=""

# Evaluate a folder of original photos instead of the existing annotated sample
SESOKO_COMPARE_DETECTORS=1 SESOKO_COMPARE_IMAGES=images SESOKO_COMPARE_REPEATS=10 \
  uv run --extra detector-onnx --extra detector-yolo \
  pytest tests/test_detector_comparison.py -s -o addopts=""
```

In PowerShell, set `$env:SESOKO_COMPARE_DETECTORS = "1"` before the equivalent `uv run` command.
`SESOKO_COMPARE_IMAGES` accepts one image or a directory (searched recursively).
`SESOKO_COMPARE_OUTPUT` overrides the default `target/detector-comparison` report directory.
Repeated comparisons overwrite generated reports and previews there, not the source images.

The test uses CPU FP32, one inference thread, 640x640 inputs, confidence 0.4, one warm-up per
image, and five measured runs by default. It reports median/p95 adapter latency separately from
checkpoint loading and saves `comparison.json`, annotated detections, and 512x512 person crops
using the same crop policy for every model. JSON includes checkpoint hashes, package versions,
boxes, confidences, person counts, and largest-person box IoU between DEIMv2 and each YOLO model.

Normal unit-test runs skip this comparison. The default existing sample must contain a detected
person; custom images may legitimately contain none. Without labeled ground truth, IoU measures
agreement, not accuracy, and confidence scores are not calibrated across models. YOLO segmentation
models still compute masks, although only boxes are used. This CPU comparison does not verify
NVIDIA execution or newer caption-model quality.

## License

Sesoko code is MIT. External model weights and runtimes retain their own licenses.
[DEIMv2's current license](https://github.com/Intellindust-AI-Lab/DEIMv2/blob/main/LICENSE.md)
permits noncommercial use; commercial use requires a separate license. Do not redistribute its
weights as MIT. Review Ultralytics licensing independently when using the optional YOLO backend.
