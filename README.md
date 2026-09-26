# local-image-processor

An offline [Model Context Protocol](https://modelcontextprotocol.io) (MCP) server for local image processing. Built on the official `mcp` SDK v2 (`mcp.server.mcpserver.MCPServer`), [Pillow](https://python-pillow.org/), and [OpenCV](https://opencv.org/). All processing runs strictly on the local machine — no external APIs, no cloud services.

## Tools (34)

### Geometry & Transformation

| Tool              | Description                                                                         |
| ----------------- | ----------------------------------------------------------------------------------- |
| `resize_image`    | Resize to exact or bounding-box dimensions with optional aspect-ratio preservation. |
| `crop_image`      | Crop a pixel bounding box (coordinates are clamped to the image).                   |
| `smart_crop`      | Auto-detect and center focal subjects (faces, AI saliency, edge density) to aspect ratio or dimensions. |
| `bulk_crop`       | Batch crop multiple images from a list, directory, or glob using smart or fixed cropping. |
| `bulk_resize`     | Batch resize multiple images with format conversion, quality control, and metadata stripping. |
| `rotate_flip`     | Rotate by 90/180/270° or any arbitrary angle, plus horizontal/vertical flips.       |
| `trim_whitespace` | Auto-detect and remove solid background/alpha borders.                              |
| `image_grid`      | Arrange multiple images into a grid or collage layout with configurable spacing.    |

### Tonal & Exposure

| Tool                     | Description                                                                                |
| ------------------------ | ------------------------------------------------------------------------------------------ |
| `adjust_brightness`      | Linear brightness multiplier (1.0 = unchanged).                                            |
| `adjust_contrast`        | Contrast multiplier around mid-gray (1.0 = unchanged).                                     |
| `auto_levels`            | Stretch each channel's dynamic range to 0–255.                                             |
| `histogram_equalization` | CLAHE on the luminance channel in LAB space.                                               |
| `normalize`              | Normalize pixel values to a specified target intensity range (minmax, zscore, percentile). |

### Color

| Tool                   | Description                                         |
| ---------------------- | --------------------------------------------------- |
| `adjust_saturation`    | Saturation scaling in HSV space (0.0 = grayscale).  |
| `convert_color_space`  | Convert to RGB, GRAY, HSV, LAB, or CMYK.            |
| `adjust_white_balance` | Shift warmth (blue↔amber) and tint (green↔magenta). |
| `color_tint`           | Apply a hex color wash at a given strength.         |

### Detail, Filters & Denoising

| Tool               | Description                                                                       |
| ------------------ | --------------------------------------------------------------------------------- |
| `sharpen_image`    | Unsharp mask (radius, percent, threshold).                                        |
| `blur_image`       | Gaussian, box, or median blur.                                                    |
| `denoise_image`    | Non-local-means or bilateral denoising.                                           |
| `edge_detection`   | Canny edge detector producing a binary mask.                                      |
| `image_difference` | Calculate pixel difference between two images with visualization modes and stats. |
| `upscale_image`    | 2× / 4× super-resolution via Real-ESRGAN or SwinIR with content-aware detection.  |

### Compositing & Watermarking

| Tool                      | Description                                                                       |
| ------------------------- | --------------------------------------------------------------------------------- |
| `overlay_image`           | Overlay a logo/watermark with position, offsets, and opacity.                     |
| `overlay_text`            | Render text with a TrueType font (automatic fallback to the default bitmap font). |
| `bulk_watermark`          | Batch apply image or text watermarks across multiple images with position and opacity. |
| `remove_background`       | Remove background using AI segmentation (U-2-Net) or Euclidean chroma keying.     |
| `bulk_remove_background`  | Batch remove backgrounds across multiple images via AI or chroma keying.         |
| `image_blend`             | Blend two images pixel-by-pixel with an adjustable alpha weight.                   |

### File Encoding & Metadata

| Tool                 | Description                                                                              |
| -------------------- | ---------------------------------------------------------------------------------------- |
| `convert_format`     | Convert between PNG, JPEG, WEBP, TIFF, BMP with quality control.                         |
| `bulk_convert`       | Batch convert multiple images to target format with quality control and metadata stripping. |
| `strip_metadata`     | Re-save an image with all EXIF/IPTC/GPS metadata removed.                                |
| `get_image_metadata` | Return dimensions, format, mode, color depth, file size, and parsed EXIF as a JSON dict. |

### Inspection

| Tool         | Description                                                                          |
| ------------ | ------------------------------------------------------------------------------------ |
| `image_info` | Return dimensions, mode, format, file size, EXIF data, ICC profile, and frame count. |

## Installation

Requires Python 3.13+.

### CPU (Default)

Runs strictly offline on any CPU with zero external drivers:

```sh
pip install -r requirements.txt
```

Dependencies: `mcp>=2.0.0,<3.0.0`, `pillow>=10.0.0,<12.0.0`, `opencv-python-headless>=4.8.0,<5.0.0`, `numpy>=1.24.0,<2.2.0`, `rembg[cpu]>=2.0.0,<3.0.0`.

### NVIDIA CUDA GPU Acceleration (Optional)

If you have an NVIDIA GPU (e.g. RTX series) and want hardware acceleration (~10×–20× faster super-resolution upscaling and AI background removal):

```sh
# Option 1: Install full GPU requirements
pip install -r requirements-gpu.txt

# Option 2: If requirements.txt is already installed, add the CUDA/cuDNN runtime wheels:
pip install onnxruntime-gpu nvidia-cudnn-cu12 nvidia-cublas-cu12
```

**How GPU execution works:**
* The server automatically locates NVIDIA CUDA and cuDNN runtime libraries on Windows (via `os.add_dll_directory`) and routes inference to `CUDAExecutionProvider`.
* If a GPU is not present or cuDNN DLLs are missing, the server performs a seamless hardware dry-run and transparently falls back to `CPUExecutionProvider` without crashing or throwing errors.

## Running

```sh
python server.py
```

The server speaks MCP over stdio (the default transport).

## Configuration

### Claude Desktop (`claude_desktop_config.json`)

```json
{
  "mcpServers": {
    "local-image-processor": {
      "command": "python",
      "args": ["C:\\AI\\mcp\\image-processor\\server.py"]
    }
  }
}
```

### Generic `mcpServers` block (python)

```json
{
  "mcpServers": {
    "local-image-processor": {
      "command": "python",
      "args": ["C:\\AI\\mcp\\image-processor\\server.py"]
    }
  }
}
```

### Generic `mcpServers` block (uv)

```json
{
  "mcpServers": {
    "local-image-processor": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "C:\\AI\\mcp\\image-processor",
        "python",
        "server.py"
      ]
    }
  }
}
```

## Project layout

```
image-processor/
├── server.py            # MCPServer instance + entry point
├── requirements.txt
├── README.md
└── tools/
    ├── __init__.py
    ├── io_utils.py      # shared I/O, validation, bulk path resolution, and conversion helpers
    ├── geometry.py      # resize (single & bulk), crop (fixed, smart focal, & bulk), rotate/flip, trim
    ├── tonal.py         # brightness, contrast, auto levels, CLAHE
    ├── color.py         # saturation, color space, white balance, tint
    ├── detail.py        # sharpen, blur, denoise, edge detection, upscaling
    ├── composite.py     # overlay & watermarking (single & bulk), background removal (single & bulk), blend
    ├── encoding.py      # format conversion (single & bulk), metadata strip/inspect
    └── info.py          # image inspection (read-only)
```

## Usage Examples

### Typical Workflow

1. **Inspect** an image:
   `image_info(input_path="photo.jpg")` → returns dimensions, EXIF, ICC profile, etc.

2. **Resize & convert** to a web-friendly format:
   `resize_image(input_path="photo.jpg", output_path="photo.webp", width=1200)`

3. **Adjust tonal balance** and sharpen:
   `auto_levels(input_path="photo.webp", output_path="levelled.webp")`
   `sharpen_image(input_path="levelled.webp", output_path="sharp.webp", percent=80)`

4. **Add a watermark** before sharing:
   `overlay_image(input_path="sharp.webp", output_path="watermarked.webp",
         overlay_path="logo.png", position="bottom-right", opacity=0.6)`

5. **Strip sensitive metadata** (optional but recommended):
   `strip_metadata(input_path="watermarked.webp", output_path="final.webp")`

### Batch Processing Pattern

Chain tools by using the output path of one tool as the input path of the next:

```
photo.jpg → resize_image → resized.webp → auto_levels → processed.webp → strip_metadata → final.webp
```

Each tool writes a file to disk, so the pipeline is explicit and inspectable at every step.

For processing collections of images, use the built-in bulk tools (`bulk_crop`, `bulk_resize`, `bulk_convert`, `bulk_watermark`, `bulk_remove_background`):

```python
# Batch resize a folder of images, convert to WebP, and strip metadata:
bulk_resize(input_paths=["photos/"], max_width=1600, target_format="webp", strip_metadata=True)
# Saves to photos/output/*.webp automatically
```

## Input & Output Conventions

| Aspect              | Detail                                                                                                       |
| ------------------- | ------------------------------------------------------------------------------------------------------------ |
| **Input path(s)**   | Single-image tools accept `input_path: str`. Bulk tools accept `input_paths: list[str]` (files, directories, or glob patterns). |
| **Output path/dir** | Single-image tools accept `output_path: str`. Bulk tools accept an optional `output_dir: str \| None = None`. If omitted, bulk tools default to an `output/` subfolder next to the input files to prevent accidental overwrites. |
| **Directories**     | Output directories are created automatically if they don't already exist.                                    |
| **Overwrites**      | Existing files at the output path are overwritten without warning.                                           |
| **Read-only tools** | `image_info` (from `info.py`) and `get_image_metadata` return data only; they take no `output_path`.         |
| **Relative paths**  | Both relative and absolute paths are supported.                                                              |

## Supported Formats

| Operation           | Formats                                                                                  |
| ------------------- | ---------------------------------------------------------------------------------------- |
| **Read**            | PNG, JPEG, WEBP, TIFF, BMP, GIF, ICO, PPM, TGA, and others supported by Pillow & OpenCV. |
| **Write / Convert** | PNG, JPEG, WEBP, TIFF, BMP (controlled via the output file extension).                   |

Quality and compression settings vary by format — see individual tool descriptions for details.

## Error Handling

Anticipated failures (missing files, unsupported formats, out-of-range parameters) raise `ToolError`, so MCP clients receive `is_error=True` with a human-readable message instead of a raw traceback. Unhandled exceptions will propagate as standard MCP errors.

## Troubleshooting

| Problem                              | Likely Cause                                    | Fix                                                                   |
| ------------------------------------ | ----------------------------------------------- | --------------------------------------------------------------------- |
| Tool fails with file-not-found       | `input_path` is wrong or the file doesn't exist | Verify the path is correct and the file exists on disk.               |
| File locked / in use                 | Another process has the file open               | Close the file in other applications or pick a different output path. |
| `ToolError: Unknown resample filter` | Invalid resample name                           | Use one of: `lanczos`, `bicubic`, `bilinear`, `nearest`, `hamming`.   |
| Memory pressure on large images      | Very high-resolution files (hundreds of MP)     | Process at a lower resolution first, or increase available RAM.       |
| `cuDNN is unavailable...` warning    | `onnxruntime-gpu` installed without cuDNN DLLs  | Safe to ignore (server falls back to CPU). To enable GPU: `pip install nvidia-cudnn-cu12 nvidia-cublas-cu12`. |
| Server exits unexpectedly            | Python version < 3.13                           | Upgrade to Python 3.13+.                                              |

## Configuration Options

| Option               | Default      | Description                                                                                |
| -------------------- | ------------ | ------------------------------------------------------------------------------------------ |
| `MAX_PIXELS` env var | `50_000_000` | Maximum allowed pixel count per image (DoS protection). Set to a higher value for very large images. |
| `MAX_GRID_MULTIPLIER` env var | `6` | Multiplier cap for grid canvas pixel limits (scales canvas limit with image count up to 6× / 300 MP by default). |
| `UPSCALER_MODEL_DIR` env var | `models/` (project root) | Local directory for storing pre-trained Real-ESRGAN and SwinIR ONNX super-resolution models. |
| `U2NET_HOME` env var | `models/` (project root) | Local directory for storing the rembg U-2-Net background removal ONNX model. |

## License

MIT License — see [LICENSE](LICENSE) for details.

This project is provided as-is without warranty.
