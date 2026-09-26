# AGENTS.md

This file provides guidance when working with code in this repository.

## Project Overview

**local-image-processor** is an offline [Model Context Protocol](https://modelcontextprotocol.io) (MCP) server for local image processing. It exposes 34 tools across geometry/transformations, tonal adjustments, color operations, detail filters, compositing/watermarking, file encoding/metadata, and inspection — all running strictly on the local machine with no external APIs or cloud services.

Built on MCP SDK v2 (`mcp.server.mcpserver.MCPServer`), Pillow, OpenCV, and NumPy. Requires Python 3.13+.

## Commands

```sh
# Install dependencies (CPU default)
pip install -r requirements.txt

# Install dependencies with NVIDIA GPU acceleration (CUDA + cuDNN)
pip install -r requirements-gpu.txt

# Run the MCP server (stdio transport)
python server.py

# Run all tests
pytest

# Run a specific test file
pytest tests/test_tools.py

# Run a specific test class or function
pytest tests/test_tools.py::TestResizeImage::test_width_only

# Run with verbose output
pytest -v
```

## Architecture

### Entry Point & Server Registration

`server.py` creates the single `MCPServer` instance (`mcp`) and is the only place it's instantiated. Tool modules register their functions onto this instance at **import time** via the `@mcp.tool()` decorator.

A critical detail: when run as `python server.py`, the module executes as `__main__`. The line `sys.modules["server"] = sys.modules[__name__]` creates an alias so that tool submodules' `from server import mcp` resolves to the same instance — otherwise Python would treat `server` and `__main__` as two distinct modules, each with its own `mcp`, resulting in a server with zero registered tools.

Tool registration happens in `server.py` via explicit submodule imports:
```python
from tools import (color, composite, detail, encoding, geometry, info, tonal)
```

### Tool Module Organization (`tools/`)

Each category module is self-contained and registers multiple tools:

| Module | Tools | Purpose |
|---|---|---|
| `geometry.py` | resize_image, crop_image, smart_crop, bulk_crop, bulk_resize, rotate_flip, trim_whitespace, image_grid | Dimension changes, manual & smart cropping, bulk crop & resize, rotation, border removal, grid layouts |
| `tonal.py` | adjust_brightness, adjust_contrast, auto_levels, histogram_equalization, normalize | Exposure, tonal adjustments, and normalization |
| `color.py` | adjust_saturation, convert_color_space, adjust_white_balance, color_tint | Color manipulation |
| `detail.py` | sharpen_image, blur_image, denoise_image, edge_detection, image_difference, upscale_image | Filters, enhancement, difference analysis, and AI super-resolution |
| `composite.py` | overlay_image, overlay_text, bulk_watermark, remove_background, bulk_remove_background, image_blend | Watermarking (single & bulk), background removal (single & bulk), and compositing |
| `encoding.py` | convert_format, bulk_convert, strip_metadata, get_image_metadata | Format conversion (single & bulk) and metadata |
| `info.py` | image_info | Read-only image inspection |

### Shared Utilities (`tools/io_utils.py`)

All tool modules depend on `io_utils.py` for:

- **Path resolution & validation**: `resolve_input()`, `resolve_output()` — validate existence, prevent path traversal when roots are configured
- **Image I/O**: `load_image()`, `load_image_with_size_check()`, `save_image()` — handle opening, size-limit checks (via `MAX_PIXELS` env var), and saving with format/quality control
- **Color space conversions**: `to_rgb()`, `to_rgba()`, `to_lum8()`, `cv_bgr_from()`, `pil_from_cv_bgr()` — bridge between Pillow and OpenCV representations
- **Format constants**: `FORMAT_BY_EXT`, `QUALITY_FORMATS`, `KNOWN_FORMATS`

### Error Handling Pattern

All anticipated failures raise `ToolError` (from `mcp.server.mcpserver.exceptions`). This ensures MCP clients receive structured error responses (`is_error=True` with a human-readable message) instead of raw tracebacks. Unhandled exceptions propagate as standard MCP errors.

## Key Patterns & Conventions

### Tool Function Signature

Every tool follows this pattern:
```python
@mcp.tool()
def tool_name(input_path: str, output_path: str, ...params...) -> str:
    """Docstring with description and Args section."""
    _, img = load_image(input_path)   # Load + validate
    # ... process img ...
    save_image(result, output_path)    # Save with format inference from extension
    return f"Result message -> {output_path}"
```

- `input_path` / `output_path` are strings (relative or absolute paths supported)
- Bulk tools (`bulk_crop`, `bulk_resize`, `bulk_convert`, `bulk_watermark`, `bulk_remove_background`) take `input_paths: list[str]` and an optional `output_dir: str | None = None` (defaulting to an `output/` subfolder next to the input files if omitted)
- Output directories are created automatically if they don't exist
- The output file extension determines the save format (`.png`, `.jpg`, `.webp`, etc.)
- Existing files at the output path are overwritten without warning

### Adding a New Tool

1. Decide which category module it belongs to (geometry, tonal, color, detail, composite, encoding, or info)
2. Import shared utilities from `tools.io_utils` as needed
3. Decorate the function with `@mcp.tool()` — registration is automatic at import time
4. Use `load_image(input_path)` to load and validate input; use `save_image(img, output_path)` to write output
5. Raise `ToolError` for any anticipated failures (invalid params, missing files, etc.)
6. Document the tool's parameters in the docstring `Args:` section — this is what MCP clients see

### Adding a New Category Module

1. Create `tools/<name>.py` with `from server import mcp` and `from tools.io_utils import ...`
2. Register tools with `@mcp.tool()` decorators
3. Add the module name to `tools/__init__.py`'s `__all__` list
4. Add an explicit import in `server.py`'s `from tools import (...)` block

### PIL ↔ OpenCV Interop

Tools needing OpenCV (e.g., denoising, CLAHE) convert via `io_utils`:
- PIL RGB → OpenCV BGR: `cv_bgr_from(image)` returns `np.ndarray`
- OpenCV BGR → PIL RGB: `pil_from_cv_bgr(arr)` returns `Image.Image`

## Configuration

| Option | Default | Description |
|---|---|---|
| `MAX_PIXELS` env var | `50_000_000` | Maximum image pixel count (DoS protection). Increase for very large images. |
| `MAX_GRID_MULTIPLIER` env var | `6` | Multiplier cap for grid canvas pixel limits (scales canvas limit with image count up to 6× / 300 MP by default). |
| `UPSCALER_MODEL_DIR` env var | `models/` (project root) | Local directory where ONNX upscaler weights are stored and cached. |
| `U2NET_HOME` env var | `models/` (project root) | Local directory where rembg U-2-Net weights are stored and cached. |
| `_DEFAULT_INPUT_ROOT` / `_DEFAULT_OUTPUT_ROOT` | `None` | In `io_utils.py`, set these to restrict file access to specific directories (path traversal prevention) |

## Testing

Tests in `tests/` import tool functions directly — no running MCP server is needed. They use pytest with temporary files (`tempfile.NamedTemporaryFile`) and assert on output file existence, dimensions, and content correctness. Run with `pytest`.
