"""Color tools: saturation, color-space conversion, white balance, tint."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from mcp.server.mcpserver.exceptions import ToolError
from PIL import Image

from server import mcp
from tools.io_utils import (
    FORMAT_BY_EXT,
    WB_SCALE,
    _parse_hex,
    _rejoin_alpha,
    _split_alpha,
    load_image,
    save_image,
)


@mcp.tool()
def adjust_saturation(input_path: str, output_path: str, factor: float) -> str:
    """Scale an image's color saturation in HSV space.

    Args:
        input_path: Path to the source image.
        output_path: Destination path; extension sets the saved format.
        factor: Saturation multiplier. 1.0 = original, 0.0 = fully
            desaturated (grayscale), <1.0 less saturated, >1.0 more
            saturated. Must be >= 0.
    """
    factor = float(factor)
    if factor < 0:
        raise ToolError(f"factor must be >= 0, got {factor}")
    _, img = load_image(input_path)

    # Preserve alpha by splitting it off first (the original code converted to
    # RGB unconditionally, silently destroying transparency).
    base, alpha = _split_alpha(img)

    hsv = base.convert("HSV")
    h, s, v = hsv.split()
    if factor == 0.0:
        s = Image.new("L", s.size, 0)
    else:
        s = s.point(lambda x: min(255, int(x * factor)))  # type: ignore[arg-type]
    result = Image.merge("HSV", (h, s, v)).convert("RGB")
    save_image(_rejoin_alpha(result, alpha), output_path)
    return f"Saturation x{factor} -> {output_path}"


@mcp.tool()
def convert_color_space(input_path: str, output_path: str, target_space: str) -> str:
    """Convert an image to a named color space.

    Args:
        input_path: Path to the source image.
        output_path: Destination path; extension sets the saved format.
        target_space: One of "RGB", "GRAY", "HSV", "LAB", "CMYK" (case-
            insensitive; "GREY" accepted as GRAY). GRAY saves single-channel;
            HSV/LAB/CMYK stored as multi-channel bands.
    """
    key = target_space.strip().upper()
    if key == "GREY":
        key = "GRAY"
    if key not in ("RGB", "GRAY", "HSV", "LAB", "CMYK"):
        raise ToolError(
            f"Unknown target_space '{target_space}'. "
            "Choose one of: RGB, GRAY, HSV, LAB, CMYK"
        )
    _, img = load_image(input_path)

    # PIL uses "L" for single-channel grayscale, not "GRAY".
    target_mode = "L" if key == "GRAY" else key

    # Multi-channel modes (CMYK, LAB, HSV) are incompatible with JPEG.
    multi_channel_modes = {"CMYK", "LAB", "HSV"}
    ext_fmt = FORMAT_BY_EXT.get(Path(output_path).suffix.lower())
    if key in multi_channel_modes and ext_fmt == "JPEG":
        raise ToolError(
            f"Target color space '{key}' requires a lossless format (PNG, TIFF). "
            f"JPEG does not support {key} mode."
        )

    save_image(img.convert(target_mode), output_path)
    return f"Converted to {key} -> {output_path}"


@mcp.tool()
def adjust_white_balance(
    input_path: str,
    output_path: str,
    temperature_shift: int = 0,
    tint_shift: int = 0,
) -> str:
    """Adjust an image's white balance by shifting color temperature and tint.

    Args:
        input_path: Path to the source image.
        output_path: Destination path; extension sets the saved format.
        temperature_shift: Color temperature shift in -100..100. Positive
            warms the image (boosts red, reduces blue); negative cools it
            (boosts blue, reduces red). Default 0 (no change).
        tint_shift: Tint shift in -100..100. Positive shifts toward magenta
            (reduces green); negative shifts toward green (boosts green).
            Default 0 (no change).
    """
    temperature_shift = int(temperature_shift)
    tint_shift = int(tint_shift)
    if not -100 <= temperature_shift <= 100:
        raise ToolError(
            f"temperature_shift must be in -100..100, got {temperature_shift}"
        )
    if not -100 <= tint_shift <= 100:
        raise ToolError(f"tint_shift must be in -100..100, got {tint_shift}")
    if temperature_shift == 0 and tint_shift == 0:
        _, img = load_image(input_path)
        save_image(img, output_path)
        return f"White balance unchanged (0, 0) -> {output_path}"

    _, img = load_image(input_path)
    base, alpha = _split_alpha(img)
    arr = np.asarray(base, dtype=np.float64)
    # Temperature: positive = warm (R up, B down); negative = cool (B up, R down).
    r_delta = temperature_shift * WB_SCALE
    b_delta = -temperature_shift * WB_SCALE
    # Tint: positive = magenta (G down); negative = green (G up).
    g_delta = -tint_shift * WB_SCALE
    arr[..., 0] += r_delta
    arr[..., 1] += g_delta
    arr[..., 2] += b_delta
    out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    save_image(_rejoin_alpha(out, alpha), output_path)
    return (
        f"White balance (temp {temperature_shift:+d}, tint {tint_shift:+d}) "
        f"-> {output_path}"
    )


@mcp.tool()
def color_tint(
    input_path: str,
    output_path: str,
    hex_color: str,
    strength: float = 0.3,
) -> str:
    """Apply a solid color tint over an image.

    Blends the image with the given color: result = image * (1 - strength)
    + tint * strength.

    Args:
        input_path: Path to the source image.
        output_path: Destination path; extension sets the saved format.
        hex_color: Tint color as #RRGGBB or #RGB (e.g. "#FF8800").
        strength: Blend strength in 0..1. 0.0 = original image,
            1.0 = solid tint color. Default 0.3.
    """
    strength = float(strength)
    if not 0.0 <= strength <= 1.0:
        raise ToolError(f"strength must be in 0..1, got {strength}")
    rgb = _parse_hex(hex_color)
    _, img = load_image(input_path)
    base, alpha = _split_alpha(img)
    base = base.convert("RGB")
    tint = Image.new("RGB", base.size, rgb)
    if strength == 0.0:
        out = base
    elif strength == 1.0:
        out = tint
    else:
        out = Image.blend(base, tint, strength)
    save_image(_rejoin_alpha(out, alpha), output_path)
    return f"Tint {hex_color} @ {strength:.2f} -> {output_path}"
