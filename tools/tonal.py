"""Tonal & exposure tools: brightness, contrast, auto levels, CLAHE."""

from __future__ import annotations

import cv2
import numpy as np
from mcp.server.mcpserver.exceptions import ToolError
from PIL import Image, ImageEnhance, ImageOps

from server import mcp
from tools.io_utils import (
    CLAHE_TILE_SIZE,
    _check_size,
    _rejoin_alpha,
    _split_alpha,
    cv_bgr_from,
    load_image,
    pil_from_cv_bgr,
    save_image,
)

# Maximum useful multiplier for ImageEnhance.Brightness and .Contrast.
# Values above ~5.0 produce no additional visual change (image is already fully
# white / maximum contrast), so the upper bound of 100 was misleading.
_MAX_ENHANCE_FACTOR = 5.0


@mcp.tool()
def adjust_brightness(input_path: str, output_path: str, factor: float) -> str:
    """Linearly adjust an image's brightness.

    Args:
        input_path: Path to the source image.
        output_path: Destination path; extension sets the saved format.
        factor: Linear brightness multiplier. 1.0 = unchanged, <1.0 darkens,
            >1.0 lightens (0.0 = pure black). Range 0.0–5.0.
    """
    factor = float(factor)
    if factor < 0 or factor > _MAX_ENHANCE_FACTOR:
        raise ToolError(f"factor must be in 0..{_MAX_ENHANCE_FACTOR}, got {factor}")
    _, img = load_image(input_path)
    _check_size(img)
    base, alpha = _split_alpha(img)
    base = ImageEnhance.Brightness(base).enhance(factor)
    save_image(_rejoin_alpha(base, alpha), output_path)
    return f"Brightness x{factor} -> {output_path}"


@mcp.tool()
def adjust_contrast(input_path: str, output_path: str, factor: float) -> str:
    """Linearly adjust an image's contrast around the midpoint.

    Args:
        input_path: Path to the source image.
        output_path: Destination path; extension sets the saved format.
        factor: Contrast multiplier centered on mid-gray. 1.0 = unchanged,
            <1.0 flattens toward gray, >1.0 increases contrast (0.0 = flat
            gray). Range 0.0–5.0.
    """
    factor = float(factor)
    if factor < 0 or factor > _MAX_ENHANCE_FACTOR:
        raise ToolError(f"factor must be in 0..{_MAX_ENHANCE_FACTOR}, got {factor}")
    _, img = load_image(input_path)
    _check_size(img)
    base, alpha = _split_alpha(img)
    base = ImageEnhance.Contrast(base).enhance(factor)
    save_image(_rejoin_alpha(base, alpha), output_path)
    return f"Contrast x{factor} -> {output_path}"


@mcp.tool()
def auto_levels(input_path: str, output_path: str) -> str:
    """Auto-level an image, stretching each channel's dynamic range to 0-255.

    Eliminates washed-out or flat tones by mapping the darkest and brightest
    pixels to black and white per channel.

    Args:
        input_path: Path to the source image.
        output_path: Destination path; extension sets the saved format.
    """
    _, img = load_image(input_path)
    base, alpha = _split_alpha(img)
    base = ImageOps.autocontrast(base, cutoff=0)
    save_image(_rejoin_alpha(base, alpha), output_path)
    return f"Auto-leveled -> {output_path}"


@mcp.tool()
def histogram_equalization(
    input_path: str, output_path: str, clip_limit: float = 2.0
) -> str:
    """Apply CLAHE histogram equalization on the luminance channel.

    Uses Contrast Limited Adaptive Histogram Equalization on the L channel of
    LAB color space for local, noise-limited contrast enhancement.

    Args:
        input_path: Path to the source image.
        output_path: Destination path; extension sets the saved format.
        clip_limit: CLAHE clip limit controlling contrast amplification
            (0 < limit; typical 2-40; higher = more aggressive). Default 2.0.
    """
    clip_limit = float(clip_limit)
    if clip_limit <= 0:
        raise ToolError(f"clip_limit must be > 0, got {clip_limit}")
    _, img = load_image(input_path)
    _check_size(img)
    base, alpha = _split_alpha(img)
    arr = cv_bgr_from(base)
    lab = cv2.cvtColor(arr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=CLAHE_TILE_SIZE)
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    save_image(_rejoin_alpha(pil_from_cv_bgr(bgr), alpha), output_path)
    return f"CLAHE (clip {clip_limit}) -> {output_path}"


@mcp.tool()
def normalize(
    input_path: str,
    output_path: str,
    range_min: float = 0.0,
    range_max: float = 255.0,
    per_channel: bool = False,
    method: str = "minmax",
) -> str:
    """Normalize pixel values to a specified target intensity range.

    Useful for contrast expansion/compression, batch consistency, and image preprocessing.

    Three normalization methods:
    - ``"minmax"``: Linear scaling from the image's min/max to ``[range_min, range_max]``.
    - ``"zscore"``: Standardizes distribution around the mean, mapping ``[-3σ, +3σ]``
      to ``[range_min, range_max]``.
    - ``"percentile"``: Clips to 1st/99th percentiles before minmax scaling to
      suppress outliers.

    Args:
        input_path: Path to source image.
        output_path: Destination path; extension sets the saved format.
        range_min: Target minimum intensity in ``0..255``. Default 0.0.
        range_max: Target maximum intensity in ``0..255``. Default 255.0.
        per_channel: Normalize each channel independently. Default False.
        method: Normalization method — ``"minmax"``, ``"zscore"``, or ``"percentile"``.
            Default "minmax".
    """
    method = method.strip().lower()
    valid_methods = ("minmax", "zscore", "percentile")
    if method not in valid_methods:
        raise ToolError(f"method must be one of: {', '.join(valid_methods)}")

    try:
        range_min = float(range_min)
        range_max = float(range_max)
    except (TypeError, ValueError):
        raise ToolError("range_min and range_max must be valid numbers")

    if not 0.0 <= range_min <= 255.0 or not 0.0 <= range_max <= 255.0:
        raise ToolError(
            f"range bounds must be in 0..255, got [{range_min}, {range_max}]"
        )

    if range_min >= range_max:
        raise ToolError(
            f"range_min ({range_min}) must be < range_max ({range_max})"
        )

    span = range_max - range_min

    _, img = load_image(input_path)
    _check_size(img)
    base, alpha = _split_alpha(img)

    arr = cv_bgr_from(base).astype(np.float64)

    if method == "minmax":
        if per_channel:
            for c in range(arr.shape[2]):
                ch_min = arr[:, :, c].min()
                ch_max = arr[:, :, c].max()
                ch_range = ch_max - ch_min
                if ch_range > 0:
                    arr[:, :, c] = (
                        (arr[:, :, c] - ch_min) / ch_range
                    ) * span + range_min
                else:
                    arr[:, :, c] = range_min
        else:
            arr_min = arr.min()
            arr_max = arr.max()
            arr_range = arr_max - arr_min
            if arr_range > 0:
                arr = ((arr - arr_min) / arr_range) * span + range_min
            else:
                arr.fill(range_min)

    elif method == "zscore":
        if per_channel:
            for c in range(arr.shape[2]):
                ch_mean = arr[:, :, c].mean()
                ch_std = arr[:, :, c].std()
                if ch_std > 0:
                    clamped_z = np.clip((arr[:, :, c] - ch_mean) / ch_std, -3.0, 3.0)
                    arr[:, :, c] = (clamped_z + 3.0) / 6.0 * span + range_min
                else:
                    arr[:, :, c] = (range_min + range_max) / 2.0
        else:
            arr_mean = arr.mean()
            arr_std = arr.std()
            if arr_std > 0:
                clamped_z = np.clip((arr - arr_mean) / arr_std, -3.0, 3.0)
                arr = (clamped_z + 3.0) / 6.0 * span + range_min
            else:
                arr.fill((range_min + range_max) / 2.0)

    elif method == "percentile":
        if per_channel:
            for c in range(arr.shape[2]):
                ch_p1 = np.percentile(arr[:, :, c], 1)
                ch_p99 = np.percentile(arr[:, :, c], 99)
                arr[:, :, c] = np.clip(arr[:, :, c], ch_p1, ch_p99)
                ch_range = ch_p99 - ch_p1
                if ch_range > 0:
                    arr[:, :, c] = (
                        (arr[:, :, c] - ch_p1) / ch_range
                    ) * span + range_min
                else:
                    arr[:, :, c] = range_min
        else:
            p1 = np.percentile(arr, 1)
            p99 = np.percentile(arr, 99)
            arr = np.clip(arr, p1, p99)
            arr_range = p99 - p1
            if arr_range > 0:
                arr = ((arr - p1) / arr_range) * span + range_min
            else:
                arr.fill(range_min)

    out_arr = np.clip(arr, range_min, range_max).round().astype(np.uint8)
    result = pil_from_cv_bgr(out_arr)
    if img.mode == "L":
        result = result.convert("L")
    save_image(_rejoin_alpha(result, alpha), output_path)

    return f"Normalized via {method} ({range_min}–{range_max}) -> {output_path}"
