"""Geometry & transformation tools: resize, crop, rotate/flip, trim."""

from __future__ import annotations

import glob
import logging
from pathlib import Path

import cv2
from mcp.server.mcpserver.exceptions import ToolError
import numpy as np
from PIL import Image, ImageChops

from server import mcp
from tools.io_utils import (
    ANGLE_EPSILON,
    FORMAT_BY_EXT,
    MAX_GRID_MULTIPLIER,
    MAX_PIXELS,
    _parse_hex,
    load_image,
    normalize_format,
    resolve_bulk_inputs,
    resolve_output_dir,
    save_image,
    to_rgba,
)

_logger = logging.getLogger(__name__)

RESAMPLES = {"lanczos", "bicubic", "bilinear", "nearest", "hamming"}
FOCUS_MODES = {"auto", "face", "saliency", "edge", "center"}



def _resample(name: str) -> int:
    try:
        return getattr(Image.Resampling, name.upper())
    except AttributeError as exc:
        raise ToolError(
            f"Unknown resample filter '{name}'. "
            f"Choose one of: {', '.join(sorted(RESAMPLES))}"
        ) from exc


def _process_resize_image(
    img: Image.Image,
    width: int | None = None,
    height: int | None = None,
    maintain_aspect_ratio: bool = True,
    resample: str = "lanczos",
) -> Image.Image:
    """Resize an image with aspect ratio and resample filter support."""
    filter = _resample(resample)
    ow, oh = img.size

    if width is not None and width < 1:
        raise ToolError(f"width must be >= 1, got {width}")
    if height is not None and height < 1:
        raise ToolError(f"height must be >= 1, got {height}")
    if width is None and height is None:
        raise ToolError("At least one of width or height is required (or both).")

    if width is not None and height is not None:
        if maintain_aspect_ratio:
            scale = min(width / ow, height / oh)  # type: ignore[operator]
            new_w = max(1, round(ow * scale))
            new_h = max(1, round(oh * scale))
        else:
            new_w, new_h = width, height
    elif width is not None:
        new_w = width
        new_h = max(1, round(oh * (width / ow)))  # type: ignore[operator]
    else:  # height only
        new_h = height
        new_w = max(1, round(ow * (height / oh)))  # type: ignore[operator]

    return img.resize((new_w, new_h), filter)  # type: ignore[arg-type]


@mcp.tool()
def resize_image(
    input_path: str,
    output_path: str,
    width: int | None = None,
    height: int | None = None,
    maintain_aspect_ratio: bool = True,
    resample: str = "lanczos",
) -> str:
    """Resize an image's pixel dimensions.

    If both width and height are given and maintain_aspect_ratio is True,
    the image is fit *within* that bounding box (no upscaling beyond the
    box on the constraining axis; one axis may be smaller). If only one of
    width/height is given, the other is derived from the original aspect
    ratio. If maintain_aspect_ratio is False, the image is stretched to the
    exact (width, height). At least one of width or height is required.

    Args:
        input_path: Path to the source image.
        output_path: Destination path; extension sets the saved format.
        width: Target width in pixels (1 or greater). Optional.
        height: Target height in pixels (1 or greater). Optional.
        maintain_aspect_ratio: Preserve aspect ratio (fit within the given
            box). Default True.
        resample: Resampling filter: "lanczos" (default), "bicubic",
            "bilinear", "nearest", or "hamming".
    """
    _, img = load_image(input_path)
    resized = _process_resize_image(
        img,
        width=width,
        height=height,
        maintain_aspect_ratio=maintain_aspect_ratio,
        resample=resample,
    )
    save_image(resized, output_path)
    return f"Resized to {resized.size[0]}x{resized.size[1]} -> {output_path}"


def _process_fixed_crop_image(
    img: Image.Image,
    x: int,
    y: int,
    width: int,
    height: int,
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Crop a fixed bounding box from an image with boundary clamping."""
    if width < 1:
        raise ToolError(f"width must be >= 1, got {width}")
    if height < 1:
        raise ToolError(f"height must be >= 1, got {height}")

    iw, ih = img.size

    # Check if the crop box is entirely outside image bounds (both positive and negative).
    if (
        int(x) >= iw
        or int(y) >= ih
        or int(x) + int(width) <= 0
        or int(y) + int(height) <= 0
    ):
        raise ToolError(
            f"Crop region ({x},{y},{width},{height}) is entirely outside "
            f"image bounds ({iw}x{ih})"
        )

    x0 = max(0, min(int(x), iw - 1))
    y0 = max(0, min(int(y), ih - 1))
    x1 = max(x0 + 1, min(int(x) + int(width), iw))
    y1 = max(y0 + 1, min(int(y) + int(height), ih))
    if x1 - x0 < 1 or y1 - y0 < 1:
        raise ToolError(
            f"Crop box ({x},{y},{width},{height}) is outside the {iw}x{ih} image."
        )
    cropped = img.crop((x0, y0, x1, y1))
    return cropped, (x0, y0, x1, y1)


@mcp.tool()
def crop_image(
    input_path: str,
    output_path: str,
    x: int,
    y: int,
    width: int,
    height: int,
) -> str:
    """Crop a rectangular region from an image (bounding box in pixels).

    The region is clamped to the image boundaries; coordinates may be
    negative or extend past the edges, which are clamped rather than
    erroring.

    Args:
        input_path: Path to the source image.
        output_path: Destination path.
        x: Left edge of the crop box in pixels (may be negative).
        y: Top edge of the crop box in pixels (may be negative).
        width: Crop width in pixels (> 0).
        height: Crop height in pixels (> 0).
    """
    _, img = load_image(input_path)
    cropped, (x0, y0, x1, y1) = _process_fixed_crop_image(img, x, y, width, height)
    save_image(cropped, output_path)
    return f"Cropped to {x1 - x0}x{y1 - y0} at ({x0},{y0}) -> {output_path}"



@mcp.tool()
def rotate_flip(
    input_path: str,
    output_path: str,
    angle: float = 0.0,
    flip_h: bool = False,
    flip_v: bool = False,
) -> str:
    """Rotate and/or reflect an image.

    ``angle`` is counter-clockwise in degrees. Exact multiples of 90 use a
    fast pixel-perfect rotation; arbitrary angles rotate with transparent
    padding (for images with alpha) or an expand bbox. Reflections are
    applied after rotation.

    Args:
        input_path: Path to the source image.
        output_path: Destination path.
        angle: Rotation in degrees, counter-clockwise. 0 = none, 90/180/270
            are special-cased for speed. Any float in (-360, 360) works.
        flip_h: Mirror horizontally (left-right). Default False.
        flip_v: Mirror vertically (top-bottom). Default False.
    """
    _, img = load_image(input_path)

    angle = float(angle) % 360.0
    # Normalize angles very close to 360 back to 0.
    if abs(angle - 360.0) < ANGLE_EPSILON:
        angle = 0.0

    if abs(angle) > ANGLE_EPSILON:
        for target in (90.0, 180.0, 270.0):
            if abs(angle - target) < ANGLE_EPSILON:
                angle = float(target)
                break
        if angle in (90.0, 180.0, 270.0):
            method = int(round(angle / 90))
            img = (
                img.transpose(Image.Transpose.ROTATE_90)
                if method == 1
                else (
                    img.transpose(Image.Transpose.ROTATE_180)
                    if method == 2
                    else img.transpose(Image.Transpose.ROTATE_270)
                )
            )
        else:
            resample = Image.Resampling.BICUBIC
            fill: tuple[int, ...] | None = None
            if img.mode == "RGBA":
                fill = (0, 0, 0, 0)
            elif img.mode == "LA":
                fill = (0, 0)
            img = img.rotate(angle, resample=resample, expand=True, fillcolor=fill)

    if flip_h:
        img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if flip_v:
        img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

    save_image(img, output_path)
    return (
        f"Rotated {float(angle)}deg, flip_h={flip_h}, flip_v={flip_v} -> {output_path}"
    )


@mcp.tool()
def trim_whitespace(
    input_path: str,
    output_path: str,
    tolerance: int = 10,
) -> str:
    """Trim solid-color borders (e.g. white/transparent margins).

    Detects the bounding box of non-background pixels and crops to it.
    Background is the color at each edge pixel, generalized by
    ``tolerance`` per channel. For images with an alpha channel, fully
    transparent pixels are always treated as background.

    Args:
        input_path: Path to the source image.
        output_path: Destination path.
        tolerance: Per-channel distance from the edge color considered
            "background" (0-255; default 10). Higher trims more.
    """
    if not 0 <= tolerance <= 255:
        raise ToolError(f"tolerance must be between 0 and 255, got {tolerance}")

    _, img = load_image(input_path)

    # Check for full transparency before any conversion.
    if img.mode in ("RGBA", "LA", "PA"):
        alpha = img.getchannel("A")
        if alpha.getextrema() == (0, 0):
            raise ToolError("Image is fully transparent; nothing to trim.")

    # Normalize to RGBA so we have a consistent multi-channel mode.
    if img.mode not in ("RGBA", "RGB"):
        img = img.convert("RGBA")

    # If RGBA, zero out invisible RGB channels (alpha == 0) to avoid false margins
    # from dirty RGB data under transparent pixels.
    if img.mode == "RGBA":
        r, g, b, a = img.split()
        mask = a.point(lambda v: 255 if v > 0 else 0)
        r = ImageChops.multiply(r, mask)
        g = ImageChops.multiply(g, mask)
        b = ImageChops.multiply(b, mask)
        img = Image.merge("RGBA", (r, g, b, a))

    # Determine the background color from the top-left corner pixel,
    # treating fully transparent corners as transparent background.
    corner = img.getpixel((0, 0))  # type: ignore[index]
    if len(corner) == 4 and corner[3] == 0:
        bg = (0, 0, 0, 0)
    else:
        bg = tuple(int(c) for c in corner)

    # Build a solid-background image and compute the per-pixel difference.
    bg_img = Image.new(img.mode, img.size, bg)
    diff = ImageChops.difference(img, bg_img)

    # Compute maximum per-channel difference so all channels (including alpha) contribute.
    bands = diff.split()
    diff_max = bands[0]
    for band in bands[1:]:
        diff_max = ImageChops.lighter(diff_max, band)

    tol = int(tolerance)
    if tol > 0:
        diff_gray = diff_max.point(lambda v: 255 if v > tol else 0)
    else:
        diff_gray = diff_max

    bbox = diff_gray.getbbox()
    if bbox is None:
        raise ToolError(
            f"Entire image matches background within tolerance {tolerance}; nothing to trim."
        )

    trimmed = img.crop(bbox)
    save_image(trimmed, output_path)
    return (
        f"Trimmed to {trimmed.size[0]}x{trimmed.size[1]} "
        f"(bbox {bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}) -> {output_path}"
    )


@mcp.tool()
def image_grid(
    input_paths: list[str],
    output_path: str,
    cols: int = 3,
    padding: int = 10,
    bg_color: str = "#000000",
    resize_to_fit: bool = True,
    cell_width: int | None = None,
    cell_height: int | None = None,
) -> str:
    """Arrange multiple images into a grid/collage layout.

    Useful for before/after comparisons, galleries, sprite sheets,
    and visual reviews.

    Args:
        input_paths: List of image paths to arrange (or a single image path).
        output_path: Destination path; extension sets the saved format.
        cols: Number of columns in the grid. Default 3.
        padding: Spacing between cells in pixels. Default 10.
        bg_color: Background color as ``#RRGGBB``, ``#RGB``, or ``"transparent"``.
            Default ``"#000000"``.
        resize_to_fit: If ``True`` and cell dimensions are not fixed, resize all
            images to match the largest dimensions (note: this stretches images if
            aspect ratios differ; set False to preserve natural sizes). Default ``True``.
        cell_width: Fixed cell width in pixels (requires ``cell_height``).
        cell_height: Fixed cell height in pixels (requires ``cell_width``).
    """
    if isinstance(input_paths, str):
        input_paths = [input_paths]
    else:
        input_paths = list(input_paths)

    if not input_paths:
        raise ToolError("At least one input image is required.")

    cols = int(cols)
    if cols < 1:
        raise ToolError(f"cols must be >= 1, got {cols}")

    padding = int(padding)
    if padding < 0:
        raise ToolError(f"padding must be >= 0, got {padding}")

    if (cell_width is None) != (cell_height is None):
        raise ToolError("Both cell_width and cell_height must be provided together.")

    if cell_width is not None and cell_height is not None:
        cell_width = int(cell_width)
        cell_height = int(cell_height)
        if cell_width < 1 or cell_height < 1:
            raise ToolError(
                f"cell dimensions must be >= 1, got ({cell_width}, {cell_height})"
            )
    elif not resize_to_fit and cols > len(input_paths):
        raise ToolError(
            f"cols ({cols}) cannot exceed number of images ({len(input_paths)}) "
            f"when resize_to_fit=False"
        )

    if bg_color.strip().lower() in ("transparent", "none"):
        bg_rgba = (0, 0, 0, 0)
    else:
        rgb = _parse_hex(bg_color)
        bg_rgba = rgb + (255,)

    images = []
    for p in input_paths:
        _, img = load_image(p)
        images.append(to_rgba(img))

    n = len(images)
    rows = (n + cols - 1) // cols

    col_widths = [0] * cols
    row_heights = [0] * rows

    if cell_width is not None and cell_height is not None:
        col_widths = [cell_width] * cols
        row_heights = [cell_height] * rows
    elif resize_to_fit:
        max_w = max(img.size[0] for img in images)
        max_h = max(img.size[1] for img in images)
        col_widths = [max_w] * cols
        row_heights = [max_h] * rows
    else:
        for i, img in enumerate(images):
            c = i % cols
            r = i // cols
            col_widths[c] = max(col_widths[c], img.size[0])
            row_heights[r] = max(row_heights[r], img.size[1])

    canvas_w = sum(col_widths) + (cols - 1) * padding
    canvas_h = sum(row_heights) + (rows - 1) * padding

    total_canvas_pixels = canvas_w * canvas_h
    effective_max_pixels = MAX_PIXELS * min(len(images), MAX_GRID_MULTIPLIER)
    if total_canvas_pixels > effective_max_pixels:
        raise ToolError(
            f"Grid canvas is too large ({canvas_w}x{canvas_h} = {total_canvas_pixels:,} pixels); "
            f"maximum allowed for {len(images)} images is {effective_max_pixels:,} pixels "
            f"(to prevent DoS)."
        )

    canvas = Image.new("RGBA", (canvas_w, canvas_h), bg_rgba)

    col_x = [0] * cols
    curr_x = 0
    for c in range(cols):
        col_x[c] = curr_x
        curr_x += col_widths[c] + padding

    row_y = [0] * rows
    curr_y = 0
    for r in range(rows):
        row_y[r] = curr_y
        curr_y += row_heights[r] + padding

    for i, img in enumerate(images):
        c = i % cols
        r = i // cols
        cw = col_widths[c]
        ch = row_heights[r]

        if cell_width is not None and cell_height is not None:
            # Scale down proportionally if larger than fixed cell; otherwise preserve size
            if img.size[0] > cw or img.size[1] > ch:
                scale = min(cw / img.size[0], ch / img.size[1])
                new_size = (
                    max(1, int(img.size[0] * scale)),
                    max(1, int(img.size[1] * scale)),
                )
                img = img.resize(new_size, Image.Resampling.LANCZOS)
        elif resize_to_fit:
            if img.size != (cw, ch):
                img = img.resize((cw, ch), Image.Resampling.LANCZOS)

        # Center image within its assigned cell, performing true alpha compositing
        ix = col_x[c] + (cw - img.size[0]) // 2
        iy = row_y[r] + (ch - img.size[1]) // 2
        canvas.alpha_composite(img, (ix, iy))

    save_image(canvas, output_path)
    return f"Created {cols}x{rows} grid ({len(images)} images, {padding}px padding) -> {output_path}"


# ---------------------------------------------------------------------------
# Smart Crop & Bulk Crop Helpers
# ---------------------------------------------------------------------------


def _parse_aspect_ratio(
    aspect_str: str | None, width: int | None, height: int | None
) -> float:
    """Parse aspect ratio from string or dimension pair."""
    if width is not None and height is not None:
        if width < 1 or height < 1:
            raise ToolError(f"width and height must be >= 1, got ({width}, {height})")
        return width / height
    if aspect_str is None or not str(aspect_str).strip():
        return 1.0
    s = str(aspect_str).strip()
    try:
        if ":" in s:
            num, den = s.split(":", 1)
            ratio = float(num) / float(den)
        elif "/" in s:
            num, den = s.split("/", 1)
            ratio = float(num) / float(den)
        else:
            ratio = float(s)
        if ratio <= 0:
            raise ValueError
        return ratio
    except Exception as exc:
        raise ToolError(
            f"Invalid aspect_ratio '{aspect_str}'. Expected format like '1:1', '16:9', '4:3', '0.75'."
        ) from exc


def _detect_focal_bbox(
    img: Image.Image,
    focus_mode: str = "auto",
) -> tuple[int, int, int, int] | None:
    """Detect subject focal bounding box based on focus_mode."""
    if focus_mode not in FOCUS_MODES:
        raise ToolError(
            f"Unknown focus_mode '{focus_mode}'. Choose one of: {', '.join(sorted(FOCUS_MODES))}"
        )

    if focus_mode == "center":
        return None

    # 1. Face detection
    if focus_mode in ("auto", "face"):
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            face_cascade = cv2.CascadeClassifier(cascade_path)
            if not face_cascade.empty():
                np_rgb = np.array(img.convert("RGB"))
                gray = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2GRAY)
                faces = face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20)
                )
                if len(faces) > 0:
                    fx0 = min(int(f[0]) for f in faces)
                    fy0 = min(int(f[1]) for f in faces)
                    fx1 = max(int(f[0] + f[2]) for f in faces)
                    fy1 = max(int(f[1] + f[3]) for f in faces)
                    # Add 25% extra headroom above the face for hair/forehead
                    headroom = int((fy1 - fy0) * 0.25)
                    fy0 = max(0, fy0 - headroom)
                    return (fx0, fy0, fx1, fy1)
        except Exception as exc:
            _logger.debug("Face detection exception: %s", exc)

        if focus_mode == "face":
            return None

    # 2. AI Saliency Detection (U-2-Net via rembg)
    if focus_mode in ("auto", "saliency"):
        try:
            import rembg
            from tools.composite import _get_rembg_session

            sess = _get_rembg_session()
            mask = rembg.remove(img, session=sess, only_mask=True)
            mask_arr = np.array(mask)
            active_ys, active_xs = np.where(mask_arr > 30)
            if len(active_xs) > 30:
                sx0 = int(np.min(active_xs))
                sy0 = int(np.min(active_ys))
                sx1 = int(np.max(active_xs)) + 1
                sy1 = int(np.max(active_ys)) + 1
                return (sx0, sy0, sx1, sy1)
        except Exception as exc:
            _logger.debug("Saliency detection exception: %s", exc)

        if focus_mode == "saliency":
            return None

    # 3. Edge & detail density fallback
    if focus_mode in ("auto", "edge"):
        try:
            np_rgb = np.array(img.convert("RGB"))
            gray = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            active_ys, active_xs = np.where(edges > 0)
            if len(active_xs) > 30:
                ex0 = int(np.percentile(active_xs, 5))
                ey0 = int(np.percentile(active_ys, 5))
                ex1 = int(np.percentile(active_xs, 95)) + 1
                ey1 = int(np.percentile(active_ys, 95)) + 1
                if ex1 > ex0 and ey1 > ey0:
                    return (ex0, ey0, ex1, ey1)
        except Exception as exc:
            _logger.debug("Edge detection exception: %s", exc)

    return None


def _calculate_smart_crop_box(
    img_size: tuple[int, int],
    target_aspect: float,
    focal_bbox: tuple[int, int, int, int] | None,
    padding_percent: float = 0.1,
    tight_frame: bool = False,
) -> tuple[int, int, int, int]:
    """Calculate the optimal crop box bounding rectangle."""
    iw, ih = img_size
    has_focal = focal_bbox is not None

    if has_focal and focal_bbox is not None:
        bx0, by0, bx1, by1 = focal_bbox
        bw = bx1 - bx0
        bh = by1 - by0
        padx = int(bw * padding_percent)
        pady = int(bh * padding_percent)
        bx0 = max(0, bx0 - padx)
        by0 = max(0, by0 - pady)
        bx1 = min(iw, bx1 + padx)
        by1 = min(ih, by1 + pady)
        bw = bx1 - bx0
        bh = by1 - by0
        scx = (bx0 + bx1) / 2.0
        scy = (by0 + by1) / 2.0
    else:
        scx = iw / 2.0
        scy = ih / 2.0
        bx0, by0, bx1, by1 = 0, 0, iw, ih
        bw, bh = 0, 0

    # Max possible window matching target_aspect inside (iw, ih)
    if (iw / ih) >= target_aspect:
        max_ch = ih
        max_cw = min(iw, max(1, int(round(ih * target_aspect))))
    else:
        max_cw = iw
        max_ch = min(ih, max(1, int(round(iw / target_aspect))))

    if tight_frame and has_focal and bw > 0 and bh > 0:
        need_cw = max(bw, int(round(bh * target_aspect)))
        need_ch = max(bh, int(round(bw / target_aspect)))
        cw = min(max_cw, max(need_cw, int(round(need_ch * target_aspect))))
        ch = min(max_ch, max(need_ch, int(round(need_cw / target_aspect))))
        if cw / ch > target_aspect:
            cw = max(1, int(round(ch * target_aspect)))
        else:
            ch = max(1, int(round(cw / target_aspect)))
        cw = min(max_cw, max(1, cw))
        ch = min(max_ch, max(1, ch))
    else:
        cw = max_cw
        ch = max_ch

    x0 = int(round(scx - cw / 2.0))
    y0 = int(round(scy - ch / 2.0))

    x0 = max(0, min(x0, iw - cw))
    y0 = max(0, min(y0, ih - ch))

    if has_focal:
        if cw >= bw:
            if x0 > bx0:
                x0 = bx0
            elif x0 + cw < bx1:
                x0 = bx1 - cw
            x0 = max(0, min(x0, iw - cw))

        if ch >= bh:
            if y0 > by0:
                y0 = by0
            elif y0 + ch < by1:
                y0 = by1 - ch
            y0 = max(0, min(y0, ih - ch))

    x1 = min(iw, x0 + cw)
    y1 = min(ih, y0 + ch)
    return (x0, y0, x1, y1)


def _process_smart_crop_image(
    img: Image.Image,
    aspect_ratio: str = "1:1",
    width: int | None = None,
    height: int | None = None,
    focus_mode: str = "auto",
    padding_percent: float = 0.1,
    tight_frame: bool = False,
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Crop an image smartly centering on the detected subject."""
    if not (0.0 <= padding_percent <= 1.0):
        raise ToolError(
            f"padding_percent must be between 0.0 and 1.0, got {padding_percent}"
        )
    if width is not None and width < 1:
        raise ToolError(f"width must be >= 1, got {width}")
    if height is not None and height < 1:
        raise ToolError(f"height must be >= 1, got {height}")

    target_aspect = _parse_aspect_ratio(aspect_ratio, width, height)
    focal_bbox = _detect_focal_bbox(img, focus_mode=focus_mode)
    crop_box = _calculate_smart_crop_box(
        img.size,
        target_aspect=target_aspect,
        focal_bbox=focal_bbox,
        padding_percent=padding_percent,
        tight_frame=tight_frame,
    )

    cropped = img.crop(crop_box)

    if width is not None and height is not None:
        cropped = cropped.resize((width, height), Image.Resampling.LANCZOS)
    elif width is not None:
        target_h = max(1, int(round(width / target_aspect)))
        cropped = cropped.resize((width, target_h), Image.Resampling.LANCZOS)
    elif height is not None:
        target_w = max(1, int(round(height * target_aspect)))
        cropped = cropped.resize((target_w, height), Image.Resampling.LANCZOS)

    return cropped, crop_box


@mcp.tool()
def smart_crop(
    input_path: str,
    output_path: str,
    aspect_ratio: str = "1:1",
    width: int | None = None,
    height: int | None = None,
    focus_mode: str = "auto",
    padding_percent: float = 0.1,
    tight_frame: bool = False,
) -> str:
    """Intelligently crop an image centering on the focal subject without cutting it off.

    Automatically detects the focal point of the image (faces, salient foreground
    objects via AI segmentation, or high-detail edge regions) and positions the crop
    window to keep the subject centered and preserved.

    Args:
        input_path: Path to the source image.
        output_path: Destination path; extension determines the saved format.
        aspect_ratio: Target aspect ratio as 'W:H', 'W/H', or float (e.g. '1:1',
            '16:9', '9:16', '4:3', '4:5'). Default '1:1'. Ignored if both width
            and height are specified.
        width: Optional output width in pixels. If specified, the cropped result
            is resized to this width.
        height: Optional output height in pixels. If specified, the cropped result
            is resized to this height.
        focus_mode: Subject detection strategy:
            - 'auto' (default): Detects human faces first; if none found, detects
              salient subjects via AI segmentation; falls back to edge density/center.
            - 'face': Prioritizes human face detection.
            - 'saliency': Uses AI salient object detection (U-2-Net).
            - 'edge': Uses OpenCV Canny edge & gradient density (fast, purely algorithmic).
            - 'center': Geometric center crop.
        padding_percent: Buffer padding margin around the detected subject (0.0 to 1.0).
            Default 0.1 (10%).
        tight_frame: If True, crops tightly around the subject plus padding (scaled to
            the aspect ratio). If False (default), uses the largest crop window fitting
            within the image while framing the subject.
    """
    _, img = load_image(input_path)
    cropped, (x0, y0, x1, y1) = _process_smart_crop_image(
        img,
        aspect_ratio=aspect_ratio,
        width=width,
        height=height,
        focus_mode=focus_mode,
        padding_percent=padding_percent,
        tight_frame=tight_frame,
    )
    save_image(cropped, output_path)
    return (
        f"Smart cropped ({focus_mode}) to {cropped.size[0]}x{cropped.size[1]} "
        f"(window [{x0},{y0},{x1},{y1}]) -> {output_path}"
    )


@mcp.tool()
def bulk_crop(
    input_paths: list[str],
    output_dir: str | None = None,
    aspect_ratio: str = "1:1",
    width: int | None = None,
    height: int | None = None,
    focus_mode: str = "auto",
    padding_percent: float = 0.1,
    tight_frame: bool = False,
    x: int | None = None,
    y: int | None = None,
    crop_width: int | None = None,
    crop_height: int | None = None,
    format: str | None = None,
    quality: int = 85,
    strip_metadata: bool = False,
) -> str:
    """Crop multiple images in bulk using smart focal centering or uniform coordinates.

    Processes a batch of images from a list of file paths, directory, or glob pattern,
    saving the cropped results into output_dir.

    If exact crop box coordinates (x, y, crop_width, crop_height) are provided,
    uniform fixed-box cropping is applied to each image. Otherwise, smart focal
    cropping (auto/face/saliency/edge/center) is applied individually to each image.

    Args:
        input_paths: List of file paths, a single directory path containing images,
            or a glob pattern string (e.g. 'images/*.jpg').
        output_dir: Directory where cropped images will be saved. If omitted, defaults
            to an 'output' subfolder next to the input images.
        aspect_ratio: Target aspect ratio for smart cropping ('1:1', '16:9', etc.).
            Default '1:1'.
        width: Optional output width in pixels for smart cropping.
        height: Optional output height in pixels for smart cropping.
        focus_mode: Smart crop strategy: 'auto', 'face', 'saliency', 'edge', 'center'.
            Default 'auto'.
        padding_percent: Padding margin around detected subjects (0.0 to 1.0). Default 0.1.
        tight_frame: If True, tightly frames detected subjects; if False, uses max window.
        x: Optional fixed left coordinate for uniform batch cropping.
        y: Optional fixed top coordinate for uniform batch cropping.
        crop_width: Optional fixed width for uniform batch cropping.
        crop_height: Optional fixed height for uniform batch cropping.
        format: Optional target format (e.g. 'webp', 'jpeg', 'png'). If omitted, preserves format.
        quality: Lossy quality 1-100 for JPEG/WEBP (default 85).
        strip_metadata: If True, strips all EXIF/GPS/IPTC metadata. Default False.
    """
    fixed_params = (x, y, crop_width, crop_height)
    has_any_fixed = any(v is not None for v in fixed_params)
    has_all_fixed = all(v is not None for v in fixed_params)
    if has_any_fixed and not has_all_fixed:
        raise ToolError(
            "To use fixed coordinates in bulk_crop, all of x, y, crop_width, and crop_height must be provided."
        )

    resolved_paths = resolve_bulk_inputs(input_paths)
    out_dir = resolve_output_dir(output_dir, input_paths=resolved_paths)

    target_fmt_name: str | None = None
    target_ext: str | None = None
    if format is not None and str(format).strip():
        target_fmt_name, target_ext = normalize_format(format)

    successes: list[str] = []
    failures: list[str] = []

    for p in resolved_paths:
        try:
            _, img = load_image(str(p))
            if has_all_fixed:
                cropped, _ = _process_fixed_crop_image(
                    img, x=x, y=y, width=crop_width, height=crop_height  # type: ignore[arg-type]
                )
            else:
                cropped, _ = _process_smart_crop_image(
                    img,
                    aspect_ratio=aspect_ratio,
                    width=width,
                    height=height,
                    focus_mode=focus_mode,
                    padding_percent=padding_percent,
                    tight_frame=tight_frame,
                )
            out_file = out_dir / (p.stem + target_ext if target_ext else p.name)
            save_image(
                cropped,
                str(out_file),
                fmt=target_fmt_name,
                quality=quality,
                strip_metadata=strip_metadata,
            )
            successes.append(p.name)
        except Exception as exc:
            failures.append(f"{p.name}: {exc}")

    if not successes:
        raise ToolError(
            f"All {len(failures)} images failed in bulk_crop: {'; '.join(failures)}"
        )

    res = f"Bulk cropped {len(successes)}/{len(resolved_paths)} images to '{out_dir}'"
    if failures:
        res += f" ({len(failures)} failed: {', '.join(failures)})"
    return res


@mcp.tool()
def bulk_resize(
    input_paths: list[str],
    output_dir: str | None = None,
    width: int | None = None,
    height: int | None = None,
    maintain_aspect_ratio: bool = True,
    resample: str = "lanczos",
    format: str | None = None,
    quality: int = 85,
    strip_metadata: bool = False,
) -> str:
    """Resize multiple images in bulk with optional format conversion, quality control, and metadata stripping.

    Args:
        input_paths: List of file paths, single directory, or glob pattern string.
        output_dir: Destination directory. If omitted, defaults to an 'output' subfolder next to the inputs.
        width: Target width in pixels (1 or greater). Optional.
        height: Target height in pixels (1 or greater). Optional.
        maintain_aspect_ratio: Preserve aspect ratio (fit within bounding box). Default True.
        resample: Resampling filter: 'lanczos' (default), 'bicubic', 'bilinear', 'nearest', 'hamming'.
        format: Optional target format (e.g. 'webp', 'jpeg', 'png'). If omitted, preserves format.
        quality: Lossy quality 1-100 for JPEG/WEBP (default 85).
        strip_metadata: If True, strips all EXIF/GPS/IPTC metadata. Default False.
    """
    resolved_paths = resolve_bulk_inputs(input_paths)
    out_dir = resolve_output_dir(output_dir, input_paths=resolved_paths)

    target_fmt_name: str | None = None
    target_ext: str | None = None
    if format is not None and str(format).strip():
        target_fmt_name, target_ext = normalize_format(format)

    successes: list[str] = []
    failures: list[str] = []

    for p in resolved_paths:
        try:
            _, img = load_image(str(p))
            resized = _process_resize_image(
                img,
                width=width,
                height=height,
                maintain_aspect_ratio=maintain_aspect_ratio,
                resample=resample,
            )
            out_file = out_dir / (p.stem + target_ext if target_ext else p.name)
            save_image(
                resized,
                str(out_file),
                fmt=target_fmt_name,
                quality=quality,
                strip_metadata=strip_metadata,
            )
            successes.append(p.name)
        except Exception as exc:
            failures.append(f"{p.name}: {exc}")

    if not successes:
        raise ToolError(
            f"All {len(failures)} images failed in bulk_resize: {'; '.join(failures)}"
        )

    res = f"Bulk resized {len(successes)}/{len(resolved_paths)} images to '{out_dir}'"
    if failures:
        res += f" ({len(failures)} failed: {', '.join(failures)})"
    return res


