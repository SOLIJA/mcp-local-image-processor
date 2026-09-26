"""Compositing & watermarking tools: image overlays, text overlays, background removal, and image blending."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import cv2
from mcp.server.mcpserver.exceptions import ToolError
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from server import mcp
from tools.io_utils import (
    _parse_hex,
    cv_bgra_from,
    load_image,
    normalize_format,
    pil_from_cv_bgra,
    resolve_bulk_inputs,
    resolve_output_dir,
    save_image,
    to_rgb,
    to_rgba,
)

# Supported overlay anchor positions.
_POSITIONS = (
    "top-left",
    "top-center",
    "top-right",
    "center",
    "bottom-left",
    "bottom-center",
    "bottom-right",
    "left",
    "right",
)

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Background removal (rembg / U-2-Net)
# ---------------------------------------------------------------------------

_REMBG_SESSION: Any = None


def _get_rembg_session() -> Any:
    """Lazily load and cache the rembg ONNX session singleton with GPU/CPU auto-detection."""
    global _REMBG_SESSION
    if _REMBG_SESSION is None:
        try:
            import onnxruntime as ort
            import rembg
        except ImportError:
            raise ToolError(
                "The 'rembg' package is required for AI-based background removal. "
                "Install it with: pip install \"rembg[cpu]\" (or pip install -r requirements-gpu.txt for CUDA GPU)"
            )
        try:
            import numpy as np

            models_dir = Path(__file__).resolve().parent.parent / "models"
            if "U2NET_HOME" not in os.environ and (models_dir / "u2net.onnx").exists():
                os.environ["U2NET_HOME"] = str(models_dir)

            if hasattr(os, "add_dll_directory"):
                for pkg in ("nvidia.cudnn", "nvidia.cublas"):
                    try:
                        mod = __import__(pkg, fromlist=["__file__"])
                        bin_dir = Path(mod.__file__).parent / "bin"
                        if bin_dir.exists():
                            os.add_dll_directory(str(bin_dir))
                    except Exception:
                        pass

            available_providers = ort.get_available_providers()
            session = None
            if "CUDAExecutionProvider" in available_providers:
                try:
                    cuda_sess = rembg.new_session(
                        "u2net",
                        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                    )
                    # Dry run to verify CUDA and cuDNN libraries are functional
                    dummy = np.zeros((1, 3, 320, 320), dtype=np.float32)
                    cuda_sess.inner_session.run(
                        None, {cuda_sess.inner_session.get_inputs()[0].name: dummy}
                    )
                    session = cuda_sess
                except Exception:
                    session = None

            if session is None:
                session = rembg.new_session("u2net", providers=["CPUExecutionProvider"])

            _REMBG_SESSION = session
        except Exception as exc:
            raise ToolError(
                f"Failed to initialize rembg session: {exc}. "
                "If running offline, verify the model is pre-downloaded to ~/.u2net/u2net.onnx"
            ) from exc
    return _REMBG_SESSION


@mcp.tool()
def remove_background(
    input_path: str,
    output_path: str,
    method: str = "ai",
    margin: int = 10,
    refine_edge: bool = True,
    key_color: str = "#00FF00",
    tolerance: int = 30,
) -> str:
    """Remove the background from an image, replacing it with transparency.

    Produces an image with an alpha channel suitable for compositing onto
    new backgrounds. Two methods available:

    - ``"ai"``: ML-based segmentation (U-2-Net via ``rembg``). High quality,
      works on arbitrary backgrounds. Supports CPU and CUDA GPU.
    - ``"chroma"``: Color distance keying in RGB space. Fast, zero extra dependencies,
      works on uniform color backgrounds (green screen, blue screen, white product backgrounds, etc.).

    Args:
        input_path: Path to the source image.
        output_path: Destination path (must be .png, .webp, .tif, or .tiff).
        method: Segmentation method — ``"ai"`` or ``"chroma"``. Default ``"ai"``.
        margin: Edge feathering / softening radius in pixels (0–50). Default 10.
        refine_edge: Apply alpha matting edge refinement (ai method only). Default True.
        key_color: Hex color to remove for chroma method (e.g. "#00FF00", "#FFFFFF", "#000000"). Default "#00FF00".
        tolerance: Color distance tolerance (1–150) for chroma method. Higher removes more color variance. Default 30.
    """
    valid_methods = ("ai", "chroma")
    if method not in valid_methods:
        raise ToolError(f"method must be one of: {', '.join(valid_methods)}")

    margin = int(margin)
    if not 0 <= margin <= 50:
        raise ToolError(f"margin must be between 0 and 50, got {margin}")

    tolerance = int(tolerance)
    if not 1 <= tolerance <= 150:
        raise ToolError(f"tolerance must be between 1 and 150, got {tolerance}")

    suffix = Path(output_path).suffix.lower()
    if suffix not in (".png", ".webp", ".tif", ".tiff"):
        raise ToolError(
            f"Output must be a format supporting transparency (.png, .webp, .tif, .tiff), got '{suffix}'"
        )

    _, img = load_image(input_path)

    if method == "ai":
        result = _remove_bg_ai(img, margin, refine_edge)
    else:
        result = _remove_bg_chroma(img, margin, key_color, tolerance)

    save_image(result, output_path)
    return f"Removed background via {method} (margin {margin}px) -> {output_path}"


def _remove_bg_ai(img: Image.Image, margin: int, refine_edge: bool) -> Image.Image:
    """AI-based background removal using rembg (U-2-Net)."""
    import numpy as np
    import rembg

    session = _get_rembg_session()
    rgb = to_rgb(img)
    result = rembg.remove(
        rgb,
        session=session,
        alpha_matting=refine_edge,
    )

    alpha = result.getchannel("A")
    if margin > 0:
        alpha = alpha.filter(ImageFilter.GaussianBlur(radius=margin / 2.0))

    # Preserve any transparency already present in the source image
    if "A" in img.getbands():
        orig_alpha = np.asarray(img.getchannel("A"))
        alpha = Image.fromarray(np.minimum(orig_alpha, np.asarray(alpha)), mode="L")

    result.putalpha(alpha)
    return result


def _remove_bg_chroma(
    img: Image.Image, margin: int, key_color: str, tolerance: int
) -> Image.Image:
    """Chroma-key background removal using Euclidean color distance."""
    import numpy as np

    rgb_img = to_rgb(img)
    arr = np.asarray(rgb_img, dtype=np.float32)

    target_rgb = np.array(_parse_hex(key_color), dtype=np.float32)

    # Compute Euclidean distance per pixel from target color
    dist = np.linalg.norm(arr - target_rgb, axis=-1)

    # Binary mask: background pixels (dist <= tolerance) -> 0, foreground -> 255
    alpha_arr = np.where(dist <= tolerance, 0, 255).astype(np.uint8)

    # Preserve any transparency already present in the source image
    if "A" in img.getbands():
        orig_a = np.asarray(img.getchannel("A"))
        alpha_arr = np.minimum(orig_a, alpha_arr)

    alpha_channel = Image.fromarray(alpha_arr, mode="L")
    if margin > 0:
        alpha_channel = alpha_channel.filter(
            ImageFilter.GaussianBlur(radius=margin / 2.0)
        )

    result = rgb_img.convert("RGBA")
    result.putalpha(alpha_channel)
    return result


def _position_xy(
    bw: int,
    bh: int,
    ow: int,
    oh: int,
    position: str,
    offset_x: int,
    offset_y: int,
) -> tuple[int, int]:
    """Compute the top-left (x, y) paste point for ``position``.

    Offsets push the overlay inward from the nearest edge(s). The result is
    clamped so the overlay's top-left corner stays inside the background.
    """
    key = position.strip().lower()
    if key not in _POSITIONS:
        raise ToolError(
            f"Unknown position '{position}'. Choose one of: {', '.join(_POSITIONS)}"
        )
    # Horizontal anchor.
    if key.endswith("left"):
        x = offset_x
    elif key.endswith("right"):
        x = bw - ow - offset_x
    elif "center" in key or key == "center":
        x = (bw - ow) // 2
    else:  # "left" / "right" edge anchors (vertically centered)
        x = offset_x if key == "left" else bw - ow - offset_x
    # Vertical anchor.
    if key.startswith("top"):
        y = offset_y
    elif key.startswith("bottom"):
        y = bh - oh - offset_y
    else:  # center / left / right -> vertically centered
        y = (bh - oh) // 2
    # Clamp so the overlay corner stays within the background.
    x = max(0, min(x, bw - 1))
    y = max(0, min(y, bh - 1))
    return x, y


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Return a TrueType font at ``size``, falling back to Pillow's default.

    Uses platform-aware font candidates (see ``tools.io_utils._font_candidates``).
    """
    from tools.io_utils import _font_candidates

    for path in _font_candidates():
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    # Last resort: Pillow's built-in bitmap font (size hint is best-effort).
    _logger.warning(
        "No TrueType font found; falling back to Pillow's default bitmap font. "
        "Text rendering may appear blocky at large sizes."
    )
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _apply_overlay_image(
    bg: Image.Image,
    ov: Image.Image,
    position: str = "bottom-right",
    offset_x: int = 20,
    offset_y: int = 20,
    opacity: float = 1.0,
    scale: float | None = None,
) -> Image.Image:
    """Composite an overlay image onto background with alpha and position support."""
    opacity = float(opacity)
    if not 0.0 <= opacity <= 1.0:
        raise ToolError(f"opacity must be in 0..1, got {opacity}")
    offset_x = int(offset_x)
    offset_y = int(offset_y)
    if offset_x < 0 or offset_y < 0:
        raise ToolError(f"offsets must be >= 0, got ({offset_x}, {offset_y})")

    bg_rgba = to_rgba(bg)
    ov_rgba = to_rgba(ov)

    # Scale the overlay if scale specified or if larger than background
    if scale is not None:
        scale = float(scale)
        if not 0.01 <= scale <= 2.0:
            raise ToolError(f"scale must be between 0.01 and 2.0, got {scale}")
        target_w = max(1, int(bg_rgba.size[0] * scale))
        target_h = max(1, int(ov_rgba.size[1] * (target_w / ov_rgba.size[0])))
        ov_rgba = ov_rgba.resize((target_w, target_h), Image.Resampling.LANCZOS)
    elif ov_rgba.size[0] > bg_rgba.size[0] or ov_rgba.size[1] > bg_rgba.size[1]:
        s = min(bg_rgba.size[0] / ov_rgba.size[0], bg_rgba.size[1] / ov_rgba.size[1])
        new_size = (max(1, int(ov_rgba.size[0] * s)), max(1, int(ov_rgba.size[1] * s)))
        ov_rgba = ov_rgba.resize(new_size, Image.Resampling.LANCZOS)

    # Apply opacity by scaling the overlay's alpha channel.
    if opacity < 1.0:
        a = ov_rgba.getchannel("A").point(lambda x: int(x * opacity))
        ov_rgba.putalpha(a)

    x, y = _position_xy(
        bg_rgba.size[0], bg_rgba.size[1], ov_rgba.size[0], ov_rgba.size[1], position, offset_x, offset_y
    )

    # Composite via a transparent intermediate layer.
    layer = Image.new("RGBA", bg_rgba.size, (0, 0, 0, 0))
    layer.paste(ov_rgba, (x, y))
    return Image.alpha_composite(bg_rgba, layer)


def _apply_overlay_text(
    img: Image.Image,
    text: str,
    position: str = "bottom-right",
    font_size: int = 24,
    font_color: str = "#FFFFFF",
    offset_x: int = 20,
    offset_y: int = 20,
    opacity: float = 1.0,
) -> Image.Image:
    """Render text onto an image with positioning, font, and opacity."""
    if not text:
        raise ToolError("text must be a non-empty string")
    font_size = int(font_size)
    if font_size < 1:
        raise ToolError(f"font_size must be >= 1, got {font_size}")
    offset_x = int(offset_x)
    offset_y = int(offset_y)
    if offset_x < 0 or offset_y < 0:
        raise ToolError(f"offsets must be >= 0, got ({offset_x}, {offset_y})")
    opacity = float(opacity)
    if not 0.0 <= opacity <= 1.0:
        raise ToolError(f"opacity must be in 0..1, got {opacity}")

    rgb = _parse_hex(font_color)
    alpha_val = int(round(255 * opacity))
    img_rgba = to_rgba(img)
    draw = ImageDraw.Draw(img_rgba)
    font = _load_font(font_size)

    # Measure the text block (multiline supported).
    l, t, r, b = draw.multiline_textbbox((0, 0), text, font=font)
    tw, th = r - l, b - t

    x, y = _position_xy(
        img_rgba.size[0], img_rgba.size[1], int(tw), int(th), position, offset_x, offset_y
    )
    x -= l
    y -= t

    if opacity < 1.0:
        layer = Image.new("RGBA", img_rgba.size, (0, 0, 0, 0))
        layer_draw = ImageDraw.Draw(layer)
        layer_draw.multiline_text((x, y), text, font=font, fill=rgb + (alpha_val,))
        return Image.alpha_composite(img_rgba, layer)
    else:
        draw.multiline_text((x, y), text, font=font, fill=rgb + (255,))
        return img_rgba


@mcp.tool()
def overlay_image(
    background_path: str,
    overlay_path: str,
    output_path: str,
    position: str = "bottom-right",
    offset_x: int = 20,
    offset_y: int = 20,
    opacity: float = 1.0,
) -> str:
    """Overlay one image (logo/watermark) onto another with alpha support.

    The overlay is composited over the background at the requested anchor
    position. If the overlay is larger than the background it is scaled down
    to fit first.

    Args:
        background_path: Path to the base image.
        overlay_path: Path to the image to composite on top (transparent
            PNGs recommended for logos/watermarks).
        output_path: Destination path; extension sets the saved format.
        position: Anchor for the overlay. One of: top-left, top-center,
            top-right, center, bottom-left, bottom-center, bottom-right,
            left, right. Default "bottom-right".
        offset_x: Horizontal inset in pixels from the nearest horizontal
            edge. Default 20.
        offset_y: Vertical inset in pixels from the nearest vertical edge.
            Default 20.
        opacity: Overlay opacity in 0..1. 1.0 = fully opaque (original
            alpha), 0.0 = fully transparent. Default 1.0.
    """
    _, bg = load_image(background_path)
    _, ov = load_image(overlay_path)
    out = _apply_overlay_image(
        bg,
        ov,
        position=position,
        offset_x=offset_x,
        offset_y=offset_y,
        opacity=opacity,
    )
    save_image(out, output_path)
    return f"Overlaid {overlay_path} at {position} -> {output_path}"


@mcp.tool()
def overlay_text(
    input_path: str,
    output_path: str,
    text: str,
    position: str = "bottom-right",
    font_size: int = 24,
    font_color: str = "#FFFFFF",
    offset_x: int = 20,
    offset_y: int = 20,
) -> str:
    """Render text onto an image (watermark/caption).

    Uses a system TrueType font when available, falling back to Pillow's
    built-in default font. The text is anchored at ``position`` and inset by
    the offsets.

    Args:
        input_path: Path to the source image.
        output_path: Destination path; extension sets the saved format.
        text: The text string to render.
        position: Anchor for the text block. One of: top-left, top-center,
            top-right, center, bottom-left, bottom-center, bottom-right,
            left, right. Default "bottom-right".
        font_size: Font size in pixels. Default 24.
        font_color: Text color as #RRGGBB or #RGB (e.g. "#FFFFFF").
            Default "#FFFFFF".
        offset_x: Horizontal inset in pixels from the nearest horizontal
            edge. Default 20.
        offset_y: Vertical inset in pixels from the nearest vertical edge.
            Default 20.
    """
    _, img = load_image(input_path)
    out = _apply_overlay_text(
        img,
        text=text,
        position=position,
        font_size=font_size,
        font_color=font_color,
        offset_x=offset_x,
        offset_y=offset_y,
    )
    save_image(out, output_path)
    return f"Overlaid text at {position} -> {output_path}"


@mcp.tool()
def image_blend(
    input_path_1: str,
    input_path_2: str,
    output_path: str,
    alpha: float = 0.5,
    resize: bool = False,
) -> str:
    """Blend two images together with an adjustable alpha weight.

    Unlike ``overlay_image`` (which composites one image at an anchor position),
    this blends pixel-by-pixel across the entire canvas — useful for
    double-exposure, cross-fades, and transparency compositing.

    Args:
        input_path_1: First source image (foreground, weighted by ``alpha``).
        input_path_2: Second source image (background, weighted by ``1-alpha``).
        output_path: Destination path; extension sets the saved format.
        alpha: Weight of image 1 in ``[0.0, 1.0]``. ``0.0`` = fully image 2,
            ``1.0`` = fully image 1. Default 0.5.
        resize: If ``True``, resize image 2 to match image 1's dimensions
            (stretched if aspect ratios differ). Default False.
    """
    try:
        alpha = float(alpha)
    except (TypeError, ValueError):
        raise ToolError(f"alpha must be a valid number, got {alpha!r}")

    if not 0.0 <= alpha <= 1.0:
        raise ToolError(f"alpha must be in 0..1, got {alpha}")

    _, img1 = load_image(input_path_1)
    _, img2 = load_image(input_path_2)

    has_alpha = ("A" in img1.getbands()) or ("A" in img2.getbands())

    img1_rgba = to_rgba(img1)
    img2_rgba = to_rgba(img2)

    if img1_rgba.size != img2_rgba.size:
        if resize:
            img2_rgba = img2_rgba.resize(img1_rgba.size, Image.Resampling.LANCZOS)
        else:
            raise ToolError(
                f"Image dimensions differ ({img1_rgba.size} vs {img2_rgba.size}). "
                f"Set resize=True to auto-resize image 2."
            )

    # Convert to BGRA arrays for OpenCV blending
    bgra1 = cv_bgra_from(img1_rgba)
    bgra2 = cv_bgra_from(img2_rgba)

    # Blend: result = img1 * alpha + img2 * (1 - alpha)
    blended = cv2.addWeighted(bgra1, alpha, bgra2, 1.0 - alpha, 0)

    result = pil_from_cv_bgra(blended)
    if not has_alpha:
        result = result.convert("RGB")
    save_image(result, output_path)
    return f"Blended images (alpha={alpha}) -> {output_path}"


@mcp.tool()
def bulk_watermark(
    input_paths: list[str],
    output_dir: str | None = None,
    watermark_image_path: str | None = None,
    text: str | None = None,
    position: str = "bottom-right",
    opacity: float = 0.7,
    offset_x: int = 20,
    offset_y: int = 20,
    scale: float | None = None,
    font_size: int = 36,
    font_color: str = "#ffffff",
    format: str | None = None,
    quality: int = 85,
    strip_metadata: bool = False,
) -> str:
    """Apply an image or text watermark across multiple images in bulk.

    Args:
        input_paths: List of file paths, directory path, or glob pattern string.
        output_dir: Destination directory. If omitted, defaults to an 'output' subfolder next to the inputs.
        watermark_image_path: Path to an image file (e.g. logo) to overlay.
        text: Text to render as a watermark (if watermark_image_path is not used).
        position: Anchor position: 'top-left', 'top-center', 'top-right', 'center',
            'bottom-left', 'bottom-center', 'bottom-right', 'left', 'right'. Default 'bottom-right'.
        opacity: Watermark opacity between 0.0 and 1.0 (default 0.7).
        offset_x: Horizontal inset in pixels (default 20).
        offset_y: Vertical inset in pixels (default 20).
        scale: Optional scale factor (0.01 to 2.0) of watermark relative to base image width.
        font_size: Font size for text watermark (default 36).
        font_color: Hex color string for text watermark (default '#ffffff').
        format: Optional target format (e.g. 'webp', 'jpeg', 'png'). If omitted, preserves format.
        quality: Lossy quality 1-100 for JPEG/WEBP (default 85).
        strip_metadata: If True, strips all EXIF/GPS/IPTC metadata. Default False.
    """
    if watermark_image_path is None and (text is None or not text.strip()):
        raise ToolError("Either watermark_image_path or text must be provided.")

    resolved_paths = resolve_bulk_inputs(input_paths)
    out_dir = resolve_output_dir(output_dir, input_paths=resolved_paths)

    target_fmt_name: str | None = None
    target_ext: str | None = None
    if format is not None and str(format).strip():
        target_fmt_name, target_ext = normalize_format(format)

    ov_img: Image.Image | None = None
    if watermark_image_path:
        _, ov_img = load_image(watermark_image_path)

    successes: list[str] = []
    failures: list[str] = []

    for p in resolved_paths:
        try:
            _, bg_img = load_image(str(p))
            if ov_img is not None:
                watermarked = _apply_overlay_image(
                    bg_img,
                    ov_img,
                    position=position,
                    offset_x=offset_x,
                    offset_y=offset_y,
                    opacity=opacity,
                    scale=scale,
                )
            else:
                watermarked = _apply_overlay_text(
                    bg_img,
                    text=text,  # type: ignore[arg-type]
                    position=position,
                    font_size=font_size,
                    font_color=font_color,
                    offset_x=offset_x,
                    offset_y=offset_y,
                    opacity=opacity,
                )
            out_file = out_dir / (p.stem + target_ext if target_ext else p.name)
            save_image(
                watermarked,
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
            f"All {len(failures)} images failed in bulk_watermark: {'; '.join(failures)}"
        )

    res = f"Bulk watermarked {len(successes)}/{len(resolved_paths)} images to '{out_dir}'"
    if failures:
        res += f" ({len(failures)} failed: {', '.join(failures)})"
    return res


@mcp.tool()
def bulk_remove_background(
    input_paths: list[str],
    output_dir: str | None = None,
    method: str = "ai",
    margin: int = 10,
    refine_edge: bool = True,
    key_color: str = "#00FF00",
    tolerance: int = 30,
    format: str = "png",
    quality: int = 85,
    strip_metadata: bool = True,
) -> str:
    """Remove background from multiple images in bulk (AI U-2-Net or chroma keying).

    Args:
        input_paths: List of file paths, single directory, or glob pattern string.
        output_dir: Destination directory. If omitted, defaults to an 'output' subfolder next to the inputs.
        method: Segmentation method — 'ai' (U-2-Net) or 'chroma'. Default 'ai'.
        margin: Edge feathering / softening radius in pixels (0–50). Default 10.
        refine_edge: Apply alpha matting edge refinement (ai method only). Default True.
        key_color: Hex color to remove for chroma method (e.g. '#00FF00', '#FFFFFF'). Default '#00FF00'.
        tolerance: Color distance tolerance (1–150) for chroma method. Default 30.
        format: Target format supporting transparency: 'png', 'webp', or 'tiff'. Default 'png'.
        quality: Lossy quality 1-100 for WEBP (ignored for PNG). Default 85.
        strip_metadata: If True (default), strips all metadata.
    """
    valid_methods = ("ai", "chroma")
    if method not in valid_methods:
        raise ToolError(f"method must be one of: {', '.join(valid_methods)}")

    fmt_name, ext = normalize_format(format)
    if fmt_name not in ("PNG", "WEBP", "TIFF"):
        raise ToolError(
            f"format for remove_background must support transparency (PNG, WEBP, TIFF), got {format}"
        )

    resolved_paths = resolve_bulk_inputs(input_paths)
    out_dir = resolve_output_dir(output_dir, input_paths=resolved_paths)

    successes: list[str] = []
    failures: list[str] = []

    for p in resolved_paths:
        try:
            _, img = load_image(str(p))
            if method == "ai":
                result = _remove_bg_ai(img, margin, refine_edge)
            else:
                result = _remove_bg_chroma(img, margin, key_color, tolerance)
            out_file = out_dir / (p.stem + ext)
            save_image(
                result,
                str(out_file),
                fmt=fmt_name,
                quality=quality,
                strip_metadata=strip_metadata,
            )
            successes.append(p.name)
        except Exception as exc:
            failures.append(f"{p.name}: {exc}")

    if not successes:
        raise ToolError(
            f"All {len(failures)} images failed in bulk_remove_background: {'; '.join(failures)}"
        )

    res = f"Bulk removed backgrounds for {len(successes)}/{len(resolved_paths)} images via {method} to '{out_dir}'"
    if failures:
        res += f" ({len(failures)} failed: {', '.join(failures)})"
    return res


