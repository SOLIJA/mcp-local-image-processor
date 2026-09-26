"""File encoding & metadata tools: format conversion, metadata stripping,
and metadata inspection.
"""

from __future__ import annotations

from pathlib import Path

from mcp.server.mcpserver.exceptions import ToolError

from server import mcp
from tools.io_utils import (
    FORMAT_BY_EXT,
    load_image,
    normalize_format,
    resolve_bulk_inputs,
    resolve_output,
    resolve_output_dir,
    save_image,
)

# Formats supported by convert_format.
CONVERTIBLE_FORMATS = {"PNG", "JPEG", "WEBP", "TIFF", "BMP"}

# Pixel mode -> bits per pixel (color depth).
DEPTH_BY_MODE: dict[str, int] = {
    "1": 1,
    "L": 8,
    "LA": 16,
    "I": 32,
    "I;16": 16,
    "P": 8,
    "RGB": 24,
    "RGBA": 32,
    "CMYK": 32,
    "YCbCr": 24,
    "LAB": 24,
    "HSV": 24,
}


def _exif_to_plain(exif) -> dict[str, object]:
    """Convert a PIL Exif object to a JSON-serializable dict of tag values.

    Tag IDs are mapped to human-readable names where known (via
    ``PIL.ExifTags.TAGS``); unknown tags fall back to their numeric ID as a
    string. Values are coerced to str/int/float/list so the result is safe to
    serialize.
    """
    try:
        from PIL.ExifTags import TAGS
    except ImportError:  # pragma: no cover - ExifTags ships with Pillow
        TAGS = {}

    result: dict[str, object] = {}
    for tag_id, value in exif.items():
        name = TAGS.get(tag_id, str(tag_id))
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        elif isinstance(value, tuple):
            value = [
                v.decode("utf-8", errors="replace") if isinstance(v, bytes) else v
                for v in value
            ]
        elif isinstance(value, (int, float, str)):
            pass
        else:
            value = str(value)
        result[name] = value
    return result


@mcp.tool()
def convert_format(
    input_path: str, output_path: str, target_format: str, quality: int = 85
) -> str:
    """Convert an image to a different file format.

    Supported target formats: PNG, JPEG, WEBP, TIFF, BMP (case-insensitive).
    The output_path extension must agree with target_format (e.g. .jpg or
    .jpeg for JPEG); a mismatch raises an error rather than silently writing
    mismatched data.

    Args:
        input_path: Path to the source image.
        output_path: Destination path; its extension must match target_format.
        target_format: One of "PNG", "JPEG", "WEBP", "TIFF", "BMP".
        quality: Lossy quality 1-100 for JPEG/WEBP (ignored for lossless
            formats). Default 85.
    """
    fmt = str(target_format).strip().upper()
    if fmt not in CONVERTIBLE_FORMATS:
        raise ToolError(
            f"Unsupported target_format '{target_format}'. "
            f"Supported: {', '.join(sorted(CONVERTIBLE_FORMATS))}"
        )
    quality = int(quality)
    if not 1 <= quality <= 100:
        raise ToolError(f"quality must be between 1 and 100, got {quality}")

    resolved = resolve_output(output_path)
    ext_fmt = FORMAT_BY_EXT.get(resolved.path.suffix.lower())
    if ext_fmt is None:
        raise ToolError(
            f"Unsupported output extension '{resolved.path.suffix}'. "
            f"Supported: {', '.join(sorted(FORMAT_BY_EXT))}"
        )
    if ext_fmt != fmt:
        raise ToolError(
            f"Output extension '{resolved.path.suffix}' implies format "
            f"{ext_fmt}, but target_format is {fmt}. Use a matching extension "
            f"(e.g. .jpg for JPEG) or a different target_format."
        )

    _, img = load_image(input_path)
    save_image(img, str(resolved.path), fmt=fmt, quality=quality)
    return f"Converted to {fmt} (quality {quality}) -> {resolved.path}"


@mcp.tool()
def strip_metadata(input_path: str, output_path: str) -> str:
    """Save an image with all metadata (EXIF, IPTC, GPS, etc.) removed.

    The pixel data and image mode are preserved exactly; only the metadata
    containers are dropped by making a copy of the image and clearing its
    ``info`` dictionary (which holds EXIF, ICC profile hints, and other
    Pillow-side metadata).  This is more faithful than the previous approach
    which copied pixel-by-pixel via ``putdata()``.

    Args:
        input_path: Path to the source image.
        output_path: Destination path; extension sets the saved format.
    """
    _, img = load_image(input_path)
    clean = img.copy()
    clean.info.clear()
    save_image(clean, output_path)
    return f"Metadata stripped -> {output_path}"


@mcp.tool()
def get_image_metadata(input_path: str) -> dict:
    """Return image metadata as a plain, JSON-serializable dictionary.

    Returns:
        dict with keys: width, height, format, mode, color_depth (bits per
        pixel), file_size_bytes, and exif (a dict of EXIF tag name -> value,
        or an empty dict when the image has no EXIF data).
    """
    p, img = load_image(input_path)

    exif_dict: dict[str, object] = {}
    try:
        raw_exif = img.getexif()
        if raw_exif:
            exif_dict = _exif_to_plain(raw_exif)
    except Exception:  # noqa: BLE001 - EXIF parsing must never fail the tool
        exif_dict = {}

    return {
        "width": img.width,
        "height": img.height,
        "format": img.format or "UNKNOWN",
        "mode": img.mode,
        "color_depth": DEPTH_BY_MODE.get(img.mode, 8),
        "file_size_bytes": p.stat().st_size,
        "exif": exif_dict,
    }


@mcp.tool()
def bulk_convert(
    input_paths: list[str],
    output_dir: str | None = None,
    target_format: str = "webp",
    quality: int = 85,
    strip_metadata: bool = True,
) -> str:
    """Convert multiple images in bulk to a target format with quality and metadata options.

    Args:
        input_paths: List of file paths, directory path, or glob pattern string.
        output_dir: Destination directory where converted images are saved. If omitted,
            defaults to an 'output' subfolder next to the input images.
        target_format: Target format name: 'webp', 'jpeg', 'png', 'tiff', or 'bmp'. Default 'webp'.
        quality: Lossy quality 1-100 for JPEG/WEBP (ignored for lossless formats). Default 85.
        strip_metadata: If True (default), strips EXIF, GPS, and camera metadata.
    """
    fmt_name, ext = normalize_format(target_format)
    if fmt_name not in CONVERTIBLE_FORMATS:
        raise ToolError(
            f"Unsupported target_format '{target_format}'. Supported: {', '.join(sorted(CONVERTIBLE_FORMATS))}"
        )
    quality = int(quality)
    if not 1 <= quality <= 100:
        raise ToolError(f"quality must be between 1 and 100, got {quality}")

    resolved_paths = resolve_bulk_inputs(input_paths)
    out_dir = resolve_output_dir(output_dir, input_paths=resolved_paths)

    successes: list[str] = []
    failures: list[str] = []

    for p in resolved_paths:
        try:
            _, img = load_image(str(p))
            out_file = out_dir / (p.stem + ext)
            save_image(
                img,
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
            f"All {len(failures)} images failed in bulk_convert: {'; '.join(failures)}"
        )

    res = f"Bulk converted {len(successes)}/{len(resolved_paths)} images to {fmt_name} in '{out_dir}'"
    if failures:
        res += f" ({len(failures)} failed: {', '.join(failures)})"
    return res

