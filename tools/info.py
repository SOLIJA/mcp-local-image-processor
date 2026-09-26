"""Image inspection tools (no output file produced)."""

from __future__ import annotations

from PIL import ExifTags, Image

from server import mcp

from .io_utils import load_image


@mcp.tool()
def image_info(input_path: str) -> dict:
    """Return metadata about an image file.

    Returns dimensions, mode, format, file size, and (when present) EXIF
    data including orientation, camera make/model, and timestamps.

    Args:
        input_path: Path to the image file to inspect.
    """
    _p, img = load_image(input_path)

    info: dict = {
        "path": str(_p),
        "width": img.width,
        "height": img.height,
        "mode": img.mode,
        "format": img.format,
        "file_size_bytes": _p.stat().st_size,
    }

    # EXIF data (getexif has been in Pillow since 6.0 — no shim needed).
    exif = img.getexif()
    if exif:
        exif_dict: dict = {}
        for tag_id, value in exif.items():
            tag = ExifTags.TAGS.get(tag_id, str(tag_id))
            if isinstance(value, bytes):
                try:
                    value = value.decode("utf-8", errors="replace")
                except Exception:
                    value = repr(value)
            exif_dict[tag] = value
        # Sub-IFDs (e.g. ExifIFD for make/model, GPSInfo for location)
        try:
            exif_ifd = exif.get_ifd(0x8769)  # ExifIFD
            for tag_id, value in exif_ifd.items():
                tag = ExifTags.TAGS.get(tag_id, str(tag_id))
                if isinstance(value, bytes):
                    try:
                        value = value.decode("utf-8", errors="replace")
                    except Exception:
                        value = repr(value)
                exif_dict[tag] = value
        except Exception:
            pass
        if exif_dict:
            info["exif"] = exif_dict

    # ICC profile presence
    icc = img.info.get("icc_profile")
    info["has_icc_profile"] = icc is not None

    # Palette / animation hints
    if img.mode == "P":
        info["palette"] = True
    n_frames = getattr(img, "n_frames", 1)
    if n_frames and n_frames > 1:
        info["n_frames"] = n_frames

    return info
