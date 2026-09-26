"""Shared I/O, validation, and conversion helpers for the image tools.

All helpers raise ``ToolError`` for anticipated failures (missing files,
unsupported formats, bad parameter values) so that MCP clients receive a
clear ``is_error=True`` result with the message instead of a raw traceback.
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from mcp.server.mcpserver.exceptions import ToolError
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Output extension -> PIL format name.
FORMAT_BY_EXT: dict[str, str] = {
    ".png": "PNG",
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".webp": "WEBP",
    ".tif": "TIFF",
    ".tiff": "TIFF",
    ".bmp": "BMP",
    ".gif": "GIF",
    ".tga": "TGA",
}

# Formats that support lossy quality on save.
QUALITY_FORMATS = {"JPEG", "WEBP"}

# Hardcoded known formats (avoids relying on Pillow's runtime registry).
KNOWN_FORMATS = {"PNG", "JPEG", "WEBP", "TIFF", "BMP", "GIF", "TGA"}

# Maximum allowed pixel count to prevent DoS from giant images.
# Configurable via MAX_PIXELS env var (default 50 megapixels).
_MAX_PIXELS_DEFAULT = 50_000_000
MAX_PIXELS: int = int(os.environ.get("MAX_PIXELS", _MAX_PIXELS_DEFAULT))

# Grid canvas multiplier cap (scales canvas limit with image count up to 6x = 300 MP).
# Configurable via MAX_GRID_MULTIPLIER env var (default 6).
_MAX_GRID_MULTIPLIER_DEFAULT = 6
MAX_GRID_MULTIPLIER: int = max(
    1, int(os.environ.get("MAX_GRID_MULTIPLIER", _MAX_GRID_MULTIPLIER_DEFAULT))
)

# White-balance delta scale factor.
WB_SCALE = 0.5

# CLAHE tile grid size.
CLAHE_TILE_SIZE = (8, 8)

# Angle comparison epsilon for rotation checks.
ANGLE_EPSILON = 1e-9

# Overlay resampling filter (high quality; change if speed is preferred).
_OVERLAY_RESAMPLE = Image.Resampling.LANCZOS

# ---------------------------------------------------------------------------
# Platform-aware font candidates for text overlay.
# ---------------------------------------------------------------------------


def _font_candidates() -> tuple[str, ...]:
    """Return platform-appropriate TrueType font paths."""
    system = platform.system()
    if system == "Windows":
        return (
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/calibri.ttf",
        )
    if system == "Darwin":  # macOS
        return (
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/SFNS.ttf",
            "/Library/Fonts/Arial.ttf",
        )
    # Linux / others
    return (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    )


# ---------------------------------------------------------------------------
# ResolvedOutput — encapsulates the output path and any dirs we created.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedOutput:
    """Result of resolving an output path."""

    path: Path
    """The resolved output file path."""
    created_dirs: list[Path]
    """Directories that were created during resolution (empty if none)."""


# ---------------------------------------------------------------------------
# Default roots for input/output sandboxing.
# Set to None to allow any path on the filesystem.
# When a Path is provided, all resolved paths must be within it (prevents traversal).
# ---------------------------------------------------------------------------

_DEFAULT_INPUT_ROOT: Path | None = None
_DEFAULT_OUTPUT_ROOT: Path | None = None


def resolve_input(input_path: str, root: Path | None = None) -> Path:
    """Validate that ``input_path`` exists and is a file; return it as a Path.

    If ``root`` is provided (or ``_DEFAULT_INPUT_ROOT`` is set), the resolved path
    must be within that directory. This prevents path-traversal attacks when the
    server is exposed to untrusted clients.
    """
    original = input_path  # keep for error messages
    p = Path(input_path).expanduser()
    effective_root = root or _DEFAULT_INPUT_ROOT
    if effective_root is not None:
        resolved = p.resolve()
        try:
            resolved.relative_to(effective_root.resolve())
        except ValueError:
            raise ToolError(
                f"Path '{original}' escapes the allowed root {effective_root}"
            )
        p = resolved
    else:
        p = p.resolve()
    if not p.exists():
        raise ToolError(f"Input file not found: {p}")
    if not p.is_file():
        raise ToolError(f"Input path is not a file: {p}")
    return p


def resolve_output(output_path: str, root: Path | None = None) -> ResolvedOutput:
    """Resolve ``output_path`` and ensure its parent directory exists.

    If ``root`` is provided (or ``_DEFAULT_OUTPUT_ROOT`` is set), the resolved path
    must be within that directory. This prevents writing files outside the intended tree.
    """
    original = output_path  # keep for error messages
    p = Path(output_path).expanduser()
    effective_root = root or _DEFAULT_OUTPUT_ROOT
    if effective_root is not None:
        resolved = p.resolve()
        try:
            resolved.relative_to(effective_root.resolve())
        except ValueError:
            raise ToolError(
                f"Path '{original}' escapes the allowed root {effective_root}"
            )
        p = resolved
    else:
        p = p.resolve()
    if p.suffix == "":
        raise ToolError(
            f"Output path has no file extension; add one (e.g. .png, .jpg): {p}"
        )
    # Track which directories we create so we can clean up on failure.
    created: list[Path] = []
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        # Collect any dirs that didn't already exist.
        for parent in p.parents:
            if not parent.exists():
                continue  # pragma: no cover - mkdir ensures they exist
            created.append(parent)
    except OSError as exc:
        raise ToolError(f"Could not create output directory for {p}: {exc}") from exc
    return ResolvedOutput(path=p, created_dirs=created)


def _try_rmdir_parents(output_path: str, created_dirs: list[Path]) -> None:
    """Remove directories that were created by ``resolve_output`` on failure.

    Only removes a directory if it is an ancestor of the output path and
    was not already present before ``resolve_output`` ran.  This avoids
    deleting user-created parent trees.
    """
    for d in reversed(created_dirs):
        try:
            if d.exists() and not any(d.iterdir()):
                d.rmdir()
        except OSError:
            break  # stop at the first non-empty or inaccessible dir


def resolve_output_dir(
    output_dir: str | None = None,
    input_paths: list[Path] | None = None,
    default_dirname: str = "output",
    root: Path | None = None,
) -> Path:
    """Resolve an output directory path, create it if needed, and validate path traversal.

    If output_dir is None or empty, defaults to a `default_dirname` (default 'output')
    subfolder inside the parent directory of the first input path (or current working directory).
    """
    if output_dir is None or not str(output_dir).strip():
        if input_paths and len(input_paths) > 0:
            first = input_paths[0]
            base_dir = first if first.is_dir() else first.parent
            p = (base_dir / default_dirname).resolve()
        else:
            p = (Path.cwd() / default_dirname).resolve()
    else:
        p = Path(output_dir).expanduser()

    effective_root = root or _DEFAULT_OUTPUT_ROOT
    if effective_root is not None:
        resolved = p.resolve()
        try:
            resolved.relative_to(effective_root.resolve())
        except ValueError:
            raise ToolError(
                f"Directory '{p}' escapes the allowed root {effective_root}"
            )
        p = resolved
    else:
        p = p.resolve()
    try:
        p.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ToolError(f"Could not create output directory at {p}: {exc}") from exc
    return p


def resolve_bulk_inputs(
    input_paths: list[str] | str, root: Path | None = None
) -> list[Path]:
    """Expand input paths, directories, or glob patterns into a list of valid image Paths."""
    import glob

    if isinstance(input_paths, str):
        raw_paths = [input_paths]
    else:
        raw_paths = list(input_paths)

    if not raw_paths:
        raise ToolError("At least one input image path or directory is required.")

    resolved: list[Path] = []
    for item in raw_paths:
        p_str = str(item).strip()
        p = Path(p_str)
        if any(char in p_str for char in ("*", "?", "[")):
            matched = glob.glob(p_str)
            for m in matched:
                mp = Path(m)
                if mp.is_file() and mp.suffix.lower() in FORMAT_BY_EXT:
                    resolved.append(resolve_input(str(mp), root=root))
        elif p.is_dir():
            for child in sorted(p.iterdir()):
                if child.is_file() and child.suffix.lower() in FORMAT_BY_EXT:
                    resolved.append(resolve_input(str(child), root=root))
        elif p.is_file():
            resolved.append(resolve_input(p_str, root=root))
        else:
            resolved.append(resolve_input(p_str, root=root))

    if not resolved:
        raise ToolError(f"No valid image files found matching input_paths: {input_paths}")

    return resolved


def normalize_format(target_format: str) -> tuple[str, str]:
    """Validate and normalize a target format string to (PIL_FORMAT, default_ext)."""
    raw = target_format.strip().lower()
    if raw in ("jpeg", "jpg"):
        return "JPEG", ".jpg"
    elif raw == "png":
        return "PNG", ".png"
    elif raw == "webp":
        return "WEBP", ".webp"
    elif raw in ("tiff", "tif"):
        return "TIFF", ".tiff"
    elif raw == "bmp":
        return "BMP", ".bmp"
    elif raw == "gif":
        return "GIF", ".gif"
    elif raw == "tga":
        return "TGA", ".tga"
    else:
        raise ToolError(
            f"Unsupported format '{target_format}'. Supported: JPEG, PNG, WEBP, TIFF, BMP, GIF, TGA"
        )


def _split_alpha(img: Image.Image) -> tuple[Image.Image, Image.Image | None]:
    """Return ``(rgb, alpha)`` with the alpha channel separated for processing.

    If the image has no alpha channel (mode not in ``RGBA``, ``LA``, ``PA``),
    returns the original image and ``None`` for alpha.
    """
    if img.mode in ("RGBA", "LA", "PA"):
        rgba = img.convert("RGBA")
        return rgba.convert("RGB"), rgba.getchannel("A")
    return img, None


def _rejoin_alpha(base: Image.Image, alpha: Image.Image | None) -> Image.Image:
    """Re-attach ``alpha`` to ``base`` if one was supplied."""
    if alpha is None:
        return base
    out = base.convert("RGBA")
    out.putalpha(alpha)
    return out


def _check_size(img: Image.Image) -> None:
    """Raise ``ToolError`` when the image exceeds ``MAX_PIXELS``."""
    pixels = img.width * img.height
    if pixels > MAX_PIXELS:
        raise ToolError(
            f"Image is too large ({img.width}x{img.height} = {pixels:,} pixels); "
            f"maximum allowed is {MAX_PIXELS:,} (to prevent DoS)."
        )


def output_format(output_path: str) -> str:
    """Infer the PIL format name from an output path's extension."""
    fmt = FORMAT_BY_EXT.get(Path(output_path).suffix.lower())
    if fmt is None:
        raise ToolError(
            f"Unsupported output extension '{Path(output_path).suffix}'. "
            f"Supported: {', '.join(sorted(FORMAT_BY_EXT))}"
        )
    return fmt


def load_image(input_path: str) -> tuple[Path, Image.Image]:
    """Open an image, validating that the path exists, pixel count <= MAX_PIXELS, and decodes."""
    p = resolve_input(input_path)
    try:
        with Image.open(p) as img:
            _check_size(img)
            img.load()
            fmt = img.format
            copied = img.copy()
            copied.format = fmt
            return p, copied
    except ToolError:
        raise
    except (OSError, ValueError) as exc:
        raise ToolError(f"Could not open image at {p}: {exc}") from exc


def load_image_with_size_check(input_path: str) -> tuple[Path, Image.Image]:
    """Like ``load_image`` but also checks the pixel count against ``MAX_PIXELS``."""
    return load_image(input_path)


def to_rgb(image: Image.Image) -> Image.Image:
    """Return an RGB copy of ``image``; flattens transparency onto white."""
    if image.mode == "RGB":
        return image.copy()
    if image.mode == "RGBA":
        rgb = Image.new("RGB", image.size, (255, 255, 255))
        rgb.paste(image, mask=image.getchannel("A"))
        return rgb
    return image.convert("RGB")


def to_rgba(image: Image.Image) -> Image.Image:
    """Return an RGBA copy of ``image`` (adds an opaque alpha channel if needed)."""
    if image.mode == "RGBA":
        return image.copy()
    if image.mode == "P":
        return image.convert("RGBA")
    return image.convert("RGBA")


def to_lum8(image: Image.Image) -> Image.Image:
    """Return a grayscale (mode 'L') copy of ``image``."""
    if image.mode == "L":
        return image.copy()
    return image.convert("L")


def save_image(
    image: Image.Image,
    output_path: str,
    fmt: str | None = None,
    quality: int | None = None,
    flatten: bool = False,
    strip_metadata: bool = False,
) -> Path:
    """Save ``image`` to ``output_path``.

    - ``fmt``: explicit PIL format name (overrides extension inference).
    - ``quality``: 1-100 lossy quality for JPEG/WEBP; ignored otherwise.
    - ``flatten``: composite transparency onto white before saving to a
      format without an alpha channel.
    - ``strip_metadata``: remove all EXIF/IPTC/ICC metadata before saving.
    """
    resolved = resolve_output(output_path)
    fmt = fmt or output_format(str(resolved.path))
    if fmt not in KNOWN_FORMATS:
        raise ToolError(f"Unsupported output format: {fmt}")

    save_kwargs: dict = {}
    if fmt in QUALITY_FORMATS and quality is not None:
        if not 1 <= quality <= 100:
            raise ToolError(f"quality must be between 1 and 100, got {quality}")
        save_kwargs["quality"] = int(quality)

    out = image
    if strip_metadata:
        out = out.copy()
        out.info.clear()

    if flatten:
        out = to_rgb(out)
    elif fmt == "JPEG" and out.mode in ("RGBA", "LA", "P"):
        out = to_rgb(out)

    try:
        out.save(resolved.path, format=fmt, **save_kwargs)
    except (OSError, ValueError, KeyError) as exc:
        _try_rmdir_parents(output_path, resolved.created_dirs)
        raise ToolError(f"Could not save image to {resolved.path}: {exc}") from exc
    return resolved.path


def cv_bgr_from(image: Image.Image) -> np.ndarray:
    """Convert an RGB PIL image to an 8-bit BGR OpenCV array.

    Note: This function always produces a 3-channel (BGR) array. Alpha is
    discarded — use ``cv_bgra_from()`` for 4-channel BGRA output.
    """
    return np.asarray(to_rgb(image))[:, :, ::-1].copy()


def cv_bgra_from(image: Image.Image) -> np.ndarray:
    """Convert a PIL image to an 8-bit BGRA OpenCV array (for denoising)."""
    arr = np.asarray(to_rgba(image))
    return arr[:, :, [2, 1, 0, 3]].copy()


def pil_from_cv_bgr(arr: np.ndarray) -> Image.Image:
    """Convert an 8-bit BGR OpenCV array to an RGB PIL image."""
    return Image.fromarray(arr[:, :, ::-1].copy())


def pil_from_cv_bgra(arr: np.ndarray) -> Image.Image:
    """Convert an 8-bit BGRA OpenCV array to an RGBA PIL image."""
    return Image.fromarray(arr[:, :, [2, 1, 0, 3]].copy())


# ---------------------------------------------------------------------------
# Shared hex-color parser (used by color.py and composite.py).
# ---------------------------------------------------------------------------


def _parse_hex(color: str) -> tuple[int, int, int]:
    """Parse a ``#RRGGBB`` (or ``#RGB``) hex color into an ``(r, g, b)`` tuple.

    Raises:
        ToolError: If the string is not a valid 3- or 6-character hex color.
    """
    c = color.strip().lstrip("#")
    if len(c) == 3:
        c = "".join(ch * 2 for ch in c)
    if len(c) != 6:
        raise ToolError(
            f"Invalid hex color '{color}'. Use #RRGGBB or #RGB (e.g. #FF8800)."
        )
    try:
        return (int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16))
    except ValueError as exc:
        raise ToolError(f"Invalid hex color '{color}': {exc}") from exc
