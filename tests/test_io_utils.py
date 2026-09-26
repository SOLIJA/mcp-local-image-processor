"""Tests for tools/io_utils.py — shared I/O, validation, and conversion helpers."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
from PIL import Image

from mcp.server.mcpserver.exceptions import ToolError

from tools.io_utils import (
    FORMAT_BY_EXT,
    KNOWN_FORMATS,
    MAX_PIXELS,
    QUALITY_FORMATS,
    ResolvedOutput,
    _check_size,
    _font_candidates,
    _parse_hex,
    _try_rmdir_parents,
    cv_bgr_from,
    cv_bgra_from,
    load_image,
    output_format,
    pil_from_cv_bgr,
    pil_from_cv_bgra,
    resolve_input,
    resolve_output,
    save_image,
    to_lum8,
    to_rgb,
    to_rgba,
)

# ---------------------------------------------------------------------------
# _parse_hex
# ---------------------------------------------------------------------------


class TestParseHex:
    def test_six_char(self):
        assert _parse_hex("#FF8800") == (255, 136, 0)

    def test_three_char(self):
        assert _parse_hex("#F80") == (255, 136, 0)

    def test_lowercase(self):
        assert _parse_hex("#ff8800") == (255, 136, 0)

    def test_no_hash(self):
        assert _parse_hex("FF8800") == (255, 136, 0)

    def test_whitespace(self):
        assert _parse_hex("  #FF8800  ") == (255, 136, 0)

    def test_too_short(self):
        with pytest.raises(ToolError):
            _parse_hex("#F0")

    def test_too_long(self):
        with pytest.raises(ToolError):
            _parse_hex("#FF88001")

    def test_non_hex(self):
        with pytest.raises(ToolError):
            _parse_hex("#GG8800")


# ---------------------------------------------------------------------------
# resolve_input / resolve_output
# ---------------------------------------------------------------------------


class TestResolveInput:
    def test_existing_file(self, tmp_path: Path) -> None:
        f = tmp_path / "test.png"
        f.write_bytes(b"fake png")
        result = resolve_input(str(f))
        assert result == f.resolve()

    def test_missing_file(self) -> None:
        with pytest.raises(ToolError):
            resolve_input("/nonexistent/path/image.png")

    def test_directory_instead_of_file(self, tmp_path: Path) -> None:
        d = tmp_path / "dir"
        d.mkdir()
        with pytest.raises(ToolError):
            resolve_input(str(d))


class TestResolveOutput:
    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        out = tmp_path / "a" / "b" / "out.png"
        resolved = resolve_output(str(out))
        assert isinstance(resolved, ResolvedOutput)
        assert resolved.path == out.resolve()
        # Parent dirs should exist now.
        assert out.parent.exists()

    def test_no_extension(self, tmp_path: Path) -> None:
        out = tmp_path / "noext"
        with pytest.raises(ToolError):
            resolve_output(str(out))

    def test_created_dirs_tracked(self, tmp_path: Path) -> None:
        out = tmp_path / "x" / "y" / "out.png"
        resolved = resolve_output(str(out))
        # x and y should be in created_dirs (they were created).
        assert len(resolved.created_dirs) >= 2

    def test_no_traversal_when_root_set(self, tmp_path: Path) -> None:
        sandbox = tmp_path / "sandbox"
        sandbox.mkdir()
        out = tmp_path / "outside" / "out.png"
        with pytest.raises(ToolError):  # escapes root
            resolve_output(str(out), root=sandbox)


class TestTryRmdirParents:
    def test_removes_empty_dirs(self, tmp_path: Path) -> None:
        d = tmp_path / "a" / "b"
        d.mkdir(parents=True)
        _try_rmdir_parents(str(tmp_path / "a" / "b" / "f.txt"), [d, d.parent])
        # b should be removed (empty). a should also be removed if empty.
        assert not d.exists()

    def test_skips_non_empty(self, tmp_path: Path) -> None:
        d = tmp_path / "a" / "b"
        d.mkdir(parents=True)
        (d / "keep.txt").write_text("x")
        _try_rmdir_parents(str(tmp_path / "a" / "b" / "f.txt"), [d, d.parent])
        assert d.exists()  # not removed because it has content


# ---------------------------------------------------------------------------
# Image conversions
# ---------------------------------------------------------------------------


class TestToRgb:
    def test_rgb_passthrough(self) -> None:
        img = Image.new("RGB", (10, 10), (255, 0, 0))
        result = to_rgb(img)
        assert result.mode == "RGB"
        assert result is not img  # should be a copy

    def test_rgba_flattens(self) -> None:
        img = Image.new("RGBA", (10, 10), (255, 0, 0, 128))
        result = to_rgb(img)
        assert result.mode == "RGB"
        # Should be white behind the red due to white background paste.

    def test_other_modes(self) -> None:
        img = Image.new("L", (10, 10), 128)
        result = to_rgb(img)
        assert result.mode == "RGB"


class TestToRgba:
    def test_rgba_passthrough(self) -> None:
        img = Image.new("RGBA", (10, 10), (255, 0, 0, 255))
        result = to_rgba(img)
        assert result.mode == "RGBA"

    def test_rgb_adds_alpha(self) -> None:
        img = Image.new("RGB", (10, 10), (255, 0, 0))
        result = to_rgba(img)
        assert result.mode == "RGBA"

    def test_palette_mode(self) -> None:
        img = Image.new("P", (10, 10))
        result = to_rgba(img)
        assert result.mode == "RGBA"


class TestToLum8:
    def test_l_passthrough(self) -> None:
        img = Image.new("L", (10, 10), 128)
        result = to_lum8(img)
        assert result.mode == "L"

    def test_rgb_to_l(self) -> None:
        img = Image.new("RGB", (10, 10), (128, 128, 128))
        result = to_lum8(img)
        assert result.mode == "L"


# ---------------------------------------------------------------------------
# OpenCV conversions
# ---------------------------------------------------------------------------


class TestCvConversions:
    def test_cv_bgr_from_shape(self) -> None:
        img = Image.new("RGB", (10, 20), (255, 128, 64))
        arr = cv_bgr_from(img)
        assert arr.shape == (20, 10, 3)  # H, W, C (OpenCV convention)

    def test_cv_bgra_from_shape(self) -> None:
        img = Image.new("RGBA", (10, 20), (255, 128, 64, 200))
        arr = cv_bgra_from(img)
        assert arr.shape == (20, 10, 4)

    def test_roundtrip_bgr(self) -> None:
        img = Image.new("RGB", (5, 5), (100, 150, 200))
        arr = cv_bgr_from(img)
        back = pil_from_cv_bgr(arr)
        assert back.mode == "RGB"

    def test_roundtrip_bgra(self) -> None:
        img = Image.new("RGBA", (5, 5), (100, 150, 200, 255))
        arr = cv_bgra_from(img)
        back = pil_from_cv_bgra(arr)
        assert back.mode == "RGBA"


# ---------------------------------------------------------------------------
# _check_size
# ---------------------------------------------------------------------------


class TestCheckSize:
    def test_within_limit(self) -> None:
        img = Image.new("RGB", (100, 100))
        _check_size(img)  # should not raise

    def test_exceeds_limit(self) -> None:
        # Use a very large image that exceeds MAX_PIXELS.
        img = Image.new("RGB", (10000, 10000))
        with pytest.raises(ToolError):
            _check_size(img)


# ---------------------------------------------------------------------------
# output_format
# ---------------------------------------------------------------------------


class TestOutputFormat:
    def test_png(self):
        assert output_format("out.png") == "PNG"

    def test_jpg(self):
        assert output_format("out.jpg") == "JPEG"

    def test_unsupported(self):
        with pytest.raises(ToolError):
            output_format("out.xyz")


# ---------------------------------------------------------------------------
# save_image / load_image round-trip
# ---------------------------------------------------------------------------


class TestSaveLoadRoundTrip:
    def test_png_roundtrip(self, tmp_path: Path) -> None:
        img = Image.new("RGB", (50, 50), (255, 128, 64))
        out = tmp_path / "round.png"
        save_image(img, str(out))
        loaded_p, loaded_img = load_image(str(out))
        assert loaded_p == out.resolve()
        assert loaded_img.size == (50, 50)
        assert loaded_img.mode == "RGB"

    def test_jpeg_roundtrip(self, tmp_path: Path) -> None:
        img = Image.new("RGB", (50, 50), (100, 200, 50))
        out = tmp_path / "round.jpg"
        save_image(img, str(out), quality=90)
        loaded_p, loaded_img = load_image(str(out))
        assert loaded_img.size == (50, 50)

    def test_rgba_save_to_png(self, tmp_path: Path) -> None:
        img = Image.new("RGBA", (10, 10), (255, 0, 0, 128))
        out = tmp_path / "rgba.png"
        save_image(img, str(out))
        loaded_p, loaded_img = load_image(str(out))
        assert loaded_img.mode == "RGBA"

    def test_rgba_save_to_jpeg_flattens(self, tmp_path: Path) -> None:
        img = Image.new("RGBA", (10, 10), (255, 0, 0, 128))
        out = tmp_path / "rgba.jpg"
        save_image(img, str(out))
        # JPEG doesn't support alpha; should be flattened to RGB.
        loaded_p, loaded_img = load_image(str(out))
        assert loaded_img.mode == "RGB"


# ---------------------------------------------------------------------------
# _font_candidates (platform check)
# ---------------------------------------------------------------------------


class TestFontCandidates:
    def test_returns_tuple(self):
        candidates = _font_candidates()
        assert isinstance(candidates, tuple)
        assert len(candidates) >= 1

    def test_paths_are_strings(self):
        for path in _font_candidates():
            assert isinstance(path, str)


# ---------------------------------------------------------------------------
# Constants sanity checks
# ---------------------------------------------------------------------------


class TestConstants:
    def test_max_pixels_configurable(self) -> None:
        original = os.environ.get("MAX_PIXELS")
        try:
            os.environ["MAX_PIXELS"] = "1000"
            # Re-import to pick up the new value.
            import importlib

            import tools.io_utils as io_mod

            importlib.reload(io_mod)
            assert io_mod.MAX_PIXELS == 1000
        finally:
            if original is not None:
                os.environ["MAX_PIXELS"] = original
            elif "MAX_PIXELS" in os.environ:
                del os.environ["MAX_PIXELS"]
            # Restore the module's MAX_PIXELS so later tests aren't affected.
            importlib.reload(io_mod)

    def test_known_formats_contains_expected(self):
        assert "PNG" in KNOWN_FORMATS
        assert "JPEG" in KNOWN_FORMATS
        assert "WEBP" in KNOWN_FORMATS

    def test_quality_formats(self):
        assert "JPEG" in QUALITY_FORMATS
        assert "WEBP" in QUALITY_FORMATS
        assert "PNG" not in QUALITY_FORMATS
