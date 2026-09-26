"""Tests for image tool functions across all 28 tools in the suite.

These tests exercise the public API of each tool module — parameter validation,
edge cases, DoS protection, and correctness. They do NOT require a running MCP server;
they import the tool modules directly to call their functions.
"""

from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from mcp.server.mcpserver.exceptions import ToolError
from tools.color import (
    adjust_saturation,
    adjust_white_balance,
    color_tint,
    convert_color_space,
)
from tools.color import _parse_hex as color_parse_hex
from tools.composite import (
    bulk_remove_background,
    bulk_watermark,
    image_blend,
    overlay_image,
    overlay_text,
    remove_background,
)
from tools.detail import (
    _detect_image_type,
    blur_image,
    denoise_image,
    edge_detection,
    image_difference,
    sharpen_image,
    upscale_image,
)
from tools.encoding import (
    bulk_convert,
    convert_format,
    get_image_metadata,
    strip_metadata,
)
from tools.geometry import (
    bulk_crop,
    bulk_resize,
    crop_image,
    image_grid,
    resize_image,
    rotate_flip,
    smart_crop,
    trim_whitespace,
)
from tools.info import image_info
from tools.io_utils import load_image
from tools.tonal import (
    adjust_brightness,
    adjust_contrast,
    auto_levels,
    histogram_equalization,
    normalize,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_png(size: tuple[int, int] = (50, 50), mode: str = "RGB", color=None) -> Path:
    """Create a temporary PNG file with the given dimensions and return its path."""
    if color is None:
        color = (128, 64, 32) if mode == "RGB" else (128, 64, 32, 200)
    img = Image.new(mode, size, color)
    p = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    p.close()
    img.save(p.name)
    return Path(p.name)


def _make_rgba_png(size: tuple[int, int] = (50, 50)) -> Path:
    """Create a temporary RGBA PNG with a gradient."""
    img = Image.new("RGBA", size)
    for x in range(size[0]):
        for y in range(size[1]):
            img.putpixel((x, y), (x * 5, y * 5, 128, 200))
    p = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    p.close()
    img.save(p.name)
    return Path(p.name)


def _cleanup(*paths: Path) -> None:
    """Remove temporary files."""
    for p in paths:
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# 1. Geometry Tools
# ---------------------------------------------------------------------------


class TestResizeImage:
    def test_width_only(self, tmp_path: Path) -> None:
        inp = _make_png((100, 50))
        out = tmp_path / "out.png"
        resize_image(str(inp), str(out), width=50)
        assert out.exists()
        _, img = load_image(str(out))
        assert img.size == (50, 25)

    def test_height_only(self, tmp_path: Path) -> None:
        inp = _make_png((100, 50))
        out = tmp_path / "out.png"
        resize_image(str(inp), str(out), height=25)
        assert out.exists()
        _, img = load_image(str(out))
        assert img.size == (50, 25)

    def test_both_with_aspect(self, tmp_path: Path) -> None:
        inp = _make_png((100, 50))
        out = tmp_path / "out.png"
        resize_image(str(inp), str(out), width=80, height=80)
        assert out.exists()
        _, img = load_image(str(out))
        assert img.size[0] <= 80 and img.size[1] <= 80

    def test_no_aspect(self, tmp_path: Path) -> None:
        inp = _make_png((100, 50))
        out = tmp_path / "out.png"
        resize_image(
            str(inp), str(out), width=80, height=80, maintain_aspect_ratio=False
        )
        assert out.exists()
        _, img = load_image(str(out))
        assert img.size == (80, 80)

    def test_invalid_width(self) -> None:
        inp = _make_png()
        with pytest.raises(ToolError, match="width must be >= 1"):
            resize_image(str(inp), "/tmp/out.png", width=0)

    def test_invalid_height(self) -> None:
        inp = _make_png()
        with pytest.raises(ToolError, match="height must be >= 1"):
            resize_image(str(inp), "/tmp/out.png", height=-5)

    def test_neither_dimension(self) -> None:
        inp = _make_png()
        with pytest.raises(ToolError, match="At least one of width or height is required"):
            resize_image(str(inp), "/tmp/out.png")

    def test_invalid_resample(self, tmp_path: Path) -> None:
        inp = _make_png()
        out = tmp_path / "out.png"
        with pytest.raises(ToolError, match="Unknown resample filter"):
            resize_image(str(inp), str(out), width=10, resample="nonexistent")


class TestCropImage:
    def test_basic_crop(self, tmp_path: Path) -> None:
        inp = _make_png((100, 100))
        out = tmp_path / "out.png"
        crop_image(str(inp), str(out), x=10, y=20, width=50, height=30)
        assert out.exists()
        _, img = load_image(str(out))
        assert img.size == (50, 30)

    def test_negative_coords_clamped(self, tmp_path: Path) -> None:
        inp = _make_png((100, 100))
        out = tmp_path / "out.png"
        crop_image(str(inp), str(out), x=-10, y=-10, width=50, height=50)
        assert out.exists()
        _, img = load_image(str(out))
        assert img.size[0] <= 50 and img.size[1] <= 50

    def test_crop_beyond_bounds(self, tmp_path: Path) -> None:
        inp = _make_png((50, 50))
        out = tmp_path / "out.png"
        crop_image(str(inp), str(out), x=40, y=40, width=50, height=50)
        assert out.exists()
        _, img = load_image(str(out))
        assert img.size[0] <= 10 and img.size[1] <= 10

    def test_crop_entirely_outside(self, tmp_path: Path) -> None:
        inp = _make_png((50, 50))
        out = tmp_path / "out.png"
        with pytest.raises(ToolError, match="entirely outside image bounds"):
            crop_image(str(inp), str(out), x=60, y=60, width=10, height=10)

    def test_invalid_dimensions(self) -> None:
        inp = _make_png()
        with pytest.raises(ToolError, match="width must be >= 1"):
            crop_image(str(inp), "/tmp/out.png", x=0, y=0, width=0, height=50)
        with pytest.raises(ToolError, match="height must be >= 1"):
            crop_image(str(inp), "/tmp/out.png", x=0, y=0, width=50, height=-1)


class TestSmartCrop:
    def test_default_square_crop(self, tmp_path: Path) -> None:
        inp = _make_png((200, 100))
        out = tmp_path / "out.png"
        msg = smart_crop(str(inp), str(out))
        assert out.exists()
        assert "Smart cropped" in msg
        _, img = load_image(str(out))
        assert img.size == (100, 100)

    def test_aspect_ratio_16_9(self, tmp_path: Path) -> None:
        inp = _make_png((500, 500))
        out = tmp_path / "out.png"
        smart_crop(str(inp), str(out), aspect_ratio="16:9")
        assert out.exists()
        _, img = load_image(str(out))
        # 500 width, aspect ratio 16:9 -> height 281
        assert img.size == (500, 281)

    def test_target_width_and_height(self, tmp_path: Path) -> None:
        inp = _make_png((200, 100))
        out = tmp_path / "out.png"
        smart_crop(str(inp), str(out), width=80, height=80)
        assert out.exists()
        _, img = load_image(str(out))
        assert img.size == (80, 80)

    def test_focal_centering_keeps_subject(self, tmp_path: Path) -> None:
        # Create a wide image (400x100) mostly white with an obvious subject on the left
        raw = Image.new("RGB", (400, 100), (255, 255, 255))
        # Draw high-contrast subject at x=10..70, y=20..80
        for x in range(10, 70):
            for y in range(20, 80):
                raw.putpixel((x, y), (0, 0, 0))
        inp = tmp_path / "subject_left.png"
        raw.save(inp)

        out = tmp_path / "out_smart.png"
        smart_crop(str(inp), str(out), aspect_ratio="1:1", focus_mode="edge")
        assert out.exists()
        _, img = load_image(str(out))
        assert img.size == (100, 100)
        # Verify the black subject was preserved (crop framed left, not middle)
        np_arr = np.array(img)
        assert np.any(np_arr == 0), "The salient subject on the left should be preserved"

    def test_tight_frame(self, tmp_path: Path) -> None:
        raw = Image.new("RGB", (400, 400), (255, 255, 255))
        for x in range(150, 250):
            for y in range(150, 250):
                raw.putpixel((x, y), (0, 0, 0))
        inp = tmp_path / "tight.png"
        raw.save(inp)

        out = tmp_path / "out_tight.png"
        smart_crop(str(inp), str(out), aspect_ratio="1:1", focus_mode="edge", tight_frame=True)
        assert out.exists()
        _, img = load_image(str(out))
        # Tight frame around 100x100 subject (+ 10% padding = 120x120) should be substantially smaller than 400x400
        assert img.size[0] < 300 and img.size[1] < 300

    def test_center_mode(self, tmp_path: Path) -> None:
        inp = _make_png((300, 100))
        out = tmp_path / "out_center.png"
        smart_crop(str(inp), str(out), aspect_ratio="1:1", focus_mode="center")
        assert out.exists()
        _, img = load_image(str(out))
        assert img.size == (100, 100)

    def test_saliency_mode(self, tmp_path: Path) -> None:
        # Create an image with an obvious salient object (red box on black background)
        raw = Image.new("RGB", (300, 100), (0, 0, 0))
        for x in range(20, 80):
            for y in range(20, 80):
                raw.putpixel((x, y), (255, 0, 0))
        inp = tmp_path / "saliency.png"
        raw.save(inp)
        out = tmp_path / "out_saliency.png"
        smart_crop(str(inp), str(out), aspect_ratio="1:1", focus_mode="saliency")
        assert out.exists()
        _, img = load_image(str(out))
        assert img.size == (100, 100)

    def test_face_mode_fallback(self, tmp_path: Path) -> None:
        inp = _make_png((300, 100))
        out = tmp_path / "out_face.png"
        smart_crop(str(inp), str(out), aspect_ratio="1:1", focus_mode="face")
        assert out.exists()
        _, img = load_image(str(out))
        assert img.size == (100, 100)

    def test_invalid_aspect_ratio(self, tmp_path: Path) -> None:
        inp = _make_png((100, 100))
        out = tmp_path / "out.png"
        with pytest.raises(ToolError, match="Invalid aspect_ratio"):
            smart_crop(str(inp), str(out), aspect_ratio="invalid")

    def test_invalid_padding_percent(self, tmp_path: Path) -> None:
        inp = _make_png((100, 100))
        out = tmp_path / "out.png"
        with pytest.raises(ToolError, match="padding_percent must be between 0.0 and 1.0"):
            smart_crop(str(inp), str(out), padding_percent=1.5)

    def test_invalid_focus_mode(self, tmp_path: Path) -> None:
        inp = _make_png((100, 100))
        out = tmp_path / "out.png"
        with pytest.raises(ToolError, match="Unknown focus_mode"):
            smart_crop(str(inp), str(out), focus_mode="nonexistent")


class TestBulkCrop:
    def test_bulk_smart_crop_paths(self, tmp_path: Path) -> None:
        inp1 = _make_png((200, 100))
        inp2 = _make_png((300, 100))
        out_dir = tmp_path / "bulk_out"
        msg = bulk_crop([str(inp1), str(inp2)], str(out_dir), aspect_ratio="1:1")
        assert "Bulk cropped 2/2" in msg
        assert (out_dir / inp1.name).exists()
        assert (out_dir / inp2.name).exists()
        _, img1 = load_image(str(out_dir / inp1.name))
        assert img1.size == (100, 100)

    def test_bulk_smart_crop_directory(self, tmp_path: Path) -> None:
        in_dir = tmp_path / "in_dir"
        in_dir.mkdir()
        img1 = in_dir / "pic1.png"
        img2 = in_dir / "pic2.png"
        Image.new("RGB", (100, 50), (200, 100, 50)).save(img1)
        Image.new("RGB", (120, 60), (50, 100, 200)).save(img2)

        out_dir = tmp_path / "out_dir"
        msg = bulk_crop([str(in_dir)], str(out_dir), aspect_ratio="1:1")
        assert "Bulk cropped 2/2" in msg
        assert (out_dir / "pic1.png").exists()
        assert (out_dir / "pic2.png").exists()

    def test_bulk_fixed_crop(self, tmp_path: Path) -> None:
        in_dir = tmp_path / "in_fixed"
        in_dir.mkdir()
        img1 = in_dir / "pic1.png"
        img2 = in_dir / "pic2.png"
        Image.new("RGB", (100, 100), (200, 100, 50)).save(img1)
        Image.new("RGB", (100, 100), (50, 100, 200)).save(img2)

        out_dir = tmp_path / "out_fixed"
        msg = bulk_crop(
            [str(in_dir)],
            str(out_dir),
            x=10,
            y=10,
            crop_width=40,
            crop_height=30,
        )
        assert "Bulk cropped 2/2" in msg
        _, loaded = load_image(str(out_dir / "pic1.png"))
        assert loaded.size == (40, 30)

    def test_bulk_partial_fixed_params_error(self, tmp_path: Path) -> None:
        inp = _make_png((100, 100))
        out_dir = tmp_path / "out"
        with pytest.raises(ToolError, match="all of x, y, crop_width, and crop_height must be provided"):
            bulk_crop([str(inp)], str(out_dir), x=10, y=10)

    def test_bulk_empty_inputs_error(self, tmp_path: Path) -> None:
        with pytest.raises(ToolError, match="At least one input image"):
            bulk_crop([], str(tmp_path / "out"))


class TestBulkResize:
    def test_bulk_resize_dimensions(self, tmp_path: Path) -> None:
        inp1 = _make_png((200, 100))
        inp2 = _make_png((300, 150))
        out_dir = tmp_path / "resized"
        msg = bulk_resize([str(inp1), str(inp2)], str(out_dir), width=100)
        assert "Bulk resized 2/2" in msg
        assert (out_dir / inp1.name).exists()
        assert (out_dir / inp2.name).exists()
        _, img1 = load_image(str(out_dir / inp1.name))
        assert img1.size == (100, 50)
        _, img2 = load_image(str(out_dir / inp2.name))
        assert img2.size == (100, 50)

    def test_bulk_resize_format_and_strip(self, tmp_path: Path) -> None:
        inp = _make_png((100, 100))
        out_dir = tmp_path / "resized_webp"
        msg = bulk_resize(
            [str(inp)],
            str(out_dir),
            width=50,
            format="webp",
            quality=80,
            strip_metadata=True,
        )
        assert "Bulk resized 1/1" in msg
        out_file = out_dir / (inp.stem + ".webp")
        assert out_file.exists()
        _, img = load_image(str(out_file))
        assert img.size == (50, 50)
        assert img.format == "WEBP"

    def test_bulk_resize_invalid_dimension(self, tmp_path: Path) -> None:
        inp = _make_png((100, 100))
        with pytest.raises(ToolError, match="width must be >= 1"):
            bulk_resize([str(inp)], str(tmp_path / "out"), width=0)

    def test_bulk_resize_default_output_dir(self, tmp_path: Path) -> None:
        in_dir = tmp_path / "inputs"
        in_dir.mkdir()
        inp = in_dir / "test.png"
        Image.new("RGB", (100, 100)).save(inp)
        msg = bulk_resize([str(inp)], width=50)
        assert "Bulk resized 1/1" in msg
        default_out = in_dir / "output"
        assert default_out.exists()
        assert (default_out / "test.png").exists()
        _, img = load_image(str(default_out / "test.png"))
        assert img.size == (50, 50)


class TestRotateFlip:
    def test_no_rotation(self, tmp_path: Path) -> None:
        inp = _make_png((100, 50))
        out = tmp_path / "out.png"
        rotate_flip(str(inp), str(out), angle=0.0)
        assert out.exists()
        _, img = load_image(str(out))
        assert img.size == (100, 50)

    def test_90_degrees(self, tmp_path: Path) -> None:
        inp = _make_png((100, 50))
        out = tmp_path / "out.png"
        rotate_flip(str(inp), str(out), angle=90.0)
        assert out.exists()
        _, img = load_image(str(out))
        assert img.size == (50, 100)

    def test_180_degrees(self, tmp_path: Path) -> None:
        inp = _make_png((100, 50))
        out = tmp_path / "out.png"
        rotate_flip(str(inp), str(out), angle=180.0)
        assert out.exists()
        _, img = load_image(str(out))
        assert img.size == (100, 50)

    def test_270_degrees(self, tmp_path: Path) -> None:
        inp = _make_png((100, 50))
        out = tmp_path / "out.png"
        rotate_flip(str(inp), str(out), angle=270.0)
        assert out.exists()
        _, img = load_image(str(out))
        assert img.size == (50, 100)

    def test_arbitrary_angle_expands_bounds(self, tmp_path: Path) -> None:
        """45 degree rotation of 100x100 square should expand bounding box to ~142x142."""
        inp = _make_png((100, 100))
        out = tmp_path / "out.png"
        rotate_flip(str(inp), str(out), angle=45.0)
        assert out.exists()
        _, img = load_image(str(out))
        assert img.size[0] > 100 and img.size[1] > 100

    def test_arbitrary_angle_rgba_preserves_alpha(self, tmp_path: Path) -> None:
        inp = _make_rgba_png((80, 80))
        out = tmp_path / "out.png"
        rotate_flip(str(inp), str(out), angle=30.0)
        assert out.exists()
        _, img = load_image(str(out))
        assert img.mode == "RGBA"

    def test_horizontal_flip(self, tmp_path: Path) -> None:
        inp = _make_png((100, 50))
        out = tmp_path / "out.png"
        rotate_flip(str(inp), str(out), flip_h=True)
        assert out.exists()

    def test_vertical_flip(self, tmp_path: Path) -> None:
        inp = _make_png((100, 50))
        out = tmp_path / "out.png"
        rotate_flip(str(inp), str(out), flip_v=True)
        assert out.exists()

    def test_angle_360_normalizes_to_zero(self, tmp_path: Path) -> None:
        inp = _make_png((100, 50))
        out = tmp_path / "out.png"
        rotate_flip(str(inp), str(out), angle=360.0)
        assert out.exists()
        _, img = load_image(str(out))
        assert img.size == (100, 50)


class TestTrimWhitespace:
    def test_trims_borders(self, tmp_path: Path) -> None:
        img = Image.new("RGB", (100, 100), (255, 255, 255))
        for x in range(40, 60):
            for y in range(40, 60):
                img.putpixel((x, y), (255, 0, 0))
        inp = tmp_path / "border.png"
        img.save(inp)

        out = tmp_path / "out.png"
        trim_whitespace(str(inp), str(out), tolerance=10)
        assert out.exists()
        _, trimmed = load_image(str(out))
        assert trimmed.size == (20, 20)

    def test_dirty_rgb_under_alpha_trimmed_correctly(self, tmp_path: Path) -> None:
        """Transparent pixels with random/dirty RGB values should be treated as transparent background."""
        img = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
        # Add dirty RGB to the transparent border
        for x in range(100):
            for y in range(100):
                img.putpixel((x, y), (255, 128, 64, 0))  # fully transparent, dirty RGB
        # Add opaque red content in the center 20x20
        for x in range(40, 60):
            for y in range(40, 60):
                img.putpixel((x, y), (255, 0, 0, 255))

        inp = tmp_path / "dirty_alpha.png"
        img.save(inp)

        out = tmp_path / "out.png"
        trim_whitespace(str(inp), str(out), tolerance=10)
        assert out.exists()
        _, trimmed = load_image(str(out))
        assert trimmed.size == (20, 20)

    def test_fully_transparent_rgba(self, tmp_path: Path) -> None:
        img = Image.new("RGBA", (50, 50), (0, 0, 0, 0))
        inp = tmp_path / "trans.png"
        img.save(inp)

        out = tmp_path / "out.png"
        with pytest.raises(ToolError, match="fully transparent"):
            trim_whitespace(str(inp), str(out))

    def test_invalid_tolerance(self) -> None:
        with pytest.raises(ToolError, match="tolerance must be between 0 and 255"):
            trim_whitespace("/tmp/fake.png", "/tmp/out.png", tolerance=-1)
        with pytest.raises(ToolError, match="tolerance must be between 0 and 255"):
            trim_whitespace("/tmp/fake.png", "/tmp/out.png", tolerance=300)

    def test_entire_image_matches_background(self, tmp_path: Path) -> None:
        inp = _make_png((50, 50), color=(100, 100, 100))
        out = tmp_path / "out.png"
        with pytest.raises(ToolError, match="Entire image matches background"):
            trim_whitespace(str(inp), str(out), tolerance=10)


class TestImageGrid:
    def test_image_grid_2x2(self, tmp_path: Path) -> None:
        """Test a 2x2 grid of 4 uniform images."""
        images = []
        for i, color in enumerate([
            (255, 0, 0, 255),
            (0, 255, 0, 255),
            (0, 0, 255, 255),
            (255, 255, 0, 255),
        ]):
            img = Image.new("RGBA", (50, 50), color)
            p = tmp_path / f"img{i}.png"
            img.save(p)
            images.append(str(p))

        out = tmp_path / "output.png"
        result = image_grid(images, str(out), cols=2, padding=10)

        assert out.exists()
        assert "Created 2x2 grid" in result
        result_img = Image.open(out)
        # 2 cols * 50 + 1 * 10 padding = 110 wide and high
        assert result_img.size == (110, 110)

    def test_image_grid_single_image(self, tmp_path: Path) -> None:
        """Test a single image in a grid produces no padding."""
        img = Image.new("RGBA", (50, 50), (255, 0, 0, 255))
        inp = tmp_path / "input.png"
        img.save(inp)

        out = tmp_path / "output.png"
        result = image_grid([str(inp)], str(out), cols=1)

        assert out.exists()
        result_img = Image.open(out)
        assert result_img.size == (50, 50)

    def test_image_grid_empty_input(self, tmp_path: Path) -> None:
        """Test that empty input raises ToolError."""
        out = tmp_path / "output.png"
        with pytest.raises(ToolError, match="At least one"):
            image_grid([], str(out))

    def test_image_grid_invalid_cols(self, tmp_path: Path) -> None:
        """Test that cols < 1 raises ToolError."""
        p = _make_png()
        out = tmp_path / "output.png"
        with pytest.raises(ToolError, match="cols must be >= 1"):
            image_grid([str(p)], str(out), cols=0)

    def test_image_grid_invalid_padding(self, tmp_path: Path) -> None:
        """Test that padding < 0 raises ToolError."""
        p = _make_png()
        out = tmp_path / "output.png"
        with pytest.raises(ToolError, match="padding must be >= 0"):
            image_grid([str(p)], str(out), padding=-5)

    def test_image_grid_mismatched_cell_dimensions(self, tmp_path: Path) -> None:
        """Test providing only cell_width raises ToolError."""
        p = _make_png()
        out = tmp_path / "output.png"
        with pytest.raises(ToolError, match="Both cell_width and cell_height"):
            image_grid([str(p)], str(out), cell_width=100)

    def test_image_grid_custom_bg(self, tmp_path: Path) -> None:
        """Test custom background color."""
        img = Image.new("RGBA", (50, 50), (255, 0, 0, 255))
        inp = tmp_path / "input.png"
        img.save(inp)

        out = tmp_path / "output.png"
        image_grid([str(inp)], str(out), cols=2, padding=10, bg_color="#FF00FF")

        assert out.exists()
        result_img = Image.open(out)
        # Check an empty cell / padding area is the magenta background
        pixel = result_img.getpixel((55, 25))
        assert pixel == (255, 0, 255, 255)

    def test_image_grid_custom_cell_size(self, tmp_path: Path) -> None:
        """Test fixed cell dimensions center smaller images without stretching."""
        img = Image.new("RGBA", (30, 30), (255, 0, 0, 255))
        inp = tmp_path / "input.png"
        img.save(inp)

        out = tmp_path / "output.png"
        image_grid(
            [str(inp)],
            str(out),
            cols=1,
            cell_width=100,
            cell_height=100,
            bg_color="#000000",
        )

        assert out.exists()
        result_img = Image.open(out)
        assert result_img.size == (100, 100)

        # Corner should be black background
        assert result_img.getpixel((0, 0)) == (0, 0, 0, 255)
        # Center should be the 30x30 red image (top-left at 35, 35)
        assert result_img.getpixel((35, 35)) == (255, 0, 0, 255)
        assert result_img.getpixel((50, 50)) == (255, 0, 0, 255)

    def test_image_grid_resize_to_fit(self, tmp_path: Path) -> None:
        """Test that images are resized to uniform cell size."""
        img1 = Image.new("RGBA", (100, 50), (255, 0, 0, 255))
        img2 = Image.new("RGBA", (30, 80), (0, 255, 0, 255))

        p1 = tmp_path / "img1.png"
        p2 = tmp_path / "img2.png"
        img1.save(p1)
        img2.save(p2)

        out = tmp_path / "output.png"
        image_grid([str(p1), str(p2)], str(out), cols=2, resize_to_fit=True)

        assert out.exists()
        result_img = Image.open(out)
        # Cell size = max(100, 30) x max(50, 80) = 100 x 80
        # Canvas = 2 * 100 + 10 = 210 wide, 80 high
        assert result_img.size == (210, 80)

    def test_image_grid_non_uniform_multi_row(self, tmp_path: Path) -> None:
        """Test multi-row non-uniform grid aligns properly without diagonal offsets."""
        img1 = Image.new("RGBA", (100, 50), (255, 0, 0, 255))
        img2 = Image.new("RGBA", (30, 80), (0, 255, 0, 255))
        img3 = Image.new("RGBA", (40, 60), (0, 0, 255, 255))
        img4 = Image.new("RGBA", (80, 40), (255, 255, 0, 255))

        paths = []
        for i, img in enumerate([img1, img2, img3, img4]):
            p = tmp_path / f"img{i}.png"
            img.save(p)
            paths.append(str(p))

        out = tmp_path / "output.png"
        image_grid(paths, str(out), cols=2, padding=10, resize_to_fit=False)

        assert out.exists()
        result_img = Image.open(out)
        # Col 0 width = max(100, 40) = 100
        # Col 1 width = max(30, 80) = 80
        # Total width = 100 + 80 + 10 = 190
        # Row 0 height = max(50, 80) = 80
        # Row 1 height = max(60, 40) = 60
        # Total height = 80 + 60 + 10 = 150
        assert result_img.size == (190, 150)

    def test_image_grid_transparent_png_blending(self, tmp_path: Path) -> None:
        """Test that images with transparency blend cleanly over the background."""
        img = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
        img.putpixel((5, 5), (0, 255, 0, 255))
        p = tmp_path / "icon.png"
        img.save(p)

        out = tmp_path / "output.png"
        image_grid([str(p)], str(out), cols=1, padding=0, bg_color="#FF0000")

        result_img = Image.open(out)
        # Background corner should be preserved as solid red
        assert result_img.getpixel((0, 0)) == (255, 0, 0, 255)
        # Center pixel is the green dot
        assert result_img.getpixel((5, 5)) == (0, 255, 0, 255)

    def test_image_grid_transparent_bg(self, tmp_path: Path) -> None:
        """Test creating a collage with transparent background."""
        img = Image.new("RGBA", (20, 20), (255, 0, 0, 255))
        p = tmp_path / "img.png"
        img.save(p)

        out = tmp_path / "output.png"
        image_grid([str(p)], str(out), cols=2, padding=10, bg_color="transparent")

        result_img = Image.open(out)
        # Padding area should be transparent
        assert result_img.getpixel((25, 10)) == (0, 0, 0, 0)

    def test_image_grid_single_string_input(self, tmp_path: Path) -> None:
        """Test that passing a single string path instead of a list works."""
        img = Image.new("RGBA", (30, 30), (0, 0, 255, 255))
        inp = tmp_path / "single.png"
        img.save(inp)

        out = tmp_path / "output.png"
        image_grid(str(inp), str(out), cols=1)  # type: ignore[arg-type]

        assert out.exists()
        result_img = Image.open(out)
        assert result_img.size == (30, 30)

    def test_image_grid_non_uniform_cols_exceeds_images(self, tmp_path: Path) -> None:
        """Test cols > len(images) in non-uniform mode raises ToolError."""
        p = _make_png()
        out = tmp_path / "output.png"

        with pytest.raises(ToolError, match="cannot exceed number of images"):
            image_grid([str(p)], str(out), cols=3, resize_to_fit=False)

    def test_image_grid_canvas_exceeds_max_pixels(self, tmp_path: Path) -> None:
        """Test that huge canvas dimensions raise ToolError before allocation."""
        p = _make_png()
        out = tmp_path / "output.png"

        with pytest.raises(ToolError, match="Grid canvas is too large"):
            image_grid([str(p)], str(out), cols=1, cell_width=100000, cell_height=100000)

    def test_image_grid_scales_canvas_limit_with_image_count(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Test that 2 images allow higher total canvas pixels than 1 image."""
        import tools.geometry as geom_mod

        monkeypatch.setattr(geom_mod, "MAX_PIXELS", 2000)
        p1 = _make_png((30, 30))
        p2 = _make_png((30, 30))
        out = tmp_path / "output.png"

        # 1 image in a 50x50 fixed cell = 2500 pixels > 2000 (1 * MAX_PIXELS)
        with pytest.raises(ToolError, match="Grid canvas is too large"):
            image_grid([str(p1)], str(out), cols=1, cell_width=50, cell_height=50)

        # 2 images in 50x50 fixed cells = 2 * 50 * 50 = 5000 (ignoring padding)
        # With cell_width=40, cell_height=40, cols=2, padding=0 -> canvas = 80x40 = 3200 pixels
        # 3200 <= 2 * 2000 = 4000 pixels -> should succeed
        image_grid([str(p1), str(p2)], str(out), cols=2, padding=0, cell_width=40, cell_height=40)
        assert out.exists()

    def test_image_grid_canvas_limit_capped_at_multiplier(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Test that canvas limit caps at MAX_GRID_MULTIPLIER * MAX_PIXELS even with 7+ images."""
        import tools.geometry as geom_mod

        monkeypatch.setattr(geom_mod, "MAX_PIXELS", 1000)
        monkeypatch.setattr(geom_mod, "MAX_GRID_MULTIPLIER", 6)

        # 7 images -> limit is min(7, 6) * 1000 = 6000 pixels (capped at 6x)
        images = []
        for i in range(7):
            p = tmp_path / f"img_{i}.png"
            Image.new("RGBA", (10, 10)).save(p)
            images.append(str(p))

        out = tmp_path / "output.png"
        # 7 images with cell_width=30, cell_height=35, cols=7, padding=0 -> 210x35 = 7350 pixels > 6000
        with pytest.raises(ToolError, match="Grid canvas is too large.*maximum allowed for 7 images is 6,000 pixels"):
            image_grid(images, str(out), cols=7, padding=0, cell_width=30, cell_height=35)


# ---------------------------------------------------------------------------
# 2. Tonal Tools
# ---------------------------------------------------------------------------


class TestTonalTools:
    def test_brightness_valid(self, tmp_path: Path) -> None:
        inp = _make_png()
        out = tmp_path / "out.png"
        adjust_brightness(str(inp), str(out), factor=1.5)
        assert out.exists()

    def test_brightness_limits(self, tmp_path: Path) -> None:
        inp = _make_png()
        out = tmp_path / "out.png"
        with pytest.raises(ToolError, match="factor must be in 0..5.0"):
            adjust_brightness(str(inp), str(out), factor=6.0)
        with pytest.raises(ToolError, match="factor must be in 0..5.0"):
            adjust_brightness(str(inp), str(out), factor=-0.1)

    def test_contrast_valid(self, tmp_path: Path) -> None:
        inp = _make_png()
        out = tmp_path / "out.png"
        adjust_contrast(str(inp), str(out), factor=1.2)
        assert out.exists()

    def test_contrast_limits(self, tmp_path: Path) -> None:
        inp = _make_png()
        out = tmp_path / "out.png"
        with pytest.raises(ToolError, match="factor must be in 0..5.0"):
            adjust_contrast(str(inp), str(out), factor=5.5)

    def test_auto_levels(self, tmp_path: Path) -> None:
        inp = _make_png()
        out = tmp_path / "out.png"
        auto_levels(str(inp), str(out))
        assert out.exists()

    def test_auto_levels_rgba_preserves_alpha(self, tmp_path: Path) -> None:
        inp = _make_rgba_png()
        out = tmp_path / "out.png"
        auto_levels(str(inp), str(out))
        assert out.exists()
        _, img = load_image(str(out))
        assert img.mode == "RGBA"

    def test_histogram_equalization(self, tmp_path: Path) -> None:
        inp = _make_png()
        out = tmp_path / "out.png"
        histogram_equalization(str(inp), str(out), clip_limit=3.0)
        assert out.exists()

    def test_histogram_equalization_invalid_clip(self, tmp_path: Path) -> None:
        inp = _make_png()
        out = tmp_path / "out.png"
        with pytest.raises(ToolError, match="clip_limit must be > 0"):
            histogram_equalization(str(inp), str(out), clip_limit=0)


class TestNormalize:
    def test_normalize_minmax(self, tmp_path: Path) -> None:
        """Test minmax normalization maps min to 0 and max to 255."""
        img = Image.new("RGB", (10, 10), (100, 150, 200))
        pixels = img.load()
        pixels[0, 0] = (20, 30, 40)
        pixels[9, 9] = (220, 230, 240)

        p = tmp_path / "input.png"
        out = tmp_path / "output.png"
        img.save(p)

        result = normalize(str(p), str(out), method="minmax")
        assert out.exists()
        assert "Normalized via minmax" in result

        result_img = Image.open(out)
        # Min pixel should map to 0 and max pixel should map to 255
        pix_min = result_img.getpixel((0, 0))
        pix_max = result_img.getpixel((9, 9))
        assert min(pix_min) == 0
        assert max(pix_max) == 255

    def test_normalize_zscore(self, tmp_path: Path) -> None:
        """Test z-score centers mean near mid-gray without clipping bottom half to black."""
        img = Image.new("RGB", (10, 10), (100, 100, 100))
        pixels = img.load()
        pixels[0, 0] = (0, 0, 0)
        pixels[9, 9] = (200, 200, 200)

        p = tmp_path / "input.png"
        out = tmp_path / "output.png"
        img.save(p)

        normalize(str(p), str(out), method="zscore")
        assert out.exists()

        result_img = Image.open(out)
        # Mean pixel (100, 100, 100) should map near midpoint (~128)
        mid_pixel = result_img.getpixel((5, 5))
        assert 110 <= mid_pixel[0] <= 145

    def test_normalize_percentile(self, tmp_path: Path) -> None:
        """Test percentile clipping normalization suppresses outliers."""
        img = Image.new("RGB", (10, 10))
        pixels = img.load()
        for x in range(10):
            for y in range(10):
                val = int(50 + (x + y) * 5)  # gradient 50 to 140
                pixels[x, y] = (val, val, val)
        # Extreme outliers:
        pixels[0, 0] = (0, 0, 0)
        pixels[9, 9] = (255, 255, 255)

        p = tmp_path / "input.png"
        out = tmp_path / "output.png"
        img.save(p)

        normalize(str(p), str(out), method="percentile")
        assert out.exists()

        result_img = Image.open(out)
        # 0 and 255 outliers are clipped to 1st/99th percentiles; range stretched across 0-255
        assert result_img.getpixel((0, 0)) == (0, 0, 0)
        assert result_img.getpixel((9, 9)) == (255, 255, 255)

    def test_normalize_grayscale_mode(self, tmp_path: Path) -> None:
        """Test that grayscale input preserves mode 'L'."""
        img = Image.new("L", (10, 10), 100)
        p = tmp_path / "gray.png"
        out = tmp_path / "output.png"
        img.save(p)

        normalize(str(p), str(out), method="minmax")
        assert out.exists()
        result_img = Image.open(out)
        assert result_img.mode == "L"

    def test_normalize_per_channel(self, tmp_path: Path) -> None:
        """Test per-channel normalization scales channels independently."""
        img = Image.new("RGB", (10, 10), (50, 100, 150))
        pixels = img.load()
        pixels[0, 0] = (10, 20, 30)
        pixels[9, 9] = (60, 110, 160)

        p = tmp_path / "input.png"
        out = tmp_path / "output.png"
        img.save(p)

        normalize(str(p), str(out), per_channel=True, method="minmax")
        assert out.exists()

        result_img = Image.open(out)
        min_p = result_img.getpixel((0, 0))
        max_p = result_img.getpixel((9, 9))
        assert min_p == (0, 0, 0)
        assert max_p == (255, 255, 255)

    def test_normalize_uniform_image(self, tmp_path: Path) -> None:
        """Test normalization on a uniform image without variance."""
        img = Image.new("RGB", (10, 10), (128, 128, 128))
        p = tmp_path / "input.png"
        out = tmp_path / "output.png"
        img.save(p)

        normalize(str(p), str(out), method="minmax")
        assert out.exists()

        result_img = Image.open(out)
        assert result_img.getpixel((5, 5)) == (0, 0, 0)

    def test_normalize_custom_range(self, tmp_path: Path) -> None:
        """Test normalization to a custom intensity range [50.0, 200.0]."""
        img = Image.new("RGB", (10, 10), (100, 100, 100))
        pixels = img.load()
        pixels[0, 0] = (0, 0, 0)
        pixels[9, 9] = (255, 255, 255)

        p = tmp_path / "input.png"
        out = tmp_path / "output.png"
        img.save(p)

        normalize(
            str(p),
            str(out),
            range_min=50.0,
            range_max=200.0,
            method="minmax",
        )
        assert out.exists()

        result_img = Image.open(out)
        assert result_img.getpixel((0, 0)) == (50, 50, 50)
        assert result_img.getpixel((9, 9)) == (200, 200, 200)

    def test_normalize_preserves_alpha(self, tmp_path: Path) -> None:
        """Test that normalization preserves the alpha channel of RGBA images."""
        img = Image.new("RGBA", (10, 10), (100, 150, 200, 128))
        pixels = img.load()
        pixels[0, 0] = (0, 0, 0, 200)
        pixels[9, 9] = (255, 255, 255, 50)

        p = tmp_path / "input.png"
        out = tmp_path / "output.png"
        img.save(p)

        normalize(str(p), str(out), method="minmax")
        assert out.exists()

        result_img = Image.open(out)
        assert result_img.mode == "RGBA"
        assert result_img.getpixel((0, 0))[3] == 200
        assert result_img.getpixel((9, 9))[3] == 50

    def test_normalize_invalid_method(self, tmp_path: Path) -> None:
        """Test that invalid method raises ToolError."""
        p = _make_png()
        out = tmp_path / "output.png"

        with pytest.raises(ToolError, match="method must be one of"):
            normalize(str(p), str(out), method="invalid")

    def test_normalize_invalid_range_order(self, tmp_path: Path) -> None:
        """Test that range_min >= range_max raises ToolError."""
        p = _make_png()
        out = tmp_path / "output.png"

        with pytest.raises(ToolError, match="range_min .* must be < range_max"):
            normalize(str(p), str(out), range_min=200.0, range_max=100.0)

        with pytest.raises(ToolError, match="range_min .* must be < range_max"):
            normalize(str(p), str(out), range_min=100.0, range_max=100.0)

    def test_normalize_invalid_range_bounds(self, tmp_path: Path) -> None:
        """Test that range bounds outside 0..255 raise ToolError."""
        p = _make_png()
        out = tmp_path / "output.png"

        with pytest.raises(ToolError, match="range bounds must be in 0..255"):
            normalize(str(p), str(out), range_min=-5.0, range_max=200.0)

        with pytest.raises(ToolError, match="range bounds must be in 0..255"):
            normalize(str(p), str(out), range_min=0.0, range_max=300.0)

    def test_normalize_invalid_types(self, tmp_path: Path) -> None:
        """Test that non-numeric range inputs raise ToolError."""
        p = _make_png()
        out = tmp_path / "output.png"

        with pytest.raises(ToolError, match="must be valid numbers"):
            normalize(str(p), str(out), range_min="zero")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 3. Color Tools
# ---------------------------------------------------------------------------


class TestColorTools:
    def test_adjust_saturation(self, tmp_path: Path) -> None:
        inp = _make_rgba_png()
        out = tmp_path / "out.png"
        adjust_saturation(str(inp), str(out), factor=0.0)
        assert out.exists()
        _, img = load_image(str(out))
        assert img.mode == "RGBA"

    def test_adjust_saturation_invalid_factor(self, tmp_path: Path) -> None:
        inp = _make_png()
        out = tmp_path / "out.png"
        with pytest.raises(ToolError, match="factor must be >= 0"):
            adjust_saturation(str(inp), str(out), factor=-1.0)

    def test_convert_color_space_gray(self, tmp_path: Path) -> None:
        inp = _make_png()
        out = tmp_path / "out.png"
        convert_color_space(str(inp), str(out), target_space="GRAY")
        assert out.exists()
        _, img = load_image(str(out))
        assert img.mode == "L"

    def test_convert_color_space_grey_alias(self, tmp_path: Path) -> None:
        inp = _make_png()
        out = tmp_path / "out.png"
        convert_color_space(str(inp), str(out), target_space="grey")
        assert out.exists()
        _, img = load_image(str(out))
        assert img.mode == "L"

    def test_convert_color_space_rgb(self, tmp_path: Path) -> None:
        inp = _make_png(mode="RGBA")
        out = tmp_path / "out.png"
        convert_color_space(str(inp), str(out), target_space="RGB")
        assert out.exists()
        _, img = load_image(str(out))
        assert img.mode == "RGB"

    def test_convert_color_space_cmyk_tiff(self, tmp_path: Path) -> None:
        inp = _make_png()
        out = tmp_path / "out.tif"
        convert_color_space(str(inp), str(out), target_space="CMYK")
        assert out.exists()
        _, img = load_image(str(out))
        assert img.mode == "CMYK"

    def test_convert_color_space_cmyk_jpeg_fails(self, tmp_path: Path) -> None:
        inp = _make_png()
        out = tmp_path / "out.jpg"
        with pytest.raises(ToolError, match="requires a lossless format"):
            convert_color_space(str(inp), str(out), target_space="CMYK")

    def test_convert_color_space_invalid(self, tmp_path: Path) -> None:
        inp = _make_png()
        out = tmp_path / "out.png"
        with pytest.raises(ToolError, match="Unknown target_space"):
            convert_color_space(str(inp), str(out), target_space="UNKNOWN")

    def test_adjust_white_balance_unchanged(self, tmp_path: Path) -> None:
        inp = _make_png()
        out = tmp_path / "out.png"
        res = adjust_white_balance(str(inp), str(out), temperature_shift=0, tint_shift=0)
        assert "unchanged" in res
        assert out.exists()

    def test_adjust_white_balance_warm_magenta(self, tmp_path: Path) -> None:
        inp = _make_rgba_png()
        out = tmp_path / "out.png"
        adjust_white_balance(str(inp), str(out), temperature_shift=20, tint_shift=10)
        assert out.exists()
        _, img = load_image(str(out))
        assert img.mode == "RGBA"

    def test_adjust_white_balance_invalid_shifts(self, tmp_path: Path) -> None:
        inp = _make_png()
        out = tmp_path / "out.png"
        with pytest.raises(ToolError, match="temperature_shift must be in -100..100"):
            adjust_white_balance(str(inp), str(out), temperature_shift=150)
        with pytest.raises(ToolError, match="tint_shift must be in -100..100"):
            adjust_white_balance(str(inp), str(out), tint_shift=-150)

    def test_color_tint(self, tmp_path: Path) -> None:
        inp = _make_rgba_png()
        out = tmp_path / "out.png"
        color_tint(str(inp), str(out), hex_color="#00FF00", strength=0.4)
        assert out.exists()
        _, img = load_image(str(out))
        assert img.mode == "RGBA"

    def test_color_tint_boundaries(self, tmp_path: Path) -> None:
        inp = _make_png()
        out0 = tmp_path / "out0.png"
        out1 = tmp_path / "out1.png"
        color_tint(str(inp), str(out0), hex_color="#FF0000", strength=0.0)
        color_tint(str(inp), str(out1), hex_color="#FF0000", strength=1.0)
        assert out0.exists()
        assert out1.exists()

    def test_color_tint_invalid_strength(self, tmp_path: Path) -> None:
        inp = _make_png()
        out = tmp_path / "out.png"
        with pytest.raises(ToolError, match="strength must be in 0..1"):
            color_tint(str(inp), str(out), hex_color="#FF0000", strength=1.5)


# ---------------------------------------------------------------------------
# 4. Detail Tools
# ---------------------------------------------------------------------------


class TestDetailTools:
    def test_sharpen_image(self, tmp_path: Path) -> None:
        inp = _make_rgba_png()
        out = tmp_path / "out.png"
        sharpen_image(str(inp), str(out), radius=1.5, percent=120, threshold=2)
        assert out.exists()

    def test_sharpen_image_invalid_params(self, tmp_path: Path) -> None:
        inp = _make_png()
        out = tmp_path / "out.png"
        with pytest.raises(ToolError, match="radius must be > 0"):
            sharpen_image(str(inp), str(out), radius=0)
        with pytest.raises(ToolError, match="percent must be >= 0"):
            sharpen_image(str(inp), str(out), percent=-10)
        with pytest.raises(ToolError, match="threshold must be in 0..255"):
            sharpen_image(str(inp), str(out), threshold=300)

    def test_blur_image_types(self, tmp_path: Path) -> None:
        inp = _make_png()
        for btype in ("gaussian", "box", "median"):
            out = tmp_path / f"out_{btype}.png"
            blur_image(str(inp), str(out), blur_type=btype, radius=3)
            assert out.exists()

    def test_blur_image_invalid(self, tmp_path: Path) -> None:
        inp = _make_png()
        out = tmp_path / "out.png"
        with pytest.raises(ToolError, match="Unknown blur_type"):
            blur_image(str(inp), str(out), blur_type="motion")
        with pytest.raises(ToolError, match="radius must be >= 1"):
            blur_image(str(inp), str(out), radius=0)

    def test_denoise_image_rgb(self, tmp_path: Path) -> None:
        inp = _make_png()
        out = tmp_path / "out.png"
        denoise_image(str(inp), str(out), strength=15)
        assert out.exists()

    def test_denoise_image_rgba(self, tmp_path: Path) -> None:
        inp = _make_rgba_png()
        out = tmp_path / "out.png"
        denoise_image(str(inp), str(out), strength=10)
        assert out.exists()
        _, img = load_image(str(out))
        assert img.mode == "RGBA"

    def test_denoise_image_invalid_strength(self, tmp_path: Path) -> None:
        inp = _make_png()
        out = tmp_path / "out.png"
        with pytest.raises(ToolError, match="strength must be in 1..100"):
            denoise_image(str(inp), str(out), strength=0)
        with pytest.raises(ToolError, match="strength must be in 1..100"):
            denoise_image(str(inp), str(out), strength=101)

    def test_edge_detection(self, tmp_path: Path) -> None:
        inp = _make_png()
        out = tmp_path / "out.png"
        edge_detection(str(inp), str(out), threshold1=50, threshold2=150)
        assert out.exists()
        _, img = load_image(str(out))
        assert img.mode == "L"

    def test_edge_detection_invalid_thresholds(self, tmp_path: Path) -> None:
        inp = _make_png()
        out = tmp_path / "out.png"
        with pytest.raises(ToolError, match="threshold1 must be in 0..255"):
            edge_detection(str(inp), str(out), threshold1=-1, threshold2=100)
        with pytest.raises(ToolError, match="must be greater than threshold1"):
            edge_detection(str(inp), str(out), threshold1=150, threshold2=100)


class TestImageDifference:
    def test_image_difference_identical(self, tmp_path: Path) -> None:
        """Test that identical images produce zero difference and an opaque image."""
        img = Image.new("RGBA", (50, 50), (128, 64, 32, 255))
        p1 = tmp_path / "img1.png"
        out = tmp_path / "output.png"
        img.save(p1)

        result = image_difference(str(p1), str(p1), str(out))

        stats = json.loads(result)
        assert stats["different_pixels"] == 0
        assert stats["max_difference"] == 0
        assert stats["difference_ratio"] == 0.0
        assert out.exists()

        out_img = Image.open(out)
        assert out_img.getpixel((25, 25)) == (0, 0, 0, 255)  # Black and opaque

    def test_image_difference_partial_diff_rgb_channels(self, tmp_path: Path) -> None:
        """Test difference in green channel only is detected (multi-channel check)."""
        img1 = Image.new("RGBA", (50, 50), (100, 100, 100, 255))
        img2 = Image.new("RGBA", (50, 50), (100, 150, 100, 255))  # diff of 50 in G

        p1 = tmp_path / "img1.png"
        p2 = tmp_path / "img2.png"
        out = tmp_path / "output.png"
        img1.save(p1)
        img2.save(p2)

        result = image_difference(str(p1), str(p2), str(out), threshold=10.0)

        stats = json.loads(result)
        assert stats["different_pixels"] == 2500
        assert stats["max_difference"] == 50
        assert stats["difference_ratio"] == 1.0

    def test_image_difference_alpha_channel(self, tmp_path: Path) -> None:
        """Test difference in alpha channel alone is detected."""
        img1 = Image.new("RGBA", (50, 50), (100, 100, 100, 255))
        img2 = Image.new("RGBA", (50, 50), (100, 100, 100, 200))  # diff of 55 in alpha

        p1 = tmp_path / "img1.png"
        p2 = tmp_path / "img2.png"
        out = tmp_path / "output.png"
        img1.save(p1)
        img2.save(p2)

        result = image_difference(str(p1), str(p2), str(out), threshold=10.0)

        stats = json.loads(result)
        assert stats["different_pixels"] == 2500
        assert stats["max_difference"] == 55
        assert stats["difference_ratio"] == 1.0

    def test_image_difference_threshold_filtering(self, tmp_path: Path) -> None:
        """Test that threshold filters out small differences."""
        img1 = Image.new("RGBA", (50, 50), (100, 100, 100, 255))
        img2 = Image.new("RGBA", (50, 50), (102, 102, 102, 255))  # diff of 2

        p1 = tmp_path / "img1.png"
        p2 = tmp_path / "img2.png"
        out = tmp_path / "output.png"
        img1.save(p1)
        img2.save(p2)

        result = image_difference(str(p1), str(p2), str(out), threshold=10.0)

        stats = json.loads(result)
        assert stats["different_pixels"] == 0
        assert stats["max_difference"] == 2

    def test_image_difference_size_mismatch(self, tmp_path: Path) -> None:
        """Test that differing dimensions raise ToolError."""
        img1 = Image.new("RGBA", (100, 100), (255, 0, 0, 255))
        img2 = Image.new("RGBA", (50, 50), (0, 255, 0, 255))

        p1 = tmp_path / "img1.png"
        p2 = tmp_path / "img2.png"
        out = tmp_path / "output.png"
        img1.save(p1)
        img2.save(p2)

        with pytest.raises(ToolError, match="dimensions differ"):
            image_difference(str(p1), str(p2), str(out))

    def test_image_difference_absolute_mode(self, tmp_path: Path) -> None:
        """Test absolute mode retains actual color differences on opaque canvas."""
        img1 = Image.new("RGBA", (50, 50), (100, 50, 25, 255))
        img2 = Image.new("RGBA", (50, 50), (150, 50, 25, 255))  # R differs by 50

        p1 = tmp_path / "img1.png"
        p2 = tmp_path / "img2.png"
        out = tmp_path / "output.png"
        img1.save(p1)
        img2.save(p2)

        image_difference(str(p1), str(p2), str(out), mode="absolute", threshold=10.0)

        out_img = Image.open(out)
        assert out_img.getpixel((25, 25)) == (50, 0, 0, 255)

    def test_image_difference_relative_mode(self, tmp_path: Path) -> None:
        """Test relative mode scales subtle difference to 255."""
        img1 = Image.new("RGBA", (50, 50), (100, 100, 100, 255))
        img2 = Image.new("RGBA", (50, 50), (110, 100, 100, 255))  # diff of 10

        p1 = tmp_path / "img1.png"
        p2 = tmp_path / "img2.png"
        out = tmp_path / "output.png"
        img1.save(p1)
        img2.save(p2)

        image_difference(str(p1), str(p2), str(out), mode="relative", threshold=5.0)

        out_img = Image.open(out)
        # 10 scaled to 255
        pixel = out_img.getpixel((25, 25))
        assert pixel[0] == 255
        assert pixel[1] == 0
        assert pixel[2] == 0
        assert pixel[3] == 255

    def test_image_difference_highlight_mode(self, tmp_path: Path) -> None:
        """Test highlight mode produces pure white (255, 255, 255, 255) on black."""
        img1 = Image.new("RGBA", (50, 50), (100, 100, 100, 255))
        img2 = Image.new("RGBA", (50, 50), (200, 50, 150, 255))

        p1 = tmp_path / "img1.png"
        p2 = tmp_path / "img2.png"
        out = tmp_path / "output.png"
        img1.save(p1)
        img2.save(p2)

        image_difference(str(p1), str(p2), str(out), mode="highlight")

        out_img = Image.open(out)
        assert out_img.getpixel((25, 25)) == (255, 255, 255, 255)

    def test_image_difference_invalid_mode(self, tmp_path: Path) -> None:
        """Test that invalid mode raises ToolError."""
        p = _make_png()
        out = tmp_path / "output.png"

        with pytest.raises(ToolError, match="mode must be one of"):
            image_difference(str(p), str(p), str(out), mode="invalid")

    def test_image_difference_invalid_threshold_range(self, tmp_path: Path) -> None:
        """Test that threshold outside 0..255 raises ToolError."""
        p = _make_png()
        out = tmp_path / "output.png"

        with pytest.raises(ToolError, match="threshold must be in 0..255"):
            image_difference(str(p), str(p), str(out), threshold=-1.0)

        with pytest.raises(ToolError, match="threshold must be in 0..255"):
            image_difference(str(p), str(p), str(out), threshold=300.0)

    def test_image_difference_invalid_threshold_type(self, tmp_path: Path) -> None:
        """Test that non-numeric threshold raises ToolError."""
        p = _make_png()
        out = tmp_path / "output.png"

        with pytest.raises(ToolError, match="threshold must be a valid number"):
            image_difference(str(p), str(p), str(out), threshold="not-a-number")  # type: ignore[arg-type]

    def test_image_difference_jpeg_output(self, tmp_path: Path) -> None:
        """Test saving difference visualization to JPEG."""
        img1 = Image.new("RGB", (40, 40), (255, 0, 0))
        img2 = Image.new("RGB", (40, 40), (0, 0, 255))
        p1 = tmp_path / "img1.png"
        p2 = tmp_path / "img2.png"
        out = tmp_path / "output.jpg"
        img1.save(p1)
        img2.save(p2)

        image_difference(str(p1), str(p2), str(out), mode="highlight")

        assert out.exists()
        out_img = Image.open(out)
        assert out_img.format == "JPEG"


class _MockIO:
    def __init__(self, name: str) -> None:
        self.name = name


class _MockUpscalerSession:
    def __init__(self, scale: int = 4) -> None:
        self.scale = scale

    def get_inputs(self):
        return [_MockIO("input")]

    def get_outputs(self):
        return [_MockIO("output")]

    def run(self, output_names, input_feed):
        tensor_in = list(input_feed.values())[0]
        out_tensor = np.repeat(np.repeat(tensor_in, self.scale, axis=2), self.scale, axis=3)
        return [out_tensor]


class TestUpscaleImage:
    def test_detect_image_type_diagram(self) -> None:
        """Diagram with straight lines and text is routed to swinir classical."""
        arr = np.full((256, 256, 3), 255, dtype=np.uint8)
        cv2.rectangle(arr, (30, 30), (220, 220), (0, 0, 0), 2)
        cv2.putText(arr, "MCP Architecture", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        img = Image.fromarray(arr)
        engine, model, desc = _detect_image_type(img)
        assert engine == "swinir"
        assert model == "classical"
        assert "diagram" in desc or "text" in desc

    def test_detect_image_type_anime(self) -> None:
        """Flat cel-shaded illustration with low color entropy is routed to real-esrgan anime."""
        arr = np.full((256, 256, 3), (255, 200, 200), dtype=np.uint8)
        cv2.circle(arr, (128, 128), 60, (100, 50, 150), -1)
        img = Image.fromarray(arr)
        engine, model, desc = _detect_image_type(img)
        assert engine == "real-esrgan"
        assert model == "anime"
        assert "anime" in desc

    def test_detect_image_type_photo_clean(self) -> None:
        """High-resolution photographic texture is routed to real-esrgan photo."""
        rng = np.random.default_rng(42)
        r = cv2.GaussianBlur(rng.integers(0, 256, (600, 600), dtype=np.uint8), (5, 5), 1.5)
        g = cv2.GaussianBlur(rng.integers(0, 256, (600, 600), dtype=np.uint8), (5, 5), 1.5)
        b = cv2.GaussianBlur(rng.integers(0, 256, (600, 600), dtype=np.uint8), (5, 5), 1.5)
        img = Image.fromarray(np.stack([r, g, b], axis=-1))
        engine, model, desc = _detect_image_type(img)
        assert engine == "real-esrgan"
        assert model == "photo"
        assert "photo (clean)" in desc

    def test_detect_image_type_photo_compressed(self) -> None:
        """Lower-resolution photographic texture (<500px) is routed to real-esrgan general."""
        rng = np.random.default_rng(42)
        r = cv2.GaussianBlur(rng.integers(0, 256, (300, 300), dtype=np.uint8), (5, 5), 1.5)
        g = cv2.GaussianBlur(rng.integers(0, 256, (300, 300), dtype=np.uint8), (5, 5), 1.5)
        b = cv2.GaussianBlur(rng.integers(0, 256, (300, 300), dtype=np.uint8), (5, 5), 1.5)
        img = Image.fromarray(np.stack([r, g, b], axis=-1))
        engine, model, desc = _detect_image_type(img)
        assert engine == "real-esrgan"
        assert model == "general"
        assert "compressed" in desc

    def test_upscale_image_invalid_scale(self, tmp_path: Path) -> None:
        inp = _make_png((20, 20))
        out = tmp_path / "out.png"
        with pytest.raises(ToolError, match="scale must be 2 or 4"):
            upscale_image(str(inp), str(out), scale=3)

    def test_upscale_image_invalid_engine(self, tmp_path: Path) -> None:
        inp = _make_png((20, 20))
        out = tmp_path / "out.png"
        with pytest.raises(ToolError, match="engine must be one of"):
            upscale_image(str(inp), str(out), engine="invalid")

    def test_upscale_image_invalid_model(self, tmp_path: Path) -> None:
        inp = _make_png((20, 20))
        out = tmp_path / "out.png"
        with pytest.raises(ToolError, match="model_name must be one of"):
            upscale_image(str(inp), str(out), engine="real-esrgan", model_name="invalid")

    def test_upscale_image_invalid_tile_parameters(self, tmp_path: Path) -> None:
        inp = _make_png((20, 20))
        out = tmp_path / "out.png"
        with pytest.raises(ToolError, match="tile_size must be >= 0"):
            upscale_image(str(inp), str(out), tile_size=-1)
        with pytest.raises(ToolError, match="tile_pad must be >= 0"):
            upscale_image(str(inp), str(out), tile_pad=-5)

    def test_upscale_image_dos_protection(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        import tools.detail as detail_mod

        inp = _make_png((20, 20))
        out = tmp_path / "out.png"
        # 20x20 upscaled 4x is 80x80 = 6,400 pixels
        monkeypatch.setattr(detail_mod, "MAX_PIXELS", 500)
        with pytest.raises(ToolError, match="Upscaled image is too large"):
            upscale_image(str(inp), str(out), scale=4)

    def test_upscale_image_mock_inference_direct(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import tools.detail as detail_mod

        monkeypatch.setattr(
            detail_mod,
            "_get_upscaler_session",
            lambda engine, model, scale: _MockUpscalerSession(scale=scale),
        )

        inp = _make_png((24, 18), color=(200, 100, 50))
        out = tmp_path / "out.png"

        res = upscale_image(str(inp), str(out), scale=4, tile_size=0)
        assert out.exists()
        out_img = Image.open(out)
        assert out_img.size == (96, 72)
        assert "[24x18 -> 96x72]" in res

    def test_upscale_image_mock_inference_tiled(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import tools.detail as detail_mod

        monkeypatch.setattr(
            detail_mod,
            "_get_upscaler_session",
            lambda engine, model, scale: _MockUpscalerSession(scale=scale),
        )

        inp = _make_png((48, 48), color=(120, 150, 180))
        out = tmp_path / "out.png"

        res = upscale_image(str(inp), str(out), scale=2, tile_size=20, tile_pad=4)
        assert out.exists()
        out_img = Image.open(out)
        assert out_img.size == (96, 96)
        assert "[48x48 -> 96x96]" in res

    def test_upscale_image_arbitrary_dimensions_and_window_padding(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Verify that arbitrary odd dimensions not divisible by window size (8) upscale cleanly."""
        import tools.detail as detail_mod

        class _AssertingWindowMockSession(_MockUpscalerSession):
            def run(self, output_names, input_feed):
                tensor_in = list(input_feed.values())[0]
                _, _, h, w = tensor_in.shape
                # Transformer attention requires dimensions to be divisible by window size 8
                assert h % 8 == 0, f"Height {h} not divisible by 8"
                assert w % 8 == 0, f"Width {w} not divisible by 8"
                return super().run(output_names, input_feed)

        monkeypatch.setattr(
            detail_mod,
            "_get_upscaler_session",
            lambda engine, model, scale: _AssertingWindowMockSession(scale=scale),
        )

        inp = _make_png((37, 53), color=(80, 100, 120))
        out = tmp_path / "out.png"

        # Test untiled
        res1 = upscale_image(str(inp), str(out), scale=4, tile_size=0)
        assert "[37x53 -> 148x212]" in res1
        out1 = Image.open(out)
        assert out1.size == (148, 212)

        # Test tiled
        out2_path = tmp_path / "out2.png"
        res2 = upscale_image(str(inp), str(out2_path), scale=2, tile_size=20, tile_pad=5)
        assert "[37x53 -> 74x106]" in res2
        out2 = Image.open(out2_path)
        assert out2.size == (74, 106)

    def test_upscale_image_preserves_alpha(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import tools.detail as detail_mod

        monkeypatch.setattr(
            detail_mod,
            "_get_upscaler_session",
            lambda engine, model, scale: _MockUpscalerSession(scale=scale),
        )

        # Create RGBA image with varied alpha
        rgba_img = Image.new("RGBA", (20, 20), (255, 0, 0, 128))
        inp = tmp_path / "transparent.png"
        rgba_img.save(inp)
        out = tmp_path / "upscaled.png"

        upscale_image(str(inp), str(out), scale=2)
        assert out.exists()
        out_img = Image.open(out)
        assert out_img.mode == "RGBA"
        assert out_img.size == (40, 40)
        # Check alpha channel preserved
        assert out_img.getchannel("A").getpixel((10, 10)) == 128

    def test_upscale_image_auto_engine_description(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import tools.detail as detail_mod

        monkeypatch.setattr(
            detail_mod,
            "_get_upscaler_session",
            lambda engine, model, scale: _MockUpscalerSession(scale=scale),
        )

        arr = np.full((100, 100, 3), 255, dtype=np.uint8)
        cv2.rectangle(arr, (10, 10), (90, 90), (0, 0, 0), 2)
        inp = tmp_path / "diagram.png"
        Image.fromarray(arr).save(inp)
        out = tmp_path / "upscaled.png"

        res = upscale_image(str(inp), str(out), scale=4, engine="auto")
        assert "[auto-detected:" in res
        assert "swinir (classical)" in res

    def test_upscale_image_explicit_engine_and_model(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import tools.detail as detail_mod

        monkeypatch.setattr(
            detail_mod,
            "_get_upscaler_session",
            lambda engine, model, scale: _MockUpscalerSession(scale=scale),
        )

        inp = _make_png((30, 30))
        out = tmp_path / "out.png"

        res = upscale_image(
            str(inp),
            str(out),
            scale=2,
            engine="swinir",
            model_name="classical",
        )
        assert "via swinir (classical)" in res


# ---------------------------------------------------------------------------
# 5. Composite Tools
# ---------------------------------------------------------------------------


class TestCompositeTools:
    def test_overlay_image(self, tmp_path: Path) -> None:
        bg = _make_png((200, 200), color=(100, 100, 100))
        ov = _make_rgba_png((50, 50))
        out = tmp_path / "out.png"
        overlay_image(
            str(bg),
            str(ov),
            str(out),
            position="top-right",
            offset_x=10,
            offset_y=15,
            opacity=0.8,
        )
        assert out.exists()
        _, img = load_image(str(out))
        assert img.size == (200, 200)

    def test_overlay_image_auto_downscales_oversized(self, tmp_path: Path) -> None:
        bg = _make_png((100, 100))
        ov = _make_png((200, 200))
        out = tmp_path / "out.png"
        overlay_image(str(bg), str(ov), str(out), position="center")
        assert out.exists()

    def test_overlay_image_invalid_params(self, tmp_path: Path) -> None:
        bg = _make_png()
        ov = _make_png()
        out = tmp_path / "out.png"
        with pytest.raises(ToolError, match="opacity must be in 0..1"):
            overlay_image(str(bg), str(ov), str(out), opacity=1.5)
        with pytest.raises(ToolError, match="offsets must be >= 0"):
            overlay_image(str(bg), str(ov), str(out), offset_x=-5)
        with pytest.raises(ToolError, match="Unknown position"):
            overlay_image(str(bg), str(ov), str(out), position="invalid-pos")

    def test_overlay_text(self, tmp_path: Path) -> None:
        inp = _make_png((200, 100))
        out = tmp_path / "out.png"
        overlay_text(
            str(inp),
            str(out),
            text="Hello MCP",
            position="bottom-left",
            font_size=20,
            font_color="#FFFF00",
        )
        assert out.exists()

    def test_overlay_text_invalid(self, tmp_path: Path) -> None:
        inp = _make_png()
        out = tmp_path / "out.png"
        with pytest.raises(ToolError, match="text must be a non-empty string"):
            overlay_text(str(inp), str(out), text="")
        with pytest.raises(ToolError, match="font_size must be >= 1"):
            overlay_text(str(inp), str(out), text="Test", font_size=0)
        with pytest.raises(ToolError, match="offsets must be >= 0"):
            overlay_text(str(inp), str(out), text="Test", offset_y=-10)

    @pytest.mark.skipif(
        importlib.util.find_spec("rembg") is None
        or not Path("~/.u2net/u2net.onnx").expanduser().exists(),
        reason="rembg not installed or u2net.onnx not cached locally",
    )
    def test_remove_background_ai(self, tmp_path: Path) -> None:
        """Test AI-based background removal produces a valid RGBA output."""
        img = Image.new("RGBA", (100, 100), (0, 200, 0, 255))
        pixels = img.load()
        for y in range(30, 70):
            for x in range(30, 70):
                pixels[x, y] = (255, 0, 0, 255)

        inp = tmp_path / "input.png"
        out = tmp_path / "output.png"
        img.save(inp)

        result = remove_background(str(inp), str(out), method="ai")
        assert out.exists()

        result_img = Image.open(out)
        assert result_img.mode == "RGBA"

        # Red square area has high alpha (kept)
        center_alpha = result_img.getpixel((50, 50))[3]
        assert center_alpha > 128, f"Center pixel alpha should be high, got {center_alpha}"

        # Green corner area has low alpha (removed)
        corner_alpha = result_img.getpixel((10, 10))[3]
        assert corner_alpha < 128, f"Corner pixel alpha should be low, got {corner_alpha}"

    def test_remove_background_chroma(self, tmp_path: Path) -> None:
        """Test chroma-key background removal makes key color transparent and preserves foreground."""
        # Green background (#00FF00) with a centered red square
        img = Image.new("RGBA", (100, 100), (0, 255, 0, 255))
        pixels = img.load()
        for y in range(30, 70):
            for x in range(30, 70):
                pixels[x, y] = (255, 0, 0, 255)

        inp = tmp_path / "input.png"
        out = tmp_path / "output.png"
        img.save(inp)

        result = remove_background(
            str(inp), str(out), method="chroma", key_color="#00FF00", margin=0
        )
        assert out.exists()

        result_img = Image.open(out)
        assert result_img.mode == "RGBA"

        # Red square must remain opaque
        assert result_img.getpixel((50, 50))[3] == 255
        # Green background must be transparent
        assert result_img.getpixel((5, 5))[3] == 0

    def test_remove_background_chroma_white_background(self, tmp_path: Path) -> None:
        """Test chroma-key with a solid white product background."""
        img = Image.new("RGBA", (100, 100), (255, 255, 255, 255))
        pixels = img.load()
        for y in range(30, 70):
            for x in range(30, 70):
                pixels[x, y] = (0, 0, 128, 255)

        inp = tmp_path / "input.png"
        out = tmp_path / "output.png"
        img.save(inp)

        result = remove_background(
            str(inp), str(out), method="chroma", key_color="#FFFFFF", margin=0, tolerance=20
        )
        assert out.exists()

        result_img = Image.open(out)
        assert result_img.getpixel((50, 50))[3] == 255
        assert result_img.getpixel((5, 5))[3] == 0

    def test_remove_background_chroma_preserves_existing_alpha(self, tmp_path: Path) -> None:
        """Test that already-transparent regions remain transparent after chroma removal."""
        img = Image.new("RGBA", (100, 100), (0, 255, 0, 255))
        pixels = img.load()
        # Foreground square
        for y in range(30, 70):
            for x in range(30, 70):
                pixels[x, y] = (255, 0, 0, 255)
        # Pre-existing transparent strip in top-left
        for y in range(0, 15):
            for x in range(0, 15):
                pixels[x, y] = (0, 0, 0, 0)

        inp = tmp_path / "input.png"
        out = tmp_path / "output.png"
        img.save(inp)

        remove_background(str(inp), str(out), method="chroma", key_color="#00FF00", margin=0)
        result_img = Image.open(out)
        # Pre-existing transparent area must stay transparent
        assert result_img.getpixel((5, 5))[3] == 0
        # Foreground must stay opaque
        assert result_img.getpixel((50, 50))[3] == 255

    def test_remove_background_invalid_output_format(self, tmp_path: Path) -> None:
        """Test that non-transparency output format (e.g. .jpg) raises ToolError."""
        img = Image.new("RGBA", (50, 50), (0, 200, 0, 255))
        inp = tmp_path / "input.png"
        out = tmp_path / "output.jpg"
        img.save(inp)

        with pytest.raises(ToolError, match="format supporting transparency"):
            remove_background(str(inp), str(out))

    def test_remove_background_invalid_margin(self, tmp_path: Path) -> None:
        """Test that margin outside 0–50 raises ToolError."""
        img = Image.new("RGBA", (50, 50), (0, 200, 0, 255))
        inp = tmp_path / "input.png"
        out = tmp_path / "output.png"
        img.save(inp)

        with pytest.raises(ToolError, match="margin"):
            remove_background(str(inp), str(out), margin=100)

    def test_remove_background_invalid_tolerance(self, tmp_path: Path) -> None:
        """Test that tolerance outside 1–150 raises ToolError."""
        img = Image.new("RGBA", (50, 50), (0, 200, 0, 255))
        inp = tmp_path / "input.png"
        out = tmp_path / "output.png"
        img.save(inp)

        with pytest.raises(ToolError, match="tolerance"):
            remove_background(str(inp), str(out), method="chroma", tolerance=200)

    def test_remove_background_invalid_method(self, tmp_path: Path) -> None:
        """Test that invalid method raises ToolError."""
        img = Image.new("RGBA", (50, 50), (0, 200, 0, 255))
        inp = tmp_path / "input.png"
        out = tmp_path / "output.png"
        img.save(inp)

        with pytest.raises(ToolError, match="method"):
            remove_background(str(inp), str(out), method="invalid")

    def test_remove_background_invalid_key_color(self, tmp_path: Path) -> None:
        """Test that invalid hex key_color raises ToolError."""
        img = Image.new("RGBA", (50, 50), (0, 200, 0, 255))
        inp = tmp_path / "input.png"
        out = tmp_path / "output.png"
        img.save(inp)

        with pytest.raises(ToolError, match="Invalid hex color"):
            remove_background(str(inp), str(out), method="chroma", key_color="not-a-color")


class TestBulkWatermark:
    def test_bulk_watermark_text(self, tmp_path: Path) -> None:
        inp1 = _make_png((100, 100))
        inp2 = _make_png((150, 100))
        out_dir = tmp_path / "watermarked_text"
        msg = bulk_watermark(
            [str(inp1), str(inp2)],
            str(out_dir),
            text="COPYRIGHT",
            position="bottom-right",
            opacity=0.8,
        )
        assert "Bulk watermarked 2/2" in msg
        assert (out_dir / inp1.name).exists()
        assert (out_dir / inp2.name).exists()

    def test_bulk_watermark_image(self, tmp_path: Path) -> None:
        bg = _make_png((200, 200))
        logo = _make_rgba_png((40, 40))
        out_dir = tmp_path / "watermarked_logo"
        msg = bulk_watermark(
            [str(bg)],
            str(out_dir),
            watermark_image_path=str(logo),
            position="center",
            opacity=0.5,
            format="webp",
            strip_metadata=True,
        )
        assert "Bulk watermarked 1/1" in msg
        out_file = out_dir / (bg.stem + ".webp")
        assert out_file.exists()

    def test_bulk_watermark_missing_source(self, tmp_path: Path) -> None:
        inp = _make_png((50, 50))
        with pytest.raises(ToolError, match="Either watermark_image_path or text"):
            bulk_watermark([str(inp)], str(tmp_path / "out"))


class TestBulkRemoveBackground:
    def test_bulk_remove_bg_chroma(self, tmp_path: Path) -> None:
        # Create image with solid green background
        raw = Image.new("RGB", (60, 60), (0, 255, 0))
        for x in range(20, 40):
            for y in range(20, 40):
                raw.putpixel((x, y), (255, 0, 0))  # red square in center
        inp = tmp_path / "chroma.png"
        raw.save(inp)

        out_dir = tmp_path / "bg_removed"
        msg = bulk_remove_background(
            [str(inp)],
            str(out_dir),
            method="chroma",
            key_color="#00FF00",
            tolerance=30,
            format="png",
        )
        assert "Bulk removed backgrounds for 1/1" in msg
        out_file = out_dir / "chroma.png"
        assert out_file.exists()
        _, img = load_image(str(out_file))
        assert img.mode == "RGBA"
        # Corner should be transparent
        alpha = img.getchannel("A")
        assert alpha.getpixel((0, 0)) == 0

    def test_bulk_remove_bg_invalid_format(self, tmp_path: Path) -> None:
        inp = _make_png((50, 50))
        with pytest.raises(ToolError, match="transparency"):
            bulk_remove_background([str(inp)], str(tmp_path / "out"), format="jpeg")


class TestImageBlend:
    def test_image_blend_equal_weights(self, tmp_path: Path) -> None:
        """Test 50/50 blend produces intermediate color (128, 0, 128)."""
        img1 = Image.new("RGBA", (50, 50), (255, 0, 0, 255))  # red
        img2 = Image.new("RGBA", (50, 50), (0, 0, 255, 255))  # blue

        p1 = tmp_path / "img1.png"
        p2 = tmp_path / "img2.png"
        out = tmp_path / "output.png"
        img1.save(p1)
        img2.save(p2)

        result = image_blend(str(p1), str(p2), str(out), alpha=0.5)

        assert out.exists()
        assert "Blended images (alpha=0.5)" in result
        result_img = Image.open(out)
        pixel = result_img.getpixel((25, 25))
        assert pixel[0] == 128, f"Expected R=128, got {pixel[0]}"
        assert pixel[1] == 0, f"Expected G=0, got {pixel[1]}"
        assert pixel[2] == 128, f"Expected B=128, got {pixel[2]}"
        assert pixel[3] == 255, f"Expected A=255, got {pixel[3]}"

    def test_image_blend_full_alpha(self, tmp_path: Path) -> None:
        """Test alpha=1.0 returns image 1 unchanged."""
        img1 = Image.new("RGBA", (50, 50), (255, 0, 0, 255))
        img2 = Image.new("RGBA", (50, 50), (0, 0, 255, 255))

        p1 = tmp_path / "img1.png"
        p2 = tmp_path / "img2.png"
        out = tmp_path / "output.png"
        img1.save(p1)
        img2.save(p2)

        image_blend(str(p1), str(p2), str(out), alpha=1.0)

        result_img = Image.open(out)
        assert result_img.getpixel((25, 25)) == (255, 0, 0, 255)

    def test_image_blend_zero_alpha(self, tmp_path: Path) -> None:
        """Test alpha=0.0 returns image 2 unchanged."""
        img1 = Image.new("RGBA", (50, 50), (255, 0, 0, 255))
        img2 = Image.new("RGBA", (50, 50), (0, 0, 255, 255))

        p1 = tmp_path / "img1.png"
        p2 = tmp_path / "img2.png"
        out = tmp_path / "output.png"
        img1.save(p1)
        img2.save(p2)

        image_blend(str(p1), str(p2), str(out), alpha=0.0)

        result_img = Image.open(out)
        assert result_img.getpixel((25, 25)) == (0, 0, 255, 255)

    def test_image_blend_resize_differing_dims(self, tmp_path: Path) -> None:
        """Test that resize=True auto-sizes image 2 to match image 1 dimensions."""
        img1 = Image.new("RGBA", (100, 80), (255, 0, 0, 255))
        img2 = Image.new("RGBA", (50, 40), (0, 0, 255, 255))

        p1 = tmp_path / "img1.png"
        p2 = tmp_path / "img2.png"
        out = tmp_path / "output.png"
        img1.save(p1)
        img2.save(p2)

        image_blend(str(p1), str(p2), str(out), resize=True)

        assert out.exists()
        result_img = Image.open(out)
        assert result_img.size == (100, 80)

    def test_image_blend_no_resize_mismatch_raises(self, tmp_path: Path) -> None:
        """Test that mismatched dimensions without resize raises ToolError."""
        img1 = Image.new("RGBA", (100, 100), (255, 0, 0, 255))
        img2 = Image.new("RGBA", (50, 50), (0, 0, 255, 255))

        p1 = tmp_path / "img1.png"
        p2 = tmp_path / "img2.png"
        out = tmp_path / "output.png"
        img1.save(p1)
        img2.save(p2)

        with pytest.raises(ToolError, match="dimensions differ"):
            image_blend(str(p1), str(p2), str(out), resize=False)

    def test_image_blend_invalid_alpha_range(self, tmp_path: Path) -> None:
        """Test that alpha outside 0..1 raises ToolError."""
        p = _make_png()
        out = tmp_path / "output.png"

        with pytest.raises(ToolError, match="alpha must be in 0..1"):
            image_blend(str(p), str(p), str(out), alpha=1.5)

        with pytest.raises(ToolError, match="alpha must be in 0..1"):
            image_blend(str(p), str(p), str(out), alpha=-0.1)

    def test_image_blend_invalid_alpha_type(self, tmp_path: Path) -> None:
        """Test that non-numeric alpha raises ToolError."""
        p = _make_png()
        out = tmp_path / "output.png"

        with pytest.raises(ToolError, match="alpha must be a valid number"):
            image_blend(str(p), str(p), str(out), alpha="not-a-number")  # type: ignore[arg-type]

    def test_image_blend_jpeg_output(self, tmp_path: Path) -> None:
        """Test saving blended result to JPEG format."""
        p1 = _make_png((40, 40), color=(200, 100, 50))
        p2 = _make_png((40, 40), color=(50, 100, 200))
        out = tmp_path / "output.jpg"

        image_blend(str(p1), str(p2), str(out), alpha=0.5)

        assert out.exists()
        result_img = Image.open(out)
        assert result_img.format == "JPEG"
        assert result_img.size == (40, 40)

    def test_image_blend_preserves_rgb_mode(self, tmp_path: Path) -> None:
        """Test that blending two RGB images preserves mode RGB without unnecessary alpha."""
        p1 = _make_png((40, 40), mode="RGB", color=(200, 0, 0))
        p2 = _make_png((40, 40), mode="RGB", color=(0, 0, 200))
        out = tmp_path / "output.png"

        image_blend(str(p1), str(p2), str(out), alpha=0.5)

        assert out.exists()
        result_img = Image.open(out)
        assert result_img.mode == "RGB"



# ---------------------------------------------------------------------------
# 6. Encoding & Metadata Tools
# ---------------------------------------------------------------------------


class TestEncodingTools:
    def test_convert_format_png_to_jpeg(self, tmp_path: Path) -> None:
        inp = _make_png()
        out = tmp_path / "out.jpg"
        convert_format(str(inp), str(out), target_format="JPEG", quality=90)
        assert out.exists()
        _, img = load_image(str(out))
        assert img.format == "JPEG"

    def test_convert_format_to_webp(self, tmp_path: Path) -> None:
        inp = _make_png()
        out = tmp_path / "out.webp"
        convert_format(str(inp), str(out), target_format="WEBP", quality=80)
        assert out.exists()

    def test_convert_format_mismatch_extension_raises(self, tmp_path: Path) -> None:
        inp = _make_png()
        out = tmp_path / "out.png"
        with pytest.raises(ToolError, match="implies format PNG, but target_format is JPEG"):
            convert_format(str(inp), str(out), target_format="JPEG")

    def test_convert_format_unsupported(self, tmp_path: Path) -> None:
        inp = _make_png()
        out = tmp_path / "out.xyz"
        with pytest.raises(ToolError, match="Unsupported target_format"):
            convert_format(str(inp), str(out), target_format="XYZ")

    def test_convert_format_invalid_quality(self, tmp_path: Path) -> None:
        inp = _make_png()
        out = tmp_path / "out.jpg"
        with pytest.raises(ToolError, match="quality must be between 1 and 100"):
            convert_format(str(inp), str(out), target_format="JPEG", quality=150)

    def test_strip_metadata(self, tmp_path: Path) -> None:
        img = Image.new("RGB", (10, 10), (255, 0, 0))
        img.info["comment"] = "test metadata"
        inp = tmp_path / "with_meta.png"
        img.save(inp)

        out = tmp_path / "clean.png"
        strip_metadata(str(inp), str(out))
        assert out.exists()
        _, cleaned = load_image(str(out))
        assert cleaned.info == {} or "comment" not in cleaned.info

    def test_get_image_metadata(self, tmp_path: Path) -> None:
        inp = _make_png((80, 60))
        meta = get_image_metadata(str(inp))
        assert isinstance(meta, dict)
        assert meta["width"] == 80
        assert meta["height"] == 60
        assert meta["format"] == "PNG"
        assert meta["mode"] == "RGB"
        assert meta["color_depth"] == 24
        assert meta["file_size_bytes"] > 0
        assert isinstance(meta["exif"], dict)


class TestBulkConvert:
    def test_bulk_convert_webp(self, tmp_path: Path) -> None:
        inp1 = _make_png((50, 50))
        inp2 = _make_png((80, 80))
        out_dir = tmp_path / "bulk_converted"
        msg = bulk_convert(
            [str(inp1), str(inp2)],
            str(out_dir),
            target_format="webp",
            quality=85,
            strip_metadata=True,
        )
        assert "Bulk converted 2/2" in msg
        assert (out_dir / (inp1.stem + ".webp")).exists()
        assert (out_dir / (inp2.stem + ".webp")).exists()
        _, img1 = load_image(str(out_dir / (inp1.stem + ".webp")))
        assert img1.format == "WEBP"

    def test_bulk_convert_invalid_format(self, tmp_path: Path) -> None:
        inp = _make_png((50, 50))
        with pytest.raises(ToolError, match="Unsupported format"):
            bulk_convert([str(inp)], str(tmp_path / "out"), target_format="UNKNOWN")

    def test_bulk_convert_invalid_quality(self, tmp_path: Path) -> None:
        inp = _make_png((50, 50))
        with pytest.raises(ToolError, match="quality must be between 1 and 100"):
            bulk_convert([str(inp)], str(tmp_path / "out"), quality=200)


# ---------------------------------------------------------------------------
# 7. Inspection Tools (Info)
# ---------------------------------------------------------------------------


class TestInfoTools:
    def test_image_info(self, tmp_path: Path) -> None:
        inp = _make_png((120, 90))
        info = image_info(str(inp))
        assert isinstance(info, dict)
        assert info["width"] == 120
        assert info["height"] == 90
        assert info["mode"] == "RGB"
        assert info["format"] == "PNG"
        assert info["file_size_bytes"] > 0
        assert "has_icc_profile" in info


# ---------------------------------------------------------------------------
# 8. Shared Helpers & Hex Parsing
# ---------------------------------------------------------------------------


class TestHexColor:
    def test_parse_hex_from_color_module(self):
        assert color_parse_hex("#FF0000") == (255, 0, 0)

    def test_parse_hex_short(self):
        assert color_parse_hex("#F00") == (255, 0, 0)


# ---------------------------------------------------------------------------
# 9. DoS (MAX_PIXELS) Protection Across Tools
# ---------------------------------------------------------------------------


class TestDoSProtection:
    def test_load_image_rejects_oversized_images(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        import tools.io_utils as io_mod

        # Create a small 50x50 image on disk (2500 pixels)
        inp = _make_png((50, 50))
        out = tmp_path / "out.png"

        # Set MAX_PIXELS artificially low to simulate DoS trigger
        monkeypatch.setattr(io_mod, "MAX_PIXELS", 100)

        # All tool categories should fail safely with ToolError before doing work
        with pytest.raises(ToolError, match="Image is too large"):
            resize_image(str(inp), str(out), width=20)

        with pytest.raises(ToolError, match="Image is too large"):
            adjust_brightness(str(inp), str(out), factor=1.2)

        with pytest.raises(ToolError, match="Image is too large"):
            adjust_saturation(str(inp), str(out), factor=0.5)

        with pytest.raises(ToolError, match="Image is too large"):
            denoise_image(str(inp), str(out), strength=10)

        with pytest.raises(ToolError, match="Image is too large"):
            overlay_text(str(inp), str(out), text="DoS test")

        with pytest.raises(ToolError, match="(Image is too large|Upscaled image is too large)"):
            upscale_image(str(inp), str(out))
