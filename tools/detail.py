"""Detail tools: unsharp sharpening, blur, denoising, edge detection, image difference, and AI upscaling."""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from mcp.server.mcpserver.exceptions import ToolError
from PIL import Image, ImageFilter

from server import mcp
from tools.io_utils import (
    MAX_PIXELS,
    _check_size,
    _rejoin_alpha,
    _split_alpha,
    cv_bgr_from,
    cv_bgra_from,
    load_image,
    pil_from_cv_bgr,
    pil_from_cv_bgra,
    save_image,
    to_lum8,
    to_rgb,
)


@mcp.tool()
def sharpen_image(
    input_path: str,
    output_path: str,
    radius: float = 1.0,
    percent: int = 150,
    threshold: int = 3,
) -> str:
    """Sharpen an image using an unsharp mask.

    Args:
        input_path: Path to the source image.
        output_path: Destination path; extension sets the saved format.
        radius: Blur radius for the unsharp mask (0 < radius; higher =
            broader, softer sharpening). Default 1.0.
        percent: Sharpening strength as a percentage (0 = no effect,
            100 = full effect, >100 = aggressive). Default 150.
        threshold: Pixel difference threshold below which sharpening is
            not applied (0-255; higher = less noise amplification).
            Default 3.
    """
    radius = float(radius)
    percent = int(percent)
    threshold = int(threshold)
    if radius <= 0:
        raise ToolError(f"radius must be > 0, got {radius}")
    if percent < 0:
        raise ToolError(f"percent must be >= 0, got {percent}")
    if not 0 <= threshold <= 255:
        raise ToolError(f"threshold must be in 0..255, got {threshold}")
    _, img = load_image(input_path)
    _check_size(img)
    # UnsharpMask works on RGB and RGBA; convert other modes.
    if img.mode not in ("RGB", "RGBA"):
        base = img.convert("RGB")
    else:
        base = img
    out = base.filter(
        ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold)
    )
    save_image(out, output_path)
    return f"Sharpened (r={radius}, p={percent}, t={threshold}) -> {output_path}"


@mcp.tool()
def blur_image(
    input_path: str, output_path: str, blur_type: str = "gaussian", radius: int = 5
) -> str:
    """Blur an image with a Gaussian, box, or median filter.

    Args:
        input_path: Path to the source image.
        output_path: Destination path; extension sets the saved format.
        blur_type: One of "gaussian", "box", "median" (case-insensitive).
            Default "gaussian".
        radius: Filter radius in pixels (>= 1). For "median" this is
            rounded up to the next odd integer (median kernels must be
            odd-sized). Default 5.
    """
    key = blur_type.strip().lower()
    if key not in ("gaussian", "box", "median"):
        raise ToolError(
            f"Unknown blur_type '{blur_type}'. Choose one of: gaussian, box, median"
        )
    radius = int(radius)
    if radius < 1:
        raise ToolError(f"radius must be >= 1, got {radius}")
    _, img = load_image(input_path)
    _check_size(img)
    if key == "gaussian":
        out = img.filter(ImageFilter.GaussianBlur(radius=radius))
    elif key == "box":
        out = img.filter(ImageFilter.BoxBlur(radius=radius))
    else:
        size = radius if radius % 2 == 1 else radius + 1
        out = img.filter(ImageFilter.MedianFilter(size=size))
    save_image(out, output_path)
    return f"Blurred ({key}, r={radius}) -> {output_path}"


@mcp.tool()
def denoise_image(input_path: str, output_path: str, strength: int = 10) -> str:
    """Reduce noise with non-local means denoising.

    Uses OpenCV's fastNlMeansDenoisingColored, which preserves edges while
    removing Gaussian noise. Slower than simple blurs but much higher quality.

    Args:
        input_path: Path to the source image.
        output_path: Destination path; extension sets the saved format.
        strength: Denoising strength (h parameter, 1-100). Higher values
            remove more noise but may smooth fine detail. Default 10.
    """
    strength = int(strength)
    if not 1 <= strength <= 100:
        raise ToolError(f"strength must be in 1..100, got {strength}")
    _, img = load_image(input_path)
    _check_size(img)
    base, alpha = _split_alpha(img)
    bgr = cv_bgr_from(base)
    denoised = cv2.fastNlMeansDenoisingColored(
        bgr, None, h=float(strength), hColor=float(strength)
    )
    out = _rejoin_alpha(pil_from_cv_bgr(denoised), alpha)
    save_image(out, output_path)
    return f"Denoised (strength {strength}) -> {output_path}"


@mcp.tool()
def edge_detection(
    input_path: str, output_path: str, threshold1: int = 100, threshold2: int = 200
) -> str:
    """Detect edges with the Canny algorithm.

    Produces a binary edge mask (white edges on black) saved as grayscale.

    Args:
        input_path: Path to the source image.
        output_path: Destination path; extension sets the saved format.
        threshold1: Lower Canny threshold (0-255). Edges with gradient
            below this are discarded. Default 100.
        threshold2: Upper Canny threshold (0-255). Edges with gradient
            above this are kept. Must be > threshold1. Default 200.
    """
    threshold1 = int(threshold1)
    threshold2 = int(threshold2)
    if not 0 <= threshold1 <= 255:
        raise ToolError(f"threshold1 must be in 0..255, got {threshold1}")
    if not 0 <= threshold2 <= 255:
        raise ToolError(f"threshold2 must be in 0..255, got {threshold2}")
    if threshold2 <= threshold1:
        raise ToolError(
            f"threshold2 ({threshold2}) must be greater than threshold1 ({threshold1})"
        )
    _, img = load_image(input_path)
    _check_size(img)
    # to_lum8 already returns mode "L" — convert directly to numpy (no double
    # conversion through cv_bgr_from → cv2.cvtColor).
    arr = np.asarray(to_lum8(img))
    edges = cv2.Canny(arr, threshold1, threshold2)
    out = Image.fromarray(edges, mode="L")
    save_image(out, output_path)
    return f"Edges (Canny {threshold1}/{threshold2}) -> {output_path}"


@mcp.tool()
def image_difference(
    input_path_1: str,
    input_path_2: str,
    output_path: str,
    mode: str = "absolute",
    threshold: float = 10.0,
) -> str:
    """Calculate the pixel difference between two images.

    Useful for change detection, visual regression testing, and QA inspection.

    Args:
        input_path_1: First source image.
        input_path_2: Second source image.
        output_path: Destination path for the difference visualization.
        mode: Visualization mode —
            ``"absolute"`` (true color differences on black),
            ``"relative"`` (normalized differences scaled to 0–255), or
            ``"highlight"`` (pure white differences on black). Default "absolute".
        threshold: Ignore pixel differences below this value in ``0..255``
            (noise filter). Default 10.0.

    Returns:
        JSON string containing difference statistics:
        - ``output_path``: Path to the generated difference visualization image.
        - ``total_pixels``: Total number of pixels evaluated.
        - ``different_pixels``: Count of pixels exceeding or meeting the threshold.
        - ``difference_ratio``: Proportion of pixels that changed (0.0 to 1.0).
        - ``max_difference``: Maximum channel difference found (0 to 255).
    """
    mode = mode.strip().lower()
    valid_modes = ("absolute", "relative", "highlight")
    if mode not in valid_modes:
        raise ToolError(f"mode must be one of: {', '.join(valid_modes)}")

    try:
        threshold = float(threshold)
    except (TypeError, ValueError):
        raise ToolError(f"threshold must be a valid number, got {threshold!r}")

    if not 0.0 <= threshold <= 255.0:
        raise ToolError(f"threshold must be in 0..255, got {threshold}")

    _, img1 = load_image(input_path_1)
    _, img2 = load_image(input_path_2)

    if img1.size != img2.size:
        raise ToolError(
            f"Image dimensions differ ({img1.size} vs {img2.size}). "
            f"Difference calculation requires identical sizes."
        )

    bgra1 = cv_bgra_from(img1)
    bgra2 = cv_bgra_from(img2)

    # Compute raw absolute differences across all channels (BGRA)
    raw_diff = cv2.absdiff(bgra1, bgra2)

    # Pixel differs if max difference across any channel (RGB or Alpha) meets or exceeds threshold
    diff_mask = np.max(raw_diff, axis=-1) >= threshold

    total_pixels = img1.size[0] * img1.size[1]
    different_pixels = int(np.count_nonzero(diff_mask))
    max_diff = int(raw_diff.max())
    diff_ratio = different_pixels / total_pixels if total_pixels > 0 else 0.0

    # Build opaque visualization (Alpha = 255 so output is visible on any format)
    vis = np.zeros_like(raw_diff)
    vis[:, :, 3] = 255

    color_diff = raw_diff[:, :, :3]
    if mode == "absolute":
        # Seamlessly display color difference or alpha difference (as grayscale) per pixel
        vis[diff_mask, :3] = np.maximum(color_diff[diff_mask], raw_diff[diff_mask, 3:4])
    elif mode == "relative":
        if max_diff > 0:
            scale = 255.0 / max_diff
            combined = np.maximum(color_diff[diff_mask], raw_diff[diff_mask, 3:4])
            vis[diff_mask, :3] = np.clip(
                combined.astype(np.float32) * scale, 0, 255
            ).astype(np.uint8)
    elif mode == "highlight":
        vis[diff_mask, :3] = 255  # Pure white on black

    result = pil_from_cv_bgra(vis)
    saved_path = save_image(result, output_path)

    return json.dumps({
        "output_path": str(saved_path),
        "total_pixels": total_pixels,
        "different_pixels": different_pixels,
        "difference_ratio": round(diff_ratio, 6),
        "max_difference": max_diff,
    })


# ---------------------------------------------------------------------------
# AI Super-Resolution Upscaling (Real-ESRGAN & SwinIR)
# ---------------------------------------------------------------------------

_UPSCALER_SESSIONS: dict[tuple[str, str, int], Any] = {}

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

UPSCALER_MODEL_DIR = Path(
    os.environ.get(
        "UPSCALER_MODEL_DIR",
        DEFAULT_MODEL_DIR,
    )
)

MODEL_REGISTRY: dict[tuple[str, str, int], dict[str, Any]] = {
    ("real-esrgan", "general", 4): {
        "filename": "realesr-general-x4v3.onnx",
        "url": "https://huggingface.co/Heliosoph/realesrgan-onnx/resolve/main/realesr-general-x4v3.onnx",
        "model_scale": 4,
    },
    ("real-esrgan", "photo", 4): {
        "filename": "RealESRGAN_x4plus.onnx",
        "url": "https://huggingface.co/fernandotonon/QtMeshEditor-realesrgan-onnx/resolve/main/RealESRGAN_x4plus.onnx",
        "aliases": ["realesrgan-x4plus.onnx"],
        "model_scale": 4,
    },
    ("real-esrgan", "anime", 4): {
        "filename": "realesrgan_anime6b.onnx",
        "url": "https://huggingface.co/RekluzLabs/realesrgan_anime6b.onnx/resolve/main/realesrgan_anime6b.onnx",
        "aliases": ["realesrgan-x4plus-anime.onnx"],
        "model_scale": 4,
    },
    ("real-esrgan", "general", 2): {
        "filename": "RealESRGAN_x2plus.onnx",
        "url": "https://huggingface.co/fernandotonon/QtMeshEditor-realesrgan-onnx/resolve/main/RealESRGAN_x2plus.onnx",
        "aliases": ["realesrgan-x2plus.onnx"],
        "model_scale": 2,
    },
    ("real-esrgan", "photo", 2): {
        "filename": "RealESRGAN_x2plus.onnx",
        "url": "https://huggingface.co/fernandotonon/QtMeshEditor-realesrgan-onnx/resolve/main/RealESRGAN_x2plus.onnx",
        "aliases": ["realesrgan-x2plus.onnx"],
        "model_scale": 2,
    },
    ("real-esrgan", "anime", 2): {
        "filename": "RealESRGAN_x2plus.onnx",
        "url": "https://huggingface.co/fernandotonon/QtMeshEditor-realesrgan-onnx/resolve/main/RealESRGAN_x2plus.onnx",
        "aliases": ["realesrgan-x2plus.onnx"],
        "model_scale": 2,
    },
    ("swinir", "classical", 4): {
        "filename": "swinir_realworld_x4.onnx",
        "url": "https://huggingface.co/rocca/swin-ir-onnx/resolve/main/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.onnx",
        "aliases": ["swinir_classical_x4.onnx", "003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.onnx"],
        "model_scale": 4,
    },
    ("swinir", "realworld", 4): {
        "filename": "swinir_realworld_x4.onnx",
        "url": "https://huggingface.co/rocca/swin-ir-onnx/resolve/main/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.onnx",
        "aliases": ["swinir_classical_x4.onnx", "003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.onnx"],
        "model_scale": 4,
    },
    ("swinir", "classical", 2): {
        "filename": "swinir_realworld_x4.onnx",
        "url": "https://huggingface.co/rocca/swin-ir-onnx/resolve/main/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.onnx",
        "aliases": ["swinir_classical_x2.onnx", "swinir_classical_x4.onnx"],
        "model_scale": 4,
    },
    ("swinir", "realworld", 2): {
        "filename": "swinir_realworld_x4.onnx",
        "url": "https://huggingface.co/rocca/swin-ir-onnx/resolve/main/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.onnx",
        "aliases": ["swinir_classical_x2.onnx", "swinir_classical_x4.onnx"],
        "model_scale": 4,
    },
}


def _detect_image_type(img: Image.Image) -> tuple[str, str, str]:
    """Analyze image content to auto-select optimal engine and preset.

    Returns:
        tuple of (engine, model_name, detected_description)
    """
    w, h = img.size
    scale = min(256.0 / max(w, h), 1.0)
    thumb_w, thumb_h = max(1, int(w * scale)), max(1, int(h * scale))
    thumb = img.resize((thumb_w, thumb_h), Image.Resampling.BILINEAR)

    arr = np.asarray(to_rgb(thumb))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    # 1. Color palette entropy for synthetic art & diagram filtering
    hist = cv2.calcHist([arr], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = hist.flatten()
    total_hist = hist.sum()
    if total_hist > 0:
        p = hist / total_hist
        p = p[p > 0]
        entropy = -float(np.sum(p * np.log2(p)))
    else:
        entropy = 0.0

    # 2. Straight line & diagram / text detection
    # Real diagrams, schematics, UI wireframes, and text documents have sparse edges (< 0.15),
    # low color entropy (< 3.2), and significant axis-aligned (horizontal/vertical) lines.
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = np.count_nonzero(edges) / (thumb_w * thumb_h) if (thumb_w * thumb_h) > 0 else 0.0
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=25, maxLineGap=5)
    rectilinear_len = 0.0
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.flatten()
            length = float(np.hypot(x2 - x1, y2 - y1))
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            if angle < 10 or angle > 170 or (80 < angle < 100):
                rectilinear_len += length

    dim_sum = thumb_w + thumb_h
    if edge_ratio < 0.15 and dim_sum > 0 and (rectilinear_len / dim_sum) >= 1.5 and entropy < 3.2:
        return ("swinir", "classical", "text/diagram")

    # 3. Color palette entropy for cel-shaded anime / illustration
    if entropy < 2.5:
        return ("real-esrgan", "anime", "anime/art")

    # 3. High-frequency texture & noise analysis for photo
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if lap_var < 100.0 or min(w, h) < 500:
        return ("real-esrgan", "general", "photo (compressed)")
    return ("real-esrgan", "photo", "photo (clean)")


def _get_upscaler_session(engine: str, model_name: str, scale: int) -> tuple[Any, int]:
    """Retrieve or lazily initialize the ONNX InferenceSession and native model scale."""
    key = (engine, model_name, scale)
    if key in _UPSCALER_SESSIONS:
        return _UPSCALER_SESSIONS[key]

    info = MODEL_REGISTRY.get(key)
    if not info:
        default_model = "classical" if engine == "swinir" else "general"
        key = (engine, default_model, scale if scale in (2, 4) else 4)
        info = MODEL_REGISTRY.get(key, MODEL_REGISTRY[("real-esrgan", "general", 4)])

    filename = info["filename"]
    urls = info.get("urls", [info["url"]])
    model_scale = info.get("model_scale", scale)

    UPSCALER_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = UPSCALER_MODEL_DIR / filename

    # Check if model exists under an alias filename
    if not model_path.exists():
        for alias in info.get("aliases", []):
            alias_path = UPSCALER_MODEL_DIR / alias
            if alias_path.exists():
                model_path = alias_path
                break

    # Check legacy ~/.cache directory if not found in models/
    if not model_path.exists():
        legacy_dir = Path.home() / ".cache" / "image-processor" / "models"
        if legacy_dir.exists():
            for name in [filename] + info.get("aliases", []):
                legacy_file = legacy_dir / name
                if legacy_file.exists():
                    model_path = legacy_file
                    break

    if not model_path.exists():
        download_errs: list[str] = []
        for url in urls:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "local-image-processor/1.0"})
                with urllib.request.urlopen(req, timeout=60) as resp, open(model_path, "wb") as f:
                    while chunk := resp.read(65536):
                        f.write(chunk)
                download_errs.clear()
                break
            except Exception as exc:
                if model_path.exists():
                    try:
                        model_path.unlink()
                    except OSError:
                        pass
                download_errs.append(f"{url} ({exc})")

        if download_errs:
            raise ToolError(
                f"Model file '{filename}' not found locally at {model_path} and automated download failed: "
                f"{'; '.join(download_errs)}. "
                f"In an offline environment, please download '{filename}' and place it in '{UPSCALER_MODEL_DIR}'."
            )

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ToolError(
            "onnxruntime is required for image upscaling. Install with: pip install onnxruntime"
        ) from exc

    try:
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
                cuda_sess = ort.InferenceSession(
                    str(model_path),
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                )
                # Dry run a tiny tensor to verify CUDA and cuDNN libraries are functional
                input_name = cuda_sess.get_inputs()[0].name
                dummy = np.zeros((1, 3, 16, 16), dtype=np.float32)
                cuda_sess.run(None, {input_name: dummy})
                session = cuda_sess
            except Exception:
                session = None

        if session is None:
            session = ort.InferenceSession(
                str(model_path), providers=["CPUExecutionProvider"]
            )

        res = (session, model_scale)
        _UPSCALER_SESSIONS[(engine, model_name, scale)] = res
        return res
    except Exception as exc:
        raise ToolError(f"Failed to load ONNX model '{filename}': {exc}") from exc


def _run_upscale_inference(
    img: Image.Image,
    session: Any,
    scale: int,
    tile_size: int = 512,
    tile_pad: int = 10,
    window_size: int = 8,
    model_scale: int | None = None,
) -> Image.Image:
    """Run super-resolution inference on img using session, with optional tiling and window padding."""
    if model_scale is None:
        model_scale = scale

    rgb = to_rgb(img)
    arr = np.asarray(rgb).astype(np.float32) / 255.0
    h, w, c = arr.shape

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    if tile_size <= 0 or (h <= tile_size and w <= tile_size):
        tensor_in = np.transpose(arr, (2, 0, 1))[np.newaxis, ...]
        pad_h = (window_size - (h % window_size)) % window_size
        pad_w = (window_size - (w % window_size)) % window_size
        if pad_h > 0 or pad_w > 0:
            tensor_in = np.pad(tensor_in, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)), mode="reflect")

        out_tensor = session.run([output_name], {input_name: tensor_in})[0]
        if pad_h > 0 or pad_w > 0:
            out_tensor = out_tensor[:, :, : h * model_scale, : w * model_scale]

        out_arr = np.squeeze(out_tensor, axis=0)
        out_arr = np.transpose(out_arr, (1, 2, 0))
        out_arr = np.clip(out_arr * 255.0, 0, 255).round().astype(np.uint8)
        res_img = Image.fromarray(out_arr, mode="RGB")
        if model_scale != scale:
            res_img = res_img.resize((w * scale, h * scale), Image.Resampling.LANCZOS)
        return res_img

    out_h = h * model_scale
    out_w = w * model_scale
    out_img_arr = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    y_steps = range(0, h, tile_size)
    x_steps = range(0, w, tile_size)

    for y in y_steps:
        for x in x_steps:
            th = min(tile_size, h - y)
            tw = min(tile_size, w - x)

            pad_top = min(tile_pad, y)
            pad_bottom = min(tile_pad, h - (y + th))
            pad_left = min(tile_pad, x)
            pad_right = min(tile_pad, w - (x + tw))

            tile_in = arr[y - pad_top : y + th + pad_bottom, x - pad_left : x + tw + pad_right]
            th_in, tw_in, _ = tile_in.shape
            tensor_in = np.transpose(tile_in, (2, 0, 1))[np.newaxis, ...]

            pad_h = (window_size - (th_in % window_size)) % window_size
            pad_w = (window_size - (tw_in % window_size)) % window_size
            if pad_h > 0 or pad_w > 0:
                tensor_in = np.pad(tensor_in, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)), mode="reflect")

            out_tensor = session.run([output_name], {input_name: tensor_in})[0]
            if pad_h > 0 or pad_w > 0:
                out_tensor = out_tensor[:, :, : th_in * model_scale, : tw_in * model_scale]

            tile_out = np.squeeze(out_tensor, axis=0)
            tile_out = np.transpose(tile_out, (1, 2, 0))

            crop_top = pad_top * model_scale
            crop_bottom = tile_out.shape[0] - (pad_bottom * model_scale)
            crop_left = pad_left * model_scale
            crop_right = tile_out.shape[1] - (pad_right * model_scale)

            cropped_tile = tile_out[crop_top:crop_bottom, crop_left:crop_right]
            cropped_uint8 = np.clip(cropped_tile * 255.0, 0, 255).round().astype(np.uint8)

            out_img_arr[y * model_scale : (y + th) * model_scale, x * model_scale : (x + tw) * model_scale] = cropped_uint8

    res_img = Image.fromarray(out_img_arr, mode="RGB")
    if model_scale != scale:
        res_img = res_img.resize((w * scale, h * scale), Image.Resampling.LANCZOS)
    return res_img


@mcp.tool()
def upscale_image(
    input_path: str,
    output_path: str,
    scale: int = 4,
    engine: str = "auto",
    model_name: str = "auto",
    tile_size: int = 512,
    tile_pad: int = 10,
) -> str:
    """Upscale an image by 2x or 4x using AI super-resolution neural networks.

    Supports Real-ESRGAN (CNN/GAN, default) and SwinIR (Vision Transformer).
    When engine="auto", the image is analyzed using classical computer vision
    to automatically select the best model:
    - Text, UI, diagrams, and architecture -> SwinIR
    - Cartoons, anime, and illustrations -> Real-ESRGAN (anime)
    - Natural photographs -> Real-ESRGAN (general / photo)

    Args:
        input_path: Path to the source image.
        output_path: Destination path; extension sets the saved format.
        scale: Upscaling factor (2 or 4). Default 4.
        engine: Engine preset — "auto" (content-aware detection),
            "real-esrgan", or "swinir". Default "auto".
        model_name: Model preset — "auto", "general", "photo", "anime",
            "classical", or "realworld". Default "auto".
        tile_size: Tile dimension in pixels for chunked inference (0 disables
            tiling; default 512 prevents out-of-memory errors). Default 512.
        tile_pad: Overlap padding in pixels between tiles to eliminate
            boundary seams. Default 10.
    """
    try:
        scale = int(scale)
    except (TypeError, ValueError):
        raise ToolError(f"scale must be an integer, got {scale!r}")

    if scale not in (2, 4):
        raise ToolError(f"scale must be 2 or 4, got {scale}")

    engine = engine.strip().lower()
    valid_engines = ("auto", "real-esrgan", "swinir")
    if engine not in valid_engines:
        raise ToolError(f"engine must be one of: {', '.join(valid_engines)}")

    model_name = model_name.strip().lower()
    valid_models = ("auto", "general", "photo", "anime", "classical", "realworld")
    if model_name not in valid_models:
        raise ToolError(f"model_name must be one of: {', '.join(valid_models)}")

    try:
        tile_size = int(tile_size)
        tile_pad = int(tile_pad)
    except (TypeError, ValueError):
        raise ToolError("tile_size and tile_pad must be integers")

    if tile_size < 0:
        raise ToolError(f"tile_size must be >= 0, got {tile_size}")
    if tile_pad < 0:
        raise ToolError(f"tile_pad must be >= 0, got {tile_pad}")

    _, img = load_image(input_path)
    _check_size(img)

    orig_w, orig_h = img.size
    proj_w = orig_w * scale
    proj_h = orig_h * scale
    proj_pixels = proj_w * proj_h

    if proj_pixels > MAX_PIXELS:
        raise ToolError(
            f"Upscaled image is too large ({proj_w}x{proj_h} = {proj_pixels:,} pixels); "
            f"maximum allowed is {MAX_PIXELS:,} (to prevent DoS)."
        )

    detected_desc: str | None = None
    if engine == "auto" or model_name == "auto":
        det_engine, det_model, detected_desc = _detect_image_type(img)
        active_engine = det_engine if engine == "auto" else engine
        active_model = det_model if model_name == "auto" else model_name
    else:
        active_engine = engine
        active_model = model_name

    base, alpha = _split_alpha(img)

    session_res = _get_upscaler_session(active_engine, active_model, scale)
    if isinstance(session_res, tuple):
        session, model_scale = session_res
    else:
        session, model_scale = session_res, scale

    upscaled_base = _run_upscale_inference(
        base, session, scale, tile_size, tile_pad, model_scale=model_scale
    )

    if alpha is not None:
        upscaled_alpha = alpha.resize((proj_w, proj_h), Image.Resampling.LANCZOS)
    else:
        upscaled_alpha = None

    result = _rejoin_alpha(upscaled_base, upscaled_alpha)
    save_image(result, output_path)

    if detected_desc:
        return (
            f"Upscaled {scale}x [auto-detected: {detected_desc} -> {active_engine} ({active_model})] "
            f"[{orig_w}x{orig_h} -> {proj_w}x{proj_h}] -> {output_path}"
        )
    return (
        f"Upscaled {scale}x via {active_engine} ({active_model}) "
        f"[{orig_w}x{orig_h} -> {proj_w}x{proj_h}] -> {output_path}"
    )
