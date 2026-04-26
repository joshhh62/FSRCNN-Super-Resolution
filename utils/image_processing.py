from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from PIL import Image, ImageFile, ImageOps, UnidentifiedImageError
import streamlit as st
import torch

from utils.logging_utils import get_logger


ImageFile.LOAD_TRUNCATED_IMAGES = False

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_UPLOAD_SIZE_MB = 20
MAX_OUTPUT_PIXELS = 12_000_000
logger = get_logger()


class ImageProcessingError(RuntimeError):
    """Raised when image validation or processing fails."""


@dataclass(frozen=True)
class ProcessedUpload:
    name: str
    original_name: str
    image: np.ndarray
    notice: str | None = None


def validate_file(filename: str, file_size: int) -> None:
    suffix = Path(filename).suffix.lower()
    logger.info("Validating upload '%s' (%d bytes)", filename, file_size)
    if suffix not in ALLOWED_EXTENSIONS:
        logger.error("Rejected upload '%s' due to unsupported extension '%s'", filename, suffix)
        raise ImageProcessingError("Unsupported file type. Please upload a JPG or PNG image.")
    max_bytes = MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if file_size > max_bytes:
        logger.error(
            "Rejected upload '%s' because it exceeds the %d MB limit",
            filename,
            MAX_UPLOAD_SIZE_MB,
        )
        raise ImageProcessingError(
            f"File is too large. Upload an image smaller than {MAX_UPLOAD_SIZE_MB} MB."
        )


@st.cache_data(show_spinner=False)
def decode_uploaded_image(file_bytes: bytes, filename: str, scale_factor: int) -> ProcessedUpload:
    validate_file(filename, len(file_bytes))
    logger.info("Decoding upload '%s' for %dx enhancement", filename, scale_factor)

    try:
        with Image.open(io.BytesIO(file_bytes)) as pil_image:
            pil_image.load()
            pil_image = ImageOps.exif_transpose(pil_image)
            pil_image = pil_image.convert("RGB")
            image = np.array(pil_image, dtype=np.uint8)
            logger.info(
                "Decoded upload '%s' successfully with shape %s",
                filename,
                image.shape,
            )
    except UnidentifiedImageError as exc:
        logger.exception("Uploaded file '%s' is not a valid image", filename)
        raise ImageProcessingError("The uploaded file is not a valid image.") from exc
    except OSError as exc:
        logger.exception("Uploaded image '%s' could not be read", filename)
        raise ImageProcessingError("The image appears to be corrupted or unreadable.") from exc

    if image.ndim != 3 or image.shape[2] != 3:
        logger.error("Upload '%s' is not RGB-compatible. Shape received: %s", filename, image.shape)
        raise ImageProcessingError("Only RGB-compatible images are supported.")

    return enforce_memory_guard(image=image, filename=filename, scale_factor=scale_factor)


def enforce_memory_guard(image: np.ndarray, filename: str, scale_factor: int) -> ProcessedUpload:
    height, width = image.shape[:2]
    expected_pixels = height * width * (scale_factor**2)
    if expected_pixels <= MAX_OUTPUT_PIXELS:
        logger.info(
            "Memory guard accepted '%s' without resizing. Input=%dx%d, projected_output_pixels=%d",
            filename,
            width,
            height,
            expected_pixels,
        )
        return ProcessedUpload(name=filename, original_name=filename, image=image, notice=None)

    ratio = (MAX_OUTPUT_PIXELS / expected_pixels) ** 0.5
    new_width = max(1, int(width * ratio))
    new_height = max(1, int(height * ratio))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    logger.warning(
        "Memory guard resized '%s' from %dx%d to %dx%d before %dx enhancement",
        filename,
        width,
        height,
        new_width,
        new_height,
        scale_factor,
    )
    notice = (
        "Input was automatically downscaled to keep inference memory-safe. "
        f"Processed at {new_width}x{new_height}px."
    )
    return ProcessedUpload(name=filename, original_name=filename, image=resized, notice=notice)


@st.cache_data(show_spinner=False)
def nearest_upscale(image: np.ndarray, scale_factor: int) -> np.ndarray:
    height, width = image.shape[:2]
    return cv2.resize(
        image,
        (width * scale_factor, height * scale_factor),
        interpolation=cv2.INTER_NEAREST,
    )


def preprocess_for_model(image: np.ndarray) -> torch.Tensor:
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    tensor = torch.from_numpy(image).unsqueeze(0)
    return tensor


def postprocess_tensor(output_tensor: torch.Tensor) -> np.ndarray:
    output = output_tensor.squeeze(0).detach().cpu().clamp(0.0, 1.0).numpy()
    output = np.transpose(output, (1, 2, 0))
    return (output * 255.0).round().astype(np.uint8)


def _unsharp_mask(image: np.ndarray, radius: float = 0.9, amount: float = 0.55) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (0, 0), radius)
    result = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
    return np.clip(result, 0, 255).astype(np.uint8)


def _clahe_enhance(image: np.ndarray, clip_limit: float = 2.5) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_ch = clahe.apply(l_ch)
    return cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2RGB)


def _saturation_boost(image: np.ndarray, factor: float = 1.12) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def post_enhance(image: np.ndarray) -> np.ndarray:
    """Four-stage pipeline to recover visual quality lost in PSNR-biased FSRCNN training.

    1. Bilateral filter  — removes ringing/banding artifacts while keeping edges intact.
    2. CLAHE             — restores local contrast that MSE loss tends to flatten.
    3. Unsharp mask      — sharpens edges and fine details.
    4. Saturation boost  — compensates for colour desaturation caused by averaging in MSE.
    """
    image = cv2.bilateralFilter(image, d=5, sigmaColor=20, sigmaSpace=20)
    image = _clahe_enhance(image)
    image = _unsharp_mask(image)
    image = _saturation_boost(image)
    return image


def _pre_sharpen_input(image: np.ndarray) -> np.ndarray:
    """Light sharpening of the low-res input so FSRCNN sees crisper edges to upscale."""
    return _unsharp_mask(image, radius=0.4, amount=0.2)


def run_fsrcnn_inference(
    image: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    scale_factor: Literal[2, 4],
    post_process: bool = True,
) -> np.ndarray:
    if scale_factor not in (2, 4):
        logger.error("Unsupported FSRCNN scale factor requested: %s", scale_factor)
        raise ImageProcessingError("FSRCNN only supports scale factors of 2 or 4 in this app.")

    logger.info(
        "Starting FSRCNN inference. Input shape=%s, scale=%dx, device=%s, post_process=%s",
        image.shape,
        scale_factor,
        device.type.upper(),
        post_process,
    )

    # Pre-sharpen low-res input so the model has crisper edges to reconstruct from.
    prepared = _pre_sharpen_input(image)
    tensor = preprocess_for_model(prepared).to(device)

    try:
        with torch.inference_mode():
            output = model(tensor)
            if scale_factor == 4:
                logger.info("Applying chained second FSRCNN pass for 4x enhancement")
                # Condition the intermediate 2x result before the second pass:
                # CLAHE restores contrast, unsharp mask adds back sharpness lost in the
                # first PSNR-biased pass, so the second pass starts from a crisp image.
                intermediate = postprocess_tensor(output.clamp(0.0, 1.0))
                intermediate = _clahe_enhance(intermediate, clip_limit=1.5)
                intermediate = _unsharp_mask(intermediate, radius=0.5, amount=0.4)
                second_tensor = preprocess_for_model(intermediate).to(device)
                output = model(second_tensor)
        result = postprocess_tensor(output)
        if post_process:
            result = post_enhance(result)
            logger.info("Post-processing applied (bilateral + CLAHE + unsharp + saturation)")
        logger.info("FSRCNN inference finished successfully with output shape=%s", result.shape)
        return result
    except RuntimeError as exc:
        logger.exception("FSRCNN inference failed with a runtime error")
        raise ImageProcessingError(f"FSRCNN inference failed: {exc}") from exc


@st.cache_data(show_spinner=False)
def encode_png(image: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return buffer.getvalue()


def build_download_name(original_name: str) -> str:
    stem = Path(original_name).stem or "image"
    return f"enhanced_{stem}.png"
