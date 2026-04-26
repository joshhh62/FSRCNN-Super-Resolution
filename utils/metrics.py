from __future__ import annotations

import math

import cv2
import numpy as np
from skimage.metrics import structural_similarity


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def compute_psnr(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference_float = reference.astype(np.float64)
    candidate_float = candidate.astype(np.float64)
    mse = np.mean((reference_float - candidate_float) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * math.log10(255.0 / math.sqrt(mse))


def compute_ssim(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference_gray = _to_grayscale(reference)
    candidate_gray = _to_grayscale(candidate)
    return float(
        structural_similarity(reference_gray, candidate_gray, data_range=255)
    )


def _laplacian_sharpness(image: np.ndarray) -> float:
    gray = _to_grayscale(image)
    return float(cv2.Laplacian(gray.astype(np.float64), cv2.CV_64F).var())


def compute_metrics(original_upscaled: np.ndarray, fsrcnn_image: np.ndarray) -> dict[str, float]:
    sharpness_input = _laplacian_sharpness(original_upscaled)
    sharpness_fsrcnn = _laplacian_sharpness(fsrcnn_image)
    return {
        "psnr": compute_psnr(original_upscaled, fsrcnn_image),
        "ssim": compute_ssim(original_upscaled, fsrcnn_image),
        "sharpness_input": sharpness_input,
        "sharpness_fsrcnn": sharpness_fsrcnn,
        "sharpness_ratio": sharpness_fsrcnn / max(sharpness_input, 1e-6),
    }
