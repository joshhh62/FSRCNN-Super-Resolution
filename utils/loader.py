from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st
import torch

from model.fsrcnn import FSRCNN
from utils.logging_utils import get_logger


DEFAULT_MODEL_NAME = "fsrcnn1.pth"
logger = get_logger()


class ModelLoadError(RuntimeError):
    """Raised when the FSRCNN checkpoint cannot be loaded."""


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def get_default_model_path() -> Path:
    return get_project_root() / "model" / DEFAULT_MODEL_NAME


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _strip_module_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    if not any(key.startswith("module.") for key in state_dict):
        return state_dict
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def _extract_state_dict(checkpoint: Any) -> dict[str, Any]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model", "weights"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, dict):
                return _strip_module_prefix(candidate)
        if checkpoint and all(isinstance(key, str) for key in checkpoint):
            return _strip_module_prefix(checkpoint)
    raise ModelLoadError("Checkpoint format is unsupported for FSRCNN loading.")


@st.cache_resource(show_spinner=False)
def load_fsrcnn_model(model_path: str | None = None) -> tuple[FSRCNN, torch.device]:
    resolved_model_path = Path(model_path) if model_path else get_default_model_path()
    logger.info("Loading FSRCNN checkpoint from %s", resolved_model_path)
    if not resolved_model_path.exists():
        logger.error("Checkpoint file not found: %s", resolved_model_path)
        raise ModelLoadError(
            f"Model checkpoint not found at '{resolved_model_path}'. "
            "Place fsrcnn1.pth inside the model directory."
        )

    device = get_device()
    logger.info("Selected device for FSRCNN inference: %s", device.type.upper())
    model = FSRCNN(scale_factor=2, d=56, s=12, m=4)

    try:
        checkpoint = torch.load(resolved_model_path, map_location=device)
        state_dict = _extract_state_dict(checkpoint)
        model.load_state_dict(state_dict, strict=True)
        model.to(device)
        model.eval()
        logger.info("FSRCNN checkpoint loaded successfully on %s", device.type.upper())
        return model, device
    except FileNotFoundError as exc:
        logger.exception("Model file disappeared while loading: %s", resolved_model_path)
        raise ModelLoadError(f"Model file is missing: {resolved_model_path}") from exc
    except RuntimeError as exc:
        logger.exception("FSRCNN checkpoint is incompatible with the application architecture")
        raise ModelLoadError(
            "Model checkpoint is incompatible with the expected FSRCNN architecture."
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error while loading FSRCNN checkpoint")
        raise ModelLoadError(f"Unable to load FSRCNN checkpoint: {exc}") from exc
