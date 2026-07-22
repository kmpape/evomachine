from __future__ import annotations

from typing import Any

import numpy as np


def array_to_preview_payload(array: np.ndarray, max_shape: tuple[int, int]) -> dict[str, Any]:
    """Return a JSON-safe downsampled preview for a 2D or RGB image."""
    preview = downsample_array(np.asarray(array), max_shape=max_shape)
    return {
        "shape": list(preview.shape),
        "dtype": str(preview.dtype),
        "data": preview.tolist(),
    }


def array_from_preview_payload(payload: dict[str, Any] | None) -> np.ndarray | None:
    """Rebuild a numpy image preview from a JSON payload."""
    if not payload:
        return None
    dtype = np.dtype(payload.get("dtype", "uint8"))
    array = np.asarray(payload["data"], dtype=dtype)
    expected_shape = tuple(payload.get("shape", ()))
    if expected_shape and array.shape != expected_shape:
        raise ValueError(f"Preview payload shape {array.shape} does not match {expected_shape}.")
    return array


def stack_to_preview_payload(array: np.ndarray, max_shape: tuple[int, int]) -> dict[str, Any]:
    """Return a JSON-safe downsampled preview for a stack of 2D or RGB images."""
    stack = np.asarray(array)
    if stack.ndim == 2:
        preview = downsample_array(stack, max_shape=max_shape)[np.newaxis, ...]
    elif stack.ndim == 3:
        preview = np.stack([downsample_array(plane, max_shape=max_shape) for plane in stack], axis=0)
    elif stack.ndim == 4 and stack.shape[-1] in {3, 4}:
        preview = np.stack([downsample_array(plane, max_shape=max_shape) for plane in stack], axis=0)
    else:
        raise ValueError(f"Expected an image stack, received shape {stack.shape}.")
    return {
        "is_stack": True,
        "shape": list(preview.shape),
        "dtype": str(preview.dtype),
        "data": preview.tolist(),
    }


def stack_from_preview_payload(payload: dict[str, Any] | None) -> np.ndarray | None:
    """Rebuild a numpy image stack preview from a JSON payload."""
    stack = array_from_preview_payload(payload)
    if stack is None:
        return None
    if not bool(payload.get("is_stack")):
        raise ValueError("Stack preview payload is missing is_stack=True.")
    if stack.ndim == 2:
        return stack[np.newaxis, ...]
    if stack.ndim not in {3, 4}:
        raise ValueError(f"Stack preview payload must be 3D or 4D, received shape {stack.shape}.")
    return stack


def downsample_array(array: np.ndarray, max_shape: tuple[int, int]) -> np.ndarray:
    """Downsample an array by index selection without changing dtype."""
    if array.ndim not in {2, 3}:
        raise ValueError(f"Expected a 2D or RGB image array, received shape {array.shape}.")
    max_height, max_width = max_shape
    height, width = array.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("Image preview source must have positive height and width.")
    scale = min(max_height / height, max_width / width, 1.0)
    target_height = max(1, int(round(height * scale)))
    target_width = max(1, int(round(width * scale)))
    if target_height == height and target_width == width:
        return array.copy()
    row_indices = np.linspace(0, height - 1, target_height).astype(int)
    col_indices = np.linspace(0, width - 1, target_width).astype(int)
    return array[row_indices][:, col_indices].copy()
