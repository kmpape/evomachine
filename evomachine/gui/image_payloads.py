from __future__ import annotations

import base64
import os
import tempfile
import time
from io import BytesIO
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import tifffile


IMAGE_TRANSPORT_ENV = "EVOMACHINE_GUI_IMAGE_TRANSPORT"
IMAGE_TRANSPORT_DIR_ENV = "EVOMACHINE_GUI_IMAGE_TRANSPORT_DIR"

IMAGE_TRANSPORT_AUTO = "auto"
IMAGE_TRANSPORT_TEMP_TIFF = "temp_tiff"
IMAGE_TRANSPORT_SOCKET_TIFF = "socket_tiff"
IMAGE_TRANSPORT_RAW = "raw"
IMAGE_TRANSPORT_CHOICES = (
    IMAGE_TRANSPORT_AUTO,
    IMAGE_TRANSPORT_TEMP_TIFF,
    IMAGE_TRANSPORT_SOCKET_TIFF,
    IMAGE_TRANSPORT_RAW,
)

TIFF_COMPRESSION = "zlib"
TEMP_IMAGE_PREFIX = "evomachine_gui_image_"
TEMP_IMAGE_SUFFIX = ".tiff"
TEMP_PROBE_PREFIX = "evomachine_gui_probe_"
TEMP_PROBE_SUFFIX = ".txt"
TEMP_FILE_MAX_AGE_SECONDS = 30 * 60


def array_to_preview_payload(array: np.ndarray, *, transport: str | None = None) -> dict[str, Any]:
    """Return a JSON-safe full-resolution image payload."""
    preview = np.asarray(array)
    validate_image_array(preview)
    return array_to_binary_payload(preview, transport=transport)


def array_from_preview_payload(payload: dict[str, Any] | None) -> np.ndarray | None:
    """Rebuild a numpy image preview from a JSON payload."""
    if not payload:
        return None
    dtype = np.dtype(payload.get("dtype", "uint8"))
    expected_shape = tuple(payload.get("shape", ()))
    encoding = payload.get("encoding")
    if encoding is None:
        array = np.asarray(payload["data"], dtype=dtype)
    elif encoding == "raw":
        array = array_from_raw_payload(payload=payload, dtype=dtype, expected_shape=expected_shape)
    elif encoding == "packed_binary":
        array = array_from_packed_binary_payload(payload=payload, dtype=dtype, expected_shape=expected_shape)
    elif encoding == "tiff":
        array = array_from_tiff_payload(payload=payload)
    elif encoding == "tiff_path":
        array = array_from_tiff_path_payload(payload=payload)
    else:
        raise ValueError(f"Unsupported image payload encoding {encoding!r}.")
    if expected_shape and array.shape != expected_shape:
        raise ValueError(f"Preview payload shape {array.shape} does not match {expected_shape}.")
    return array


def stack_to_preview_payload(array: np.ndarray, *, transport: str | None = None) -> dict[str, Any]:
    """Return a JSON-safe full-resolution stack payload."""
    stack = np.asarray(array)
    if stack.ndim == 2:
        preview = stack[np.newaxis, ...]
    elif stack.ndim in {3, 4}:
        if stack.ndim == 4 and stack.shape[-1] not in {3, 4}:
            raise ValueError(f"Expected an RGB image stack, received shape {stack.shape}.")
        preview = stack
    else:
        raise ValueError(f"Expected an image stack, received shape {stack.shape}.")
    return {
        **array_to_binary_payload(preview, transport=transport, is_stack=True),
        "is_stack": True,
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


def validate_image_array(array: np.ndarray) -> None:
    """Validate a single image array for GUI transport."""
    if array.ndim not in {2, 3}:
        raise ValueError(f"Expected a 2D or RGB image array, received shape {array.shape}.")
    if array.ndim == 3 and array.shape[-1] not in {3, 4}:
        raise ValueError(f"Expected a 2D or RGB image array, received shape {array.shape}.")
    if array.shape[0] <= 0 or array.shape[1] <= 0:
        raise ValueError("Image preview source must have positive height and width.")


def array_to_binary_payload(array: np.ndarray, *, transport: str | None = None, is_stack: bool = False) -> dict[str, Any]:
    """Return a JSON-safe exact binary payload for a numpy array."""
    contiguous = np.ascontiguousarray(array)
    if is_binary_uint8_array(contiguous):
        packed = np.packbits((contiguous.reshape(-1) > 0).astype(np.uint8), bitorder="big")
        return {
            "shape": list(contiguous.shape),
            "dtype": str(contiguous.dtype),
            "encoding": "packed_binary",
            "bitorder": "big",
            "data": base64.b64encode(packed.tobytes()).decode("ascii"),
        }
    transport_mode = resolved_image_transport(transport)
    if transport_mode == IMAGE_TRANSPORT_TEMP_TIFF:
        return array_to_tiff_path_payload(contiguous, is_stack=is_stack)
    if transport_mode == IMAGE_TRANSPORT_SOCKET_TIFF:
        return array_to_tiff_payload(contiguous, is_stack=is_stack)
    return {
        "shape": list(contiguous.shape),
        "dtype": str(contiguous.dtype),
        "encoding": "raw",
        "data": base64.b64encode(contiguous.tobytes()).decode("ascii"),
    }


def normalise_image_transport(value: str | None) -> str:
    """Return a canonical GUI image transport mode."""
    if value is None:
        value = os.environ.get(IMAGE_TRANSPORT_ENV, IMAGE_TRANSPORT_AUTO)
    mode = str(value).strip().lower().replace("-", "_")
    aliases = {
        "file": IMAGE_TRANSPORT_TEMP_TIFF,
        "path": IMAGE_TRANSPORT_TEMP_TIFF,
        "temp": IMAGE_TRANSPORT_TEMP_TIFF,
        "tiff_path": IMAGE_TRANSPORT_TEMP_TIFF,
        "socket": IMAGE_TRANSPORT_SOCKET_TIFF,
        "tiff": IMAGE_TRANSPORT_SOCKET_TIFF,
    }
    mode = aliases.get(mode, mode)
    if mode not in IMAGE_TRANSPORT_CHOICES:
        raise ValueError(
            f"Unknown GUI image transport {value!r}. "
            f"Expected one of {', '.join(IMAGE_TRANSPORT_CHOICES)}."
        )
    return mode


def resolved_image_transport(transport: str | None = None) -> str:
    """Return the concrete transport to use for one encoded payload."""
    mode = normalise_image_transport(transport)
    if mode != IMAGE_TRANSPORT_AUTO:
        return mode
    try:
        probe = create_image_transport_probe_payload()
        probe_path = Path(probe["path"])
        if probe_path.read_text(encoding="ascii") == probe["token"]:
            probe_path.unlink(missing_ok=True)
            return IMAGE_TRANSPORT_TEMP_TIFF
    except Exception:
        return IMAGE_TRANSPORT_SOCKET_TIFF
    return IMAGE_TRANSPORT_SOCKET_TIFF


def create_image_transport_probe_payload() -> dict[str, str]:
    """Create a small file the GUI can read to prove path transport works."""
    transport_dir = image_transport_directory()
    transport_dir.mkdir(parents=True, exist_ok=True)
    token = uuid4().hex
    probe_path = transport_dir / f"{TEMP_PROBE_PREFIX}{token}{TEMP_PROBE_SUFFIX}"
    probe_path.write_text(token, encoding="ascii")
    return {
        "mode": IMAGE_TRANSPORT_TEMP_TIFF,
        "path": str(probe_path),
        "token": token,
    }


def image_transport_directory() -> Path:
    """Return the shared temporary directory used for path-based image previews."""
    configured = os.environ.get(IMAGE_TRANSPORT_DIR_ENV)
    if configured:
        return Path(configured)
    return Path(tempfile.gettempdir()) / "evomachine_gui_image_transport"


def array_to_tiff_payload(array: np.ndarray, *, is_stack: bool = False) -> dict[str, Any]:
    """Return an exact compressed TIFF image payload embedded in JSON."""
    tiff_bytes, compression = array_to_tiff_bytes(array, is_stack=is_stack)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "encoding": "tiff",
        "compression": compression,
        "data": base64.b64encode(tiff_bytes).decode("ascii"),
    }


def array_to_tiff_path_payload(array: np.ndarray, *, is_stack: bool = False) -> dict[str, Any]:
    """Return an exact compressed TIFF image payload stored as a temporary file."""
    transport_dir = image_transport_directory()
    transport_dir.mkdir(parents=True, exist_ok=True)
    prune_stale_transport_files(transport_dir)
    path = transport_dir / f"{TEMP_IMAGE_PREFIX}{uuid4().hex}{TEMP_IMAGE_SUFFIX}"
    _compression = write_tiff(path=path, array=array, is_stack=is_stack)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "encoding": "tiff_path",
        "path": str(path),
        "delete_after_read": True,
    }


def array_to_tiff_bytes(array: np.ndarray, *, is_stack: bool = False) -> tuple[bytes, str | None]:
    """Encode an array as TIFF bytes, preferring lossless compression."""
    buffer = BytesIO()
    compression = write_tiff(file=buffer, array=array, is_stack=is_stack)
    return buffer.getvalue(), compression


def write_tiff(
        *,
        array: np.ndarray,
        path: Path | None = None,
        file: BytesIO | None = None,
        is_stack: bool = False,
) -> str | None:
    """Write a TIFF to a path or file-like object and return the compression used."""
    target = file if file is not None else path
    if target is None:
        raise ValueError("write_tiff requires either path or file.")
    kwargs = tiff_write_kwargs(array=array, is_stack=is_stack)
    try:
        tifffile.imwrite(target, array, compression=TIFF_COMPRESSION, **kwargs)
    except Exception:
        if file is not None:
            file.seek(0)
            file.truncate(0)
        tifffile.imwrite(target, array, **kwargs)
        return None
    return TIFF_COMPRESSION


def tiff_write_kwargs(*, array: np.ndarray, is_stack: bool) -> dict[str, Any]:
    """Return tifffile metadata that preserves image vs stack intent."""
    if is_stack and array.ndim == 3:
        return {"photometric": "minisblack"}
    if array.ndim == 3 and array.shape[-1] in {3, 4}:
        return {"photometric": "rgb"}
    return {}


def array_from_tiff_payload(payload: dict[str, Any]) -> np.ndarray:
    """Decode an exact TIFF image payload embedded in JSON."""
    raw = base64.b64decode(payload["data"].encode("ascii"))
    return np.asarray(tifffile.imread(BytesIO(raw)))


def array_from_tiff_path_payload(payload: dict[str, Any]) -> np.ndarray:
    """Decode an exact TIFF image payload from a temporary file path."""
    path = Path(payload["path"])
    array = np.asarray(tifffile.imread(path))
    if payload.get("delete_after_read", False):
        path.unlink(missing_ok=True)
    return array


def prune_stale_transport_files(directory: Path, *, max_age_seconds: int = TEMP_FILE_MAX_AGE_SECONDS) -> None:
    """Delete stale GUI transport temp files left by interrupted sessions."""
    deadline = time.time() - max_age_seconds
    for pattern in (f"{TEMP_IMAGE_PREFIX}*{TEMP_IMAGE_SUFFIX}", f"{TEMP_PROBE_PREFIX}*{TEMP_PROBE_SUFFIX}"):
        for path in directory.glob(pattern):
            try:
                if path.stat().st_mtime < deadline:
                    path.unlink(missing_ok=True)
            except OSError:
                continue


def array_from_raw_payload(
        *,
        payload: dict[str, Any],
        dtype: np.dtype,
        expected_shape: tuple[int, ...],
) -> np.ndarray:
    """Decode an exact raw-byte image payload."""
    if not expected_shape:
        raise ValueError("Raw image payload is missing shape.")
    raw = base64.b64decode(payload["data"].encode("ascii"))
    expected_size = int(np.prod(expected_shape))
    array = np.frombuffer(raw, dtype=dtype)
    if array.size != expected_size:
        raise ValueError(f"Raw image payload contains {array.size} values, expected {expected_size}.")
    return array.reshape(expected_shape).copy()


def array_from_packed_binary_payload(
        *,
        payload: dict[str, Any],
        dtype: np.dtype,
        expected_shape: tuple[int, ...],
) -> np.ndarray:
    """Decode a lossless bit-packed uint8 binary image payload."""
    if dtype != np.dtype("uint8"):
        raise ValueError(f"Packed binary image payload must be uint8, received {dtype}.")
    if not expected_shape:
        raise ValueError("Packed binary image payload is missing shape.")
    raw = base64.b64decode(payload["data"].encode("ascii"))
    packed = np.frombuffer(raw, dtype=np.uint8)
    expected_size = int(np.prod(expected_shape))
    unpacked = np.unpackbits(packed, bitorder=payload.get("bitorder", "big"))[:expected_size]
    if unpacked.size != expected_size:
        raise ValueError(f"Packed binary image payload contains {unpacked.size} values, expected {expected_size}.")
    return (unpacked.reshape(expected_shape) * 255).astype(np.uint8, copy=False)


def is_binary_uint8_array(array: np.ndarray) -> bool:
    """Return True when an array can be represented exactly as a DMD-style bit mask."""
    return array.dtype == np.dtype("uint8") and not np.any((array != 0) & (array != 255))
