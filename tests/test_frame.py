"""Tests for frame data containers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from evomachine.frame import Frame, FrameMetaData
from evomachine.types import LEDType, UNKNOWN_FOV_ID


def _metadata(frame_id: int = 0, fov_id: int = 7) -> FrameMetaData:
    """
    Return simple frame metadata for Frame tests.

    Parameters
    ----------
    frame_id
        Frame ID to assign.
    fov_id
        FoV ID to assign.

    Returns
    -------
    FrameMetaData
        Valid metadata object.
    """
    return FrameMetaData(
        frame_id=frame_id,
        leds={LEDType.LED_450_NM: 20},
        filter_wheel=None,
        exposure=50,
        fov_id=fov_id,
    )


def _array(n_frames: int) -> np.ndarray:
    """
    Return a deterministic image stack.

    Parameters
    ----------
    n_frames
        Leading-axis frame count.

    Returns
    -------
    np.ndarray
        Image stack.
    """
    return np.zeros((n_frames, 3, 4), dtype=np.uint16)


def test_frame_fov_id_derives_from_metadata() -> None:
    """Check Frame.fov_id defaults to the metadata FoV ID."""
    frame = Frame(frame_metadata=[_metadata(fov_id=3)], array=_array(1))

    assert frame.fov_id == 3


def test_frame_unknown_metadata_keeps_unknown_fov_id() -> None:
    """Check unknown metadata FoV IDs produce an unknown Frame FoV ID."""
    frame = Frame(frame_metadata=[_metadata(fov_id=UNKNOWN_FOV_ID)], array=_array(1))

    assert frame.fov_id == UNKNOWN_FOV_ID


def test_frame_accepts_explicit_matching_fov_id() -> None:
    """Check explicit Frame.fov_id is accepted when it matches metadata."""
    frame = Frame(frame_metadata=[_metadata(fov_id=5)], array=_array(1), fov_id=5)

    assert frame.fov_id == 5


def test_frame_rejects_mixed_metadata_fov_ids() -> None:
    """Check one Frame cannot contain multiple FoV IDs."""
    with pytest.raises(ValueError, match="same fov_id"):
        Frame(
            frame_metadata=[_metadata(frame_id=0, fov_id=3), _metadata(frame_id=1, fov_id=4)],
            array=_array(2),
        )


def test_frame_rejects_explicit_mismatched_fov_id() -> None:
    """Check Frame.fov_id must agree with metadata."""
    with pytest.raises(ValueError, match="must match"):
        Frame(frame_metadata=[_metadata(fov_id=5)], array=_array(1), fov_id=6)


def test_frame_omitted_or_empty_saved_paths_default_to_none_per_metadata() -> None:
    """Check omitted and empty saved paths are expanded per image plane."""
    metadata = [_metadata(frame_id=0), _metadata(frame_id=1)]

    omitted_paths_frame = Frame(frame_metadata=metadata, array=_array(2))
    empty_paths_frame = Frame(frame_metadata=metadata, array=_array(2), saved_paths=[])

    assert omitted_paths_frame.saved_paths == [None, None]
    assert empty_paths_frame.saved_paths == [None, None]


def test_frame_saved_paths_validate_length_and_type() -> None:
    """Check saved_paths stays aligned with metadata and image planes."""
    metadata = [_metadata()]

    with pytest.raises(ValueError, match="saved_paths length"):
        Frame(frame_metadata=metadata, array=_array(1), saved_paths=[None, Path("frame.tif")])
    with pytest.raises(TypeError, match="Path or None"):
        Frame(frame_metadata=metadata, array=_array(1), saved_paths=["frame.tif"])
