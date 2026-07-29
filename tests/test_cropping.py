import numpy as np
import pytest

from evomachine.utils import EvoCroppingBox


def test_shape_size_and_full() -> None:
    box = EvoCroppingBox(xtl=2, ytl=3, xbr=7, ybr=9)

    assert box.shape == (6, 5)
    assert box.size == 30
    assert EvoCroppingBox.full(np.zeros((4, 6))) == EvoCroppingBox(0, 0, 6, 4)


def test_crop_inside_image() -> None:
    image = np.arange(20, dtype=np.uint16).reshape(4, 5)

    cropped = EvoCroppingBox(xtl=1, ytl=1, xbr=4, ybr=3).crop(image)

    assert np.array_equal(cropped, image[1:3, 1:4])


def test_crop_pads_outside_image() -> None:
    image = np.arange(12, dtype=np.uint16).reshape(3, 4)

    cropped = EvoCroppingBox(xtl=-2, ytl=-1, xbr=3, ybr=4).crop(image)

    expected = np.zeros((5, 5), dtype=np.uint16)
    expected[1:4, 2:5] = image[:, :3]
    assert np.array_equal(cropped, expected)


def test_patch_clips_to_image_boundary() -> None:
    image = np.zeros((3, 4), dtype=np.uint8)
    patch = np.arange(20, dtype=np.uint8).reshape(4, 5)

    result = EvoCroppingBox(xtl=-2, ytl=-1, xbr=3, ybr=3).patch(image, patch)

    assert result is image
    assert np.array_equal(image[:, :3], patch[1:4, 2:5])


def test_resize_returns_scaled_copy_and_preserves_none_axis() -> None:
    box = EvoCroppingBox(xtl=2, ytl=3, xbr=8, ybr=9)

    resized = box.resize(fx=0.5, fy=2)
    horizontal_only = box.resize(fx=2)

    assert resized == EvoCroppingBox(xtl=1, ytl=6, xbr=4, ybr=18)
    assert horizontal_only == EvoCroppingBox(xtl=4, ytl=3, xbr=16, ybr=9)
    assert box == EvoCroppingBox(xtl=2, ytl=3, xbr=8, ybr=9)


def test_resize_rejects_invalid_factors() -> None:
    box = EvoCroppingBox(0, 0, 2, 2)

    with pytest.raises(TypeError, match="fx must be numeric"):
        box.resize(fx="two")
    with pytest.raises(ValueError, match="greater than zero"):
        box.resize(fy=0)
