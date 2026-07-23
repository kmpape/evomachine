from __future__ import annotations

import numpy as np

from evomachine.config import DMD_WIDTH_HEIGHT
from evomachine.gui.central_workspace import (
    DMD_DISPLAY_SHAPE,
    DMD_RECT,
    MAIN_RECT,
    SPECTRUM_RECT,
    WORKSPACE_SHAPE,
    dmd_array_to_display,
    make_brightness_histogram,
    make_dmd_placeholder,
    make_visual_workspace_stack,
    make_visual_workspace,
)
from evomachine.gui.image_payloads import (
    array_from_preview_payload,
    array_to_preview_payload,
    stack_from_preview_payload,
    stack_to_preview_payload,
)


def test_dmd_array_to_display_uses_physical_screen_orientation() -> None:
    dmd_array = np.zeros(DMD_WIDTH_HEIGHT, dtype=np.uint8)

    display_array = dmd_array_to_display(dmd_array)

    assert display_array.shape == DMD_DISPLAY_SHAPE
    assert display_array.shape[1] > display_array.shape[0]


def test_dmd_placeholder_has_wide_display_shape() -> None:
    placeholder = make_dmd_placeholder()

    assert placeholder.shape == DMD_DISPLAY_SHAPE
    assert placeholder.shape[1] > placeholder.shape[0]


def test_brightness_histogram_is_rgb_and_updates_from_image() -> None:
    blank = make_brightness_histogram()
    image = np.array([[0, 0, 255], [128, 128, 255]], dtype=np.uint8)
    active = make_brightness_histogram(image)

    assert blank.ndim == 3
    assert blank.shape[-1] == 3
    assert active.sum() > blank.sum()


def test_visual_workspace_is_one_rgb_dashboard_image() -> None:
    workspace = make_visual_workspace()

    assert workspace.shape == (*WORKSPACE_SHAPE, 3)
    assert workspace.dtype == np.uint8


def test_visual_workspace_panels_are_vertically_stacked() -> None:
    assert MAIN_RECT[0] == SPECTRUM_RECT[0] == DMD_RECT[0]
    assert MAIN_RECT[1] + MAIN_RECT[3] < SPECTRUM_RECT[1]
    assert SPECTRUM_RECT[1] + SPECTRUM_RECT[3] < DMD_RECT[1]
    assert WORKSPACE_SHAPE[1] == MAIN_RECT[0] + MAIN_RECT[2] + MAIN_RECT[0]
    assert WORKSPACE_SHAPE[0] == DMD_RECT[1] + DMD_RECT[3] + MAIN_RECT[1]


def test_visual_workspace_stack_has_one_dashboard_per_plane() -> None:
    stack = np.arange(3 * 12, dtype=np.uint16).reshape(3, 3, 4)

    workspace_stack = make_visual_workspace_stack(image_stack=stack)

    assert workspace_stack.shape == (3, *WORKSPACE_SHAPE, 3)
    assert workspace_stack.dtype == np.uint8


def test_preview_payload_round_trip_downsamples_large_images() -> None:
    image = np.arange(100 * 200, dtype=np.uint16).reshape(100, 200)

    payload = array_to_preview_payload(image, max_shape=(10, 20))
    preview = array_from_preview_payload(payload)

    assert preview.shape == (10, 20)
    assert preview.dtype == np.dtype("uint16")


def test_stack_preview_payload_round_trip_downsamples_each_plane() -> None:
    stack = np.arange(3 * 100 * 200, dtype=np.uint16).reshape(3, 100, 200)

    payload = stack_to_preview_payload(stack, max_shape=(10, 20))
    preview = stack_from_preview_payload(payload)

    assert payload["is_stack"] is True
    assert preview.shape == (3, 10, 20)
    assert preview.dtype == np.dtype("uint16")


def test_dmd_array_to_display_accepts_display_oriented_preview() -> None:
    preview = np.zeros((10, 20), dtype=np.uint8)

    display = dmd_array_to_display(preview)

    assert display.shape == preview.shape
