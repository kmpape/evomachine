from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from evomachine.config import DMD_WIDTH_HEIGHT
from evomachine.gui.central_workspace import (
    DMD_DISPLAY_SHAPE,
    CENTRAL_VIEW_ZOOM,
    DMD_RECT,
    HISTOGRAM_BINS,
    MAIN_RECT,
    SPECTRUM_RECT,
    WORKSPACE_SHAPE,
    dmd_array_to_display,
    fit_central_viewer,
    make_brightness_histogram,
    make_dmd_placeholder,
    make_visual_workspace_stack,
    make_visual_workspace,
    percentile_contrast_limits,
    _content_rect,
    _visual_workspace_layout,
)
from evomachine.gui.image_payloads import (
    IMAGE_TRANSPORT_DIR_ENV,
    IMAGE_TRANSPORT_RAW,
    IMAGE_TRANSPORT_SOCKET_TIFF,
    IMAGE_TRANSPORT_TEMP_TIFF,
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


def test_brightness_histogram_exposes_all_eight_bit_display_levels() -> None:
    assert HISTOGRAM_BINS == 256


def test_visual_workspace_is_one_rgb_dashboard_image() -> None:
    workspace = make_visual_workspace()

    assert workspace.shape == (*WORKSPACE_SHAPE, 3)
    assert workspace.dtype == np.uint8


def test_central_viewer_fit_reduces_default_outer_margin() -> None:
    viewer = SimpleNamespace(
        camera=SimpleNamespace(zoom=1.0),
        reset_count=0,
    )

    def reset_view():
        viewer.reset_count += 1
        viewer.camera.zoom = 2.0

    viewer.reset_view = reset_view

    fit_central_viewer(viewer)

    assert viewer.reset_count == 1
    assert viewer.camera.zoom == 2.0 * CENTRAL_VIEW_ZOOM


def test_visual_workspace_magnifies_small_camera_image_to_dmd_width() -> None:
    image = np.arange(48 * 64, dtype=np.uint16).reshape(48, 64)
    dmd_pattern = np.zeros(DMD_DISPLAY_SHAPE, dtype=np.uint8)

    workspace = make_visual_workspace(last_image=image, dmd_pattern=dmd_pattern)
    main_rect, _spectrum_rect, dmd_rect, workspace_shape = _visual_workspace_layout(
        image.shape,
        dmd_pattern.shape,
    )
    main_content = _content_rect(main_rect, top=58, pad=18)
    dmd_content = _content_rect(dmd_rect, top=48, pad=18)

    assert workspace.shape == (*workspace_shape, 3)
    assert main_content[2] == dmd_content[2]
    assert workspace[
        main_content[1]:main_content[1] + main_content[3],
        main_content[0]:main_content[0] + main_content[2],
    ].sum() > 0


def test_visual_workspace_panels_are_vertically_stacked() -> None:
    assert MAIN_RECT[0] == SPECTRUM_RECT[0] == DMD_RECT[0]
    assert MAIN_RECT[1] + MAIN_RECT[3] < SPECTRUM_RECT[1]
    assert SPECTRUM_RECT[1] + SPECTRUM_RECT[3] < DMD_RECT[1]
    assert WORKSPACE_SHAPE[1] == MAIN_RECT[0] + MAIN_RECT[2] + MAIN_RECT[0]
    assert WORKSPACE_SHAPE[0] == DMD_RECT[1] + DMD_RECT[3] + MAIN_RECT[1]


def test_visual_workspace_stack_has_one_dashboard_per_plane() -> None:
    stack = np.arange(3 * 12, dtype=np.uint16).reshape(3, 3, 4)

    workspace_stack = make_visual_workspace_stack(image_stack=stack)
    single_workspace = make_visual_workspace(last_image=stack[0])

    assert workspace_stack.shape == (3, *single_workspace.shape)
    assert workspace_stack.dtype == np.uint8


def test_percentile_contrast_limits_ignore_sparse_outliers() -> None:
    image = np.full((100, 100), 100, dtype=np.uint16)
    image[:100, :10] = np.arange(1000, dtype=np.uint16).reshape(100, 10)
    image[0, 0] = np.iinfo(np.uint16).max

    low, high = percentile_contrast_limits(image)

    assert low >= 50
    assert high < np.iinfo(np.uint16).max


def test_percentile_contrast_limits_expand_constant_images() -> None:
    assert percentile_contrast_limits(np.full((3, 4), 7, dtype=np.uint16)) == (7.0, 8.0)


def test_preview_payload_round_trip_preserves_full_image_as_raw_bytes() -> None:
    image = np.arange(100 * 200, dtype=np.uint16).reshape(100, 200)

    payload = array_to_preview_payload(image, transport=IMAGE_TRANSPORT_RAW)
    preview = array_from_preview_payload(payload)

    assert payload["encoding"] == "raw"
    assert preview.shape == image.shape
    assert preview.dtype == np.dtype("uint16")
    assert np.array_equal(preview, image)


def test_preview_payload_round_trip_preserves_full_image_as_socket_tiff() -> None:
    image = np.arange(100 * 200, dtype=np.uint16).reshape(100, 200)

    payload = array_to_preview_payload(image, transport=IMAGE_TRANSPORT_SOCKET_TIFF)
    preview = array_from_preview_payload(payload)

    assert payload["encoding"] == "tiff"
    assert preview.shape == image.shape
    assert preview.dtype == np.dtype("uint16")
    assert np.array_equal(preview, image)


def test_preview_payload_round_trip_preserves_full_image_as_temp_tiff(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(IMAGE_TRANSPORT_DIR_ENV, str(tmp_path))
    image = np.arange(100 * 200, dtype=np.uint16).reshape(100, 200)

    payload = array_to_preview_payload(image, transport=IMAGE_TRANSPORT_TEMP_TIFF)
    tiff_path = Path(payload["path"])

    assert payload["encoding"] == "tiff_path"
    assert tiff_path.exists()
    preview = array_from_preview_payload(payload)
    assert not tiff_path.exists()
    assert preview.shape == image.shape
    assert preview.dtype == np.dtype("uint16")
    assert np.array_equal(preview, image)


def test_preview_payload_preserves_thin_bright_features() -> None:
    image = np.zeros((100, 200), dtype=np.uint8)
    image[:, 97] = 255

    payload = array_to_preview_payload(image)
    preview = array_from_preview_payload(payload)

    assert payload["encoding"] == "packed_binary"
    assert preview.shape == image.shape
    assert preview.max() == 255
    assert np.array_equal(preview, image)


def test_preview_payload_packs_binary_masks_losslessly() -> None:
    image = np.zeros((7, 11), dtype=np.uint8)
    image[2:5, 3:8] = 255

    payload = array_to_preview_payload(image)
    preview = array_from_preview_payload(payload)

    assert payload["encoding"] == "packed_binary"
    assert isinstance(payload["data"], str)
    assert preview.dtype == np.dtype("uint8")
    assert np.array_equal(preview, image)


def test_stack_preview_payload_round_trip_preserves_each_plane() -> None:
    stack = np.arange(3 * 100 * 200, dtype=np.uint16).reshape(3, 100, 200)

    payload = stack_to_preview_payload(stack, transport=IMAGE_TRANSPORT_SOCKET_TIFF)
    preview = stack_from_preview_payload(payload)

    assert payload["is_stack"] is True
    assert payload["encoding"] == "tiff"
    assert preview.shape == stack.shape
    assert preview.dtype == np.dtype("uint16")
    assert np.array_equal(preview, stack)


def test_dmd_array_to_display_accepts_display_oriented_preview() -> None:
    preview = np.zeros((10, 20), dtype=np.uint8)

    display = dmd_array_to_display(preview)

    assert display.shape == preview.shape
