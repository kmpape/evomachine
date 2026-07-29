from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from evomachine.config import DMD_WIDTH_HEIGHT
from evomachine.gui.image_payloads import array_from_preview_payload, stack_from_preview_payload


VISUAL_WORKSPACE_LAYER = "Visual workspace"
LAST_IMAGE_LAYER = "Last acquired image"
BRIGHTNESS_LAYER = "Brightness spectrum"
DMD_LAYER = "DMD pattern"

DEFAULT_CAMERA_DISPLAY_SHAPE = (512, 512)
DMD_DISPLAY_SHAPE = (DMD_WIDTH_HEIGHT[1], DMD_WIDTH_HEIGHT[0])
HISTOGRAM_BINS = 256
AUTO_CONTRAST_PERCENTILES = (0.5, 99.5)

BACKGROUND = np.array([9, 11, 14], dtype=np.uint8)
PANEL = np.array([21, 24, 29], dtype=np.uint8)
PANEL_ALT = np.array([28, 32, 38], dtype=np.uint8)
BORDER = (78, 85, 96)
TEXT = (230, 234, 241)
MUTED_TEXT = (150, 158, 170)

PANEL_X = 28
PANEL_Y = 36
PANEL_GAP = 34
PANEL_WIDTH = 764
PANEL_PAD = 18
MAIN_CONTENT_TOP = 58
SPECTRUM_CONTENT_TOP = 42
DMD_CONTENT_TOP = 48
SPECTRUM_HEIGHT = 210
MIN_CONTENT_WIDTH = PANEL_WIDTH - 2 * PANEL_PAD


def _magnified_shape(shape: tuple[int, int], target_width: int) -> tuple[int, int]:
    height, width = shape
    if width >= target_width:
        return height, width
    scale = target_width / width
    return max(1, int(round(height * scale))), target_width


def _visual_workspace_layout(
        main_shape: tuple[int, int],
        dmd_shape: tuple[int, int],
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int], tuple[int, int, int, int], tuple[int, int]]:
    content_width = max(MIN_CONTENT_WIDTH, main_shape[1], dmd_shape[1])
    main_display_shape = _magnified_shape(main_shape, content_width)
    dmd_display_shape = _magnified_shape(dmd_shape, content_width)
    panel_width = content_width + 2 * PANEL_PAD
    main_rect = (PANEL_X, PANEL_Y, panel_width, MAIN_CONTENT_TOP + main_display_shape[0] + PANEL_PAD)
    spectrum_rect = (
        PANEL_X,
        main_rect[1] + main_rect[3] + PANEL_GAP,
        panel_width,
        SPECTRUM_HEIGHT,
    )
    dmd_rect = (
        PANEL_X,
        spectrum_rect[1] + spectrum_rect[3] + PANEL_GAP,
        panel_width,
        DMD_CONTENT_TOP + dmd_display_shape[0] + PANEL_PAD,
    )
    workspace_shape = (dmd_rect[1] + dmd_rect[3] + PANEL_Y, PANEL_X + panel_width + PANEL_X)
    return main_rect, spectrum_rect, dmd_rect, workspace_shape


MAIN_RECT, SPECTRUM_RECT, DMD_RECT, WORKSPACE_SHAPE = _visual_workspace_layout(
    DEFAULT_CAMERA_DISPLAY_SHAPE,
    DMD_DISPLAY_SHAPE,
)


def dmd_array_to_display(array: np.ndarray) -> np.ndarray:
    """Return a DMD image in normal image-display orientation."""
    if array.ndim != 2:
        raise ValueError(f"DMD display image must be 2D, received shape {array.shape}.")
    if array.shape == DMD_WIDTH_HEIGHT:
        return array.T
    return array


def make_main_image_placeholder(shape: tuple[int, int] = DEFAULT_CAMERA_DISPLAY_SHAPE) -> np.ndarray:
    """Create a visible blank image placeholder for the central viewer."""
    image = np.zeros(shape, dtype=np.uint16)
    border = max(1, min(shape) // 96)
    marker = np.iinfo(image.dtype).max // 12
    image[:border, :] = marker
    image[-border:, :] = marker
    image[:, :border] = marker
    image[:, -border:] = marker
    return image


def make_dmd_placeholder() -> np.ndarray:
    """Create a visible DMD placeholder with the physical DMD aspect ratio."""
    image = np.zeros(DMD_DISPLAY_SHAPE, dtype=np.uint8)
    border = max(1, min(DMD_DISPLAY_SHAPE) // 120)
    image[:border, :] = 64
    image[-border:, :] = 64
    image[:, :border] = 64
    image[:, -border:] = 64
    image[::160, :] = 24
    image[:, ::160] = 24
    return image


def make_brightness_histogram(
        image: np.ndarray | None = None,
        size: tuple[int, int] = (140, 464),
) -> np.ndarray:
    """Create an RGB histogram showing pixel brightness distribution."""
    height, width = size
    histogram = np.zeros((height, width, 3), dtype=np.uint8)
    histogram[:, :, :] = PANEL
    pil_histogram = Image.fromarray(histogram)
    draw = ImageDraw.Draw(pil_histogram)
    font = _font(14)

    left, top, right, bottom = 42, 12, width - 18, height - 30
    draw.line((left, top, left, bottom), fill=BORDER, width=1)
    draw.line((left, bottom, right, bottom), fill=BORDER, width=1)

    if image is None:
        draw.text((left + 82, top + 40), "No image acquired", fill=MUTED_TEXT, font=font)
        return np.asarray(pil_histogram)

    values = _grayscale_values(image)
    if values.size == 0:
        draw.text((left + 90, top + 40), "No pixel data", fill=MUTED_TEXT, font=font)
        return np.asarray(pil_histogram)

    x_min, x_max = _brightness_axis_range(image, values)
    if x_max <= x_min:
        x_max = x_min + 1.0
    counts, _edges = np.histogram(values, bins=HISTOGRAM_BINS, range=(x_min, x_max))
    max_count = int(counts.max())
    if max_count > 0:
        bar_area_width = right - left
        bar_width = max(1, bar_area_width / HISTOGRAM_BINS)
        plot_height = bottom - top
        for index, count in enumerate(counts):
            x0 = left + int(round(index * bar_width))
            x1 = left + int(round((index + 1) * bar_width))
            y1 = bottom
            y0 = bottom - int(round(plot_height * int(count) / max_count))
            draw.rectangle((x0, y0, max(x0 + 1, x1 - 1), y1), fill=(190, 202, 220))

    return np.asarray(pil_histogram)


def make_visual_workspace(
        *,
        last_image: np.ndarray | None = None,
        camera_shape: tuple[int, int] = DEFAULT_CAMERA_DISPLAY_SHAPE,
    dmd_pattern: np.ndarray | None = None,
    show_last_image: bool = True,
) -> np.ndarray:
    """Compose the central visual dashboard as one Napari RGB image layer."""
    main_source = last_image if last_image is not None else make_main_image_placeholder(camera_shape)
    dmd_source = make_dmd_placeholder() if dmd_pattern is None else dmd_array_to_display(dmd_pattern)
    main_rect, spectrum_rect, dmd_rect, workspace_shape = _visual_workspace_layout(
        main_source.shape[:2],
        dmd_source.shape[:2],
    )

    canvas = np.zeros((*workspace_shape, 3), dtype=np.uint8)
    canvas[:, :, :] = BACKGROUND
    pil_canvas = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil_canvas)

    _draw_panel(draw, main_rect, "Last acquired image")
    displayed_main_source = (
        main_source
        if show_last_image
        else np.zeros(main_source.shape[:2], dtype=np.uint8)
    )
    _paste_into_rect(
        pil_canvas,
        displayed_main_source,
        _content_rect(main_rect, top=MAIN_CONTENT_TOP, pad=PANEL_PAD),
        magnify_to_width=True,
    )

    _draw_panel(draw, spectrum_rect, "Brightness histogram")
    spectrum_content_rect = _content_rect(spectrum_rect, top=SPECTRUM_CONTENT_TOP, pad=PANEL_PAD)
    histogram = make_brightness_histogram(
        last_image,
        size=(spectrum_content_rect[3], spectrum_content_rect[2]),
    )
    _paste_into_rect(pil_canvas, histogram, spectrum_content_rect)

    _draw_panel(draw, dmd_rect, "DMD window")
    _paste_into_rect(
        pil_canvas,
        dmd_source,
        _content_rect(dmd_rect, top=DMD_CONTENT_TOP, pad=PANEL_PAD),
        magnify_to_width=True,
    )

    return np.asarray(pil_canvas)


def make_visual_workspace_stack(
        *,
        image_stack: np.ndarray,
        camera_shape: tuple[int, int] = DEFAULT_CAMERA_DISPLAY_SHAPE,
    dmd_pattern: np.ndarray | None = None,
    show_last_image: bool = True,
) -> np.ndarray:
    """Compose a stack of central visual dashboards, one per acquired plane."""
    if image_stack.ndim == 2:
        planes = image_stack[np.newaxis, ...]
    elif image_stack.ndim == 3 or (image_stack.ndim == 4 and image_stack.shape[-1] in {3, 4}):
        planes = image_stack
    else:
        raise ValueError(f"Expected image stack, received shape {image_stack.shape}.")
    return np.stack([
        make_visual_workspace(
            last_image=plane,
            camera_shape=camera_shape,
            dmd_pattern=dmd_pattern,
            show_last_image=show_last_image,
        )
        for plane in planes
    ], axis=0)


class CentralVisualWorkspace:
    """Own the Napari central visual layer used by the EvoMachine GUI."""

    def __init__(self, viewer: Any, controller: Any):
        self.viewer = viewer
        self.controller = controller
        self.camera_shape = DEFAULT_CAMERA_DISPLAY_SHAPE
        self.last_image: np.ndarray | None = None
        self.last_stack: np.ndarray | None = None
        self.dmd_pattern: np.ndarray | None = None
        self._setup_layer()
        self._connect_controller()

    def _setup_layer(self) -> None:
        self._remove_obsolete_layers()
        layer = self._layer(VISUAL_WORKSPACE_LAYER)
        data = self._workspace_image()
        if layer is None:
            self.viewer.add_image(data, name=VISUAL_WORKSPACE_LAYER, rgb=True)
        else:
            layer.data = data
        layer = self._layer(LAST_IMAGE_LAYER)
        if layer is not None:
            self.viewer.layers.remove(layer)
        self.viewer.grid.enabled = False
        self.viewer.reset_view()

    def _connect_controller(self) -> None:
        self.controller.camera_status_received.connect(self.update_camera_status)
        self.controller.dmd_status_received.connect(self.update_dmd_status)
        self.controller.frame_received.connect(self.update_frame)

    def update_camera_status(self, payload: dict) -> None:
        image_shape = payload.get("image_shape")
        if self.last_image is None and self._is_image_shape(image_shape):
            self.camera_shape = tuple(image_shape)
            self._refresh()

    def update_last_image(self, image: np.ndarray) -> None:
        self.last_stack = None
        self.last_image = image
        self._refresh()
        self._update_camera_layer(image)

    def update_last_stack(self, stack: np.ndarray) -> None:
        self.last_stack = stack
        self.last_image = stack[-1]
        self._refresh()
        self._update_camera_layer(stack)

    def update_dmd_pattern(self, pattern: np.ndarray) -> None:
        self.dmd_pattern = pattern
        self._refresh()

    def update_dmd_status(self, payload: dict) -> None:
        preview = array_from_preview_payload(payload.get("preview"))
        if preview is not None:
            self.update_dmd_pattern(preview)

    def update_frame(self, payload: dict) -> None:
        stack_preview = stack_from_preview_payload(payload.get("stack_preview"))
        if stack_preview is not None:
            self.update_last_stack(stack_preview)
            return
        preview = array_from_preview_payload(payload.get("preview"))
        if preview is not None:
            self.update_last_image(preview)

    def _refresh(self) -> None:
        layer = self._layer(VISUAL_WORKSPACE_LAYER)
        if layer is not None:
            layer.data = self._workspace_image()
            self.viewer.reset_view()

    def _workspace_image(self) -> np.ndarray:
        if self.last_stack is not None:
            return make_visual_workspace_stack(
                image_stack=self.last_stack,
                camera_shape=self.camera_shape,
                dmd_pattern=self.dmd_pattern,
                show_last_image=False,
            )
        return make_visual_workspace(
            last_image=self.last_image,
            camera_shape=self.camera_shape,
            dmd_pattern=self.dmd_pattern,
            show_last_image=False,
        )

    def _remove_obsolete_layers(self) -> None:
        obsolete = {BRIGHTNESS_LAYER, DMD_LAYER}
        for layer in list(self.viewer.layers):
            if layer.name in obsolete:
                self.viewer.layers.remove(layer)

    def _layer(self, name: str) -> Any | None:
        try:
            return self.viewer.layers[name]
        except KeyError:
            return None

    def _update_camera_layer(self, data: np.ndarray) -> None:
        """Add or update the raw camera layer without replacing display settings."""
        layer = self._layer(LAST_IMAGE_LAYER)
        if layer is None:
            layer = self.viewer.add_image(
                data,
                name=LAST_IMAGE_LAYER,
                colormap="gray",
                contrast_limits=percentile_contrast_limits(data),
            )
        else:
            layer.data = data
        self._position_camera_layer(layer=layer, image_shape=data.shape[-2:], ndim=data.ndim)

    @staticmethod
    def _position_camera_layer(layer: Any, image_shape: tuple[int, int], ndim: int) -> None:
        main_rect, _spectrum_rect, _dmd_rect, _workspace_shape = _visual_workspace_layout(
            image_shape,
            DMD_DISPLAY_SHAPE,
        )
        content_rect = _content_rect(main_rect, top=MAIN_CONTENT_TOP, pad=PANEL_PAD)
        scale = content_rect[2] / image_shape[1]
        leading_dimensions = max(0, ndim - 2)
        layer.scale = (*([1.0] * leading_dimensions), scale, scale)
        layer.translate = (
            *([0.0] * leading_dimensions),
            content_rect[1],
            content_rect[0],
        )

    @staticmethod
    def _is_image_shape(value: Any) -> bool:
        return (
            isinstance(value, list | tuple)
            and len(value) == 2
            and all(isinstance(item, int) and item > 0 for item in value)
        )


def percentile_contrast_limits(
        image: np.ndarray,
        percentiles: tuple[float, float] = AUTO_CONTRAST_PERCENTILES,
) -> tuple[float, float]:
    """Return robust display limits while ignoring non-finite pixels."""
    values = np.asarray(image)
    if not np.issubdtype(values.dtype, np.number):
        raise TypeError(f"Contrast image must be numeric, received dtype {values.dtype}.")
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    low, high = (float(value) for value in np.percentile(finite, percentiles))
    if high <= low:
        high = low + 1.0
    return low, high


def _draw_panel(
        draw: ImageDraw.ImageDraw,
        rect: tuple[int, int, int, int],
        title: str,
        subtitle: str | None = None,
) -> None:
    x, y, width, height = rect
    draw.rounded_rectangle((x, y, x + width, y + height), radius=6, fill=tuple(PANEL.tolist()), outline=BORDER)
    draw.text((x + 18, y + 14), title, fill=TEXT, font=_font(22))
    if subtitle is not None:
        draw.text((x + 18, y + 40), subtitle, fill=MUTED_TEXT, font=_font(16))


def _content_rect(rect: tuple[int, int, int, int], *, top: int, pad: int) -> tuple[int, int, int, int]:
    x, y, width, height = rect
    return x + pad, y + top, width - 2 * pad, height - top - pad


def _paste_into_rect(
        canvas: Image.Image,
        source: np.ndarray,
        rect: tuple[int, int, int, int],
        *,
        magnify_to_width: bool = False,
) -> None:
    x, y, width, height = rect
    rgb = _normalise_to_rgb(source)
    image = Image.fromarray(rgb)
    if magnify_to_width and image.width < width:
        target_height = max(1, int(round(image.height * width / image.width)))
        image = image.resize((width, target_height), Image.Resampling.NEAREST)
    if image.width > width or image.height > height:
        raise ValueError(
            f"Source image shape {(image.height, image.width)} exceeds target rect {(height, width)}."
        )
    backing = Image.new("RGB", (width, height), tuple(PANEL_ALT.tolist()))
    offset = ((width - image.width) // 2, (height - image.height) // 2)
    backing.paste(image, offset)
    canvas.paste(backing, (x, y))


def _grayscale_values(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        values = image.astype(np.float64, copy=False).ravel()
    elif image.ndim == 3 and image.shape[-1] >= 3:
        rgb = image[..., :3].astype(np.float64, copy=False)
        values = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).ravel()
    else:
        raise ValueError(f"Expected a 2D or RGB image array, received shape {image.shape}.")
    return values[np.isfinite(values)]


def _brightness_axis_range(image: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    if np.issubdtype(image.dtype, np.unsignedinteger):
        info = np.iinfo(image.dtype)
        return float(info.min), float(info.max)
    if np.issubdtype(image.dtype, np.signedinteger):
        info = np.iinfo(image.dtype)
        return float(info.min), float(info.max)
    if values.size and float(values.min()) >= 0.0 and float(values.max()) <= 1.0:
        return 0.0, 1.0
    return float(values.min()), float(values.max())


def _format_axis_value(value: float) -> str:
    if abs(value) >= 1000:
        return f"{value:.0f}"
    if value.is_integer():
        return f"{value:.0f}"
    return f"{value:.2f}"


def _normalise_to_rgb(array: np.ndarray) -> np.ndarray:
    if array.ndim == 3 and array.shape[-1] in {3, 4}:
        rgb = array[..., :3]
        if rgb.dtype == np.uint8:
            return rgb
        return _scale_to_uint8(rgb)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D or RGB image array, received shape {array.shape}.")
    gray = _scale_to_uint8(array)
    return np.repeat(gray[:, :, np.newaxis], 3, axis=2)


def _scale_to_uint8(array: np.ndarray) -> np.ndarray:
    if array.dtype == np.uint8:
        return array
    values = array.astype(np.float64, copy=False)
    minimum = float(np.nanmin(values))
    maximum = float(np.nanmax(values))
    if maximum <= minimum:
        return np.zeros(array.shape, dtype=np.uint8)
    scaled = (values - minimum) / (maximum - minimum)
    return np.clip(scaled * 255, 0, 255).astype(np.uint8)


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", size)
    except OSError:
        return ImageFont.load_default()
