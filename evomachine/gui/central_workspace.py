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

WORKSPACE_SHAPE = (900, 1400)
DEFAULT_CAMERA_DISPLAY_SHAPE = (512, 512)
DMD_DISPLAY_SHAPE = (DMD_WIDTH_HEIGHT[1], DMD_WIDTH_HEIGHT[0])
HISTOGRAM_BINS = 64

BACKGROUND = np.array([9, 11, 14], dtype=np.uint8)
PANEL = np.array([21, 24, 29], dtype=np.uint8)
PANEL_ALT = np.array([28, 32, 38], dtype=np.uint8)
BORDER = (78, 85, 96)
TEXT = (230, 234, 241)
MUTED_TEXT = (150, 158, 170)

MAIN_RECT = (36, 52, 780, 820)
SPECTRUM_RECT = (860, 52, 504, 210)
DMD_CONTENT_WIDTH = 468
DMD_CONTENT_HEIGHT = round(DMD_CONTENT_WIDTH * DMD_DISPLAY_SHAPE[0] / DMD_DISPLAY_SHAPE[1])
DMD_RECT = (860, 332, 504, 48 + DMD_CONTENT_HEIGHT + 18)


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
) -> np.ndarray:
    """Compose the central visual dashboard as one Napari RGB image layer."""
    canvas = np.zeros((*WORKSPACE_SHAPE, 3), dtype=np.uint8)
    canvas[:, :, :] = BACKGROUND
    pil_canvas = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil_canvas)

    _draw_panel(draw, MAIN_RECT, "Last acquired image")
    main_source = last_image if last_image is not None else make_main_image_placeholder(camera_shape)
    _paste_into_rect(pil_canvas, main_source, _content_rect(MAIN_RECT, top=58, pad=18))

    _draw_panel(draw, SPECTRUM_RECT, "Brightness histogram")
    histogram = make_brightness_histogram(last_image)
    _paste_into_rect(pil_canvas, histogram, _content_rect(SPECTRUM_RECT, top=42, pad=18))

    _draw_panel(draw, DMD_RECT, "DMD window")
    dmd_source = make_dmd_placeholder() if dmd_pattern is None else dmd_array_to_display(dmd_pattern)
    _paste_into_rect(pil_canvas, dmd_source, _content_rect(DMD_RECT, top=48, pad=18), nearest=True)

    return np.asarray(pil_canvas)


def make_visual_workspace_stack(
        *,
        image_stack: np.ndarray,
        camera_shape: tuple[int, int] = DEFAULT_CAMERA_DISPLAY_SHAPE,
        dmd_pattern: np.ndarray | None = None,
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

    def update_last_stack(self, stack: np.ndarray) -> None:
        self.last_stack = stack
        self.last_image = stack[-1]
        self._refresh()

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
            )
        return make_visual_workspace(
            last_image=self.last_image,
            camera_shape=self.camera_shape,
            dmd_pattern=self.dmd_pattern,
        )

    def _remove_obsolete_layers(self) -> None:
        obsolete = {LAST_IMAGE_LAYER, BRIGHTNESS_LAYER, DMD_LAYER}
        for layer in list(self.viewer.layers):
            if layer.name in obsolete:
                self.viewer.layers.remove(layer)

    def _layer(self, name: str) -> Any | None:
        try:
            return self.viewer.layers[name]
        except KeyError:
            return None

    @staticmethod
    def _is_image_shape(value: Any) -> bool:
        return (
            isinstance(value, list | tuple)
            and len(value) == 2
            and all(isinstance(item, int) and item > 0 for item in value)
        )


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
        nearest: bool = False,
) -> None:
    x, y, width, height = rect
    rgb = _normalise_to_rgb(source)
    image = Image.fromarray(rgb)
    scale = min(width / image.width, height / image.height)
    target_size = (max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale))))
    resampling = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    image = image.resize(target_size, resampling)
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
