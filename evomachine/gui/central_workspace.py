from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFrame, QHBoxLayout, QLabel, QTabBar

from evomachine.config import DMD_WIDTH_HEIGHT
from evomachine.gui.image_payloads import array_from_preview_payload, stack_from_preview_payload


MICROSCOPE_LAYER = "Microscope"
DMD_LAYER = "DMD"
DEFAULT_CAMERA_DISPLAY_SHAPE = (512, 512)
DMD_DISPLAY_SHAPE = (DMD_WIDTH_HEIGHT[1], DMD_WIDTH_HEIGHT[0])
DMD_FRAME_COLOR = "#ff3030"
DMD_FRAME_THICKNESS = 3.0
FOCUS_CROP_LAYER = "Focus crops"
FOCUS_CROP_COLOR = "#26d9d9"
AUTO_CONTRAST_PERCENTILES = (0.5, 99.5)
CAMERA_VIEW_MARGIN = 0.05
DISPLAY_TAB_HEIGHT = 30
METADATA_STRIP_HEIGHT = 30


@dataclass(frozen=True)
class DisplayMetadata:
    """Describe compact display metadata.

    Args:
        text: Short text displayed beneath the image canvas.
        tooltip: Full text shown when hovering over the metadata strip.
    """

    text: str
    tooltip: str = ""


def fit_central_viewer(viewer: Any) -> None:
    """Fit the currently selected central image layer in the Napari canvas.

    Args:
        viewer: Napari viewer exposing ``reset_view``.
    """
    viewer.reset_view(margin=CAMERA_VIEW_MARGIN)


def dmd_array_to_display(array: np.ndarray) -> np.ndarray:
    """Return a two-dimensional DMD array in normal display orientation.

    Args:
        array: DMD pattern in hardware or already display-oriented coordinates.

    Returns:
        The pattern oriented with its wider dimension horizontal.
    """
    if array.ndim != 2:
        raise ValueError(f"DMD display image must be 2D, received shape {array.shape}.")
    if array.shape == DMD_WIDTH_HEIGHT:
        return array.T
    return array


def make_main_image_placeholder(shape: tuple[int, int] = DEFAULT_CAMERA_DISPLAY_SHAPE) -> np.ndarray:
    """Create a blank unsigned 16-bit camera placeholder.

    Args:
        shape: Placeholder height and width.

    Returns:
        A bordered two-dimensional placeholder image.
    """
    image = np.zeros(shape, dtype=np.uint16)
    border = max(1, min(shape) // 96)
    marker = np.iinfo(image.dtype).max // 12
    image[:border, :] = marker
    image[-border:, :] = marker
    image[:, :border] = marker
    image[:, -border:] = marker
    return image


def make_dmd_placeholder() -> np.ndarray:
    """Create a visible DMD placeholder using the physical aspect ratio.

    Returns:
        An eight-bit display-oriented DMD image.
    """
    image = np.zeros(DMD_DISPLAY_SHAPE, dtype=np.uint8)
    border = max(1, min(DMD_DISPLAY_SHAPE) // 120)
    image[:border, :] = 64
    image[-border:, :] = 64
    image[:, :border] = 64
    image[:, -border:] = 64
    return image


class CentralVisualWorkspace:
    """Manage mutually exclusive microscope and DMD layers in Napari.

    Args:
        viewer: Napari viewer receiving the two image layers.
        controller: Shared GUI controller providing camera, DMD, and frame signals.
    """

    def __init__(self, viewer: Any, controller: Any):
        """Create image, DMD, and crop layers and connect their GUI signals."""
        self.viewer = viewer
        self.controller = controller
        self.camera_shape = DEFAULT_CAMERA_DISPLAY_SHAPE
        self.last_image: np.ndarray | None = None
        self.last_stack: np.ndarray | None = None
        self.dmd_pattern: np.ndarray | None = None
        self.last_frame_payload: dict[str, Any] | None = None
        self.last_dmd_payload: dict[str, Any] | None = None
        self._has_real_camera_image = False
        self._has_real_dmd_image = False
        self.display_tabs = self._create_display_tabs()
        self.metadata_strip = self._create_metadata_strip()
        self._setup_layers()
        self._install_canvas_chrome()
        self._connect_controller()
        self._connect_viewer()
        self.select_display(0)

    def _create_display_tabs(self) -> QTabBar:
        """Create and return the compact image-mode selector."""
        tabs = QTabBar()
        tabs.addTab("Microscope")
        tabs.addTab("DMD")
        tabs.setExpanding(True)
        tabs.setFixedHeight(DISPLAY_TAB_HEIGHT)
        tabs.currentChanged.connect(self.select_display)
        return tabs

    def _create_metadata_strip(self) -> QFrame:
        """Create and return the fixed-height metadata strip."""
        frame = QFrame()
        frame.setObjectName("centralMetadataStrip")
        frame.setFixedHeight(METADATA_STRIP_HEIGHT)
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(8, 2, 8, 2)
        self.metadata_text = QLabel("No microscope image acquired")
        self.metadata_text.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.metadata_text)
        return frame

    def _setup_layers(self) -> None:
        """Replace obsolete dashboard layers with microscope and DMD layers."""
        for layer in list(self.viewer.layers):
            if layer.name in {"Visual workspace", "Brightness spectrum", "Last acquired image", DMD_LAYER, FOCUS_CROP_LAYER}:
                self.viewer.layers.remove(layer)
        self.microscope_layer = self.viewer.add_image(
            make_main_image_placeholder(self.camera_shape), name=MICROSCOPE_LAYER, colormap="gray"
        )
        self.microscope_layer.bounding_box.visible = True
        self.microscope_layer.bounding_box.lines = True
        self.microscope_layer.bounding_box.line_color = DMD_FRAME_COLOR
        self.microscope_layer.bounding_box.line_thickness = DMD_FRAME_THICKNESS
        self.microscope_layer.bounding_box.points = False
        self.dmd_layer = self.viewer.add_image(
            make_dmd_placeholder(), name=DMD_LAYER, colormap="gray", visible=False
        )
        self.dmd_layer.bounding_box.visible = True
        self.dmd_layer.bounding_box.lines = True
        self.dmd_layer.bounding_box.line_color = DMD_FRAME_COLOR
        self.dmd_layer.bounding_box.line_thickness = DMD_FRAME_THICKNESS
        self.dmd_layer.bounding_box.points = False
        self.focus_crop_layer = self.viewer.add_shapes(
            [],
            name=FOCUS_CROP_LAYER,
            shape_type="rectangle",
            edge_color=FOCUS_CROP_COLOR,
            face_color="transparent",
            edge_width=3,
        )
        self.focus_crop_layer.editable = False
        self.viewer.grid.enabled = False

    def _install_canvas_chrome(self) -> None:
        """Place display tabs above and metadata beneath the Napari canvas."""
        qt_viewer = getattr(getattr(self.viewer, "window", None), "_qt_viewer", None)
        if qt_viewer is None:
            return
        qt_viewer.insertWidget(0, self.display_tabs)
        qt_viewer.addWidget(self.metadata_strip)
        qt_viewer.setCollapsible(qt_viewer.indexOf(self.display_tabs), False)
        qt_viewer.setCollapsible(qt_viewer.indexOf(self.metadata_strip), False)

    def _connect_controller(self) -> None:
        """Connect shared controller responses to central display updates."""
        self.controller.camera_status_received.connect(self.update_camera_status)
        self.controller.dmd_status_received.connect(self.update_dmd_status)
        self.controller.frame_received.connect(self.update_frame)
        self.controller.software_focus_status_received.connect(self.update_focus_crops)

    def _connect_viewer(self) -> None:
        """Connect stack-plane navigation to metadata refreshes."""
        current_step = getattr(getattr(self.viewer.dims, "events", None), "current_step", None)
        if current_step is not None:
            current_step.connect(self._on_dims_step_changed)

    def select_display(self, index: int) -> None:
        """Show one display layer and fit it to the canvas.

        Args:
            index: Zero for Microscope or one for DMD.
        """
        microscope_selected = index == 0
        self.microscope_layer.visible = microscope_selected
        self.dmd_layer.visible = not microscope_selected
        self.focus_crop_layer.visible = microscope_selected
        selected = self.microscope_layer if microscope_selected else self.dmd_layer
        selection = getattr(self.viewer.layers, "selection", None)
        if selection is not None:
            selection.active = selected
        self._set_dims_visible(microscope_selected and self.microscope_layer.ndim > 2)
        self._update_metadata()
        fit_central_viewer(self.viewer)

    def update_focus_crops(self, payload: dict[str, Any]) -> None:
        """Display authoritative focus crops as read-only microscope rectangles.

        Args:
            payload: Unified software-focus status containing configuration crops.
        """
        config = payload.get("config") or {}
        rectangles = []
        for box in config.get("cropping_boxes") or []:
            left, top = float(box["x_left"]), float(box["y_top"])
            right, bottom = float(box["x_right"]), float(box["y_bottom"])
            rectangles.append(np.asarray([
                [top, left], [top, right], [bottom, right], [bottom, left],
            ]))
        self.focus_crop_layer.data = rectangles
        self.focus_crop_layer.visible = self.display_tabs.currentIndex() == 0

    def update_camera_status(self, payload: dict[str, Any]) -> None:
        """Update placeholder dimensions from a camera status payload.

        Args:
            payload: Existing camera status response.
        """
        image_shape = payload.get("image_shape")
        if not self._has_real_camera_image and self._is_image_shape(image_shape):
            self.camera_shape = tuple(image_shape)
            self.microscope_layer.data = make_main_image_placeholder(self.camera_shape)
            self._update_metadata()

    def update_last_image(self, image: np.ndarray) -> None:
        """Display one microscope image while preserving later user zoom.

        Args:
            image: Two-dimensional acquired or loaded image.
        """
        first_image = not self._has_real_camera_image
        self.last_stack = None
        self.last_image = image
        self.microscope_layer.data = image
        if first_image:
            self.microscope_layer.contrast_limits = percentile_contrast_limits(image)
        self._has_real_camera_image = True
        self._set_dims_visible(False)
        self._update_metadata()
        if first_image and self.display_tabs.currentIndex() == 0:
            fit_central_viewer(self.viewer)

    def update_last_stack(self, stack: np.ndarray) -> None:
        """Display a navigable microscope Z-stack.

        Args:
            stack: Three-dimensional plane-height-width array.
        """
        first_image = not self._has_real_camera_image
        self.last_stack = stack
        self.last_image = stack[-1]
        self.microscope_layer.data = stack
        if first_image:
            self.microscope_layer.contrast_limits = percentile_contrast_limits(stack)
        self._has_real_camera_image = True
        self._set_dims_visible(self.display_tabs.currentIndex() == 0)
        self._update_metadata()
        if first_image and self.display_tabs.currentIndex() == 0:
            fit_central_viewer(self.viewer)

    def update_dmd_pattern(self, pattern: np.ndarray) -> None:
        """Update only the DMD preview layer.

        Args:
            pattern: DMD pattern in hardware or display orientation.
        """
        first_pattern = not self._has_real_dmd_image
        self.dmd_pattern = pattern
        self.dmd_layer.data = dmd_array_to_display(pattern)
        self._has_real_dmd_image = True
        self._update_metadata()
        if first_pattern and self.display_tabs.currentIndex() == 1:
            fit_central_viewer(self.viewer)

    def update_dmd_status(self, payload: dict[str, Any]) -> None:
        """Apply a DMD status payload and its optional preview.

        Args:
            payload: Existing DMD status response.
        """
        self.last_dmd_payload = dict(payload)
        preview = array_from_preview_payload(payload.get("preview"))
        if preview is not None:
            self.update_dmd_pattern(preview)
        else:
            self._update_metadata()

    def update_frame(self, payload: dict[str, Any]) -> None:
        """Apply an existing frame payload to the microscope layer.

        Args:
            payload: Frame response containing an image or stack preview.
        """
        self.last_frame_payload = dict(payload)
        stack_preview = stack_from_preview_payload(payload.get("stack_preview"))
        if stack_preview is not None:
            self.update_last_stack(stack_preview)
            return
        preview = array_from_preview_payload(payload.get("preview"))
        if preview is not None:
            self.update_last_image(preview)

    def _on_dims_step_changed(self, _event: Any = None) -> None:
        """Refresh stack metadata after selected-plane changes.

        Args:
            _event: Optional Napari dimension event, unused.
        """
        self._update_metadata()

    def _update_metadata(self) -> None:
        """Refresh compact text for the selected display."""
        metadata = self._microscope_metadata() if self.display_tabs.currentIndex() == 0 else self._dmd_metadata()
        self.metadata_text.setText(metadata.text)
        self.metadata_text.setToolTip(metadata.tooltip)

    def _microscope_metadata(self) -> DisplayMetadata:
        """Return compact metadata for the current microscope plane."""
        payload = self.last_frame_payload
        if payload is None:
            return DisplayMetadata("No microscope image acquired")
        plane, count = self._current_plane(payload)
        parts = [str(payload.get("kind", "frame")).replace("_", " ")]
        if count > 1:
            parts.append(f"plane {plane + 1}/{count}")
        fov_id = payload.get("fov_id")
        if fov_id is not None and fov_id != -1:
            parts.append(f"FoV {fov_id}")
        z_positions = payload.get("z_positions") or []
        if plane < len(z_positions):
            parts.append(f"Z {z_positions[plane]}")
        shape = payload.get("image_shape") or payload.get("stack_shape")
        if isinstance(shape, list | tuple) and len(shape) >= 2:
            parts.append(f"{shape[-1]}×{shape[-2]}")
        if payload.get("dtype"):
            parts.append(str(payload["dtype"]))
        paths = payload.get("saved_paths") or []
        path = paths[min(plane, len(paths) - 1)] if paths else payload.get("loaded_path")
        tooltip = str(path) if path else ""
        parts.append(Path(path).name if path else "not saved")
        return DisplayMetadata("  |  ".join(parts), tooltip)

    def _dmd_metadata(self) -> DisplayMetadata:
        """Return compact metadata for the current DMD status."""
        payload = self.last_dmd_payload or {}
        pattern = payload.get("last_pattern") or "No pattern"
        dimensions = payload.get("width_height")
        if isinstance(dimensions, list | tuple) and len(dimensions) == 2:
            dimension_text = f"{dimensions[0]}×{dimensions[1]}"
        else:
            dimension_text = f"{self.dmd_layer.data.shape[1]}×{self.dmd_layer.data.shape[0]}"
        calibration = payload.get("calibration_file")
        calibrated = payload.get("is_calibrated")
        status_text = "calibrated" if calibrated else "not calibrated"
        calibration_text = Path(calibration).name if calibration else "calibration unavailable"
        return DisplayMetadata(
            "  |  ".join((str(pattern), dimension_text, status_text, calibration_text)),
            str(calibration) if calibration else "",
        )

    def _set_dims_visible(self, visible: bool) -> None:
        """Set visibility of Napari's stack navigation widget.

        Args:
            visible: Whether stack navigation should be visible.
        """
        dims_widget = getattr(getattr(self.viewer.window, "_qt_viewer", None), "dims", None)
        if dims_widget is not None:
            dims_widget.setVisible(visible)

    def _current_plane(self, payload: dict[str, Any]) -> tuple[int, int]:
        """Return selected zero-based plane and available plane count.

        Args:
            payload: Current frame payload.

        Returns:
            Selected plane index and total plane count.
        """
        count = max(1, int(payload.get("planes", 1)))
        if count == 1:
            return 0, count
        current_step = tuple(getattr(self.viewer.dims, "current_step", ()))
        plane = int(current_step[-3]) if len(current_step) >= 3 else 0
        return min(max(plane, 0), count - 1), count

    @staticmethod
    def _is_image_shape(value: Any) -> bool:
        """Return whether a value is a valid two-dimensional image shape."""
        return (
            isinstance(value, list | tuple)
            and len(value) == 2
            and all(isinstance(item, int) and item > 0 for item in value)
        )


def percentile_contrast_limits(
    image: np.ndarray,
    percentiles: tuple[float, float] = AUTO_CONTRAST_PERCENTILES,
) -> tuple[float, float]:
    """Return robust display limits while ignoring non-finite pixels.

    Args:
        image: Numeric image or stack used to calculate limits.
        percentiles: Low and high percentiles used as display limits.

    Returns:
        Increasing low and high contrast limits.
    """
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
