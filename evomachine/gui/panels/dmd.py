from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (
    QComboBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from evomachine.config import CAM_WIDTH_HEIGHT, DMD_WIDTH_HEIGHT
from evomachine.gui.panels.config_dialog import ConfigDialog, ConfigFieldSpec


PATTERN_ACTIONS = (
    ("Empty", "empty"),
    ("Full", "full"),
    ("Rectangle", "rectangle"),
    ("Circle", "circle"),
    ("Checkerboard", "checkerboard"),
    ("Crosshair", "crosshair"),
)

UTILITY_ACTIONS = (("Refresh", "refresh"),)

CAM_ROWS, CAM_COLS = CAM_WIDTH_HEIGHT
DEFAULT_RECTANGLE_HEIGHT = CAM_ROWS // 2
DEFAULT_RECTANGLE_WIDTH = CAM_COLS // 2

SHAPE_CONFIG_FIELDS = (
    ("Rect row", "rectangle_row", (CAM_ROWS - DEFAULT_RECTANGLE_HEIGHT) // 2, 0, CAM_ROWS - 1),
    ("Rect col", "rectangle_col", (CAM_COLS - DEFAULT_RECTANGLE_WIDTH) // 2, 0, CAM_COLS - 1),
    ("Rect height", "rectangle_height", DEFAULT_RECTANGLE_HEIGHT, 1, CAM_ROWS),
    ("Rect width", "rectangle_width", DEFAULT_RECTANGLE_WIDTH, 1, CAM_COLS),
    ("Checker box", "checkerboard_box_size", 200, 1, max(CAM_WIDTH_HEIGHT)),
    ("Cross row", "crosshair_row", CAM_ROWS // 2, 0, CAM_ROWS - 1),
    ("Cross col", "crosshair_col", CAM_COLS // 2, 0, CAM_COLS - 1),
    ("Cross width", "crosshair_width", 1, 1, max(CAM_WIDTH_HEIGHT)),
    ("Circle row", "circle_row", CAM_ROWS // 2, 0, CAM_ROWS - 1),
    ("Circle col", "circle_col", CAM_COLS // 2, 0, CAM_COLS - 1),
    ("Circle radius", "circle_radius", min(CAM_WIDTH_HEIGHT) // 8, 1, max(CAM_WIDTH_HEIGHT)),
)


class DmdPanel(QGroupBox):
    """DMD controls and pattern buttons."""

    def __init__(self, controller, parent: QWidget | None = None):
        super().__init__("DMD", parent)
        self.controller = controller
        self.devices_initialised = False
        self.strategy_running = False
        self.pattern_buttons: dict[str, QPushButton] = {}
        self.utility_buttons: dict[str, QPushButton] = {}
        self.calibration_buttons: dict[str, QPushButton] = {}
        self.config_values = self._default_shape_config_values()
        self.config_limits = {
            field_name: (minimum, maximum)
            for _label, field_name, _default, minimum, maximum in SHAPE_CONFIG_FIELDS
        }
        self.calibration_file_combo = QComboBox()
        self.configure_pattern_button = QPushButton("Configure Pattern")
        self.load_calibration_button = QPushButton("Load Existing")
        self.show_calibration_plot_button = QPushButton("Show Calibration Plot")
        self.calibration_plot_windows: list[DmdCalibrationPlotWindow] = []
        self.status_label = QLabel("Run Initialise Devices before using DMD controls.")
        self.status_label.setWordWrap(True)
        self.status_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)

        layout = QVBoxLayout()
        self._add_button_section(layout, "Patterns", PATTERN_ACTIONS, self.pattern_buttons)
        self._add_config_section(layout)
        self._add_calibration_section(layout)
        self._add_button_section(layout, "Utilities", UTILITY_ACTIONS, self.utility_buttons)
        layout.addWidget(self.status_label)
        self.setLayout(layout)

        for pattern, button in self.pattern_buttons.items():
            button.clicked.connect(
                lambda _checked=False, selected=pattern: self._display_pattern(selected)
            )
        self.configure_pattern_button.clicked.connect(self._open_pattern_config_dialog)
        self.calibration_buttons["calibrate"].clicked.connect(self._calibrate)
        self.load_calibration_button.clicked.connect(self._load_selected_calibration)
        self.show_calibration_plot_button.clicked.connect(self._request_calibration_plot)
        self.utility_buttons["refresh"].clicked.connect(self.controller.refresh_dmd)
        self.calibration_buttons["calibrate"].setToolTip(
            "Runs the default DMD calibration workflow and saves a new file."
        )
        self.load_calibration_button.setToolTip("Loads the selected stored DMD calibration file.")
        self.show_calibration_plot_button.setToolTip(
            "Opens a separate plot of paired DMD and camera calibration points."
        )
        self.controller.dmd_status_received.connect(self.update_status)
        self.controller.dmd_calibration_points_received.connect(self._show_calibration_plot)
        self.controller.lifecycle_status_received.connect(self.update_lifecycle_status)
        self.controller.strategy_status_received.connect(self.update_strategy_status)
        self.controller.response_error.connect(self._show_error)
        self._sync_controls_enabled()

    def _add_button_section(
        self,
        layout: QVBoxLayout,
        title: str,
        actions: tuple[tuple[str, str], ...],
        buttons: dict[str, QPushButton],
    ) -> None:
        section_label = QLabel(title)
        section_label.setStyleSheet("font-weight: 600;")
        grid = QGridLayout()
        for index, (button_label, action_name) in enumerate(actions):
            button = QPushButton(button_label)
            button.setProperty("dmd_action", action_name)
            button.setEnabled(False)
            buttons[action_name] = button
            row, column = divmod(index, 2)
            grid.addWidget(button, row, column)
        layout.addWidget(section_label)
        layout.addLayout(grid)

    def _add_config_section(self, layout: QVBoxLayout) -> None:
        section_label = QLabel("Pattern config")
        section_label.setStyleSheet("font-weight: 600;")
        layout.addWidget(section_label)
        layout.addWidget(self.configure_pattern_button)

    def _add_calibration_section(self, layout: QVBoxLayout) -> None:
        section_label = QLabel("Calibration")
        section_label.setStyleSheet("font-weight: 600;")
        layout.addWidget(section_label)

        file_grid = QGridLayout()
        self.calibration_file_combo.setEnabled(False)
        file_grid.addWidget(QLabel("Existing file"), 0, 0)
        file_grid.addWidget(self.calibration_file_combo, 0, 1)
        layout.addLayout(file_grid)

        self.calibration_buttons["calibrate"] = QPushButton("Run New Calibration")
        for button in (
            *self.calibration_buttons.values(),
            self.load_calibration_button,
            self.show_calibration_plot_button,
        ):
            button.setEnabled(False)
        grid = QGridLayout()
        grid.addWidget(self.calibration_buttons["calibrate"], 0, 0)
        grid.addWidget(self.load_calibration_button, 0, 1)
        grid.addWidget(self.show_calibration_plot_button, 1, 0, 1, 2)
        layout.addLayout(grid)

    def _display_pattern(self, pattern: str) -> None:
        if not self.devices_initialised:
            self.status_label.setText("Run Initialise Devices before using DMD controls.")
            return
        self.status_label.setText(f"Displaying {self._format_pattern(pattern)}")
        self.controller.display_dmd_pattern(pattern=pattern, config=self._shape_config_payload())

    def _calibrate(self) -> None:
        if not self.devices_initialised:
            self.status_label.setText("Run Initialise Devices before using DMD controls.")
            return
        self.status_label.setText("Starting DMD calibration.")
        self.controller.calibrate_dmd()

    def _load_selected_calibration(self) -> None:
        if not self.devices_initialised:
            self.status_label.setText("Run Initialise Devices before using DMD controls.")
            return
        filename = self.calibration_file_combo.currentData()
        if not filename:
            self.status_label.setText("No calibration file selected.")
            return
        self.status_label.setText(f"Loading calibration {Path(filename).name}.")
        self.controller.load_dmd_calibration(str(filename))

    def _request_calibration_plot(self) -> None:
        if not self.devices_initialised:
            self.status_label.setText("Run Initialise Devices before using DMD controls.")
            return
        self.status_label.setText("Loading DMD calibration plot data.")
        self.controller.request_dmd_calibration_points()

    def _show_calibration_plot(self, payload: dict) -> None:
        if not payload.get("dmd_points") or not payload.get("cam_points"):
            self.status_label.setText("No DMD calibration points are loaded.")
            return
        window = DmdCalibrationPlotWindow(payload)
        self.calibration_plot_windows.append(window)
        window.destroyed.connect(
            lambda _object=None, closed=window: self._forget_calibration_plot(closed)
        )
        window.show()
        self.status_label.setText("DMD calibration plot opened.")

    def _forget_calibration_plot(self, window: "DmdCalibrationPlotWindow") -> None:
        if window in self.calibration_plot_windows:
            self.calibration_plot_windows.remove(window)

    def update_status(self, payload: dict) -> None:
        if "is_initialised" in payload:
            self.devices_initialised = bool(payload.get("is_initialised")) and bool(
                payload.get("is_alive", True)
            )
        self._update_config_limits(payload.get("camera_width_height"))
        self._update_calibration_files(
            calibration_files=payload.get("calibration_files"),
            current_file=payload.get("calibration_file"),
        )
        last_pattern = payload.get("last_pattern")
        pattern_text = f", pattern: {self._format_pattern(last_pattern)}" if last_pattern else ""
        calibration_file = self._format_calibration_file(payload.get("calibration_file"))
        self.status_label.setText(
            f"initialised: {payload.get('is_initialised')}, alive: {payload.get('is_alive')}, "
            f"calibrated: {payload.get('is_calibrated')}{pattern_text}, calibration: {calibration_file}"
        )

    def update_lifecycle_status(self, payload: dict) -> None:
        self.devices_initialised = bool(payload.get("devices_initialised"))
        if payload.get("shutdown"):
            self.devices_initialised = False
        self._sync_controls_enabled()
        if not self.devices_initialised:
            self.status_label.setText("Run Initialise Devices before using DMD controls.")
        elif self.status_label.text().startswith("Run Initialise Devices"):
            self.status_label.setText("Refresh DMD to read status.")

    def update_strategy_status(self, payload: dict) -> None:
        self.strategy_running = bool(payload.get("running"))
        self._sync_controls_enabled()

    def _sync_controls_enabled(self) -> None:
        manual_controls_enabled = self.devices_initialised and not self.strategy_running
        for button in (
            *self.pattern_buttons.values(),
            *self.calibration_buttons.values(),
        ):
            button.setEnabled(manual_controls_enabled)
        self.configure_pattern_button.setEnabled(not self.strategy_running)
        for button in (*self.utility_buttons.values(), self.show_calibration_plot_button):
            button.setEnabled(self.devices_initialised)
        has_calibration_files = self.calibration_file_combo.count() > 0
        self.calibration_file_combo.setEnabled(not self.strategy_running and has_calibration_files)
        self.load_calibration_button.setEnabled(manual_controls_enabled and has_calibration_files)

    def _show_error(self, error: str) -> None:
        text = self.status_label.text()
        if "dmd" in error.lower() or "calibrat" in error.lower() or text.startswith("Starting DMD"):
            self.status_label.setText(error)

    def _shape_config_payload(self) -> dict[str, int]:
        return dict(self.config_values)

    @staticmethod
    def _default_shape_config_values() -> dict[str, int]:
        return {
            field_name: default
            for _label, field_name, default, _minimum, _maximum in SHAPE_CONFIG_FIELDS
        }

    def _open_pattern_config_dialog(self) -> None:
        dialog = ConfigDialog(
            title="DMD Pattern Configuration",
            fields=self._shape_config_fields(),
            parent=self,
        )
        if dialog.exec_() != dialog.Accepted:
            return
        self.config_values.update(
            {key: int(value) for key, value in dialog.values().items() if key in self.config_values}
        )

    def _shape_config_fields(self) -> list[ConfigFieldSpec]:
        fields = []
        for label, field_name, default, _minimum, _maximum in SHAPE_CONFIG_FIELDS:
            minimum, maximum = self.config_limits[field_name]
            fields.append(
                ConfigFieldSpec(
                    label=label,
                    key=field_name,
                    value=self.config_values.get(field_name, default),
                    kind="int",
                    minimum=minimum,
                    maximum=maximum,
                )
            )
        return fields

    def _update_config_limits(self, width_height: list[int] | tuple[int, int] | None) -> None:
        if (
            not isinstance(width_height, list | tuple)
            or len(width_height) != 2
            or not all(isinstance(value, int) and value > 0 for value in width_height)
        ):
            return
        rows, cols = width_height
        ranges = {
            "rectangle_row": (0, rows - 1),
            "rectangle_col": (0, cols - 1),
            "rectangle_height": (1, rows),
            "rectangle_width": (1, cols),
            "checkerboard_box_size": (1, max(rows, cols)),
            "crosshair_row": (0, rows - 1),
            "crosshair_col": (0, cols - 1),
            "crosshair_width": (1, max(rows, cols)),
            "circle_row": (0, rows - 1),
            "circle_col": (0, cols - 1),
            "circle_radius": (1, max(rows, cols)),
        }
        for field_name, (minimum, maximum) in ranges.items():
            if field_name in self.config_limits:
                self.config_limits[field_name] = (minimum, maximum)
                self.config_values[field_name] = min(
                    max(self.config_values[field_name], minimum), maximum
                )
        self.config_values["rectangle_row"] = min(
            self.config_values["rectangle_row"],
            rows - self.config_values["rectangle_height"],
        )
        self.config_values["rectangle_col"] = min(
            self.config_values["rectangle_col"],
            cols - self.config_values["rectangle_width"],
        )

    def _update_calibration_files(
        self,
        calibration_files: list[dict] | None,
        current_file: str | None,
    ) -> None:
        if not isinstance(calibration_files, list):
            self._sync_controls_enabled()
            return
        previous_file = self.calibration_file_combo.currentData()
        selected_file = current_file or previous_file
        self.calibration_file_combo.clear()
        selected_index = -1
        for file_payload in calibration_files:
            if not isinstance(file_payload, dict):
                continue
            path = file_payload.get("path")
            if not path:
                continue
            label = file_payload.get("label") or Path(str(path)).name
            self.calibration_file_combo.addItem(str(label), str(path))
            if file_payload.get("is_current") or str(path) == selected_file:
                selected_index = self.calibration_file_combo.count() - 1
        if selected_index >= 0:
            self.calibration_file_combo.setCurrentIndex(selected_index)
        self._sync_controls_enabled()

    @staticmethod
    def _format_pattern(pattern: str | None) -> str:
        if pattern is None:
            return "-"
        labels = {
            action_name: label
            for label, action_name in (
                *PATTERN_ACTIONS,
                ("Run New Calibration", "calibrate"),
            )
        }
        return labels.get(pattern, pattern)

    @staticmethod
    def _format_calibration_file(calibration_file: str | None) -> str:
        if not calibration_file:
            return "-"
        return Path(calibration_file).name


class DmdCalibrationPlotWindow(QWidget):
    """Separate window showing paired DMD and camera calibration points."""

    def __init__(self, payload: dict, parent: QWidget | None = None):
        super().__init__(parent)
        calibration_file = payload.get("calibration_file")
        title = "DMD Calibration Points"
        if calibration_file:
            title = f"{title}: {Path(str(calibration_file)).name}"
        self.setWindowTitle(title)

        self.figure = Figure(figsize=(10, 5), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        self.resize(1000, 520)

        self._draw(payload)

    def _draw(self, payload: dict) -> None:
        dmd_shape = self._shape_from_payload(payload.get("dmd_shape"), default=DMD_WIDTH_HEIGHT)
        cam_shape = self._shape_from_payload(payload.get("cam_shape"), default=CAM_WIDTH_HEIGHT)
        dmd_points = self._dmd_display_points(self._points_from_payload(payload.get("dmd_points")))
        cam_points = self._points_from_payload(payload.get("cam_points"))

        dmd_image = np.ones(self._dmd_display_shape(dmd_shape), dtype=np.uint8) * 100
        cam_image = np.ones(cam_shape, dtype=np.uint8) * 100
        axes = self.figure.subplots(1, 2)
        self._draw_points_axis(axes[0], image=dmd_image, points=dmd_points, title="DMD Points")
        self._draw_points_axis(axes[1], image=cam_image, points=cam_points, title="Camera Points")
        self.canvas.draw_idle()

    @staticmethod
    def _draw_points_axis(
        axis, *, image: np.ndarray, points: list[tuple[int, int]], title: str
    ) -> None:
        axis.imshow(image, cmap="gray", vmin=0, vmax=255)
        for index, (row, col) in enumerate(points):
            axis.text(col, row, str(index), color="tab:red", fontsize=6, ha="center", va="center")
        axis.set_title(title)
        axis.set_xlabel("Column")
        axis.set_ylabel("Row")

    @staticmethod
    def _shape_from_payload(value, *, default: tuple[int, int]) -> tuple[int, int]:
        if isinstance(value, list | tuple) and len(value) == 2:
            rows, cols = value
            if isinstance(rows, int) and isinstance(cols, int) and rows > 0 and cols > 0:
                return rows, cols
        return default

    @staticmethod
    def _dmd_display_shape(dmd_shape: tuple[int, int]) -> tuple[int, int]:
        return dmd_shape[1], dmd_shape[0]

    @staticmethod
    def _dmd_display_points(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
        return [(col, row) for row, col in points]

    @staticmethod
    def _points_from_payload(value) -> list[tuple[int, int]]:
        if not isinstance(value, list):
            return []
        points: list[tuple[int, int]] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            row = item.get("row")
            col = item.get("col")
            if isinstance(row, int) and isinstance(col, int):
                points.append((row, col))
        return points
