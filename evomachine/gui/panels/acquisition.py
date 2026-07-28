from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from evomachine.gui.image_payloads import IMAGE_TRANSPORT_SOCKET_TIFF


class FrameAcquisitionSettingsPanel(QGroupBox):
    """Frame acquisition settings currently represented in the backend dataclass."""

    SETTINGS = (
        ("save", "Save", False),
        ("normalise", "Normalise", False),
        ("illuminate_dmd", "Illuminate DMD", True),
        ("clear_dmd_after", "Clear DMD After", False),
        ("restore_leds_after", "Restore LEDs After", True),
        ("disable_leds_after", "Disable LEDs After", False),
    )

    def __init__(self, parent: QWidget | None = None):
        super().__init__("Frame Acquisition Settings", parent)
        self.checkboxes: dict[str, QCheckBox] = {}

        layout = QVBoxLayout()
        for key, label, default in self.SETTINGS:
            checkbox = QCheckBox(label)
            checkbox.setChecked(default)
            self.checkboxes[key] = checkbox
            layout.addWidget(checkbox)
        self.setLayout(layout)

    def payload(self) -> dict:
        return {
            "settings": {
                key: checkbox.isChecked()
                for key, checkbox in self.checkboxes.items()
            }
        }


class ManualAcquisitionPanel(QGroupBox):
    """Manual single-frame acquisition controls."""

    def __init__(
            self,
            controller,
            settings_provider: Callable[[], dict] | None = None,
            parent: QWidget | None = None,
    ):
        super().__init__("Manual Acquisition", parent)
        self.controller = controller
        self.settings_provider = settings_provider or (lambda: {})
        self.devices_initialised = False
        self.acquire_button = QPushButton("Acquire Frame")
        self.status_label = QLabel("Run Initialise Devices before manual acquisition.")
        self.exposure_checkbox = QCheckBox("Override exposure")
        self.exposure_input = self._exposure_input()

        form = QFormLayout()
        form.addRow("Exposure ms", self.exposure_input)

        layout = QVBoxLayout()
        layout.addWidget(self.exposure_checkbox)
        layout.addLayout(form)
        layout.addWidget(self.acquire_button)
        layout.addWidget(self.status_label)
        self.setLayout(layout)

        self.acquire_button.clicked.connect(self._acquire_frame)
        self.controller.frame_received.connect(self.update_frame_status)
        self.controller.lifecycle_status_received.connect(self.update_lifecycle_status)
        self.controller.response_error.connect(self._show_error)
        self._sync_controls_enabled()

    @staticmethod
    def _exposure_input() -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setRange(1.0, 1000.0)
        box.setDecimals(1)
        box.setSingleStep(10.0)
        box.setValue(200.0)
        return box

    def _acquire_frame(self) -> None:
        if not self.devices_initialised:
            self.status_label.setText("Run Initialise Devices before manual acquisition.")
            return
        self.status_label.setText("Acquiring frame.")
        payload = self.settings_provider()
        if self.exposure_checkbox.isChecked():
            payload = {**payload, "exposure": self.exposure_input.value()}
        self.controller.acquire_frame(payload)

    def update_frame_status(self, payload: dict) -> None:
        if payload.get("kind") == "z_stack":
            return
        self.status_label.setText(
            f"Last frame: {payload.get('image_shape')}, dtype: {payload.get('dtype')}"
        )

    def update_lifecycle_status(self, payload: dict) -> None:
        self.devices_initialised = bool(payload.get("devices_initialised"))
        if payload.get("shutdown"):
            self.devices_initialised = False
        self._sync_controls_enabled()
        if not self.devices_initialised:
            self.status_label.setText("Run Initialise Devices before manual acquisition.")
        elif self.status_label.text().startswith("Run Initialise Devices"):
            self.status_label.setText("Ready to acquire a frame.")

    def _sync_controls_enabled(self) -> None:
        self.acquire_button.setEnabled(self.devices_initialised)
        self.exposure_checkbox.setEnabled(self.devices_initialised)
        self.exposure_input.setEnabled(self.devices_initialised)

    def _show_error(self, error: str) -> None:
        if self.status_label.text().startswith("Acquiring frame"):
            self.status_label.setText(error)


class ZStackSettingsPanel(QGroupBox):
    """Minimum useful z-stack acquisition controls."""

    def __init__(
            self,
            controller,
            settings_provider: Callable[[], dict] | None = None,
            parent: QWidget | None = None,
    ):
        super().__init__("Z Stack Acquisition", parent)
        self.controller = controller
        self.settings_provider = settings_provider or (lambda: {})
        self.devices_initialised = False
        self.start_input = self._float_input()
        self.end_input = self._float_input()
        self.step_input = self._float_input()
        self.step_input.setRange(0.001, 1e7)
        self.step_input.setValue(1.0)
        self.acquire_button = QPushButton("Acquire Z Stack")
        self.status_label = QLabel("Run Initialise Devices before z-stack acquisition.")

        form = QFormLayout()
        form.addRow("Start Z", self.start_input)
        form.addRow("End Z", self.end_input)
        form.addRow("Step", self.step_input)

        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addWidget(self.acquire_button)
        layout.addWidget(self.status_label)
        self.setLayout(layout)

        self.acquire_button.clicked.connect(self._acquire_z_stack)
        self.controller.frame_received.connect(self.update_frame_status)
        self.controller.lifecycle_status_received.connect(self.update_lifecycle_status)
        self.controller.response_error.connect(self._show_error)
        self._sync_controls_enabled()

    @staticmethod
    def _float_input() -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setRange(-1e7, 1e7)
        box.setDecimals(3)
        box.setSingleStep(1.0)
        return box

    def _acquire_z_stack(self) -> None:
        if not self.devices_initialised:
            self.status_label.setText("Run Initialise Devices before z-stack acquisition.")
            return
        payload = {
            **self.settings_provider(),
            "start_z": self.start_input.value(),
            "end_z": self.end_input.value(),
            "step_z": self.step_input.value(),
        }
        self.status_label.setText("Acquiring z-stack.")
        self.controller.acquire_z_stack(payload)

    def update_frame_status(self, payload: dict) -> None:
        if payload.get("kind") != "z_stack":
            return
        self.status_label.setText(
            f"Last z-stack: {payload.get('planes')} planes, shape: {payload.get('stack_shape')}"
        )

    def update_lifecycle_status(self, payload: dict) -> None:
        self.devices_initialised = bool(payload.get("devices_initialised"))
        if payload.get("shutdown"):
            self.devices_initialised = False
        self._sync_controls_enabled()
        if not self.devices_initialised:
            self.status_label.setText("Run Initialise Devices before z-stack acquisition.")
        elif self.status_label.text().startswith("Run Initialise Devices"):
            self.status_label.setText("Ready to acquire a z-stack.")

    def _sync_controls_enabled(self) -> None:
        for widget in (self.start_input, self.end_input, self.step_input, self.acquire_button):
            widget.setEnabled(self.devices_initialised)

    def _show_error(self, error: str) -> None:
        if self.status_label.text().startswith("Acquiring z-stack"):
            self.status_label.setText(error)


class SavedImageLoaderPanel(QGroupBox):
    """Load previously saved acquisition TIFFs into the central viewer."""

    def __init__(self, controller, parent: QWidget | None = None):
        super().__init__("Load Saved Image", parent)
        self.controller = controller
        self.file_combo = QComboBox()
        self.force_socket_checkbox = QCheckBox("Force socket transport")
        self.refresh_button = QPushButton("Refresh Files")
        self.load_button = QPushButton("Load Selected")
        self.path_label = QLabel("path: -")
        self.status_label = QLabel("Refresh files to load a saved TIFF.")
        self.path_label.setWordWrap(True)

        button_grid = QGridLayout()
        button_grid.addWidget(self.refresh_button, 0, 0)
        button_grid.addWidget(self.load_button, 0, 1)

        layout = QVBoxLayout()
        layout.addWidget(self.force_socket_checkbox)
        layout.addWidget(self.file_combo)
        layout.addWidget(self.path_label)
        layout.addLayout(button_grid)
        layout.addWidget(self.status_label)
        self.setLayout(layout)

        self.refresh_button.clicked.connect(self._refresh_files)
        self.load_button.clicked.connect(self._load_selected)
        self.file_combo.currentIndexChanged.connect(self._update_selected_path_label)
        self.controller.acquisition_files_received.connect(self.update_file_list)
        self.controller.frame_received.connect(self.update_frame_status)
        self.controller.response_error.connect(self._show_error)
        self._sync_controls_enabled()

    def _refresh_files(self) -> None:
        self.status_label.setText("Refreshing files.")
        self.controller.refresh_acquisition_files()

    def _load_selected(self) -> None:
        filename = self._selected_filename()
        if filename is None:
            self.status_label.setText("No saved TIFF selected.")
            return
        image_transport = IMAGE_TRANSPORT_SOCKET_TIFF if self.force_socket_checkbox.isChecked() else None
        self.status_label.setText("Loading through socket." if image_transport else "Loading selected file.")
        self.controller.load_acquisition_frame(filename, image_transport=image_transport)

    def _selected_filename(self) -> str | None:
        filename = self.file_combo.currentData()
        return filename if isinstance(filename, str) and filename else None

    def update_file_list(self, files: list) -> None:
        selected = self._selected_filename()
        self.file_combo.clear()
        for file_payload in files:
            if isinstance(file_payload, dict):
                path = file_payload.get("path")
                label = file_payload.get("label") or (Path(str(path)).name if path else None)
            else:
                path = str(file_payload)
                label = Path(path).name
            if not path or not label:
                continue
            self.file_combo.addItem(str(label), str(path))
        if selected is not None:
            index = self.file_combo.findData(selected)
            if index >= 0:
                self.file_combo.setCurrentIndex(index)
        count = self.file_combo.count()
        self.status_label.setText(f"{count} saved TIFF file(s) available." if count else "No saved TIFF files found.")
        self._update_selected_path_label()
        self._sync_controls_enabled()

    def update_frame_status(self, payload: dict) -> None:
        if payload.get("source") != "file":
            return
        path = payload.get("loaded_path") or "-"
        self.status_label.setText(f"Loaded {Path(str(path)).name}: {payload.get('image_shape')}")

    def _sync_controls_enabled(self) -> None:
        has_file = self.file_combo.count() > 0
        self.load_button.setEnabled(has_file)

    def _update_selected_path_label(self, _index: int | None = None) -> None:
        filename = self._selected_filename()
        self.path_label.setText(f"path: {filename}" if filename is not None else "path: -")

    def _show_error(self, error: str) -> None:
        if self.status_label.text().startswith(("Refreshing files", "Loading")):
            self.status_label.setText(error)


class AcquisitionStatusPanel(QGroupBox):
    """Small readout for the latest acquisition result."""

    def __init__(self, controller, parent: QWidget | None = None):
        super().__init__("Acquisition Status", parent)
        self.controller = controller
        self.ready_label = QLabel("not ready")
        self.last_label = QLabel("last acquisition: -")
        self.saved_label = QLabel("saved: -")

        layout = QGridLayout()
        layout.addWidget(QLabel("Ready"), 0, 0)
        layout.addWidget(self.ready_label, 0, 1)
        layout.addWidget(QLabel("Last"), 1, 0)
        layout.addWidget(self.last_label, 1, 1)
        layout.addWidget(QLabel("Saved"), 2, 0)
        layout.addWidget(self.saved_label, 2, 1)
        self.setLayout(layout)

        self.controller.frame_received.connect(self.update_frame_status)
        self.controller.lifecycle_status_received.connect(self.update_lifecycle_status)
        self.controller.response_error.connect(self._show_error)

    def update_frame_status(self, payload: dict) -> None:
        kind = "z-stack" if payload.get("kind") == "z_stack" else "frame"
        self.last_label.setText(
            f"{kind}: {payload.get('planes')} plane(s), image {payload.get('image_shape')}"
        )
        saved_paths = [path for path in payload.get("saved_paths", []) if path]
        if not saved_paths:
            self.saved_label.setText("-")
            self.saved_label.setToolTip("")
            return
        latest_name = saved_paths[-1].split("/")[-1]
        if len(saved_paths) == 1:
            self.saved_label.setText(f"1 file: {latest_name}")
        else:
            self.saved_label.setText(f"{len(saved_paths)} files; latest: {latest_name}")
        self.saved_label.setToolTip("\n".join(saved_paths))

    def update_lifecycle_status(self, payload: dict) -> None:
        ready = bool(payload.get("devices_initialised")) and not bool(payload.get("shutdown"))
        self.ready_label.setText("ready" if ready else "not ready")

    def _show_error(self, error: str) -> None:
        self.last_label.setText(error)
