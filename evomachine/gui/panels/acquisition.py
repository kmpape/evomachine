from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from PyQt5.QtWidgets import (
    QComboBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from evomachine.gui.image_payloads import IMAGE_TRANSPORT_SOCKET_TIFF
from evomachine.gui.panels.config_dialog import ConfigDialog, ConfigFieldSpec
from evomachine.gui.panels.leds import LED_GROUPS, LED_LABELS
from evomachine.types import FilterWheelType, LEDType


USE_CURRENT_MAIN_CONTROLS_KEY = "use_current_main_controls"
DMD_PATTERN_CHOICES = (
    "full",
    "empty",
    "rectangle",
    "circle",
    "checkerboard",
    "crosshair",
)
FILTER_WHEEL_CHOICES = tuple(
    filter_type.name
    for filter_type in FilterWheelType
    if filter_type is not FilterWheelType.UNKNOWN
)
ACQUISITION_LED_TYPES = tuple(
    led_type
    for _group_name, led_types in LED_GROUPS
    for led_type in led_types
)


class FrameAcquisitionSettingsPanel(QGroupBox):
    """Config values used by manual frame and z-stack acquisition panels."""

    SETTINGS = (
        ("save", "Save", False),
        ("normalise", "Normalise", False),
        ("illuminate_dmd", "Illuminate DMD", False),
        ("clear_dmd_after", "Clear DMD After", False),
        ("restore_leds_after", "Restore LEDs After", True),
        ("disable_leds_after", "Disable LEDs After", False),
    )

    def __init__(self, parent: QWidget | None = None):
        super().__init__("Acquisition Configuration", parent)
        self.config_values = self._default_config_values()
        self.configure_button = QPushButton("Configure")
        self.summary_label = QLabel()
        self.summary_label.setWordWrap(True)

        layout = QVBoxLayout()
        layout.addWidget(self.summary_label)
        layout.addWidget(self.configure_button)
        self.setLayout(layout)

        self.configure_button.clicked.connect(self._open_config_dialog)
        self._update_summary()

    def payload(self) -> dict:
        settings = {
            key: bool(self.config_values[key])
            for key, _label, _default in self.SETTINGS
        }
        use_current_main_controls = bool(self.config_values[USE_CURRENT_MAIN_CONTROLS_KEY])
        if use_current_main_controls:
            settings["illuminate_dmd"] = False

        payload: dict[str, Any] = {
            "settings": settings,
            USE_CURRENT_MAIN_CONTROLS_KEY: use_current_main_controls,
        }
        if not use_current_main_controls:
            payload["exposure"] = float(self.config_values["exposure"])
            payload.update(self._explicit_peripheral_payload())
        return payload

    def z_stack_payload(self) -> dict:
        payload = self.payload()
        payload.update({
            "start_z": float(self.config_values["start_z"]),
            "end_z": float(self.config_values["end_z"]),
            "step_z": float(self.config_values["step_z"]),
        })
        return payload

    def _open_config_dialog(self) -> None:
        dialog = ConfigDialog(
            title="Acquisition Configuration",
            fields=self._config_fields(),
            parent=self,
        )
        if dialog.exec_() != dialog.Accepted:
            return
        self.config_values.update(dialog.values())
        self._update_summary()

    def _config_fields(self) -> list[ConfigFieldSpec]:
        fields = [
            ConfigFieldSpec(
                "Use current main controls",
                USE_CURRENT_MAIN_CONTROLS_KEY,
                self.config_values[USE_CURRENT_MAIN_CONTROLS_KEY],
                kind="bool",
            ),
            ConfigFieldSpec(
                "Exposure ms",
                "exposure",
                self.config_values["exposure"],
                kind="float",
                minimum=1.0,
                maximum=1000.0,
                decimals=1,
                single_step=10.0,
                enabled_when_key=USE_CURRENT_MAIN_CONTROLS_KEY,
                enabled_when_value=False,
            ),
            ConfigFieldSpec(
                "Start ΔZ (µm)",
                "start_z",
                self.config_values["start_z"],
                kind="float",
                minimum=-1e7,
                maximum=1e7,
                decimals=3,
                single_step=1.0,
            ),
            ConfigFieldSpec(
                "End ΔZ (µm)",
                "end_z",
                self.config_values["end_z"],
                kind="float",
                minimum=-1e7,
                maximum=1e7,
                decimals=3,
                single_step=1.0,
            ),
            ConfigFieldSpec(
                "ΔZ step (µm)",
                "step_z",
                self.config_values["step_z"],
                kind="float",
                minimum=0.001,
                maximum=1e7,
                decimals=3,
                single_step=1.0,
            ),
        ]
        for key, label, _default in self.SETTINGS:
            kwargs = (
                {
                    "enabled_when_key": USE_CURRENT_MAIN_CONTROLS_KEY,
                    "enabled_when_value": False,
                }
                if key == "illuminate_dmd"
                else {}
            )
            fields.append(ConfigFieldSpec(label, key, self.config_values[key], kind="bool", **kwargs))
        fields.extend([
            ConfigFieldSpec(
                "Filter wheel",
                "filter_wheel",
                self.config_values["filter_wheel"],
                kind="choice",
                choices=FILTER_WHEEL_CHOICES,
                enabled_when_key=USE_CURRENT_MAIN_CONTROLS_KEY,
                enabled_when_value=False,
            ),
            ConfigFieldSpec(
                "DMD pattern",
                "dmd_pattern",
                self.config_values["dmd_pattern"],
                kind="choice",
                choices=DMD_PATTERN_CHOICES,
                enabled_when_key=USE_CURRENT_MAIN_CONTROLS_KEY,
                enabled_when_value=False,
            ),
        ])
        for led_type in ACQUISITION_LED_TYPES:
            fields.extend([
                ConfigFieldSpec(
                    f"Use {LED_LABELS.get(led_type, led_type.name)}",
                    self._led_enabled_key(led_type),
                    self.config_values[self._led_enabled_key(led_type)],
                    kind="bool",
                    enabled_when_key=USE_CURRENT_MAIN_CONTROLS_KEY,
                    enabled_when_value=False,
                ),
                ConfigFieldSpec(
                    f"{LED_LABELS.get(led_type, led_type.name)} brightness",
                    self._led_brightness_key(led_type),
                    self.config_values[self._led_brightness_key(led_type)],
                    kind="float",
                    minimum=0.0,
                    maximum=100.0,
                    decimals=0,
                    single_step=1.0,
                    enabled_when_key=USE_CURRENT_MAIN_CONTROLS_KEY,
                    enabled_when_value=False,
                ),
            ])
        return fields

    @classmethod
    def _default_config_values(cls) -> dict[str, Any]:
        values: dict[str, Any] = {
            USE_CURRENT_MAIN_CONTROLS_KEY: True,
            "exposure": 200.0,
            "start_z": -10.0,
            "end_z": 10.0,
            "step_z": 1.0,
            "filter_wheel": FilterWheelType.NO_FILTER.name,
            "dmd_pattern": "full",
        }
        values.update({key: default for key, _label, default in cls.SETTINGS})
        for led_type in ACQUISITION_LED_TYPES:
            values[cls._led_enabled_key(led_type)] = False
            values[cls._led_brightness_key(led_type)] = 29.0
        return values

    def _explicit_peripheral_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        leds = {
            led_type.name: float(self.config_values[self._led_brightness_key(led_type)])
            for led_type in ACQUISITION_LED_TYPES
            if bool(self.config_values[self._led_enabled_key(led_type)])
        }
        if leds:
            payload["leds"] = leds
        filter_wheel = str(self.config_values["filter_wheel"])
        payload["filter_wheel"] = filter_wheel
        dmd_pattern = str(self.config_values["dmd_pattern"])
        if bool(self.config_values["illuminate_dmd"]):
            payload["dmd_pattern"] = dmd_pattern
        return payload

    def _update_summary(self) -> None:
        use_current_main_controls = bool(self.config_values[USE_CURRENT_MAIN_CONTROLS_KEY])
        exposure = "main controls" if use_current_main_controls else f"{float(self.config_values['exposure']):.1f} ms"
        dmd_pattern = (
            "main controls"
            if use_current_main_controls
            else str(self.config_values["dmd_pattern"])
            if bool(self.config_values["illuminate_dmd"])
            else "off"
        )
        mode = "current main controls" if use_current_main_controls else "explicit acquisition config"
        settings = ", ".join(
            label
            for key, label, _default in self.SETTINGS
            if bool(self.config_values[key])
        )
        self.summary_label.setText(
            f"mode: {mode}; exposure: {exposure}; "
            f"DMD: {dmd_pattern}; "
            f"Δz: {float(self.config_values['start_z']):.3f} -> "
            f"{float(self.config_values['end_z']):.3f} by "
            f"{float(self.config_values['step_z']):.3f}; "
            f"settings: {settings or '-'}"
        )

    @staticmethod
    def _led_enabled_key(led_type: LEDType) -> str:
        return f"led_{led_type.name}_enabled"

    @staticmethod
    def _led_brightness_key(led_type: LEDType) -> str:
        return f"led_{led_type.name}_brightness"


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

        layout = QVBoxLayout()
        layout.addWidget(self.acquire_button)
        layout.addWidget(self.status_label)
        self.setLayout(layout)

        self.acquire_button.clicked.connect(self._acquire_frame)
        self.controller.frame_received.connect(self.update_frame_status)
        self.controller.lifecycle_status_received.connect(self.update_lifecycle_status)
        self.controller.response_error.connect(self._show_error)
        self._sync_controls_enabled()

    def _acquire_frame(self) -> None:
        if not self.devices_initialised:
            self.status_label.setText("Run Initialise Devices before manual acquisition.")
            return
        self.status_label.setText("Acquiring frame.")
        payload = self.settings_provider()
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
        self.acquire_button = QPushButton("Acquire Z Stack")
        self.status_label = QLabel("Run Initialise Devices before z-stack acquisition.")

        layout = QVBoxLayout()
        layout.addWidget(self.acquire_button)
        layout.addWidget(self.status_label)
        self.setLayout(layout)

        self.acquire_button.clicked.connect(self._acquire_z_stack)
        self.controller.frame_received.connect(self.update_frame_status)
        self.controller.lifecycle_status_received.connect(self.update_lifecycle_status)
        self.controller.response_error.connect(self._show_error)
        self._sync_controls_enabled()

    def _acquire_z_stack(self) -> None:
        if not self.devices_initialised:
            self.status_label.setText("Run Initialise Devices before z-stack acquisition.")
            return
        payload = self.settings_provider()
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
        self.acquire_button.setEnabled(self.devices_initialised)

    def _show_error(self, error: str) -> None:
        if self.status_label.text().startswith("Acquiring z-stack"):
            self.status_label.setText(error)


class SavedImageLoaderPanel(QGroupBox):
    """Select experiment folders and load their acquisition TIFFs."""

    def __init__(self, controller, parent: QWidget | None = None):
        super().__init__("Experiment Files", parent)
        self.controller = controller
        self.experiment_combo = QComboBox()
        self.file_combo = QComboBox()
        self.experiment_name_input = QLineEdit()
        self.experiment_name_input.setPlaceholderText("Experiment name")
        self.directory = ""
        self.force_socket_transport = False
        self.refresh_experiments_button = QPushButton("Refresh Experiments")
        self.refresh_button = QPushButton("Refresh Images")
        self.create_experiment_button = QPushButton("Create New Experiment")
        self.configure_button = QPushButton("Configure")
        self.load_button = QPushButton("Load Selected")
        self.path_label = QLabel("path: -")
        self.experiment_label = QLabel("active experiment: -")
        self.status_label = QLabel("Refresh files to load a saved TIFF.")
        self.transport_label = QLabel()
        self.path_label.setWordWrap(True)

        button_grid = QGridLayout()
        button_grid.addWidget(self.refresh_experiments_button, 0, 0)
        button_grid.addWidget(self.refresh_button, 0, 1)
        button_grid.addWidget(self.configure_button, 1, 0)
        button_grid.addWidget(self.create_experiment_button, 1, 1)
        button_grid.addWidget(self.load_button, 2, 0, 1, 2)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Experiment"))
        layout.addWidget(self.experiment_combo)
        layout.addWidget(self.experiment_name_input)
        layout.addWidget(self.experiment_label)
        layout.addWidget(QLabel("Image"))
        layout.addWidget(self.file_combo)
        layout.addWidget(self.path_label)
        layout.addWidget(self.transport_label)
        layout.addLayout(button_grid)
        layout.addWidget(self.status_label)
        self.setLayout(layout)

        self.refresh_experiments_button.clicked.connect(self._refresh_experiments)
        self.refresh_button.clicked.connect(self._refresh_files)
        self.create_experiment_button.clicked.connect(self._create_experiment)
        self.configure_button.clicked.connect(self._open_config_dialog)
        self.load_button.clicked.connect(self._load_selected)
        self.experiment_combo.currentIndexChanged.connect(self._select_experiment)
        self.file_combo.currentIndexChanged.connect(self._update_selected_path_label)
        self.controller.acquisition_files_received.connect(self.update_file_list)
        self.controller.acquisition_directory_received.connect(self.update_acquisition_directory)
        self.controller.acquisition_experiments_received.connect(self.update_experiment_list)
        self.controller.frame_received.connect(self.update_frame_status)
        self.controller.response_error.connect(self._show_error)
        self._update_transport_label()
        self._sync_controls_enabled()
        self._refresh_experiments()

    def _refresh_files(self) -> None:
        self.status_label.setText("Refreshing images.")
        self.controller.refresh_acquisition_files()

    def _refresh_experiments(self) -> None:
        self.status_label.setText("Refreshing experiments.")
        self.controller.refresh_acquisition_experiments()

    def _select_experiment(self, _index: int | None = None) -> None:
        name = self.experiment_combo.currentData()
        if not isinstance(name, str) or not name:
            return
        self.status_label.setText(f"Selecting experiment {name}.")
        self.controller.select_acquisition_experiment(name)

    def _load_selected(self) -> None:
        filename = self._selected_filename()
        if filename is None:
            self.status_label.setText("No saved TIFF selected.")
            return
        image_transport = IMAGE_TRANSPORT_SOCKET_TIFF if self.force_socket_transport else None
        self.status_label.setText(
            "Loading through socket." if image_transport else "Loading selected file."
        )
        self.controller.load_acquisition_frame(filename, image_transport=image_transport)

    def _create_experiment(self) -> None:
        name = self.experiment_name_input.text().strip()
        if not name:
            self.status_label.setText("Enter an experiment name.")
            return
        self.status_label.setText(f"Creating experiment {name}.")
        self.controller.create_acquisition_experiment(name)

    def _open_config_dialog(self) -> None:
        dialog = ConfigDialog(
            title="Saved Image Loader Configuration",
            fields=[
                ConfigFieldSpec(
                    "Force socket transport",
                    "force_socket_transport",
                    self.force_socket_transport,
                    kind="bool",
                )
            ],
            parent=self,
        )
        if dialog.exec_() != dialog.Accepted:
            return
        values = dialog.values()
        self.force_socket_transport = bool(values["force_socket_transport"])
        self._update_transport_label()
        self._update_selected_path_label()

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
        self.status_label.setText(
            f"{count} saved TIFF file(s) available." if count else "No saved TIFF files found."
        )
        self._update_selected_path_label()
        self._sync_controls_enabled()

    def update_experiment_list(self, payload: dict) -> None:
        active_experiment = payload.get("active_experiment")
        was_blocked = self.experiment_combo.blockSignals(True)
        self.experiment_combo.clear()
        for experiment in payload.get("experiments", []):
            if not isinstance(experiment, dict):
                continue
            name = experiment.get("name")
            if isinstance(name, str) and name:
                self.experiment_combo.addItem(name, name)
        if isinstance(active_experiment, str):
            index = self.experiment_combo.findData(active_experiment)
            if index >= 0:
                self.experiment_combo.setCurrentIndex(index)
        self.experiment_combo.blockSignals(was_blocked)
        self._sync_controls_enabled()

    def update_frame_status(self, payload: dict) -> None:
        if payload.get("source") != "file":
            return
        path = payload.get("loaded_path") or "-"
        self.status_label.setText(f"Loaded {Path(str(path)).name}: {payload.get('image_shape')}")

    def update_acquisition_directory(self, payload: dict) -> None:
        directory = str(payload.get("directory") or "")
        if not directory:
            return
        self.directory = directory
        experiment_name = payload.get("experiment_name") or Path(directory).name
        self.experiment_label.setText(f"active experiment: {experiment_name}")
        index = self.experiment_combo.findData(experiment_name)
        if index >= 0:
            was_blocked = self.experiment_combo.blockSignals(True)
            self.experiment_combo.setCurrentIndex(index)
            self.experiment_combo.blockSignals(was_blocked)
        self.experiment_name_input.clear()
        self.status_label.setText(f"Saving acquisitions to {directory}.")
        self._update_selected_path_label()

    def _sync_controls_enabled(self) -> None:
        has_file = self.file_combo.count() > 0
        self.load_button.setEnabled(has_file)
        self.experiment_combo.setEnabled(self.experiment_combo.count() > 0)
        self.refresh_experiments_button.setEnabled(True)
        self.create_experiment_button.setEnabled(True)
        self.configure_button.setEnabled(True)

    def _update_selected_path_label(self, _index: int | None = None) -> None:
        filename = self._selected_filename()
        directory = self.directory.strip()
        if filename is not None:
            self.path_label.setText(f"path: {filename}")
        elif directory:
            self.path_label.setText(f"folder: {directory}")
        else:
            self.path_label.setText("path: -")

    def _update_transport_label(self) -> None:
        transport = "socket" if self.force_socket_transport else "auto"
        self.transport_label.setText(f"transport: {transport}")

    def _show_error(self, error: str) -> None:
        if self.status_label.text().startswith(
            ("Creating experiment", "Selecting experiment", "Refreshing", "Loading")
        ):
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
