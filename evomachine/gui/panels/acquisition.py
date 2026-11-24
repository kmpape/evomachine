from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QCheckBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from evomachine.gui.image_payloads import IMAGE_TRANSPORT_SOCKET_TIFF
from evomachine.filemanager import FileNameConfig, FileNameField
from evomachine.gui.panels.common import NoWheelComboBox
from evomachine.gui.panels.config_dialog import ConfigDialog, ConfigFieldSpec, ConfigForm
from evomachine.gui.panels.leds import LED_GROUPS, LED_LABELS
from evomachine.gui.protocol import GuiCommandType
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

FILENAME_FIELD_HELP = (
    (FileNameField.CHANNEL, "configured LED channel name"),
    (FileNameField.FOV_ID, "field-of-view ID, blank when unknown"),
    (FileNameField.X, "stage X coordinate, blank when unavailable"),
    (FileNameField.Y, "stage Y coordinate, blank when unavailable"),
    (FileNameField.Z, "stage Z coordinate, or 'auto' when unavailable"),
    (FileNameField.FILTER_WHEEL, "filter-wheel value, or 'None' when unset"),
    (FileNameField.TIMESTAMP, "acquisition timestamp formatted for filenames"),
    (FileNameField.TIMEZONE, "acquisition timezone identifier"),
    (FileNameField.FRAME_ID, "image-plane identifier"),
    (FileNameField.CALLBACK_ID, "strategy callback identifier, blank when unset"),
    (FileNameField.FILE_ID, "unique zero-padded file number, increasing per output folder"),
    (FileNameField.LIGHT, "condensed LED wavelength, for example 450NM, or NOLED"),
    (FileNameField.DATETIME, "compact acquisition date and time, for example 261108_155754"),
    (FileNameField.Z_RANGE, "z-stack offsets as min,step,max around the centre Z; the Z position for single frames"),
)


class AcquisitionConfigurationDialog(QDialog):
    """Edit manual acquisition settings and an optional virtual image source."""

    def __init__(
            self,
            fields: list[ConfigFieldSpec],
            controller: Any | None,
            virtual_source: dict[str, Any] | None,
            parent: QWidget | None = None,
    ):
        """
        Initialise a tabbed acquisition configuration dialog.

        Parameters
        ----------
        fields
            Acquisition setting fields displayed on the first tab.
        controller
            GUI controller used to submit a virtual-source update.
        virtual_source
            Latest authoritative virtual-source payload, or None when unsupported.
        parent
            Optional parent widget.

        Returns
        -------
        None
        """
        super().__init__(parent)
        self.setWindowTitle("Acquisition Configuration")
        self.controller = controller
        self._virtual_source = None if virtual_source is None else dict(virtual_source)
        self._source_update_pending = False
        self._form = ConfigForm(fields=fields)
        self._widgets = self._form._widgets
        self.tabs = QTabWidget()
        self.tabs.addTab(self._form, "Acquisition Settings")

        self.source_mode: NoWheelComboBox | None = None
        self.source_directory: QLineEdit | None = None
        self.source_glob: QLineEdit | None = None
        self.source_status_label: QLabel | None = None
        self.source_browse_button: QPushButton | None = None
        if virtual_source is not None and bool(virtual_source.get("supported")):
            self.tabs.addTab(self._build_virtual_source_tab(virtual_source), "Virtual Image Source")

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self._accept_configuration)
        self.button_box.rejected.connect(self.reject)
        if controller is not None:
            controller.virtual_image_source_received.connect(self._receive_virtual_source)
            controller.request_error.connect(self._receive_request_error)

        layout = QVBoxLayout(self)
        layout.addWidget(self.tabs)
        layout.addWidget(self.button_box)
        self.setMinimumWidth(460)

    def values(self) -> dict[str, Any]:
        """
        Return the values from the acquisition settings tab.

        Returns
        -------
        dict[str, Any]
            Current acquisition setting values keyed by field name.
        """
        return self._form.values()

    def _build_virtual_source_tab(self, source: dict[str, Any]) -> QWidget:
        """
        Build controls for the virtual image-source tab.

        Parameters
        ----------
        source
            Authoritative virtual-source payload used for initial values.

        Returns
        -------
        QWidget
            Populated virtual image-source tab.
        """
        widget = QWidget()
        self.source_mode = NoWheelComboBox()
        self.source_mode.addItem("Random images", "random")
        self.source_mode.addItem("Image folder", "folder")
        mode_index = self.source_mode.findData(source.get("mode", "random"))
        self.source_mode.setCurrentIndex(max(0, mode_index))
        self.source_directory = QLineEdit(source.get("directory") or "")
        self.source_glob = QLineEdit(str(source.get("filename_glob", "*.tiff")))
        self.source_browse_button = QPushButton("Choose Folder")
        self.source_status_label = QLabel()
        self.source_status_label.setWordWrap(True)
        help_label = QLabel(
            "Folder mode uses naturally sorted 3200 × 3200 uint16 TIFF images in a "
            "fixed cyclic sequence. The relative glob selects files, for example *.tiff."
        )
        help_label.setWordWrap(True)

        directory_row = QHBoxLayout()
        directory_row.addWidget(self.source_directory, stretch=1)
        directory_row.addWidget(self.source_browse_button)
        layout = QVBoxLayout(widget)
        layout.addWidget(QLabel("Source mode"))
        layout.addWidget(self.source_mode)
        layout.addWidget(QLabel("Image folder"))
        layout.addLayout(directory_row)
        layout.addWidget(QLabel("File glob"))
        layout.addWidget(self.source_glob)
        layout.addWidget(help_label)
        layout.addWidget(self.source_status_label)
        layout.addStretch(1)

        self.source_mode.currentIndexChanged.connect(self._sync_source_controls)
        self.source_browse_button.clicked.connect(self._choose_source_directory)
        self._sync_source_controls()
        self._update_source_status(source)
        return widget

    def _sync_source_controls(self) -> None:
        """
        Enable folder controls only while folder mode is selected.

        Returns
        -------
        None
        """
        if self.source_mode is None:
            return
        enabled = self.source_mode.currentData() == "folder"
        for control in (self.source_directory, self.source_glob, self.source_browse_button):
            if control is not None:
                control.setEnabled(enabled)

    def _choose_source_directory(self) -> None:
        """
        Select an existing folder for virtual camera images.

        Returns
        -------
        None
        """
        if self.source_directory is None:
            return
        selected = QFileDialog.getExistingDirectory(
            self,
            "Choose Virtual Image Folder",
            self.source_directory.text() or str(Path.home()),
        )
        if selected:
            self.source_directory.setText(selected)

    def _source_values(self) -> tuple[str, str | None, str]:
        """
        Return values from the virtual-source controls.

        Returns
        -------
        tuple[str, str | None, str]
            Mode, optional folder, and relative filename glob.
        """
        if self.source_mode is None or self.source_directory is None or self.source_glob is None:
            return "random", None, "*.tiff"
        mode = str(self.source_mode.currentData())
        directory = self.source_directory.text().strip() or None
        return mode, directory if mode == "folder" else None, self.source_glob.text().strip()

    def _source_has_changed(self) -> bool:
        """
        Return whether virtual-source controls differ from server state.

        Returns
        -------
        bool
            True when submitting the dialog requires a server update.
        """
        if self._virtual_source is None or self.source_mode is None:
            return False
        mode, directory, filename_glob = self._source_values()
        current_directory = self._virtual_source.get("directory")
        if mode == "random":
            current_directory = None
        return (
            mode != self._virtual_source.get("mode")
            or directory != current_directory
            or filename_glob != self._virtual_source.get("filename_glob")
        )

    def _accept_configuration(self) -> None:
        """
        Apply a changed source before accepting the complete dialog.

        Returns
        -------
        None
        """
        if not self._source_has_changed():
            self.accept()
            return
        if self.controller is None or self.source_status_label is None:
            self.accept()
            return
        mode, directory, filename_glob = self._source_values()
        if not filename_glob:
            self.source_status_label.setText("File glob must not be empty.")
            return
        if mode == "folder" and directory is None:
            self.source_status_label.setText("Choose an image folder before applying folder mode.")
            return
        self._source_update_pending = True
        self.button_box.setEnabled(False)
        self.source_status_label.setText("Validating virtual image source...")
        self.controller.set_virtual_image_source(mode, directory, filename_glob)

    def _receive_virtual_source(self, source: dict[str, Any]) -> None:
        """
        Accept the dialog after a pending server update succeeds.

        Parameters
        ----------
        source
            Authoritative source state returned by the server.

        Returns
        -------
        None
        """
        if not self._source_update_pending:
            self._virtual_source = dict(source)
            self._update_source_status(source)
            return
        self._virtual_source = dict(source)
        self._source_update_pending = False
        self.button_box.setEnabled(True)
        self._update_source_status(source)
        self.accept()

    def _receive_request_error(self, command: object, error: str) -> None:
        """
        Display a rejected source update without closing the dialog.

        Parameters
        ----------
        command
            GUI command associated with the failed request.
        error
            User-facing validation or request error.

        Returns
        -------
        None
        """
        if command is not GuiCommandType.VIRTUAL_IMAGE_SOURCE_SET or not self._source_update_pending:
            return
        self._source_update_pending = False
        self.button_box.setEnabled(True)
        if self.source_status_label is not None:
            self.source_status_label.setText(f"Source error: {error}")

    def _update_source_status(self, source: dict[str, Any]) -> None:
        """
        Render usable, skipped, and next-file source information.

        Parameters
        ----------
        source
            Virtual-source payload containing counts and file information.

        Returns
        -------
        None
        """
        if self.source_status_label is None:
            return
        skipped = source.get("skipped_files", [])
        skipped_text = "; ".join(
            f"{Path(item['filename']).name}: {item['reason']}"
            for item in skipped[:3]
        )
        next_filename = source.get("next_filename")
        summary = (
            f"usable: {source.get('usable_count', 0)}; skipped: {source.get('skipped_count', 0)}; "
            f"next: {Path(next_filename).name if next_filename else '-'}"
        )
        if skipped_text:
            summary += f"\nSkipped: {skipped_text}"
        self.source_status_label.setText(summary)


@dataclass
class ManualAcquisitionSaveConfig:
    """Share the manual save choice between acquisition and output panels.

    Args:
        enabled: Whether workflow manual frames and Z-stacks should be saved.
    """

    enabled: bool = False


class FrameAcquisitionSettingsPanel(QGroupBox):
    """Configure general and device-specific manual acquisition values."""

    SETTINGS = (
        ("normalise", "Normalise", False),
        ("illuminate_dmd", "Illuminate DMD", False),
        ("clear_dmd_after", "Clear DMD After", False),
        ("restore_leds_after", "Restore LEDs After", True),
        ("disable_leds_after", "Disable LEDs After", False),
    )

    def __init__(
        self,
        controller: Any | None = None,
        parent: QWidget | None = None,
        save_config: ManualAcquisitionSaveConfig | None = None,
    ):
        """Create acquisition configuration controls.

        Args:
            controller: Optional GUI controller used for virtual-source state.
            parent: Optional owning Qt widget.
            save_config: Shared manual-save choice used in emitted payloads.
        """
        super().__init__("Acquisition Configuration", parent)
        self.controller = controller
        self.save_config = (
            save_config if save_config is not None else ManualAcquisitionSaveConfig()
        )
        self.virtual_source: dict[str, Any] | None = None
        self.config_values = self._default_config_values()
        self.use_current_device_settings = QCheckBox("Use current device settings")
        self.use_current_device_settings.setChecked(True)
        self.use_current_device_settings.setToolTip(
            "Use the exposure, LEDs, filter wheel, and DMD state already selected in Device "
            "Controls. General image processing and cleanup options still apply."
        )
        self.configure_button = QPushButton("Configure")
        self.configure_button.setToolTip(
            "Configure exposure, normalisation, illumination cleanup, and virtual images."
        )
        self.leds_button = QPushButton("LEDs")
        self.leds_button.setToolTip("Choose acquisition LED channels and brightness values.")
        self.dmd_button = QPushButton("DMD")
        self.dmd_button.setToolTip("Choose the DMD pattern used by explicit acquisitions.")
        self.filter_wheel_button = QPushButton("Filter Wheel")
        self.filter_wheel_button.setToolTip("Choose the acquisition filter-wheel position.")
        self.summary_label = QLabel()
        self.summary_label.setWordWrap(True)

        device_buttons = QGridLayout()
        device_buttons.addWidget(self.leds_button, 0, 0)
        device_buttons.addWidget(self.dmd_button, 0, 1)
        device_buttons.addWidget(self.filter_wheel_button, 1, 0, 1, 2)
        layout = QVBoxLayout()
        layout.addWidget(self.use_current_device_settings)
        layout.addWidget(self.summary_label)
        layout.addWidget(self.configure_button)
        layout.addLayout(device_buttons)
        self.setLayout(layout)

        self.use_current_device_settings.toggled.connect(self._set_use_current_device_settings)
        self.configure_button.clicked.connect(self._open_config_dialog)
        self.leds_button.clicked.connect(self._open_led_config_dialog)
        self.dmd_button.clicked.connect(self._open_dmd_config_dialog)
        self.filter_wheel_button.clicked.connect(self._open_filter_wheel_config_dialog)
        if self.controller is not None:
            self.controller.virtual_image_source_received.connect(self.update_virtual_source)
            self.controller.refresh_virtual_image_source()
        self._sync_device_buttons()
        self._update_summary()

    def payload(self) -> dict:
        """Return the existing GUI acquisition payload from the reorganised controls."""
        settings = {
            key: bool(self.config_values[key])
            for key, _label, _default in self.SETTINGS
        }
        settings["save"] = self.save_config.enabled
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

    def _open_config_dialog(self) -> None:
        """Open the general acquisition and optional virtual-source dialog."""
        dialog = AcquisitionConfigurationDialog(
            fields=self._config_fields(),
            controller=self.controller,
            virtual_source=self.virtual_source,
            parent=self,
        )
        if self.controller is not None:
            self.controller.refresh_virtual_image_source()
        if dialog.exec_() != dialog.Accepted:
            return
        self.config_values.update(dialog.values())
        self._update_summary()

    def _open_led_config_dialog(self) -> None:
        """Open and apply explicit acquisition LED settings."""
        self._apply_simple_dialog("Acquisition LEDs", self._led_config_fields())

    def _open_dmd_config_dialog(self) -> None:
        """Open and apply the explicit acquisition DMD pattern."""
        self._apply_simple_dialog("Acquisition DMD", self._dmd_config_fields())

    def _open_filter_wheel_config_dialog(self) -> None:
        """Open and apply the explicit acquisition filter position."""
        self._apply_simple_dialog("Acquisition Filter Wheel", self._filter_config_fields())

    def _apply_simple_dialog(self, title: str, fields: list[ConfigFieldSpec]) -> None:
        """Apply values from one accepted device-specific configuration dialog.

        Args:
            title: Dialog window title.
            fields: Device-specific fields to display.
        """
        dialog = ConfigDialog(title=title, fields=fields, parent=self)
        if dialog.exec_() != dialog.Accepted:
            return
        self.config_values.update(dialog.values())
        self._update_summary()

    def _set_use_current_device_settings(self, checked: bool) -> None:
        """Update acquisition mode from the visible checkbox.

        Args:
            checked: Whether current live device settings should be used.
        """
        self.config_values[USE_CURRENT_MAIN_CONTROLS_KEY] = checked
        self._sync_device_buttons()
        self._update_summary()

    def _sync_device_buttons(self) -> None:
        """Disable explicit device dialogs while live device settings are selected."""
        enabled = not bool(self.config_values[USE_CURRENT_MAIN_CONTROLS_KEY])
        for button in (self.leds_button, self.dmd_button, self.filter_wheel_button):
            button.setEnabled(enabled)

    def update_virtual_source(self, payload: dict[str, Any]) -> None:
        """
        Store authoritative virtual-source state for the configuration dialog.

        Parameters
        ----------
        payload
            Virtual-source support or configuration payload from the server.

        Returns
        -------
        None
        """
        self.virtual_source = dict(payload) if bool(payload.get("supported")) else None

    def _config_fields(self) -> list[ConfigFieldSpec]:
        """Return general acquisition fields retained in the main dialog."""
        use_current = bool(self.config_values[USE_CURRENT_MAIN_CONTROLS_KEY])
        fields = [
            ConfigFieldSpec(
                "Exposure ms",
                "exposure",
                self.config_values["exposure"],
                kind="float",
                minimum=1.0,
                maximum=1000.0,
                decimals=1,
                single_step=10.0,
                editable=not use_current,
                help_text=(
                    "Camera exposure in milliseconds for explicit acquisition settings. "
                    "Ignored while current device settings are selected."
                ),
            ),
        ]
        for key, label, _default in self.SETTINGS:
            help_texts = {
                "normalise": "Request camera-supported normalisation for the returned image.",
                "illuminate_dmd": (
                    "Display the configured DMD pattern before capture. Ignored while current "
                    "device settings are selected."
                ),
                "clear_dmd_after": "Blank the DMD after acquisition completes.",
                "restore_leds_after": (
                    "Restore LED states captured before acquisition unless Disable LEDs After is set."
                ),
                "disable_leds_after": (
                    "Disable all LEDs after acquisition; this takes precedence over LED restoration."
                ),
            }
            editable = not use_current if key == "illuminate_dmd" else True
            fields.append(
                ConfigFieldSpec(
                    label,
                    key,
                    self.config_values[key],
                    kind="bool",
                    editable=editable,
                    help_text=help_texts[key],
                )
            )
        return fields

    def _filter_config_fields(self) -> list[ConfigFieldSpec]:
        """Return the explicit acquisition filter-wheel field."""
        return [
            ConfigFieldSpec(
                "Filter wheel",
                "filter_wheel",
                self.config_values["filter_wheel"],
                kind="choice",
                choices=FILTER_WHEEL_CHOICES,
                help_text=(
                    "Filter-wheel position applied before an explicitly configured capture. "
                    "BANDPASS_405_445_514_561_640nm is the Chroma "
                    "ZET405/445/514/561/640m multi-bandpass filter."
                ),
            ),
        ]

    def _dmd_config_fields(self) -> list[ConfigFieldSpec]:
        """Return the explicit acquisition DMD pattern field."""
        return [
            ConfigFieldSpec(
                "DMD pattern",
                "dmd_pattern",
                self.config_values["dmd_pattern"],
                kind="choice",
                choices=DMD_PATTERN_CHOICES,
                help_text=(
                    "Built-in DMD pattern prepared when Illuminate DMD is enabled for an "
                    "explicitly configured capture."
                ),
            ),
        ]

    def _led_config_fields(self) -> list[ConfigFieldSpec]:
        """Return channel-enable and brightness fields for explicit acquisition LEDs."""
        fields: list[ConfigFieldSpec] = []
        for led_type in ACQUISITION_LED_TYPES:
            fields.extend([
                ConfigFieldSpec(
                    f"Use {LED_LABELS.get(led_type, led_type.name)}",
                    self._led_enabled_key(led_type),
                    self.config_values[self._led_enabled_key(led_type)],
                    kind="bool",
                    help_text=(
                        f"Include the {LED_LABELS.get(led_type, led_type.name)} channel in "
                        "explicit acquisition metadata."
                    ),
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
                    enabled_when_key=self._led_enabled_key(led_type),
                    enabled_when_value=True,
                    help_text="LED brightness percentage from 0 to 100 for this acquisition.",
                ),
            ])
        return fields

    @classmethod
    def _default_config_values(cls) -> dict[str, Any]:
        values: dict[str, Any] = {
            USE_CURRENT_MAIN_CONTROLS_KEY: True,
            "exposure": 200.0,
            "filter_wheel": FilterWheelType.NO_FILTER.name,
            "dmd_pattern": "full",
        }
        values.update({key: default for key, _label, default in cls.SETTINGS})
        for led_type in ACQUISITION_LED_TYPES:
            values[cls._led_enabled_key(led_type)] = False
            values[cls._led_brightness_key(led_type)] = 10.0
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
        exposure = (
            "current device settings"
            if use_current_main_controls
            else f"{float(self.config_values['exposure']):.1f} ms"
        )
        dmd_pattern = (
            "current device settings"
            if use_current_main_controls
            else str(self.config_values["dmd_pattern"])
            if bool(self.config_values["illuminate_dmd"])
            else "off"
        )
        mode = (
            "current device settings"
            if use_current_main_controls
            else "explicit acquisition config"
        )
        settings = ", ".join(
            label
            for key, label, _default in self.SETTINGS
            if bool(self.config_values[key])
        )
        self.summary_label.setText(
            f"mode: {mode}; exposure: {exposure}; "
            f"DMD: {dmd_pattern}; "
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
        self.acquisition_running = False
        self.start_z = -10.0
        self.end_z = 10.0
        self.step_z = 1.0
        self.summary_label = QLabel()
        self.summary_label.setWordWrap(True)
        self.configure_button = QPushButton("Configure")
        self.configure_button.setToolTip("Configure the relative Z-stack range and step size.")
        self.acquire_button = QPushButton("Acquire Z Stack")
        self.status_label = QLabel("Run Initialise Devices before z-stack acquisition.")
        self.operation_poll_timer = QTimer(self)
        self.operation_poll_timer.setInterval(250)

        layout = QVBoxLayout()
        layout.addWidget(self.summary_label)
        layout.addWidget(self.configure_button)
        layout.addWidget(self.acquire_button)
        layout.addWidget(self.status_label)
        self.setLayout(layout)

        self.configure_button.clicked.connect(self._open_config_dialog)
        self.acquire_button.clicked.connect(self._acquire_z_stack)
        self.controller.frame_received.connect(self.update_frame_status)
        self.controller.lifecycle_status_received.connect(self.update_lifecycle_status)
        self.controller.response_error.connect(self._show_error)
        self.controller.operation_status_received.connect(self.update_operation_status)
        self.operation_poll_timer.timeout.connect(self.controller.refresh_z_stack_operation)
        self._update_summary()
        self._sync_controls_enabled()

    def _open_config_dialog(self) -> None:
        """Open and apply relative Z-stack range configuration."""
        dialog = ConfigDialog(
            title="Z Stack Configuration",
            fields=self._config_fields(),
            parent=self,
        )
        if dialog.exec_() != dialog.Accepted:
            return
        values = dialog.values()
        self.start_z = float(values["start_z"])
        self.end_z = float(values["end_z"])
        self.step_z = float(values["step_z"])
        self._update_summary()

    def _config_fields(self) -> list[ConfigFieldSpec]:
        """Return relative Z-stack range fields."""
        return [
            ConfigFieldSpec(
                "Start ΔZ (µm)", "start_z", self.start_z, kind="float",
                minimum=-1e7, maximum=1e7, decimals=3, single_step=1.0,
                help_text="Relative Z offset in micrometres for the first stack plane.",
            ),
            ConfigFieldSpec(
                "End ΔZ (µm)", "end_z", self.end_z, kind="float",
                minimum=-1e7, maximum=1e7, decimals=3, single_step=1.0,
                help_text="Relative Z offset in micrometres for the final stack plane.",
            ),
            ConfigFieldSpec(
                "ΔZ step (µm)", "step_z", self.step_z, kind="float",
                minimum=0.001, maximum=1e7, decimals=3, single_step=1.0,
                help_text="Positive spacing in micrometres between successive stack planes.",
            ),
        ]

    def _update_summary(self) -> None:
        """Display the active relative Z-stack range."""
        self.summary_label.setText(
            f"ΔZ: {self.start_z:.3f} → {self.end_z:.3f} µm; step: {self.step_z:.3f} µm"
        )

    def _acquire_z_stack(self) -> None:
        if not self.devices_initialised:
            self.status_label.setText("Run Initialise Devices before z-stack acquisition.")
            return
        payload = self.settings_provider()
        payload.update({
            "start_z": self.start_z,
            "end_z": self.end_z,
            "step_z": self.step_z,
        })
        self.status_label.setText("Acquiring z-stack.")
        self.acquisition_running = True
        self.operation_poll_timer.start()
        self._sync_controls_enabled()
        self.controller.acquire_z_stack(payload)

    def update_operation_status(self, operation: dict) -> None:
        if operation.get("kind") != "z_stack_acquisition":
            return
        state = str(operation.get("state", "-"))
        self.acquisition_running = state == "running"
        if not self.acquisition_running:
            self.operation_poll_timer.stop()
            if state == "failed" and operation.get("error"):
                self.status_label.setText(str(operation["error"]))
            elif state != "completed":
                self.status_label.setText(f"Z-stack acquisition {state}.")
        self._sync_controls_enabled()

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
        self.configure_button.setEnabled(not self.acquisition_running)
        self.acquire_button.setEnabled(self.devices_initialised and not self.acquisition_running)

    def _show_error(self, error: str) -> None:
        if self.status_label.text().startswith("Acquiring z-stack"):
            self.acquisition_running = False
            self.operation_poll_timer.stop()
            self._sync_controls_enabled()
            self.status_label.setText(error)


class FilenamePatternDialog(QDialog):
    """Edit and validate one output filename pattern without applying it directly."""

    def __init__(self, current_pattern: str, parent: QWidget | None = None):
        """
        Initialise the filename-pattern editor.

        Parameters
        ----------
        current_pattern
            Active pattern shown initially in the editable field.
        parent
            Optional parent widget that owns the modal dialog.

        Returns
        -------
        None
        """
        super().__init__(parent)
        self.setWindowTitle("Filename Pattern")
        self.pattern_input = QLineEdit(current_pattern)
        self.validate_button = QPushButton("Validate Pattern")
        self.validation_label = QLabel("")
        self.validation_label.setWordWrap(True)
        self.help_label = QLabel(self._help_text())
        self.help_label.setWordWrap(True)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Filename pattern"))
        layout.addWidget(self.pattern_input)
        layout.addWidget(self.validate_button)
        layout.addWidget(self.validation_label)
        layout.addWidget(self.help_label)
        layout.addWidget(self.buttons)

        self.pattern_input.textChanged.connect(self._clear_validation)
        self.validate_button.clicked.connect(self.validate_pattern)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.setMinimumWidth(560)

    def pattern(self) -> str:
        """
        Return the pattern currently entered by the user.

        Returns
        -------
        str
            Unmodified text from the pattern input field.
        """
        return self.pattern_input.text()

    def validate_pattern(self) -> bool:
        """
        Validate the entered pattern and display an immediate result.

        Returns
        -------
        bool
            True when FileNameConfig accepts the pattern, otherwise False.
        """
        try:
            FileNameConfig(directory=Path("."), filename_pattern=self.pattern())
        except (TypeError, ValueError) as error:
            self.validation_label.setStyleSheet("color: #d9534f;")
            self.validation_label.setText(str(error))
            return False
        self.validation_label.setStyleSheet("color: #4caf50;")
        self.validation_label.setText("Pattern is valid.")
        return True

    def accept(self) -> None:
        """
        Accept the dialog only when the current pattern validates successfully.

        Returns
        -------
        None
        """
        if self.validate_pattern():
            super().accept()

    def _clear_validation(self, _text: str) -> None:
        """
        Clear stale validation feedback after the pattern text changes.

        Parameters
        ----------
        _text
            Updated input text supplied by Qt; the value is already available
            from ``pattern_input`` and is intentionally unused.

        Returns
        -------
        None
        """
        self.validation_label.clear()
        self.validation_label.setStyleSheet("")

    @staticmethod
    def _help_text() -> str:
        """
        Build the filename-pattern syntax and field reference shown to users.

        Returns
        -------
        str
            Multi-line instructions covering syntax, fields, and uniqueness.
        """
        field_lines = "\n".join(
            f"  {{{field.value}}}: {description}"
            for field, description in FILENAME_FIELD_HELP
        )
        return (
            "Combine literal text with fields in braces, for example "
            "FOV{fov_id}_{timestamp}_{frame_id}. Use doubled braces {{ and }} "
            "for literal braces. Python format specifications are supported, "
            "for example {frame_id:04d}.\n\nAvailable fields:\n"
            f"{field_lines}\n\nThe .tiff extension is added automatically. Include "
            "timestamp, frame_id, or callback_id to avoid overwriting files."
        )


class OutputConfigurationPanel(QGroupBox):
    """Configure the shared output destination and filename pattern."""

    def __init__(
        self,
        controller,
        parent: QWidget | None = None,
        save_config: ManualAcquisitionSaveConfig | None = None,
    ):
        """
        Initialise output-folder and filename-pattern controls.

        Parameters
        ----------
        controller
            GUI controller used to read and update server-side configuration.
        parent
            Optional parent widget.
        save_config
            Optional shared manual-save state. When supplied, this panel exposes
            the manual frame and Z-stack saving checkbox.

        Returns
        -------
        None
        """
        super().__init__("Output Configuration", parent)
        self.controller = controller
        self.save_config = save_config
        self.directory = ""
        self.filename_pattern = ""
        self._pattern_update_pending = False
        self.path_label = QLabel("folder: loading configured destination...")
        self.path_label.setWordWrap(True)
        self.pattern_label = QLabel("pattern: loading configured pattern...")
        self.pattern_label.setWordWrap(True)
        self.choose_directory_button = QPushButton("Choose Output Folder")
        self.modify_pattern_button = QPushButton("Modify Pattern")
        self.save_checkbox: QCheckBox | None = None

        layout = QVBoxLayout()
        if self.save_config is not None:
            self.save_checkbox = QCheckBox("Save manual acquisitions")
            self.save_checkbox.setChecked(self.save_config.enabled)
            self.save_checkbox.setToolTip(
                "Save frames acquired from the Acquisition workflow and its Z-stack control."
            )
            self.save_checkbox.toggled.connect(self._set_manual_save)
            layout.addWidget(self.save_checkbox)
            save_help = QLabel(
                "Applies to GUI manual frames and Z-stacks only. Python and AutoStrat "
                "strategies control their own saving."
            )
            save_help.setWordWrap(True)
            layout.addWidget(save_help)
        layout.addWidget(self.path_label)
        layout.addWidget(self.choose_directory_button)
        layout.addWidget(QLabel("Used by saved manual acquisitions, z-stacks, and strategies."))
        layout.addWidget(self.pattern_label)
        layout.addWidget(self.modify_pattern_button)
        self.setLayout(layout)

        self.choose_directory_button.clicked.connect(self._choose_directory)
        self.modify_pattern_button.clicked.connect(self._open_pattern_dialog)
        self.controller.acquisition_directory_received.connect(self.update_directory)
        self.controller.output_configuration_received.connect(self.update_configuration)
        self.controller.response_error.connect(self._show_error)
        self.controller.request_error.connect(self._show_request_error)
        self.controller.refresh_acquisition_files()

    def _set_manual_save(self, enabled: bool) -> None:
        """Store the manual acquisition save choice.

        Args:
            enabled: Whether workflow frames and Z-stacks should be saved.
        """
        if self.save_config is not None:
            self.save_config.enabled = enabled

    def _choose_directory(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            "Choose Acquisition Output Folder",
            self.directory,
        )
        if directory:
            self.path_label.setText(f"selecting: {directory}")
            self.controller.set_acquisition_directory(directory)

    def update_directory(self, payload: dict) -> None:
        """
        Update the displayed output directory from a controller response.

        Parameters
        ----------
        payload
            Mapping containing the authoritative ``directory`` value.

        Returns
        -------
        None
        """
        directory = str(payload.get("directory") or "")
        if directory:
            self.directory = directory
            self.path_label.setText(f"folder: {directory}")

    def update_configuration(self, payload: dict) -> None:
        """
        Display authoritative output configuration returned by the server.

        Parameters
        ----------
        payload
            Mapping containing optional directory and filename-pattern values.

        Returns
        -------
        None
        """
        self.update_directory(payload)
        filename_pattern = payload.get("filename_pattern")
        if isinstance(filename_pattern, str) and filename_pattern:
            self.filename_pattern = filename_pattern
            self.pattern_label.setText(f"pattern: {filename_pattern}")
            self._pattern_update_pending = False

    def _open_pattern_dialog(self) -> None:
        """
        Open the pattern editor and submit an accepted valid pattern.

        Returns
        -------
        None
        """
        dialog = FilenamePatternDialog(current_pattern=self.filename_pattern, parent=self)
        if dialog.exec_() != dialog.Accepted:
            return
        self._pattern_update_pending = True
        self.pattern_label.setText("pattern: updating...")
        self.controller.set_acquisition_filename_pattern(dialog.pattern())

    def _show_error(self, error: str) -> None:
        """
        Display an output-directory request failure when one is pending.

        Parameters
        ----------
        error
            User-facing error text emitted by the controller.

        Returns
        -------
        None
        """
        if self.path_label.text().startswith("selecting:"):
            self.path_label.setText(error)

    def _show_request_error(self, command: object, error: str) -> None:
        """
        Display a failed filename-pattern update without affecting folder state.

        Parameters
        ----------
        command
            GuiCommandType associated with the failed request.
        error
            User-facing error returned by the server.

        Returns
        -------
        None
        """
        if command is GuiCommandType.ACQUISITION_SET_FILENAME_PATTERN and self._pattern_update_pending:
            self.pattern_label.setText(f"pattern error: {error}")
            self._pattern_update_pending = False


class SavedImageLoaderPanel(QGroupBox):
    """Browse one local folder and load its acquisition TIFFs."""

    def __init__(self, controller, parent: QWidget | None = None):
        super().__init__("Load Saved Images", parent)
        self.controller = controller
        self.file_combo = NoWheelComboBox()
        self.loading_directory = ""
        self.force_socket_transport = False
        self.refresh_button = QPushButton("Refresh Images")
        self.choose_directory_button = QPushButton("Choose Loading Folder")
        self.configure_button = QPushButton("Configure")
        self.load_button = QPushButton("Load Selected")
        self.path_label = QLabel("loading folder: -")
        self.status_label = QLabel("Refresh files to load a saved TIFF.")
        self.transport_label = QLabel()
        self.path_label.setWordWrap(True)

        button_grid = QGridLayout()
        button_grid.addWidget(self.choose_directory_button, 0, 0, 1, 2)
        button_grid.addWidget(self.refresh_button, 1, 0)
        button_grid.addWidget(self.configure_button, 1, 1)
        button_grid.addWidget(self.load_button, 2, 0, 1, 2)

        layout = QVBoxLayout()
        layout.addWidget(self.path_label)
        layout.addWidget(QLabel("Image"))
        layout.addWidget(self.file_combo)
        layout.addWidget(self.transport_label)
        layout.addLayout(button_grid)
        layout.addWidget(self.status_label)
        self.setLayout(layout)

        self.refresh_button.clicked.connect(self._refresh_files)
        self.choose_directory_button.clicked.connect(self._choose_directory)
        self.configure_button.clicked.connect(self._open_config_dialog)
        self.load_button.clicked.connect(self._load_selected)
        self.file_combo.currentIndexChanged.connect(self._update_selected_path_label)
        self.controller.acquisition_files_received.connect(self.update_file_list)
        self.controller.acquisition_loading_directory_received.connect(self.update_loading_directory)
        self.controller.frame_received.connect(self.update_frame_status)
        self.controller.response_error.connect(self._show_error)
        self._update_transport_label()
        self._sync_controls_enabled()
        self._refresh_files()

    def _refresh_files(self) -> None:
        self.status_label.setText("Refreshing images.")
        self.controller.refresh_acquisition_files(self.loading_directory or None)

    def _choose_directory(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            "Choose Folder Containing Images",
            self.loading_directory,
        )
        if not directory:
            return
        self.loading_directory = directory
        self.status_label.setText(f"Loading images from {directory}.")
        self.controller.refresh_acquisition_files(directory)

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

    def update_frame_status(self, payload: dict) -> None:
        if payload.get("source") != "file":
            return
        path = payload.get("loaded_path") or "-"
        self.status_label.setText(f"Loaded {Path(str(path)).name}: {payload.get('image_shape')}")

    def update_loading_directory(self, directory: str) -> None:
        directory = str(directory or "")
        if not directory:
            return
        self.loading_directory = directory
        self.status_label.setText(f"Loading images from {directory}.")
        self._update_selected_path_label()

    def _sync_controls_enabled(self) -> None:
        has_file = self.file_combo.count() > 0
        self.load_button.setEnabled(has_file)
        self.choose_directory_button.setEnabled(True)
        self.configure_button.setEnabled(True)

    def _update_selected_path_label(self, _index: int | None = None) -> None:
        filename = self._selected_filename()
        directory = self.loading_directory.strip()
        if filename is not None:
            self.path_label.setText(f"selected image: {filename}")
        elif directory:
            self.path_label.setText(f"loading folder: {directory}")
        else:
            self.path_label.setText("loading folder: -")

    def _update_transport_label(self) -> None:
        transport = "socket" if self.force_socket_transport else "auto"
        self.transport_label.setText(f"transport: {transport}")

    def _show_error(self, error: str) -> None:
        if self.status_label.text().startswith(
            ("Loading images", "Refreshing", "Loading selected")
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
        # Every plane of a z-stack points at the same stack file, so count files once.
        saved_paths = list(dict.fromkeys(path for path in payload.get("saved_paths", []) if path))
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
