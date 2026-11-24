"""Adaptive manual controls for the unified software-focus engine."""

from __future__ import annotations

from PyQt5.QtCore import QSignalBlocker, QTimer
from PyQt5.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QFormLayout, QGridLayout, QGroupBox,
    QLabel, QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

from evomachine.gui.panels.config_dialog import ConfigDialog, ConfigFieldSpec
from evomachine.gui.panels.focus_settings import ReferenceSettingsDialog
from evomachine.gui.panels.reference_focus import FocusCurves


class FocusCropDialog(QDialog):
    """Edit the GUI-supported optional single focus crop."""

    def __init__(self, crop: dict | None, camera_shape: tuple[int, int], parent=None):
        """Create a bounded four-coordinate crop editor."""
        super().__init__(parent)
        self.setWindowTitle("Focus Crop")
        height, width = camera_shape
        values = crop or {"x_left": 0, "y_top": 0, "x_right": width, "y_bottom": height}
        self.inputs = {}
        form = QFormLayout()
        for label, key, maximum in (
            ("Left X", "x_left", width - 1), ("Top Y", "y_top", height - 1),
            ("Right X", "x_right", width), ("Bottom Y", "y_bottom", height),
        ):
            control = QSpinBox()
            control.setRange(0, max(0, maximum))
            control.setValue(int(values[key]))
            control.setToolTip("Pixel coordinate; right and bottom are exclusive.")
            self.inputs[key] = control
            form.addRow(label, control)
        self.message = QLabel("Clear selects whole-image scoring.")
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)
        buttons.addButton("Clear", QDialogButtonBox.ResetRole).clicked.connect(self._clear)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.message)
        layout.addWidget(buttons)
        self._cleared = False

    def _validate_and_accept(self) -> None:
        """Accept only a positive-area rectangle."""
        values = self.values()
        if values and (values["x_right"] <= values["x_left"] or values["y_bottom"] <= values["y_top"]):
            self.message.setText("Right and bottom must exceed left and top.")
            return
        self.accept()

    def _clear(self) -> None:
        """Select whole-image scoring and accept the dialog."""
        self._cleared = True
        self.accept()

    def values(self) -> dict | None:
        """Return one crop payload or ``None`` for the whole image."""
        return None if self._cleared else {key: value.value() for key, value in self.inputs.items()}


class SoftwareFocusPanel(QGroupBox):
    """Control maximum and reference-profile focus through one engine."""

    SCORING_TYPES = ("LAPLACIAN_VAR", "SQUARED_GRAD_AVG", "STEEL", "BANDPASS_FFT")
    SELECTION_TYPES = ("MAXIMUM", "REFERENCE_PROFILE")

    def __init__(self, controller, parent: QWidget | None = None):
        """Create controls and connect authoritative controller responses."""
        super().__init__("Software Focus", parent)
        self.controller = controller
        self.devices_initialised = self.strategy_running = self.focus_running = False
        self.reference_accepted = False
        self.reference_curve_available = False
        self.reference_needs_calibration = False
        self._latest_config = {}
        self._camera_shape = (3200, 3200)
        self.status_label = QLabel("Run Initialise Devices before using software focus controls.")
        self.result_label = QLabel("result: -")
        self.crop_label = QLabel("crop: whole image")
        self.selection_combo = QComboBox()
        self.selection_combo.addItems(self.SELECTION_TYPES)
        self.scoring_combo = QComboBox()
        self.scoring_combo.addItems(self.SCORING_TYPES)
        self.configure_selection_button = QPushButton("Configure Focus Method")
        self.configure_scoring_button = QPushButton("Configure Scoring")
        self.configure_imaging_button = QPushButton("Configure Imaging")
        self.configure_crop_button = QPushButton("Configure Crop")
        self.refresh_button = QPushButton("Refresh")
        self.run_button = QPushButton("Run")
        self.calibrate_button = QPushButton("Calibrate")
        self.accept_button = QPushButton("Accept Reference")
        self.test_button = QPushButton("Test Reference")
        self.clear_button = QPushButton("Clear Reference")
        self.cancel_button = QPushButton("Cancel")
        self.curves = FocusCurves()
        form = QFormLayout()
        form.addRow("Focus method", self.selection_combo)
        form.addRow("Scoring method", self.scoring_combo)
        configs = QGridLayout()
        config_buttons = (
            self.configure_selection_button,
            self.configure_scoring_button,
            self.configure_imaging_button,
            self.configure_crop_button,
        )
        for index, button in enumerate(config_buttons):
            configs.addWidget(button, index // 2, index % 2)
        actions = QGridLayout()
        action_buttons = (
            self.run_button,
            self.calibrate_button,
            self.accept_button,
            self.test_button,
            self.clear_button,
            self.cancel_button,
        )
        for index, button in enumerate(action_buttons):
            actions.addWidget(button, index // 2, index % 2)
        layout = QVBoxLayout(self)
        layout.addWidget(self.status_label)
        layout.addLayout(form)
        layout.addLayout(configs)
        layout.addWidget(self.crop_label)
        layout.addWidget(self.result_label)
        layout.addLayout(actions)
        layout.addWidget(self.refresh_button)
        layout.addWidget(self.curves)
        self.operation_poll_timer = QTimer(self)
        self.operation_poll_timer.setInterval(250)
        self.operation_poll_timer.timeout.connect(controller.refresh_software_focus_operation)
        self.refresh_button.clicked.connect(controller.refresh_software_focus)
        self.run_button.clicked.connect(lambda: self._run("run"))
        self.calibrate_button.clicked.connect(lambda: self._run("calibrate"))
        self.test_button.clicked.connect(lambda: self._run("test"))
        self.accept_button.clicked.connect(lambda: controller.edit_software_focus("accept"))
        self.clear_button.clicked.connect(lambda: controller.edit_software_focus("clear"))
        self.cancel_button.clicked.connect(controller.cancel_software_focus)
        self.selection_combo.currentTextChanged.connect(self._selection_changed)
        self.scoring_combo.currentTextChanged.connect(self._scoring_changed)
        self.configure_selection_button.clicked.connect(self._configure_selection)
        self.configure_scoring_button.clicked.connect(self._configure_scoring)
        self.configure_imaging_button.clicked.connect(self._configure_imaging)
        self.configure_crop_button.clicked.connect(self._configure_crop)
        controller.software_focus_status_received.connect(self.update_status)
        controller.lifecycle_status_received.connect(self.update_lifecycle_status)
        controller.strategy_status_received.connect(self.update_strategy_status)
        controller.operation_status_received.connect(self.update_operation_status)
        controller.response_error.connect(self._show_error)
        self._sync_controls_enabled()

    def _run(self, action: str) -> None:
        """Start one focus operation."""
        if not self.devices_initialised:
            return
        self.focus_running = True
        self.operation_poll_timer.start()
        self.controller.run_software_focus(action)
        self._sync_controls_enabled()

    def _selection_changed(self, name: str) -> None:
        """Submit the selected focal-Z policy."""
        if name and name != self._latest_config.get("selection_type"):
            self.controller.configure_software_focus({"selection_type": name, "selection_config": {}})

    def _scoring_changed(self, name: str) -> None:
        """Submit the selected scoring implementation."""
        if name and name != self._latest_config.get("scoring_type"):
            self.controller.configure_software_focus({"scoring_type": name, "scoring_config": {}})

    def _configure_selection(self) -> None:
        """Open settings belonging to the selected focus method."""
        config = self._latest_config.get("selection_config") or {}
        if self.selection_combo.currentText() == "MAXIMUM":
            specs = [
                ConfigFieldSpec(
                    "Relative range (µm)", "rel_range",
                    config.get("rel_range", 50), kind="float", minimum=0.1,
                ),
                ConfigFieldSpec(
                    "Step size (µm)", "step_size",
                    config.get("step_size", 5), kind="float", minimum=0.1,
                ),
            ]
            self._submit_dialog("Focus Method Configuration", specs, "selection_config")
        else:
            dialog = ReferenceSettingsDialog(config, self)
            if dialog.exec_() == QDialog.Accepted:
                self.controller.configure_software_focus(
                    {"selection_config": dialog.values()}
                )

    def _configure_scoring(self) -> None:
        """Open settings belonging to the selected scorer."""
        config = self._latest_config.get("scoring_config") or {}
        name = self.scoring_combo.currentText()
        specs = {
            "LAPLACIAN_VAR": [],
            "SQUARED_GRAD_AVG": [
                ConfigFieldSpec(
                    "Threshold", "threshold", config.get("threshold") or 0,
                    kind="float",
                    help_text="Horizontal gradients below this value contribute zero.",
                )
            ],
            "STEEL": [
                ConfigFieldSpec(
                    "Row shift (px)", "rowshift", config.get("rowshift", 25),
                    kind="int", minimum=0,
                    help_text="Pixel displacement used for vertical shifted differences.",
                ),
                ConfigFieldSpec(
                    "Column shift (px)", "colshift", config.get("colshift", 50),
                    kind="int", minimum=0,
                    help_text="Pixel displacement used for horizontal shifted differences.",
                ),
                ConfigFieldSpec(
                    "Normalise", "normalise", config.get("normalise", False),
                    kind="bool",
                    help_text="Divide the Steel score by twice the image area.",
                ),
            ],
            "BANDPASS_FFT": [
                ConfigFieldSpec(
                    "Cell radius (px)", "cell_radius", config.get("cell_radius", 20),
                    kind="float", minimum=0.1,
                ),
                ConfigFieldSpec(
                    "Downsample", "downsample", config.get("downsample", 4),
                    kind="int", minimum=1,
                ),
                ConfigFieldSpec(
                    "Filter order", "order", config.get("order", 4),
                    kind="int", minimum=1,
                ),
                ConfigFieldSpec(
                    "Saturation ceiling", "saturation",
                    config.get("saturation") or 0, kind="float", minimum=0,
                    help_text="Upper intensity clip; zero is submitted as no clipping.",
                ),
                ConfigFieldSpec(
                    "Normalise brightness", "normalize_brightness",
                    config.get("normalize_brightness", True), kind="bool",
                ),
            ],
        }[name]
        if not specs:
            self.status_label.setText("Laplacian variance has no configurable options.")
            return
        dialog = ConfigDialog(title="Scoring Configuration", fields=specs, parent=self)
        if dialog.exec_() == QDialog.Accepted:
            values = dialog.values()
            if name == "BANDPASS_FFT" and values["saturation"] == 0:
                values["saturation"] = None
            self.controller.configure_software_focus({"scoring_config": values})

    def _configure_imaging(self) -> None:
        """Configure the first focus frame and shared save/normalise settings."""
        frames = self._latest_config.get("focus_frames") or [{}]
        acquisition = self._latest_config.get("acquisition_settings") or {}
        dialog = ConfigDialog(
            title="Focus Imaging Configuration",
            fields=[
                ConfigFieldSpec(
                    "Exposure (ms)", "exposure", frames[0].get("exposure", 200),
                    kind="float", minimum=0.01,
                ),
                ConfigFieldSpec(
                    "Save focus stacks", "save", acquisition.get("save", False),
                    kind="bool",
                ),
                ConfigFieldSpec(
                    "Normalise images", "normalise",
                    acquisition.get("normalise", False), kind="bool",
                ),
            ],
            parent=self,
        )
        if dialog.exec_() == QDialog.Accepted:
            values = dialog.values()
            frame = dict(frames[0], exposure=values.pop("exposure"))
            self.controller.configure_software_focus({
                "focus_frames": [frame],
                "acquisition_settings": dict(acquisition, **values),
            })

    def _configure_crop(self) -> None:
        """Edit zero or one crop; applying replaces any Python-configured list."""
        crops = self._latest_config.get("cropping_boxes") or []
        dialog = FocusCropDialog(crops[0] if crops else None, self._camera_shape, self)
        if dialog.exec_() == QDialog.Accepted:
            crop = dialog.values()
            self.controller.configure_software_focus(
                {"cropping_boxes": [] if crop is None else [crop]}
            )

    def _submit_dialog(self, title: str, specs: list[ConfigFieldSpec], key: str) -> None:
        """Submit one accepted typed subconfiguration."""
        dialog = ConfigDialog(title=title, fields=specs, parent=self)
        if dialog.exec_() == QDialog.Accepted:
            self.controller.configure_software_focus({key: dialog.values()})

    def update_status(self, payload: dict) -> None:
        """Apply an authoritative unified focus response."""
        self._latest_config = dict(payload.get("config") or {})
        config = self._latest_config
        self.reference_accepted = bool(payload.get("accepted"))
        self.reference_curve_available = payload.get("curve") is not None
        self.reference_needs_calibration = bool(payload.get("needs_calibration"))
        shape = payload.get("camera_shape")
        if isinstance(shape, (list, tuple)) and len(shape) == 2:
            self._camera_shape = (int(shape[0]), int(shape[1]))
        with QSignalBlocker(self.selection_combo):
            self.selection_combo.setCurrentText(config.get("selection_type", "MAXIMUM"))
        with QSignalBlocker(self.scoring_combo):
            self.scoring_combo.setCurrentText(config.get("scoring_type", "STEEL"))
        crops = config.get("cropping_boxes") or []
        crop_text = "crop: whole image" if not crops else "crop: " + "; ".join(
            f"({box['x_left']}, {box['y_top']})–({box['x_right']}, {box['y_bottom']})"
            for box in crops
        )
        self.crop_label.setText(crop_text)
        result = payload.get("last_result") or payload.get("result")
        status = None if not result else result.get("status", result.get("focus_status"))
        if isinstance(status, dict):
            status = status.get("name")
        self.result_label.setText("result: -" if status is None else f"result: {status}")
        reference = config.get("selection_type") == "REFERENCE_PROFILE"
        self.curves.setVisible(reference)
        self.curves.update_curves(payload)
        self.status_label.setText(
            payload.get("message") or "Software focus configuration loaded."
        )
        self._sync_controls_enabled()

    def update_operation_status(self, operation: dict) -> None:
        """Update busy state from the shared operation."""
        if operation.get("kind") != "software_focus":
            return
        self.focus_running = operation.get("state") == "running"
        if not self.focus_running:
            self.operation_poll_timer.stop()
        self._sync_controls_enabled()

    def update_lifecycle_status(self, payload: dict) -> None:
        """Update device lifecycle gating."""
        self.devices_initialised = bool(payload.get("devices_initialised")) and not payload.get(
            "shutdown"
        )
        self._sync_controls_enabled()

    def update_strategy_status(self, payload: dict) -> None:
        """Update strategy mutation gating."""
        self.strategy_running = bool(payload.get("running"))
        self._sync_controls_enabled()

    def _sync_controls_enabled(self) -> None:
        """Apply lifecycle and method-specific control gating."""
        editable = self.devices_initialised and not self.strategy_running and not self.focus_running
        reference = self.selection_combo.currentText() == "REFERENCE_PROFILE"
        editable_widgets = (
            self.selection_combo,
            self.scoring_combo,
            self.configure_selection_button,
            self.configure_scoring_button,
            self.configure_imaging_button,
            self.configure_crop_button,
        )
        for widget in editable_widgets:
            widget.setEnabled(editable)
        self.refresh_button.setEnabled(not self.focus_running)
        self.run_button.setEnabled(editable and (not reference or self.reference_accepted))
        self.calibrate_button.setEnabled(editable and reference)
        self.accept_button.setEnabled(
            editable
            and reference
            and self.reference_curve_available
            and not self.reference_accepted
            and not self.reference_needs_calibration
        )
        self.test_button.setEnabled(editable and reference and self.reference_accepted)
        self.clear_button.setEnabled(
            editable and reference and (self.reference_curve_available or self.reference_accepted)
        )
        self.cancel_button.setEnabled(self.focus_running)

    def _show_error(self, error: str) -> None:
        """Show focus-related request errors."""
        if "focus" in error.lower():
            self.status_label.setText(error)
