from __future__ import annotations

from PyQt5.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from evomachine.gui.panels.common import muted_label


class AutofocusPanel(QGroupBox):
    """Hardware autofocus controls."""

    def __init__(self, controller, parent: QWidget | None = None):
        super().__init__("Hardware Autofocus", parent)
        self.controller = controller
        self.devices_initialised = False
        self.status_label = QLabel("Run Initialise Devices before using autofocus controls.")
        self.status_label.setWordWrap(True)
        self.state_label = QLabel("status: -")
        self.locked_label = QLabel("locked: -")
        self.note_label = muted_label("CRISP configuration values are applied before calibration.")
        self.note_label.setWordWrap(True)
        self.config_inputs = self._make_config_inputs()
        self.lock_after_calibration_checkbox = QCheckBox("Lock after calibration")
        self.lock_after_calibration_checkbox.setChecked(False)

        self.refresh_button = QPushButton("Refresh")
        self.apply_config_button = QPushButton("Apply Config")
        self.run_calibration_button = QPushButton("Run Calibration")
        self.lock_button = QPushButton("Lock")
        self.unlock_button = QPushButton("Unlock")

        config_form = QFormLayout()
        config_form.addRow("Averaging", self.config_inputs["averaging"])
        config_form.addRow("LED intensity", self.config_inputs["led_intensity"])
        config_form.addRow("Lock range", self.config_inputs["lock_range"])
        config_form.addRow("Loop gain", self.config_inputs["loop_gain"])
        config_form.addRow("Update rate", self.config_inputs["update_rate"])
        config_form.addRow("Objective NA", self.config_inputs["objective_na"])
        config_form.addRow("Minimum SNR", self.config_inputs["min_snr"])
        config_form.addRow("Minimum error", self.config_inputs["min_error"])

        buttons = QGridLayout()
        buttons.addWidget(self.refresh_button, 0, 0)
        buttons.addWidget(self.apply_config_button, 0, 1)
        buttons.addWidget(self.run_calibration_button, 1, 0)
        buttons.addWidget(self.lock_after_calibration_checkbox, 1, 1)
        buttons.addWidget(self.lock_button, 2, 0)
        buttons.addWidget(self.unlock_button, 2, 1)

        layout = QVBoxLayout()
        layout.addWidget(self.status_label)
        layout.addWidget(self.state_label)
        layout.addWidget(self.locked_label)
        layout.addWidget(self.note_label)
        layout.addLayout(config_form)
        layout.addLayout(buttons)
        self.setLayout(layout)

        self.refresh_button.clicked.connect(self.controller.refresh_autofocus)
        self.apply_config_button.clicked.connect(self._apply_config)
        self.run_calibration_button.clicked.connect(self._run_calibration)
        self.lock_button.clicked.connect(self._lock_autofocus)
        self.unlock_button.clicked.connect(self._unlock_autofocus)
        self.controller.autofocus_status_received.connect(self.update_status)
        self.controller.lifecycle_status_received.connect(self.update_lifecycle_status)
        self.controller.response_error.connect(self._show_error)
        self._set_default_config_values()
        self._sync_controls_enabled()

    @staticmethod
    def _make_config_inputs() -> dict[str, QSpinBox | QDoubleSpinBox]:
        inputs: dict[str, QSpinBox | QDoubleSpinBox] = {}
        int_specs = {
            "averaging": (0, 99, 5),
            "led_intensity": (2, 100, 70),
            "loop_gain": (1, 100, 10),
            "update_rate": (0, 1000, 10),
            "min_error": (0, 100000, 100),
        }
        for key, (minimum, maximum, value) in int_specs.items():
            box = QSpinBox()
            box.setRange(minimum, maximum)
            box.setValue(value)
            inputs[key] = box

        double_specs = {
            "lock_range": (0.001, 0.999, 0.1, 3),
            "objective_na": (0.01, 10.0, 0.9, 3),
            "min_snr": (0.0, 1000.0, 2.0, 2),
        }
        for key, (minimum, maximum, value, decimals) in double_specs.items():
            box = QDoubleSpinBox()
            box.setRange(minimum, maximum)
            box.setDecimals(decimals)
            box.setSingleStep(0.1)
            box.setValue(value)
            inputs[key] = box
        return inputs

    def _set_default_config_values(self) -> None:
        self.config_inputs["averaging"].setValue(5)
        self.config_inputs["led_intensity"].setValue(70)
        self.config_inputs["lock_range"].setValue(0.1)
        self.config_inputs["loop_gain"].setValue(10)
        self.config_inputs["update_rate"].setValue(10)
        self.config_inputs["objective_na"].setValue(0.9)
        self.config_inputs["min_snr"].setValue(2.0)
        self.config_inputs["min_error"].setValue(100)

    def _config_payload(self) -> dict[str, float | int]:
        payload: dict[str, float | int] = {}
        for key, widget in self.config_inputs.items():
            payload[key] = widget.value()
        return payload

    def _apply_config(self) -> None:
        if not self._ensure_devices_initialised():
            return
        self.status_label.setText("Applying autofocus configuration.")
        self.controller.configure_autofocus(config=self._config_payload())

    def _run_calibration(self) -> None:
        if not self._ensure_devices_initialised():
            return
        self.status_label.setText("Running autofocus calibration.")
        self.controller.initialise_autofocus(
            lock_after_initialise=self.lock_after_calibration_checkbox.isChecked(),
            config=self._config_payload(),
        )

    def _lock_autofocus(self) -> None:
        if not self._ensure_devices_initialised():
            return
        self.status_label.setText("Locking autofocus.")
        self.controller.lock_autofocus()

    def _unlock_autofocus(self) -> None:
        if not self._ensure_devices_initialised():
            return
        self.status_label.setText("Unlocking autofocus.")
        self.controller.unlock_autofocus()

    def update_status(self, payload: dict) -> None:
        status = payload.get("status", {})
        status_name = status.get("name") if isinstance(status, dict) else status
        self.status_label.setText(
            f"initialised: {payload.get('is_initialised')}, alive: {payload.get('is_alive')}"
        )
        self.state_label.setText(f"status: {self._format_status(status_name)}")
        self.locked_label.setText(f"locked: {payload.get('is_locked')}")
        config = payload.get("config")
        if isinstance(config, dict):
            self._update_config_inputs(config)
        self._sync_controls_enabled()

    def _update_config_inputs(self, config: dict) -> None:
        for key, widget in self.config_inputs.items():
            if key not in config or config[key] is None:
                continue
            value = config[key]
            if isinstance(widget, QSpinBox):
                widget.setValue(int(value))
            else:
                widget.setValue(float(value))

    def update_lifecycle_status(self, payload: dict) -> None:
        self.devices_initialised = bool(payload.get("devices_initialised"))
        if payload.get("shutdown"):
            self.devices_initialised = False
        self._sync_controls_enabled()
        if not self.devices_initialised:
            self.status_label.setText("Run Initialise Devices before using autofocus controls.")
        elif self.status_label.text().startswith("Run Initialise Devices"):
            self.status_label.setText("Refresh autofocus to read status.")

    def _ensure_devices_initialised(self) -> bool:
        if self.devices_initialised:
            return True
        self.status_label.setText("Run Initialise Devices before using autofocus controls.")
        return False

    def _sync_controls_enabled(self) -> None:
        for widget in (
            self.refresh_button,
            self.apply_config_button,
            self.run_calibration_button,
            self.lock_after_calibration_checkbox,
            self.lock_button,
            self.unlock_button,
            *self.config_inputs.values(),
        ):
            widget.setEnabled(self.devices_initialised)

    def _show_error(self, error: str) -> None:
        if "autofocus" in error.lower() or self.status_label.text().endswith("autofocus."):
            self.status_label.setText(error)

    @staticmethod
    def _format_status(status_name: str | None) -> str:
        if not status_name:
            return "-"
        labels = {
            "IDLE": "Idle",
            "READY": "Ready",
            "DIM": "Dim",
            "OUT_OF_FOCUS": "Out of focus",
            "IN_FOCUS": "In focus",
            "INHIBIT": "Inhibit",
            "ERROR": "Error",
            "LOG_CAL": "Log calibration",
        }
        return labels.get(status_name, str(status_name))
