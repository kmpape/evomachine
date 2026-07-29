from __future__ import annotations

from PyQt5.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from evomachine.gui.panels.common import muted_label
from evomachine.gui.panels.config_dialog import ConfigDialog, ConfigFieldSpec


class AutofocusPanel(QGroupBox):
    """Hardware autofocus controls."""

    CONFIG_FIELDS = (
        ConfigFieldSpec("Averaging", "averaging", 5, kind="int", minimum=0, maximum=99),
        ConfigFieldSpec("LED intensity", "led_intensity", 70, kind="int", minimum=2, maximum=100),
        ConfigFieldSpec(
            "Lock range",
            "lock_range",
            0.1,
            kind="float",
            minimum=0.001,
            maximum=0.999,
            decimals=3,
            single_step=0.1,
        ),
        ConfigFieldSpec("Loop gain", "loop_gain", 10, kind="int", minimum=1, maximum=100),
        ConfigFieldSpec("Update rate", "update_rate", 10, kind="int", minimum=0, maximum=1000),
        ConfigFieldSpec(
            "Objective NA",
            "objective_na",
            0.9,
            kind="float",
            minimum=0.01,
            maximum=10.0,
            decimals=3,
            single_step=0.1,
            editable=False,
        ),
        ConfigFieldSpec(
            "Minimum SNR",
            "min_snr",
            2.0,
            kind="float",
            minimum=0.0,
            maximum=1000.0,
            decimals=2,
            single_step=0.1,
        ),
        ConfigFieldSpec("Minimum error", "min_error", 100, kind="int", minimum=0, maximum=100000),
    )

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
        self.config_values = self._default_config_values()
        self.lock_after_calibration_checkbox = QCheckBox("Lock after calibration")
        self.lock_after_calibration_checkbox.setChecked(False)

        self.refresh_button = QPushButton("Refresh")
        self.configure_button = QPushButton("Configure")
        self.run_calibration_button = QPushButton("Run Calibration")
        self.lock_button = QPushButton("Lock")
        self.unlock_button = QPushButton("Unlock")

        buttons = QGridLayout()
        buttons.addWidget(self.refresh_button, 0, 0)
        buttons.addWidget(self.configure_button, 0, 1)
        buttons.addWidget(self.run_calibration_button, 1, 0)
        buttons.addWidget(self.lock_after_calibration_checkbox, 1, 1)
        buttons.addWidget(self.lock_button, 2, 0)
        buttons.addWidget(self.unlock_button, 2, 1)

        layout = QVBoxLayout()
        layout.addWidget(self.status_label)
        layout.addWidget(self.state_label)
        layout.addWidget(self.locked_label)
        layout.addWidget(self.note_label)
        layout.addLayout(buttons)
        self.setLayout(layout)

        self.refresh_button.clicked.connect(self.controller.refresh_autofocus)
        self.configure_button.clicked.connect(self._open_config_dialog)
        self.run_calibration_button.clicked.connect(self._run_calibration)
        self.lock_button.clicked.connect(self._lock_autofocus)
        self.unlock_button.clicked.connect(self._unlock_autofocus)
        self.controller.autofocus_status_received.connect(self.update_status)
        self.controller.lifecycle_status_received.connect(self.update_lifecycle_status)
        self.controller.response_error.connect(self._show_error)
        self._sync_controls_enabled()

    @classmethod
    def _default_config_values(cls) -> dict[str, float | int]:
        return {field.key: field.value for field in cls.CONFIG_FIELDS}

    def _config_payload(self, *, runtime_only: bool = False) -> dict[str, float | int]:
        if not runtime_only:
            return dict(self.config_values)
        editable_keys = {field.key for field in self.CONFIG_FIELDS if field.editable}
        return {
            key: value
            for key, value in self.config_values.items()
            if key in editable_keys
        }

    def _open_config_dialog(self) -> None:
        if not self._ensure_devices_initialised():
            return
        dialog = ConfigDialog(
            title="Hardware Autofocus Configuration",
            fields=self._config_fields(),
            parent=self,
        )
        if dialog.exec_() != dialog.Accepted:
            return
        self._update_config_values({
            key: value
            for key, value in dialog.values().items()
            if key in {field.key for field in self.CONFIG_FIELDS if field.editable}
        })
        self._apply_config()

    def _config_fields(self) -> list[ConfigFieldSpec]:
        return [
            ConfigFieldSpec(
                label=field.label,
                key=field.key,
                value=self.config_values.get(field.key, field.value),
                kind=field.kind,
                minimum=field.minimum,
                maximum=field.maximum,
                decimals=field.decimals,
                single_step=field.single_step,
                editable=field.editable,
            )
            for field in self.CONFIG_FIELDS
        ]

    def _apply_config(self) -> None:
        if not self._ensure_devices_initialised():
            return
        self.status_label.setText("Applying autofocus configuration.")
        self.controller.configure_autofocus(config=self._config_payload(runtime_only=True))

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
            self._update_config_values(config)
        self._sync_controls_enabled()

    def _update_config_values(self, config: dict) -> None:
        for key in self.config_values:
            if key not in config or config[key] is None:
                continue
            self.config_values[key] = config[key]

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
            self.configure_button,
            self.run_calibration_button,
            self.lock_after_calibration_checkbox,
            self.lock_button,
            self.unlock_button,
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
