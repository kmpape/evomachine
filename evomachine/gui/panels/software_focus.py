from __future__ import annotations

from PyQt5.QtWidgets import QGridLayout, QGroupBox, QLabel, QPushButton, QVBoxLayout, QWidget

from evomachine.gui.panels.config_dialog import ConfigDialog, ConfigFieldSpec


class SoftwareFocusPanel(QGroupBox):
    """Software focus controls."""

    def __init__(self, controller, parent: QWidget | None = None):
        super().__init__("Software Focus", parent)
        self.controller = controller
        self.devices_initialised = False
        self.strategy_running = False
        self._latest_config: dict = {}
        self.status_label = QLabel("Run Initialise Devices before using software focus controls.")
        self.status_label.setWordWrap(True)
        self.config_label = QLabel("config: -")
        self.result_label = QLabel("result: -")

        self.refresh_button = QPushButton("Refresh")
        self.configure_button = QPushButton("Configure")
        self.run_button = QPushButton("Run Software Focus")

        buttons = QGridLayout()
        buttons.addWidget(self.refresh_button, 0, 0)
        buttons.addWidget(self.configure_button, 0, 1)
        buttons.addWidget(self.run_button, 1, 0, 1, 2)

        layout = QVBoxLayout()
        layout.addWidget(self.status_label)
        layout.addWidget(self.config_label)
        layout.addWidget(self.result_label)
        layout.addLayout(buttons)
        self.setLayout(layout)

        self.refresh_button.clicked.connect(self.controller.refresh_software_focus)
        self.configure_button.clicked.connect(self._open_config_dialog)
        self.run_button.clicked.connect(self._run_software_focus)
        self.controller.software_focus_status_received.connect(self.update_status)
        self.controller.lifecycle_status_received.connect(self.update_lifecycle_status)
        self.controller.strategy_status_received.connect(self.update_strategy_status)
        self.controller.response_error.connect(self._show_error)
        self._sync_controls_enabled()

    def _run_software_focus(self) -> None:
        if not self._ensure_devices_initialised():
            return
        self.status_label.setText("Running software focus.")
        self.controller.run_software_focus()

    def _open_config_dialog(self) -> None:
        if not self._ensure_devices_initialised():
            return
        dialog = ConfigDialog(
            title="Software Focus Configuration",
            fields=self._config_fields(),
            parent=self,
        )
        dialog.exec_()

    def update_status(self, payload: dict) -> None:
        self.status_label.setText(f"available: {payload.get('available')}")
        config = payload.get("config") or {}
        self._latest_config = dict(config)
        self.config_label.setText(
            f"config: range {config.get('rel_range', '-')}, step {config.get('step_size', '-')}, "
            f"{self._format_enum_name(config.get('algorithm'))}"
        )
        result = payload.get("last_result")
        if isinstance(result, dict):
            self.result_label.setText(self._result_text(result))
        else:
            self.result_label.setText("result: -")
        self._sync_controls_enabled()

    def update_lifecycle_status(self, payload: dict) -> None:
        self.devices_initialised = bool(payload.get("devices_initialised"))
        if payload.get("shutdown"):
            self.devices_initialised = False
        self._sync_controls_enabled()
        if not self.devices_initialised:
            self.status_label.setText(
                "Run Initialise Devices before using software focus controls."
            )
        elif self.status_label.text().startswith("Run Initialise Devices"):
            self.status_label.setText("Refresh software focus to read status.")

    def update_strategy_status(self, payload: dict) -> None:
        self.strategy_running = bool(payload.get("running"))
        self._sync_controls_enabled()

    def _ensure_devices_initialised(self) -> bool:
        if self.devices_initialised:
            return True
        self.status_label.setText("Run Initialise Devices before using software focus controls.")
        return False

    def _sync_controls_enabled(self) -> None:
        self.refresh_button.setEnabled(self.devices_initialised)
        manual_controls_enabled = self.devices_initialised and not self.strategy_running
        self.configure_button.setEnabled(manual_controls_enabled)
        self.run_button.setEnabled(manual_controls_enabled)

    def _show_error(self, error: str) -> None:
        if "software focus" in error.lower() or self.status_label.text().startswith(
            "Running software focus"
        ):
            self.status_label.setText(error)

    def _result_text(self, result: dict) -> str:
        focus_status = self._status_name(result.get("focus_status"))
        best_coordinate = result.get("best_coordinate")
        best_z = best_coordinate.get("z") if isinstance(best_coordinate, dict) else "-"
        z_points = result.get("z_points", "-")
        return f"result: {focus_status}, best z {best_z}, points {z_points}"

    @staticmethod
    def _status_name(value: object) -> str:
        if isinstance(value, dict):
            return SoftwareFocusPanel._format_enum_name(value.get("name"))
        return SoftwareFocusPanel._format_enum_name(value)

    @staticmethod
    def _format_enum_name(value: object) -> str:
        if not value:
            return "-"
        return str(value).replace("_", " ").lower()

    def _config_fields(self) -> list[ConfigFieldSpec]:
        return [
            ConfigFieldSpec(
                "Relative range", "rel_range", self._latest_config.get("rel_range"), editable=False
            ),
            ConfigFieldSpec(
                "Step size", "step_size", self._latest_config.get("step_size"), editable=False
            ),
            ConfigFieldSpec(
                "Algorithm",
                "algorithm",
                self._format_enum_name(self._latest_config.get("algorithm")),
                editable=False,
            ),
        ]
