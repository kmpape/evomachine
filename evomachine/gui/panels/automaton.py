from __future__ import annotations

from PyQt5.QtWidgets import QGridLayout, QGroupBox, QLabel, QPushButton, QVBoxLayout, QWidget


class AutomatonPanel(QGroupBox):
    """Top-level automaton lifecycle controls."""

    def __init__(self, controller, parent: QWidget | None = None):
        super().__init__("Automaton", parent)
        self.controller = controller
        self.status_label = QLabel("Disconnected")
        self.initialisation_label = QLabel("Run Initialise Devices before using hardware controls.")
        self.initialisation_label.setWordWrap(True)
        self.initialisation_label.setStyleSheet("color: #aab2bd;")

        self.ping_button = QPushButton("Ping")
        self.initialise_button = QPushButton("Initialise Devices")
        self.stop_button = QPushButton("Stop")
        self.shutdown_button = QPushButton("Shutdown Automaton")
        self.shutdown_button.setToolTip("Terminal shutdown. Restart the GUI to reconnect afterwards.")

        buttons = QGridLayout()
        buttons.addWidget(self.ping_button, 0, 0)
        buttons.addWidget(self.initialise_button, 0, 1)
        buttons.addWidget(self.stop_button, 1, 0)
        buttons.addWidget(self.shutdown_button, 1, 1)

        layout = QVBoxLayout()
        layout.addWidget(self.status_label)
        layout.addWidget(self.initialisation_label)
        layout.addLayout(buttons)
        self.setLayout(layout)

        self.ping_button.clicked.connect(self.controller.ping)
        self.initialise_button.clicked.connect(self.controller.initialise_devices)
        self.stop_button.clicked.connect(self.controller.stop)
        self.shutdown_button.clicked.connect(self.controller.shutdown_automaton)
        self.controller.response_error.connect(self._show_error)
        self.controller.lifecycle_status_received.connect(self._show_status)

    def _show_error(self, error: str) -> None:
        self.status_label.setText(error)
        if self._is_connection_error(error):
            self.initialisation_label.setText("Automaton connection lost. Restart the GUI to reconnect.")
            self._set_controls_enabled(False)
            return
        self.initialisation_label.setText("Check automaton status before using hardware controls.")

    def _show_status(self, payload: dict) -> None:
        self.status_label.setText(
            f"devices: {payload.get('devices_initialised')}, "
            f"strategy: {payload.get('strategy_active')}, stopped: {payload.get('stopped')}, "
            f"shutdown: {payload.get('shutdown')}"
        )
        if payload.get("shutdown"):
            self.initialisation_label.setText("Automaton has shut down. Restart the GUI to reconnect.")
            self._set_controls_enabled(False)
        elif payload.get("devices_initialised"):
            self.initialisation_label.setText("Devices initialised. Hardware controls are ready.")
            self._set_controls_enabled(True)
        else:
            self.initialisation_label.setText("Run Initialise Devices before using hardware controls.")
            self._set_controls_enabled(True)

    def _set_controls_enabled(self, enabled: bool) -> None:
        self.ping_button.setEnabled(enabled)
        self.initialise_button.setEnabled(enabled)
        self.stop_button.setEnabled(enabled)
        self.shutdown_button.setEnabled(enabled)

    @staticmethod
    def _is_connection_error(error: str) -> bool:
        lowered = error.lower()
        return "connection" in lowered or "socket closed" in lowered or "broken pipe" in lowered
