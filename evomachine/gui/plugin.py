from __future__ import annotations

from PyQt5.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

from evomachine.gui.controller import EvoMachineGuiController
from evomachine.gui.panels.leds import LedManagerPanel
from evomachine.gui.panels.stage import StagePanel


class EvoMachineNapariWidget(QWidget):
    """Napari dock widget containing modular evomachine peripheral panels."""

    def __init__(self, napari_viewer=None):
        super().__init__()
        self.viewer = napari_viewer
        self.controller = EvoMachineGuiController()

        self.status_label = QLabel("Disconnected")
        ping_button = QPushButton("Ping")
        initialise_button = QPushButton("Initialise Devices")
        stop_button = QPushButton("Stop")
        shutdown_button = QPushButton("Shutdown Automaton")

        layout = QVBoxLayout()
        layout.addWidget(self.status_label)
        layout.addWidget(ping_button)
        layout.addWidget(initialise_button)
        layout.addWidget(stop_button)
        layout.addWidget(shutdown_button)
        layout.addWidget(StagePanel(controller=self.controller))
        layout.addWidget(LedManagerPanel(controller=self.controller))
        layout.addStretch(1)
        self.setLayout(layout)

        ping_button.clicked.connect(self.controller.ping)
        initialise_button.clicked.connect(self.controller.initialise_devices)
        stop_button.clicked.connect(self.controller.stop)
        shutdown_button.clicked.connect(self.controller.shutdown_automaton)
        self.controller.response_error.connect(self._show_error)
        self.controller.lifecycle_status_received.connect(self._show_status)

    def closeEvent(self, event) -> None:  # noqa: N802
        self.controller.close()
        super().closeEvent(event)

    def _show_error(self, error: str) -> None:
        self.status_label.setText(error)

    def _show_status(self, payload: dict) -> None:
        self.status_label.setText(
            f"devices: {payload.get('devices_initialised')}, "
            f"strategy: {payload.get('strategy_active')}, shutdown: {payload.get('shutdown')}"
        )

