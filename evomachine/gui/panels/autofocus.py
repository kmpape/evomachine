from __future__ import annotations

from PyQt5.QtWidgets import QGridLayout, QGroupBox, QLabel, QPushButton, QVBoxLayout, QWidget

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
        self.note_label = muted_label("Uses the backend default autofocus configuration.")
        self.note_label.setWordWrap(True)

        self.refresh_button = QPushButton("Refresh")
        self.lock_button = QPushButton("Lock")
        self.unlock_button = QPushButton("Unlock")

        buttons = QGridLayout()
        buttons.addWidget(self.refresh_button, 0, 0)
        buttons.addWidget(self.lock_button, 0, 1)
        buttons.addWidget(self.unlock_button, 1, 0, 1, 2)

        layout = QVBoxLayout()
        layout.addWidget(self.status_label)
        layout.addWidget(self.state_label)
        layout.addWidget(self.locked_label)
        layout.addWidget(self.note_label)
        layout.addLayout(buttons)
        self.setLayout(layout)

        self.refresh_button.clicked.connect(self.controller.refresh_autofocus)
        self.lock_button.clicked.connect(self._lock_autofocus)
        self.unlock_button.clicked.connect(self._unlock_autofocus)
        self.controller.autofocus_status_received.connect(self.update_status)
        self.controller.lifecycle_status_received.connect(self.update_lifecycle_status)
        self.controller.response_error.connect(self._show_error)
        self._sync_controls_enabled()

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
        self._sync_controls_enabled()

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
