from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget


@dataclass(frozen=True)
class ControllerStatus:
    """Small display model for one peripheral controller row."""

    name: str
    connected: bool
    description: str = ""


class StatusDot(QLabel):
    """Round red/green peripheral controller status indicator."""

    def __init__(self, connected: bool = False, parent: QWidget | None = None):
        super().__init__(parent)
        self.setFixedSize(12, 12)
        self.set_connected(connected)

    def set_connected(self, connected: bool) -> None:
        color = "#2fb344" if connected else "#d64545"
        border = "#86d993" if connected else "#ee8f8f"
        self.setAccessibleName("Connected" if connected else "Disconnected")
        self.setToolTip("Initialised / connected" if connected else "Not initialised / disconnected")
        self.setStyleSheet(
            "QLabel {"
            f"background-color: {color};"
            f"border: 1px solid {border};"
            "border-radius: 6px;"
            "}"
        )


class ControllerStatusRow(QFrame):
    """One compact row showing a peripheral controller and its status."""

    def __init__(self, status: ControllerStatus, parent: QWidget | None = None):
        super().__init__(parent)
        self.dot = StatusDot(status.connected)
        self.name_label = QLabel(status.name)
        self.description_label = QLabel(status.description)
        self.name_label.setWordWrap(True)
        self.description_label.setWordWrap(True)
        self.description_label.setStyleSheet("color: #aab2bd;")

        text_layout = QVBoxLayout()
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(2)
        text_layout.addWidget(self.name_label)
        if status.description:
            text_layout.addWidget(self.description_label)

        layout = QHBoxLayout()
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(8)
        layout.addWidget(self.dot, 0, Qt.AlignTop)
        layout.addLayout(text_layout)
        self.setLayout(layout)

        self.setFrameShape(QFrame.StyledPanel)
        self.set_connected(status.connected)

    def set_connected(self, connected: bool) -> None:
        self.dot.set_connected(connected)


class PeripheralControllerStatusPanel(QWidget):
    """Panel showing liveness for connected peripheral controllers."""

    def __init__(
            self,
            controller,
            statuses: Iterable[ControllerStatus] | None = None,
            refresh_interval_ms: int = 20 * 60 * 1000,
            parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.controller = controller

        title = QLabel("Peripheral Controllers")
        title.setStyleSheet("font-weight: 600;")
        self.legend_label = QLabel("Key: green = initialised + alive; red = not initialised or not alive.")
        self.legend_label.setWordWrap(True)
        self.legend_label.setStyleSheet("color: #aab2bd;")
        self.status_label = QLabel("Refresh controller status.")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #aab2bd;")
        self.refresh_button = QPushButton("Refresh")
        self.rows_layout = QVBoxLayout()
        self.rows_layout.setContentsMargins(0, 0, 0, 0)
        self.rows_layout.setSpacing(6)

        layout = QVBoxLayout()
        layout.addWidget(title)
        layout.addWidget(self.legend_label)
        layout.addWidget(self.status_label)
        layout.addWidget(self.refresh_button)
        layout.addLayout(self.rows_layout)
        layout.addStretch(1)
        self.setLayout(layout)

        self.refresh_button.clicked.connect(self.controller.refresh_controller_status)
        self.controller.controller_status_received.connect(self.update_status)
        self.controller.response_error.connect(self._show_error)
        self.set_statuses(statuses or ())

        self.refresh_timer = QTimer(self)
        self.refresh_timer.setInterval(refresh_interval_ms)
        self.refresh_timer.timeout.connect(self.controller.refresh_controller_status)
        self.refresh_timer.start()
        QTimer.singleShot(0, self.controller.refresh_controller_status)

    def set_statuses(self, statuses: Iterable[ControllerStatus]) -> None:
        while self.rows_layout.count():
            item = self.rows_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        statuses = list(statuses)
        if not statuses:
            self.rows_layout.addWidget(QLabel("No peripheral controllers reported."))
            return
        for status in statuses:
            self.rows_layout.addWidget(ControllerStatusRow(status))

    def update_status(self, payload: dict) -> None:
        controllers = payload.get("controllers", [])
        self.set_statuses(
            ControllerStatus(
                name=item.get("name") or item.get("type") or "Peripheral controller",
                connected=bool(item.get("connected")),
                description=self._description(item),
            )
            for item in controllers
        )
        checked_at = payload.get("checked_at")
        self.status_label.setText(f"Last checked: {self._format_checked_at(checked_at)}")

    def _show_error(self, error: str) -> None:
        if "controller" in error.lower() or "controllers" in error.lower():
            self.status_label.setText(error)

    @staticmethod
    def _description(item: dict) -> str:
        owners = item.get("owners") or []
        details = []
        if owners:
            details.append(", ".join(str(owner) for owner in owners))
        details.append(
            f"initialised: {bool(item.get('is_initialised'))}, alive: {bool(item.get('is_alive'))}"
        )
        if item.get("error"):
            details.append(str(item["error"]))
        return " | ".join(details)

    @staticmethod
    def _format_checked_at(value: object) -> str:
        if not isinstance(value, str) or not value:
            return "-"
        try:
            return datetime.fromisoformat(value).strftime("%H:%M:%S")
        except ValueError:
            return value
