from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from evomachine.gui.protocol import GuiCommandType


@dataclass(frozen=True)
class ControllerStatus:
    """Describe one flat Automaton or peripheral status row.

    Args:
        name: User-facing component name.
        connected: Whether the row should display a green status dot.
        description: Detailed status text displayed beside the name.
    """

    name: str
    connected: bool
    description: str = ""


class StatusDot(QLabel):
    """Display an accessible red or green component-status indicator."""

    def __init__(self, connected: bool = False, parent: QWidget | None = None):
        """Create a status dot.

        Args:
            connected: Initial responding state shown as green when true.
            parent: Optional owning Qt widget.
        """
        super().__init__(parent)
        self.setFixedSize(12, 12)
        self.set_connected(connected)

    def set_connected(self, connected: bool) -> None:
        """Update colour, accessible text, and tooltip.

        Args:
            connected: True for green and False for red.
        """
        color = "#2fb344" if connected else "#d64545"
        border = "#86d993" if connected else "#ee8f8f"
        self.setAccessibleName("Connected" if connected else "Disconnected")
        self.setToolTip("Responding" if connected else "Not responding or unavailable")
        self.setStyleSheet(
            "QLabel {"
            f"background-color: {color};"
            f"border: 1px solid {border};"
            "border-radius: 6px;"
            "}"
        )


class ControllerStatusRow(QFrame):
    """Show one component as an unframed dot, name, and description row."""

    def __init__(self, status: ControllerStatus, parent: QWidget | None = None):
        """Create an unframed status row.

        Args:
            status: Initial component state rendered by the row.
            parent: Optional owning Qt widget.
        """
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
        text_layout.addWidget(self.description_label)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 3, 4, 3)
        layout.setSpacing(8)
        layout.addWidget(self.dot, 0, Qt.AlignTop)
        layout.addLayout(text_layout)
        self.setFrameShape(QFrame.NoFrame)

    def update_status(self, status: ControllerStatus) -> None:
        """Replace displayed component state.

        Args:
            status: Latest component name, health, and description.
        """
        self.name_label.setText(status.name)
        self.description_label.setText(status.description)
        self.dot.set_connected(status.connected)


class PeripheralControllerStatusPanel(QWidget):
    """Display flat Automaton lifecycle and peripheral-controller status."""

    def __init__(
        self,
        controller,
        statuses: Iterable[ControllerStatus] | None = None,
        refresh_interval_ms: int = 20 * 60 * 1000,
        parent: QWidget | None = None,
    ):
        """Create the combined status and lifecycle panel.

        Args:
            controller: Shared GUI controller used for status requests and signals.
            statuses: Optional initial peripheral status records.
            refresh_interval_ms: Milliseconds between automatic Ping requests.
            parent: Optional owning Qt widget.
        """
        super().__init__(parent)
        self.controller = controller
        self.automaton_row = ControllerStatusRow(
            ControllerStatus("Automaton", False, "No status response received.")
        )
        self.initialise_button = QPushButton("Initialise")
        self.stop_button = QPushButton("Stop")
        self.shutdown_button = QPushButton("Shutdown")
        self.refresh_button = QPushButton("Refresh Status")
        self.refresh_button.setToolTip(
            "Ping the Automaton and refresh both lifecycle and peripheral-controller status."
        )
        self.status_label = QLabel("Last checked: -")
        self.status_label.setStyleSheet("color: #aab2bd;")
        self.rows_layout = QVBoxLayout()
        self.rows_layout.setContentsMargins(0, 0, 0, 0)
        self.rows_layout.setSpacing(3)

        actions = QVBoxLayout()
        actions.addWidget(self.initialise_button)
        actions.addWidget(self.stop_button)
        actions.addWidget(self.shutdown_button)
        actions.addWidget(self.refresh_button)
        layout = QVBoxLayout(self)
        layout.addLayout(actions)
        layout.addWidget(self.automaton_row)
        layout.addLayout(self.rows_layout)
        layout.addWidget(self.status_label)
        layout.addStretch(1)

        self.initialise_button.clicked.connect(self.controller.initialise_devices)
        self.stop_button.clicked.connect(self.controller.stop)
        self.shutdown_button.clicked.connect(self.controller.shutdown_automaton)
        self.refresh_button.clicked.connect(self.controller.ping)
        self.controller.controller_status_received.connect(self.update_status)
        self.controller.lifecycle_status_received.connect(self.update_lifecycle_status)
        self.controller.request_error.connect(self._show_request_error)
        self.set_statuses(statuses or ())

        self.refresh_timer = QTimer(self)
        self.refresh_timer.setInterval(refresh_interval_ms)
        self.refresh_timer.timeout.connect(self.controller.ping)
        self.refresh_timer.start()
        QTimer.singleShot(0, self.controller.ping)

    def set_statuses(self, statuses: Iterable[ControllerStatus]) -> None:
        """Replace the peripheral rows while retaining the Automaton row.

        Args:
            statuses: Current peripheral-controller display records.
        """
        while self.rows_layout.count():
            item = self.rows_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        for status in statuses:
            self.rows_layout.addWidget(ControllerStatusRow(status))

    def update_status(self, payload: dict) -> None:
        """Render peripheral status returned by Ping or controller status.

        Args:
            payload: Response containing controller entries and check time.
        """
        controllers = payload.get("controllers", [])
        self.set_statuses(
            ControllerStatus(
                name=item.get("name") or item.get("type") or "Peripheral controller",
                connected=bool(item.get("connected")),
                description=self._description(item),
            )
            for item in controllers
        )
        self.status_label.setText(f"Last checked: {self._format_checked_at(payload.get('checked_at'))}")

    def update_lifecycle_status(self, payload: dict) -> None:
        """Render Automaton health and lifecycle values.

        Args:
            payload: Lifecycle payload emitted after a successful backend response.
        """
        shutdown = bool(payload.get("shutdown"))
        description = (
            f"devices initialised: {bool(payload.get('devices_initialised'))}, "
            f"strategy running: {bool(payload.get('strategy_active'))}, "
            f"stopped: {bool(payload.get('stopped'))}, shutdown: {shutdown}"
        )
        self.automaton_row.update_status(
            ControllerStatus("Automaton", not shutdown, description)
        )
        self.initialise_button.setEnabled(not shutdown)
        self.stop_button.setEnabled(not shutdown)
        self.shutdown_button.setEnabled(not shutdown)
        self.refresh_button.setEnabled(True)

    def _show_request_error(self, command: object, error: str) -> None:
        """Mark Automaton unavailable after a failed status refresh.

        Args:
            command: Failed GUI command.
            error: User-facing failure text.
        """
        if command is not GuiCommandType.PING:
            return
        self.automaton_row.update_status(ControllerStatus("Automaton", False, error))
        self.status_label.setText("Last checked: failed")
        self.refresh_button.setEnabled(True)

    @staticmethod
    def _description(item: dict) -> str:
        """Return compact controller ownership and health details."""
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
        """Format an ISO status time as a short local display value."""
        if not isinstance(value, str) or not value:
            return "-"
        try:
            return datetime.fromisoformat(value).strftime("%H:%M:%S")
        except ValueError:
            return value
