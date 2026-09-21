from __future__ import annotations

from html import escape

from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QFontDatabase
from PyQt5.QtWidgets import QPlainTextEdit, QVBoxLayout, QWidget


class ApplicationLogPanel(QWidget):
    """Display a bounded, live view of informational and higher-level logs."""

    def __init__(self, controller, history_limit: int = 200, parent: QWidget | None = None):
        super().__init__(parent)
        if not isinstance(history_limit, int) or isinstance(history_limit, bool) or history_limit < 1:
            raise ValueError("ApplicationLogPanel history_limit must be a positive integer.")
        self.controller = controller
        self.latest_sequence = 0

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.log_view.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
        self.log_view.document().setMaximumBlockCount(history_limit)

        layout = QVBoxLayout()
        layout.addWidget(self.log_view)
        self.setLayout(layout)

        self.controller.logs_received.connect(self.update_logs)

        self.poll_timer = QTimer(self)
        self.poll_timer.setInterval(500)
        self.poll_timer.timeout.connect(self.refresh)
        self.poll_timer.start()
        self.refresh()

    def refresh(self) -> None:
        self.controller.refresh_logs(after_sequence=self.latest_sequence)

    def update_logs(self, payload: dict) -> None:
        records = payload.get("records", [])
        for record in records:
            sequence = int(record["sequence"])
            if sequence <= self.latest_sequence:
                continue
            self.log_view.appendHtml(self._render_record(record))
            self.latest_sequence = sequence
        self.latest_sequence = max(
            self.latest_sequence,
            int(payload.get("latest_sequence", self.latest_sequence)),
        )
        if records:
            scrollbar = self.log_view.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())

    @staticmethod
    def _render_record(record: dict) -> str:
        level = str(record.get("level", "INFO")).upper()
        colour = {
            "WARNING": "#d18b00",
            "ERROR": "#d33c3c",
            "CRITICAL": "#d33c3c",
        }.get(level, "#808080")
        timestamp = escape(str(record.get("timestamp", "-")))
        logger_name = escape(str(record.get("logger", "")))
        message = escape(str(record.get("message", ""))).replace("\n", "<br>")
        return (
            f'<span style="color:{colour}">[{timestamp}] <b>{escape(level)}</b> '
            f'{logger_name}: {message}</span>'
        )
