from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import logging


class GuiLogBufferHandler(logging.Handler):
    """Retain a bounded, structured view of recent application log records."""

    def __init__(self, capacity: int = 200, level: int = logging.INFO):
        if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity < 1:
            raise ValueError("GuiLogBufferHandler capacity must be a positive integer.")
        super().__init__(level=level)
        self.capacity = capacity
        self._records: deque[dict[str, object]] = deque(maxlen=capacity)
        self._sequence = 0

    def emit(self, record: logging.LogRecord) -> None:
        self._sequence += 1
        formatter = self.formatter
        timestamp = (
            formatter.formatTime(record)
            if formatter is not None
            else datetime.fromtimestamp(record.created, tz=timezone.utc).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )
        message = record.getMessage()
        if record.exc_info is not None and formatter is not None:
            message = f"{message}\n{formatter.formatException(record.exc_info)}"
        self._records.append(
            {
                "sequence": self._sequence,
                "timestamp": timestamp,
                "level": record.levelname,
                "logger": record.name,
                "message": message,
            }
        )

    def records_after(self, sequence: int = 0) -> tuple[dict[str, object], ...]:
        """Return retained records newer than the supplied sequence number."""
        if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 0:
            raise ValueError("Log sequence must be a non-negative integer.")
        self.acquire()
        try:
            return tuple(dict(record) for record in self._records if record["sequence"] > sequence)
        finally:
            self.release()

    @property
    def latest_sequence(self) -> int:
        self.acquire()
        try:
            return self._sequence
        finally:
            self.release()

    def clear(self) -> None:
        """Clear retained records while keeping sequence numbers monotonic."""
        self.acquire()
        try:
            self._records.clear()
        finally:
            self.release()
