from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
import threading
from typing import Any
from uuid import uuid4


OperationReporter = Callable[[float, str], None]
OperationRunner = Callable[[threading.Event, OperationReporter], None]


@dataclass
class GuiOperation:
    """Thread-safe state for one background hardware operation."""

    operation_id: str
    kind: str
    state: str
    progress: float
    message: str
    error: str | None
    started_at: str
    finished_at: str | None
    cancel_event: threading.Event


class GuiOperationManager:
    """Run at most one long hardware operation while keeping GUI RPC responsive."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._operations: dict[str, GuiOperation] = {}
        self._active_kind: str | None = None

    def start(self, kind: str, runner: OperationRunner) -> dict[str, Any]:
        with self._lock:
            if self._active_kind is not None:
                active = self._operations[self._active_kind]
                raise RuntimeError(
                    f"Cannot start {kind}: {active.kind} operation {active.operation_id} is running."
                )
            operation = GuiOperation(
                operation_id=str(uuid4()),
                kind=kind,
                state="running",
                progress=0.0,
                message="Starting.",
                error=None,
                started_at=_utc_now(),
                finished_at=None,
                cancel_event=threading.Event(),
            )
            self._operations[kind] = operation
            self._active_kind = kind
            thread = threading.Thread(
                target=self._run,
                args=(operation, runner),
                name=f"GuiOperation-{kind}",
                daemon=True,
            )
            thread.start()
            return self._snapshot_locked(operation)

    def status(self, kind: str) -> dict[str, Any] | None:
        with self._lock:
            operation = self._operations.get(kind)
            return None if operation is None else self._snapshot_locked(operation)

    def cancel(self, kind: str) -> dict[str, Any]:
        with self._lock:
            operation = self._operations.get(kind)
            if operation is None:
                raise RuntimeError(f"No {kind} operation has been started.")
            if operation.state != "running":
                return self._snapshot_locked(operation)
            operation.cancel_event.set()
            operation.message = "Cancellation requested; waiting for a safe stopping point."
            return self._snapshot_locked(operation)

    def active(self) -> dict[str, Any] | None:
        with self._lock:
            if self._active_kind is None:
                return None
            return self._snapshot_locked(self._operations[self._active_kind])

    def _run(self, operation: GuiOperation, runner: OperationRunner) -> None:
        def report(progress: float, message: str) -> None:
            with self._lock:
                if operation.state != "running":
                    return
                operation.progress = max(0.0, min(1.0, float(progress)))
                operation.message = str(message)

        try:
            runner(operation.cancel_event, report)
        except Exception as error:
            with self._lock:
                operation.state = "cancelled" if operation.cancel_event.is_set() else "failed"
                operation.error = None if operation.cancel_event.is_set() else f"{type(error).__name__}: {error}"
                operation.message = "Cancelled." if operation.cancel_event.is_set() else "Failed."
                self._finish_locked(operation)
            return
        with self._lock:
            if operation.cancel_event.is_set():
                operation.state = "cancelled"
                operation.message = "Cancelled."
            else:
                operation.state = "completed"
                operation.progress = 1.0
                operation.message = "Completed."
            self._finish_locked(operation)

    def _finish_locked(self, operation: GuiOperation) -> None:
        operation.finished_at = _utc_now()
        if self._active_kind == operation.kind:
            self._active_kind = None

    @staticmethod
    def _snapshot_locked(operation: GuiOperation) -> dict[str, Any]:
        return {
            "operation_id": operation.operation_id,
            "kind": operation.kind,
            "state": operation.state,
            "progress": operation.progress,
            "message": operation.message,
            "error": operation.error,
            "started_at": operation.started_at,
            "finished_at": operation.finished_at,
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
