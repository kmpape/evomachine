from __future__ import annotations

from collections.abc import Iterable
from decimal import Decimal, ROUND_HALF_UP
from math import isfinite

from PyQt5.QtGui import QWheelEvent
from PyQt5.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


def format_micrometre_value(value: object) -> str:
    """Format a GUI measurement to the nearest whole micrometre."""
    if not isinstance(value, int | float) or isinstance(value, bool):
        return "-"
    numeric_value = float(value)
    if not isfinite(numeric_value):
        return "-"
    return str(int(Decimal(str(numeric_value)).quantize(Decimal("1"), rounding=ROUND_HALF_UP)))


class NoWheelComboBox(QComboBox):
    """Combo box that lets wheel events continue to a surrounding scroll area."""

    def wheelEvent(self, event: QWheelEvent) -> None:
        event.ignore()


class NoWheelSpinBox(QSpinBox):
    """Integer input that cannot be changed accidentally while scrolling."""

    def wheelEvent(self, event: QWheelEvent) -> None:
        event.ignore()


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    """Float input that cannot be changed accidentally while scrolling."""

    def wheelEvent(self, event: QWheelEvent) -> None:
        event.ignore()


def muted_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setStyleSheet("color: #aab2bd;")
    return label


class DisabledPanelShell(QGroupBox):
    """Disabled panel shell for controls that do not have GUI commands yet."""

    def __init__(
            self,
            title: str,
            actions: Iterable[str] = (),
            note: str = "Backend command needed",
            parent: QWidget | None = None,
    ):
        super().__init__(title, parent)
        layout = QVBoxLayout()
        for action in actions:
            button = QPushButton(action)
            button.setEnabled(False)
            layout.addWidget(button)
        layout.addWidget(muted_label(note))
        self.setLayout(layout)
