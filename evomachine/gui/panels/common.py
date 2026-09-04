from __future__ import annotations

from collections.abc import Iterable

from PyQt5.QtWidgets import QGroupBox, QLabel, QPushButton, QVBoxLayout, QWidget


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
