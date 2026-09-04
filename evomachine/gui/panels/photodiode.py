from __future__ import annotations

from PyQt5.QtWidgets import QWidget

from evomachine.gui.panels.common import DisabledPanelShell


class PhotodiodePanel(DisabledPanelShell):
    """Photodiode controls shell."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(
            "Photodiode",
            actions=("Read", "Record"),
            note="Not currently configured",
            parent=parent,
        )
