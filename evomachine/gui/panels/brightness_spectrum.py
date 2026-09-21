from __future__ import annotations

from PyQt5.QtWidgets import QWidget

from evomachine.gui.panels.common import DisabledPanelShell


class BrightnessSpectrumPanel(DisabledPanelShell):
    """Brightness spectrum display shell."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(
            "Brightness Spectrum",
            actions=("Refresh Spectrum",),
            parent=parent,
        )
