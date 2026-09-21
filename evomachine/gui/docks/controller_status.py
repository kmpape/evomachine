from __future__ import annotations

from PyQt5.QtWidgets import QVBoxLayout, QWidget

from evomachine.gui.controller import EvoMachineGuiController
from evomachine.gui.panels.status import PeripheralControllerStatusPanel


class PeripheralControllerStatusDock(QWidget):
    """Napari dock widget containing peripheral controller status."""

    def __init__(self, controller: EvoMachineGuiController | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        self.controller = controller if controller is not None else EvoMachineGuiController()
        self._owns_controller = controller is None

        layout = QVBoxLayout()
        layout.addWidget(PeripheralControllerStatusPanel(controller=self.controller))
        self.setLayout(layout)

    def closeEvent(self, event) -> None:  # noqa: N802
        if self._owns_controller:
            self.controller.close()
        super().closeEvent(event)
