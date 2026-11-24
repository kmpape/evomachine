from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QScrollArea, QVBoxLayout, QWidget

from evomachine.gui.controller import EvoMachineGuiController
from evomachine.gui.panels.status import PeripheralControllerStatusPanel


class PeripheralControllerStatusDock(QWidget):
    """Napari dock containing Automaton lifecycle and peripheral status."""

    def __init__(self, controller: EvoMachineGuiController | None = None, parent: QWidget | None = None):
        """Create a scrollable status dock around one shared controller.

        Args:
            controller: Existing GUI controller, or None to construct and own one.
            parent: Optional owning Qt widget.
        """
        super().__init__(parent)
        self.controller = controller if controller is not None else EvoMachineGuiController()
        self._owns_controller = controller is None

        self.controller_status_panel = PeripheralControllerStatusPanel(controller=self.controller)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.addWidget(self.controller_status_panel)
        content_layout.addStretch(1)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setWidget(content)
        layout = QVBoxLayout()
        layout.addWidget(scroll_area)
        self.setLayout(layout)

    def closeEvent(self, event) -> None:  # noqa: N802
        if self._owns_controller:
            self.controller.close()
        super().closeEvent(event)
