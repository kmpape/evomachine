from __future__ import annotations

from PyQt5.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class StagePanel(QGroupBox):
    """Simple Stage control panel."""

    def __init__(self, controller, parent: QWidget | None = None):
        super().__init__("Stage", parent)
        self.controller = controller
        self.coordinate_label = QLabel("x: -, y: -, z: -")
        self.status_label = QLabel("Not connected")
        self.x_input = self._axis_input()
        self.y_input = self._axis_input()
        self.z_input = self._axis_input()

        refresh_button = QPushButton("Refresh")
        move_button = QPushButton("Move")
        stop_button = QPushButton("Stop")

        form = QFormLayout()
        form.addRow("X", self.x_input)
        form.addRow("Y", self.y_input)
        form.addRow("Z", self.z_input)

        buttons = QGridLayout()
        buttons.addWidget(refresh_button, 0, 0)
        buttons.addWidget(move_button, 0, 1)
        buttons.addWidget(stop_button, 0, 2)

        layout = QVBoxLayout()
        layout.addWidget(self.coordinate_label)
        layout.addWidget(self.status_label)
        layout.addLayout(form)
        layout.addLayout(buttons)
        self.setLayout(layout)

        refresh_button.clicked.connect(self.controller.refresh_stage)
        move_button.clicked.connect(self._move_absolute)
        stop_button.clicked.connect(self.controller.stop_stage)
        self.controller.stage_coordinates_received.connect(self.update_coordinates)
        self.controller.stage_status_received.connect(self.update_status)

    @staticmethod
    def _axis_input() -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setRange(-1e7, 1e7)
        box.setDecimals(3)
        box.setSingleStep(1.0)
        return box

    def _move_absolute(self) -> None:
        self.controller.move_stage_absolute(
            x=self.x_input.value(),
            y=self.y_input.value(),
            z=self.z_input.value(),
        )

    def update_coordinates(self, payload: dict) -> None:
        coordinate = payload.get("coordinate", {})
        self.coordinate_label.setText(
            f"x: {coordinate.get('x')}, y: {coordinate.get('y')}, z: {coordinate.get('z')}"
        )
        if "stage" in payload:
            self.update_status(payload["stage"])

    def update_status(self, payload: dict) -> None:
        self.status_label.setText(
            f"initialised: {payload.get('is_initialised')}, alive: {payload.get('is_alive')}, "
            f"fov: {payload.get('fov_id')}"
        )

