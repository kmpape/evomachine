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
        self.status_label = QLabel("Run Initialise Devices before using stage controls.")
        self.devices_initialised = False
        self.x_input = self._axis_input()
        self.y_input = self._axis_input()
        self.z_input = self._axis_input()

        self.refresh_button = QPushButton("Refresh")
        self.move_button = QPushButton("Move")
        self.stop_button = QPushButton("Stop")

        form = QFormLayout()
        form.addRow("X", self.x_input)
        form.addRow("Y", self.y_input)
        form.addRow("Z", self.z_input)

        buttons = QGridLayout()
        buttons.addWidget(self.refresh_button, 0, 0)
        buttons.addWidget(self.move_button, 0, 1)
        buttons.addWidget(self.stop_button, 0, 2)

        layout = QVBoxLayout()
        layout.addWidget(self.coordinate_label)
        layout.addWidget(self.status_label)
        layout.addLayout(form)
        layout.addLayout(buttons)
        self.setLayout(layout)

        self.refresh_button.clicked.connect(self.controller.refresh_stage)
        self.move_button.clicked.connect(self._move_absolute)
        self.stop_button.clicked.connect(self.controller.stop_stage)
        self.controller.stage_coordinates_received.connect(self.update_coordinates)
        self.controller.stage_status_received.connect(self.update_status)
        self.controller.lifecycle_status_received.connect(self.update_lifecycle_status)
        self._sync_controls_enabled()

    @staticmethod
    def _axis_input() -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setRange(-1e7, 1e7)
        box.setDecimals(3)
        box.setSingleStep(1.0)
        return box

    def _move_absolute(self) -> None:
        if not self.devices_initialised:
            self.status_label.setText("Run Initialise Devices before using stage controls.")
            return
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

    def update_lifecycle_status(self, payload: dict) -> None:
        self.devices_initialised = bool(payload.get("devices_initialised"))
        if payload.get("shutdown"):
            self.devices_initialised = False
        self._sync_controls_enabled()
        if not self.devices_initialised:
            self.status_label.setText("Run Initialise Devices before using stage controls.")
        elif self.status_label.text().startswith("Run Initialise Devices"):
            self.status_label.setText("Refresh stage to read coordinates.")

    def _sync_controls_enabled(self) -> None:
        for widget in (
            self.x_input,
            self.y_input,
            self.z_input,
            self.refresh_button,
            self.move_button,
            self.stop_button,
        ):
            widget.setEnabled(self.devices_initialised)
