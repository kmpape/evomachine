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

from evomachine.gui.panels.config_dialog import ConfigDialog, ConfigFieldSpec


class StagePanel(QGroupBox):
    """Simple Stage control panel."""

    FOV_DIRECTIONS = (
        ("Up", "UP", 0, 1),
        ("Left", "LEFT", 1, 0),
        ("Right", "RIGHT", 1, 2),
        ("Down", "DOWN", 2, 1),
    )

    def __init__(self, controller, parent: QWidget | None = None):
        super().__init__("Stage", parent)
        self.controller = controller
        self._latest_coordinate: dict = {}
        self._latest_status: dict = {}
        self.coordinate_label = QLabel("x: -, y: -, z: -")
        self.machine_coordinate_label = QLabel("machine x: -, y: -, z: -")
        self.fov_step_label = QLabel("camera FoV step: -")
        self.status_label = QLabel("Run Initialise Devices before using stage controls.")
        self.devices_initialised = False
        self.strategy_running = False
        self.fov_buttons: list[QPushButton] = []
        self.x_input = self._axis_input()
        self.y_input = self._axis_input()
        self.z_input = self._axis_input()

        self.refresh_button = QPushButton("Refresh")
        self.move_button = QPushButton("Move by ΔXYZ")
        self.move_button.setToolTip("Move relative to the current calibration coordinates.")
        self.stop_button = QPushButton("Stop")
        self.zero_button = QPushButton("Set Current XYZ as Zero")
        self.zero_button.setToolTip(
            "Set the current position to user XYZ = 0 while retaining the original machine coordinates."
        )
        self.origin_button = QPushButton("Return to Calibration Origin")
        self.origin_button.setToolTip("Move the stage to user/calibration coordinates XYZ = 0.")
        self.configure_button = QPushButton("Configure")

        form = QFormLayout()
        form.addRow("ΔX (µm)", self.x_input)
        form.addRow("ΔY (µm)", self.y_input)
        form.addRow("ΔZ (µm)", self.z_input)

        buttons = QGridLayout()
        buttons.addWidget(self.refresh_button, 0, 0)
        buttons.addWidget(self.move_button, 0, 1)
        buttons.addWidget(self.stop_button, 0, 2)
        buttons.addWidget(self.zero_button, 1, 0, 1, 3)
        buttons.addWidget(self.origin_button, 2, 0, 1, 3)
        buttons.addWidget(self.configure_button, 3, 0, 1, 3)

        fov_buttons = QGridLayout()
        for label, direction, row, column in self.FOV_DIRECTIONS:
            button = QPushButton(label)
            button.setEnabled(False)
            button.clicked.connect(
                lambda _checked=False, selected=direction: self._move_camera_fov(selected)
            )
            self.fov_buttons.append(button)
            fov_buttons.addWidget(button, row, column)

        layout = QVBoxLayout()
        layout.addWidget(self.coordinate_label)
        layout.addWidget(self.machine_coordinate_label)
        layout.addWidget(self.fov_step_label)
        layout.addWidget(self.status_label)
        layout.addLayout(form)
        layout.addLayout(buttons)
        layout.addWidget(QLabel("Move by one camera FoV"))
        layout.addLayout(fov_buttons)
        self.setLayout(layout)

        self.refresh_button.clicked.connect(self.controller.refresh_stage)
        self.move_button.clicked.connect(self._move_delta)
        self.stop_button.clicked.connect(self.controller.stop_stage)
        self.zero_button.clicked.connect(self._zero_stage)
        self.origin_button.clicked.connect(self._return_to_origin)
        self.configure_button.clicked.connect(self._open_config_dialog)
        self.controller.stage_coordinates_received.connect(self.update_coordinates)
        self.controller.stage_status_received.connect(self.update_status)
        self.controller.lifecycle_status_received.connect(self.update_lifecycle_status)
        self.controller.strategy_status_received.connect(self.update_strategy_status)
        self._sync_controls_enabled()

    @staticmethod
    def _axis_input() -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setRange(-1e7, 1e7)
        box.setDecimals(3)
        box.setSingleStep(1.0)
        return box

    def _move_delta(self) -> None:
        if not self.devices_initialised:
            self.status_label.setText("Run Initialise Devices before using stage controls.")
            return
        self.status_label.setText("Moving stage by relative ΔXYZ.")
        self.controller.move_stage_relative(
            dx=self.x_input.value(),
            dy=self.y_input.value(),
            dz=self.z_input.value(),
        )

    def _move_camera_fov(self, direction: str) -> None:
        if not self.devices_initialised:
            self.status_label.setText("Run Initialise Devices before using stage controls.")
            return
        self.status_label.setText(f"Moving one camera FoV {direction.lower()}.")
        self.controller.move_stage_fov(direction=direction, multiplier=1.0)

    def _zero_stage(self) -> None:
        if not self.devices_initialised:
            self.status_label.setText("Run Initialise Devices before zeroing the stage.")
            return
        self.status_label.setText("Setting current XYZ as the calibration zero.")
        self.controller.zero_stage()

    def _return_to_origin(self) -> None:
        if not self.devices_initialised:
            self.status_label.setText("Run Initialise Devices before returning to the origin.")
            return
        self.status_label.setText("Returning stage to calibration origin XYZ = 0.")
        self.controller.return_stage_to_origin()

    def _open_config_dialog(self) -> None:
        if not self.devices_initialised:
            self.status_label.setText("Run Initialise Devices before using stage controls.")
            return
        dialog = ConfigDialog(
            title="Stage Configuration",
            fields=self._config_fields(),
            parent=self,
        )
        dialog.exec_()

    def update_coordinates(self, payload: dict) -> None:
        coordinate = payload.get("coordinate", {})
        self._latest_coordinate = dict(coordinate)
        self.coordinate_label.setText(
            f"x: {coordinate.get('x')}, y: {coordinate.get('y')}, z: {coordinate.get('z')}"
        )
        machine_coordinate = payload.get("machine_coordinate", {})
        self.machine_coordinate_label.setText(
            f"machine x: {machine_coordinate.get('x')}, "
            f"y: {machine_coordinate.get('y')}, z: {machine_coordinate.get('z')}"
        )
        if "stage" in payload:
            self.update_status(payload["stage"])

    def update_status(self, payload: dict) -> None:
        self._latest_status = dict(payload)
        self.status_label.setText(
            f"initialised: {payload.get('is_initialised')}, alive: {payload.get('is_alive')}, "
            f"fov: {payload.get('fov_id')}"
        )
        fov_step_size = payload.get("camera_fov_step_size", payload.get("fov_step_size"))
        if isinstance(fov_step_size, int | float):
            fov_step_text = f"{float(fov_step_size):.3f} um"
        else:
            fov_step_text = "-"
        self.fov_step_label.setText(f"camera FoV step: {fov_step_text}")

    def update_lifecycle_status(self, payload: dict) -> None:
        self.devices_initialised = bool(payload.get("devices_initialised"))
        if payload.get("shutdown"):
            self.devices_initialised = False
        self._sync_controls_enabled()
        if not self.devices_initialised:
            self.status_label.setText("Run Initialise Devices before using stage controls.")
        elif self.status_label.text().startswith("Run Initialise Devices"):
            self.status_label.setText("Refresh stage to read coordinates.")

    def update_strategy_status(self, payload: dict) -> None:
        self.strategy_running = bool(payload.get("running"))
        self._sync_controls_enabled()

    def _sync_controls_enabled(self) -> None:
        manual_controls_enabled = self.devices_initialised and not self.strategy_running
        for widget in (
            self.x_input,
            self.y_input,
            self.z_input,
            self.move_button,
            self.zero_button,
            self.origin_button,
            self.configure_button,
            *self.fov_buttons,
        ):
            widget.setEnabled(manual_controls_enabled)
        self.refresh_button.setEnabled(self.devices_initialised)
        self.stop_button.setEnabled(self.devices_initialised)

    def _config_fields(self) -> list[ConfigFieldSpec]:
        return [
            ConfigFieldSpec("X", "x", self._latest_coordinate.get("x"), editable=False),
            ConfigFieldSpec("Y", "y", self._latest_coordinate.get("y"), editable=False),
            ConfigFieldSpec("Z", "z", self._latest_coordinate.get("z"), editable=False),
            ConfigFieldSpec("FoV ID", "fov_id", self._latest_status.get("fov_id"), editable=False),
            ConfigFieldSpec(
                "Stage FoV step",
                "fov_step_size",
                self._latest_status.get("fov_step_size"),
                editable=False,
            ),
            ConfigFieldSpec(
                "Camera FoV step",
                "camera_fov_step_size",
                self._latest_status.get("camera_fov_step_size"),
                editable=False,
            ),
        ]
