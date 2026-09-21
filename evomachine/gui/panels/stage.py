from __future__ import annotations

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QDoubleSpinBox,
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
        self.coordinate_label = QLabel("Current absolute position: x: -, y: -, z: -")
        self.limits_label = QLabel("active limits: -")
        self.limits_label.setWordWrap(True)
        self.fov_step_label = QLabel("camera FoV step: -")
        self.status_label = QLabel("Run Initialise Devices before using stage controls.")
        self.devices_initialised = False
        self.strategy_running = False
        self.movement_running = False
        self.fov_buttons: list[QPushButton] = []
        self.x_input = self._axis_input("ΔX (µm): ")
        self.y_input = self._axis_input("ΔY (µm): ")
        self.z_input = self._axis_input("ΔZ (µm): ")
        self.absolute_x_input = self._axis_input("X (µm): ")
        self.absolute_y_input = self._axis_input("Y (µm): ")
        self.absolute_z_input = self._axis_input("Z (µm): ")

        self.refresh_button = QPushButton("Refresh")
        self.move_button = QPushButton("Move Relative")
        self.move_button.setToolTip("Apply ΔX, ΔY, and ΔZ as offsets from the current position.")
        self.absolute_move_button = QPushButton("Move Absolute")
        self.absolute_move_button.setToolTip("Move to the target X, Y, and Z stage coordinates.")
        self.stop_button = QPushButton("Stop")
        self.origin_button = QPushButton("Return to Origin")
        self.origin_button.setToolTip("Move the stage to Tiger coordinates XYZ = 0.")
        self.configure_button = QPushButton("Configure")
        self.movement_poll_timer = QTimer(self)
        self.movement_poll_timer.setInterval(250)

        relative_form = self._movement_layout(
            self.x_input,
            self.y_input,
            self.z_input,
            self.move_button,
        )
        self.relative_movement_group = QGroupBox("Relative movement")
        self.relative_movement_group.setToolTip("Offsets are applied from the current stage position.")
        self.relative_movement_group.setLayout(relative_form)

        absolute_form = self._movement_layout(
            self.absolute_x_input,
            self.absolute_y_input,
            self.absolute_z_input,
            self.absolute_move_button,
            self.origin_button,
        )
        self.absolute_movement_group = QGroupBox("Absolute movement")
        self.absolute_movement_group.setToolTip("Targets use absolute zeroed Tiger coordinates.")
        self.absolute_movement_group.setLayout(absolute_form)

        buttons = QGridLayout()
        buttons.addWidget(self.refresh_button, 0, 0)
        buttons.addWidget(self.stop_button, 0, 1)
        buttons.addWidget(self.configure_button, 1, 0, 1, 2)

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
        layout.addWidget(self.limits_label)
        layout.addWidget(self.fov_step_label)
        layout.addWidget(self.status_label)
        layout.addWidget(self.relative_movement_group)
        layout.addWidget(self.absolute_movement_group)
        layout.addLayout(buttons)
        layout.addWidget(QLabel("Relative movement by one camera FoV"))
        layout.addLayout(fov_buttons)
        self.setLayout(layout)

        self.refresh_button.clicked.connect(self.controller.refresh_stage)
        self.move_button.clicked.connect(self._move_delta)
        self.absolute_move_button.clicked.connect(self._move_absolute)
        self.stop_button.clicked.connect(self.controller.stop_stage)
        self.origin_button.clicked.connect(self._return_to_origin)
        self.configure_button.clicked.connect(self._open_config_dialog)
        self.controller.stage_coordinates_received.connect(self.update_coordinates)
        self.controller.stage_status_received.connect(self.update_status)
        self.controller.lifecycle_status_received.connect(self.update_lifecycle_status)
        self.controller.strategy_status_received.connect(self.update_strategy_status)
        self.controller.operation_status_received.connect(self.update_operation_status)
        self.movement_poll_timer.timeout.connect(
            self.controller.refresh_stage_movement_operation
        )
        self._sync_controls_enabled()

    @staticmethod
    def _axis_input(prefix: str) -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setRange(-1e7, 1e7)
        box.setDecimals(3)
        box.setSingleStep(1.0)
        box.setPrefix(prefix)
        box.setAlignment(Qt.AlignCenter)
        return box

    @staticmethod
    def _movement_layout(*widgets: QWidget) -> QVBoxLayout:
        layout = QVBoxLayout()
        layout.setSpacing(8)
        for widget in widgets:
            layout.addWidget(widget)
        return layout

    def _move_delta(self) -> None:
        if not self.devices_initialised or self.movement_running:
            self.status_label.setText("Run Initialise Devices before using stage controls.")
            return
        self.status_label.setText("Moving stage by relative ΔXYZ.")
        self._start_movement()
        self.controller.move_stage_relative(
            dx=self.x_input.value(),
            dy=self.y_input.value(),
            dz=self.z_input.value(),
        )

    def _move_absolute(self) -> None:
        if not self.devices_initialised or self.movement_running:
            self.status_label.setText("Run Initialise Devices before using stage controls.")
            return
        self.status_label.setText("Moving stage to absolute target XYZ.")
        self._start_movement()
        self.controller.move_stage_absolute(
            x=self.absolute_x_input.value(),
            y=self.absolute_y_input.value(),
            z=self.absolute_z_input.value(),
        )

    def _move_camera_fov(self, direction: str) -> None:
        if not self.devices_initialised or self.movement_running:
            self.status_label.setText("Run Initialise Devices before using stage controls.")
            return
        self.status_label.setText(f"Moving one camera FoV {direction.lower()}.")
        self._start_movement()
        self.controller.move_stage_fov(direction=direction, multiplier=1.0)

    def _return_to_origin(self) -> None:
        if not self.devices_initialised or self.movement_running:
            self.status_label.setText("Run Initialise Devices before returning to the origin.")
            return
        self.status_label.setText("Returning stage to origin XYZ = 0.")
        self._start_movement()
        self.controller.return_stage_to_origin()

    def _start_movement(self) -> None:
        self.movement_running = True
        self.movement_poll_timer.start()
        self._sync_controls_enabled()

    def update_operation_status(self, operation: dict) -> None:
        if operation.get("kind") != "stage_movement":
            return
        state = str(operation.get("state", "-"))
        self.movement_running = state == "running"
        if not self.movement_running:
            self.movement_poll_timer.stop()
            if state == "failed" and operation.get("error"):
                self.status_label.setText(str(operation["error"]))
            else:
                self.status_label.setText(f"Stage movement {state}.")
        self._sync_controls_enabled()

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
            "Current absolute position: "
            f"x: {coordinate.get('x')}, y: {coordinate.get('y')}, z: {coordinate.get('z')}"
        )
        for axis, target_input in (
            ("x", self.absolute_x_input),
            ("y", self.absolute_y_input),
            ("z", self.absolute_z_input),
        ):
            value = coordinate.get(axis)
            if isinstance(value, int | float) and not isinstance(value, bool):
                target_input.setValue(float(value))
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
        self._update_limits(payload.get("coordinate_bounds"))

    def _update_limits(self, bounds: dict | None) -> None:
        if not isinstance(bounds, dict):
            self.limits_label.setText("active limits: -")
            return
        low = bounds.get("low", {})
        high = bounds.get("high", {})
        self.limits_label.setText(
            "active limits (µm):\n"
            f"X [{low.get('x')}, {high.get('x')}]\n"
            f"Y [{low.get('y')}, {high.get('y')}]\n"
            f"Z [{low.get('z')}, {high.get('z')}]"
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

    def update_strategy_status(self, payload: dict) -> None:
        self.strategy_running = bool(payload.get("running"))
        self._sync_controls_enabled()

    def _sync_controls_enabled(self) -> None:
        manual_controls_enabled = (
            self.devices_initialised
            and not self.strategy_running
            and not self.movement_running
        )
        for widget in (
            self.x_input,
            self.y_input,
            self.z_input,
            self.absolute_x_input,
            self.absolute_y_input,
            self.absolute_z_input,
            self.move_button,
            self.absolute_move_button,
            self.origin_button,
            *self.fov_buttons,
        ):
            widget.setEnabled(manual_controls_enabled)
        self.configure_button.setEnabled(self.devices_initialised and not self.strategy_running)
        self.refresh_button.setEnabled(self.devices_initialised and not self.movement_running)
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
