from __future__ import annotations

from math import isfinite

from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from evomachine.coordinates import Coordinate, CoordinateFactory
from evomachine.gui.panels.common import muted_label


class FovSetupPanel(QGroupBox):
    """FoV table used to initialise the automaton for strategy runs."""

    COLUMNS = ("FoV", "X", "Y", "Z")

    def __init__(self, controller, parent: QWidget | None = None):
        super().__init__("FoV Setup", parent)
        self.controller = controller
        self.current_coordinate: dict | None = None
        self.linear_start: dict | None = None
        self.linear_end: dict | None = None
        self.camera_fov_step_size: float | None = None
        self.fov_id_input = QSpinBox()
        self.fov_id_input.setRange(0, 9999)
        self.x_input = self._axis_input()
        self.y_input = self._axis_input()
        self.z_input = self._axis_input()
        self.use_autofocus_checkbox = QCheckBox("Use autofocus")
        self.use_autofocus_checkbox.setToolTip(
            "Use autofocus while the configured fields of view are initialised."
        )
        self.current_label = QLabel("current stage: -")
        self.current_label.setWordWrap(True)
        self.status_label = QLabel("Add at least one FoV before initialising.")
        self.status_label.setWordWrap(True)
        self.refresh_stage_button = QPushButton("Refresh")
        self.refresh_stage_button.setToolTip("Refresh the current stage coordinates.")
        self.use_current_button = QPushButton("Use Current")
        self.use_current_button.setToolTip("Copy the current stage coordinates into the FoV form.")
        self.add_button = QPushButton("Add / Update")
        self.add_button.setToolTip("Add this FoV or update the FoV with the same ID.")
        self.remove_button = QPushButton("Remove")
        self.remove_button.setToolTip("Remove the selected FoV.")
        self.clear_button = QPushButton("Clear")
        self.initialise_button = QPushButton("Initialise")
        self.initialise_button.setToolTip("Initialise the configured fields of view.")
        self.linear_start_label = QLabel("start: -")
        self.linear_start_label.setWordWrap(True)
        self.linear_end_label = QLabel("end: -")
        self.linear_end_label.setWordWrap(True)
        self.linear_spacing_label = QLabel("spacing: -")
        self.set_linear_start_button = QPushButton("Use Current as Start")
        self.set_linear_start_button.setToolTip(
            "Use the current stage coordinates as the start of a linear FoV path."
        )
        self.set_linear_end_button = QPushButton("Use Current as End")
        self.set_linear_end_button.setToolTip(
            "Use the current stage coordinates as the end of a linear FoV path."
        )
        self.generate_line_button = QPushButton("Generate Line")
        self.generate_line_button.setToolTip(
            "Append a straight, non-overlapping line of camera-sized FoVs to the table."
        )
        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.verticalHeader().setVisible(False)

        form = QFormLayout()
        form.addRow("FoV ID", self.fov_id_input)
        form.addRow("X", self.x_input)
        form.addRow("Y", self.y_input)
        form.addRow("Z", self.z_input)

        buttons = QGridLayout()
        buttons.addWidget(self.refresh_stage_button, 0, 0)
        buttons.addWidget(self.use_current_button, 0, 1)
        buttons.addWidget(self.add_button, 1, 0)
        buttons.addWidget(self.remove_button, 1, 1)
        buttons.addWidget(self.clear_button, 2, 0)
        buttons.addWidget(self.initialise_button, 2, 1)

        linear_group = QGroupBox("Linear FoVs")
        linear_layout = QVBoxLayout()
        linear_layout.addWidget(self.linear_start_label)
        linear_layout.addWidget(self.linear_end_label)
        linear_layout.addWidget(self.linear_spacing_label)
        linear_buttons = QGridLayout()
        linear_buttons.addWidget(self.set_linear_start_button, 0, 0)
        linear_buttons.addWidget(self.set_linear_end_button, 1, 0)
        linear_buttons.addWidget(self.generate_line_button, 2, 0)
        linear_layout.addLayout(linear_buttons)
        linear_group.setLayout(linear_layout)

        layout = QVBoxLayout()
        layout.addWidget(self.current_label)
        layout.addLayout(form)
        layout.addWidget(self.use_autofocus_checkbox)
        layout.addLayout(buttons)
        layout.addWidget(linear_group)
        layout.addWidget(self.table)
        layout.addWidget(self.status_label)
        layout.addWidget(muted_label("FoVs are stage positions visited by strategies."))
        self.setLayout(layout)

        self.refresh_stage_button.clicked.connect(self.controller.refresh_stage)
        self.use_current_button.clicked.connect(self._use_current_stage)
        self.add_button.clicked.connect(self._add_or_update_fov)
        self.remove_button.clicked.connect(self._remove_selected_fov)
        self.clear_button.clicked.connect(self._clear_fovs)
        self.initialise_button.clicked.connect(self._initialise_fovs)
        self.set_linear_start_button.clicked.connect(
            lambda _checked=False: self._set_linear_endpoint("start")
        )
        self.set_linear_end_button.clicked.connect(
            lambda _checked=False: self._set_linear_endpoint("end")
        )
        self.generate_line_button.clicked.connect(self._generate_linear_fovs)
        self.table.itemSelectionChanged.connect(self._load_selected_fov)
        self.controller.stage_coordinates_received.connect(self.update_current_coordinate)
        self.controller.stage_status_received.connect(self.update_stage_status)
        self.controller.fovs_received.connect(self.update_initialised_fovs)
        self.controller.response_error.connect(self._show_error)
        self._sync_buttons()

    @staticmethod
    def _axis_input() -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setRange(-1e7, 1e7)
        box.setDecimals(3)
        box.setSingleStep(1.0)
        return box

    def update_current_coordinate(self, payload: dict) -> None:
        self.current_coordinate = self._validated_coordinate_payload(payload.get("coordinate"))
        if self.current_coordinate is None:
            self.current_label.setText("current stage: unavailable")
        else:
            self.current_label.setText(
                f"current stage: {self._format_coordinate(self.current_coordinate)}"
            )
        self.update_stage_status(payload.get("stage", {}))
        self._sync_buttons()

    def update_stage_status(self, payload: dict) -> None:
        value = payload.get("camera_fov_step_size")
        if (
            isinstance(value, int | float)
            and not isinstance(value, bool)
            and isfinite(value)
            and value > 0
        ):
            self.camera_fov_step_size = float(value)
            self.linear_spacing_label.setText(
                f"spacing: {self._format_float(self.camera_fov_step_size)} µm"
            )
        else:
            self.camera_fov_step_size = None
            self.linear_spacing_label.setText("spacing: unavailable")
        self._sync_buttons()

    def update_initialised_fovs(self, fovs: list[dict]) -> None:
        self.status_label.setText(f"Initialised {len(fovs)} FoV(s).")

    def _use_current_stage(self) -> None:
        if self.current_coordinate is None:
            self.status_label.setText("Refresh the stage before using current coordinates.")
            return
        self.x_input.setValue(self.current_coordinate["x"])
        self.y_input.setValue(self.current_coordinate["y"])
        self.z_input.setValue(self.current_coordinate["z"])

    def _add_or_update_fov(self) -> None:
        fov = {
            "fov_id": self.fov_id_input.value(),
            "x": self.x_input.value(),
            "y": self.y_input.value(),
            "z": self.z_input.value(),
        }
        row = self._row_for_fov_id(fov["fov_id"])
        if row is None:
            row = self.table.rowCount()
            self.table.insertRow(row)
        self._set_row(row=row, fov=fov)
        self._sort_rows()
        self.fov_id_input.setValue(self._next_fov_id())
        self.status_label.setText(f"{self.table.rowCount()} FoV(s) staged for initialisation.")
        self._sync_buttons()

    def _set_linear_endpoint(self, endpoint: str) -> None:
        if self.current_coordinate is None:
            self.status_label.setText("Refresh the stage before setting a linear path endpoint.")
            return
        coordinate = dict(self.current_coordinate)
        if endpoint == "start":
            self.linear_start = coordinate
            self.linear_start_label.setText(f"start: {self._format_coordinate(coordinate)}")
        elif endpoint == "end":
            self.linear_end = coordinate
            self.linear_end_label.setText(f"end: {self._format_coordinate(coordinate)}")
        else:
            raise ValueError(f"Unknown linear FoV endpoint {endpoint!r}.")
        self._sync_buttons()

    def _generate_linear_fovs(self) -> None:
        if self.linear_start is None or self.linear_end is None:
            self.status_label.setText("Set both linear FoV endpoints before generating a line.")
            return
        if self.camera_fov_step_size is None:
            self.status_label.setText("Camera FoV spacing is unavailable; refresh the stage.")
            return

        factory = CoordinateFactory(dfov=self.camera_fov_step_size)
        coordinates = factory.make_grid(
            start=self._coordinate_from_payload(self.linear_start),
            stop=self._coordinate_from_payload(self.linear_end),
        )
        first_id = self._next_fov_id()
        if first_id + len(coordinates) - 1 > self.fov_id_input.maximum():
            self.status_label.setText("The generated line would exceed the maximum FoV ID.")
            return
        for offset, coordinate in enumerate(coordinates):
            row = self.table.rowCount()
            self.table.insertRow(row)
            self._set_row(
                row=row,
                fov={
                    "fov_id": first_id + offset,
                    "x": coordinate.x,
                    "y": coordinate.y,
                    "z": coordinate.z,
                },
            )
        self.fov_id_input.setValue(self._next_fov_id())
        self.status_label.setText(
            f"Generated {len(coordinates)} linear FoV(s); "
            f"{self.table.rowCount()} FoV(s) staged for initialisation."
        )
        self._sync_buttons()

    def _remove_selected_fov(self) -> None:
        row = self._selected_row()
        if row is None:
            return
        self.table.removeRow(row)
        self.status_label.setText(f"{self.table.rowCount()} FoV(s) staged for initialisation.")
        self._sync_buttons()

    def _clear_fovs(self) -> None:
        self.table.setRowCount(0)
        self.fov_id_input.setValue(0)
        self.linear_start = None
        self.linear_end = None
        self.linear_start_label.setText("start: -")
        self.linear_end_label.setText("end: -")
        self.status_label.setText("Add at least one FoV before initialising.")
        self._sync_buttons()

    def _initialise_fovs(self) -> None:
        fovs = self._fov_payload()
        if not fovs:
            self.status_label.setText("Add at least one FoV before initialising.")
            return
        self.status_label.setText("Initialising FoVs.")
        self.controller.initialise_fovs(
            fovs=fovs,
            use_autofocus=self.use_autofocus_checkbox.isChecked(),
        )

    def _load_selected_fov(self) -> None:
        row = self._selected_row()
        if row is None:
            return
        self.fov_id_input.setValue(int(float(self.table.item(row, 0).text())))
        self.x_input.setValue(float(self.table.item(row, 1).text()))
        self.y_input.setValue(float(self.table.item(row, 2).text()))
        self.z_input.setValue(float(self.table.item(row, 3).text()))
        self._sync_buttons()

    def _fov_payload(self) -> list[dict]:
        fovs = []
        for row in range(self.table.rowCount()):
            fovs.append(
                {
                    "fov_id": int(float(self.table.item(row, 0).text())),
                    "x": float(self.table.item(row, 1).text()),
                    "y": float(self.table.item(row, 2).text()),
                    "z": float(self.table.item(row, 3).text()),
                    "channel_id": 0,
                }
            )
        return fovs

    def _set_row(self, row: int, fov: dict) -> None:
        values = (
            str(fov["fov_id"]),
            self._format_float(fov["x"]),
            self._format_float(fov["y"]),
            self._format_float(fov["z"]),
        )
        for column, value in enumerate(values):
            item = QTableWidgetItem(value)
            self.table.setItem(row, column, item)

    def _sort_rows(self) -> None:
        fovs = sorted(self._fov_payload(), key=lambda item: item["fov_id"])
        self.table.setRowCount(0)
        for fov in fovs:
            row = self.table.rowCount()
            self.table.insertRow(row)
            self._set_row(row=row, fov=fov)

    def _row_for_fov_id(self, fov_id: int) -> int | None:
        for row in range(self.table.rowCount()):
            if int(float(self.table.item(row, 0).text())) == fov_id:
                return row
        return None

    def _selected_row(self) -> int | None:
        selected = self.table.selectionModel().selectedRows()
        if not selected:
            return None
        return selected[0].row()

    def _next_fov_id(self) -> int:
        if self.table.rowCount() == 0:
            return 0
        return max(fov["fov_id"] for fov in self._fov_payload()) + 1

    def _sync_buttons(self) -> None:
        has_rows = self.table.rowCount() > 0
        self.use_current_button.setEnabled(self.current_coordinate is not None)
        self.set_linear_start_button.setEnabled(self.current_coordinate is not None)
        self.set_linear_end_button.setEnabled(self.current_coordinate is not None)
        self.generate_line_button.setEnabled(
            self.linear_start is not None
            and self.linear_end is not None
            and self.camera_fov_step_size is not None
        )
        self.remove_button.setEnabled(self._selected_row() is not None)
        self.clear_button.setEnabled(has_rows)
        self.initialise_button.setEnabled(has_rows)

    def _show_error(self, error: str) -> None:
        self.status_label.setText(error)

    @staticmethod
    def _format_float(value: float) -> str:
        return f"{float(value):.3f}"

    @classmethod
    def _format_coordinate(cls, coordinate: dict) -> str:
        return ", ".join(
            f"{axis}={cls._format_float(coordinate[axis])}" for axis in ("x", "y", "z")
        )

    @staticmethod
    def _coordinate_from_payload(coordinate: dict) -> Coordinate:
        return Coordinate(
            x=float(coordinate["x"]),
            y=float(coordinate["y"]),
            z=float(coordinate["z"]),
        )

    @staticmethod
    def _validated_coordinate_payload(coordinate: object) -> dict[str, float] | None:
        if not isinstance(coordinate, dict):
            return None
        values = {axis: coordinate.get(axis) for axis in ("x", "y", "z")}
        if any(
            not isinstance(value, int | float) or isinstance(value, bool) or not isfinite(value)
            for value in values.values()
        ):
            return None
        return {axis: float(value) for axis, value in values.items()}


class StrategySetupPanel(QGroupBox):
    """Strategy selection and lifecycle shell."""

    def __init__(self, controller, parent: QWidget | None = None):
        super().__init__("Strategy Setup", parent)
        self.controller = controller
        self.strategy_combo = QComboBox()
        self.strategy_combo.setEnabled(False)
        self.refresh_button = QPushButton("Refresh Strategies")
        self.set_button = QPushButton("Set Strategy")
        self.start_button = QPushButton("Start Strategy")
        self.stop_button = QPushButton("Stop Strategy")
        self.file_label = QLabel("-")
        self.file_label.setWordWrap(True)
        self.notes_label = QLabel("-")
        self.notes_label.setWordWrap(True)
        self.status_label = QLabel("No strategy status yet.")
        self.status_label.setWordWrap(True)
        self._last_strategy_status: dict = {}

        form = QFormLayout()
        form.addRow("Strategy", self.strategy_combo)
        form.addRow("File", self.file_label)
        form.addRow("Notes", self.notes_label)

        buttons = QGridLayout()
        buttons.addWidget(self.refresh_button, 0, 0)
        buttons.addWidget(self.set_button, 0, 1)
        buttons.addWidget(self.start_button, 1, 0)
        buttons.addWidget(self.stop_button, 1, 1)

        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addLayout(buttons)
        layout.addWidget(self.status_label)
        layout.addWidget(muted_label("Strategies must be set before they can be started."))
        self.setLayout(layout)

        self.refresh_button.clicked.connect(self.controller.refresh_strategies)
        self.set_button.clicked.connect(self._set_strategy)
        self.start_button.clicked.connect(self.controller.start_strategy)
        self.stop_button.clicked.connect(self.controller.stop_strategy)
        self.strategy_combo.currentIndexChanged.connect(self._show_selected_strategy)
        self.controller.strategies_received.connect(self.update_strategies)
        self.controller.strategy_status_received.connect(self.update_status)
        self.controller.response_error.connect(self._show_error)
        self._sync_controls(strategy_status={})
        self.controller.refresh_strategies()

    def update_strategies(self, strategies: list[dict]) -> None:
        self.strategy_combo.blockSignals(True)
        self.strategy_combo.clear()
        for strategy in strategies:
            self.strategy_combo.addItem(strategy.get("name", "<unnamed>"), strategy)
        self.strategy_combo.blockSignals(False)
        self.strategy_combo.setEnabled(bool(strategies))
        self._show_selected_strategy()
        selected = self._selected_strategy()
        if selected is None or not selected.get("error"):
            self.status_label.setText(f"{len(strategies)} strategy option(s) available.")
        self._sync_controls(strategy_status=self._last_strategy_status)

    def update_status(self, payload: dict) -> None:
        self._last_strategy_status = payload
        name = payload.get("name") or "-"
        state = "running" if payload.get("running") else "not running"
        initialised = "initialised" if payload.get("is_initialised") else "not initialised"
        fovs = "FoVs ready" if payload.get("fovs_initialised") else "FoVs not ready"
        self.status_label.setText(f"{name}: {initialised}, {state}, {fovs}.")
        self._sync_controls(strategy_status=payload)

    def _set_strategy(self) -> None:
        strategy = self._selected_strategy()
        if strategy is None:
            return
        self.controller.set_strategy(
            name=strategy["name"],
            file_path=strategy.get("file_path"),
        )

    def _show_selected_strategy(self) -> None:
        strategy = self._selected_strategy()
        if strategy is None:
            self.file_label.setText("-")
            self.notes_label.setText("-")
            return
        self.file_label.setText(strategy.get("file_path") or "built in")
        notes = strategy.get("notes") or []
        self.notes_label.setText("\n".join(notes) if notes else "-")
        error = strategy.get("error")
        if error:
            self.status_label.setText(error)
        self._sync_controls(strategy_status=self._last_strategy_status)

    def _sync_controls(self, strategy_status: dict) -> None:
        running = bool(strategy_status.get("running"))
        started = bool(strategy_status.get("started"))
        stopped = bool(strategy_status.get("stopped"))
        can_start = (
            bool(strategy_status.get("is_initialised"))
            and not started
            and not stopped
            and not running
        )
        self.set_button.setEnabled(self._selected_strategy() is not None and not started)
        self.start_button.setEnabled(can_start)
        self.stop_button.setEnabled(running)

    def _selected_strategy(self) -> dict | None:
        index = self.strategy_combo.currentIndex()
        if index < 0:
            return None
        data = self.strategy_combo.itemData(index)
        return data if isinstance(data, dict) else None

    def _show_error(self, error: str) -> None:
        self.status_label.setText(error)


class AiAssistancePanel(QGroupBox):
    """Placeholder home for future strategy-writing assistance."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__("AI Assistance", parent)
        prompt_count = QSpinBox()
        prompt_count.setRange(1, 20)
        prompt_count.setValue(3)
        prompt_count.setEnabled(False)
        generate_button = QPushButton("Draft Strategy")
        generate_button.setEnabled(False)

        form = QFormLayout()
        form.addRow("Suggestions", prompt_count)

        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addWidget(generate_button)
        layout.addWidget(muted_label("Placeholder only"))
        self.setLayout(layout)
