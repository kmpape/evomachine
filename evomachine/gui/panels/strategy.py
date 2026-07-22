from __future__ import annotations

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

from evomachine.gui.panels.common import muted_label


class FovSetupPanel(QGroupBox):
    """FoV table used to initialise the automaton for strategy runs."""

    COLUMNS = ("FoV", "X", "Y", "Z")

    def __init__(self, controller, parent: QWidget | None = None):
        super().__init__("FoV Setup", parent)
        self.controller = controller
        self.current_coordinate: dict | None = None
        self.fov_id_input = QSpinBox()
        self.fov_id_input.setRange(0, 9999)
        self.x_input = self._axis_input()
        self.y_input = self._axis_input()
        self.z_input = self._axis_input()
        self.use_autofocus_checkbox = QCheckBox("Use autofocus during FoV setup")
        self.current_label = QLabel("current stage: -")
        self.current_label.setWordWrap(True)
        self.status_label = QLabel("Add at least one FoV before initialising.")
        self.status_label.setWordWrap(True)
        self.refresh_stage_button = QPushButton("Refresh Stage")
        self.use_current_button = QPushButton("Use Current Stage")
        self.add_button = QPushButton("Add / Update FoV")
        self.remove_button = QPushButton("Remove Selected")
        self.clear_button = QPushButton("Clear")
        self.initialise_button = QPushButton("Initialise FoVs")
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

        layout = QVBoxLayout()
        layout.addWidget(self.current_label)
        layout.addLayout(form)
        layout.addWidget(self.use_autofocus_checkbox)
        layout.addLayout(buttons)
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
        self.table.itemSelectionChanged.connect(self._load_selected_fov)
        self.controller.stage_coordinates_received.connect(self.update_current_coordinate)
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
        coordinate = payload.get("coordinate", {})
        self.current_coordinate = coordinate
        self.current_label.setText(
            f"current stage: x={coordinate.get('x')}, y={coordinate.get('y')}, z={coordinate.get('z')}"
        )
        self._sync_buttons()

    def update_initialised_fovs(self, fovs: list[dict]) -> None:
        self.status_label.setText(f"Initialised {len(fovs)} FoV(s).")

    def _use_current_stage(self) -> None:
        if self.current_coordinate is None:
            self.status_label.setText("Refresh the stage before using current coordinates.")
            return
        self.x_input.setValue(float(self.current_coordinate.get("x") or 0.0))
        self.y_input.setValue(float(self.current_coordinate.get("y") or 0.0))
        self.z_input.setValue(float(self.current_coordinate.get("z") or 0.0))

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
            fovs.append({
                "fov_id": int(float(self.table.item(row, 0).text())),
                "x": float(self.table.item(row, 1).text()),
                "y": float(self.table.item(row, 2).text()),
                "z": float(self.table.item(row, 3).text()),
                "channel_id": 0,
            })
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
        self.remove_button.setEnabled(self._selected_row() is not None)
        self.clear_button.setEnabled(has_rows)
        self.initialise_button.setEnabled(has_rows)

    def _show_error(self, error: str) -> None:
        self.status_label.setText(error)

    @staticmethod
    def _format_float(value: float) -> str:
        return f"{float(value):.3f}"


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
        can_start = bool(strategy_status.get("is_initialised")) and not started and not stopped and not running
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
