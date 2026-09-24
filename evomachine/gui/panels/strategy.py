from __future__ import annotations

from math import isfinite

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFontDatabase

from PyQt5.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QCheckBox,
    QDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHeaderView,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from evomachine.coordinates import Coordinate, CoordinateFactory
from evomachine.gui.panels.common import muted_label
from evomachine.gui.protocol import GuiCommandType


class FovSetupPanel(QGroupBox):
    """Capture stage positions; never move hardware from the setup table."""

    COLUMNS = ("FoV", "X (µm)", "Y (µm)", "Z (µm)")

    def __init__(self, controller, parent: QWidget | None = None):
        super().__init__("FoV Setup", parent)
        self.controller = controller
        self.current_coordinate: dict | None = None
        self.linear_start: dict | None = None
        self.linear_end: dict | None = None
        self.camera_fov_step_size: float | None = None
        self._pending_capture: str | None = None
        self._running = False
        self._moving = False
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Manual", "Linear"])
        self.current_label = QLabel("Current stage: unavailable")
        self.current_label.setWordWrap(True)
        self.status_label = QLabel("Move using the stage controls, then add the current position.")
        self.status_label.setWordWrap(True)
        self.add_button = QPushButton("Add current")
        self.add_button.setToolTip("Read the stage position and append it as the next FoV.")
        self.remove_button = QPushButton("Remove last")
        self.clear_button = QPushButton("Clear")
        self.initialise_button = QPushButton("Initialise")
        self.linear_start_label = QLabel("X: -\nY: -\nZ: -")
        self.linear_end_label = QLabel("X: -\nY: -\nZ: -")
        for label in (self.linear_start_label, self.linear_end_label):
            label.setFixedHeight(label.fontMetrics().lineSpacing() * 3 + 6)
            label.setMinimumWidth(0)
        self.linear_spacing_label = QLabel("spacing: unavailable")
        self.set_linear_start_button = QPushButton("Start")
        self.set_linear_end_button = QPushButton("End")
        self.set_linear_start_button.setToolTip("Capture the current stage position as the start.")
        self.set_linear_end_button.setToolTip("Capture the current stage position as the end.")
        self.generate_line_button = QPushButton("Generate line")
        self.linear_group = QGroupBox("Linear FoVs")
        linear_layout = QVBoxLayout(self.linear_group)
        endpoints = QGridLayout()
        endpoints.addWidget(self.set_linear_start_button, 0, 0, alignment=Qt.AlignVCenter)
        endpoints.addWidget(self.linear_start_label, 0, 1)
        endpoints.addWidget(self.set_linear_end_button, 1, 0, alignment=Qt.AlignVCenter)
        endpoints.addWidget(self.linear_end_label, 1, 1)
        endpoints.setColumnStretch(1, 1)
        linear_layout.addLayout(endpoints)
        linear_layout.addWidget(self.linear_spacing_label)
        linear_layout.addWidget(self.generate_line_button)
        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setMinimumSectionSize(28)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        for column in range(1, len(self.COLUMNS)):
            self.table.horizontalHeader().setSectionResizeMode(column, QHeaderView.Stretch)
        self.table.setMinimumWidth(0)
        self.table.setFixedHeight(140)
        buttons = QGridLayout()
        buttons.addWidget(self.remove_button, 0, 0)
        buttons.addWidget(self.clear_button, 0, 1)
        layout = QVBoxLayout(self)
        hint = muted_label("Move the stage using the controls on the left.")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        layout.addWidget(self.current_label)
        layout.addWidget(self.mode_combo)
        layout.addWidget(self.add_button)
        layout.addWidget(self.linear_group)
        layout.addLayout(buttons)
        layout.addWidget(self.table)
        layout.addWidget(self.initialise_button)
        layout.addWidget(self.status_label)
        self.add_button.clicked.connect(lambda: self._capture("add"))
        self.set_linear_start_button.clicked.connect(lambda: self._capture("start"))
        self.set_linear_end_button.clicked.connect(lambda: self._capture("end"))
        self.generate_line_button.clicked.connect(self._generate_linear_fovs)
        self.remove_button.clicked.connect(self._remove_last_fov)
        self.clear_button.clicked.connect(self._clear_fovs)
        self.initialise_button.clicked.connect(self._initialise_fovs)
        self.mode_combo.currentIndexChanged.connect(self._sync_buttons)
        self.controller.stage_coordinates_received.connect(self.update_current_coordinate)
        self.controller.stage_status_received.connect(self.update_stage_status)
        self.controller.operation_status_received.connect(self.update_operation_status)
        self.controller.strategy_status_received.connect(self.update_strategy_status)
        self.controller.fovs_received.connect(self.update_initialised_fovs)
        self.controller.request_error.connect(self._show_request_error)
        self._sync_buttons()

    def _capture(self, action: str) -> None:
        if self._running or self._moving or self._pending_capture is not None:
            return
        self._pending_capture = action
        self.status_label.setText("Reading current stage position…")
        self._sync_buttons()
        self.controller.refresh_stage()

    def update_current_coordinate(self, payload: dict) -> None:
        self.current_coordinate = self._validated_coordinate_payload(payload.get("coordinate"))
        self.current_label.setText(
            "Current stage: unavailable" if self.current_coordinate is None
            else f"Current stage: {self._format_coordinate(self.current_coordinate)}"
        )
        if "stage" in payload:
            self.update_stage_status(payload["stage"])
        action, self._pending_capture = self._pending_capture, None
        if action and not self._running and not self._moving:
            if self.current_coordinate is None:
                self.status_label.setText("Stage position unavailable; no FoV was captured.")
            elif action == "add":
                self._append_coordinate(self.current_coordinate)
                self._mark_changed()
            else:
                coordinate = dict(self.current_coordinate)
                if action == "start":
                    self.linear_start = coordinate
                    self._show_endpoint(self.linear_start_label, coordinate)
                else:
                    self.linear_end = coordinate
                    self._show_endpoint(self.linear_end_label, coordinate)
                self.status_label.setText(f"Captured linear {action}.")
        self._sync_buttons()

    def update_stage_status(self, payload: dict) -> None:
        value = payload.get("camera_fov_step_size")
        if isinstance(value, int | float) and not isinstance(value, bool) and isfinite(value) and value > 0:
            self.camera_fov_step_size = float(value)
            self.linear_spacing_label.setText(f"spacing: {value:.3f} µm")
        else:
            self.camera_fov_step_size = None
            self.linear_spacing_label.setText("spacing: unavailable")
        self._sync_buttons()

    def update_operation_status(self, payload: dict) -> None:
        if payload.get("kind") == "stage_movement":
            self._moving = payload.get("state") == "running"
            if self._moving:
                self._pending_capture = None
            self._sync_buttons()

    def update_strategy_status(self, payload: dict) -> None:
        self._running = bool(payload.get("running"))
        if self._running:
            self._pending_capture = None
        self._sync_buttons()

    def update_initialised_fovs(self, fovs: list[dict]) -> None:
        self.status_label.setText(f"Initialised {len(fovs)} FoV(s).")

    def _append_coordinate(self, coordinate: dict) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        for column, value in enumerate((str(row), *(f"{coordinate[axis]:.3f}" for axis in ("x", "y", "z")))):
            self.table.setItem(row, column, QTableWidgetItem(value))

    def _generate_linear_fovs(self) -> None:
        if self.linear_start is None or self.linear_end is None or self.camera_fov_step_size is None:
            return
        coordinates = CoordinateFactory(dfov=self.camera_fov_step_size).make_grid(
            start=Coordinate(**self.linear_start), stop=Coordinate(**self.linear_end),
        )
        for coordinate in coordinates:
            self._append_coordinate(dict(x=coordinate.x, y=coordinate.y, z=coordinate.z))
        self._mark_changed()
        self.status_label.setText(f"Generated {len(coordinates)} linear FoV(s); initialise to apply.")

    def _remove_last_fov(self) -> None:
        if self.table.rowCount():
            self.table.removeRow(self.table.rowCount() - 1)
            self._mark_changed()

    def _clear_fovs(self) -> None:
        self.table.setRowCount(0)
        self.linear_start = self.linear_end = None
        self.linear_start_label.setText("X: -\nY: -\nZ: -")
        self.linear_end_label.setText("X: -\nY: -\nZ: -")
        self.linear_start_label.setToolTip("")
        self.linear_end_label.setToolTip("")
        self._mark_changed()

    def _mark_changed(self) -> None:
        self.status_label.setText(f"{self.table.rowCount()} FoV(s) staged; initialise to apply.")
        self._sync_buttons()

    def _initialise_fovs(self) -> None:
        if self.table.rowCount():
            self.status_label.setText("Initialising FoVs…")
            self.controller.initialise_fovs(fovs=self._fov_payload(), use_autofocus=False)

    def _fov_payload(self) -> list[dict]:
        return [
            dict(fov_id=row, x=float(self.table.item(row, 1).text()),
                 y=float(self.table.item(row, 2).text()), z=float(self.table.item(row, 3).text()),
                 channel_id=0)
            for row in range(self.table.rowCount())
        ]

    def _sync_buttons(self) -> None:
        editable = not self._running and not self._moving and self._pending_capture is None
        manual = self.mode_combo.currentIndex() == 0
        self.add_button.setVisible(manual)
        self.linear_group.setVisible(not manual)
        self.mode_combo.setEnabled(editable)
        for button in (self.add_button, self.set_linear_start_button, self.set_linear_end_button):
            button.setEnabled(editable)
        self.generate_line_button.setEnabled(
            editable and self.linear_start is not None and self.linear_end is not None
            and self.camera_fov_step_size is not None
        )
        for button in (self.remove_button, self.clear_button, self.initialise_button):
            button.setEnabled(editable and self.table.rowCount() > 0)

    def _show_error(self, error: str) -> None:
        self._pending_capture = None
        self.status_label.setText(error)
        self._sync_buttons()

    def _show_request_error(self, command: GuiCommandType, error: str) -> None:
        if command in {GuiCommandType.STAGE_GET_COORDINATES, GuiCommandType.FOV_INITIALISE}:
            self._show_error(error)

    @staticmethod
    def _show_endpoint(label: QLabel, coordinate: dict) -> None:
        text = "\n".join(f"{axis.upper()}: {coordinate[axis]:.3f}" for axis in ("x", "y", "z"))
        label.setText(text)
        label.setToolTip(text)

    @staticmethod
    def _format_coordinate(coordinate: dict) -> str:
        return ", ".join(f"{axis}={coordinate[axis]:.3f}" for axis in ("x", "y", "z"))

    @staticmethod
    def _validated_coordinate_payload(coordinate: object) -> dict[str, float] | None:
        if not isinstance(coordinate, dict):
            return None
        values = {axis: coordinate.get(axis) for axis in ("x", "y", "z")}
        if any(not isinstance(value, int | float) or isinstance(value, bool) or not isfinite(value)
               for value in values.values()):
            return None
        return {axis: float(value) for axis, value in values.items()}


class StrategySetupPanel(QGroupBox):
    """Strategy selection and lifecycle shell."""

    def __init__(self, controller, parent: QWidget | None = None):
        super().__init__("Strategy Setup", parent)
        self.controller = controller
        self.source_combo = QComboBox()
        self.source_combo.addItems(["Fixed strategy", "AutoStrat"])
        self._autostrat_enabled = False
        self.auth_label = QLabel("AutoStrat disabled: no API key supplied. Restart the GUI and enter a key to enable it. Fixed strategies remain available.")
        self.auth_label.setWordWrap(True)
        self._generation_id: str | None = None
        self._generation_busy = False
        self._generation_pending = False
        self._generated_prompt = ""
        self.generation_timer = QTimer(self)
        self.generation_timer.setInterval(500)
        self.generation_timer.timeout.connect(self._poll_generation)
        self.auto_group = QWidget()
        auto_layout = QVBoxLayout(self.auto_group)
        auto_layout.setContentsMargins(0, 0, 0, 0)
        self.prompt_input = QPlainTextEdit()
        self.prompt_input.setPlaceholderText("Describe the imaging strategy and its stopping condition…")
        self.prompt_input.setFixedHeight(120)
        self.generate_button = QPushButton("Generate strategy")
        self.cancel_generation_button = QPushButton("Cancel generation")
        self.generation_label = QLabel("Generate, review the strategy code, then Set Strategy. Generation does not run hardware.")
        self.generation_label.setWordWrap(True)
        self.dsl_output = QPlainTextEdit()
        self.dsl_output.setReadOnly(True)
        self.dsl_output.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.dsl_output.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
        self.dsl_output.setFixedHeight(180)
        self.expand_dsl_button = QPushButton("Expand strategy code")
        self.diagnostics_toggle = QCheckBox("Show generation diagnostics")
        self.diagnostics_output = QPlainTextEdit()
        self.diagnostics_output.setReadOnly(True)
        self.diagnostics_output.setFixedHeight(150)
        self.diagnostics_output.hide()
        for widget in (self.prompt_input, self.generate_button, self.cancel_generation_button,
                       self.generation_label,
                       self.dsl_output, self.expand_dsl_button, self.diagnostics_toggle,
                       self.diagnostics_output):
            auto_layout.addWidget(widget)
        self.strategy_combo = QComboBox()
        self.strategy_combo.setEnabled(False)
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
        self._status_pending = False
        self.status_timer = QTimer(self)
        self.status_timer.setInterval(1000)
        self.status_timer.timeout.connect(self._poll_status)

        form = QFormLayout()
        form.addRow("Strategy", self.strategy_combo)
        form.addRow("File", self.file_label)
        form.addRow("Notes", self.notes_label)
        self.fixed_group = QWidget()
        self.fixed_group.setLayout(form)

        buttons = QGridLayout()
        buttons.addWidget(self.set_button, 0, 0, 1, 2)
        buttons.addWidget(self.start_button, 1, 0)
        buttons.addWidget(self.stop_button, 1, 1)

        layout = QVBoxLayout()
        layout.addWidget(self.source_combo)
        layout.addWidget(self.auth_label)
        layout.addWidget(self.fixed_group)
        layout.addWidget(self.auto_group)
        layout.addLayout(buttons)
        layout.addWidget(self.status_label)
        self.setLayout(layout)

        self.source_combo.currentIndexChanged.connect(self._source_changed)
        self.generate_button.clicked.connect(self._generate)
        self.cancel_generation_button.clicked.connect(self._cancel_generation)
        self.prompt_input.textChanged.connect(self._prompt_changed)
        self.expand_dsl_button.clicked.connect(self._expand_dsl)
        self.diagnostics_toggle.toggled.connect(self.diagnostics_output.setVisible)
        self.controller.strategy_generation_received.connect(self.update_generation)
        self.controller.autostrat_configuration_received.connect(self._update_autostrat_configuration)

        self.set_button.clicked.connect(self._set_strategy)
        self.start_button.clicked.connect(self.controller.start_strategy)
        self.stop_button.clicked.connect(self.controller.stop_strategy)
        self.strategy_combo.currentIndexChanged.connect(self._show_selected_strategy)
        self.controller.strategies_received.connect(self.update_strategies)
        self.controller.strategy_status_received.connect(self.update_status)
        self.controller.request_error.connect(self._show_request_error)
        self._sync_controls(strategy_status={})
        self.controller.refresh_strategies()

    def _source_changed(self) -> None:
        self.update_status(self._last_strategy_status)

    def _update_autostrat_configuration(self, payload: dict) -> None:
        self._autostrat_enabled = bool(payload.get("enabled"))
        if not self._autostrat_enabled:
            self._generation_id = None
        self._sync_controls(self._last_strategy_status)

    def _prompt_changed(self) -> None:
        if self.prompt_input.toPlainText().strip() != self._generated_prompt:
            self._generation_id = None
            if self._generated_prompt:
                self.generation_label.setText("Prompt changed; generate again before setting this strategy.")
        self._sync_controls(self._last_strategy_status)

    def _generate(self) -> None:
        prompt = self.prompt_input.toPlainText().strip()
        if not self._autostrat_enabled or not prompt or self._generation_busy:
            return
        self._generated_prompt = prompt
        self._generation_id = None
        self._generation_busy = True
        self._generation_pending = True
        self.dsl_output.clear()
        self.diagnostics_output.clear()
        self.generation_label.setText("Generating and verifying in the background…")
        self._sync_controls(self._last_strategy_status)
        self.controller.generate_strategy(prompt)

    def _poll_generation(self) -> None:
        if not self._generation_pending:
            self._generation_pending = True
            self.controller.refresh_strategy_generation()

    def _cancel_generation(self) -> None:
        if self._generation_busy:
            self.cancel_generation_button.setEnabled(False)
            self.generation_label.setText("Cancelling generation…")
            self.controller.cancel_strategy_generation()

    def update_generation(self, payload: dict) -> None:
        self._generation_pending = False
        self._generation_busy = payload.get("state") == "running"
        if self._generation_busy:
            self.generation_timer.start()
            self.generation_label.setText("Generating and verifying in the background…")
        else:
            self.generation_timer.stop()
            accepted = bool(payload.get("accepted"))
            self._generation_id = payload.get("operation_id") if accepted else None
            self.dsl_output.setPlainText(payload.get("dsl") or "")
            diagnostics = payload.get("diagnostics") or payload.get("error") or ""
            self.diagnostics_output.setPlainText(diagnostics)
            if payload.get("state") == "cancelled":
                self.generation_label.setText("Generation cancelled; no strategy was installed.")
            else:
                self.generation_label.setText(
                    "Accepted. Review the strategy code, then press Set Strategy."
                    if accepted else "Generation failed. See diagnostics; no strategy was installed."
                )
            if not accepted:
                self.diagnostics_toggle.setChecked(True)
        self._sync_controls(self._last_strategy_status)

    def _expand_dsl(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("AutoStrat strategy code — review only")
        editor = QPlainTextEdit(dialog)
        editor.setReadOnly(True)
        editor.setFont(self.dsl_output.font())
        editor.setLineWrapMode(QPlainTextEdit.NoWrap)
        editor.setPlainText(self.dsl_output.toPlainText())
        layout = QVBoxLayout(dialog)
        layout.addWidget(editor)
        dialog.resize(900, 650)
        dialog.exec_()

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

    def _poll_status(self) -> None:
        if not self._status_pending:
            self._status_pending = True
            self.controller.refresh_strategy_status()

    def update_status(self, payload: dict) -> None:
        self._status_pending = False
        if payload.get("running"):
            self.status_timer.start()
        else:
            self.status_timer.stop()
        self._last_strategy_status = payload
        name = payload.get("name") or "-"
        state = "stopped" if payload.get("stopped") else "running" if payload.get("running") else "not running"
        initialised = "ready" if payload.get("is_initialised") else "initialise FoVs to prepare"
        fovs = "FoVs ready" if payload.get("fovs_initialised") else "FoVs not ready"
        self.status_label.setText(f"{name}: {initialised}, {state}, {fovs}.")
        if payload.get("started") and not payload.get("running"):
            self.status_label.setText(self.status_label.text() + " Press Set Strategy to prepare a fresh run.")
        selected = self._selected_strategy()
        if self.source_combo.currentIndex() == 0 and selected and selected.get("name") != payload.get("name"):
            self.status_label.setText(
                self.status_label.text() + f" Select Set Strategy to load {selected['name']}."
            )
        self._sync_controls(strategy_status=payload)

    def _set_strategy(self) -> None:
        if self.source_combo.currentIndex() == 1:
            if self._generation_id is not None:
                self.controller.set_generated_strategy(self._generation_id)
            return
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
        elif self._last_strategy_status:
            self.update_status(self._last_strategy_status)
        self._sync_controls(strategy_status=self._last_strategy_status)

    def _sync_controls(self, strategy_status: dict) -> None:
        running = bool(strategy_status.get("running"))
        started = bool(strategy_status.get("started"))
        stopped = bool(strategy_status.get("stopped"))
        selected = self._selected_strategy()
        automatic = self.source_combo.currentIndex() == 1
        self.auth_label.setVisible(automatic and not self._autostrat_enabled)
        self.auto_group.setEnabled(self._autostrat_enabled)
        self.fixed_group.setVisible(not automatic)
        self.auto_group.setVisible(automatic)
        self.prompt_input.setReadOnly(self._generation_busy)
        self.generate_button.setEnabled(not self._generation_busy and bool(self.prompt_input.toPlainText().strip()))
        self.cancel_generation_button.setEnabled(self._generation_busy)
        self.expand_dsl_button.setEnabled(bool(self.dsl_output.toPlainText()))
        matches_installed = (
            self._generation_id is not None and self._generation_id == strategy_status.get("generation_id")
            if automatic else selected is not None and selected.get("name") == strategy_status.get("name")
        )
        can_start = (
            bool(strategy_status.get("is_initialised"))
            and (not automatic or self._autostrat_enabled)
            and matches_installed
            and not started
            and not stopped
            and not running
        )
        can_set = self._autostrat_enabled and self._generation_id is not None if automatic else selected is not None and not selected.get("error")
        self.set_button.setEnabled(can_set and not running)
        self.strategy_combo.setEnabled(self.strategy_combo.count() > 0 and not running)
        self.start_button.setEnabled(can_start)
        self.stop_button.setEnabled(running)

    def _selected_strategy(self) -> dict | None:
        index = self.strategy_combo.currentIndex()
        if index < 0:
            return None
        data = self.strategy_combo.itemData(index)
        return data if isinstance(data, dict) else None

    def _show_error(self, error: str) -> None:
        self._status_pending = False
        self.status_label.setText(error)

    def _show_request_error(self, command: GuiCommandType, error: str) -> None:
        if command in {
            GuiCommandType.STRATEGY_LIST,
            GuiCommandType.STRATEGY_SET,
            GuiCommandType.STRATEGY_START,
            GuiCommandType.STRATEGY_STOP,
            GuiCommandType.STRATEGY_STATUS,
            GuiCommandType.AUTOSTRAT_CONFIGURE,
        }:
            self._show_error(error)
