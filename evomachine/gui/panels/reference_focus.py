from __future__ import annotations

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from evomachine.gui.panels.focus_settings import ReferenceSettingsDialog
from evomachine.gui.panels.common import NoWheelDoubleSpinBox
from evomachine.gui.protocol import GuiCommandType


class FocusCurves(QWidget):
    def __init__(self, parent=None, *, show_match=True):
        super().__init__(parent)
        self.figure = Figure(figsize=(6, 5), constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(330)
        self.reference = self.figure.add_subplot(211 if show_match else 111)
        self.correlation = self.figure.add_subplot(212) if show_match else None
        layout = QVBoxLayout(self)
        layout.addWidget(NavigationToolbar2QT(self.canvas, self))
        layout.addWidget(self.canvas)
        self.update_curves({})

    def update_curves(self, payload):
        self.reference.clear()
        if self.correlation is not None:
            self.correlation.clear()
        curve = payload.get("curve")
        result = payload.get("result") or {}
        if curve:
            self.reference.plot(curve["z"], curve["scores"], "o", markersize=3, label="Measured")
            self.reference.plot(curve["fit_z"], curve["fit_scores"], label="Spline")
            self.reference.axvline(0, linestyle="--", color="black", label="Selected zero")
            self.reference.legend(fontsize="small")
        self.reference.set(
            xlabel="Relative Z (µm)", ylabel="Focus score", title="Reference profile"
        )
        if self.correlation is not None:
            if result.get("shifts"):
                self.correlation.plot(result["shifts"], result["correlations"])
                if result.get("shift") is not None:
                    self.correlation.axvline(result["shift"], linestyle="--", color="black")
            self.correlation.set(
                xlabel="Z correction (µm)", ylabel="Pearson correlation", title="Latest match"
            )
        self.canvas.draw_idle()


class ReferenceFocusPanel(QGroupBox):
    """Independent manual experiments; never edits strategy focus configuration."""

    def __init__(self, controller, parent=None):
        super().__init__("Reference Profile Focus — Manual", parent)
        self.controller = controller
        self.config = {}
        self.ready = self.busy = self.running = self.pending = False
        self.payload = {}
        self.label = QLabel("Initialise devices, manually focus, then calibrate a reference.")
        self.label.setWordWrap(True)
        self.curves = FocusCurves()
        self.buttons = {}
        actions = (
            ("refresh", "Refresh", controller.refresh_reference_focus),
            ("configure", "Configure", self._configure),
            ("calibrate", "Calibrate at current Z", lambda: self._run("calibrate")),
            ("accept", "Accept reference", lambda: controller.edit_reference_focus("accept")),
            ("test", "Test match (no correction)", lambda: self._run("test")),
            ("run", "Run focus", lambda: self._run("run")),
            ("clear", "Clear reference", lambda: controller.edit_reference_focus("clear")),
            ("cancel", "Cancel", controller.cancel_reference_focus),
        )
        grid = QGridLayout()
        for index, (key, text, callback) in enumerate(actions):
            button = QPushButton(text)
            button.clicked.connect(callback)
            self.buttons[key] = button
            grid.addWidget(button, index // 2, index % 2)
        self.automatic = QCheckBox("Automatic spline smoothing")
        self.automatic.setChecked(True)
        self.smoothing = NoWheelDoubleSpinBox()
        self.smoothing.setRange(0, 1e12)
        self.smoothing.setDecimals(4)
        self.smoothing.setEnabled(False)
        self.automatic.toggled.connect(lambda enabled: self.smoothing.setEnabled(not enabled))
        self.refit = QPushButton("Refit cached reference")
        self.refit.clicked.connect(
            lambda: controller.edit_reference_focus(
                "refit", None if self.automatic.isChecked() else self.smoothing.value()
            )
        )
        smoothing = QHBoxLayout()
        smoothing.addWidget(self.automatic)
        smoothing.addWidget(self.smoothing)
        smoothing.addWidget(self.refit)
        layout = QVBoxLayout(self)
        layout.addWidget(self.label)
        layout.addLayout(grid)
        layout.addLayout(smoothing)
        layout.addWidget(self.curves)
        self.timer = QTimer(self)
        self.timer.setInterval(300)
        self.timer.timeout.connect(self._poll)
        controller.reference_focus_received.connect(self.update_status)
        controller.operation_status_received.connect(self._operation)
        controller.lifecycle_status_received.connect(self._lifecycle)
        controller.strategy_status_received.connect(self._strategy)
        controller.request_error.connect(self._error)
        self._sync()
        controller.refresh_reference_focus()

    def _configure(self):
        dialog = ReferenceSettingsDialog(self.config, self)
        if dialog.exec_() == QDialog.Accepted:
            self.controller.configure_reference_focus(dialog.values())

    def _run(self, action):
        self.busy = True
        self.pending = False
        self.timer.start()
        self._sync()
        self.controller.run_reference_focus(action)

    def _poll(self):
        if not self.pending:
            self.pending = True
            self.controller.refresh_reference_operation()

    def _operation(self, payload):
        if payload.get("kind") != "reference_focus":
            return
        self.pending = False
        self.busy = payload.get("state") == "running"
        if not self.busy:
            self.timer.stop()
        self.label.setText(
            str(payload.get("error") or payload.get("message") or payload.get("state"))
        )
        self._sync()

    def _lifecycle(self, payload):
        if "devices_initialised" in payload:
            self.ready = bool(payload["devices_initialised"])
        if "strategy_active" in payload:
            self.running = bool(payload["strategy_active"])
        self._sync()

    def _strategy(self, payload):
        self.running = bool(payload.get("running"))
        self._sync()

    def _error(self, command, error):
        if command in {
            GuiCommandType.REFERENCE_FOCUS_RUN,
            GuiCommandType.REFERENCE_FOCUS_OPERATION,
            GuiCommandType.REFERENCE_FOCUS_CONFIGURE,
            GuiCommandType.REFERENCE_FOCUS_EDIT,
            GuiCommandType.REFERENCE_FOCUS_CANCEL,
        }:
            self.pending = False
            if command == GuiCommandType.REFERENCE_FOCUS_RUN:
                self.busy = False
                self.timer.stop()
            self.label.setText(error)
            self._sync()

    def update_status(self, payload):
        self.payload = payload
        self.config = payload.get("config") or self.config
        smoothing = self.config.get("smoothing")
        self.automatic.setChecked(smoothing is None)
        if smoothing is not None:
            self.smoothing.setValue(smoothing)
        result = payload.get("result") or {}
        curve = payload.get("curve")
        text = (
            "Accepted"
            if payload.get("accepted")
            else "Awaiting acceptance"
            if curve
            else "No reference"
        )
        if curve:
            text += f"; reference Z {curve['focus_z']:g} µm"
        if result:
            text += f"; {result['status']}"
            if not result.get("correction_applied"):
                text += "; no correction applied"
            if result.get("reference_reset"):
                text += "; zero reset to highest peak"
            if result.get("correlation") is not None:
                text += f"; correlation {result['correlation']:.3f}"
            if result.get("failure_reason"):
                text += f"; {result['failure_reason']}"
        self.label.setText(text)
        self.curves.update_curves(payload)
        self._sync()

    def _sync(self):
        available = self.ready and not self.running and not self.busy
        for key, button in self.buttons.items():
            button.setEnabled(available)
        self.buttons["refresh"].setEnabled(not self.busy)
        self.buttons["cancel"].setEnabled(self.busy)
        self.buttons["accept"].setEnabled(
            available and bool(self.payload.get("curve")) and not self.payload.get("accepted")
        )
        for key in ("run", "test"):
            self.buttons[key].setEnabled(available and bool(self.payload.get("accepted")))
        self.refit.setEnabled(available and bool(self.payload.get("curve")))


class StrategyReferenceDialog(QDialog):
    """Review a strategy-owned calibration without changing manual experiment state."""

    def __init__(self, controller, review, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.identifier = review["id"]
        self.decision = None
        self.setWindowTitle(f"Accept strategy reference — FoV {review['fov_id']}")
        self.setWindowModality(Qt.ApplicationModal)
        layout = QVBoxLayout(self)
        self.label = QLabel(
            f"Saved focal Z {review['curve']['focus_z']:g} µm defines zero. "
            "Inspect the curve before imaging continues."
        )
        self.label.setWordWrap(True)
        layout.addWidget(self.label)
        curves = FocusCurves(show_match=False)
        curves.update_curves({"curve": review["curve"]})
        layout.addWidget(curves)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.button(QDialogButtonBox.Ok).setText("Accept and continue")
        self.buttons.button(QDialogButtonBox.Cancel).setText("Reject / stop")
        self.buttons.accepted.connect(lambda: self._decide(True))
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        controller.strategy_focus_received.connect(self._status)
        controller.request_error.connect(self._error)
        self.resize(760, 650)

    def _decide(self, accepted):
        self.decision = accepted
        self.buttons.setEnabled(False)
        self.label.setText("Waiting for the backend…")
        self.controller.decide_strategy_reference(self.identifier, accepted)

    def reject(self):
        self._decide(False)

    def _status(self, payload):
        review = payload.get("review")
        if review is None or review.get("id") != self.identifier:
            if self.decision:
                super().accept()
            else:
                super().reject()

    def _error(self, command, error):
        if command == GuiCommandType.STRATEGY_REFERENCE_DECIDE:
            self.buttons.setEnabled(True)
            self.label.setText(error)
