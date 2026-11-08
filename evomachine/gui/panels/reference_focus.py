from __future__ import annotations

from math import isfinite
from textwrap import fill

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QPalette
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


_PLOT_BACKGROUND = "#262930"
_AXES_BACKGROUND = "#202328"
_PLOT_TEXT = "#d9dde5"
_PLOT_BORDER = "#606774"
_MEASURED_COLOR = "#7bc8f6"
_SPLINE_COLOR = "#ffb66e"
_SAMPLE_COLOR = "#f28eb1"


class FocusCurves(QWidget):
    def __init__(self, parent=None, *, show_match=True):
        super().__init__(parent)
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor(_PLOT_BACKGROUND))
        palette.setColor(QPalette.WindowText, QColor(_PLOT_TEXT))
        palette.setColor(QPalette.Button, QColor(_PLOT_BACKGROUND))
        palette.setColor(QPalette.ButtonText, QColor(_PLOT_TEXT))
        self.setPalette(palette)
        self.setAutoFillBackground(True)
        self.figure = Figure(figsize=(6, 5), constrained_layout=True, facecolor=_PLOT_BACKGROUND)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(330)
        self.reference = self.figure.add_subplot(211 if show_match else 111)
        self.correlation = self.figure.add_subplot(212) if show_match else None
        self.payload = {}
        layout = QVBoxLayout(self)
        if show_match:
            controls = QVBoxLayout()
            self.align_samples = QCheckBox("Align sample crosses to candidate shift")
            self.scale_samples = QCheckBox("Scale crosses to reference scores")
            for checkbox in (self.align_samples, self.scale_samples):
                checkbox.setChecked(True)
                checkbox.toggled.connect(lambda _: self.update_curves(self.payload))
                controls.addWidget(checkbox)
            layout.addLayout(controls)
        layout.addWidget(NavigationToolbar2QT(self.canvas, self))
        layout.addWidget(self.canvas)
        self.update_curves({})

    def update_curves(self, payload):
        self.payload = payload
        self.reference.clear()
        if self.correlation is not None:
            self.correlation.clear()
        curve = payload.get("curve")
        result = payload.get("result") or {}
        if curve:
            self.reference.plot(
                curve["z"],
                curve["scores"],
                "o",
                markersize=3,
                color=_MEASURED_COLOR,
                label="Measured",
            )
            self.reference.plot(
                curve["fit_z"], curve["fit_scores"], color=_SPLINE_COLOR, label="Spline"
            )
            self.reference.axvline(0, linestyle="--", color=_PLOT_TEXT, label="Selected zero")
        self.reference.set(
            xlabel="Relative Z (µm)", ylabel="Focus score", title="Reference profile"
        )
        if self.correlation is not None:
            attempts = result.get("attempts") or ([dict(result, label="Latest")] if result else [])
            if attempts and attempts[-1].get("offsets") and not result.get("reference_reset"):
                latest = attempts[-1]
                aligned = self.align_samples.isChecked() and latest.get("shift") is not None
                scaled = self.scale_samples.isChecked() and latest.get("scaled", False)
                offsets = latest["aligned_offsets"] if aligned else latest["offsets"]
                scores = latest["display_scores"] if scaled else latest["scores"]
                description = "aligned" if aligned else "scan offsets"
                if scaled:
                    description += ", scaled"
                self.reference.plot(
                    offsets,
                    scores,
                    "x",
                    color=_SAMPLE_COLOR,
                    markersize=6,
                    label=f"{latest['label']} samples ({description})",
                )
            drawn = False
            unavailable = []
            for index, attempt in enumerate(attempts):
                color = _MEASURED_COLOR if index == 0 else _SPLINE_COLOR
                correlations = attempt.get("correlations", [])
                finite = sum(value is not None and isfinite(value) for value in correlations)
                if finite:
                    drawn = True
                    status = (
                        "passed"
                        if attempt.get("accepted", result.get("status") == "IN_FOCUS")
                        else "rejected"
                    )
                    self.correlation.plot(
                        attempt["shifts"],
                        correlations,
                        color=color,
                        marker="." if finite < 3 else None,
                        label=f"{attempt['label']} ({status})",
                    )
                    if attempt.get("shift") is not None:
                        self.correlation.axvline(attempt["shift"], linestyle="--", color=color)
                else:
                    unavailable.append(attempt["label"])
            if attempts and attempts[-1].get("shifts"):
                shifts = attempts[-1]["shifts"]
                self.correlation.set_xlim(shifts[0], shifts[-1])
            self.correlation.set(
                xlabel="Z correction (µm)", ylabel="Pearson correlation", title="Latest match"
            )
            if result and not drawn:
                reason = (
                    result.get("failure_reason")
                    or "Scores are constant or the scan does not fit inside the reference."
                )
                self.correlation.text(
                    0.5,
                    0.5,
                    "Correlation unavailable\n" + fill(reason, 60),
                    transform=self.correlation.transAxes,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=_PLOT_TEXT,
                )
            elif unavailable:
                self.correlation.text(
                    0.02,
                    0.95,
                    f"{', '.join(unavailable)}: correlation unavailable",
                    transform=self.correlation.transAxes,
                    va="top",
                    fontsize=8,
                    color=_PLOT_TEXT,
                )
        for axis in self.figure.axes:
            axis.set_facecolor(_AXES_BACKGROUND)
            axis.tick_params(colors=_PLOT_TEXT)
            for text in (
                axis.title,
                axis.xaxis.label,
                axis.yaxis.label,
                axis.xaxis.get_offset_text(),
                axis.yaxis.get_offset_text(),
            ):
                text.set_color(_PLOT_TEXT)
            for spine in axis.spines.values():
                spine.set_color(_PLOT_BORDER)
            if axis.get_legend_handles_labels()[0]:
                axis.legend(
                    fontsize="small",
                    facecolor=_AXES_BACKGROUND,
                    edgecolor=_PLOT_BORDER,
                    labelcolor=_PLOT_TEXT,
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
            "Recalibration required"
            if payload.get("needs_calibration")
            else (
                "Accepted"
                if payload.get("accepted")
                else "Awaiting acceptance"
                if curve
                else "No reference"
            )
        )
        if curve:
            text += f"; reference Z {curve['focus_z']:g} µm"
        if result:
            method = result.get("method") or (
                "recovery" if result.get("recovery_attempted") else "sparse"
            )
            method = method if method == "peak reset" else method + " matching"
            if result["status"] == "IN_FOCUS":
                text += f"\n{'Test passed' if not result.get('correction_applied') else 'Focused'} via {method}"
            else:
                text += (
                    "\nSparse and recovery matching failed"
                    if len(result.get("attempts", [])) > 1
                    else "\nFocus not accepted"
                )
            if not result.get("correction_applied"):
                text += "; no correction applied"
            if result.get("reference_reset"):
                text += "; zero reset to highest peak"
            if result.get("correlation") is not None:
                text += f"; correlation {result['correlation']:.3f}"
            attempts = result.get("attempts", [])
            rejected = [attempt for attempt in attempts if not attempt.get("accepted")]
            for attempt in rejected:
                text += f"\n{attempt['label']}: " + "; ".join(attempt["rejection_reasons"])
            if result.get("failure_reason") and not rejected:
                text += f"\n{result['failure_reason']}"
        if payload.get("message"):
            text += "\n" + payload["message"]
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
            available
            and bool(self.payload.get("curve"))
            and not self.payload.get("accepted")
            and not self.payload.get("needs_calibration")
        )
        for key in ("run", "test"):
            self.buttons[key].setEnabled(available and bool(self.payload.get("accepted")))
        self.refit.setEnabled(
            available
            and bool(self.payload.get("curve"))
            and not self.payload.get("needs_calibration")
        )


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
