from __future__ import annotations

from math import isfinite
from textwrap import fill

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QVBoxLayout,
    QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

_PLOT_BACKGROUND = "#262930"
_AXES_BACKGROUND = "#202328"
_PLOT_TEXT = "#d9dde5"
_PLOT_BORDER = "#606774"
_MEASURED_COLOR = "#7bc8f6"
_SPLINE_COLOR = "#ffb66e"
_SAMPLE_COLOR = "#f28eb1"


class FocusCurves(QWidget):
    """Plot reference, latest sample crosses and both Pearson attempts.

    Alignment and score-scaling controls affect display only, including for
    rejected candidates. Undefined correlations show a reason instead of a
    fabricated curve; the sampled points remain available for inspection.
    """

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
