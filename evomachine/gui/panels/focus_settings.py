"""Mode-specific strategy editors, also reused by manual reference experiments."""

from __future__ import annotations

from copy import deepcopy
import numpy as np

from PyQt5.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from evomachine.acquisition import FrameAcquisitionSettings
from evomachine.focus_registration import ShiftSearchConfig
from evomachine.referencefocus import SamplingPlan
from evomachine.gui.panels.common import NoWheelComboBox
from evomachine.gui.panels.config_dialog import ConfigFieldSpec as F, ConfigForm
from evomachine.types import FilterWheelType, LEDType


METRICS = {
    "SQUARED_GRAD_AVG": {"threshold": 0.0},
    "LAPLACIAN_VAR": {},
    "STEEL": {"rowshift": 1, "colshift": 1, "normalise": True},
    "BANDPASS_FFT": {
        "cell_radius": 20.0,
        "downsample": 4,
        "order": 4,
        "saturation": None,
        "normalize_brightness": True,
    },
}
CRISP_DEFAULTS = dict(
    averaging=5,
    led_intensity=70,
    lock_range=0.1,
    loop_gain=10,
    update_rate=10,
    objective_na=0.9,
    min_snr=2.0,
    min_error=100,
)


def _group(layout, title, widget):
    box = QGroupBox(title)
    body = QVBoxLayout(box)
    body.addWidget(widget)
    layout.addWidget(box)
    return widget


def _field(key, value, **kwargs):
    kind = "bool" if isinstance(value, bool) else "int" if isinstance(value, int) else "float"
    return F(key.replace("_", " ").capitalize(), key, value, kind=kind, **kwargs)


class MetricEditor(QWidget):
    def __init__(self, config):
        super().__init__()
        self.algorithm = NoWheelComboBox()
        self.algorithm.addItems(METRICS)
        self.pages = QStackedWidget()
        self.forms = {}
        for name, defaults in METRICS.items():
            values = defaults | (
                config.get("algorithm_kwargs", {}) if config.get("algorithm") == name else {}
            )
            fields = [_field(key, value) for key, value in values.items() if key != "saturation"]
            if name == "BANDPASS_FFT":
                fields += [
                    F(
                        "Disable saturation clipping",
                        "no_saturation",
                        values["saturation"] is None,
                        kind="bool",
                    ),
                    F(
                        "Saturation ceiling",
                        "saturation",
                        values["saturation"] or 4095,
                        kind="float",
                        minimum=0.001,
                        enabled_when_key="no_saturation",
                        enabled_when_value=False,
                    ),
                ]
            form = ConfigForm(fields)
            self.forms[name] = form
            self.pages.addWidget(form)
        self.algorithm.currentIndexChanged.connect(self.pages.setCurrentIndex)
        self.algorithm.setCurrentText(config.get("algorithm", "SQUARED_GRAD_AVG"))
        layout = QVBoxLayout(self)
        layout.addWidget(self.algorithm)
        layout.addWidget(self.pages)

    def values(self):
        name = self.algorithm.currentText()
        kwargs = self.forms[name].values()
        if name == "BANDPASS_FFT":
            if kwargs.pop("no_saturation"):
                kwargs["saturation"] = None
        return dict(algorithm=name, algorithm_kwargs=kwargs)


class FrameEditor(ConfigForm):
    def __init__(self, config=None):
        config = config or dict(exposure=200, leds={"LED_450_NM": 29}, filter_wheel=None)
        leds = config.get("leds")
        fields = [
            F(
                "Use current camera exposure",
                "current_exposure",
                config.get("exposure") is None,
                kind="bool",
            ),
            F(
                "Exposure (ms)",
                "exposure",
                config.get("exposure") or 200,
                kind="float",
                minimum=1,
                maximum=1000,
                enabled_when_key="current_exposure",
                enabled_when_value=False,
            ),
            F(
                "Filter wheel",
                "filter_wheel",
                config.get("filter_wheel") or "NONE",
                kind="choice",
                choices=("NONE", *(value.name for value in FilterWheelType)),
            ),
            F("Use current LED states", "current_leds", leds is None, kind="bool"),
        ]
        for led in LEDType:
            fields += [
                F(
                    f"Use {led.name}",
                    f"use_{led.name}",
                    leds is not None and led.name in leds,
                    kind="bool",
                    enabled_when_key="current_leds",
                    enabled_when_value=False,
                ),
                F(
                    f"{led.name} brightness (%)",
                    led.name,
                    (leds or {}).get(led.name, 29),
                    kind="float",
                    minimum=0,
                    maximum=100,
                    enabled_when_key=f"use_{led.name}",
                ),
            ]
        super().__init__(fields)

    def values(self):
        values = super().values()
        return dict(
            exposure=None if values["current_exposure"] else values["exposure"],
            filter_wheel=None if values["filter_wheel"] == "NONE" else values["filter_wheel"],
            leds=None
            if values["current_leds"]
            else {led.name: values[led.name] for led in LEDType if values[f"use_{led.name}"]},
        )


class AcquisitionEditor(ConfigForm):
    def __init__(self, config=None):
        values = dict(FrameAcquisitionSettings().__dict__) | (config or {})
        super().__init__([_field(key, value) for key, value in values.items()])


class SoftwareEditor(QWidget):
    def __init__(self, config=None):
        super().__init__()
        config = config or {}
        layout = QVBoxLayout(self)
        self.scan = ConfigForm(
            [
                F(
                    "Range ±Z (µm)",
                    "rel_range",
                    config.get("rel_range", 50),
                    kind="int",
                    minimum=1,
                    maximum=1999,
                ),
                F(
                    "Z step (µm)",
                    "step_size",
                    config.get("step_size", 5),
                    kind="int",
                    minimum=1,
                    maximum=1999,
                ),
            ]
        )
        _group(layout, "Scan", self.scan)
        self.metric = _group(layout, "Image score", MetricEditor(config))
        self.frames = QTabWidget()
        for frame in config.get("focus_frames", [None]):
            self.add_frame(frame)
        _group(layout, "Images captured at each Z", self.frames)
        buttons = QHBoxLayout()
        add, remove = QPushButton("Add image"), QPushButton("Remove image")
        add.clicked.connect(lambda: self.add_frame())
        remove.clicked.connect(self.remove_frame)
        buttons.addWidget(add)
        buttons.addWidget(remove)
        layout.addLayout(buttons)
        self.acquisition = _group(
            layout, "Acquisition", AcquisitionEditor(config.get("acquisition_settings"))
        )

    def add_frame(self, frame=None):
        self.frames.addTab(FrameEditor(frame), f"Image {self.frames.count() + 1}")

    def remove_frame(self):
        if self.frames.count() > 1:
            self.frames.removeTab(self.frames.currentIndex())

    def values(self):
        values = self.scan.values() | self.metric.values()
        if values["step_size"] > values["rel_range"]:
            raise ValueError("Software-focus step must not exceed its range.")
        return values | dict(
            focus_frames=[self.frames.widget(i).values() for i in range(self.frames.count())],
            acquisition_settings=self.acquisition.values(),
        )


class SamplingEditor(QWidget):
    """Generate image Z positions or preserve an explicitly supplied custom list.

    Infer generated controls only when they reproduce every saved position;
    otherwise keep Custom mode so opening a dialog cannot change an irregular
    scan. Preview includes the exact image count and generated positions.
    """

    def __init__(self, offsets):
        super().__init__()
        self.mode = NoWheelComboBox()
        self.mode.addItems(["Generated positions", "Custom positions"])
        plan = self._infer_plan(offsets)
        self.form = ConfigForm(
            [
                F(
                    "Outer range ±Z (µm)",
                    "half_range",
                    plan.half_range if plan else 20,
                    kind="float",
                    minimum=0.001,
                ),
                F(
                    "Outer image step (µm)",
                    "step",
                    plan.step if plan else 5,
                    kind="float",
                    minimum=0.001,
                    single_step=0.5,
                ),
                F(
                    "Denser centre",
                    "dense",
                    plan is not None and plan.central_range is not None,
                    kind="bool",
                ),
                F(
                    "Central range ±Z (µm)",
                    "central_range",
                    (plan.central_range if plan else None) or 5,
                    kind="float",
                    minimum=0.001,
                    enabled_when_key="dense",
                ),
                F(
                    "Central image step (µm)",
                    "central_step",
                    plan.central_step
                    if plan and plan.central_range is not None
                    else min(0.5, plan.step / 2)
                    if plan
                    else 0.5,
                    kind="float",
                    minimum=0.001,
                    single_step=0.1,
                    enabled_when_key="dense",
                ),
            ]
        )
        self.custom = QLineEdit(", ".join(f"{value:g}" for value in offsets))
        self.custom.setPlaceholderText("Z offsets in µm, separated by commas")
        self.pages = QStackedWidget()
        self.pages.addWidget(self.form)
        self.pages.addWidget(self.custom)
        self.preview = QLabel()
        self.preview.setWordWrap(True)
        self.mode.setCurrentIndex(0 if plan else 1)
        layout = QVBoxLayout(self)
        layout.addWidget(self.mode)
        layout.addWidget(self.pages)
        layout.addWidget(self.preview)
        self.mode.currentIndexChanged.connect(self._refresh)
        self.custom.textChanged.connect(self._refresh)
        for widget in self.form._widgets.values():
            signal = widget.toggled if isinstance(widget, QCheckBox) else widget.valueChanged
            signal.connect(self._refresh)
        self._refresh()

    @staticmethod
    def _infer_plan(offsets):
        values = np.asarray(offsets, dtype=float)
        if values.size < 5 or not np.allclose(values, -values[::-1]):
            return None
        positive = values[values > 0]
        if not np.any(values == 0) or not positive.size:
            return None
        coarse = float(np.max(np.diff(values)))
        fine = float(positive[0])
        end = fine
        for value in positive[1:]:
            if not np.isclose(value - end, fine):
                break
            end = float(value)
        dense = fine < coarse and not np.isclose(fine, coarse)
        plan = SamplingPlan(float(values[-1]), coarse, end if dense else None, fine)
        generated = plan.offsets()
        return plan if len(generated) == len(values) and np.allclose(generated, values) else None

    def values(self):
        if self.mode.currentIndex() == 0:
            values = self.form.values()
            dense = values.pop("dense")
            if not dense:
                values["central_range"] = None
            return list(SamplingPlan(**values).offsets())
        values = [float(item.strip()) for item in self.custom.text().split(",")]
        if len(values) < 5 or not np.all(np.isfinite(values)) or np.any(np.diff(values) <= 0):
            raise ValueError("Custom sampling needs at least five increasing, finite Z offsets.")
        return values

    def _refresh(self, *_):
        self.pages.setCurrentIndex(self.mode.currentIndex())
        try:
            values = self.values()
            self.preview.setText(
                f"{len(values)} images: " + ", ".join(f"{value:g}" for value in values)
            )
        except (ValueError, TypeError) as error:
            self.preview.setText(str(error))


class ReferenceEditor(QWidget):
    def __init__(self, config=None):
        super().__init__()
        config = config or {}
        layout = QVBoxLayout(self)
        self.calibration = _group(
            layout,
            "Initial calibration — saved focal Z defines zero",
            ConfigForm(
                [
                    F(
                        "Range ±Z (µm)",
                        "calibration_range",
                        config.get("calibration_range", 40),
                        kind="float",
                        minimum=0.001,
                    ),
                    F(
                        "Z step (µm)",
                        "calibration_step",
                        config.get("calibration_step", 1),
                        kind="float",
                        minimum=0.001,
                    ),
                    F(
                        "Automatic smoothing (GCV)",
                        "automatic",
                        config.get("smoothing", 0) is None,
                        kind="bool",
                    ),
                    F(
                        "Smoothing lambda",
                        "smoothing",
                        config.get("smoothing") or 0,
                        kind="float",
                        minimum=0,
                        enabled_when_key="automatic",
                        enabled_when_value=False,
                    ),
                ]
            ),
        )
        hint = QLabel(
            "Sampling selects where images are captured. Matching tests Z corrections against the spline; it acquires no extra images."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.sparse_samples = _group(
            layout,
            "Sparse images",
            SamplingEditor(config.get("sample_offsets", SamplingPlan().offsets())),
        )
        self.recovery_samples = _group(
            layout,
            "Recovery images",
            SamplingEditor(config.get("recovery_offsets", SamplingPlan(central_range=5).offsets())),
        )
        self.samples = _group(
            layout,
            "Sampling and recovery",
            ConfigForm(
                [
                    F(
                        "Allow highest-peak reset after failed recovery",
                        "allow_peak_reset",
                        config.get("allow_peak_reset", False),
                        kind="bool",
                    ),
                ]
            ),
        )
        hint = QLabel(
            "Peak reset changes the chosen focal plane. In strategies, it affects only the failed FoV."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.searches = {}
        for key, title, defaults in (
            (
                "search",
                "Sparse matching — candidate corrections",
                vars(ShiftSearchConfig(shift_min=-20, shift_max=20, shift_step=0.1)),
            ),
            (
                "recovery_search",
                "Recovery matching — candidate corrections",
                vars(ShiftSearchConfig(shift_min=-20, shift_max=20, shift_step=0.1)),
            ),
        ):
            values = dict(defaults) | config.get(key, {})
            self.searches[key] = _group(
                layout,
                title,
                ConfigForm(
                    [
                        _field(k, float(v), single_step=0.01 if k.startswith("min_") else 0.1)
                        for k, v in values.items()
                    ]
                ),
            )
        self.metric = _group(layout, "Image score", MetricEditor(config))
        self.frame = _group(
            layout,
            "Focus image — whole image is always scored",
            FrameEditor(config.get("focus_frame")),
        )
        self.acquisition = _group(
            layout, "Acquisition", AcquisitionEditor(config.get("acquisition_settings"))
        )

    def values(self):
        values = self.calibration.values()
        values["smoothing"] = None if values.pop("automatic") else values["smoothing"]
        values |= self.samples.values() | self.metric.values()
        for key, title, editor in (
            ("sample_offsets", "Sparse images", self.sparse_samples),
            ("recovery_offsets", "Recovery images", self.recovery_samples),
        ):
            try:
                values[key] = editor.values()
            except (ValueError, TypeError) as error:
                raise ValueError(f"{title}: {error}") from error
        for key, form in self.searches.items():
            values[key] = form.values()
            try:
                ShiftSearchConfig(**values[key])
            except ValueError as error:
                raise ValueError(
                    f"{'Sparse' if key == 'search' else 'Recovery'} matching: {error}"
                ) from error
        if not 0 < values["calibration_step"] <= values["calibration_range"] / 2:
            raise ValueError("Calibration needs a positive Z step and at least five planes.")
        return values | dict(
            focus_frame=self.frame.values(), acquisition_settings=self.acquisition.values()
        )


class StandardEditor(QWidget):
    def __init__(self, config):
        super().__init__()
        layout = QVBoxLayout(self)
        defaults = dict(
            use_autofocus=True,
            refocus=True,
            refocus_using_software_focus=True,
            refocus_on_all_fovs=False,
            max_refocus_trials=10,
            out_of_focus_wait_s=10.0,
            post_autofocus_wait_s=3.0,
            post_move_wait_s=1.0,
            toggle_autofocus_on_channel_change=True,
            initialise_from_first_fov_z=False,
        )
        self.navigation = _group(
            layout,
            "Movement and recovery",
            ConfigForm([_field(key, config.get(key, value)) for key, value in defaults.items()]),
        )
        flags = dict(
            lock_autofocus_during_arrival_move=True,
            lock_autofocus_on_fov=True,
            lock_autofocus_during_departure_move=True,
            run_software_focus_on_arrival=False,
        )
        self.policy = _group(
            layout,
            "FoV arrival/departure",
            ConfigForm(
                [
                    _field(key, config.get("default_fov_config", {}).get(key, value))
                    for key, value in flags.items()
                ]
            ),
        )
        self.crisp = _group(
            layout,
            "CRISP calibration",
            ConfigForm(
                [
                    _field(key, (config.get("autofocus") or {}).get(key, value))
                    for key, value in CRISP_DEFAULTS.items()
                ]
            ),
        )
        self.software = _group(
            layout, "Software autofocus", SoftwareEditor(config.get("software_focus"))
        )

    def values(self):
        return self.navigation.values() | dict(
            default_fov_config=self.policy.values(),
            autofocus=self.crisp.values(),
            software_focus=self.software.values(),
        )


class FocusSettingsDialog(QDialog):
    def __init__(self, title, config, *, global_config=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.per_fov = global_config is not None
        inherited = deepcopy(global_config or config)
        self.inherited_mode = inherited.get("focus_mode", "STANDARD")
        active = inherited | config
        self.mode = NoWheelComboBox()
        if self.per_fov:
            self.mode.addItem("Inherit global", "INHERIT")
        self.mode.addItem("Standard", "STANDARD")
        self.mode.addItem("Reference profile", "REFERENCE_PROFILE")
        selected = config.get("focus_mode", "INHERIT" if self.per_fov else "STANDARD")
        self.mode.setCurrentIndex(max(0, self.mode.findData(selected)))
        self.override = QCheckBox("Override settings for this FoV")
        self.override.setChecked(config.get("override_settings", bool(config)))
        self.override.setVisible(self.per_fov)
        self.pages = QStackedWidget()
        self.standard = StandardEditor(active)
        self.reference = ReferenceEditor(active.get("reference_profile"))
        self.pages.addWidget(self.standard)
        self.pages.addWidget(self.reference)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.pages)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Focus mode"))
        layout.addWidget(self.mode)
        layout.addWidget(self.override)
        layout.addWidget(scroll)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.mode.currentIndexChanged.connect(self._sync)
        self.override.toggled.connect(self._sync)
        self._sync()
        self.resize(740, 750)

    def _sync(self):
        mode = self.mode.currentData()
        if mode == "INHERIT":
            mode = self.inherited_mode
        self.pages.setCurrentIndex(1 if mode == "REFERENCE_PROFILE" else 0)
        self.pages.setEnabled(not self.per_fov or self.override.isChecked())

    def values(self):
        mode = self.mode.currentData()
        if self.per_fov and not self.override.isChecked():
            return {} if mode == "INHERIT" else dict(focus_mode=mode, override_settings=False)
        values = self.standard.values() | dict(
            focus_mode=mode, reference_profile=self.reference.values()
        )
        if self.per_fov:
            values["override_settings"] = True
        return values

    def accept(self):
        try:
            self.values()
        except (ValueError, TypeError) as error:
            QMessageBox.warning(self, "Invalid focus settings", str(error))
            return
        super().accept()


class ReferenceSettingsDialog(QDialog):
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Manual Reference Profile Settings")
        self.editor = ReferenceEditor(config)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.editor)
        layout = QVBoxLayout(self)
        note = QLabel(
            "Scan/search edits keep the accepted reference. Smoothing or metric changes refit cached images for review. Changed imaging settings require a new calibration; the old curve remains visible."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        layout.addWidget(scroll)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.resize(740, 750)

    def values(self):
        return self.editor.values()

    def accept(self):
        try:
            self.values()
        except (ValueError, TypeError) as error:
            QMessageBox.warning(self, "Invalid focus settings", str(error))
            return
        super().accept()
