from __future__ import annotations

import os

import pytest

if os.environ.get("EVOMACHINE_GUI_RUN_QT_TESTS") != "1":
    pytest.skip("Qt widget tests are opt-in in headless environments.", allow_module_level=True)

from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QApplication, QFileDialog

from evomachine.gui.panels.acquisition import (
    FrameAcquisitionSettingsPanel,
    ManualAcquisitionPanel,
    SavedImageLoaderPanel,
    ZStackSettingsPanel,
)
from evomachine.gui.panels.autofocus import AutofocusPanel
from evomachine.gui.panels.camera import CameraPanel
from evomachine.gui.panels.dmd import DmdPanel
from evomachine.gui.panels.filterwheel import FilterWheelPanel
from evomachine.gui.panels.leds import LedManagerPanel
from evomachine.gui.panels.logs import ApplicationLogPanel
from evomachine.gui.panels.software_focus import SoftwareFocusPanel
from evomachine.gui.panels.stage import StagePanel
from evomachine.gui.panels.strategy import FovSetupPanel, StrategySetupPanel
from evomachine.types import FilterWheelType, LEDType


_QT_APP: QApplication | None = None


class FakeController(QObject):
    stage_coordinates_received = pyqtSignal(dict)
    stage_status_received = pyqtSignal(dict)
    camera_status_received = pyqtSignal(dict)
    acquisition_files_received = pyqtSignal(list)
    acquisition_directory_received = pyqtSignal(dict)
    acquisition_experiments_received = pyqtSignal(dict)
    frame_received = pyqtSignal(dict)
    filter_wheel_status_received = pyqtSignal(dict)
    led_list_received = pyqtSignal(list)
    led_state_received = pyqtSignal(dict)
    dmd_status_received = pyqtSignal(dict)
    dmd_calibration_points_received = pyqtSignal(dict)
    autofocus_status_received = pyqtSignal(dict)
    software_focus_status_received = pyqtSignal(dict)
    operation_status_received = pyqtSignal(dict)
    logs_received = pyqtSignal(dict)
    strategies_received = pyqtSignal(list)
    strategy_status_received = pyqtSignal(dict)
    lifecycle_status_received = pyqtSignal(dict)
    response_error = pyqtSignal(str)
    fovs_received = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self.calls = []

    def refresh_stage(self):
        self.calls.append(("refresh_stage",))

    def refresh_logs(self, after_sequence=0):
        self.calls.append(("refresh_logs", after_sequence))

    def initialise_fovs(self, fovs, use_autofocus=False):
        self.calls.append(("initialise_fovs", fovs, use_autofocus))

    def move_stage_absolute(self, x, y, z):
        self.calls.append(("move_stage_absolute", x, y, z))

    def move_stage_relative(self, dx, dy, dz):
        self.calls.append(("move_stage_relative", dx, dy, dz))

    def move_stage_fov(self, direction, multiplier=1.0):
        self.calls.append(("move_stage_fov", direction, multiplier))

    def stop_stage(self):
        self.calls.append(("stop_stage",))

    def zero_stage(self):
        self.calls.append(("zero_stage",))

    def return_stage_to_origin(self):
        self.calls.append(("return_stage_to_origin",))

    def refresh_camera(self):
        self.calls.append(("refresh_camera",))

    def set_camera_exposure(self, exposure):
        self.calls.append(("set_camera_exposure", exposure))

    def acquire_frame(self, payload=None):
        self.calls.append(("acquire_frame", payload))

    def acquire_z_stack(self, payload=None):
        self.calls.append(("acquire_z_stack", payload))

    def refresh_acquisition_files(self):
        self.calls.append(("refresh_acquisition_files",))

    def refresh_acquisition_experiments(self):
        self.calls.append(("refresh_acquisition_experiments",))

    def create_acquisition_experiment(self, name):
        self.calls.append(("create_acquisition_experiment", name))

    def select_acquisition_experiment(self, name):
        self.calls.append(("select_acquisition_experiment", name))

    def load_acquisition_frame(self, filename, image_transport=None):
        self.calls.append(("load_acquisition_frame", filename, image_transport))

    def refresh_filter_wheel(self):
        self.calls.append(("refresh_filter_wheel",))

    def set_filter_wheel(self, filter_wheel):
        self.calls.append(("set_filter_wheel", filter_wheel))

    def refresh_leds(self):
        self.calls.append(("refresh_leds",))

    def refresh_led_state(self, led):
        self.calls.append(("refresh_led_state", led))

    def set_led(self, led, brightness, duration=None):
        self.calls.append(("set_led", led, brightness, duration))

    def disable_led(self, led):
        self.calls.append(("disable_led", led))

    def disable_all_leds(self):
        self.calls.append(("disable_all_leds",))

    def refresh_dmd(self):
        self.calls.append(("refresh_dmd",))

    def display_dmd_pattern(self, pattern, config=None):
        self.calls.append(("display_dmd_pattern", pattern, config))

    def load_dmd_pattern(self, filename):
        self.calls.append(("load_dmd_pattern", filename))

    def display_loaded_dmd_pattern(self):
        self.calls.append(("display_loaded_dmd_pattern",))

    def calibrate_dmd(self):
        self.calls.append(("calibrate_dmd",))

    def refresh_dmd_calibration_operation(self):
        self.calls.append(("refresh_dmd_calibration_operation",))

    def cancel_dmd_calibration(self):
        self.calls.append(("cancel_dmd_calibration",))

    def load_dmd_calibration(self, filename):
        self.calls.append(("load_dmd_calibration", filename))

    def request_dmd_calibration_points(self):
        self.calls.append(("request_dmd_calibration_points",))

    def refresh_autofocus(self):
        self.calls.append(("refresh_autofocus",))

    def configure_autofocus(self, config=None):
        self.calls.append(("configure_autofocus", config))

    def initialise_autofocus(self, lock_after_initialise=False, config=None):
        self.calls.append(("initialise_autofocus", lock_after_initialise, config))

    def refresh_autofocus_calibration_operation(self):
        self.calls.append(("refresh_autofocus_calibration_operation",))

    def cancel_autofocus_calibration(self):
        self.calls.append(("cancel_autofocus_calibration",))

    def lock_autofocus(self):
        self.calls.append(("lock_autofocus",))

    def unlock_autofocus(self):
        self.calls.append(("unlock_autofocus",))

    def disable_autofocus(self):
        self.calls.append(("disable_autofocus",))

    def refresh_software_focus(self):
        self.calls.append(("refresh_software_focus",))

    def run_software_focus(self):
        self.calls.append(("run_software_focus",))

    def refresh_strategies(self):
        self.calls.append(("refresh_strategies",))

    def refresh_strategy_status(self):
        self.calls.append(("refresh_strategy_status",))

    def set_strategy(self, name, file_path=None):
        self.calls.append(("set_strategy", name, file_path))

    def start_strategy(self):
        self.calls.append(("start_strategy",))

    def stop_strategy(self):
        self.calls.append(("stop_strategy",))


def _app():
    global _QT_APP
    _QT_APP = QApplication.instance() or _QT_APP or QApplication([])
    return _QT_APP


def test_stage_panel_sends_relative_delta_move_request() -> None:
    _app()
    controller = FakeController()
    panel = StagePanel(controller=controller)
    panel.update_lifecycle_status({"devices_initialised": True})
    panel.x_input.setValue(1)
    panel.y_input.setValue(2)
    panel.z_input.setValue(3)

    panel._move_delta()

    assert controller.calls == [("move_stage_relative", 1.0, 2.0, 3.0)]


def test_stage_panel_sends_zero_request() -> None:
    _app()
    controller = FakeController()
    panel = StagePanel(controller=controller)
    panel.update_lifecycle_status({"devices_initialised": True})

    panel.zero_button.click()

    assert controller.calls == [("zero_stage",)]


def test_stage_panel_sends_return_to_origin_request() -> None:
    _app()
    controller = FakeController()
    panel = StagePanel(controller=controller)
    panel.update_lifecycle_status({"devices_initialised": True})

    panel.origin_button.click()

    assert controller.calls == [("return_stage_to_origin",)]


def test_stage_panel_sends_camera_fov_move_request() -> None:
    _app()
    controller = FakeController()
    panel = StagePanel(controller=controller)
    panel.update_lifecycle_status({"devices_initialised": True})

    panel._move_camera_fov("RIGHT")

    assert controller.calls == [("move_stage_fov", "RIGHT", 1.0)]


def test_camera_panel_sends_set_exposure_request() -> None:
    _app()
    controller = FakeController()
    panel = CameraPanel(controller=controller)
    panel.update_lifecycle_status({"devices_initialised": True})
    panel.exposure_input.setValue(150)

    panel._set_exposure()

    assert controller.calls == [("set_camera_exposure", 150.0)]


def test_camera_panel_sends_acquire_frame_request() -> None:
    _app()
    controller = FakeController()
    panel = CameraPanel(controller=controller)
    panel.update_lifecycle_status({"devices_initialised": True})

    panel.acquire_frame_button.click()

    assert controller.calls == [("acquire_frame", None)]


def test_manual_acquisition_panel_sends_settings_payload() -> None:
    _app()
    controller = FakeController()
    settings_panel = FrameAcquisitionSettingsPanel()
    settings_panel.config_values["normalise"] = True
    panel = ManualAcquisitionPanel(controller=controller, settings_provider=settings_panel.payload)
    panel.update_lifecycle_status({"devices_initialised": True})

    panel.acquire_button.click()

    assert controller.calls == [
        (
            "acquire_frame",
            {
                "settings": {
                    "save": False,
                    "normalise": True,
                    "illuminate_dmd": False,
                    "clear_dmd_after": False,
                    "restore_leds_after": True,
                    "disable_leds_after": False,
                },
                "use_current_main_controls": True,
            },
        )
    ]


def test_acquisition_config_does_not_offer_unknown_filter_wheel() -> None:
    _app()
    panel = FrameAcquisitionSettingsPanel()

    filter_field = next(field for field in panel._config_fields() if field.key == "filter_wheel")

    assert FilterWheelType.UNKNOWN.name not in filter_field.choices


def test_z_stack_panel_sends_request() -> None:
    _app()
    controller = FakeController()
    settings_panel = FrameAcquisitionSettingsPanel()
    settings_panel.config_values["start_z"] = -1.0
    settings_panel.config_values["end_z"] = 1.0
    settings_panel.config_values["step_z"] = 0.5
    panel = ZStackSettingsPanel(controller=controller, settings_provider=settings_panel.z_stack_payload)
    panel.update_lifecycle_status({"devices_initialised": True})

    panel.acquire_button.click()

    assert controller.calls == [
        (
            "acquire_z_stack",
            {
                "settings": {
                    "save": False,
                    "normalise": False,
                    "illuminate_dmd": False,
                    "clear_dmd_after": False,
                    "restore_leds_after": True,
                    "disable_leds_after": False,
                },
                "use_current_main_controls": True,
                "start_z": -1.0,
                "end_z": 1.0,
                "step_z": 0.5,
            },
        )
    ]


def test_z_stack_defaults_are_relative_ten_micrometres_around_origin() -> None:
    _app()
    settings_panel = FrameAcquisitionSettingsPanel()

    payload = settings_panel.z_stack_payload()

    assert payload["start_z"] == -10.0
    assert payload["end_z"] == 10.0
    assert payload["step_z"] == 1.0


def test_saved_image_loader_panel_sends_load_requests() -> None:
    _app()
    controller = FakeController()
    panel = SavedImageLoaderPanel(controller=controller)
    controller.calls.clear()
    panel.update_file_list([{"label": "test.tiff", "path": "/tmp/test.tiff"}])

    panel.load_button.click()
    panel.force_socket_transport = True
    panel.load_button.click()

    assert controller.calls == [
        ("load_acquisition_frame", "/tmp/test.tiff", None),
        ("load_acquisition_frame", "/tmp/test.tiff", "socket_tiff"),
    ]


def test_saved_image_loader_panel_refreshes_configured_directory() -> None:
    _app()
    controller = FakeController()
    panel = SavedImageLoaderPanel(controller=controller)
    controller.calls.clear()

    panel.refresh_button.click()

    assert controller.calls == [("refresh_acquisition_files",)]


def test_saved_image_loader_panel_creates_and_displays_experiment() -> None:
    _app()
    controller = FakeController()
    panel = SavedImageLoaderPanel(controller=controller)
    controller.calls.clear()
    panel.experiment_name_input.setText("2026-07-29 calibration")

    panel.create_experiment_button.click()

    assert controller.calls == [
        ("create_acquisition_experiment", "2026-07-29 calibration")
    ]

    panel.update_acquisition_directory({
        "directory": "/tmp/images/2026-07-29 calibration",
        "experiment_root": "/tmp/images",
        "experiment_name": "2026-07-29 calibration",
    })

    assert panel.directory == "/tmp/images/2026-07-29 calibration"
    assert panel.experiment_label.text() == "active experiment: 2026-07-29 calibration"


def test_saved_image_loader_panel_selects_experiment_from_dropdown() -> None:
    _app()
    controller = FakeController()
    panel = SavedImageLoaderPanel(controller=controller)
    controller.calls.clear()

    panel.update_experiment_list({
        "experiments": [
            {"name": "AD_experiment_1", "directory": "/tmp/images/AD_experiment_1"},
            {"name": "hardware_z_stack_test", "directory": "/tmp/images/hardware_z_stack_test"},
        ],
        "active_experiment": "AD_experiment_1",
        "experiment_root": "/tmp/images",
    })
    panel.experiment_combo.setCurrentIndex(1)

    assert controller.calls == [
        ("select_acquisition_experiment", "hardware_z_stack_test")
    ]


def test_filter_wheel_panel_sends_set_request() -> None:
    _app()
    controller = FakeController()
    panel = FilterWheelPanel(controller=controller)
    panel.update_lifecycle_status({"devices_initialised": True})
    panel.update_status(
        {
            "available_filters": [
                {"name": "NO_FILTER", "value": 4},
                {"name": "BLOCKING", "value": 5},
            ],
            "current_filter": {"name": "NO_FILTER", "value": 4},
            "is_initialised": True,
            "is_alive": True,
        }
    )
    panel.filter_combo.setCurrentIndex(panel.filter_combo.findData("BLOCKING"))

    panel.set_button.click()

    assert controller.calls == [("set_filter_wheel", "BLOCKING")]


def test_led_panel_sends_set_request() -> None:
    _app()
    controller = FakeController()
    panel = LedManagerPanel(controller=controller)
    panel.update_lifecycle_status({"devices_initialised": True})
    panel.update_leds(["LED_450_NM"])
    panel.brightness_inputs[LEDType.LED_450_NM].setValue(12)
    panel.led_buttons[LEDType.LED_450_NM].setChecked(True)

    assert controller.calls == [("set_led", "LED_450_NM", 12.0, None)]


def test_led_panel_sends_optional_high_brightness_duration_in_milliseconds() -> None:
    _app()
    controller = FakeController()
    panel = LedManagerPanel(controller=controller)
    panel.update_lifecycle_status({"devices_initialised": True})
    panel.update_leds(["LED_450_NM"])
    panel.custom_duration_checkbox.setChecked(True)
    panel.high_brightness_duration_input.setValue(4.5)

    panel.brightness_inputs[LEDType.LED_450_NM].setValue(50)
    panel.led_buttons[LEDType.LED_450_NM].setChecked(True)

    assert controller.calls == [("set_led", "LED_450_NM", 50.0, 4500.0)]


def test_led_panel_omits_custom_duration_at_safe_continuous_brightness() -> None:
    _app()
    controller = FakeController()
    panel = LedManagerPanel(controller=controller)
    panel.update_lifecycle_status({"devices_initialised": True})
    panel.update_leds(["LED_450_NM"])
    panel.custom_duration_checkbox.setChecked(True)
    panel.high_brightness_duration_input.setValue(4.5)

    panel.brightness_inputs[LEDType.LED_450_NM].setValue(29)
    panel.led_buttons[LEDType.LED_450_NM].setChecked(True)

    assert controller.calls == [("set_led", "LED_450_NM", 29.0, None)]


def test_led_panel_shows_timed_state_and_muted_wavelength_indicators() -> None:
    _app()
    panel = LedManagerPanel(controller=FakeController())

    panel.update_state(
        {"led": "LED_450_NM", "brightness": 50, "is_on": True, "stop_time": 1003.0}
    )

    assert "timed illumination active" in panel.state_label.text()
    assert panel.led_buttons[LEDType.LED_450_NM].styleSheet() == ""
    assert "#4f7197" in panel.wavelength_indicators[LEDType.LED_450_NM].styleSheet()
    assert panel.led_buttons[LEDType.LED_OVERHEAD].styleSheet() == ""
    assert panel.wavelength_indicators[LEDType.LED_OVERHEAD].styleSheet() == ""


def test_led_panel_allows_full_backend_brightness_range() -> None:
    _app()
    controller = FakeController()
    panel = LedManagerPanel(controller=controller)

    assert panel.brightness_inputs[LEDType.LED_450_NM].maximum() == 100.0


def test_led_panel_toggles_one_led_off() -> None:
    _app()
    controller = FakeController()
    panel = LedManagerPanel(controller=controller)
    panel.update_lifecycle_status({"devices_initialised": True})
    panel.update_leds(["LED_450_NM"])

    panel.led_buttons[LEDType.LED_450_NM].setChecked(True)
    panel.led_buttons[LEDType.LED_450_NM].setChecked(False)

    assert controller.calls == [
        ("set_led", "LED_450_NM", 29.0, None),
        ("disable_led", "LED_450_NM"),
    ]


def test_led_panel_refreshes_timed_state_after_stop_time(monkeypatch) -> None:
    _app()
    controller = FakeController()
    panel = LedManagerPanel(controller=controller)
    panel.update_lifecycle_status({"devices_initialised": True})
    panel.update_leds(["LED_450_NM"])
    callbacks = []
    monkeypatch.setattr("evomachine.gui.panels.leds.time.time", lambda: 1000.0)
    monkeypatch.setattr(
        "evomachine.gui.panels.leds.QTimer.singleShot",
        lambda _delay, callback: callbacks.append(callback),
    )

    panel.update_state({"led": "LED_450_NM", "brightness": 50, "is_on": True, "stop_time": 1003.0})
    callbacks[0]()

    assert controller.calls == [("refresh_led_state", "LED_450_NM")]


def test_dmd_panel_sends_pattern_request() -> None:
    _app()
    controller = FakeController()
    panel = DmdPanel(controller=controller)
    panel.update_lifecycle_status({"devices_initialised": True})

    panel.pattern_buttons["checkerboard"].click()

    assert controller.calls == [
        ("display_dmd_pattern", "checkerboard", panel._shape_config_payload())
    ]


def test_dmd_pattern_configuration_uses_camera_dimensions() -> None:
    _app()
    panel = DmdPanel(controller=FakeController())

    panel.update_status(
        {
            "is_initialised": True,
            "is_alive": True,
            "camera_width_height": [100, 200],
        }
    )
    fields = {field.key: field for field in panel._shape_config_fields()}

    assert fields["rectangle_row"].maximum == 99
    assert fields["rectangle_col"].maximum == 199
    assert fields["rectangle_height"].maximum == 100
    assert fields["rectangle_width"].maximum == 200
    assert fields["circle_radius"].maximum == 200
    assert (
        panel.config_values["rectangle_row"] + panel.config_values["rectangle_height"]
        <= 100
    )
    assert (
        panel.config_values["rectangle_col"] + panel.config_values["rectangle_width"]
        <= 200
    )


def test_dmd_panel_loads_previews_and_displays_custom_pattern(monkeypatch) -> None:
    _app()
    controller = FakeController()
    panel = DmdPanel(controller=controller)
    panel.update_lifecycle_status({"devices_initialised": True})
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        lambda *_args, **_kwargs: ("/tmp/custom.png", "Pattern images (*.png)"),
    )

    panel.select_custom_pattern_button.click()
    assert controller.calls == [("load_dmd_pattern", "/tmp/custom.png")]
    assert not panel.display_custom_pattern_button.isEnabled()

    panel.update_status(
        {
            "is_initialised": True,
            "is_alive": True,
            "custom_pattern": {
                "filename": "/tmp/custom.png",
                "source_shape": [100, 200],
                "coordinate_space": "camera",
            },
        }
    )
    assert panel.display_custom_pattern_button.isEnabled()
    assert "warped using loaded calibration" in panel.custom_pattern_space_label.text()

    panel.display_custom_pattern_button.click()
    assert controller.calls[-1] == ("display_loaded_dmd_pattern",)

    panel.status_label.setText("Loading custom DMD pattern invalid.png.")
    panel._show_error("RuntimeError: img_to_dmd_array: no calibration data provided.")
    assert not panel.display_custom_pattern_button.isEnabled()
    assert panel.custom_pattern_space_label.text() == "coordinates: load failed"


def test_dmd_panel_sends_load_calibration_request() -> None:
    _app()
    controller = FakeController()
    panel = DmdPanel(controller=controller)
    panel.update_lifecycle_status({"devices_initialised": True})
    panel.update_status(
        {
            "is_initialised": True,
            "is_alive": True,
            "is_calibrated": True,
            "calibration_file": "/tmp/current_calibration.pkl",
            "calibration_files": [
                {
                    "label": "current_calibration.pkl",
                    "path": "/tmp/current_calibration.pkl",
                    "is_current": True,
                }
            ],
        }
    )

    panel.load_calibration_button.click()

    assert controller.calls == [("load_dmd_calibration", "/tmp/current_calibration.pkl")]


def test_dmd_panel_sends_calibration_plot_request() -> None:
    _app()
    controller = FakeController()
    panel = DmdPanel(controller=controller)
    panel.update_lifecycle_status({"devices_initialised": True})

    panel.show_calibration_plot_button.click()

    assert controller.calls == [("request_dmd_calibration_points",)]


def test_autofocus_panel_sends_lock_request() -> None:
    _app()
    controller = FakeController()
    panel = AutofocusPanel(controller=controller)
    panel.update_lifecycle_status({"devices_initialised": True})

    panel.lock_button.click()

    assert controller.calls == [("lock_autofocus",)]


def test_autofocus_panel_sends_config_request() -> None:
    _app()
    controller = FakeController()
    panel = AutofocusPanel(controller=controller)
    panel.update_lifecycle_status({"devices_initialised": True})
    panel.config_values["led_intensity"] = 80
    panel.config_values["objective_na"] = 1.4

    panel._apply_config()

    assert controller.calls == [
        (
            "configure_autofocus",
            {
                "averaging": 5,
                "led_intensity": 80,
                "lock_range": 0.1,
                "loop_gain": 10,
                "update_rate": 10,
                "min_error": 100,
                "min_snr": 2.0,
            },
        ),
    ]


def test_autofocus_panel_sends_calibration_request() -> None:
    _app()
    controller = FakeController()
    panel = AutofocusPanel(controller=controller)
    panel.update_lifecycle_status({"devices_initialised": True})
    panel.lock_after_calibration_checkbox.setChecked(True)

    panel.run_calibration_button.click()

    assert controller.calls == [
        (
            "initialise_autofocus",
            True,
            {
                "averaging": 5,
                "led_intensity": 70,
                "lock_range": 0.1,
                "loop_gain": 10,
                "update_rate": 10,
                "min_error": 100,
                "objective_na": 0.9,
                "min_snr": 2.0,
            },
        ),
    ]


def test_autofocus_panel_maps_crisp_status_and_displays_calibration_diagnostics() -> None:
    _app()
    panel = AutofocusPanel(controller=FakeController())
    payload = {
        "is_initialised": True,
        "is_alive": True,
        "is_locked": False,
        "status": {"name": "READY", "value": "R"},
        "config": {"min_snr": 2.0, "min_error": 100},
        "calibration_result": {
            "success": True,
            "measurements": {"snr": 10.0, "error": 200.0},
            "failure_reason": None,
            "cancelled": False,
        },
    }

    panel.update_status(payload)
    assert panel.state_label.text() == "status: Calibrated"
    assert "SNR 10" in panel.diagnostics_label.text()
    assert "error 200" in panel.diagnostics_label.text()

    payload["status"] = {"name": "ERROR", "value": "E"}
    panel.update_status(payload)
    assert panel.state_label.text() == "status: Out of range"

    payload["status"] = {"name": "OUT_OF_FOCUS", "value": "K"}
    panel.update_status(payload)
    assert panel.state_label.text() == "status: Out of focus"

    payload["status"] = {"name": "IN_FOCUS", "value": "F"}
    panel.update_status(payload)
    assert panel.state_label.text() == "status: Locked"

    payload["status"] = {"name": "READY", "value": "R"}
    payload["calibration_result"] = {
        "success": False,
        "measurements": {"snr": 1.0, "error": 200.0},
        "failure_reason": "SNR below minimum.",
        "cancelled": False,
    }
    panel.update_status(payload)
    assert panel.state_label.text() == "status: Calibration failed"
    assert "SNR below minimum" in panel.diagnostics_label.text()


def test_software_focus_panel_sends_run_request() -> None:
    _app()
    controller = FakeController()
    panel = SoftwareFocusPanel(controller=controller)
    panel.update_lifecycle_status({"devices_initialised": True})

    panel.run_button.click()

    assert controller.calls == [("run_software_focus",)]


def test_application_log_panel_is_incremental_and_bounded() -> None:
    _app()
    controller = FakeController()
    panel = ApplicationLogPanel(controller=controller, history_limit=2)
    assert controller.calls[-1] == ("refresh_logs", 0)

    controller.logs_received.emit(
        {
            "records": [
                {"sequence": 1, "timestamp": "10:00:00", "level": "INFO", "logger": "a", "message": "first"},
                {"sequence": 2, "timestamp": "10:00:01", "level": "WARNING", "logger": "b", "message": "second"},
                {"sequence": 3, "timestamp": "10:00:02", "level": "ERROR", "logger": "c", "message": "third"},
            ],
            "latest_sequence": 3,
        }
    )
    controller.logs_received.emit(
        {
            "records": [
                {"sequence": 3, "timestamp": "10:00:02", "level": "ERROR", "logger": "c", "message": "third"},
            ],
            "latest_sequence": 3,
        }
    )

    rendered = panel.log_view.toPlainText()
    assert "first" not in rendered
    assert rendered.count("second") == 1
    assert rendered.count("third") == 1
    panel.refresh()
    assert controller.calls[-1] == ("refresh_logs", 3)


def test_long_operation_status_updates_controls_and_cancellation() -> None:
    _app()
    controller = FakeController()
    dmd = DmdPanel(controller=controller)
    autofocus = AutofocusPanel(controller=controller)
    for panel in (dmd, autofocus):
        panel.update_lifecycle_status({"devices_initialised": True})

    controller.operation_status_received.emit(
        {
            "kind": "dmd_calibration",
            "state": "running",
            "progress": 0.4,
            "message": "Scanning.",
        }
    )
    assert dmd.calibration_operation_label.text() == "operation: running — Scanning."
    assert dmd.cancel_calibration_button.isEnabled()
    assert not dmd.pattern_buttons["full"].isEnabled()
    dmd.cancel_calibration_button.click()
    assert controller.calls[-1] == ("cancel_dmd_calibration",)

    controller.operation_status_received.emit(
        {
            "kind": "autofocus_calibration",
            "state": "failed",
            "progress": 0.5,
            "message": "Failed.",
            "error": "RuntimeError: CRISP failed",
        }
    )
    assert autofocus.calibration_operation_label.text() == "operation: failed — Failed."
    assert "CRISP failed" in autofocus.status_label.text()

def test_peripheral_panels_lock_unsafe_controls_while_strategy_runs() -> None:
    _app()
    controller = FakeController()
    stage = StagePanel(controller=controller)
    camera = CameraPanel(controller=controller)
    filter_wheel = FilterWheelPanel(controller=controller)
    leds = LedManagerPanel(controller=controller)
    dmd = DmdPanel(controller=controller)
    autofocus = AutofocusPanel(controller=controller)
    software_focus = SoftwareFocusPanel(controller=controller)
    panels = (stage, camera, filter_wheel, leds, dmd, autofocus, software_focus)

    for panel in panels:
        panel.update_lifecycle_status({"devices_initialised": True})
    filter_wheel.update_status(
        {
            "available_filters": [{"name": "FILTER_465nm"}],
            "current_filter": {"name": "FILTER_465nm"},
        }
    )
    leds.update_leds(["LED_450_NM"])
    dmd.update_status(
        {
            "is_initialised": True,
            "is_alive": True,
            "custom_pattern": {
                "filename": "/tmp/custom.png",
                "source_shape": [10, 20],
                "coordinate_space": "dmd",
            },
        }
    )

    unsafe_controls = (
        stage.configure_button,
        stage.move_button,
        camera.configure_button,
        camera.acquire_frame_button,
        filter_wheel.configure_button,
        filter_wheel.set_button,
        leds.configure_button,
        leds.led_buttons[LEDType.LED_450_NM],
        dmd.configure_pattern_button,
        dmd.pattern_buttons["full"],
        dmd.select_custom_pattern_button,
        dmd.display_custom_pattern_button,
        autofocus.configure_button,
        autofocus.lock_button,
        software_focus.configure_button,
        software_focus.run_button,
    )
    safe_controls = (
        stage.refresh_button,
        stage.stop_button,
        camera.refresh_button,
        filter_wheel.refresh_button,
        leds.refresh_button,
        dmd.utility_buttons["refresh"],
        autofocus.refresh_button,
        software_focus.refresh_button,
    )

    assert all(control.isEnabled() for control in (*unsafe_controls, *safe_controls))

    controller.strategy_status_received.emit({"running": True})

    assert not any(control.isEnabled() for control in unsafe_controls)
    assert all(control.isEnabled() for control in safe_controls)

    controller.strategy_status_received.emit({"running": False})

    assert all(control.isEnabled() for control in (*unsafe_controls, *safe_controls))


def test_strategy_panel_sends_lifecycle_requests() -> None:
    _app()
    controller = FakeController()
    panel = StrategySetupPanel(controller=controller)
    controller.calls.clear()
    panel.update_strategies(
        [
            {
                "name": "NoStrategy",
                "file_path": None,
                "commands": [],
                "built_in": True,
            },
            {
                "name": "SimpleImagingStrategy",
                "file_path": "/tmp/strategy_simple_imaging.py",
                "commands": ["IMAGE", "MOVE", "WAIT"],
                "built_in": False,
            },
        ]
    )
    panel.strategy_combo.setCurrentIndex(1)

    panel.set_button.click()
    panel.update_status(
        {
            "name": "SimpleImagingStrategy",
            "is_initialised": True,
            "running": False,
            "fovs_initialised": True,
        }
    )
    panel.start_button.click()
    panel.update_status(
        {
            "name": "SimpleImagingStrategy",
            "is_initialised": True,
            "running": True,
            "fovs_initialised": True,
        }
    )
    panel.stop_button.click()

    assert controller.calls == [
        ("set_strategy", "SimpleImagingStrategy", "/tmp/strategy_simple_imaging.py"),
        ("start_strategy",),
        ("stop_strategy",),
    ]


def test_fov_setup_panel_sends_initialise_request() -> None:
    _app()
    controller = FakeController()
    panel = FovSetupPanel(controller=controller)
    panel.fov_id_input.setValue(2)
    panel.x_input.setValue(10)
    panel.y_input.setValue(20)
    panel.z_input.setValue(30)

    panel.add_button.click()
    panel.initialise_button.click()

    assert controller.calls == [
        (
            "initialise_fovs",
            [{"fov_id": 2, "x": 10.0, "y": 20.0, "z": 30.0, "channel_id": 0}],
            False,
        )
    ]
