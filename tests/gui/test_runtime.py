from __future__ import annotations

from types import SimpleNamespace

import pytest

from evomachine.bindings.binding_types import BindingType
from evomachine.gui.runtime import (
    PVCAM_FRAME_TIMEOUT_MS,
    HardwareGuiRuntimeSettings,
    _find_serial_port_by_hwid_fragment,
)
from evomachine.types import LEDType


def test_hardware_gui_settings_can_be_overridden_from_env(monkeypatch) -> None:
    monkeypatch.setenv("EVOMACHINE_GUI_CAMERA_WIDTH", "123")
    monkeypatch.setenv("EVOMACHINE_GUI_CAMERA_HEIGHT", "456")
    monkeypatch.setenv("EVOMACHINE_GUI_USE_DMD", "false")
    monkeypatch.setenv("EVOMACHINE_GUI_USE_KWR103", "false")
    monkeypatch.setenv("EVOMACHINE_GUI_TIGER_PORT", "/dev/ttyTiger")
    monkeypatch.setenv("EVOMACHINE_GUI_KWR103_PORT", "/dev/ttyKWR103")
    monkeypatch.setenv("EVOMACHINE_GUI_STAGE_MIN_X_UM", "-123")
    monkeypatch.setenv("EVOMACHINE_GUI_STAGE_MAX_X_UM", "456")
    monkeypatch.setenv("EVOMACHINE_GUI_CAMERA_BINDING", "pvcam")
    monkeypatch.setenv("EVOMACHINE_GUI_USE_SYNCBOARD", "0")
    monkeypatch.setenv("EVOMACHINE_GUI_CRISP_CARD_ADDRESS", "3")
    monkeypatch.setenv("EVOMACHINE_GUI_LED_CARD_ADDRESS", "1")
    monkeypatch.setenv("EVOMACHINE_GUI_TIGER_LED_CHANNEL", "x")

    settings = HardwareGuiRuntimeSettings.from_env()

    assert settings.camera_size == (123, 456)
    assert settings.use_dmd is False
    assert settings.use_kwr103 is False
    assert settings.camera_binding == "PVCAM"
    assert settings.use_syncboard is False
    assert settings.crisp_card_address == 3
    assert settings.led_card_address == 1
    assert settings.tiger_led_channel == "X"
    assert settings.tiger_port == "/dev/ttyTiger"
    assert settings.kwr103_port == "/dev/ttyKWR103"
    assert settings.stage_bounds.low.x == -123
    assert settings.stage_bounds.high.x == 456
    assert settings.tiger_stage_limits["X"] == (-1230, 4560)


def test_hardware_gui_settings_reject_inverted_stage_limits() -> None:
    with pytest.raises(ValueError, match="stage X minimum"):
        HardwareGuiRuntimeSettings(stage_min_x_um=1, stage_max_x_um=1)

    with pytest.raises(ValueError, match="stage Z limits must be finite"):
        HardwareGuiRuntimeSettings(stage_max_z_um=float("nan"))

    with pytest.raises(ValueError, match="stage Y limits must include the startup zero"):
        HardwareGuiRuntimeSettings(stage_min_y_um=1, stage_max_y_um=2)


def test_hardware_gui_settings_default_to_microscope_one_wiring() -> None:
    settings = HardwareGuiRuntimeSettings()

    assert settings.camera_binding == "MMC"
    assert settings.camera_binding_type is BindingType.MMC
    assert settings.camera_binding_options == {"camera_device": "Camera-1", "readout_mode_property": "Port"}
    assert settings.use_syncboard is True
    assert settings.software_focus_led is LEDType.LED_515_NM
    assert (settings.crisp_card_address, settings.led_card_address, settings.filter_card_address) == (2, 7, 8)
    assert settings.tiger_led_channel == "Y"


def test_hardware_gui_settings_pvcam_without_syncboard() -> None:
    settings = HardwareGuiRuntimeSettings(camera_binding="PVCAM", use_syncboard=False)

    assert settings.camera_binding_type is BindingType.PVCAM
    assert settings.camera_binding_options == {"frame_timeout_ms": PVCAM_FRAME_TIMEOUT_MS}
    assert settings.software_focus_led is LEDType.LED_OVERHEAD_TIGER


def test_hardware_gui_settings_reject_unknown_camera_binding_and_led_channel() -> None:
    with pytest.raises(ValueError, match="camera_binding must be one of"):
        HardwareGuiRuntimeSettings(camera_binding="VIRTUAL")

    with pytest.raises(ValueError, match="tiger_led_channel must be one of"):
        HardwareGuiRuntimeSettings(tiger_led_channel="Q")

    with pytest.raises(ValueError, match="led_card_address must be a non-negative int"):
        HardwareGuiRuntimeSettings(led_card_address=-1)


def test_serial_port_lookup_uses_hwid_fragment() -> None:
    ports = [
        SimpleNamespace(device="/dev/ttyA", hwid="USB VID:PID=1111:2222"),
        SimpleNamespace(device="/dev/ttyB", hwid="USB VID:PID=10C4:EA60 SER=0001"),
    ]

    port = _find_serial_port_by_hwid_fragment("10C4:EA60", "ASI Tiger", ports)

    assert port == "/dev/ttyB"


def test_serial_port_lookup_reports_missing_port() -> None:
    ports = [SimpleNamespace(device="/dev/ttyA", hwid="USB VID:PID=1111:2222")]

    with pytest.raises(RuntimeError, match="No ASI Tiger serial port"):
        _find_serial_port_by_hwid_fragment("10C4:EA60", "ASI Tiger", ports)
