from __future__ import annotations

from types import SimpleNamespace

import pytest

from evomachine.gui.runtime import HardwareGuiRuntimeSettings, _find_serial_port_by_hwid_fragment


def test_hardware_gui_settings_can_be_overridden_from_env(monkeypatch) -> None:
    monkeypatch.setenv("EVOMACHINE_GUI_CAMERA_WIDTH", "123")
    monkeypatch.setenv("EVOMACHINE_GUI_CAMERA_HEIGHT", "456")
    monkeypatch.setenv("EVOMACHINE_GUI_USE_DMD", "false")
    monkeypatch.setenv("EVOMACHINE_GUI_USE_KWR103", "false")
    monkeypatch.setenv("EVOMACHINE_GUI_TIGER_PORT", "/dev/ttyTiger")
    monkeypatch.setenv("EVOMACHINE_GUI_KWR103_PORT", "/dev/ttyKWR103")
    monkeypatch.setenv("EVOMACHINE_GUI_STAGE_MIN_X_UM", "-123")
    monkeypatch.setenv("EVOMACHINE_GUI_STAGE_MAX_X_UM", "456")

    settings = HardwareGuiRuntimeSettings.from_env()

    assert settings.camera_size == (123, 456)
    assert settings.use_dmd is False
    assert settings.use_kwr103 is False
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
