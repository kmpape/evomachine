from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from evomachine.gui.runtime import (
    HardwareGuiRuntimeSettings,
    _find_serial_port_by_hwid_fragment,
    _hardware_software_focus_config,
    build_virtual_automaton,
)
from evomachine.types import FocusSelectionType, LEDType


def test_virtual_gui_camera_uses_full_size_uint16_frames() -> None:
    """
    Check the virtual GUI camera matches the full-size TIFF source contract.

    Returns
    -------
    None
    """
    automaton = build_virtual_automaton()

    assert automaton.acq_mngr.camera.image.shape == (3200, 3200)
    assert automaton.acq_mngr.camera.image.pxl_dtype == np.dtype("uint16")


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


def test_hardware_software_focus_config_builds_one_focus_frame() -> None:
    """
    Check the hardware GUI software-focus config builds without hardware.

    Returns
    -------
    None
    """
    settings = HardwareGuiRuntimeSettings(camera_exposure_ms=150.0)

    config = _hardware_software_focus_config(settings)

    assert len(config.focus_frames) == 1
    assert config.focus_frames[0].leds == {LEDType.LED_515_NM: 29}
    assert config.focus_frames[0].exposure == 150.0
    selection_type, selection_config = config.selection
    assert selection_type == FocusSelectionType.MAXIMUM
    assert (selection_config.rel_range, selection_config.step_size) == (100, 5)


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
