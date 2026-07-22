from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from evomachine.acquisition import FrameAcquisitionSettings
from evomachine.config import DMD_WIDTH_HEIGHT
from evomachine.coordinates import Coordinate
from evomachine.frame import FrameMetaDataFactory
from evomachine.gui.image_payloads import array_to_preview_payload, stack_to_preview_payload
from evomachine.peripherals.dmd import DmdCalibrationConfigFactory
from evomachine.strategy import NoStrategy, create_strategy_from_definition, list_strategy_definitions
from evomachine.types import UNKNOWN_FOV_ID
from evomachine.types import FilterWheelType, LEDType
from evomachine.gui.protocol import GuiCommandType


GuiRequestHandler = Callable[[Any, dict[str, Any]], dict[str, Any]]
DMD_PATTERNS = frozenset({"clear", "full", "checkerboard", "calibration_image", "half", "crosshair"})
DMD_PREVIEW_SHAPE = (220, 360)
FRAME_PREVIEW_SHAPE = (512, 512)
STACK_PREVIEW_SHAPE = (256, 256)
MAX_Z_STACK_PLANES = 10000


def gui_coordinate_to_payload(coordinate: Coordinate) -> dict[str, Any]:
    """Serialize an evomachine Coordinate for JSON transport."""
    return {
        "x": coordinate.x,
        "y": coordinate.y,
        "z": coordinate.z,
        "channel_id": coordinate.get_channel_id(),
    }


def gui_coordinate_from_payload(payload: dict[str, Any]) -> Coordinate:
    """Build a Coordinate from JSON payload fields."""
    return Coordinate(
        x=payload.get("x"),
        y=payload.get("y"),
        z=payload.get("z"),
        channel_id=int(payload.get("channel_id", 0)),
    )


def gui_led_type_from_payload(value: Any) -> LEDType:
    """Convert a JSON LED identifier to LEDType."""
    if isinstance(value, LEDType):
        return value
    if isinstance(value, str):
        try:
            return LEDType[value]
        except KeyError:
            return LEDType(value)
    return LEDType(value)


def gui_filter_wheel_type_from_payload(value: Any) -> FilterWheelType:
    """Convert a JSON filter-wheel identifier to FilterWheelType."""
    if isinstance(value, FilterWheelType):
        return value
    if isinstance(value, str):
        try:
            return FilterWheelType[value]
        except KeyError:
            return FilterWheelType(value)
    return FilterWheelType(value)


def gui_autofocus_config_from_payload(payload: dict[str, Any]) -> Any | None:
    """Build an optional Tiger autofocus config from GUI payload fields."""
    config_payload = payload.get("config")
    if config_payload is None:
        return None
    if not isinstance(config_payload, dict):
        raise TypeError("Autofocus config payload must be a dict.")

    from evomachine.bindings.asitiger.autofocus import TigerAutofocusConfig, TigerAutofocusConfigFactory

    preset = config_payload.get("preset")
    if preset == "oil":
        base_config = TigerAutofocusConfigFactory.default_oil_config()
    else:
        base_config = TigerAutofocusConfigFactory.default_config()
    values = dict(base_config.__dict__)
    for key in (
            "averaging",
            "led_intensity",
            "lock_range",
            "loop_gain",
            "update_rate",
            "objective_na",
            "min_snr",
            "min_error",
    ):
        if key not in config_payload:
            continue
        value = config_payload[key]
        if isinstance(value, str):
            value = TigerAutofocusConfig.get_attr_from_str(key, value)
        values[key] = value
    return TigerAutofocusConfig(**values)


def gui_frame_acquisition_settings_from_payload(payload: dict[str, Any]) -> FrameAcquisitionSettings:
    """Build frame acquisition settings from GUI payload booleans."""
    settings_payload = payload.get("settings", {})
    if settings_payload is None:
        settings_payload = {}
    if not isinstance(settings_payload, dict):
        raise TypeError("Acquisition settings payload must be a dict.")
    allowed_fields = {
        "save",
        "normalise",
        "illuminate_dmd",
        "clear_dmd_after",
        "restore_leds_after",
        "disable_leds_after",
    }
    unknown_fields = sorted(set(settings_payload) - allowed_fields)
    if unknown_fields:
        raise ValueError(f"Unknown acquisition setting(s): {', '.join(unknown_fields)}.")
    return FrameAcquisitionSettings(**{
        field_name: bool(settings_payload[field_name])
        for field_name in allowed_fields
        if field_name in settings_payload
    })


def gui_z_coordinates_from_payload(payload: dict[str, Any]) -> list[Coordinate]:
    """Build an inclusive Z-coordinate list from start/end/step GUI fields."""
    start_z = float(payload["start_z"])
    end_z = float(payload["end_z"])
    step_z = abs(float(payload["step_z"]))
    if step_z <= 0:
        raise ValueError("Z-stack step must be greater than zero.")
    direction = 1.0 if end_z >= start_z else -1.0
    signed_step = step_z * direction
    z_values: list[float] = []
    current_z = start_z
    for _index in range(MAX_Z_STACK_PLANES):
        if direction > 0 and current_z > end_z:
            break
        if direction < 0 and current_z < end_z:
            break
        z_values.append(current_z)
        current_z += signed_step
    else:
        raise ValueError(f"Z-stack cannot exceed {MAX_Z_STACK_PLANES} planes.")
    if not z_values or not np.isclose(z_values[-1], end_z):
        z_values.append(end_z)
    return [Coordinate(None, None, z) for z in z_values]


def gui_fovs_from_payload(payload: dict[str, Any]) -> dict[int, Coordinate]:
    """Build an FoV coordinate map from GUI table rows."""
    fov_rows = payload.get("fovs")
    if not isinstance(fov_rows, list) or not fov_rows:
        raise ValueError("FoV initialisation requires at least one FoV row.")
    fovs: dict[int, Coordinate] = {}
    for row in fov_rows:
        if not isinstance(row, dict):
            raise TypeError("Every FoV row must be a dict.")
        fov_id = int(row["fov_id"])
        if fov_id in fovs:
            raise ValueError(f"Duplicate FoV ID {fov_id}.")
        fovs[fov_id] = Coordinate(
            x=None if row.get("x") is None else float(row.get("x")),
            y=None if row.get("y") is None else float(row.get("y")),
            z=None if row.get("z") is None else float(row.get("z")),
            channel_id=int(row.get("channel_id", 0)),
        )
    return fovs


def gui_strategy_notes(command_names: list[str]) -> list[str]:
    """Create user-facing notes for strategies with extra setup needs."""
    notes = []
    if "PROJECT_ROI" in command_names:
        notes.append("Requires ROI/DeLTA outputs before DMD projection can run.")
        notes.append("Uses the DMD ROI projection mode; this still needs GUI validation.")
    return notes


def gui_frame_payload(frame: Any, *, kind: str, z_positions: list[float] | None = None) -> dict[str, Any]:
    """Serialize an acquired Frame preview for GUI display/status panels."""
    image = frame.array[-1]
    saved_paths = getattr(frame, "saved_paths", None) or []
    payload: dict[str, Any] = {
        "kind": kind,
        "preview": array_to_preview_payload(image, max_shape=FRAME_PREVIEW_SHAPE),
        "image_shape": list(image.shape),
        "stack_shape": list(frame.array.shape),
        "dtype": str(image.dtype),
        "fov_id": frame.fov_id,
        "planes": int(frame.array.shape[0]),
        "saved_paths": [None if path is None else str(path) for path in saved_paths],
    }
    if payload["planes"] > 1:
        payload["stack_preview"] = stack_to_preview_payload(frame.array, max_shape=STACK_PREVIEW_SHAPE)
    if z_positions is not None:
        payload["z_positions"] = list(z_positions)
    return payload


def gui_led_state_to_payload(state: Any) -> dict[str, Any]:
    """Serialize a LedState-like object."""
    led_type = getattr(state, "led_type")
    return {
        "led": led_type.name if isinstance(led_type, LEDType) else str(led_type),
        "brightness": getattr(state, "brightness"),
        "is_on": getattr(state, "is_on"),
        "stop_time": getattr(state, "stop_time"),
    }


def gui_require_devices_initialised(facade: Any, control_name: str) -> None:
    """Reject hardware commands before the automaton has initialised devices."""
    if not facade.automaton.devices_is_initialised():
        raise RuntimeError(f"Initialise devices before using {control_name} controls.")


def gui_ping(facade: Any, payload: dict[str, Any]) -> dict[str, Any]:
    return {"status": "ok", **facade.gui_status_payload(), **facade.gui_controller_status_payload()}


def gui_initialise_devices(facade: Any, payload: dict[str, Any]) -> dict[str, Any]:
    facade.gui_initialise_controllers()
    facade.automaton.initialise_devices()
    return {**facade.gui_status_payload(), **facade.gui_controller_status_payload()}


def gui_stop(facade: Any, payload: dict[str, Any]) -> dict[str, Any]:
    facade.automaton.stop()
    return facade.gui_status_payload()


def gui_shutdown(facade: Any, payload: dict[str, Any]) -> dict[str, Any]:
    devices_were_initialised = bool(facade.automaton.devices_is_initialised())
    try:
        if devices_were_initialised:
            facade.automaton.shutdown()
        else:
            facade.gui_mark_automaton_shutdown()
    finally:
        facade.gui_shutdown_controllers()
    return {**facade.gui_status_payload(), **facade.gui_controller_status_payload()}


def gui_controller_status(facade: Any, payload: dict[str, Any]) -> dict[str, Any]:
    return facade.gui_controller_status_payload()


def gui_fov_initialise(facade: Any, payload: dict[str, Any]) -> dict[str, Any]:
    fovs = gui_fovs_from_payload(payload)
    facade.gui_initialise_controllers()
    facade.automaton.initialise(
        fovs=fovs,
        use_autofocus=bool(payload.get("use_autofocus", False)),
    )
    return {
        **facade.gui_status_payload(),
        "fovs": facade.gui_fov_status_payload(),
        "strategy": facade.gui_strategy_status_payload(),
        **facade.gui_controller_status_payload(),
    }


def gui_stage_status(facade: Any, payload: dict[str, Any]) -> dict[str, Any]:
    gui_require_devices_initialised(facade, "stage")
    return {"stage": facade.gui_stage_status_payload()}


def gui_stage_get_coordinates(facade: Any, payload: dict[str, Any]) -> dict[str, Any]:
    gui_require_devices_initialised(facade, "stage")
    return facade.gui_stage_coordinates_payload(query_hardware=bool(payload.get("query_hardware", True)))


def gui_stage_move_absolute(facade: Any, payload: dict[str, Any]) -> dict[str, Any]:
    gui_require_devices_initialised(facade, "stage")
    coordinate = gui_coordinate_from_payload(payload)
    facade.gui_stage().move(target=coordinate, block=bool(payload.get("block", True)))
    return facade.gui_stage_coordinates_payload(query_hardware=False)


def gui_stage_move_relative(facade: Any, payload: dict[str, Any]) -> dict[str, Any]:
    gui_require_devices_initialised(facade, "stage")
    current = facade.gui_stage().get_coordinates(query_hardware=True)
    target = Coordinate(
        x=None if payload.get("dx") is None else current.x + payload.get("dx"),
        y=None if payload.get("dy") is None else current.y + payload.get("dy"),
        z=None if payload.get("dz") is None else current.z + payload.get("dz"),
    )
    facade.gui_stage().move(target=target, block=bool(payload.get("block", True)))
    return facade.gui_stage_coordinates_payload(query_hardware=False)


def gui_stage_stop(facade: Any, payload: dict[str, Any]) -> dict[str, Any]:
    gui_require_devices_initialised(facade, "stage")
    facade.gui_stage().stop()
    return {"stage": facade.gui_stage_status_payload()}


def gui_camera_status(facade: Any, payload: dict[str, Any]) -> dict[str, Any]:
    gui_require_devices_initialised(facade, "camera")
    return {"camera": facade.gui_camera_status_payload()}


def gui_camera_set_exposure(facade: Any, payload: dict[str, Any]) -> dict[str, Any]:
    gui_require_devices_initialised(facade, "camera")
    facade.gui_camera().set_exposure(exposure_time=payload["exposure"])
    return {"camera": facade.gui_camera_status_payload()}


def gui_acquisition_take_frame(facade: Any, payload: dict[str, Any]) -> dict[str, Any]:
    gui_require_devices_initialised(facade, "acquisition")
    acq_mngr = facade.gui_acquisition_manager()
    metadata = FrameMetaDataFactory.default(
        exposure=payload.get("exposure"),
        fov_id=int(payload.get("fov_id", UNKNOWN_FOV_ID)),
    )
    settings = gui_frame_acquisition_settings_from_payload(payload)
    frame = acq_mngr.take_frame(frame_metadata=metadata, settings=settings)
    return {
        "frame": gui_frame_payload(frame=frame, kind="frame"),
    }


def gui_acquisition_take_z_stack(facade: Any, payload: dict[str, Any]) -> dict[str, Any]:
    gui_require_devices_initialised(facade, "acquisition")
    acq_mngr = facade.gui_acquisition_manager()
    z_coordinates = gui_z_coordinates_from_payload(payload)
    metadata = FrameMetaDataFactory.default(
        exposure=payload.get("exposure"),
        fov_id=int(payload.get("fov_id", UNKNOWN_FOV_ID)),
    )
    settings = gui_frame_acquisition_settings_from_payload(payload)
    frame = acq_mngr.take_z_stack(
        frame_metadata=metadata,
        z_coordinates=z_coordinates,
        settings=settings,
    )
    return {
        "frame": gui_frame_payload(
            frame=frame,
            kind="z_stack",
            z_positions=[coordinate.z for coordinate in z_coordinates],
        ),
    }


def gui_filter_wheel_status(facade: Any, payload: dict[str, Any]) -> dict[str, Any]:
    gui_require_devices_initialised(facade, "filter wheel")
    return {"filter_wheel": facade.gui_filter_wheel_status_payload()}


def gui_filter_wheel_set(facade: Any, payload: dict[str, Any]) -> dict[str, Any]:
    gui_require_devices_initialised(facade, "filter wheel")
    filter_type = gui_filter_wheel_type_from_payload(payload["filter_wheel"])
    facade.gui_filter_wheel().set_filter_wheel(filter_type=filter_type)
    return {"filter_wheel": facade.gui_filter_wheel_status_payload()}


def gui_led_list(facade: Any, payload: dict[str, Any]) -> dict[str, Any]:
    gui_require_devices_initialised(facade, "LED")
    led_manager = facade.gui_led_manager()
    return {"leds": [led_type.name for led_type in led_manager.get_available_leds()]}


def gui_led_set(facade: Any, payload: dict[str, Any]) -> dict[str, Any]:
    gui_require_devices_initialised(facade, "LED")
    led_type = gui_led_type_from_payload(payload["led"])
    led_manager = facade.gui_led_manager()
    led_manager.set_led(
        led_type=led_type,
        brightness=payload.get("brightness", 100.0),
        duration=payload.get("duration"),
    )
    return {"state": gui_led_state_to_payload(led_manager.get_led_state(led_type))}


def gui_led_disable(facade: Any, payload: dict[str, Any]) -> dict[str, Any]:
    gui_require_devices_initialised(facade, "LED")
    led_type = gui_led_type_from_payload(payload["led"])
    led_manager = facade.gui_led_manager()
    led_manager.disable_led(led_type=led_type)
    return {"state": gui_led_state_to_payload(led_manager.get_led_state(led_type))}


def gui_led_disable_all(facade: Any, payload: dict[str, Any]) -> dict[str, Any]:
    led_manager = facade.gui_led_manager()
    led_manager.disable_led()
    return {"leds": [led_type.name for led_type in led_manager.get_available_leds()]}


def gui_led_get_state(facade: Any, payload: dict[str, Any]) -> dict[str, Any]:
    gui_require_devices_initialised(facade, "LED")
    led_type = gui_led_type_from_payload(payload["led"])
    led_manager = facade.gui_led_manager()
    return {"state": gui_led_state_to_payload(led_manager.get_led_state(led_type))}


def gui_dmd_status(facade: Any, payload: dict[str, Any]) -> dict[str, Any]:
    gui_require_devices_initialised(facade, "DMD")
    return {"dmd": facade.gui_dmd_status_payload()}


def gui_dmd_display_pattern(facade: Any, payload: dict[str, Any]) -> dict[str, Any]:
    gui_require_devices_initialised(facade, "DMD")
    pattern = payload["pattern"]
    if pattern not in DMD_PATTERNS:
        raise ValueError(f"Unsupported DMD pattern {pattern!r}.")
    dmd = facade.gui_dmd()
    pattern_array = gui_dmd_pattern_array(dmd=dmd, pattern=pattern)
    dmd.display_image(pattern_array, _is_full_display=pattern == "full")
    facade._last_dmd_pattern = pattern
    facade._last_dmd_preview = gui_dmd_preview_payload(dmd=dmd, pattern_array=pattern_array)
    return {"dmd": facade.gui_dmd_status_payload()}


def gui_dmd_calibrate(facade: Any, payload: dict[str, Any]) -> dict[str, Any]:
    gui_require_devices_initialised(facade, "DMD")
    config_updates = payload.get("config", {})
    if not isinstance(config_updates, dict):
        raise TypeError("DMD calibration config payload must be a dict.")
    if "channel" in config_updates:
        channel = config_updates["channel"]
        if isinstance(channel, list):
            config_updates = {**config_updates, "channel": [gui_led_type_from_payload(item) for item in channel]}
        else:
            config_updates = {**config_updates, "channel": gui_led_type_from_payload(channel)}
    config = DmdCalibrationConfigFactory.default().update_from_mapping(config_updates)
    calibration_data, _homography, _homography_inv, calibration_file = facade.automaton.dmd_calibrate(
        cfg=config,
        filename=payload.get("filename"),
    )
    return {
        "dmd": {
            **facade.gui_dmd_status_payload(),
            "calibration_file": None if calibration_file is None else str(calibration_file),
            "calibration_points": len(calibration_data) if calibration_data else 0,
        },
    }


def gui_autofocus_status(facade: Any, payload: dict[str, Any]) -> dict[str, Any]:
    gui_require_devices_initialised(facade, "autofocus")
    return {"autofocus": facade.gui_autofocus_status_payload()}


def gui_autofocus_configure(facade: Any, payload: dict[str, Any]) -> dict[str, Any]:
    gui_require_devices_initialised(facade, "autofocus")
    configured = facade.gui_autofocus().configure(config=gui_autofocus_config_from_payload(payload))
    return {"autofocus": {**facade.gui_autofocus_status_payload(), "configured": bool(configured)}}


def gui_autofocus_initialise(facade: Any, payload: dict[str, Any]) -> dict[str, Any]:
    gui_require_devices_initialised(facade, "autofocus")
    initialised = facade.gui_autofocus().initialise_autofocus(
        config=gui_autofocus_config_from_payload(payload),
        lock_after_initialise=bool(payload.get("lock_after_initialise", False)),
    )
    return {"autofocus": {**facade.gui_autofocus_status_payload(), "autofocus_initialised": bool(initialised)}}


def gui_autofocus_lock(facade: Any, payload: dict[str, Any]) -> dict[str, Any]:
    gui_require_devices_initialised(facade, "autofocus")
    facade.gui_autofocus().lock()
    return {"autofocus": facade.gui_autofocus_status_payload()}


def gui_autofocus_unlock(facade: Any, payload: dict[str, Any]) -> dict[str, Any]:
    gui_require_devices_initialised(facade, "autofocus")
    facade.gui_autofocus().unlock()
    return {"autofocus": facade.gui_autofocus_status_payload()}


def gui_autofocus_disable(facade: Any, payload: dict[str, Any]) -> dict[str, Any]:
    gui_require_devices_initialised(facade, "autofocus")
    facade.gui_autofocus().disable()
    return {"autofocus": facade.gui_autofocus_status_payload()}


def gui_software_focus_status(facade: Any, payload: dict[str, Any]) -> dict[str, Any]:
    gui_require_devices_initialised(facade, "software focus")
    return {"software_focus": facade.gui_software_focus_status_payload()}


def gui_software_focus_run(facade: Any, payload: dict[str, Any]) -> dict[str, Any]:
    gui_require_devices_initialised(facade, "software focus")
    fov_id = payload.get("fov_id")
    result = facade.gui_software_focus().run(fov_id=None if fov_id is None else int(fov_id))
    result_payload = facade.gui_software_focus_result_payload(result)
    return {
        "software_focus": {
            **facade.gui_software_focus_status_payload(),
            "last_result": result_payload,
        },
    }


def gui_strategy_status(facade: Any, payload: dict[str, Any]) -> dict[str, Any]:
    return {"strategy": facade.gui_strategy_status_payload()}


def gui_strategy_list(facade: Any, payload: dict[str, Any]) -> dict[str, Any]:
    cfg = facade.gui_strategy_config()
    definitions = [
        {
            "name": "NoStrategy",
            "file_path": None,
            "commands": [],
            "notes": [],
            "built_in": True,
        }
    ]
    for definition in list_strategy_definitions():
        commands: list[str] = []
        error = None
        try:
            strategy = create_strategy_from_definition(
                name=definition.name,
                file_path=definition.file_path,
                cfg=cfg,
            )
            commands = sorted(command_type.name for command_type in strategy.register_automaton_commands())
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        entry = {
            "name": definition.name,
            "file_path": str(definition.file_path),
            "commands": commands,
            "notes": gui_strategy_notes(commands),
            "built_in": False,
        }
        if error is not None:
            entry["error"] = error
        definitions.append(entry)
    return {
        "strategies": definitions,
        "strategy": facade.gui_strategy_status_payload(),
    }


def gui_strategy_set(facade: Any, payload: dict[str, Any]) -> dict[str, Any]:
    name = str(payload["name"])
    cfg = facade.gui_strategy_config()
    if name == "NoStrategy":
        strategy = NoStrategy(cfg=cfg)
    else:
        file_path = payload.get("file_path")
        if file_path is None:
            matches = [
                definition
                for definition in list_strategy_definitions()
                if definition.name == name
            ]
            if len(matches) != 1:
                raise RuntimeError(f"Expected one strategy named {name}, found {len(matches)}.")
            file_path = matches[0].file_path
        strategy = create_strategy_from_definition(name=name, file_path=file_path, cfg=cfg)
    facade.automaton.set_strategy(strategy=strategy)
    return {
        **facade.gui_status_payload(),
        "strategy": facade.gui_strategy_status_payload(),
    }


def gui_strategy_start(facade: Any, payload: dict[str, Any]) -> dict[str, Any]:
    facade.automaton.start_strategy()
    return {
        **facade.gui_status_payload(),
        "strategy": facade.gui_strategy_status_payload(),
    }


def gui_strategy_stop(facade: Any, payload: dict[str, Any]) -> dict[str, Any]:
    facade.automaton.stop_strategy()
    return {
        **facade.gui_status_payload(),
        "strategy": facade.gui_strategy_status_payload(),
    }


GUI_REQUEST_HANDLERS: dict[GuiCommandType, GuiRequestHandler] = {
    GuiCommandType.PING: gui_ping,
    GuiCommandType.INITIALISE_DEVICES: gui_initialise_devices,
    GuiCommandType.STOP: gui_stop,
    GuiCommandType.SHUTDOWN: gui_shutdown,
    GuiCommandType.CONTROLLER_STATUS: gui_controller_status,
    GuiCommandType.FOV_INITIALISE: gui_fov_initialise,
    GuiCommandType.STAGE_STATUS: gui_stage_status,
    GuiCommandType.STAGE_GET_COORDINATES: gui_stage_get_coordinates,
    GuiCommandType.STAGE_MOVE_ABSOLUTE: gui_stage_move_absolute,
    GuiCommandType.STAGE_MOVE_RELATIVE: gui_stage_move_relative,
    GuiCommandType.STAGE_STOP: gui_stage_stop,
    GuiCommandType.CAMERA_STATUS: gui_camera_status,
    GuiCommandType.CAMERA_SET_EXPOSURE: gui_camera_set_exposure,
    GuiCommandType.ACQUISITION_TAKE_FRAME: gui_acquisition_take_frame,
    GuiCommandType.ACQUISITION_TAKE_Z_STACK: gui_acquisition_take_z_stack,
    GuiCommandType.FILTER_WHEEL_STATUS: gui_filter_wheel_status,
    GuiCommandType.FILTER_WHEEL_SET: gui_filter_wheel_set,
    GuiCommandType.LED_LIST: gui_led_list,
    GuiCommandType.LED_SET: gui_led_set,
    GuiCommandType.LED_DISABLE: gui_led_disable,
    GuiCommandType.LED_DISABLE_ALL: gui_led_disable_all,
    GuiCommandType.LED_GET_STATE: gui_led_get_state,
    GuiCommandType.DMD_STATUS: gui_dmd_status,
    GuiCommandType.DMD_DISPLAY_PATTERN: gui_dmd_display_pattern,
    GuiCommandType.DMD_CALIBRATE: gui_dmd_calibrate,
    GuiCommandType.AUTOFOCUS_STATUS: gui_autofocus_status,
    GuiCommandType.AUTOFOCUS_CONFIGURE: gui_autofocus_configure,
    GuiCommandType.AUTOFOCUS_INITIALISE: gui_autofocus_initialise,
    GuiCommandType.AUTOFOCUS_LOCK: gui_autofocus_lock,
    GuiCommandType.AUTOFOCUS_UNLOCK: gui_autofocus_unlock,
    GuiCommandType.AUTOFOCUS_DISABLE: gui_autofocus_disable,
    GuiCommandType.SOFTWARE_FOCUS_STATUS: gui_software_focus_status,
    GuiCommandType.SOFTWARE_FOCUS_RUN: gui_software_focus_run,
    GuiCommandType.STRATEGY_STATUS: gui_strategy_status,
    GuiCommandType.STRATEGY_LIST: gui_strategy_list,
    GuiCommandType.STRATEGY_SET: gui_strategy_set,
    GuiCommandType.STRATEGY_START: gui_strategy_start,
    GuiCommandType.STRATEGY_STOP: gui_strategy_stop,
}


def gui_dmd_pattern_array(dmd: Any, pattern: str) -> np.ndarray:
    """Build the DMD array sent for a built-in GUI pattern."""
    if pattern == "clear":
        return dmd.get_zero_array()
    if pattern == "full":
        return dmd.get_one_array()
    if pattern == "checkerboard":
        return dmd.get_checkerboard()
    if pattern == "calibration_image":
        return dmd.get_calibration_image()
    if pattern == "half":
        image = dmd.get_zero_array()
        image[image.shape[0] // 4:image.shape[0] * 3 // 4, :] = 255
        return image
    if pattern == "crosshair":
        image = dmd.get_zero_array()
        line_width = int(getattr(dmd, "default_line_width", getattr(dmd, "DEFAULT_LINE_WIDTH", 5)))
        row_start, row_end = _centered_slice(center=image.shape[0] // 2, width=line_width, length=image.shape[0])
        col_start, col_end = _centered_slice(center=image.shape[1] // 2, width=line_width, length=image.shape[1])
        image[row_start:row_end, :] = 255
        image[:, col_start:col_end] = 255
        return image
    raise ValueError(f"Unsupported DMD pattern {pattern!r}.")


def gui_dmd_preview_payload(dmd: Any, pattern_array: np.ndarray) -> dict[str, Any]:
    """Return a display-oriented preview for a DMD array."""
    dmd_shape = tuple(getattr(dmd, "width_height_DMD", DMD_WIDTH_HEIGHT))
    display_array = pattern_array.T if pattern_array.shape == dmd_shape else pattern_array
    return array_to_preview_payload(display_array, max_shape=DMD_PREVIEW_SHAPE)


def _centered_slice(center: int, width: int, length: int) -> tuple[int, int]:
    half_width = max(1, width) // 2
    return max(0, center - half_width), min(length, center + half_width + 1)
