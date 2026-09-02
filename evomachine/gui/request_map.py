from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from evomachine.acquisition import FrameAcquisitionSettings
from evomachine.config import CAM_WIDTH_HEIGHT, DMD_WIDTH_HEIGHT, gui_log_handler
from evomachine.coordinates import Coordinate
from evomachine.filemanager import FileManager
from evomachine.frame import FrameMetaDataFactory
from evomachine.gui.image_payloads import (
    IMAGE_TRANSPORT_SOCKET_TIFF,
    array_to_preview_payload,
    create_image_transport_probe_payload,
    stack_to_preview_payload,
)
from evomachine.peripherals.autofocus import AutofocusCalibrationConfig
from evomachine.peripherals.dmd import DMD_BUILT_IN_PATTERNS, DmdCalibrationConfigFactory, DmdShapeConfig
from evomachine.strategy import NoStrategy, create_strategy_from_definition, list_strategy_definitions
from evomachine.types import FilterWheelType, LEDType
from evomachine.types import FovDirectionType
from evomachine.gui.protocol import GuiCommandType


GuiRequestHandler = Callable[[Any, dict[str, Any]], dict[str, Any]]
DMD_PATTERNS = DMD_BUILT_IN_PATTERNS
MAX_Z_STACK_PLANES = 10000
GUI_CAMERA_FOV_DIRECTION_DELTAS = {
    FovDirectionType.UP: (0.0, -1.0),
    FovDirectionType.DOWN: (0.0, 1.0),
    FovDirectionType.LEFT: (1.0, 0.0),
    FovDirectionType.RIGHT: (-1.0, 0.0),
}


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


def gui_recent_logs(facade: Any, payload: dict[str, Any]) -> dict[str, Any]:
    """Return buffered application logs newer than the GUI's cursor."""
    del facade
    after_sequence = payload.get("after_sequence", 0)
    if not isinstance(after_sequence, int) or isinstance(after_sequence, bool):
        raise TypeError("after_sequence must be an integer.")
    records = gui_log_handler.records_after(after_sequence)
    return {
        "logs": {
            "records": list(records),
            "latest_sequence": gui_log_handler.latest_sequence,
        }
    }


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


def gui_fov_direction_from_payload(value: Any) -> FovDirectionType:
    """Convert a JSON FoV movement direction to FovDirectionType."""
    if isinstance(value, FovDirectionType):
        return value
    if isinstance(value, str):
        try:
            return FovDirectionType[value.upper()]
        except KeyError:
            return FovDirectionType(value)
    return FovDirectionType(value)


def gui_camera_fov_move_coordinate(stage: Any, camera: Any, direction: FovDirectionType, multiplier: float) -> Coordinate:
    """Build the stage target coordinate for one camera FoV movement."""
    if direction not in GUI_CAMERA_FOV_DIRECTION_DELTAS:
        raise ValueError("Camera FoV movement only supports UP, DOWN, LEFT, and RIGHT.")
    if multiplier <= 0:
        raise ValueError(f"Camera FoV movement multiplier must be positive, received {multiplier}.")
    current = stage.get_coordinates(query_hardware=True)
    if current.x is None or current.y is None:
        raise RuntimeError("Camera FoV movement requires current X/Y stage coordinates.")
    step_size = float(camera.fov_size())
    x_delta, y_delta = GUI_CAMERA_FOV_DIRECTION_DELTAS[direction]
    return Coordinate(
        x=round(current.x + x_delta * step_size * multiplier, 3),
        y=round(current.y + y_delta * step_size * multiplier, 3),
        z=None,
        channel_id=current.get_channel_id(),
    )


def gui_autofocus_config_from_payload(payload: dict[str, Any]) -> AutofocusCalibrationConfig | None:
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


def gui_acquisition_uses_current_main_controls(payload: dict[str, Any]) -> bool:
    """Return whether acquisition should leave main-control peripheral state alone."""
    return bool(payload.get("use_current_main_controls", False))


def gui_acquisition_leds_from_payload(payload: dict[str, Any]) -> dict[LEDType, float] | None:
    """Build optional explicit acquisition LED settings from a GUI payload."""
    if gui_acquisition_uses_current_main_controls(payload):
        return None
    led_payload = payload.get("leds")
    if led_payload is None:
        return None
    if not isinstance(led_payload, dict):
        raise TypeError("Acquisition LEDs payload must be a dict.")
    return {
        gui_led_type_from_payload(led): float(brightness)
        for led, brightness in led_payload.items()
    }


def gui_acquisition_filter_wheel_from_payload(payload: dict[str, Any]) -> FilterWheelType | None:
    """Build an optional explicit acquisition filter-wheel setting."""
    if gui_acquisition_uses_current_main_controls(payload):
        return None
    filter_wheel = payload.get("filter_wheel")
    if filter_wheel is None:
        return None
    return gui_filter_wheel_type_from_payload(filter_wheel)


def gui_acquisition_dmd_pattern_from_payload(facade: Any, payload: dict[str, Any]) -> np.ndarray | None:
    """Build an optional explicit acquisition DMD pattern array."""
    if gui_acquisition_uses_current_main_controls(payload):
        return None
    pattern = payload.get("dmd_pattern")
    if pattern in (None, "full"):
        return None
    if pattern not in DMD_PATTERNS:
        raise ValueError(f"Unsupported acquisition DMD pattern {pattern!r}.")
    config_payload = {"config": payload.get("dmd_config", {})}
    return gui_dmd_pattern_array(
        dmd=facade.gui_dmd(),
        pattern=pattern,
        config=gui_dmd_shape_config_from_payload(config_payload),
    )


def gui_frame_metadata_from_payload(facade: Any, payload: dict[str, Any]) -> Any:
    """Build frame metadata from optional GUI acquisition config fields."""
    return FrameMetaDataFactory.default(
        leds=gui_acquisition_leds_from_payload(payload),
        filter_wheel=gui_acquisition_filter_wheel_from_payload(payload),
        exposure=payload.get("exposure"),
        dmd_pattern=gui_acquisition_dmd_pattern_from_payload(facade, payload),
    )


def gui_dmd_shape_config_from_payload(payload: dict[str, Any]) -> DmdShapeConfig:
    """Build DMD built-in shape configuration from GUI payload fields."""
    config_payload = payload.get("config", {})
    if config_payload is None:
        config_payload = {}
    if not isinstance(config_payload, dict):
        raise TypeError("DMD pattern config payload must be a dict.")
    return DmdShapeConfig().update_from_mapping(config_payload)


def gui_z_coordinates_from_payload(payload: dict[str, Any]) -> list[Coordinate]:
    """Build an inclusive list of relative Z deltas from the GUI fields."""
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


def gui_frame_payload(
        frame: Any,
        *,
        kind: str,
        z_positions: list[float] | None = None,
        image_transport: str | None = None,
) -> dict[str, Any]:
    """Serialize an acquired Frame preview for GUI display/status panels."""
    image = frame.array[-1]
    saved_paths = getattr(frame, "saved_paths", None) or []
    payload: dict[str, Any] = {
        "kind": kind,
        "image_shape": list(image.shape),
        "stack_shape": list(frame.array.shape),
        "dtype": str(image.dtype),
        "fov_id": frame.fov_id,
        "planes": int(frame.array.shape[0]),
        "saved_paths": [None if path is None else str(path) for path in saved_paths],
    }
    if payload["planes"] > 1:
        payload["stack_preview"] = stack_to_preview_payload(frame.array, transport=image_transport)
    else:
        payload["preview"] = array_to_preview_payload(image, transport=image_transport)
    if z_positions is not None:
        payload["z_positions"] = list(z_positions)
    return payload


def gui_loaded_image_payload(
        image: np.ndarray,
        *,
        filename: Path,
        image_transport: str | None = None,
) -> dict[str, Any]:
    """Serialize an image loaded from disk for GUI display/status panels."""
    loaded = np.asarray(image)
    payload: dict[str, Any] = {
        "source": "file",
        "loaded_path": str(filename),
        "dtype": str(loaded.dtype),
        "saved_paths": [str(filename)],
    }
    if loaded.ndim == 2 or (loaded.ndim == 3 and loaded.shape[-1] in {3, 4}):
        payload.update({
            "kind": "loaded_frame",
            "image_shape": list(loaded.shape),
            "stack_shape": [1, *loaded.shape],
            "planes": 1,
            "preview": array_to_preview_payload(loaded, transport=image_transport),
        })
        return payload
    payload.update({
        "kind": "loaded_z_stack",
        "image_shape": list(loaded[-1].shape),
        "stack_shape": list(loaded.shape),
        "planes": int(loaded.shape[0]),
        "stack_preview": stack_to_preview_payload(loaded, transport=image_transport),
    })
    return payload


def gui_acquisition_file_manager(facade: Any) -> FileManager:
    """Return the file manager used by the GUI acquisition manager."""
    acq_mngr = facade.gui_acquisition_manager()
    file_manager = getattr(acq_mngr, "file_manager", None)
    if file_manager is None:
        raise RuntimeError("GUI acquisition file request ignored because no file manager is configured.")
    return file_manager


def gui_acquisition_file_path(facade: Any, filename: str) -> Path:
    """Resolve a GUI-selected acquisition filename."""
    path = Path(filename)
    if path.is_absolute():
        return path
    file_manager = gui_acquisition_file_manager(facade)
    return file_manager.config.directory / path


def gui_acquisition_file_payload(path: Path) -> dict[str, Any]:
    """Serialize one loadable acquisition file for a GUI dropdown."""
    stat = path.stat()
    return {
        "label": path.name,
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "modified_time": stat.st_mtime,
    }


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


def gui_image_transport_probe(facade: Any, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        probe = create_image_transport_probe_payload()
    except Exception as error:
        probe = {"mode": IMAGE_TRANSPORT_SOCKET_TIFF, "error": f"{type(error).__name__}: {error}"}
    return {"image_transport_probe": probe}


def gui_initialise_devices(facade: Any, payload: dict[str, Any]) -> dict[str, Any]:
    facade.gui_initialise_controllers()
    facade.automaton.initialise_devices()
    return {**facade.gui_status_payload(), **facade.gui_controller_status_payload()}


def gui_stop(facade: Any, payload: dict[str, Any]) -> dict[str, Any]:
    facade.automaton.stop()
    return facade.gui_status_payload()


def gui_shutdown(facade: Any, payload: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    try:
        facade.automaton.shutdown()
    except Exception as error:
        errors.append(f"automaton: {type(error).__name__}: {error}")
        facade.gui_mark_automaton_shutdown()
    try:
        facade.gui_shutdown_controllers()
    except Exception as error:
        errors.append(f"controllers: {type(error).__name__}: {error}")
    if errors:
        raise RuntimeError("GUI shutdown completed with errors: " + "; ".join(errors))
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


def gui_stage_move_fov(facade: Any, payload: dict[str, Any]) -> dict[str, Any]:
    gui_require_devices_initialised(facade, "stage")
    direction = gui_fov_direction_from_payload(payload["direction"])
    multiplier = float(payload.get("multiplier", 1.0))
    stage = facade.gui_stage()
    camera = facade.gui_camera()
    target = gui_camera_fov_move_coordinate(stage=stage, camera=camera, direction=direction, multiplier=multiplier)
    stage.move(
        target=target,
        block=bool(payload.get("block", True)),
    )
    return facade.gui_stage_coordinates_payload(query_hardware=False)


def gui_stage_stop(facade: Any, payload: dict[str, Any]) -> dict[str, Any]:
    gui_require_devices_initialised(facade, "stage")
    facade.gui_stage().stop()
    return {"stage": facade.gui_stage_status_payload()}


def gui_stage_zero(facade: Any, payload: dict[str, Any]) -> dict[str, Any]:
    gui_require_devices_initialised(facade, "stage")
    facade.gui_stage().zero_coordinates()
    return facade.gui_stage_coordinates_payload(query_hardware=False)


def gui_stage_return_origin(facade: Any, payload: dict[str, Any]) -> dict[str, Any]:
    gui_require_devices_initialised(facade, "stage")
    facade.gui_stage().move(target=Coordinate(0, 0, 0), block=False)
    return facade.gui_stage_coordinates_payload(query_hardware=False)


def gui_camera_status(facade: Any, payload: dict[str, Any]) -> dict[str, Any]:
    gui_require_devices_initialised(facade, "camera")
    return {"camera": facade.gui_camera_status_payload()}


def gui_camera_set_exposure(facade: Any, payload: dict[str, Any]) -> dict[str, Any]:
    gui_require_devices_initialised(facade, "camera")
    facade.gui_camera().set_exposure(exposure_time=payload["exposure"])
    return {"camera": facade.gui_camera_status_payload()}


def gui_acquisition_files_payload(file_manager: FileManager) -> dict[str, Any]:
    directory = file_manager.config.directory
    paths: dict[Path, None] = {}
    if directory.exists():
        for pattern in ("*.tiff", "*.tif"):
            for path in FileManager.list_filenames(directory=directory, filename_pattern=pattern):
                paths[path] = None
    files = sorted(paths, key=lambda item: item.stat().st_mtime, reverse=True)
    return {
        "acquisition_directory": str(directory),
        "experiment_root": str(file_manager.experiment_root),
        "acquisition_files": [gui_acquisition_file_payload(path) for path in files],
    }


def gui_acquisition_list_files(facade: Any, payload: dict[str, Any]) -> dict[str, Any]:
    return gui_acquisition_files_payload(gui_acquisition_file_manager(facade))


def gui_acquisition_experiments_payload(file_manager: FileManager) -> dict[str, Any]:
    return {
        "experiment_root": str(file_manager.experiment_root),
        "active_experiment": file_manager.config.directory.name,
        "experiments": [
            {"name": path.name, "directory": str(path)}
            for path in file_manager.list_experiments()
        ],
    }


def gui_acquisition_list_experiments(facade: Any, payload: dict[str, Any]) -> dict[str, Any]:
    file_manager = gui_acquisition_file_manager(facade)
    return gui_acquisition_experiments_payload(file_manager) | gui_acquisition_files_payload(file_manager)


def gui_acquisition_create_experiment(facade: Any, payload: dict[str, Any]) -> dict[str, Any]:
    file_manager = gui_acquisition_file_manager(facade)
    experiment_directory = file_manager.create_experiment(payload.get("name"))
    return gui_acquisition_files_payload(file_manager) | gui_acquisition_experiments_payload(file_manager) | {
        "acquisition_directory": str(experiment_directory),
        "experiment_name": experiment_directory.name,
    }


def gui_acquisition_select_experiment(facade: Any, payload: dict[str, Any]) -> dict[str, Any]:
    file_manager = gui_acquisition_file_manager(facade)
    experiment_directory = file_manager.select_experiment(payload.get("name"))
    return gui_acquisition_files_payload(file_manager) | gui_acquisition_experiments_payload(file_manager) | {
        "acquisition_directory": str(experiment_directory),
        "experiment_name": experiment_directory.name,
    }


def gui_acquisition_load_frame(facade: Any, payload: dict[str, Any]) -> dict[str, Any]:
    filename = payload.get("filename")
    if not isinstance(filename, str) or not filename:
        raise ValueError("Acquisition image filename must be a non-empty string.")
    path = gui_acquisition_file_path(facade, filename=filename)
    image = FileManager.load_image(path)
    return {
        "frame": gui_loaded_image_payload(
            image=image,
            filename=path,
            image_transport=payload.get("image_transport"),
        ),
    }


def gui_acquisition_take_frame(facade: Any, payload: dict[str, Any]) -> dict[str, Any]:
    gui_require_devices_initialised(facade, "acquisition")
    acq_mngr = facade.gui_acquisition_manager()
    metadata = gui_frame_metadata_from_payload(facade=facade, payload=payload)
    settings = gui_frame_acquisition_settings_from_payload(payload)
    frame = acq_mngr.take_frame(frame_metadata=metadata, settings=settings)
    return {
        "frame": gui_frame_payload(
            frame=frame,
            kind="frame",
            image_transport=payload.get("image_transport"),
        ),
    }


def gui_acquisition_take_z_stack(facade: Any, payload: dict[str, Any]) -> dict[str, Any]:
    gui_require_devices_initialised(facade, "acquisition")
    acq_mngr = facade.gui_acquisition_manager()
    z_deltas = gui_z_coordinates_from_payload(payload)
    current_coordinate = facade.gui_stage().get_coordinates(query_hardware=True)
    if current_coordinate.z is None:
        raise RuntimeError("Relative Z-stack acquisition requires a current stage Z coordinate.")
    z_coordinates = [
        Coordinate(None, None, current_coordinate.z + delta.z)
        for delta in z_deltas
    ]
    metadata = gui_frame_metadata_from_payload(facade=facade, payload=payload)
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
            z_positions=[coordinate.z for coordinate in z_deltas],
            image_transport=payload.get("image_transport"),
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
    warp = payload.get("warp", True)
    if not isinstance(warp, bool):
        raise TypeError(f"DMD pattern warp must be bool, received {type(warp)}.")
    dmd = facade.gui_dmd()
    pattern_array = gui_dmd_pattern_array(
        dmd=dmd,
        pattern=pattern,
        config=gui_dmd_shape_config_from_payload(payload),
        warp=warp,
    )
    dmd.display_image(pattern_array, _is_full_display=pattern == "full" and not warp)
    facade._last_dmd_pattern = pattern
    facade._last_dmd_preview = gui_dmd_preview_payload(dmd=dmd, pattern_array=pattern_array)
    return {"dmd": facade.gui_dmd_status_payload()}


def gui_dmd_load_pattern(facade: Any, payload: dict[str, Any]) -> dict[str, Any]:
    """Load and preview a custom pattern without displaying it on the DMD."""
    gui_require_devices_initialised(facade, "DMD")
    filename = payload.get("filename")
    if not isinstance(filename, str) or not filename:
        raise ValueError("DMD pattern filename must be a non-empty string.")
    dmd = facade.gui_dmd()
    facade._loaded_dmd_pattern = None
    facade._last_dmd_preview = None
    pattern_array = dmd.load_image(filename, display_image=False)
    info = dmd.get_loaded_image_info()
    if info is None:
        raise RuntimeError("DMD did not report metadata for the loaded custom pattern.")
    facade._loaded_dmd_pattern = {
        "filename": str(info.filename),
        "source_shape": list(info.source_shape),
        "coordinate_space": info.coordinate_space,
    }
    facade._last_dmd_preview = gui_dmd_preview_payload(dmd=dmd, pattern_array=pattern_array)
    return {"dmd": facade.gui_dmd_status_payload()}


def gui_dmd_display_loaded_pattern(facade: Any, payload: dict[str, Any]) -> dict[str, Any]:
    """Display the custom pattern most recently loaded through the GUI."""
    gui_require_devices_initialised(facade, "DMD")
    dmd = facade.gui_dmd()
    dmd.display_loaded_image()
    facade._last_dmd_pattern = "custom"
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
    filename = payload.get("filename")

    def run(cancel_event, report):
        projection_manager = facade.automaton.proj_mngr
        previous_stop_requested = projection_manager.stop_requested
        projection_manager.stop_requested = lambda: cancel_event.is_set() or (
            previous_stop_requested is not None and previous_stop_requested()
        )
        try:
            facade.automaton.dmd_calibrate(
                cfg=config,
                filename=filename,
                progress_callback=report,
            )
        finally:
            projection_manager.stop_requested = previous_stop_requested

    return {"operation": facade.gui_operations.start("dmd_calibration", run)}


def gui_dmd_calibration_status(facade: Any, payload: dict[str, Any]) -> dict[str, Any]:
    return {"operation": _require_operation_status(facade, "dmd_calibration")}


def gui_dmd_cancel_calibration(facade: Any, payload: dict[str, Any]) -> dict[str, Any]:
    return {"operation": facade.gui_operations.cancel("dmd_calibration")}


def gui_dmd_load_calibration(facade: Any, payload: dict[str, Any]) -> dict[str, Any]:
    gui_require_devices_initialised(facade, "DMD")
    filename = payload.get("filename")
    if not isinstance(filename, str) or not filename:
        raise ValueError("DMD calibration filename must be a non-empty string.")
    dmd = facade.gui_dmd()
    dmd.calibrate_from_path(Path(filename))
    return {"dmd": facade.gui_dmd_status_payload()}


def gui_dmd_calibration_points(facade: Any, payload: dict[str, Any]) -> dict[str, Any]:
    gui_require_devices_initialised(facade, "DMD")
    dmd = facade.gui_dmd()
    calibration_data = getattr(dmd, "_calib_data", None)
    if calibration_data is None:
        raise RuntimeError("No DMD calibration data is loaded.")
    dmd_points = getattr(calibration_data, "dmd_points", None)
    cam_points = getattr(calibration_data, "cam_points", None)
    if not dmd_points or not cam_points:
        raise RuntimeError("No DMD calibration point correspondences are loaded.")
    if len(dmd_points) != len(cam_points):
        raise RuntimeError("DMD and camera calibration point counts do not match.")

    calibration_file = getattr(calibration_data, "path", None)
    if calibration_file is None and hasattr(dmd, "get_calibration_filename"):
        calibration_file = dmd.get_calibration_filename()
    return {
        "dmd_calibration_points": {
            "dmd_points": gui_pixel_points_payload(dmd_points),
            "cam_points": gui_pixel_points_payload(cam_points),
            "dmd_shape": list(getattr(dmd, "width_height_DMD", DMD_WIDTH_HEIGHT)),
            "cam_shape": list(getattr(dmd, "width_height_CAM", CAM_WIDTH_HEIGHT)),
            "calibration_file": None if calibration_file is None else str(calibration_file),
        },
    }


def gui_autofocus_status(facade: Any, payload: dict[str, Any]) -> dict[str, Any]:
    gui_require_devices_initialised(facade, "autofocus")
    return {"autofocus": facade.gui_autofocus_status_payload()}


def gui_autofocus_configure(facade: Any, payload: dict[str, Any]) -> dict[str, Any]:
    gui_require_devices_initialised(facade, "autofocus")
    configured = facade.gui_autofocus().apply_config(config=gui_autofocus_config_from_payload(payload))
    return {"autofocus": {**facade.gui_autofocus_status_payload(), "configured": bool(configured)}}


def gui_autofocus_initialise(facade: Any, payload: dict[str, Any]) -> dict[str, Any]:
    gui_require_devices_initialised(facade, "autofocus")
    config = gui_autofocus_config_from_payload(payload)
    lock_after_calibration = bool(payload.get("lock_after_initialise", False))

    def run(cancel_event, report):
        initialised = facade.gui_autofocus().run_calibration(
            config=config,
            lock_after_calibration=lock_after_calibration,
            stop_event=cancel_event,
            progress_callback=report,
        )
        if not initialised and not cancel_event.is_set():
            raise RuntimeError("Autofocus calibration did not pass its acceptance checks.")

    return {"operation": facade.gui_operations.start("autofocus_calibration", run)}


def gui_autofocus_calibration_status(facade: Any, payload: dict[str, Any]) -> dict[str, Any]:
    return {"operation": _require_operation_status(facade, "autofocus_calibration")}


def gui_autofocus_cancel_calibration(facade: Any, payload: dict[str, Any]) -> dict[str, Any]:
    return {"operation": facade.gui_operations.cancel("autofocus_calibration")}


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


def _require_operation_status(facade: Any, kind: str) -> dict[str, Any]:
    status = facade.gui_operations.status(kind)
    if status is None:
        raise RuntimeError(f"No {kind} operation has been started.")
    return status


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
    GuiCommandType.IMAGE_TRANSPORT_PROBE: gui_image_transport_probe,
    GuiCommandType.INITIALISE_DEVICES: gui_initialise_devices,
    GuiCommandType.STOP: gui_stop,
    GuiCommandType.SHUTDOWN: gui_shutdown,
    GuiCommandType.CONTROLLER_STATUS: gui_controller_status,
    GuiCommandType.LOGS_RECENT: gui_recent_logs,
    GuiCommandType.FOV_INITIALISE: gui_fov_initialise,
    GuiCommandType.STAGE_STATUS: gui_stage_status,
    GuiCommandType.STAGE_GET_COORDINATES: gui_stage_get_coordinates,
    GuiCommandType.STAGE_MOVE_ABSOLUTE: gui_stage_move_absolute,
    GuiCommandType.STAGE_MOVE_RELATIVE: gui_stage_move_relative,
    GuiCommandType.STAGE_MOVE_FOV: gui_stage_move_fov,
    GuiCommandType.STAGE_STOP: gui_stage_stop,
    GuiCommandType.STAGE_ZERO: gui_stage_zero,
    GuiCommandType.STAGE_RETURN_ORIGIN: gui_stage_return_origin,
    GuiCommandType.CAMERA_STATUS: gui_camera_status,
    GuiCommandType.CAMERA_SET_EXPOSURE: gui_camera_set_exposure,
    GuiCommandType.ACQUISITION_CREATE_EXPERIMENT: gui_acquisition_create_experiment,
    GuiCommandType.ACQUISITION_LIST_EXPERIMENTS: gui_acquisition_list_experiments,
    GuiCommandType.ACQUISITION_SELECT_EXPERIMENT: gui_acquisition_select_experiment,
    GuiCommandType.ACQUISITION_LIST_FILES: gui_acquisition_list_files,
    GuiCommandType.ACQUISITION_LOAD_FRAME: gui_acquisition_load_frame,
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
    GuiCommandType.DMD_LOAD_PATTERN: gui_dmd_load_pattern,
    GuiCommandType.DMD_DISPLAY_LOADED_PATTERN: gui_dmd_display_loaded_pattern,
    GuiCommandType.DMD_LOAD_CALIBRATION: gui_dmd_load_calibration,
    GuiCommandType.DMD_CALIBRATION_POINTS: gui_dmd_calibration_points,
    GuiCommandType.DMD_CALIBRATE: gui_dmd_calibrate,
    GuiCommandType.DMD_CALIBRATION_STATUS: gui_dmd_calibration_status,
    GuiCommandType.DMD_CANCEL_CALIBRATION: gui_dmd_cancel_calibration,
    GuiCommandType.AUTOFOCUS_STATUS: gui_autofocus_status,
    GuiCommandType.AUTOFOCUS_CONFIGURE: gui_autofocus_configure,
    GuiCommandType.AUTOFOCUS_INITIALISE: gui_autofocus_initialise,
    GuiCommandType.AUTOFOCUS_CALIBRATION_STATUS: gui_autofocus_calibration_status,
    GuiCommandType.AUTOFOCUS_CANCEL_CALIBRATION: gui_autofocus_cancel_calibration,
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


def gui_dmd_pattern_array(
        dmd: Any,
        pattern: str,
        config: DmdShapeConfig | None = None,
        warp: bool = True,
) -> np.ndarray:
    """Build the DMD array sent for a built-in GUI pattern."""
    return dmd.get_pattern(pattern=pattern, config=config, warp=warp)


def gui_dmd_preview_payload(dmd: Any, pattern_array: np.ndarray) -> dict[str, Any]:
    """Return a display-oriented preview for a DMD array."""
    dmd_shape = tuple(getattr(dmd, "width_height_DMD", DMD_WIDTH_HEIGHT))
    display_array = pattern_array.T if pattern_array.shape == dmd_shape else pattern_array
    return array_to_preview_payload(display_array)


def gui_pixel_points_payload(points: list[tuple[int, int]]) -> list[dict[str, int]]:
    """Serialize row/column pixel point pairs for GUI plotting."""
    return [
        {"row": int(row), "col": int(col)}
        for row, col in points
    ]
