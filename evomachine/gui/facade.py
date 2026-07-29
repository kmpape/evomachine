from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evomachine.config import EVOMACHINE_DIR, get_logger
from evomachine.gui.protocol import (
    ALWAYS_ALLOWED_MUTATING_COMMANDS,
    MUTATING_COMMANDS,
    GuiCommandType,
    GuiRequest,
    GuiResponse,
)
from evomachine.gui.request_map import GUI_REQUEST_HANDLERS, gui_coordinate_to_payload


logger = get_logger(name=__name__)


class AutomatonGuiFacade:
    """Allowlisted GUI command facade around one Automaton instance."""

    def __init__(self, automaton: Any):
        self.automaton = automaton
        self._last_dmd_pattern: str | None = None
        self._last_dmd_preview: dict[str, Any] | None = None
        self._last_software_focus_result: dict[str, Any] | None = None

    def handle(self, request: GuiRequest) -> GuiResponse:
        if self.gui_strategy_active() and self.gui_is_rejected_during_strategy(request.command):
            return GuiResponse(
                request_id=request.request_id,
                ok=False,
                error=f"{request.command.value} is not allowed while a strategy is running.",
            )
        try:
            payload = self.gui_handle_payload(request.command, request.payload)
            return GuiResponse(request_id=request.request_id, ok=True, payload=payload)
        except Exception as error:
            return GuiResponse(
                request_id=request.request_id, ok=False, error=f"{type(error).__name__}: {error}"
            )

    def gui_strategy_active(self) -> bool:
        started = bool(self.automaton.strategy_has_started())
        stopped = bool(self.automaton.strategy_has_stopped())
        return started and not stopped

    @staticmethod
    def gui_is_rejected_during_strategy(command: GuiCommandType) -> bool:
        return command in MUTATING_COMMANDS and command not in ALWAYS_ALLOWED_MUTATING_COMMANDS

    def gui_handle_payload(
        self, command: GuiCommandType, payload: dict[str, Any]
    ) -> dict[str, Any]:
        try:
            handler = GUI_REQUEST_HANDLERS[command]
        except KeyError as error:
            raise ValueError(f"Unsupported GUI command {command}.") from error
        return handler(self, payload)

    def gui_stage(self) -> Any:
        focus_nav = getattr(self.automaton, "focus_nav", None)
        stage = getattr(focus_nav, "stage", None)
        if stage is None:
            logger.warning(
                "AutomatonGuiFacade: GUI stage request ignored because no stage is configured."
            )
            raise RuntimeError("GUI stage request ignored because no stage is configured.")
        return stage

    def gui_acquisition_manager(self) -> Any:
        acq_mngr = getattr(self.automaton, "acq_mngr", None)
        if acq_mngr is None:
            logger.warning(
                "AutomatonGuiFacade: GUI acquisition request ignored because no acquisition manager is configured."
            )
            raise RuntimeError(
                "GUI acquisition request ignored because no acquisition manager is configured."
            )
        return acq_mngr

    def gui_led_manager(self) -> Any:
        acq_mngr = self.gui_acquisition_manager()
        led_manager = getattr(acq_mngr, "led_manager", None)
        if led_manager is None:
            logger.warning(
                "AutomatonGuiFacade: GUI LED request ignored because no LED manager is configured."
            )
            raise RuntimeError("GUI LED request ignored because no LED manager is configured.")
        return led_manager

    def gui_camera(self) -> Any:
        acq_mngr = self.gui_acquisition_manager()
        camera = getattr(acq_mngr, "camera", None)
        if camera is None:
            logger.warning(
                "AutomatonGuiFacade: GUI camera request ignored because no camera is configured."
            )
            raise RuntimeError("GUI camera request ignored because no camera is configured.")
        return camera

    def gui_filter_wheel(self) -> Any:
        acq_mngr = self.gui_acquisition_manager()
        filter_wheel = getattr(acq_mngr, "filter_wheel", None)
        if filter_wheel is None:
            filter_wheel = getattr(self.automaton, "_filt_wheel", None)
        if filter_wheel is None:
            logger.warning(
                "AutomatonGuiFacade: GUI filter wheel request ignored because no filter wheel is configured."
            )
            raise RuntimeError(
                "GUI filter wheel request ignored because no filter wheel is configured."
            )
        return filter_wheel

    def gui_dmd(self) -> Any:
        acq_mngr = getattr(self.automaton, "acq_mngr", None)
        dmd = getattr(acq_mngr, "dmd", None)
        if dmd is None:
            dmd = getattr(self.automaton, "_dmd", None)
        if dmd is None:
            logger.warning(
                "AutomatonGuiFacade: GUI DMD request ignored because no DMD is configured."
            )
            raise RuntimeError("GUI DMD request ignored because no DMD is configured.")
        return dmd

    def gui_autofocus(self) -> Any:
        focus_nav = getattr(self.automaton, "focus_nav", None)
        autofocus = getattr(focus_nav, "autofocus", None)
        if autofocus is None:
            autofocus = getattr(self.automaton, "_autofocus", None)
        if autofocus is None:
            logger.warning(
                "AutomatonGuiFacade: GUI autofocus request ignored because no autofocus is configured."
            )
            raise RuntimeError("GUI autofocus request ignored because no autofocus is configured.")
        return autofocus

    def gui_software_focus(self) -> Any:
        focus_nav = getattr(self.automaton, "focus_nav", None)
        software_focus = getattr(focus_nav, "software_focus", None)
        if software_focus is None:
            software_focus = getattr(self.automaton, "_swfocus", None)
        if software_focus is None:
            logger.warning(
                "AutomatonGuiFacade: GUI software focus request ignored because no software focus is configured."
            )
            raise RuntimeError(
                "GUI software focus request ignored because no software focus is configured."
            )
        return software_focus

    def gui_status_payload(self) -> dict[str, Any]:
        return {
            "devices_initialised": bool(self.automaton.devices_is_initialised()),
            "strategy_active": self.gui_strategy_active(),
            "stopped": bool(self.automaton.stopped()),
            "shutdown": bool(self.automaton.has_shutdown()),
        }

    def gui_strategy_config(self) -> Any:
        cfg = getattr(self.automaton, "_cfg", None)
        if cfg is None:
            raise RuntimeError(
                "GUI strategy request ignored because no image processor config is configured."
            )
        return cfg

    def gui_strategy_status_payload(self) -> dict[str, Any]:
        strategy = getattr(self.automaton, "_strategy", None)
        command_names: list[str] = []
        command_error = None
        if strategy is not None:
            try:
                command_names = sorted(
                    command_type.name for command_type in strategy.register_automaton_commands()
                )
            except Exception as error:
                command_error = f"{type(error).__name__}: {error}"
        payload = {
            "name": None if strategy is None else strategy.name(),
            "is_initialised": bool(getattr(self.automaton, "_strategy_is_initialised", False)),
            "fovs_initialised": bool(getattr(self.automaton, "_fov_list_is_initialised", False)),
            "running": self.gui_strategy_active(),
            "started": bool(self.automaton.strategy_has_started()),
            "stopped": bool(self.automaton.strategy_has_stopped()),
            "next_commands": len(getattr(self.automaton, "next_commands", []) or []),
            "last_commands": len(getattr(self.automaton, "last_commands", []) or []),
            "commands": command_names,
        }
        if command_error is not None:
            payload["error"] = command_error
        return payload

    def gui_fov_status_payload(self) -> list[dict[str, Any]]:
        fovs = getattr(self.automaton, "_fovs", {}) or {}
        return [
            {
                "fov_id": fov_id,
                **gui_coordinate_to_payload(coordinate),
            }
            for fov_id, coordinate in sorted(fovs.items())
        ]

    def gui_controller_status_payload(self) -> dict[str, Any]:
        return {
            "controllers": [
                self._gui_controller_entry_payload(
                    controller=entry["controller"], owners=entry["owners"]
                )
                for entry in self._gui_controller_entries()
            ],
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }

    def gui_initialise_controllers(self) -> None:
        """Initialise every peripheral controller used by configured devices."""
        for entry in self._gui_controller_entries():
            controller = entry["controller"]
            initialise = getattr(controller, "initialise", None)
            is_initialised, _error = self._safe_bool_call(controller.is_initialised)
            if not is_initialised and callable(initialise):
                initialise()

    def gui_shutdown_controllers(self) -> None:
        """Shutdown every peripheral controller used by configured devices."""
        for entry in reversed(self._gui_controller_entries()):
            controller = entry["controller"]
            shutdown = getattr(controller, "shutdown", None)
            is_initialised, _error = self._safe_bool_call(controller.is_initialised)
            if is_initialised and callable(shutdown):
                shutdown()

    def gui_mark_automaton_shutdown(self) -> None:
        """Set automaton shutdown events when normal shutdown cannot run safely."""
        for attr_name in (
            "_stop_strategy_event",
            "_start_strategy_event",
            "_stop_event",
            "_shutdown_event",
        ):
            event = getattr(self.automaton, attr_name, None)
            set_event = getattr(event, "set", None)
            if callable(set_event):
                set_event()

    def gui_stage_status_payload(self) -> dict[str, Any]:
        stage = self.gui_stage()
        camera_fov_step_size = None
        try:
            camera_fov_step_size = float(self.gui_camera().fov_size())
        except RuntimeError:
            logger.debug("AutomatonGuiFacade: camera FoV size unavailable for stage status.")
        return {
            "is_initialised": bool(stage.is_initialised()),
            "is_alive": bool(stage.is_alive()),
            "fov_id": stage.get_fov_id(),
            "fov_step_size": stage.get_fov_step_size(),
            "camera_fov_step_size": camera_fov_step_size,
        }

    def gui_stage_coordinates_payload(self, query_hardware: bool) -> dict[str, Any]:
        coordinate = self.gui_stage().get_coordinates(query_hardware=query_hardware)
        return {
            "coordinate": gui_coordinate_to_payload(coordinate),
            "stage": self.gui_stage_status_payload(),
        }

    def gui_camera_status_payload(self) -> dict[str, Any]:
        camera = self.gui_camera()
        config = getattr(camera, "config", None)
        objective_config = getattr(config, "objective_config", None)
        readout_mode = getattr(camera, "readout_mode", None)
        return {
            "name": getattr(camera, "name", "Camera"),
            "is_initialised": bool(camera.is_initialised()),
            "is_alive": bool(camera.is_alive()),
            "exposure": camera.get_exposure(),
            "default_exposure": getattr(camera, "default_exposure_time", None),
            "readout_mode": getattr(readout_mode, "value", readout_mode),
            "image_shape": list(camera.image.shape),
            "dtype": str(camera.image.pxl_dtype),
            "sensor_pixel_size_um": getattr(config, "sensor_pixel_size_um", None),
            "objective": None
            if objective_config is None
            else {
                "na": getattr(objective_config, "na", None),
                "mag": getattr(objective_config, "mag", None),
                "descr": getattr(objective_config, "descr", None),
            },
        }

    def gui_filter_wheel_status_payload(self) -> dict[str, Any]:
        filter_wheel = self.gui_filter_wheel()
        current_filter = filter_wheel.get_filter_wheel()
        return {
            "name": getattr(filter_wheel, "name", "Filter Wheel"),
            "is_initialised": bool(filter_wheel.is_initialised()),
            "is_alive": bool(filter_wheel.is_alive()),
            "current_filter": {
                "name": getattr(current_filter, "name", str(current_filter)),
                "value": getattr(current_filter, "value", None),
            },
            "available_filters": [
                {"name": filter_type.name, "value": filter_type.value}
                for filter_type in filter_wheel.get_available_filters()
            ],
        }

    def gui_dmd_status_payload(self) -> dict[str, Any]:
        dmd = self.gui_dmd()
        calibration_filename = None
        if hasattr(dmd, "get_calibration_filename"):
            calibration_filename = dmd.get_calibration_filename()
        return {
            "name": getattr(dmd, "name", "DMD"),
            "is_initialised": bool(dmd.is_initialised()),
            "is_alive": bool(dmd.is_alive()),
            "is_full_display": bool(dmd.is_full_display()),
            "is_calibrated": bool(dmd.is_calibrated()),
            "width_height": list(getattr(dmd, "width_height_DMD", ())),
            "calibration_file": None if calibration_filename is None else str(calibration_filename),
            "calibration_files": self.gui_dmd_calibration_files_payload(
                current_file=calibration_filename
            ),
            "last_pattern": self._last_dmd_pattern,
            "preview": self._last_dmd_preview,
        }

    @staticmethod
    def gui_dmd_calibration_files_payload(
        current_file: Path | str | None = None,
    ) -> list[dict[str, Any]]:
        current_path = Path(current_file) if current_file is not None else None
        candidates: list[Path] = []
        calibration_directories = (
            EVOMACHINE_DIR / "calibration_data" / "dmd",
            EVOMACHINE_DIR / "evomachine" / "calibration_data" / "dmd",
        )
        for calibration_directory in calibration_directories:
            if calibration_directory.exists():
                candidates.extend(calibration_directory.glob("*.pkl"))
        packaged_calibrations = (
            EVOMACHINE_DIR / "evomachine" / "dmd_calibration_data.pkl",
            EVOMACHINE_DIR / "dmd_calibration_data.pkl",
        )
        candidates.extend(path for path in packaged_calibrations if path.exists())
        if current_path is not None and current_path.exists():
            candidates.append(current_path)

        seen: set[str] = set()
        files: list[dict[str, Any]] = []
        for path in sorted(candidates, key=lambda item: item.name):
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            files.append(
                {
                    "label": path.name,
                    "path": key,
                    "is_current": current_path is not None and path == current_path,
                }
            )
        return files

    def gui_autofocus_status_payload(self) -> dict[str, Any]:
        autofocus = self.gui_autofocus()
        status = autofocus.get_status()
        config = getattr(autofocus, "tiger_config", None)
        return {
            "name": getattr(autofocus, "name", "Autofocus"),
            "is_initialised": bool(autofocus.is_initialised()),
            "is_alive": bool(autofocus.is_alive()),
            "status": {
                "name": getattr(status, "name", str(status)),
                "value": getattr(status, "value", None),
            },
            "is_locked": bool(autofocus.is_locked()),
            "config": self._gui_autofocus_config_payload(config),
        }

    @staticmethod
    def _gui_autofocus_config_payload(config: Any | None) -> dict[str, Any] | None:
        if config is None:
            return None
        return {
            key: getattr(config, key)
            for key in (
                "averaging",
                "led_intensity",
                "lock_range",
                "loop_gain",
                "update_rate",
                "objective_na",
                "min_snr",
                "min_error",
            )
            if hasattr(config, key)
        }

    def gui_software_focus_status_payload(self) -> dict[str, Any]:
        software_focus = self.gui_software_focus()
        config = getattr(software_focus, "default_config", None)
        algorithm = getattr(config, "algorithm", None)
        return {
            "available": True,
            "config": None
            if config is None
            else {
                "rel_range": getattr(config, "rel_range", None),
                "step_size": getattr(config, "step_size", None),
                "algorithm": getattr(algorithm, "name", str(algorithm)),
            },
            "last_result": self._last_software_focus_result,
        }

    def gui_software_focus_result_payload(self, result: Any) -> dict[str, Any]:
        focus_status = getattr(result, "focus_status", None)
        curve_status = getattr(result, "curve_status", None)
        best_coordinate = getattr(result, "best_coordinate", None)
        previous_coordinate = getattr(result, "previous_coordinate", None)
        z_coordinates = getattr(result, "z_coordinates", None)
        payload = {
            "focus_status": {
                "name": getattr(focus_status, "name", str(focus_status)),
                "value": getattr(focus_status, "value", None),
            },
            "curve_status": {
                "name": getattr(curve_status, "name", str(curve_status)),
                "value": getattr(curve_status, "value", None),
            },
            "best_coordinate": None
            if best_coordinate is None
            else gui_coordinate_to_payload(best_coordinate),
            "previous_coordinate": None
            if previous_coordinate is None
            else gui_coordinate_to_payload(previous_coordinate),
            "z_points": None
            if z_coordinates is None
            else int(getattr(z_coordinates, "size", len(z_coordinates))),
        }
        self._last_software_focus_result = payload
        return payload

    def _gui_controller_entries(self) -> list[dict[str, Any]]:
        entries: dict[int, dict[str, Any]] = {}
        for source in self._gui_controller_sources():
            controller = getattr(source, "peripheral_ctrl", None)
            if not self._looks_like_controller(controller):
                continue
            entry = entries.setdefault(id(controller), {"controller": controller, "owners": []})
            owner_name = getattr(source, "name", type(source).__name__)
            if owner_name not in entry["owners"]:
                entry["owners"].append(owner_name)
        return list(entries.values())

    def _gui_controller_sources(self) -> list[Any]:
        roots: list[Any] = []
        iter_peripherals = getattr(self.automaton, "_iter_peripherals", None)
        if callable(iter_peripherals):
            try:
                roots.extend(iter_peripherals())
            except Exception:
                logger.exception(
                    "AutomatonGuiFacade: failed to iterate automaton peripherals for controller status."
                )
        acq_mngr = getattr(self.automaton, "acq_mngr", None)
        focus_nav = getattr(self.automaton, "focus_nav", None)
        proj_mngr = getattr(self.automaton, "proj_mngr", None)
        for container, attr_names in (
            (acq_mngr, ("camera", "stage", "led_manager", "filter_wheel", "dmd")),
            (focus_nav, ("stage", "autofocus")),
            (proj_mngr, ("camera", "dmd", "led_manager", "filter_wheel", "photodiode")),
            (
                self.automaton,
                (
                    "_camera",
                    "_stage",
                    "_led_mngr",
                    "_filt_wheel",
                    "_dmd",
                    "_autofocus",
                    "_photodiode",
                ),
            ),
        ):
            if container is None:
                continue
            for attr_name in attr_names:
                value = getattr(container, attr_name, None)
                if value is not None:
                    roots.append(value)

        sources: list[Any] = []
        seen_ids: set[int] = set()
        for root in roots:
            for source in self._gui_expand_controller_sources(root):
                if id(source) in seen_ids:
                    continue
                sources.append(source)
                seen_ids.add(id(source))
        return sources

    def _gui_expand_controller_sources(self, source: Any) -> list[Any]:
        sources = [source]
        led_sources = getattr(source, "led_sources", None)
        if isinstance(led_sources, list):
            sources.extend(led_sources)
        return sources

    @staticmethod
    def _looks_like_controller(value: Any) -> bool:
        return (
            value is not None
            and callable(getattr(value, "is_initialised", None))
            and callable(getattr(value, "is_alive", None))
        )

    def _gui_controller_entry_payload(self, controller: Any, owners: list[str]) -> dict[str, Any]:
        is_initialised, initialised_error = self._safe_bool_call(controller.is_initialised)
        is_alive, alive_error = self._safe_bool_call(controller.is_alive)
        return {
            "name": getattr(controller, "name", type(controller).__name__),
            "type": type(controller).__name__,
            "is_initialised": is_initialised,
            "is_alive": is_alive,
            "connected": bool(is_initialised and is_alive),
            "owners": owners,
            "error": initialised_error or alive_error,
        }

    @staticmethod
    def _safe_bool_call(method: Any) -> tuple[bool, str | None]:
        try:
            return bool(method()), None
        except Exception as error:
            return False, f"{type(error).__name__}: {error}"
