from __future__ import annotations

from typing import Any

from evomachine.config import get_logger
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
            return GuiResponse(request_id=request.request_id, ok=False, error=f"{type(error).__name__}: {error}")

    def gui_strategy_active(self) -> bool:
        started = bool(self.automaton.strategy_has_started())
        stopped = bool(self.automaton.strategy_has_stopped())
        return started and not stopped

    @staticmethod
    def gui_is_rejected_during_strategy(command: GuiCommandType) -> bool:
        return command in MUTATING_COMMANDS and command not in ALWAYS_ALLOWED_MUTATING_COMMANDS

    def gui_handle_payload(self, command: GuiCommandType, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            handler = GUI_REQUEST_HANDLERS[command]
        except KeyError as error:
            raise ValueError(f"Unsupported GUI command {command}.") from error
        return handler(self, payload)

    def gui_stage(self) -> Any:
        focus_nav = getattr(self.automaton, "focus_nav", None)
        stage = getattr(focus_nav, "stage", None)
        if stage is None:
            logger.warning("AutomatonGuiFacade: GUI stage request ignored because no stage is configured.")
            raise RuntimeError("GUI stage request ignored because no stage is configured.")
        return stage

    def gui_led_manager(self) -> Any:
        acq_mngr = getattr(self.automaton, "acq_mngr", None)
        led_manager = getattr(acq_mngr, "led_manager", None)
        if led_manager is None:
            logger.warning("AutomatonGuiFacade: GUI LED request ignored because no LED manager is configured.")
            raise RuntimeError("GUI LED request ignored because no LED manager is configured.")
        return led_manager

    def gui_status_payload(self) -> dict[str, Any]:
        return {
            "devices_initialised": bool(self.automaton.devices_is_initialised()),
            "strategy_active": self.gui_strategy_active(),
            "shutdown": bool(self.automaton.has_shutdown()),
        }

    def gui_stage_status_payload(self) -> dict[str, Any]:
        stage = self.gui_stage()
        return {
            "is_initialised": bool(stage.is_initialised()),
            "is_alive": bool(stage.is_alive()),
            "fov_id": stage.get_fov_id(),
            "fov_step_size": stage.get_fov_step_size(),
        }

    def gui_stage_coordinates_payload(self, query_hardware: bool) -> dict[str, Any]:
        coordinate = self.gui_stage().get_coordinates(query_hardware=query_hardware)
        return {
            "coordinate": gui_coordinate_to_payload(coordinate),
            "stage": self.gui_stage_status_payload(),
        }
