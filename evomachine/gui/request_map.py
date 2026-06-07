from __future__ import annotations

from collections.abc import Callable
from typing import Any

from evomachine.coordinates import Coordinate
from evomachine.types import LEDType
from evomachine.gui.protocol import GuiCommandType


GuiRequestHandler = Callable[[Any, dict[str, Any]], dict[str, Any]]


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


def gui_led_state_to_payload(state: Any) -> dict[str, Any]:
    """Serialize a LedState-like object."""
    led_type = getattr(state, "led_type")
    return {
        "led": led_type.name if isinstance(led_type, LEDType) else str(led_type),
        "brightness": getattr(state, "brightness"),
        "is_on": getattr(state, "is_on"),
        "stop_time": getattr(state, "stop_time"),
    }


def gui_ping(facade: Any, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "ok",
        "strategy_active": facade.gui_strategy_active(),
        "shutdown": bool(facade.automaton.has_shutdown()),
    }


def gui_initialise_devices(facade: Any, payload: dict[str, Any]) -> dict[str, Any]:
    facade.automaton.initialise_devices()
    return facade.gui_status_payload()


def gui_stop(facade: Any, payload: dict[str, Any]) -> dict[str, Any]:
    facade.automaton.stop()
    return facade.gui_status_payload()


def gui_shutdown(facade: Any, payload: dict[str, Any]) -> dict[str, Any]:
    facade.automaton.shutdown()
    return {"shutdown": True}


def gui_stage_status(facade: Any, payload: dict[str, Any]) -> dict[str, Any]:
    return {"stage": facade.gui_stage_status_payload()}


def gui_stage_get_coordinates(facade: Any, payload: dict[str, Any]) -> dict[str, Any]:
    return facade.gui_stage_coordinates_payload(query_hardware=bool(payload.get("query_hardware", True)))


def gui_stage_move_absolute(facade: Any, payload: dict[str, Any]) -> dict[str, Any]:
    coordinate = gui_coordinate_from_payload(payload)
    facade.automaton.stage.move(target=coordinate, block=bool(payload.get("block", True)))
    return facade.gui_stage_coordinates_payload(query_hardware=False)


def gui_stage_move_relative(facade: Any, payload: dict[str, Any]) -> dict[str, Any]:
    current = facade.automaton.stage.get_coordinates(query_hardware=True)
    target = Coordinate(
        x=None if payload.get("dx") is None else current.x + payload.get("dx"),
        y=None if payload.get("dy") is None else current.y + payload.get("dy"),
        z=None if payload.get("dz") is None else current.z + payload.get("dz"),
    )
    facade.automaton.stage.move(target=target, block=bool(payload.get("block", True)))
    return facade.gui_stage_coordinates_payload(query_hardware=False)


def gui_stage_stop(facade: Any, payload: dict[str, Any]) -> dict[str, Any]:
    facade.automaton.stage.stop()
    return {"stage": facade.gui_stage_status_payload()}


def gui_led_list(facade: Any, payload: dict[str, Any]) -> dict[str, Any]:
    return {"leds": [led_type.name for led_type in facade.automaton.led_manager.get_available_leds()]}


def gui_led_set(facade: Any, payload: dict[str, Any]) -> dict[str, Any]:
    led_type = gui_led_type_from_payload(payload["led"])
    facade.automaton.led_manager.set_led(
        led_type=led_type,
        brightness=payload.get("brightness", 100.0),
        duration=payload.get("duration"),
    )
    return {"state": gui_led_state_to_payload(facade.automaton.led_manager.get_led_state(led_type))}


def gui_led_disable(facade: Any, payload: dict[str, Any]) -> dict[str, Any]:
    led_type = gui_led_type_from_payload(payload["led"])
    facade.automaton.led_manager.disable_led(led_type=led_type)
    return {"state": gui_led_state_to_payload(facade.automaton.led_manager.get_led_state(led_type))}


def gui_led_disable_all(facade: Any, payload: dict[str, Any]) -> dict[str, Any]:
    facade.automaton.led_manager.disable_led()
    return {"leds": [led_type.name for led_type in facade.automaton.led_manager.get_available_leds()]}


def gui_led_get_state(facade: Any, payload: dict[str, Any]) -> dict[str, Any]:
    led_type = gui_led_type_from_payload(payload["led"])
    return {"state": gui_led_state_to_payload(facade.automaton.led_manager.get_led_state(led_type))}


GUI_REQUEST_HANDLERS: dict[GuiCommandType, GuiRequestHandler] = {
    GuiCommandType.PING: gui_ping,
    GuiCommandType.INITIALISE_DEVICES: gui_initialise_devices,
    GuiCommandType.STOP: gui_stop,
    GuiCommandType.SHUTDOWN: gui_shutdown,
    GuiCommandType.STAGE_STATUS: gui_stage_status,
    GuiCommandType.STAGE_GET_COORDINATES: gui_stage_get_coordinates,
    GuiCommandType.STAGE_MOVE_ABSOLUTE: gui_stage_move_absolute,
    GuiCommandType.STAGE_MOVE_RELATIVE: gui_stage_move_relative,
    GuiCommandType.STAGE_STOP: gui_stage_stop,
    GuiCommandType.LED_LIST: gui_led_list,
    GuiCommandType.LED_SET: gui_led_set,
    GuiCommandType.LED_DISABLE: gui_led_disable,
    GuiCommandType.LED_DISABLE_ALL: gui_led_disable_all,
    GuiCommandType.LED_GET_STATE: gui_led_get_state,
}
