from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import uuid4


PROTOCOL_VERSION = 1
GUI_HOST_ENV = "EVOMACHINE_GUI_HOST"
GUI_PORT_ENV = "EVOMACHINE_GUI_PORT"
GuiRequestProcessor = Callable[[int], int | None]


class GuiCommandType(str, Enum):
    """Allowlisted commands accepted by the automaton GUI facade."""

    PING = "ping"
    IMAGE_TRANSPORT_PROBE = "image_transport.probe"
    INITIALISE_DEVICES = "initialise_devices"
    STOP = "stop"
    SHUTDOWN = "shutdown"
    CONTROLLER_STATUS = "controllers.status"
    FOV_INITIALISE = "fov.initialise"
    STAGE_STATUS = "stage.status"
    STAGE_GET_COORDINATES = "stage.get_coordinates"
    STAGE_MOVE_ABSOLUTE = "stage.move_absolute"
    STAGE_MOVE_RELATIVE = "stage.move_relative"
    STAGE_MOVE_FOV = "stage.move_fov"
    STAGE_STOP = "stage.stop"
    STAGE_ZERO = "stage.zero"
    CAMERA_STATUS = "camera.status"
    CAMERA_SET_EXPOSURE = "camera.set_exposure"
    ACQUISITION_LIST_FILES = "acquisition.list_files"
    ACQUISITION_LOAD_FRAME = "acquisition.load_frame"
    ACQUISITION_TAKE_FRAME = "acquisition.take_frame"
    ACQUISITION_TAKE_Z_STACK = "acquisition.take_z_stack"
    FILTER_WHEEL_STATUS = "filter_wheel.status"
    FILTER_WHEEL_SET = "filter_wheel.set"
    LED_LIST = "led.list"
    LED_SET = "led.set"
    LED_DISABLE = "led.disable"
    LED_DISABLE_ALL = "led.disable_all"
    LED_GET_STATE = "led.get_state"
    DMD_STATUS = "dmd.status"
    DMD_DISPLAY_PATTERN = "dmd.display_pattern"
    DMD_LOAD_CALIBRATION = "dmd.load_calibration"
    DMD_CALIBRATION_POINTS = "dmd.calibration_points"
    DMD_CALIBRATE = "dmd.calibrate"
    AUTOFOCUS_STATUS = "autofocus.status"
    AUTOFOCUS_CONFIGURE = "autofocus.configure"
    AUTOFOCUS_INITIALISE = "autofocus.initialise_autofocus"
    AUTOFOCUS_LOCK = "autofocus.lock"
    AUTOFOCUS_UNLOCK = "autofocus.unlock"
    AUTOFOCUS_DISABLE = "autofocus.disable"
    SOFTWARE_FOCUS_STATUS = "software_focus.status"
    SOFTWARE_FOCUS_RUN = "software_focus.run"
    STRATEGY_STATUS = "strategy.status"
    STRATEGY_LIST = "strategy.list"
    STRATEGY_SET = "strategy.set"
    STRATEGY_START = "strategy.start"
    STRATEGY_STOP = "strategy.stop"


MUTATING_COMMANDS = frozenset(
    {
        GuiCommandType.INITIALISE_DEVICES,
        GuiCommandType.FOV_INITIALISE,
        GuiCommandType.STAGE_MOVE_ABSOLUTE,
        GuiCommandType.STAGE_MOVE_RELATIVE,
        GuiCommandType.STAGE_MOVE_FOV,
        GuiCommandType.STAGE_STOP,
        GuiCommandType.STAGE_ZERO,
        GuiCommandType.CAMERA_SET_EXPOSURE,
        GuiCommandType.ACQUISITION_TAKE_FRAME,
        GuiCommandType.ACQUISITION_TAKE_Z_STACK,
        GuiCommandType.FILTER_WHEEL_SET,
        GuiCommandType.LED_SET,
        GuiCommandType.LED_DISABLE,
        GuiCommandType.LED_DISABLE_ALL,
        GuiCommandType.DMD_DISPLAY_PATTERN,
        GuiCommandType.DMD_LOAD_CALIBRATION,
        GuiCommandType.DMD_CALIBRATE,
        GuiCommandType.AUTOFOCUS_CONFIGURE,
        GuiCommandType.AUTOFOCUS_INITIALISE,
        GuiCommandType.AUTOFOCUS_LOCK,
        GuiCommandType.AUTOFOCUS_UNLOCK,
        GuiCommandType.AUTOFOCUS_DISABLE,
        GuiCommandType.SOFTWARE_FOCUS_RUN,
        GuiCommandType.STRATEGY_SET,
        GuiCommandType.STRATEGY_START,
        GuiCommandType.STRATEGY_STOP,
    }
)

ALWAYS_ALLOWED_MUTATING_COMMANDS = frozenset(
    {
        GuiCommandType.STOP,
        GuiCommandType.SHUTDOWN,
        GuiCommandType.STAGE_STOP,
        GuiCommandType.LED_DISABLE_ALL,
        GuiCommandType.STRATEGY_STOP,
    }
)

NO_RESPONSE_TIMEOUT_COMMANDS = frozenset(
    {
        GuiCommandType.ACQUISITION_TAKE_Z_STACK,
    }
)


@dataclass(kw_only=True)
class GuiRequest:
    """One typed request from a GUI client to the automaton process."""

    command: GuiCommandType
    payload: dict[str, Any] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: str(uuid4()))
    version: int = PROTOCOL_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.command, GuiCommandType):
            self.command = GuiCommandType(self.command)
        if not isinstance(self.payload, dict):
            raise TypeError(f"GuiRequest: payload must be dict, received {type(self.payload)}.")
        if not isinstance(self.request_id, str) or not self.request_id:
            raise TypeError("GuiRequest: request_id must be a non-empty str.")
        if self.version != PROTOCOL_VERSION:
            raise ValueError(f"GuiRequest: unsupported protocol version {self.version}.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "request",
            "version": self.version,
            "request_id": self.request_id,
            "command": self.command.value,
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GuiRequest":
        if not isinstance(data, dict):
            raise TypeError(f"GuiRequest.from_dict: data must be dict, received {type(data)}.")
        if data.get("kind") != "request":
            raise ValueError(f"GuiRequest.from_dict: expected kind='request', received {data.get('kind')!r}.")
        return cls(
            version=data.get("version"),
            request_id=data.get("request_id"),
            command=GuiCommandType(data.get("command")),
            payload=data.get("payload", {}),
        )


@dataclass(kw_only=True)
class GuiResponse:
    """One response from the automaton process to a GUI request."""

    request_id: str
    ok: bool
    payload: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    version: int = PROTOCOL_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id:
            raise TypeError("GuiResponse: request_id must be a non-empty str.")
        if not isinstance(self.ok, bool):
            raise TypeError(f"GuiResponse: ok must be bool, received {type(self.ok)}.")
        if not isinstance(self.payload, dict):
            raise TypeError(f"GuiResponse: payload must be dict, received {type(self.payload)}.")
        if self.error is not None and not isinstance(self.error, str):
            raise TypeError(f"GuiResponse: error must be str or None, received {type(self.error)}.")
        if self.version != PROTOCOL_VERSION:
            raise ValueError(f"GuiResponse: unsupported protocol version {self.version}.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "response",
            "version": self.version,
            "request_id": self.request_id,
            "ok": self.ok,
            "payload": self.payload,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GuiResponse":
        if not isinstance(data, dict):
            raise TypeError(f"GuiResponse.from_dict: data must be dict, received {type(data)}.")
        if data.get("kind") != "response":
            raise ValueError(f"GuiResponse.from_dict: expected kind='response', received {data.get('kind')!r}.")
        return cls(
            version=data.get("version"),
            request_id=data.get("request_id"),
            ok=data.get("ok"),
            payload=data.get("payload", {}),
            error=data.get("error"),
        )


@dataclass(kw_only=True)
class GuiEvent:
    """Asynchronous event emitted by the automaton-side GUI bridge."""

    event_type: str
    payload: dict[str, Any] = field(default_factory=dict)
    version: int = PROTOCOL_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.event_type, str) or not self.event_type:
            raise TypeError("GuiEvent: event_type must be a non-empty str.")
        if not isinstance(self.payload, dict):
            raise TypeError(f"GuiEvent: payload must be dict, received {type(self.payload)}.")
        if self.version != PROTOCOL_VERSION:
            raise ValueError(f"GuiEvent: unsupported protocol version {self.version}.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "event",
            "version": self.version,
            "event_type": self.event_type,
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GuiEvent":
        if not isinstance(data, dict):
            raise TypeError(f"GuiEvent.from_dict: data must be dict, received {type(data)}.")
        if data.get("kind") != "event":
            raise ValueError(f"GuiEvent.from_dict: expected kind='event', received {data.get('kind')!r}.")
        return cls(
            version=data.get("version"),
            event_type=data.get("event_type"),
            payload=data.get("payload", {}),
        )


def response_from_exception(request_id: str, error: Exception) -> GuiResponse:
    """Return a standard error response for a failed request."""
    return GuiResponse(request_id=request_id, ok=False, error=f"{type(error).__name__}: {error}")
