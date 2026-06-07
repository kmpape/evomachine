from __future__ import annotations

import pytest

from evomachine.gui.protocol import GuiCommandType, GuiEvent, GuiRequest, GuiResponse


def test_request_response_and_event_round_trip() -> None:
    request = GuiRequest(command=GuiCommandType.STAGE_GET_COORDINATES, payload={"query_hardware": False})
    response = GuiResponse(request_id=request.request_id, ok=True, payload={"x": 1})
    event = GuiEvent(event_type="status", payload={"message": "ready"})

    assert GuiRequest.from_dict(request.to_dict()) == request
    assert GuiResponse.from_dict(response.to_dict()) == response
    assert GuiEvent.from_dict(event.to_dict()) == event


def test_request_rejects_unknown_command_and_bad_version() -> None:
    with pytest.raises(ValueError):
        GuiRequest.from_dict(
            {
                "kind": "request",
                "version": 1,
                "request_id": "1",
                "command": "bad.command",
                "payload": {},
            }
        )

    with pytest.raises(ValueError):
        GuiRequest(command=GuiCommandType.PING, version=99)

