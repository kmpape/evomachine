from __future__ import annotations

from evomachine.gui.protocol import GuiCommandType
from evomachine.gui.request_map import GUI_REQUEST_HANDLERS


def test_gui_request_map_handles_every_command_type() -> None:
    assert set(GUI_REQUEST_HANDLERS) == set(GuiCommandType)
