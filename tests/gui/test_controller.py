from evomachine.gui.controller import EvoMachineGuiController
from evomachine.gui.protocol import GuiCommandType, GuiResponse


class RecordingClient:
    def __init__(self):
        self.requests = []

    def request_object(self, request):
        self.requests.append(request)
        return GuiResponse(request_id=request.request_id, ok=True)

    def close(self):
        return None


def test_gui_stage_moves_are_non_blocking_so_stop_can_be_processed() -> None:
    client = RecordingClient()
    controller = EvoMachineGuiController(client=client, start_worker=False)

    controller.move_stage_absolute(1, 2, 3)
    controller.move_stage_relative(4, 5, 6)
    controller.move_stage_fov("RIGHT")
    controller.zero_stage()
    controller.return_stage_to_origin()

    assert [request.command for request in client.requests] == [
        GuiCommandType.STAGE_MOVE_ABSOLUTE,
        GuiCommandType.STAGE_MOVE_RELATIVE,
        GuiCommandType.STAGE_MOVE_FOV,
        GuiCommandType.STAGE_ZERO,
        GuiCommandType.STAGE_RETURN_ORIGIN,
    ]
    assert all(request.payload["block"] is False for request in client.requests[:3])


def test_gui_controller_sends_create_experiment_request() -> None:
    client = RecordingClient()
    controller = EvoMachineGuiController(client=client, start_worker=False)

    controller.create_acquisition_experiment("experiment-one")

    assert client.requests[-1].command is GuiCommandType.ACQUISITION_CREATE_EXPERIMENT
    assert client.requests[-1].payload == {"name": "experiment-one"}
