from evomachine.gui.controller import EvoMachineGuiController
from evomachine.gui.protocol import GuiCommandType, GuiResponse


class RecordingClient:
    def __init__(self):
        self.requests = []
        self.closed = False

    def request_object(self, request):
        self.requests.append(request)
        return GuiResponse(request_id=request.request_id, ok=True)

    def close(self):
        self.closed = True


def test_generation_requests_and_errors_use_the_generation_channel() -> None:
    client = RecordingClient()
    controller = EvoMachineGuiController(client=client, start_worker=False)
    received, errors = [], []
    controller.strategy_generation_received.connect(received.append)
    controller.response_error.connect(errors.append)
    controller.generate_strategy("Image then finish")
    controller.refresh_strategy_generation()
    controller.set_generated_strategy("accepted-id")
    assert [request.command for request in client.requests] == [
        GuiCommandType.STRATEGY_GENERATE, GuiCommandType.STRATEGY_GENERATION_STATUS,
        GuiCommandType.STRATEGY_SET,
    ]
    assert client.requests[2].payload == {"generation_id": "accepted-id"}
    failed_request = client.requests[0]
    controller._generation_requests.add(failed_request.request_id)
    controller._handle_request_failure(failed_request, "endpoint unavailable")
    assert received == [{"state": "failed", "error": "endpoint unavailable"}]
    assert not errors


def test_gui_stage_moves_are_non_blocking_so_stop_can_be_processed() -> None:
    client = RecordingClient()
    controller = EvoMachineGuiController(client=client, start_worker=False)

    controller.move_stage_absolute(1, 2, 3)
    controller.move_stage_relative(4, 5, 6)
    controller.move_stage_fov("RIGHT")
    controller.return_stage_to_origin()

    assert [request.command for request in client.requests] == [
        GuiCommandType.STAGE_MOVE_ABSOLUTE,
        GuiCommandType.STAGE_MOVE_RELATIVE,
        GuiCommandType.STAGE_MOVE_FOV,
        GuiCommandType.STAGE_RETURN_ORIGIN,
    ]
    assert all(request.payload["block"] is False for request in client.requests[:3])


def test_gui_controller_dispatches_completed_stage_movement_coordinates() -> None:
    controller = EvoMachineGuiController(client=RecordingClient(), start_worker=False)
    received = []
    controller.stage_coordinates_received.connect(received.append)
    result = {
        "coordinate": {"x": 1, "y": 2, "z": 3, "channel_id": 0},
        "stage": {"is_initialised": True},
    }

    controller._handle_response(
        GuiResponse(
            request_id="stage-movement",
            ok=True,
            payload={
                "operation": {
                    "kind": "stage_movement",
                    "state": "completed",
                    "result": result,
                }
            },
        )
    )

    assert received == [result]


def test_gui_controller_sends_output_directory_request() -> None:
    client = RecordingClient()
    controller = EvoMachineGuiController(client=client, start_worker=False)

    controller.set_acquisition_directory("/tmp/images")

    assert client.requests[-1].command is GuiCommandType.ACQUISITION_SET_DIRECTORY
    assert client.requests[-1].payload == {"directory": "/tmp/images"}

    controller.refresh_acquisition_files("/tmp/previous-images")
    assert client.requests[-1].command is GuiCommandType.ACQUISITION_LIST_FILES
    assert client.requests[-1].payload == {"directory": "/tmp/previous-images"}


def test_gui_controller_requests_and_dispatches_incremental_logs() -> None:
    client = RecordingClient()
    controller = EvoMachineGuiController(client=client, start_worker=False)
    received = []
    controller.logs_received.connect(received.append)

    controller.refresh_logs(after_sequence=12)
    controller._handle_response(
        GuiResponse(
            request_id="logs",
            ok=True,
            payload={"logs": {"records": [], "latest_sequence": 12}},
        )
    )

    assert client.requests[-1].command is GuiCommandType.LOGS_RECENT
    assert client.requests[-1].payload == {"after_sequence": 12}
    assert received == [{"records": [], "latest_sequence": 12}]


def test_gui_controller_requests_shutdown_before_closing_connection() -> None:
    client = RecordingClient()
    controller = EvoMachineGuiController(client=client, start_worker=False)

    controller.close()
    controller.close()

    assert [request.command for request in client.requests] == [GuiCommandType.SHUTDOWN]
    assert client.closed
