from __future__ import annotations

import threading
import time

import pytest

from evomachine.gui.protocol import GuiCommandType, GuiRequest, GuiResponse
from evomachine.gui.socket_transport import GuiRpcServer, GuiSocketClient


class EchoHandler:
    def handle(self, request: GuiRequest) -> GuiResponse:
        return GuiResponse(request_id=request.request_id, ok=True, payload={"command": request.command.value})


def test_socket_server_queues_requests_for_bounded_processing() -> None:
    server = GuiRpcServer(handler=EchoHandler(), port=0)
    host, port = server.start()
    responses = []

    def call_client():
        with GuiSocketClient(host=host, port=port) as client:
            responses.append(client.request(GuiCommandType.PING))

    thread = threading.Thread(target=call_client)
    thread.start()
    deadline = time.time() + 2
    processed = 0
    while time.time() < deadline and processed == 0:
        processed = server.process_pending(max_jobs=1)
        time.sleep(0.01)
    thread.join(timeout=2)
    server.stop()

    assert processed == 1
    assert responses[0].ok
    assert responses[0].payload == {"command": "ping"}


def test_client_rejects_response_request_id_mismatch() -> None:
    class BadHandler:
        def handle(self, request: GuiRequest) -> GuiResponse:
            return GuiResponse(request_id="different", ok=True)

    server = GuiRpcServer(handler=BadHandler(), port=0)
    host, port = server.start()

    with GuiSocketClient(host=host, port=port) as client:
        result = []

        def process():
            deadline = time.time() + 2
            while time.time() < deadline and not result:
                if server.process_pending(max_jobs=1):
                    result.append(True)
                time.sleep(0.01)

        thread = threading.Thread(target=process)
        thread.start()
        with pytest.raises(RuntimeError):
            client.request(GuiCommandType.PING)
        thread.join(timeout=2)
    server.stop()


def test_z_stack_request_has_no_response_timeout() -> None:
    class SlowZStackHandler:
        def handle(self, request: GuiRequest) -> GuiResponse:
            time.sleep(0.05)
            return GuiResponse(request_id=request.request_id, ok=True)

    server = GuiRpcServer(handler=SlowZStackHandler(), port=0, response_timeout=0.01)
    host, port = server.start()
    processed = []

    def process() -> None:
        deadline = time.time() + 1
        while time.time() < deadline and not processed:
            if server.process_pending(max_jobs=1):
                processed.append(True)
            time.sleep(0.001)

    thread = threading.Thread(target=process)
    thread.start()
    with GuiSocketClient(host=host, port=port, timeout=0.01) as client:
        response = client.request(GuiCommandType.ACQUISITION_TAKE_Z_STACK)
    thread.join(timeout=1)
    server.stop()

    assert processed == [True]
    assert response.ok
