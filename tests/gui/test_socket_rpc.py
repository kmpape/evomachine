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


def test_z_stack_request_uses_normal_response_timeout() -> None:
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
        with pytest.raises(TimeoutError):
            client.request(GuiCommandType.ACQUISITION_TAKE_Z_STACK)
    thread.join(timeout=1)
    server.stop()

    assert processed == [True]

def test_background_operation_keeps_same_socket_usable_with_normal_timeout() -> None:
    operation_started = threading.Event()
    allow_operation_to_finish = threading.Event()
    operation_finished = threading.Event()

    class BackgroundHandler:
        def handle(self, request: GuiRequest) -> GuiResponse:
            if request.command == GuiCommandType.SOFTWARE_FOCUS_RUN:

                def run() -> None:
                    operation_started.set()
                    allow_operation_to_finish.wait(timeout=2.0)
                    operation_finished.set()

                threading.Thread(target=run, daemon=True).start()
                payload = {"operation": {"state": "running"}}
            else:
                payload = {"command": request.command.value}

            return GuiResponse(
                request_id=request.request_id,
                ok=True,
                payload=payload,
            )

    server = GuiRpcServer(
        handler=BackgroundHandler(),
        port=0,
        response_timeout=0.5,
    )
    host, port = server.start()
    stop_processing = threading.Event()

    def process() -> None:
        while not stop_processing.is_set():
            server.process_pending(max_jobs=4)
            time.sleep(0.001)

    thread = threading.Thread(target=process)
    thread.start()

    try:
        with GuiSocketClient(host=host, port=port, timeout=0.5) as client:
            started = client.request(GuiCommandType.SOFTWARE_FOCUS_RUN)

            assert operation_started.wait(timeout=1)
            assert not operation_finished.is_set()

            ping = client.request(GuiCommandType.PING)

            assert started.payload["operation"]["state"] == "running"
            assert ping.payload == {"command": "ping"}
            assert not operation_finished.is_set()

            allow_operation_to_finish.set()
            assert operation_finished.wait(timeout=1)

    finally:
        allow_operation_to_finish.set()
        stop_processing.set()
        thread.join(timeout=1)
        server.stop()