from __future__ import annotations

import argparse
import importlib
import multiprocessing as mp
import os
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from typing import Any

from evomachine.gui.facade import AutomatonGuiFacade
from evomachine.gui.protocol import GUI_HOST_ENV, GUI_PORT_ENV, GuiCommandType
from evomachine.gui.socket_transport import GuiRpcServer, GuiSocketClient


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
PLUGIN_NAME = "evomachine.gui"


def _build_common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no-napari", action="store_true", help="Start and stop the automaton RPC server only.")
    parser.add_argument("napari_args", nargs=argparse.REMAINDER)
    return parser


def _load_runtime_factory(spec: str) -> Callable[[], Any]:
    module_name, sep, function_name = spec.partition(":")
    if not sep:
        raise ValueError("Runtime spec must use 'module:function' format.")
    module = importlib.import_module(module_name)
    factory = getattr(module, function_name)
    if not callable(factory):
        raise TypeError(f"Runtime factory {spec!r} is not callable.")
    return factory


def _serve_automaton(automaton, host: str, port: int, ready_queue) -> None:
    facade = AutomatonGuiFacade(automaton=automaton)
    server = GuiRpcServer(handler=facade, host=host, port=port)
    bound_host, bound_port = server.start()
    automaton.gui_set_request_processor(server.process_pending)
    ready_queue.put((bound_host, bound_port))
    try:
        automaton.run()
    finally:
        server.stop()


def _demo_automaton_process(host: str, port: int, ready_queue) -> None:
    from evomachine.gui.runtime import build_virtual_automaton

    _serve_automaton(build_virtual_automaton(), host=host, port=port, ready_queue=ready_queue)


def _hardware_automaton_process(runtime_spec: str, host: str, port: int, ready_queue) -> None:
    factory = _load_runtime_factory(runtime_spec)
    automaton = factory()
    _serve_automaton(automaton, host=host, port=port, ready_queue=ready_queue)


def _run_napari(host: str, port: int, napari_args: Sequence[str]) -> int:
    env = dict(os.environ)
    env[GUI_HOST_ENV] = host
    env[GUI_PORT_ENV] = str(port)
    command = ["napari", *napari_args, "--with", PLUGIN_NAME]
    completed = subprocess.run(command, env=env, check=False)
    return int(completed.returncode)


def _shutdown_child(process: mp.Process, host: str, port: int) -> None:
    try:
        with GuiSocketClient(host=host, port=port, timeout=1.0) as client:
            client.request(GuiCommandType.SHUTDOWN)
    except Exception:
        pass
    process.join(timeout=3.0)
    if process.is_alive():
        process.terminate()
        process.join(timeout=3.0)


def _launch_with_process(process: mp.Process, ready_queue, no_napari: bool, napari_args: Sequence[str]) -> int:
    process.start()
    host, port = ready_queue.get(timeout=15.0)
    if no_napari:
        _shutdown_child(process=process, host=host, port=port)
        return 0
    try:
        return _run_napari(host=host, port=port, napari_args=napari_args)
    finally:
        _shutdown_child(process=process, host=host, port=port)


def demo_main(argv: Sequence[str] | None = None) -> int:
    parser = _build_common_parser("Launch the evomachine GUI with virtual peripherals.")
    args = parser.parse_args(argv)
    ready_queue = mp.Queue()
    process = mp.Process(
        target=_demo_automaton_process,
        name="EvoMachineDemoAutomaton",
        args=(args.host, args.port, ready_queue),
    )
    return _launch_with_process(process=process, ready_queue=ready_queue, no_napari=args.no_napari, napari_args=args.napari_args)


def hardware_main(argv: Sequence[str] | None = None) -> int:
    parser = _build_common_parser("Launch the evomachine GUI with a user-provided hardware runtime.")
    parser.add_argument("--runtime", required=True, help="Callable in 'module:function' format returning an Automaton.")
    args = parser.parse_args(argv)
    ready_queue = mp.Queue()
    process = mp.Process(
        target=_hardware_automaton_process,
        name="EvoMachineHardwareAutomaton",
        args=(args.runtime, args.host, args.port, ready_queue),
    )
    return _launch_with_process(process=process, ready_queue=ready_queue, no_napari=args.no_napari, napari_args=args.napari_args)


if __name__ == "__main__":
    raise SystemExit(demo_main(sys.argv[1:]))
