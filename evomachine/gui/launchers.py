from __future__ import annotations

import argparse
import importlib
import multiprocessing as mp
import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from evomachine.gui.facade import AutomatonGuiFacade
from evomachine.gui.image_payloads import IMAGE_TRANSPORT_CHOICES, IMAGE_TRANSPORT_ENV, normalise_image_transport
from evomachine.gui.protocol import GUI_HOST_ENV, GUI_PORT_ENV, GuiCommandType
from evomachine.gui.socket_transport import GuiRpcServer, GuiSocketClient


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_HARDWARE_RUNTIME = "evomachine.gui.runtime:build_hardware_automaton"


def _repo_root() -> Path:
    """Return the source checkout root that contains the evomachine package."""
    return Path(__file__).resolve().parents[2]


def _build_common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--image-transport",
        default=None,
        type=normalise_image_transport,
        choices=IMAGE_TRANSPORT_CHOICES,
        help="How acquired image previews are sent to Napari: auto, temp_tiff, socket_tiff, or raw.",
    )
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


def _require_hardware_gui_mmc_camera(automaton: Any) -> None:
    from evomachine.bindings.binding_types import BindingType

    acq_mngr = getattr(automaton, "acq_mngr", None)
    camera = getattr(acq_mngr, "camera", None) or getattr(automaton, "_camera", None)
    if camera is None:
        raise RuntimeError("Hardware GUI runtime must provide an acquisition camera.")

    config = getattr(camera, "config", None)
    binding = getattr(config, "binding", None)
    if binding != BindingType.MMC:
        camera_name = getattr(camera, "name", type(camera).__name__)
        binding_name = getattr(binding, "name", str(binding))
        raise RuntimeError(
            "Hardware GUI camera must use BindingType.MMC (Micro-Manager); "
            f"{camera_name} is configured with {binding_name}."
        )


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


def _virtual_automaton_process(host: str, port: int, ready_queue) -> None:
    from evomachine.gui.runtime import build_virtual_automaton

    _serve_automaton(build_virtual_automaton(), host=host, port=port, ready_queue=ready_queue)


def _hardware_automaton_process(runtime_spec: str, host: str, port: int, ready_queue) -> None:
    try:
        factory = _load_runtime_factory(runtime_spec)
        automaton = factory()
        _require_hardware_gui_mmc_camera(automaton)
        _serve_automaton(automaton, host=host, port=port, ready_queue=ready_queue)
    except Exception as error:
        ready_queue.put({"error": f"{type(error).__name__}: {error}"})
        raise


def _run_napari(host: str, port: int, napari_args: Sequence[str], *, image_transport: str | None = None) -> int:
    env = dict(os.environ)
    env[GUI_HOST_ENV] = host
    env[GUI_PORT_ENV] = str(port)
    if image_transport is not None:
        env[IMAGE_TRANSPORT_ENV] = normalise_image_transport(image_transport)
    repo_root = str(_repo_root())
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = repo_root if not pythonpath else os.pathsep.join([repo_root, pythonpath])
    command = [sys.executable, "-m", "evomachine.gui.napari_app", *napari_args]
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


def _launch_with_process(
        process: mp.Process,
        ready_queue,
        no_napari: bool,
        napari_args: Sequence[str],
        *,
        image_transport: str | None = None,
) -> int:
    process.start()
    ready = ready_queue.get(timeout=15.0)
    if isinstance(ready, dict) and "error" in ready:
        process.join(timeout=1.0)
        raise RuntimeError(f"Automaton process failed to start: {ready['error']}")
    host, port = ready
    if no_napari:
        _shutdown_child(process=process, host=host, port=port)
        return 0
    try:
        return _run_napari(host=host, port=port, napari_args=napari_args, image_transport=image_transport)
    finally:
        _shutdown_child(process=process, host=host, port=port)


def virtual_main(argv: Sequence[str] | None = None) -> int:
    parser = _build_common_parser("Launch the evomachine GUI with virtual peripherals.")
    args = parser.parse_args(argv)
    ready_queue = mp.Queue()
    process = mp.Process(
        target=_virtual_automaton_process,
        name="EvoMachineVirtualAutomaton",
        args=(args.host, args.port, ready_queue),
    )
    return _launch_with_process(
        process=process,
        ready_queue=ready_queue,
        no_napari=args.no_napari,
        napari_args=args.napari_args,
        image_transport=args.image_transport,
    )


def hardware_main(argv: Sequence[str] | None = None) -> int:
    parser = _build_common_parser("Launch the evomachine GUI with hardware peripherals.")
    parser.add_argument(
        "--runtime",
        default=DEFAULT_HARDWARE_RUNTIME,
        help="Callable in 'module:function' format returning an Automaton.",
    )
    args = parser.parse_args(argv)
    ready_queue = mp.Queue()
    process = mp.Process(
        target=_hardware_automaton_process,
        name="EvoMachineHardwareAutomaton",
        args=(args.runtime, args.host, args.port, ready_queue),
    )
    return _launch_with_process(
        process=process,
        ready_queue=ready_queue,
        no_napari=args.no_napari,
        napari_args=args.napari_args,
        image_transport=args.image_transport,
    )


if __name__ == "__main__":
    raise SystemExit(virtual_main(sys.argv[1:]))
