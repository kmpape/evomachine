"""Exercise generation RPC and lifecycle without network calls or microscope access."""

import json
import threading
import time

import pytest

from evomachine.gui.facade import AutomatonGuiFacade
from evomachine.gui.protocol import GuiCommandType, GuiRequest
from evomachine.strategy_generation.files import StrategyFileStore
from evomachine.strategy_generation.preview import GenerationPreview
from tests.gui.test_facade import FakeAutomaton
from tests.test_strategy_generation_integration import _verified


def request(facade, command, **payload):
    return facade.handle(GuiRequest(command=command, payload=payload))


def finish_generation(facade):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        result = request(facade, GuiCommandType.STRATEGY_GENERATION_STATUS)
        if result.payload["generation"]["state"] != "running":
            return result.payload["generation"]
        time.sleep(0.005)
    pytest.fail("Generation did not finish")


def test_background_generation_does_not_lock_hardware_and_requires_explicit_set(monkeypatch):
    from evomachine.gui import strategy_generation

    verified = _verified("initialise\nstep\n    terminate\nfinalise\n")
    started, release = threading.Event(), threading.Event()

    def generate(prompt, cancel_event):
        assert prompt == "finish"
        assert not cancel_event.is_set()
        started.set()
        assert release.wait(3)
        return GenerationPreview(elapsed_seconds=1, verified=verified)

    monkeypatch.setattr(strategy_generation, "generate_cancellable", generate)
    automaton = FakeAutomaton()
    facade = AutomatonGuiFacade(automaton)
    facade.autostrat_enabled = True  # Model requests below are mocked.
    try:
        first = request(facade, GuiCommandType.STRATEGY_GENERATE, prompt="finish")
        assert first.ok and started.wait(1)
        generation_id = first.payload["generation"]["operation_id"]
        assert first.payload["generation"]["state"] == "running"
        assert facade.gui_operations.active() is None
        assert request(facade, GuiCommandType.STRATEGY_SET, name="NoStrategy").ok
        assert not request(facade, GuiCommandType.STRATEGY_SET, generation_id=generation_id).ok
        assert not request(facade, GuiCommandType.STRATEGY_GENERATE, prompt="finish").ok
        release.set()
        generated = finish_generation(facade)
        assert generated["accepted"] and generated["dsl"] == verified.source
        json.dumps(generated)  # No Python strategy objects cross the wire.
        assert automaton._strategy.name() == "NoStrategy"
        assert not request(facade, GuiCommandType.STRATEGY_SET, generation_id="stale").ok
        installed = request(facade, GuiCommandType.STRATEGY_SET, generation_id=generation_id)
        assert installed.ok and installed.payload["strategy"]["generation_id"] == generation_id
        assert automaton._strategy.source == verified.source
        assert not automaton.strategy_started
        assert request(facade, GuiCommandType.FOV_INITIALISE, fovs=[{
            "fov_id": 0, "x": 0, "y": 0, "z": 0,
        }]).ok
        assert request(facade, GuiCommandType.STRATEGY_START).ok
        assert automaton.strategy_started
        assert request(facade, GuiCommandType.STRATEGY_STOP).ok
    finally:
        release.set()


@pytest.mark.parametrize("exception", [False, True])
def test_failed_generation_has_diagnostics_and_cannot_be_installed(monkeypatch, exception):
    from evomachine.gui import strategy_generation

    def generate(prompt, cancel_event):
        assert not cancel_event.is_set()
        if exception:
            raise ValueError("Missing model configuration")
        return GenerationPreview(elapsed_seconds=1, error=RuntimeError("Rejected candidate"))

    monkeypatch.setattr(strategy_generation, "generate_cancellable", generate)
    facade = AutomatonGuiFacade(FakeAutomaton())
    facade.autostrat_enabled = True
    assert request(facade, GuiCommandType.STRATEGY_GENERATE, prompt="finish").ok
    result = finish_generation(facade)
    assert not result["accepted"] and result["dsl"] == "" and result["diagnostics"]
    assert not request(facade, GuiCommandType.STRATEGY_SET, generation_id=result["operation_id"]).ok
    assert facade.automaton._strategy is None


def test_generation_worker_has_event_loop_and_reuses_preview_diagnostics(monkeypatch):
    import asyncio
    from types import SimpleNamespace
    from evomachine.gui import strategy_generation

    verified = _verified("initialise\nstep\n    terminate\nfinalise\n")

    def run(prompt):
        assert not asyncio.get_event_loop().is_closed()
        return verified

    monkeypatch.setattr(strategy_generation, "create_pipeline", lambda: SimpleNamespace(run=run))
    previews = []
    thread = threading.Thread(target=lambda: previews.append(strategy_generation.generate("finish")))
    thread.start()
    thread.join(3)
    assert not thread.is_alive()
    assert previews[0].verified is verified
    assert "Semantic candidates: 1" in previews[0].diagnostics()


def test_cancellable_generation_stops_its_worker_process(monkeypatch):
    from evomachine.gui import strategy_generation

    cancel_event = threading.Event()
    cancel_event.set()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    preview = strategy_generation.generate_cancellable("finish", cancel_event)
    assert isinstance(preview.error, RuntimeError)
    assert "cancelled" in str(preview.error).lower()


def test_generation_timeout_configuration_is_validated(monkeypatch):
    from evomachine.gui import strategy_generation

    monkeypatch.setenv(strategy_generation.GENERATION_TIMEOUT_ENV, "0")
    with pytest.raises(ValueError, match="positive number"):
        strategy_generation.generation_timeout_seconds()


def test_generation_can_be_cancelled_without_waiting_for_model_completion(monkeypatch):
    from evomachine.gui import strategy_generation

    started = threading.Event()

    def generate(prompt, cancel_event):
        assert prompt == "finish"
        started.set()
        assert cancel_event.wait(3)
        return GenerationPreview(elapsed_seconds=0, error=RuntimeError("cancelled"))

    monkeypatch.setattr(strategy_generation, "generate_cancellable", generate)
    facade = AutomatonGuiFacade(FakeAutomaton())
    facade.autostrat_enabled = True
    assert request(facade, GuiCommandType.STRATEGY_GENERATE, prompt="finish").ok
    assert started.wait(1)
    cancelled = request(facade, GuiCommandType.STRATEGY_GENERATION_CANCEL)
    assert cancelled.ok
    result = finish_generation(facade)
    assert result["state"] == "cancelled"
    assert not result["accepted"]


def test_blank_startup_key_disables_autostrat_even_with_inherited_key(monkeypatch):
    import os

    monkeypatch.setenv("OPENAI_API_KEY", "inherited-test-key")
    facade = AutomatonGuiFacade(FakeAutomaton())
    response = request(facade, GuiCommandType.AUTOSTRAT_CONFIGURE, api_key="session-test-key")
    assert response.payload == {"autostrat": {"enabled": True}}
    assert os.environ["OPENAI_API_KEY"] == "session-test-key"
    response = request(facade, GuiCommandType.AUTOSTRAT_CONFIGURE, api_key="")
    assert response.payload == {"autostrat": {"enabled": False}}
    assert "OPENAI_API_KEY" not in os.environ
    assert not request(facade, GuiCommandType.STRATEGY_GENERATE, prompt="finish").ok
    assert request(facade, GuiCommandType.STRATEGY_SET, name="NoStrategy").ok


def test_user_authored_strategies_save_load_validate_and_install_without_api_key(tmp_path):
    facade = AutomatonGuiFacade(FakeAutomaton())
    facade.autostrat_strategy_store = StrategyFileStore(tmp_path)
    source = "initialise\nstep\n    terminate\nfinalise\n"

    assert request(facade, GuiCommandType.AUTOSTRAT_STRATEGY_LIST).payload == {
        "autostrat_strategies": [],
    }
    saved = request(
        facade,
        GuiCommandType.AUTOSTRAT_STRATEGY_SAVE,
        name="My strategy.strat",
        source=source,
        overwrite=False,
    )
    assert saved.ok
    assert saved.payload["autostrat_strategy"] == {
        "name": "My strategy",
        "filename": "My strategy.strat",
        "source": source,
        "saved": True,
    }
    assert (tmp_path / "My strategy.strat").read_text() == source
    assert not request(
        facade,
        GuiCommandType.AUTOSTRAT_STRATEGY_SAVE,
        name="My strategy",
        source=source,
        overwrite=False,
    ).ok

    loaded = request(
        facade, GuiCommandType.AUTOSTRAT_STRATEGY_LOAD, name="My strategy"
    )
    assert loaded.ok and loaded.payload["autostrat_strategy"]["source"] == source
    validated = request(
        facade, GuiCommandType.AUTOSTRAT_STRATEGY_VALIDATE, source=source
    )
    assert validated.ok and validated.payload["autostrat_validation"]["valid"]

    installed = request(
        facade,
        GuiCommandType.STRATEGY_SET,
        autostrat_name="My strategy",
        autostrat_source=source,
    )
    assert installed.ok
    assert installed.payload["strategy"]["autostrat_name"] == "My strategy"
    assert facade.automaton._strategy.source == source


def test_strategy_file_store_rejects_unsafe_names_and_invalid_source(tmp_path):
    store = StrategyFileStore(tmp_path)
    with pytest.raises(ValueError, match="Strategy name"):
        store.save("../escape", "initialise\nstep\nfinalise\n")
    with pytest.raises(ValueError, match="non-empty"):
        store.save("empty", "  \n")


def test_strategy_file_store_preserves_source_text(tmp_path):
    store = StrategyFileStore(tmp_path)
    source = "\ninitialise\nstep\nfinalise"
    store.save("exact", source)
    _, loaded = store.load("exact")
    assert loaded == source
