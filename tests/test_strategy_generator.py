"""Tests for OpenAI-backed strategy DSL generation."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from evomachine.dsl import CaptureImage, DSLGenerationError, DSLSyntaxError
from evomachine.strategy_generator import StrategyGenerator, save_generated_strategy


VALID_DSL = """initialise
    move first_fov

callback
    image exposure 50ms

finalise
"""

TEN_SECOND_WAIT_DSL = """initialise

callback
    wait 10.0s

finalise
"""


class FakeResponsesAPI:
    """Return a configured response while recording request arguments."""

    def __init__(self, output_text: str) -> None:
        self.output_text = output_text
        self.request: dict[str, object] = {}

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.request = kwargs
        return SimpleNamespace(id="response-123", output_text=self.output_text)


class FakeOpenAIClient:
    """Minimal client matching the interface used by ``StrategyGenerator``."""

    def __init__(self, output_text: str) -> None:
        self.responses = FakeResponsesAPI(output_text)


def test_generate_returns_validated_strategy() -> None:
    client = FakeOpenAIClient(VALID_DSL)
    generator = StrategyGenerator(client=client)

    generated = generator.generate("Take a 50 ms image on every callback.")

    assert generated.dsl_text == VALID_DSL
    assert generated.parsed_strategy.callback == (CaptureImage(exposure_ms=50.0),)
    assert generated.response_id == "response-123"
    assert client.responses.request["model"] == "gpt-5.6"
    assert "## USER REQUEST" in str(client.responses.request["input"])


def test_generate_accepts_an_enclosing_code_fence() -> None:
    client = FakeOpenAIClient(f"```dsl\n{VALID_DSL}```")

    generated = StrategyGenerator(client=client).generate("Generate a strategy.")

    assert generated.dsl_text == VALID_DSL


def test_generate_accepts_ten_second_wait() -> None:
    generated = StrategyGenerator(client=FakeOpenAIClient(TEN_SECOND_WAIT_DSL)).generate(
        "Wait for ten seconds on every callback."
    )

    assert generated.parsed_strategy.callback[0].duration_seconds == 10.0


def test_generate_rejects_empty_or_invalid_responses() -> None:
    with pytest.raises(DSLGenerationError, match="empty"):
        StrategyGenerator(client=FakeOpenAIClient("  ")).generate("Generate a strategy.")

    with pytest.raises(DSLSyntaxError):
        StrategyGenerator(client=FakeOpenAIClient("not valid DSL")).generate(
            "Generate a strategy."
        )


def test_save_generated_strategy_writes_validated_dsl(tmp_path: Path) -> None:
    generated = StrategyGenerator(client=FakeOpenAIClient(VALID_DSL)).generate(
        "Generate a strategy."
    )
    destination = tmp_path / "generated" / "strategy.dsl"

    saved_path = save_generated_strategy(generated, destination)

    assert saved_path == destination.resolve()
    assert destination.read_text(encoding="utf-8") == VALID_DSL
    with pytest.raises(FileExistsError):
        save_generated_strategy(generated, destination)
