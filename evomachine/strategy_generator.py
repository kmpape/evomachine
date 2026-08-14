"""Generate, validate, and optionally save strategy DSL using OpenAI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from evomachine.dsl import validator as dsl_validator
from evomachine.dsl.errors import DSLGenerationError
from evomachine.dsl.model import ParsedStrategy
from evomachine.dsl.parser import DSLParser
from evomachine.prompt_constructor import PromptConstructor


DEFAULT_MODEL = "gpt-5.6"
DEFAULT_MAX_OUTPUT_TOKENS = 2_000


class _OpenAIResponse(Protocol):
    """Subset of an OpenAI response used by the strategy generator."""

    id: str
    output_text: str


class _ResponsesAPI(Protocol):
    """Subset of the OpenAI Responses API used by the generator."""

    def create(
            self,
            *,
            model: str,
            input: str,
            max_output_tokens: int,
    ) -> _OpenAIResponse: ...


class _OpenAIClient(Protocol):
    """Client interface required by ``StrategyGenerator``."""

    responses: _ResponsesAPI


@dataclass(frozen=True)
class GeneratedStrategy:
    """Hold generated DSL together with its validated representation."""

    dsl_text: str
    parsed_strategy: ParsedStrategy
    model: str
    response_id: str


class StrategyGenerator:
    """Generate strategy DSL through OpenAI and validate it locally.

    By default, the OpenAI SDK reads the API key from the
    ``OPENAI_API_KEY`` environment variable. A compatible client can be
    supplied for testing.
    """

    def __init__(
            self,
            model: str = DEFAULT_MODEL,
            max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
            client: _OpenAIClient | None = None,
            parser: DSLParser | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("model must not be empty.")
        if max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive.")

        self.model = model
        self.max_output_tokens = max_output_tokens
        self.client = client or self._create_openai_client()
        self.parser = parser or DSLParser()

    @staticmethod
    def _create_openai_client() -> _OpenAIClient:
        """Create the official client using environment-based authentication."""
        from openai import OpenAI

        return cast(_OpenAIClient, OpenAI())

    @staticmethod
    def _extract_dsl(response_text: str) -> str:
        """Extract plain DSL, tolerating one enclosing Markdown code fence."""
        dsl_text = response_text.strip()
        if not dsl_text:
            raise DSLGenerationError("OpenAI returned an empty strategy response.")

        lines = dsl_text.splitlines()
        if len(lines) >= 2 and lines[0].strip().startswith("```"):
            if lines[-1].strip() != "```":
                raise DSLGenerationError("OpenAI returned an unterminated Markdown code fence.")
            dsl_text = "\n".join(lines[1:-1]).strip()

        if not dsl_text:
            raise DSLGenerationError("OpenAI returned no DSL inside its code fence.")
        return f"{dsl_text}\n"

    def generate(self, natural_language_input: str) -> GeneratedStrategy:
        """Generate DSL, parse it, validate it, and return the result."""
        prompt = PromptConstructor(nat_lang_input=natural_language_input).construct_prompt()
        response = self.client.responses.create(
            model=self.model,
            input=prompt,
            max_output_tokens=self.max_output_tokens,
        )
        dsl_text = self._extract_dsl(response.output_text)
        parsed_strategy = dsl_validator.validate_strategy(
            self.parser.parse(dsl_text=dsl_text)
        )
        return GeneratedStrategy(
            dsl_text=dsl_text,
            parsed_strategy=parsed_strategy,
            model=self.model,
            response_id=response.id,
        )


def save_generated_strategy(
        generated_strategy: GeneratedStrategy,
        destination: str | Path,
        *,
        overwrite: bool = False,
) -> Path:
    """Save validated DSL to a ``.dsl`` file and return its resolved path."""
    destination_path = Path(destination).expanduser()
    if destination_path.suffix.lower() != ".dsl":
        raise ValueError("Generated strategies must be saved with a .dsl extension.")
    if destination_path.exists() and not overwrite:
        raise FileExistsError(f"Strategy file already exists: {destination_path}")

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.write_text(generated_strategy.dsl_text, encoding="utf-8")
    return destination_path.resolve()
