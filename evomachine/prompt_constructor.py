"""Construct prompts for generating EvoMachine strategy DSL text."""

from importlib.resources import files
from typing import Final


PROMPT_RESOURCE_PACKAGE: Final[str] = "evomachine.dsl"
GRAMMAR_FILE: Final[str] = "grammar.ebnf"
SEMANTIC_GUIDANCE_FILE: Final[str] = "semantic_guidance.txt"
FEW_SHOT_EXAMPLES_FILE: Final[str] = "few_shot_examples.txt"


class PromptConstructor:
    """Combine DSL instructions, references, and a user request into one prompt.

    This class only constructs text. Sending the prompt to an LLM and handling
    its response belong to the generation layer.
    """

    def __init__(
            self,
            nat_lang_input: str,
            grammar_text: str | None = None,
            semantic_guidance_text: str | None = None,
            few_shot_examples_text: str | None = None,
    ) -> None:
        if not isinstance(nat_lang_input, str):
            raise TypeError(
                "nat_lang_input must be str, "
                f"received {type(nat_lang_input).__name__}."
            )

        self.nat_lang_input = nat_lang_input
        self.grammar_text = self._resolve_text(grammar_text, GRAMMAR_FILE)
        self.semantic_guidance_text = self._resolve_text(
            semantic_guidance_text,
            SEMANTIC_GUIDANCE_FILE,
        )
        self.few_shot_examples_text = self._resolve_text(
            few_shot_examples_text,
            FEW_SHOT_EXAMPLES_FILE,
        )
        self.prompt = ""

    @staticmethod
    def _read_resource(filename: str) -> str:
        """Read a UTF-8 prompt resource packaged with the DSL."""
        return (
            files(PROMPT_RESOURCE_PACKAGE)
            .joinpath(filename)
            .read_text(encoding="utf-8")
            .strip()
        )

    @classmethod
    def _resolve_text(cls, supplied_text: str | None, filename: str) -> str:
        """Use explicitly supplied text, otherwise load its packaged resource."""
        return supplied_text if supplied_text is not None else cls._read_resource(filename)

    def _append_section(self, label: str, content: str) -> None:
        """Append a labelled prompt section separated by blank lines."""
        self.prompt += f"## {label}\n{content.strip()}\n\n"

    def _grammar_prompter(self) -> None:
        """Append the formal DSL grammar."""
        self._append_section("DSL GRAMMAR", self.grammar_text)

    def _semantic_prompter(self) -> None:
        """Append command meanings and semantic constraints."""
        self._append_section("DSL SEMANTICS", self.semantic_guidance_text)

    def _few_shot_prompter(self) -> None:
        """Append examples mapping natural-language requests to DSL programs."""
        self._append_section("EXAMPLES", self.few_shot_examples_text)

    def construct_prompt(self) -> str:
        """Build and return a complete prompt for DSL generation.

        Rebuilding resets the accumulated text, making repeated calls
        idempotent.
        """
        self.prompt = (
            "You generate EvoMachine imaging strategies in the strategy DSL.\n"
            "Translate the user request using only the grammar and semantics "
            "provided below.\n"
            "Do not invent commands or syntax.\n\n"
        )
        self._grammar_prompter()
        self._semantic_prompter()
        self._few_shot_prompter()
        self._append_section("USER REQUEST", self.nat_lang_input)
        self.prompt += (
            "## RESPONSE REQUIREMENTS\n"
            "Return only the complete DSL strategy as plain text.\n"
            "Do not use Markdown fences, commentary, or explanations.\n"
            "If the request is nonsensical or cannot be represented using the "
            "supported DSL commands, return an empty strategy: include only "
            "the initialise, callback, and finalise section headings.\n"
            "Include initialise, callback, and finalise in that order, even "
            "when a section is empty.\n"
        )
        return self.prompt
