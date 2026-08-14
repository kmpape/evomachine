"""Parse strategy DSL text into its intermediate representation."""

from importlib.resources import files

from lark import Lark, Token, Tree
from lark.exceptions import UnexpectedInput
from lark.indenter import Indenter

from evomachine.dsl.errors import DSLSyntaxError
from evomachine.dsl.model import (
    CaptureImage,
    Move,
    MoveTarget,
    ParsedCommand,
    ParsedStrategy,
    Wait,
)
from evomachine.types import LEDType


LED_WAVELENGTHS = {
    "385nm": LEDType.LED_385_NM,
    "450nm": LEDType.LED_450_NM,
    "515nm": LEDType.LED_515_NM,
    "565nm": LEDType.LED_565_NM,
    "645nm": LEDType.LED_645_NM,
}


class DSLIndenter(Indenter):
    """Generate indentation tokens for the strategy DSL grammar."""

    NL_type = "_NL"
    INDENT_type = "_INDENT"
    DEDENT_type = "_DEDENT"
    OPEN_PAREN_types: list[str] = []
    CLOSE_PAREN_types: list[str] = []
    tab_len = 8


class DSLParser:
    """Parse and transform source text written in the EvoMachine strategy DSL."""

    def __init__(self, grammar_text: str | None = None) -> None:
        self.grammar_text = grammar_text or files("evomachine.dsl").joinpath("grammar.ebnf").read_text()
        self._parser = Lark(self.grammar_text, parser="lalr", postlex=DSLIndenter())

    def parse_tree(self, dsl_text: str) -> Tree[Token]:
        """Parse source text and return its Lark parse tree."""
        if not isinstance(dsl_text, str):
            raise TypeError(f"dsl_text must be str, received {type(dsl_text).__name__}.")
        source = dsl_text if dsl_text.endswith(("\n", "\r")) else f"{dsl_text}\n"
        try:
            return self._parser.parse(source)
        except UnexpectedInput as error:
            context = error.get_context(source).strip()
            message = f"Invalid strategy DSL at line {error.line}, column {error.column}."
            if context:
                message = f"{message}\n{context}"
            raise DSLSyntaxError(message) from error

    @staticmethod
    def _transform_instruction(tree: Tree[Token]) -> ParsedCommand:
        """Convert one Lark instruction tree into a parsed command."""
        rule_name = str(tree.data)
        if rule_name == "move":
            return Move(target=MoveTarget(str(tree.children[0])))
        if rule_name == "image":
            return CaptureImage(
                exposure_ms=float(tree.children[0]),
                led_type=(
                    LED_WAVELENGTHS[str(tree.children[1])]
                    if len(tree.children) > 1 and tree.children[1] is not None
                    else None
                ),
            )
        if rule_name == "wait":
            return Wait(duration_seconds=float(tree.children[0]))
        raise DSLSyntaxError(f"Unsupported DSL instruction rule: {rule_name}.")

    def _transform_section(self, section_tree: Tree[Token]) -> tuple[ParsedCommand, ...]:
        """Convert one lifecycle parse-tree node into parsed commands."""
        return tuple(
            self._transform_instruction(instruction_tree)
            for instruction_tree in section_tree.children
            if isinstance(instruction_tree, Tree)
        )

    def transform(self, parse_tree: Tree[Token]) -> ParsedStrategy:
        """Convert a complete Lark parse tree into a parsed strategy."""
        try:
            initialise_tree = next(parse_tree.find_data("initialise"))
            callback_tree = next(parse_tree.find_data("callback"))
            finalise_tree = next(parse_tree.find_data("finalise"))
        except StopIteration as error:
            raise DSLSyntaxError("The strategy must define initialise, callback, and finalise sections.") from error

        return ParsedStrategy(
            initialise=self._transform_section(initialise_tree),
            callback=self._transform_section(callback_tree),
            finalise=self._transform_section(finalise_tree),
        )

    def parse(self, dsl_text: str) -> ParsedStrategy:
        """Parse DSL source text and return its intermediate representation."""
        return self.transform(self.parse_tree(dsl_text=dsl_text))


def parse_dsl(dsl_text: str) -> ParsedStrategy:
    """Parse DSL text using the packaged grammar."""
    return DSLParser().parse(dsl_text=dsl_text)
