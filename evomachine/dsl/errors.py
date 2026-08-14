"""Exceptions raised while processing strategy DSL source text."""


class DSLError(Exception):
    """Base class for all strategy DSL errors."""


class DSLSyntaxError(DSLError):
    """Raised when DSL text does not conform to the grammar."""


class DSLValidationError(DSLError):
    """Raised when parsed DSL is structurally valid but semantically invalid."""


class DSLInterpretationError(DSLError):
    """Raised when parsed DSL cannot be mapped to automaton commands."""


class DSLGenerationError(DSLError):
    """Raised when an LLM response does not contain strategy DSL text."""
