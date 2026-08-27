"""Evaluate the small, validated AutoStrat conditional language."""

from __future__ import annotations

import operator
from collections.abc import Callable

from autostrat.language.model import (
    ControlAction,
    QuantityValue,
    ReferenceExpression,
    ValidatedComparisonExpression,
    ValidatedIfStatement,
    ValidatedStatement,
)

from evomachine.strategy_generation.runtime import (
    ActiveRuntimeError,
    InterpretationResult,
    StrategyInterpretationError,
    StrategyRuntimeContext,
)


_OPERATORS: dict[str, Callable[[object, object], bool]] = {
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
    ">=": operator.ge,
    ">": operator.gt,
}


class ConditionalInterpreter:
    """Select validated command calls from one runtime-state snapshot."""

    def interpret(
        self,
        statements: tuple[ValidatedStatement, ...],
        context: StrategyRuntimeContext,
    ) -> InterpretationResult:
        if not isinstance(context, StrategyRuntimeContext):
            raise TypeError("context must be a StrategyRuntimeContext.")
        return self._interpret(statements, context, retry_error=None)

    def _interpret(
        self,
        statements: tuple[ValidatedStatement, ...],
        context: StrategyRuntimeContext,
        *,
        retry_error: ActiveRuntimeError | None,
    ) -> InterpretationResult:
        calls = []
        for statement in statements:
            if isinstance(statement, ValidatedIfStatement):
                condition_is_true = self._condition(statement.condition, context)
                branch = statement.body if condition_is_true else statement.else_body
                branch_retry_error = retry_error
                if condition_is_true and isinstance(statement.condition, ReferenceExpression):
                    if statement.condition.namespace == "error":
                        branch_retry_error = context.errors[statement.condition.name]
                branch_result = self._interpret(
                    branch,
                    context,
                    retry_error=branch_retry_error,
                )
                calls.extend(branch_result.calls)
                if branch_result.action is not None:
                    return InterpretationResult(
                        calls=tuple(calls),
                        action=branch_result.action,
                        retry_error=branch_result.retry_error,
                    )
                continue

            if isinstance(statement, ControlAction):
                if statement.action == "retry":
                    if retry_error is None or retry_error.failed_call is None:
                        raise StrategyInterpretationError(
                            "retry requires an active error associated with a failed validated command."
                        )
                    return InterpretationResult(
                        calls=tuple(calls),
                        action="retry",
                        retry_error=retry_error,
                    )
                return InterpretationResult(calls=tuple(calls), action=statement.action)

            calls.append(statement)
        return InterpretationResult(calls=tuple(calls))

    def _condition(self, expression, context: StrategyRuntimeContext) -> bool:
        if isinstance(expression, ReferenceExpression):
            if expression.namespace == "error":
                return expression.name in context.errors
            if expression.name not in context.observations:
                raise StrategyInterpretationError(
                    f"Required observation {expression.name!r} was not supplied."
                )
            value = context.observations[expression.name]
            if type(value) is not bool:
                raise StrategyInterpretationError(
                    f"Bare observation {expression.name!r} must have a boolean runtime value."
                )
            return value

        if not isinstance(expression, ValidatedComparisonExpression):
            raise StrategyInterpretationError(
                f"Unsupported validated condition {type(expression).__name__}."
            )
        if expression.left.namespace != "observation":
            raise StrategyInterpretationError("Only observations can be compared with constants.")
        name = expression.left.name
        if name not in context.observations:
            raise StrategyInterpretationError(f"Required observation {name!r} was not supplied.")
        left, right = self._comparison_values(context.observations[name], expression.right)
        try:
            return bool(_OPERATORS[expression.operator](left, right))
        except TypeError as error:
            raise StrategyInterpretationError(
                f"Observation {name!r} cannot be compared with its validated constant."
            ) from error

    @staticmethod
    def _comparison_values(left, right) -> tuple[object, object]:
        if isinstance(right, QuantityValue):
            if not isinstance(left, QuantityValue) or left.unit != right.unit:
                raise StrategyInterpretationError(
                    f"Quantity comparison requires matching unit {right.unit!r}."
                )
            return left.magnitude, right.magnitude
        if isinstance(left, QuantityValue):
            raise StrategyInterpretationError("A quantity observation requires a quantity constant.")
        return left, right


__all__ = ["ConditionalInterpreter"]
