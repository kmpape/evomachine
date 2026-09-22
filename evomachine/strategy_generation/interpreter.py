"""Application boundary for the domain-independent AutoStrat evaluator."""

from autostrat.domain import DomainPack
from autostrat.language.evaluator import EvaluationResult, StrategyEvaluator
from autostrat.language.model import ValidatedStatement

from evomachine.strategy_generation.runtime import StrategyRuntimeContext


class ConditionalInterpreter:
    def __init__(self, domain: DomainPack) -> None:
        self._evaluator = StrategyEvaluator(domain)

    def interpret(
        self,
        statements: tuple[ValidatedStatement, ...],
        context: StrategyRuntimeContext,
        *,
        section: str = "step",
    ) -> EvaluationResult:
        return self._evaluator.evaluate(statements, context.observations, section=section)
