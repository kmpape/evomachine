"""Hardware-free generation previews for the lightweight testing notebook."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from autostrat.pipeline import StrategyAttempt, VerifiedStrategy

from evomachine.strategy_generation.service import StrategyPipelineRunner


PYTHON_WRAPPER = """from evomachine.strategy_generation import (
    AutoStratStrategy,
    MicroscopyCommandAdapter,
    MicroscopyObservationProvider,
    MicroscopyRuntimeErrorProvider,
)


def build_strategy(verified, domain, cfg):
    # verified is the accepted DSL result displayed above.
    # The shared evaluator handles expressions, conditions and recovery.
    return AutoStratStrategy(
        cfg=cfg,
        verified=verified,
        domain=domain,
        command_adapter=MicroscopyCommandAdapter(
            segment_images=False, save_images=False,
        ),
        observation_provider=MicroscopyObservationProvider(),
        runtime_error_provider=MicroscopyRuntimeErrorProvider(),
    )


# When ready, supply your application configuration:
# strategy = build_strategy(preview.verified, domain, cfg)
# Constructing a strategy does not start the microscope.
"""


@dataclass(frozen=True, slots=True)
class GenerationPreview:
    """One run, with no stale accepted output carried over from earlier requests."""

    elapsed_seconds: float
    verified: VerifiedStrategy | None = None
    error: Exception | None = None

    @property
    def dsl(self) -> str | None:
        return self.verified.source if self.verified is not None else None

    @property
    def python(self) -> str | None:
        return PYTHON_WRAPPER if self.verified is not None else None

    @property
    def attempts(self) -> tuple[StrategyAttempt, ...]:
        if self.verified is not None:
            return self.verified.attempts
        attempts = getattr(self.error, "attempts", ())
        return attempts if isinstance(attempts, tuple) else ()

    def diagnostics(self) -> str:
        lines = [
            f"Status: {'accepted' if self.verified is not None else 'failed'}",
            f"Elapsed: {self.elapsed_seconds:.2f} s",
        ]
        if self.attempts:
            lines.extend(
                (
                    f"Semantic candidates: {len(self.attempts)}",
                    f"Semantic revisions: {len(self.attempts) - 1}",
                )
            )
        if self.error is not None:
            lines.append(f"{type(self.error).__name__}: {self.error}")
            count = getattr(self.error, "attempts", None)
            if type(count) is int:
                lines.append(f"Failed deterministic attempts in final generation: {count}")
        lines.append(
            "Deterministic retries and verifier-format retries on successful runs: "
            "not exposed by AutoStrat (not assumed to be zero)."
        )
        lines.append("Hardware execution: not run.")
        for attempt in self.attempts:
            lines.append(f"\nCandidate {attempt.number}: accepted={attempt.verdict.accepted}")
            for issue in attempt.verdict.issues:
                lines.append(f"  {issue.category}: {issue.message}")
            lines.append(attempt.generated.source)
        return "\n".join(lines)


def generate_preview(pipeline: StrategyPipelineRunner, request: str) -> GenerationPreview:
    """Run the real pipeline, but never construct or execute microscope commands."""
    started = perf_counter()
    try:
        verified = pipeline.run(request)
    except Exception as error:
        return GenerationPreview(elapsed_seconds=perf_counter() - started, error=error)
    return GenerationPreview(elapsed_seconds=perf_counter() - started, verified=verified)
