"""Build AutoStrat-backed strategies synchronously or on one background worker."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Protocol

from autostrat.domain import DomainPack
from autostrat.pipeline import VerifiedStrategy

from evomachine.image_processing_config import ImageProcessorConfig
from evomachine.strategy_generation.interfaces import (
    CommandAdapter,
    EmptyObservationProvider,
    EmptyRuntimeErrorProvider,
    ObservationProvider,
    RuntimeErrorProvider,
)
from evomachine.strategy_generation.strategy import AutoStratStrategy


class StrategyPipelineRunner(Protocol):
    """Structural interface implemented by AutoStrat's synchronous pipeline."""

    def run(self, request: str) -> VerifiedStrategy:
        """Return a verified strategy for one natural-language request."""


class StrategyGenerationService:
    """Keep the synchronous model pipeline off GUI and automaton event-loop threads."""

    def __init__(
        self,
        pipeline: StrategyPipelineRunner,
        *,
        domain: DomainPack,
        command_adapter: CommandAdapter,
        observation_provider: ObservationProvider | None = None,
        runtime_error_provider: RuntimeErrorProvider | None = None,
    ) -> None:
        if not callable(getattr(pipeline, "run", None)):
            raise TypeError("pipeline must expose a callable run(request) method.")
        if not isinstance(domain, DomainPack):
            raise TypeError("domain must be a DomainPack.")
        if not isinstance(command_adapter, CommandAdapter):
            raise TypeError("command_adapter must be a CommandAdapter.")
        self._pipeline = pipeline
        self._domain = domain
        self._command_adapter = command_adapter
        self._observation_provider = observation_provider or EmptyObservationProvider()
        self._runtime_error_provider = runtime_error_provider or EmptyRuntimeErrorProvider()
        if not isinstance(self._observation_provider, ObservationProvider):
            raise TypeError("observation_provider must be an ObservationProvider.")
        if not isinstance(self._runtime_error_provider, RuntimeErrorProvider):
            raise TypeError("runtime_error_provider must be a RuntimeErrorProvider.")
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="strategy-generation")

    def build(self, request: str, cfg: ImageProcessorConfig) -> AutoStratStrategy:
        """Build on the calling thread, primarily for scripts and notebooks."""
        verified = self._pipeline.run(request)
        return AutoStratStrategy(
            cfg=cfg,
            verified=verified,
            domain=self._domain,
            command_adapter=self._command_adapter,
            observation_provider=self._observation_provider,
            runtime_error_provider=self._runtime_error_provider,
        )

    def submit(self, request: str, cfg: ImageProcessorConfig) -> Future[AutoStratStrategy]:
        """Submit the synchronous pipeline to the dedicated background worker."""
        return self._executor.submit(self.build, request, cfg)

    def close(self, *, wait: bool = True) -> None:
        """Release the background worker after outstanding work completes or is cancelled."""
        self._executor.shutdown(wait=wait, cancel_futures=True)

    def __enter__(self) -> StrategyGenerationService:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        del exc_type, exc_value, traceback
        self.close()


__all__ = [
    "StrategyGenerationService",
    "StrategyPipelineRunner",
]
