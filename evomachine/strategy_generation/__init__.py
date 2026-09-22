"""Integrate validated AutoStrat programs with EvoMachine strategy execution."""

from evomachine.strategy_generation.interfaces import (
    CommandAdapter,
    CollectionProvider,
    CommandBuildContext,
    ObservationProvider,
    RuntimeErrorProvider,
)
from evomachine.strategy_generation.microscopy import (
    MicroscopyCommandAdapter,
    MicroscopyCollectionProvider,
    MicroscopyObservationProvider,
    MicroscopyRuntimeErrorProvider,
)
from evomachine.strategy_generation.service import StrategyGenerationService
from evomachine.strategy_generation.strategy import AutoStratStrategy

__all__ = [
    "AutoStratStrategy",
    "CollectionProvider",
    "MicroscopyCollectionProvider",
    "CommandAdapter",
    "CommandBuildContext",
    "MicroscopyCommandAdapter",
    "MicroscopyObservationProvider",
    "MicroscopyRuntimeErrorProvider",
    "ObservationProvider",
    "RuntimeErrorProvider",
    "StrategyGenerationService",
]
