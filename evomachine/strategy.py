from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import inspect
import importlib.util
from pathlib import Path
from types import ModuleType

from delta.rt import PositionRT

from evomachine.commands import AutomatonCommand, CommandFactory
from evomachine.config import get_logger
from evomachine.coordinates import Coordinate
from evomachine.frame import FrameMetaDataFactory
from evomachine.image_processing_config import ImageProcessorConfig
from evomachine.peripherals.camera import CameraSystemConfig
from evomachine.peripherals.dmd import Dmd
from evomachine.types import LEDType

logger = get_logger(name=__name__)


@dataclass(frozen=True)
class StrategyDefinition:
    """
    Describe a discoverable strategy class.

    Parameters
    ----------
    name
        Class name of the strategy.
    file_path
        Python file that defines the strategy class.

    Returns
    -------
    StrategyDefinition
        Immutable strategy descriptor used for discovery and creation.
    """

    name: str
    file_path: Path


class AbstractStrategy(ABC):
    """
    Base class for strategies executed by the automaton.

    Parameters
    ----------
    cfg
        Image processor configuration used by command validation.

    Returns
    -------
    AbstractStrategy
        Strategy instance that can be initialised by Automaton.
    """

    def __init__(self, cfg: ImageProcessorConfig):
        self.callback_counter: int = 0
        "Incremented after each callback. First callback is 0."
        self.cfg: ImageProcessorConfig = cfg
        "Image processor configuration object."
        self.fovs: dict[int, Coordinate] = {}
        "Dictionary indexed by FoV ID with FoV coordinates."
        self.region_of_interests: dict[int, list[int]] = {}
        "Dictionary indexed by FoV ID with ROI IDs."
        self.fov_processors: dict[int, PositionRT] = {}
        "Processors for FoV data, ROI boxes, and cell lineages keyed by FoV ID. Treat as read-only."
        self.path_to_save: Path | None = None
        "Optional path to save images."
        self.command_factory: CommandFactory = CommandFactory(cfg=cfg)
        "Factory object used to create AutomatonCommand objects."
        self.config_camera: CameraSystemConfig | None = None
        "Camera configuration object injected during initialise()."
        self.dmd: Dmd | None = None
        "DMD object injected during initialise()."

    def __getstate__(self) -> dict:
        """
        Return serialisable strategy state.

        Parameters
        ----------
        None

        Returns
        -------
        dict
            Strategy state without runtime DMD and FoV processors.
        """
        state = self.__dict__.copy()
        state.pop("dmd", None)
        state.pop("fov_processors", None)
        return state

    def __setstate__(self, state: dict) -> None:
        """
        Restore strategy state after deserialisation.

        Parameters
        ----------
        state
            Serialised strategy state.

        Returns
        -------
        None
        """
        state["dmd"] = None
        state["fov_processors"] = {}
        self.__dict__.update(state)

    @abstractmethod
    def _initialise(self) -> list[AutomatonCommand]:
        """
        Return commands for the first strategy step.

        Parameters
        ----------
        None

        Returns
        -------
        list[AutomatonCommand]
            Commands to execute after initialisation.
        """
        raise NotImplementedError

    def initialise(
            self,
            fovs: dict[int, Coordinate],
            region_of_interests: dict[int, list[int]],
            config_camera: CameraSystemConfig | None,
            fov_processors: dict[int, PositionRT],
            dmd: Dmd,
    ) -> list[AutomatonCommand]:
        """
        Reset runtime state and initialise the strategy.

        Parameters
        ----------
        fovs
            Mapping from FoV ID to Coordinate.
        region_of_interests
            Mapping from FoV ID to ROI IDs.
        config_camera
            Camera configuration object, when available.
        fov_processors
            FoV processors keyed by FoV ID.
        dmd
            DMD object available for pattern construction.

        Returns
        -------
        list[AutomatonCommand]
            Valid commands produced by the strategy.
        """
        if not isinstance(fovs, dict):
            raise TypeError(f"AbstractStrategy.initialise: fovs must be dict, received {type(fovs)}.")
        if not isinstance(region_of_interests, dict):
            raise TypeError(
                "AbstractStrategy.initialise: region_of_interests must be dict[int, list[int]]."
            )
        if not isinstance(fov_processors, dict):
            raise TypeError(
                f"AbstractStrategy.initialise: fov_processors must be dict, received {type(fov_processors)}."
            )
        if not isinstance(dmd, Dmd):
            raise TypeError(f"AbstractStrategy.initialise: dmd must be Dmd, received {type(dmd)}.")
        self.callback_counter = 0
        self.fovs = fovs
        self.region_of_interests = region_of_interests
        self.command_factory.update_region_of_interests(region_of_interests=region_of_interests)
        self.config_camera = config_camera
        self.fov_processors = fov_processors
        self.dmd = dmd
        command_list = self._initialise()
        if not self.is_valid_command_list(command_list):
            raise RuntimeError(f"AbstractStrategy.initialise: invalid command list ({command_list}).")
        return command_list

    @abstractmethod
    def _callback(
            self,
            fov_id: int,
            data: list[AutomatonCommand],
            errors: list[Exception],
    ) -> list[AutomatonCommand]:
        """
        Return commands for a strategy callback.

        Parameters
        ----------
        fov_id
            Current FoV ID.
        data
            Commands executed at the previous step.
        errors
            Errors raised while executing the previous step.

        Returns
        -------
        list[AutomatonCommand]
            Commands for the next step.
        """
        raise NotImplementedError

    def callback(
            self,
            fov_id: int,
            data: list[AutomatonCommand],
            errors: list[Exception],
    ) -> list[AutomatonCommand]:
        """
        Validate and return commands from the strategy callback.

        Parameters
        ----------
        fov_id
            Current FoV ID.
        data
            Commands executed at the previous step.
        errors
            Errors raised while executing the previous step.

        Returns
        -------
        list[AutomatonCommand]
            Valid commands for the next step.
        """
        command_list = self._callback(fov_id=fov_id, data=data, errors=errors)
        if not self.is_valid_command_list(command_list):
            raise RuntimeError(f"AbstractStrategy.callback: invalid command list ({command_list}).")
        self.callback_counter += 1
        return command_list

    @abstractmethod
    def finalise(self) -> list[AutomatonCommand]:
        """
        Return commands for strategy finalisation.

        Parameters
        ----------
        None

        Returns
        -------
        list[AutomatonCommand]
            Final commands to execute.
        """
        raise NotImplementedError

    def name(self) -> str:
        """
        Return the strategy class name.

        Parameters
        ----------
        None

        Returns
        -------
        str
            Strategy class name.
        """
        return self.__class__.__name__

    @staticmethod
    def is_valid_command_list(command_list: list[AutomatonCommand]) -> bool:
        """
        Return whether all entries are AutomatonCommand objects.

        Parameters
        ----------
        command_list
            Candidate command list.

        Returns
        -------
        bool
            True when every entry is an AutomatonCommand.
        """
        return isinstance(command_list, list) and all(isinstance(cmd, AutomatonCommand) for cmd in command_list)


class NoStrategy(AbstractStrategy):
    """Strategy that never schedules commands."""

    def _callback(
            self,
            fov_id: int,
            data: list[AutomatonCommand],
            errors: list[Exception],
    ) -> list[AutomatonCommand]:
        """Return no callback commands."""
        return []

    def _initialise(self) -> list[AutomatonCommand]:
        """Return no initial commands."""
        return []

    def finalise(self) -> list[AutomatonCommand]:
        """Return no final commands."""
        return []


class BasicStrategy(AbstractStrategy):
    """Simple built-in imaging strategy for tests and minimal acquisitions."""

    def __init__(self, cfg: ImageProcessorConfig, save_path: str):
        """
        Initialise the basic strategy.

        Parameters
        ----------
        cfg
            Image processor configuration.
        save_path
            Directory path where images should be saved.

        Returns
        -------
        None
        """
        super().__init__(cfg=cfg)
        self.path_to_save = Path(save_path)
        self.imaging_interval: int = 60 * 3
        self.imaging_channels: list[LEDType] = [LEDType.LED_565_NM]
        self.exposure_time: int = 100
        self.num_fovs: int | None = None

    def _commands_for_fovs(self, segment: bool) -> list[AutomatonCommand]:
        """
        Build a move/image/wait cycle for every FoV.

        Parameters
        ----------
        segment
            Whether image commands should request segmentation.

        Returns
        -------
        list[AutomatonCommand]
            Commands for one imaging cycle.
        """
        self.num_fovs = len(self.fovs)
        commands: list[AutomatonCommand] = []
        for fov_id in self.fovs:
            commands.append(self.command_factory.command_move(fov_id=fov_id))
            channels = self.imaging_channels
            if segment:
                channels = list(dict.fromkeys([*self.imaging_channels, *self.cfg.channels_seg]))
            frame_metadata = [
                FrameMetaDataFactory.default(leds={channel: 10}, exposure=self.exposure_time, fov_id=fov_id)
                for channel in channels
            ]
            commands.append(
                self.command_factory.command_image(
                    frame_metadata=frame_metadata,
                    segment=segment,
                    save=True,
                )
            )
            commands.append(
                self.command_factory.command_wait(
                    duration=self.imaging_interval / max(self.num_fovs, 1),
                    set_live_mode=False,
                    channel=LEDType.LED_450_NM,
                    brightness=10,
                )
            )
        return commands

    def _initialise(self) -> list[AutomatonCommand]:
        """Return segmented imaging commands for the first cycle."""
        logger.info("Initialised BasicStrategy for %s FoVs.", len(self.fovs))
        return self._commands_for_fovs(segment=True)

    def _callback(
            self,
            fov_id: int,
            data: list[AutomatonCommand],
            errors: list[Exception],
    ) -> list[AutomatonCommand]:
        """Return non-segmenting imaging commands for subsequent cycles."""
        logger.info("BasicStrategy callback at FoV %s with %s errors.", fov_id, len(errors))
        return self._commands_for_fovs(segment=False)

    def finalise(self) -> list[AutomatonCommand]:
        """Enable live mode as the final command."""
        logger.info("Finalising BasicStrategy.")
        return [self.command_factory.command_live_mode(status=True)]


# TODO(CODEX): Move this _default_strategies_folder to class definition as class attribute
def _default_strategies_folder() -> Path:
    """
    Return the default repository strategies folder.

    Parameters
    ----------
    None

    Returns
    -------
    Path
        Absolute path to the default strategies folder.
    """
    return Path(__file__).resolve().parents[1] / "strategies"


def _load_strategy_module(file_path: Path) -> ModuleType:
    """
    Load a strategy module from a Python file.

    Parameters
    ----------
    file_path
        Python file to load.

    Returns
    -------
    ModuleType
        Loaded Python module.
    """
    module_name = f"evomachine_user_strategy_{file_path.stem}_{abs(hash(file_path))}"
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load strategy file {file_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _strategy_classes_from_module(module: ModuleType) -> list[tuple[str, type[AbstractStrategy]]]:
    """
    Return concrete AbstractStrategy subclasses defined by a module.

    Parameters
    ----------
    module
        Loaded strategy module.

    Returns
    -------
    list[tuple[str, type[AbstractStrategy]]]
        Strategy class names and class objects.
    """
    strategy_classes: list[tuple[str, type[AbstractStrategy]]] = []
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if obj.__module__ != module.__name__:
            continue
        if issubclass(obj, AbstractStrategy) and obj not in (AbstractStrategy, NoStrategy):
            if not inspect.isabstract(obj):
                strategy_classes.append((name, obj))
    return strategy_classes


def list_strategy_definitions(strategy_folder: str | Path | None = None) -> list[StrategyDefinition]:
    """
    List strategy names and files discoverable in a folder.

    Parameters
    ----------
    strategy_folder
        Folder to scan. If None, the repository default strategies folder is used.

    Returns
    -------
    list[StrategyDefinition]
        Strategy descriptors sorted by file path and name.
    """
    folder = _default_strategies_folder() if strategy_folder is None else Path(strategy_folder)
    folder = folder.resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"Strategy folder does not exist: {folder}.")
    definitions: list[StrategyDefinition] = []
    seen: set[tuple[str, Path]] = set()
    for file_path in sorted(folder.glob("*.py")):
        if file_path.name == "__init__.py":
            continue
        module = _load_strategy_module(file_path=file_path)
        for name, _strategy_class in _strategy_classes_from_module(module=module):
            definition = StrategyDefinition(name=name, file_path=file_path.resolve())
            pair = (definition.name, definition.file_path)
            if pair in seen:
                raise RuntimeError(f"Duplicate strategy definition found: {definition.name} in {definition.file_path}.")
            seen.add(pair)
            definitions.append(definition)
    return sorted(definitions, key=lambda item: (str(item.file_path), item.name))


def create_strategy_from_definition(name: str, file_path: str | Path, cfg: ImageProcessorConfig) -> AbstractStrategy:
    """
    Create a strategy object from a discovered name/file pair.

    Parameters
    ----------
    name
        Strategy class name.
    file_path
        Python file containing the strategy class.
    cfg
        Image processor configuration passed to the strategy constructor.

    Returns
    -------
    AbstractStrategy
        Instantiated strategy object.
    """
    file_path = Path(file_path).resolve()
    module = _load_strategy_module(file_path=file_path)
    matches = [
        strategy_class
        for strategy_name, strategy_class in _strategy_classes_from_module(module=module)
        if strategy_name == name
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one strategy named {name} in {file_path}, found {len(matches)}.")
    strategy = matches[0](cfg=cfg)
    if not isinstance(strategy, AbstractStrategy):
        raise TypeError(f"Strategy {name} from {file_path} did not create an AbstractStrategy.")
    return strategy


def get_all_strategies() -> list[tuple[type[AbstractStrategy], str]]:
    """
    Return discoverable strategy classes using the legacy tuple format.

    Parameters
    ----------
    None

    Returns
    -------
    list[tuple[type[AbstractStrategy], str]]
        Strategy classes and class names.
    """
    strategies: list[tuple[type[AbstractStrategy], str]] = [(BasicStrategy, "BasicStrategy")]
    for definition in list_strategy_definitions():
        module = _load_strategy_module(file_path=definition.file_path)
        for name, strategy_class in _strategy_classes_from_module(module=module):
            if name == definition.name:
                strategies.append((strategy_class, name))
    return strategies
