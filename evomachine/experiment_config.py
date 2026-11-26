"""
Shareable experiment configuration files.

An experiment configuration holds the settings describing how an experiment is
run, such as acquisition behaviour, where images are saved, and how software
focus works. It deliberately excludes microscope-specific hardware settings
(serial ports, camera size, objective, stage limits) and per-session state such
as FoV positions, which are selected in the GUI before every initialisation.

Configurations are immutable, validated on construction, and stored as
versioned TOML files that can be shared between users and loaded back through
the same validation used for new configurations.
"""

from __future__ import annotations

import dataclasses
import os
import tomllib
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Annotated, Any

import tomli_w
from delta.utils import CroppingBox
from pydantic import (
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    ValidationError,
    field_validator,
    model_validator,
)

from evomachine.acquisition import FrameAcquisitionSettings
from evomachine.config import TimeConfig, get_logger
from evomachine.config_models import EvoConfig
from evomachine.filemanager import FileNameConfig
from evomachine.frame import FrameMetaData
from evomachine.softwarefocus import (
    SCORING_CONFIG_TYPES,
    FocusSelectionConfig,
    MaximumFocusSelectionConfig,
    ReferenceFocusSelectionConfig,
    SoftwareFocusConfig,
    SteelScoringConfig,
)
from evomachine.types import FilterWheelType, FocusScoringType, FocusSelectionType, LEDType

logger = get_logger(name=__name__)

SCHEMA_VERSION = 1
"Experiment-file schema version written by this code and the newest version it can read."

ALL_OPTIONS_FILE_NAME = "all_options.toml"
"Name of the generated reference file listing every value accepted in experiment files."

MATCHING_CONFIG_TYPES: dict[FocusSelectionType, type[FocusSelectionConfig]] = {
    FocusSelectionType.MAXIMUM: MaximumFocusSelectionConfig,
    FocusSelectionType.REFERENCE_PROFILE: ReferenceFocusSelectionConfig,
}
"Parameter class required by each software-focus matching type."

_EXCLUDED_OPTIONS: dict[type[Enum], frozenset[Enum]] = {
    LEDType: frozenset({LEDType.NO_LED}),
    FilterWheelType: frozenset({FilterWheelType.UNKNOWN}),
}
"Enum members that exist in the code but are not valid choices in an experiment file."

_NONE_NAMES: dict[tuple[type, str], str] = {
    (ReferenceFocusSelectionConfig, "smoothing"): "auto",
}
"Words that stand for None in parameters whose default is not None, because TOML has no null."


class ExperimentConfigError(ValueError):
    """Raised when an experiment configuration cannot be read, parsed, validated, or written."""


# Option names


def allowed_names(enum_type: type[Enum]) -> tuple[str, ...]:
    """
    Return the names accepted in experiment files for one option type.

    Parameters
    ----------
    enum_type
        LEDType, FilterWheelType, FocusScoringType, or FocusSelectionType.

    Returns
    -------
    tuple[str, ...]
        Canonical member names in definition order, excluding placeholder members.
    """
    excluded = _EXCLUDED_OPTIONS.get(enum_type, frozenset())
    return tuple(member.name for member in enum_type if member not in excluded)


def _parse_enum_name(value: object, enum_type: type[Enum], label: str) -> Enum:
    """Return the enum member named by a string, accepting backward-compatible aliases."""
    excluded = _EXCLUDED_OPTIONS.get(enum_type, frozenset())
    if isinstance(value, enum_type) and value not in excluded:
        return value
    member = enum_type.__members__.get(value) if isinstance(value, str) else None
    if member is None or member in excluded:
        raise ValueError(
            f"unknown {label} {value!r}. Allowed: {', '.join(allowed_names(enum_type))} "
            f"(see experiments/{ALL_OPTIONS_FILE_NAME})."
        )
    return member


def _enum_name_type(enum_type: type[Enum], label: str) -> Any:
    """Return an annotated enum type stored by member and written to files by name."""
    return Annotated[
        enum_type,
        BeforeValidator(partial(_parse_enum_name, enum_type=enum_type, label=label)),
        PlainSerializer(lambda member: member.name, return_type=str),
    ]


FilterName = _enum_name_type(FilterWheelType, "filter")


# Output


class ExperimentOutputConfig(EvoConfig):
    """
    Output location and filename pattern for saved images.

    The directory should be a full path. ``~`` is accepted and expands to the
    home directory of the user running EvoMachine. A relative path is accepted
    with a warning because it would be resolved against the working directory
    of the running process.
    """

    model_config = ConfigDict(frozen=True)

    directory: Path
    "Directory where saved images are written."
    filename_pattern: str = FileNameConfig.model_fields["filename_pattern"].default
    "Python format pattern for image filenames; see FileNameConfig for the allowed fields."

    @field_validator("directory", mode="before")
    @classmethod
    def _validate_directory(cls, value: object) -> object:
        if not isinstance(value, Path | str):
            raise ValueError(f"directory must be a path string, received {type(value).__name__}.")
        if str(value).strip() == "":
            raise ValueError("directory must not be empty.")
        return value

    @model_validator(mode="after")
    def _validate_with_file_name_config(self) -> ExperimentOutputConfig:
        """Reuse FileNameConfig validation and warn about relative directories."""
        try:
            self.to_file_name_config()
        except (TypeError, ValueError) as error:
            raise ValueError(_error_message(error)) from error
        if not self.directory.expanduser().is_absolute():
            logger.warning(
                "Experiment output directory '%s' is not a full path; it will be resolved "
                "relative to the working directory of the running process (currently '%s').",
                self.directory,
                Path.cwd(),
            )
        return self

    def to_file_name_config(self, time_config: TimeConfig | None = None) -> FileNameConfig:
        """
        Build a new FileNameConfig for a FileManager.

        Parameters
        ----------
        time_config
            Optional timezone configuration used for timestamps in filenames.
            When omitted, the FileNameConfig default is used.

        Returns
        -------
        FileNameConfig
            New validated configuration with ``~`` expanded in the directory.
        """
        fields: dict[str, Any] = dict(
            directory=self.directory.expanduser(),
            filename_pattern=self.filename_pattern,
        )
        if time_config is not None:
            fields["time_config"] = time_config
        return FileNameConfig(**fields)


# Software focus


def _parse_leds(value: object) -> object:
    """Convert a ``{LED name = brightness}`` table into an immutable tuple of pairs."""
    if isinstance(value, tuple):
        return value
    if not isinstance(value, dict):
        raise ValueError("leds must be a table of LED name = brightness, e.g. { LED_450_NM = 29 }.")
    if not value:
        raise ValueError("leds must name at least one LED.")
    return tuple(
        (_parse_enum_name(name, enum_type=LEDType, label="LED"), brightness)
        for name, brightness in value.items()
    )


def _serialize_leds(leds: tuple[tuple[LEDType, float], ...]) -> dict[str, float]:
    return {led.name: brightness for led, brightness in leds}


class FocusFrameConfig(EvoConfig):
    """
    One image acquired at every Z position of a software-focus scan.

    Every field is required: focus imaging settings have no defaults.
    """

    model_config = ConfigDict(frozen=True)

    leds: Annotated[
        tuple[tuple[LEDType, float], ...],
        BeforeValidator(_parse_leds),
        PlainSerializer(_serialize_leds),
    ]
    "LEDs switched on for the image, with brightness in [0, 100]."
    exposure_ms: float
    "Camera exposure in milliseconds, in [1, 1000]."
    filter: FilterName
    "Filter-wheel position used for the image."

    @model_validator(mode="after")
    def _validate_with_frame_metadata(self) -> FocusFrameConfig:
        """Reuse FrameMetaData validation for brightness and exposure ranges."""
        try:
            self.to_frame_metadata()
        except (TypeError, ValueError) as error:
            raise ValueError(_error_message(error)) from error
        return self

    def to_frame_metadata(self) -> FrameMetaData:
        """
        Build new frame metadata for a software-focus scan.

        Returns
        -------
        FrameMetaData
            New metadata object; it is never shared with other callers.
        """
        return FrameMetaData(
            frame_id=-1,
            leds=dict(self.leds),
            filter_wheel=self.filter,
            exposure=self.exposure_ms,
        )


class CropBoxConfig(EvoConfig):
    """Image region scored by software focus; left/top inclusive, right/bottom exclusive."""

    model_config = ConfigDict(frozen=True)

    xtl: int
    "Left edge in pixels."
    ytl: int
    "Top edge in pixels."
    xbr: int
    "Right edge in pixels."
    ybr: int
    "Bottom edge in pixels."

    def to_cropping_box(self) -> CroppingBox:
        """Return a new DeLTA CroppingBox for this region."""
        return CroppingBox(xtl=self.xtl, ytl=self.ytl, xbr=self.xbr, ybr=self.ybr)


def _parse_typed_pair(
        value: object,
        enum_type: type[Enum],
        config_types: dict[Any, type],
        label: str,
) -> tuple[Any, Any]:
    """
    Convert a ``{type = ..., <parameters>}`` table into a ``(type, parameters)`` pair.

    The parameters are validated by constructing the parameter class required
    by the selected type, so the rules stay defined in one place.
    """
    if isinstance(value, tuple):
        if (
            len(value) != 2
            or value[0] not in config_types
            or not isinstance(value[1], config_types[value[0]])
        ):
            raise ValueError(f"{label} must pair a {label} type with its parameter class.")
        return value
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a table with a 'type' key.")
    parameters = dict(value)
    if "type" not in parameters:
        raise ValueError(
            f"{label} is missing 'type'. Allowed: {', '.join(allowed_names(enum_type))}."
        )
    member = _parse_enum_name(parameters.pop("type"), enum_type=enum_type, label=f"{label} type")
    config_type = config_types[member]
    field_names = [field.name for field in dataclasses.fields(config_type)]
    unknown = sorted(set(parameters) - set(field_names))
    if unknown:
        raise ValueError(
            f"unknown {member.name} parameter(s): {', '.join(unknown)}. "
            f"Allowed: {', '.join(field_names) or 'none'}."
        )
    for name, raw in parameters.items():
        none_name = _NONE_NAMES.get((config_type, name))
        if none_name is not None and raw == none_name:
            parameters[name] = None
        elif isinstance(raw, list):
            parameters[name] = tuple(raw)
    try:
        config = config_type(**parameters)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid {member.name} parameters: {error}") from error
    return member, config


def _serialize_typed_pair(pair: tuple[Enum, Any]) -> dict[str, Any]:
    """Write a ``(type, parameters)`` pair as a ``{type = ..., <parameters>}`` table."""
    member, config = pair
    table: dict[str, Any] = {"type": member.name}
    for field in dataclasses.fields(config):
        value = getattr(config, field.name)
        if value is None:
            none_name = _NONE_NAMES.get((type(config), field.name))
            if none_name is not None:
                table[field.name] = none_name
            elif field.default is not None:
                raise ValueError(
                    f"{type(config).__name__}.{field.name} is None, which TOML cannot store."
                )
            continue
        table[field.name] = list(value) if isinstance(value, tuple) else value
    return table


ScoringPair = Annotated[
    tuple[Any, Any],
    BeforeValidator(
        partial(
            _parse_typed_pair,
            enum_type=FocusScoringType,
            config_types=SCORING_CONFIG_TYPES,
            label="scoring",
        )
    ),
    PlainSerializer(_serialize_typed_pair),
]
MatchingPair = Annotated[
    tuple[Any, Any],
    BeforeValidator(
        partial(
            _parse_typed_pair,
            enum_type=FocusSelectionType,
            config_types=MATCHING_CONFIG_TYPES,
            label="matching",
        )
    ),
    PlainSerializer(_serialize_typed_pair),
]


class ExperimentSoftwareFocusConfig(EvoConfig):
    """
    Software-focus settings: which images to take, how to score them, and how to pick Z.

    ``matching`` selects how the focal Z is chosen from the image scores:
    ``MAXIMUM`` takes the Z with the highest score, which fails when it lies at
    either end of the scan; ``REFERENCE_PROFILE`` compares the measured focus
    curve with a reference curve recorded during initialisation, which is more
    robust but requires that initialisation.
    """

    model_config = ConfigDict(frozen=True)

    frames: tuple[FocusFrameConfig, ...]
    "Images acquired at every Z position; required, no default."
    acquisition: FrameAcquisitionSettings = Field(
        default_factory=FrameAcquisitionSettings.experiment_defaults
    )
    "Acquisition behaviour of focus-scan images."
    scoring: ScoringPair = (FocusScoringType.STEEL, SteelScoringConfig())
    "Image scoring type and its parameters."
    matching: MatchingPair = (FocusSelectionType.REFERENCE_PROFILE, ReferenceFocusSelectionConfig())
    "Z matching type and its parameters."
    crops: tuple[CropBoxConfig, ...] = ()
    "Image regions to score; empty scores the whole image."

    @field_validator("frames")
    @classmethod
    def _validate_frames_not_empty(cls, value: tuple[FocusFrameConfig, ...]) -> object:
        if not value:
            raise ValueError("at least one focus frame is required.")
        return value

    @model_validator(mode="after")
    def _validate_with_software_focus_config(self) -> ExperimentSoftwareFocusConfig:
        """Reuse SoftwareFocusConfig validation, e.g. for crop geometry."""
        try:
            self.to_software_focus_config()
        except (TypeError, ValueError) as error:
            raise ValueError(_error_message(error)) from error
        return self

    def to_software_focus_config(self) -> SoftwareFocusConfig:
        """
        Build a new runtime software-focus configuration.

        Returns
        -------
        SoftwareFocusConfig
            New configuration whose focus frames and crops are never shared
            with other callers.
        """
        return SoftwareFocusConfig(
            focus_frames=tuple(frame.to_frame_metadata() for frame in self.frames),
            acquisition_settings=self.acquisition,
            scoring=self.scoring,
            selection=self.matching,
            cropping_boxes=tuple(crop.to_cropping_box() for crop in self.crops),
        )


class ExperimentFocusConfig(EvoConfig):
    """Focus settings of an experiment."""

    model_config = ConfigDict(frozen=True)

    software: ExperimentSoftwareFocusConfig
    "Software (image-based) focus settings."


# Experiment


class ExperimentConfig(EvoConfig):
    """
    Complete shareable configuration of one experiment.

    Instances are immutable. Use ``updated()`` to obtain a validated copy with
    changed fields, ``load()``/``save()`` for TOML files, and
    ``from_toml()``/``to_toml()`` for TOML text.
    """

    model_config = ConfigDict(frozen=True)

    schema_version: int = SCHEMA_VERSION
    "Version of the experiment-file format; must equal SCHEMA_VERSION once loaded."
    name: str = ""
    "Short human-readable experiment name."
    description: str = ""
    "Free-text description of the experiment."
    acquisition: FrameAcquisitionSettings = Field(
        default_factory=FrameAcquisitionSettings.experiment_defaults
    )
    "Default acquisition behaviour applied to every image acquisition."
    output: ExperimentOutputConfig
    "Where saved images are written and how they are named."
    focus: ExperimentFocusConfig
    "Focus settings."

    @field_validator("schema_version", mode="before")
    @classmethod
    def _validate_schema_version(cls, value: object) -> object:
        if type(value) is not int or value != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION}, received {value!r}.")
        return value

    @classmethod
    def from_toml(cls, text: str, source: str = "<string>") -> ExperimentConfig:
        """
        Parse and validate an experiment configuration from TOML text.

        Parameters
        ----------
        text
            TOML document.
        source
            Name used in error messages, normally the file path.

        Returns
        -------
        ExperimentConfig
            Validated configuration.

        Raises
        ------
        ExperimentConfigError
            If the text is not valid TOML, has an unsupported schema version,
            or does not describe a valid configuration.
        """
        try:
            document = tomllib.loads(text)
        except tomllib.TOMLDecodeError as error:
            raise ExperimentConfigError(f"{source}: not a valid TOML file: {error}.") from error
        _check_schema_version(document, source)
        try:
            return cls.model_validate(document)
        except ValidationError as error:
            raise ExperimentConfigError(
                f"{source}: invalid experiment configuration:\n{_format_validation_error(error)}"
            ) from error
        except (TypeError, ValueError) as error:
            raise ExperimentConfigError(
                f"{source}: invalid experiment configuration: {error}"
            ) from error

    @classmethod
    def load(cls, path: Path | str) -> ExperimentConfig:
        """
        Read and validate an experiment configuration file.

        Parameters
        ----------
        path
            TOML file path; ``~`` is expanded.

        Returns
        -------
        ExperimentConfig
            Validated configuration.

        Raises
        ------
        ExperimentConfigError
            If the file cannot be read or does not contain a valid configuration.
        """
        file_path = Path(path).expanduser()
        try:
            text = file_path.read_text(encoding="utf-8")
        except OSError as error:
            raise ExperimentConfigError(
                f"Cannot read experiment configuration '{file_path}': {error}."
            ) from error
        return cls.from_toml(text, source=str(file_path))

    def to_toml(self) -> str:
        """
        Serialise this configuration as TOML text.

        Returns
        -------
        str
            TOML document that ``from_toml()`` reads back into an equal configuration.
        """
        return tomli_w.dumps(self.model_dump(mode="json"))

    def save(self, path: Path | str) -> Path:
        """
        Write this configuration to a TOML file.

        The file is written to a temporary file first and then moved into
        place, so an interrupted save never leaves a partially written file.

        Parameters
        ----------
        path
            Destination file path; ``~`` is expanded. The parent directory must exist.

        Returns
        -------
        Path
            Path of the written file.

        Raises
        ------
        ExperimentConfigError
            If the file cannot be written.
        """
        file_path = Path(path).expanduser()
        temporary_path = file_path.with_name(f".{file_path.name}.tmp")
        try:
            temporary_path.write_text(self.to_toml(), encoding="utf-8")
            os.replace(temporary_path, file_path)
        except OSError as error:
            temporary_path.unlink(missing_ok=True)
            raise ExperimentConfigError(
                f"Cannot write experiment configuration '{file_path}': {error}."
            ) from error
        return file_path


# Reference list of options


def all_options() -> dict[str, Any]:
    """
    Return every value accepted in experiment files, with parameter defaults.

    This is the content of ``experiments/all_options.toml`` and could later
    populate drop-down lists, for example in a GUI experiment editor.

    Returns
    -------
    dict[str, Any]
        Option names per kind, default parameters per scoring and matching type,
        and the optional parameters that are unset unless given a value.
    """
    parameter_types = {**SCORING_CONFIG_TYPES, **MATCHING_CONFIG_TYPES}
    return {
        "leds": list(allowed_names(LEDType)),
        "filters": list(allowed_names(FilterWheelType)),
        "scoring_types": list(allowed_names(FocusScoringType)),
        "matching_types": list(allowed_names(FocusSelectionType)),
        "scoring_parameters": {
            member.name: _without_type(_serialize_typed_pair((member, config_type())))
            for member, config_type in SCORING_CONFIG_TYPES.items()
        },
        "matching_parameters": {
            member.name: _without_type(_serialize_typed_pair((member, config_type())))
            for member, config_type in MATCHING_CONFIG_TYPES.items()
        },
        "optional_parameters": {
            member.name: names
            for member, config_type in parameter_types.items()
            if (names := [f.name for f in dataclasses.fields(config_type) if f.default is None])
        },
    }


def all_options_toml() -> str:
    """
    Return the text of ``experiments/all_options.toml``.

    Returns
    -------
    str
        Commented TOML listing every accepted option and parameter default.
    """
    header = "\n".join(
        [
            "# Every value accepted in EvoMachine experiment configuration files.",
            "#",
            "# This is a reference list, not an experiment file. Which LEDs and filters are",
            "# installed depends on the microscope. The parameter tables show the default",
            "# value of every scoring and matching parameter. [optional_parameters] lists",
            "# parameters that are unset by default; give them a value to use them.",
            "#",
            "# Future idea: offer these values as drop-down lists, for example in a GUI",
            "# experiment editor, instead of typing names by hand.",
            "#",
            "# Generated from the code; do not edit by hand. Regenerate with:",
            "#   python -c \"from evomachine.experiment_config import all_options_toml; "
            "print(all_options_toml(), end='')\" > experiments/all_options.toml",
            "",
            "",
        ]
    )
    return header + tomli_w.dumps(all_options())


# Helpers


def _without_type(table: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in table.items() if key != "type"}


def _check_schema_version(document: dict[str, Any], source: str) -> None:
    """
    Reject documents whose schema version is missing or unsupported.

    Older versions do not exist yet. When SCHEMA_VERSION is increased, documents
    with an older supported version should be upgraded here before validation.
    """
    version = document.get("schema_version")
    if version is None:
        raise ExperimentConfigError(
            f"{source}: missing 'schema_version'. "
            f"Add 'schema_version = {SCHEMA_VERSION}' at the top of the file."
        )
    if type(version) is not int:
        raise ExperimentConfigError(
            f"{source}: 'schema_version' must be an integer, received {version!r}."
        )
    if version > SCHEMA_VERSION:
        raise ExperimentConfigError(
            f"{source}: file uses experiment schema version {version}, but this version "
            f"of EvoMachine reads versions up to {SCHEMA_VERSION}. "
            "Update EvoMachine to load this file."
        )
    if version < SCHEMA_VERSION:
        raise ExperimentConfigError(
            f"{source}: experiment schema version {version} is not supported; "
            f"the oldest supported version is {SCHEMA_VERSION}."
        )


def _format_validation_error(error: ValidationError) -> str:
    """Return one readable line per validation problem, prefixed by its location in the file."""
    lines = []
    for problem in error.errors():
        location = ".".join(str(part) for part in problem["loc"]) or "(top level)"
        lines.append(f"  - {location}: {_strip_value_error_prefix(problem['msg'])}")
    return "\n".join(lines)


def _error_message(error: Exception) -> str:
    """Return a compact message for a nested config construction error."""
    if isinstance(error, ValidationError):
        return "; ".join(_strip_value_error_prefix(problem["msg"]) for problem in error.errors())
    return str(error)


def _strip_value_error_prefix(message: str) -> str:
    """Remove the 'Value error, ' prefixes pydantic adds to messages from custom validators."""
    prefix = "Value error, "
    while message.startswith(prefix):
        message = message[len(prefix):]
    return message
