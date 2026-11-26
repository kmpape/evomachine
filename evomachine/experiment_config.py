"""
Shareable experiment configuration files.

An experiment configuration holds the settings describing how an experiment is
run, such as acquisition behaviour and where images are saved. It deliberately
excludes microscope-specific hardware settings (serial ports, camera size,
objective, stage limits) and per-session state such as FoV positions, which are
selected in the GUI before every initialisation.

Configurations are immutable, validated on construction, and stored as
versioned TOML files that can be shared between users and loaded back through
the same validation used for new configurations.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

import tomli_w
from pydantic import ConfigDict, Field, ValidationError, field_validator, model_validator

from evomachine.acquisition import FrameAcquisitionSettings
from evomachine.config import TimeConfig, get_logger
from evomachine.config_models import EvoConfig
from evomachine.filemanager import FileNameConfig

logger = get_logger(name=__name__)

SCHEMA_VERSION = 1
"Experiment-file schema version written by this code and the newest version it can read."


class ExperimentConfigError(ValueError):
    """Raised when an experiment configuration cannot be read, parsed, validated, or written."""


class OutputConfig(EvoConfig):
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
    def _validate_with_file_name_config(self) -> OutputConfig:
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
    acquisition: FrameAcquisitionSettings = Field(default_factory=FrameAcquisitionSettings)
    "Default acquisition behaviour applied to every image acquisition."
    output: OutputConfig
    "Where saved images are written and how they are named."

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
