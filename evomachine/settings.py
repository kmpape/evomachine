"""
User-adjustable settings that must not live inside the code checkout.

Everything evomachine writes at runtime (acquired images, logs, strategies saved from the
GUI) goes under one data directory. The directory is resolved once at import time from two
optional TOML files, later entries winning:

1. The machine file, ``/etc/evomachine/config.toml``: defaults for every user of one
   microscope computer, for example where that machine's large disk is mounted.
2. The user file, ``~/.config/evomachine/config.toml``: one user's own choice.

If neither file sets a value, the data directory is the code checkout itself, which keeps
the historical layout (``images/``, ``logs/`` and ``strategies/autostrat/`` next to the
package). Machines with a read-only shared installation must therefore ship a machine file.

Both files share the same schema::

    [paths]
    data_dir = "/scratch/$USER/microscope-data"

Values may use ``~`` and ``$VARIABLES``; they are expanded for the user running evomachine,
which is what makes one machine file work for everyone.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tomllib

MACHINE_CONFIG_FILE = Path("/etc/evomachine/config.toml")
USER_CONFIG_FILE = (
    Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "evomachine" / "config.toml"
)
# The checkout root. Computed here rather than imported from evomachine.config, which
# imports this module.
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1]

_KNOWN_TABLES = {"paths"}
_KNOWN_PATH_KEYS = {"data_dir"}


@dataclass(frozen=True)
class EvoSettings:
    """
    Resolved settings.

    Parameters
    ----------
    data_dir
        Root directory for everything evomachine writes at runtime.
    source_files
        Configuration files that were found and read, in the order applied.

    Returns
    -------
    EvoSettings
        Immutable settings.
    """

    data_dir: Path
    source_files: tuple[Path, ...] = ()

    @property
    def images_dir(self) -> Path:
        """Directory acquired frames are written to."""
        return self.data_dir / "images"

    @property
    def log_dir(self) -> Path:
        """Directory log files are written to."""
        return self.data_dir / "logs"

    @property
    def autostrat_strategies_dir(self) -> Path:
        """Directory AutoStrat strategies saved from the GUI are written to."""
        return self.data_dir / "strategies" / "autostrat"


def _expand_path(value: str) -> Path:
    """
    Expand ``~`` and environment variables in a configured path.

    Parameters
    ----------
    value
        Path text as written in a configuration file.

    Returns
    -------
    Path
        Absolute path for the current user.
    """
    return Path(os.path.expandvars(value)).expanduser()


def _read_config_file(path: Path) -> dict[str, object]:
    """
    Read one configuration file and reject keys this version does not understand.

    Parameters
    ----------
    path
        File to read. A missing file yields an empty mapping.

    Returns
    -------
    dict[str, object]
        The ``[paths]`` table with values validated.

    Raises
    ------
    ValueError
        If the file is not valid TOML, contains unknown tables or keys, or has a
        non-string path value.
    """
    if not path.is_file():
        return {}
    try:
        with path.open("rb") as handle:
            document = tomllib.load(handle)
    except tomllib.TOMLDecodeError as error:
        raise ValueError(f"Invalid settings file {path}: {error}") from error

    unknown_tables = set(document) - _KNOWN_TABLES
    if unknown_tables:
        raise ValueError(f"Unknown section(s) {sorted(unknown_tables)} in settings file {path}.")
    paths = document.get("paths", {})
    if not isinstance(paths, dict):
        raise ValueError(f"[paths] must be a table in settings file {path}.")
    unknown_keys = set(paths) - _KNOWN_PATH_KEYS
    if unknown_keys:
        raise ValueError(f"Unknown key(s) {sorted(unknown_keys)} under [paths] in settings file {path}.")
    for key, value in paths.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"[paths].{key} must be a non-empty string in settings file {path}.")
    return paths


def load_settings(
        machine_file: Path = MACHINE_CONFIG_FILE,
        user_file: Path = USER_CONFIG_FILE,
        default_data_dir: Path = DEFAULT_DATA_DIR,
) -> EvoSettings:
    """
    Resolve settings from the machine file, then the user file, then built-in defaults.

    Parameters
    ----------
    machine_file
        Per-machine defaults, normally ``/etc/evomachine/config.toml``.
    user_file
        Per-user overrides, normally ``~/.config/evomachine/config.toml``.
    default_data_dir
        Data directory used when neither file sets one.

    Returns
    -------
    EvoSettings
        Resolved settings; ``source_files`` lists the files that existed.
    """
    data_dir = default_data_dir
    source_files: list[Path] = []
    for path in (machine_file, user_file):
        paths = _read_config_file(path)
        if not paths and not path.is_file():
            continue
        source_files.append(path)
        if "data_dir" in paths:
            data_dir = _expand_path(str(paths["data_dir"]))
    return EvoSettings(data_dir=data_dir, source_files=tuple(source_files))


SETTINGS = load_settings()
