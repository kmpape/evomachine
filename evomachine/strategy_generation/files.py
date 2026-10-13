"""Persistent user-authored AutoStrat strategy files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import tempfile


STRATEGY_EXTENSION = ".strat"
_VALID_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _-]{0,79}$")


@dataclass(frozen=True, slots=True)
class StoredStrategy:
    """One named AutoStrat source file."""

    name: str
    path: Path


class StrategyFileStore:
    """List, load, and atomically save `.strat` files under one directory."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)

    @staticmethod
    def normalise_name(name: str) -> str:
        if not isinstance(name, str):
            raise TypeError("Strategy name must be text.")
        candidate = name.strip()
        if candidate.lower().endswith(STRATEGY_EXTENSION):
            candidate = candidate[:-len(STRATEGY_EXTENSION)].rstrip()
        if not _VALID_NAME.fullmatch(candidate):
            raise ValueError(
                "Strategy name must be 1-80 characters and contain only letters, "
                "numbers, spaces, underscores, or hyphens."
            )
        return candidate

    def path_for(self, name: str) -> Path:
        normalised = self.normalise_name(name)
        path = self.directory / f"{normalised}{STRATEGY_EXTENSION}"
        if path.resolve().parent != self.directory.resolve():
            raise ValueError("Strategy file must remain inside the AutoStrat directory.")
        return path

    def list(self) -> list[StoredStrategy]:
        if not self.directory.exists():
            return []
        strategies = []
        for path in sorted(
                self.directory.glob(f"*{STRATEGY_EXTENSION}"),
                key=lambda item: item.name.lower(),
        ):
            if not path.is_file() or path.is_symlink():
                continue
            try:
                name = self.normalise_name(path.stem)
            except ValueError:
                continue
            strategies.append(StoredStrategy(name=name, path=path))
        return strategies

    def load(self, name: str) -> tuple[StoredStrategy, str]:
        path = self.path_for(name)
        if not path.is_file():
            raise FileNotFoundError(f"AutoStrat strategy does not exist: {path.name}")
        return StoredStrategy(name=path.stem, path=path), path.read_text(encoding="utf-8")

    def save(self, name: str, source: str, *, overwrite: bool = False) -> StoredStrategy:
        if not isinstance(source, str) or not source.strip():
            raise ValueError("Strategy source must be non-empty text.")
        path = self.path_for(name)
        if path.exists() and not overwrite:
            raise FileExistsError(f"AutoStrat strategy already exists: {path.name}")
        self.directory.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    newline="\n",
                    dir=self.directory,
                    prefix=f".{path.name}.",
                    suffix=".tmp",
                    delete=False,
            ) as temporary:
                temporary.write(source)
                temporary_path = Path(temporary.name)
            temporary_path.replace(path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        return StoredStrategy(name=path.stem, path=path)
