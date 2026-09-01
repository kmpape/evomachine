"""Paths to domain packs owned and deployed by EvoMachine."""

from pathlib import Path
from typing import Final

MICROSCOPY_DOMAIN_PACK_PATH: Final[Path] = Path(__file__).with_name("microscopy")

__all__ = ["MICROSCOPY_DOMAIN_PACK_PATH"]
