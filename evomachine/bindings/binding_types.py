"""Shared binding identifiers used by evomachine device factories."""

from __future__ import annotations

from enum import auto

from evomachine.types import EvoType


class BindingType(EvoType):
    """Binding identifiers supported by evomachine peripheral factories."""

    VIRTUAL = auto()
    """Supported by peripheral controllers, stages, filter wheels, LED sources, DMDs, cameras, autofocus, and photodiodes."""

    ASI_TIGER = auto()
    """Supported by peripheral controllers, stages, filter wheels, LED sources, and autofocus."""

    SYNCBOARD = auto()
    """Supported by peripheral controllers, LED sources, and photodiodes."""

    KWR103 = auto()
    """Supported by peripheral controllers and LED sources."""

    EM_DMD_WINDOW = auto()
    """Supported by DMDs."""

    PYGAME = auto()
    """Supported by DMDs."""

    MMC = auto()
    """Supported by Micro-Manager cameras."""

    PVCAM = auto()
    """Supported by PVCAM cameras."""
