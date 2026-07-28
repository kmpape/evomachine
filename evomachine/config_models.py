from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class EvoConfig(BaseModel):
    """Base model for EvoMachine configuration objects."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    def copy(self):
        """Return a shallow same-type copy for compatibility with old dataclass configs."""
        return self.model_copy()

    def updated(self, **kwargs: Any):
        """Return a validated copy with selected fields updated."""
        unknown_keys = [key for key in kwargs if key not in type(self).model_fields]
        if unknown_keys:
            raise ValueError(f"{type(self).__name__}.updated: unknown fields {unknown_keys}.")
        values = dict(self.__dict__)
        values.update(kwargs)
        return type(self)(**values)

    def update_from_mapping(self, updates: dict[str, Any]):
        """Return a validated copy updated from a mapping."""
        if not isinstance(updates, dict):
            raise TypeError(f"{type(self).__name__}.update_from_mapping: updates must be dict.")
        return self.updated(**updates)

    def __str__(self) -> str:
        lines = [type(self).__name__]
        for index, (key, value) in enumerate(self.__dict__.items()):
            lines.append(f"{' └─ ' if index == len(self.__dict__) - 1 else ' ├─ '}{key}: {value}")
        return "\n".join(lines)
