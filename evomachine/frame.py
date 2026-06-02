from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np

from evomachine.coordinates import Coordinate
from evomachine.types import BrightnessType, EvoType, ExposureType, FilterWheelType, LEDType, UNKNOWN_FOV_ID


@dataclass
class FrameMetaData:
    """Store frame acquisition settings and metadata associated with one image frame."""

    frame_id: int
    leds: dict[LEDType, BrightnessType] | None
    filter_wheel: FilterWheelType | None
    exposure: ExposureType | None
    dmd_pattern: np.ndarray | None = None
    fov_id: int = UNKNOWN_FOV_ID
    coordinate: Coordinate | None = None
    creation_time: datetime = field(default_factory=datetime.now)
    execution_time: datetime | None = None
    callback_id: int | None = None
    additional_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.frame_id, int) or isinstance(self.frame_id, bool):
            raise TypeError(f"FrameMetaData: frame_id must be int, received {type(self.frame_id)}.")
        if self.leds is not None:
            if not isinstance(self.leds, dict):
                raise TypeError(f"FrameMetaData: leds must be dict[LEDType, BrightnessType] or None, received {type(self.leds)}.")
            for led_type, brightness in self.leds.items():
                if not isinstance(led_type, LEDType):
                    raise TypeError(f"FrameMetaData: LED keys must be LEDType, received {type(led_type)}.")
                if not isinstance(brightness, int | float):
                    raise TypeError(f"FrameMetaData: LED brightness must be numeric, received {type(brightness)}.")
                if not 0 <= float(brightness) <= 100:
                    raise ValueError(f"FrameMetaData: LED brightness must be in [0, 100], received {brightness}.")
        if self.filter_wheel is not None and not isinstance(self.filter_wheel, FilterWheelType):
            raise TypeError(f"FrameMetaData: filter_wheel must be FilterWheelType or None, received {type(self.filter_wheel)}.")
        if self.exposure is not None:
            if not isinstance(self.exposure, int | float):
                raise TypeError(f"FrameMetaData: exposure must be numeric or None, received {type(self.exposure)}.")
            if not 1 <= float(self.exposure) <= 1000:
                raise ValueError(f"FrameMetaData: exposure must be in [1, 1000], received {self.exposure}.")
        if self.dmd_pattern is not None and not isinstance(self.dmd_pattern, np.ndarray):
            raise TypeError(f"FrameMetaData: dmd_pattern must be np.ndarray or None, received {type(self.dmd_pattern)}.")
        if not isinstance(self.fov_id, int) or isinstance(self.fov_id, bool):
            raise TypeError(f"FrameMetaData: fov_id must be int, received {type(self.fov_id)}.")
        if self.coordinate is not None and not isinstance(self.coordinate, Coordinate):
            raise TypeError(f"FrameMetaData: coordinate must be Coordinate or None, received {type(self.coordinate)}.")
        if not isinstance(self.creation_time, datetime):
            raise TypeError(f"FrameMetaData: creation_time must be datetime, received {type(self.creation_time)}.")
        if self.execution_time is not None and not isinstance(self.execution_time, datetime):
            raise TypeError(f"FrameMetaData: execution_time must be datetime or None, received {type(self.execution_time)}.")
        if self.callback_id is not None and (not isinstance(self.callback_id, int) or isinstance(self.callback_id, bool)):
            raise TypeError(f"FrameMetaData: callback_id must be int or None, received {type(self.callback_id)}.")
        if not isinstance(self.additional_metadata, dict):
            raise TypeError(f"FrameMetaData: additional_metadata must be dict[str, Any], received {type(self.additional_metadata)}.")
        if not all(isinstance(key, str) for key in self.additional_metadata):
            raise TypeError("FrameMetaData: additional_metadata keys must be str.")

    def to_metadata_dict(self) -> dict[str, Any]:
        metadata = {
            "frame_id": self.frame_id,
            "callback_id": self.callback_id,
            "leds": None if self.leds is None else {
                led_type.name: {"value": led_type.value, "brightness": float(brightness)}
                for led_type, brightness in self.leds.items()
            },
            "filter_wheel": self._enum_to_metadata(self.filter_wheel),
            "exposure": self.exposure,
            "dmd_pattern": self._dmd_pattern_to_metadata(),
            "fov_id": self.fov_id,
            "coordinate": None if self.coordinate is None else self.coordinate.to_dict(),
            "creation_time": self.creation_time.isoformat(),
            "execution_time": None if self.execution_time is None else self.execution_time.isoformat(),
            "additional_metadata": self.additional_metadata,
        }
        try:
            json.dumps(metadata)
        except TypeError as error:
            raise TypeError("FrameMetaData.to_metadata_dict: metadata must be JSON serializable.") from error
        return metadata

    def __str__(self) -> str:
        values = {
            "frame_id": self.frame_id,
            "leds": self.leds,
            "filter_wheel": self.filter_wheel,
            "exposure": self.exposure,
            "dmd_pattern": self._dmd_pattern_to_metadata(),
            "fov_id": self.fov_id,
            "coordinate": self.coordinate,
            "creation_time": self.creation_time,
            "execution_time": self.execution_time,
            "callback_id": self.callback_id,
            "additional_metadata": self.additional_metadata,
        }
        lines = ["FrameMetaData"]
        for index, (key, value) in enumerate(values.items()):
            lines.append(f"{' └─ ' if index == len(values) - 1 else ' ├─ '}{key}: {value}")
        return "\n".join(lines)

    @staticmethod
    def _enum_to_metadata(value: EvoType | None) -> dict[str, Any] | None:
        if value is None:
            return None
        return {"name": value.name, "value": value.value}

    def _dmd_pattern_to_metadata(self) -> dict[str, Any] | None:
        if self.dmd_pattern is None:
            return None
        return {"present": True, "shape": list(self.dmd_pattern.shape), "dtype": str(self.dmd_pattern.dtype)}


class FrameMetaDataFactory:
    """Factory for common FrameMetaData instances with generated frame IDs."""

    _next_frame_id = 0

    @classmethod
    def reset_counter(cls, start: int = 0) -> None:
        if not isinstance(start, int) or isinstance(start, bool):
            raise TypeError(f"FrameMetaDataFactory.reset_counter: start must be int, received {type(start)}.")
        cls._next_frame_id = start

    @classmethod
    def default(
            cls,
            leds: dict[LEDType, BrightnessType] | None = None,
            filter_wheel: FilterWheelType | None = None,
            exposure: ExposureType | None = None,
            dmd_pattern: np.ndarray | None = None,
            fov_id: int = UNKNOWN_FOV_ID,
            coordinate: Coordinate | None = None,
            frame_id: int | None = None,
            callback_id: int | None = None,
            additional_metadata: dict[str, Any] | None = None,
    ) -> FrameMetaData:
        if frame_id is None:
            frame_id = cls._next_frame_id
            cls._next_frame_id += 1
        return FrameMetaData(
            frame_id=frame_id,
            leds=leds,
            filter_wheel=filter_wheel,
            exposure=exposure,
            dmd_pattern=dmd_pattern,
            fov_id=fov_id,
            coordinate=coordinate,
            callback_id=callback_id,
            additional_metadata={} if additional_metadata is None else additional_metadata,
        )


@dataclass
class Frame:
    """Store one or more acquired image arrays with their frame metadata."""

    frame_metadata: list[FrameMetaData]
    array: np.ndarray
    saved_paths: list[Path | None] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.frame_metadata, list):
            raise TypeError(f"Frame: frame_metadata must be list[FrameMetaData], received {type(self.frame_metadata)}.")
        if not self.frame_metadata:
            raise ValueError("Frame: frame_metadata must not be empty.")
        if not all(isinstance(metadata, FrameMetaData) for metadata in self.frame_metadata):
            raise TypeError("Frame: every frame_metadata entry must be FrameMetaData.")
        if not isinstance(self.array, np.ndarray):
            raise TypeError(f"Frame: array must be np.ndarray, received {type(self.array)}.")
        if self.array.ndim != 3:
            raise ValueError(f"Frame: array must be 3D with shape (n_frames, height, width), received {self.array.shape}.")
        if self.array.shape[0] != len(self.frame_metadata):
            raise ValueError(
                f"Frame: leading array dimension {self.array.shape[0]} must match {len(self.frame_metadata)} metadata entries."
            )
        if self.saved_paths is None:
            self.saved_paths = [None] * len(self.frame_metadata)
        if not isinstance(self.saved_paths, list):
            raise TypeError(f"Frame: saved_paths must be list[Path | None], received {type(self.saved_paths)}.")
        if len(self.saved_paths) != len(self.frame_metadata):
            raise ValueError("Frame: saved_paths length must match frame_metadata length.")
        if not all(path is None or isinstance(path, Path) for path in self.saved_paths):
            raise TypeError("Frame: saved_paths entries must be Path or None.")
