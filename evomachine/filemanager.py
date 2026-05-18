from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np
import skimage.io
import tifffile

from evomachine.config_types import ConfigFrame, FileNameConfig
from evomachine.coordinates import Coordinate
from evomachine.types import EvoType, FilterWheelType, LEDType

# TODO(CODEX):
# - Extend the filemanager with functions for easy loading of images, i.e. if a user wants to load images after an experiment, there should be some helpers for this
# - Maybe also save the filemanager config into the corresponding folder (each time the config it is changed, with a hint saying from which filename on the new config holds)

class FileManager:
    """Manager for frame filenames, save directories, and frame image metadata."""

    TIFF_EXTENSIONS = {"tif", "tiff"}

    def __init__(self, config: FileNameConfig):
        """
        Initialise a file manager from a file name configuration.

        Parameters
        ----------
        config
            FileNameConfig defining the output directory, filename pattern, and
            default extension.

        Returns
        -------
        None
        """
        if not isinstance(config, FileNameConfig):
            raise TypeError(f"FileManager.__init__: config must be FileNameConfig, received {type(config)}.")
        self.config: FileNameConfig = config
        self._ensure_directory()

    def update_config(self, config: FileNameConfig | None = None, **updates: Any) -> None:
        """
        Replace or update the active file name configuration.

        Parameters
        ----------
        config
            Optional replacement FileNameConfig. If None, updates are applied to
            the current config.
        **updates
            FileNameConfig field values to update when config is None.

        Returns
        -------
        None
        """
        if config is not None and updates:
            raise ValueError("FileManager.update_config: provide config or updates, not both.")
        if config is not None:
            if not isinstance(config, FileNameConfig):
                raise TypeError(f"FileManager.update_config: config must be FileNameConfig, received {type(config)}.")
            self.config = config
        else:
            self.config = self.config.updated(**updates)
        self._ensure_directory()

    def _ensure_directory(self) -> None:
        """
        Create or validate the configured output directory.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if self.config.create_directory:
            self.config.directory.mkdir(parents=True, exist_ok=True)
        elif not self.config.directory.exists():
            raise FileNotFoundError(f"FileManager: directory does not exist: {self.config.directory}.")
        if not self.config.directory.is_dir():
            raise NotADirectoryError(f"FileManager: output path is not a directory: {self.config.directory}.")

    def get_filename(
            self,
            config_frame: ConfigFrame,
            filename_pattern: str | None = None,
            suffix: str | None = None,
            **values: Any,
    ) -> Path:
        """
        Return an output path generated from the active filename pattern.

        Parameters
        ----------
        config_frame
            ConfigFrame supplying default filename variables.
        filename_pattern
            Optional pattern overriding FileNameConfig.filename_pattern.
        suffix
            Optional suffix inserted through the {suffix} pattern variable.
        **values
            Additional or overriding format variables for the filename pattern.

        Returns
        -------
        Path
            Full output path under the configured directory.
        """
        if not isinstance(config_frame, ConfigFrame):
            raise TypeError(
                f"FileManager.get_filename: config_frame must be ConfigFrame, received {type(config_frame)}."
            )
        if filename_pattern is not None and not isinstance(filename_pattern, str):
            raise TypeError(
                f"FileManager.get_filename: filename_pattern must be str or None, received {type(filename_pattern)}."
            )
        if suffix is not None and not isinstance(suffix, str):
            raise TypeError(f"FileManager.get_filename: suffix must be str or None, received {type(suffix)}.")
        pattern = self.config.filename_pattern if filename_pattern is None else filename_pattern
        format_values = self._format_values(config_frame=config_frame, suffix=suffix)
        format_values.update(values)
        filename = pattern.format(**format_values)
        filename_path = Path(filename)
        if filename_path.is_absolute():
            raise ValueError("FileManager.get_filename: filename_pattern must produce a relative path.")
        return self.config.directory / filename_path

    def save_frame(
            self,
            frame: np.ndarray,
            config_frame: ConfigFrame,
            filename_pattern: str | None = None,
            suffix: str | None = None,
            **values: Any,
    ) -> Path:
        """
        Save one frame image and return the output path.

        Parameters
        ----------
        frame
            Image array to save.
        config_frame
            ConfigFrame to include in TIFF metadata and filename variables.
        filename_pattern
            Optional pattern overriding FileNameConfig.filename_pattern.
        suffix
            Optional suffix inserted through the {suffix} pattern variable.
        **values
            Additional or overriding format variables for the filename pattern.

        Returns
        -------
        Path
            Path where the frame was saved.
        """
        if not isinstance(frame, np.ndarray):
            raise TypeError(f"FileManager.save_frame: frame must be np.ndarray, received {type(frame)}.")
        if config_frame.execution_time is None:
            config_frame.execution_time = datetime.now()
        filename = self.get_filename(
            config_frame=config_frame,
            filename_pattern=filename_pattern,
            suffix=suffix,
            **values,
        )
        filename.parent.mkdir(parents=True, exist_ok=True)
        extension = filename.suffix.lower().lstrip(".") or self.config.extension
        if extension in self.TIFF_EXTENSIONS:
            metadata = {"ConfigFrame": self.config_frame_to_metadata(config_frame=config_frame)}
            tifffile.imwrite(filename, frame, description=json.dumps(metadata))
        else:
            skimage.io.imsave(filename, frame, check_contrast=False)
        return filename

    def _format_values(self, config_frame: ConfigFrame, suffix: str | None) -> dict[str, Any]:
        """
        Return default filename format values derived from a ConfigFrame.

        Parameters
        ----------
        config_frame
            ConfigFrame supplying metadata for filename variables.
        suffix
            Optional suffix value.

        Returns
        -------
        dict[str, Any]
            Format variable mapping for filename_pattern.
        """
        coordinate = config_frame.coordinate
        coordinate_dict = coordinate.to_dict() if coordinate is not None else {}
        timestamp_source = config_frame.execution_time or config_frame.creation_time
        return {
            "channel": self._format_channel(config_frame=config_frame),
            "position_id": config_frame.position_id,
            "x": coordinate_dict.get("X", ""),
            "y": coordinate_dict.get("Y", ""),
            "z": coordinate_dict.get("Z", "auto"),
            "filter_wheel": self._format_filter_wheel(config_frame.filter_wheel),
            "timestamp": timestamp_source.strftime("%Y-%m-%d_%H-%M-%S-%f"),
            "suffix": "" if suffix is None else suffix,
            "extension": self.config.extension,
        }

    @staticmethod
    def _format_channel(config_frame: ConfigFrame) -> str:
        """
        Return a filename-safe channel label for a ConfigFrame.

        Parameters
        ----------
        config_frame
            ConfigFrame whose LED settings should be represented.

        Returns
        -------
        str
            LED channel label for filenames.
        """
        if not config_frame.leds:
            return LEDType.NO_LED.name
        return "+".join(led_type.name.replace("_", "") for led_type in config_frame.leds)

    @staticmethod
    def _format_filter_wheel(filter_wheel: FilterWheelType | None) -> str:
        """
        Return a filename-safe filter wheel label.

        Parameters
        ----------
        filter_wheel
            FilterWheelType value or None.

        Returns
        -------
        str
            Filter wheel label for filenames.
        """
        return "None" if filter_wheel is None else str(filter_wheel.value)

    @classmethod
    def config_frame_to_metadata(cls, config_frame: ConfigFrame) -> dict[str, Any]:
        """
        Return a JSON-serializable metadata payload for a ConfigFrame.

        Parameters
        ----------
        config_frame
            ConfigFrame to serialize.

        Returns
        -------
        dict[str, Any]
            JSON-serializable representation of config_frame.
        """
        if not isinstance(config_frame, ConfigFrame):
            raise TypeError(
                f"FileManager.config_frame_to_metadata: config_frame must be ConfigFrame, "
                f"received {type(config_frame)}."
            )
        return {
            "leds": None if config_frame.leds is None else {
                led_type.name: {
                    "value": led_type.value,
                    "brightness": float(brightness),
                }
                for led_type, brightness in config_frame.leds.items()
            },
            "filter_wheel": cls._enum_to_metadata(config_frame.filter_wheel),
            "exposure": config_frame.exposure,
            "position_id": config_frame.position_id,
            "coordinate": cls._coordinate_to_metadata(config_frame.coordinate),
            "creation_time": config_frame.creation_time.isoformat(),
            "execution_time": None if config_frame.execution_time is None else config_frame.execution_time.isoformat(),
            "force_settings": config_frame.force_settings,
            "disable_leds_before": config_frame.disable_leds_before,
            "disable_leds_after": config_frame.disable_leds_after,
            "reset_leds_after": config_frame.reset_leds_after,
        }

    @staticmethod
    def _enum_to_metadata(value: EvoType | None) -> dict[str, Any] | None:
        """
        Return a JSON-serializable enum representation.

        Parameters
        ----------
        value
            EvoType enum value or None.

        Returns
        -------
        dict[str, Any] | None
            Enum name and value, or None.
        """
        if value is None:
            return None
        return {"name": value.name, "value": value.value}

    @staticmethod
    def _coordinate_to_metadata(coordinate: Coordinate | None) -> dict[str, float | int | None] | None:
        """
        Return a JSON-serializable coordinate representation.

        Parameters
        ----------
        coordinate
            Coordinate value or None.

        Returns
        -------
        dict[str, float | int | None] | None
            Coordinate dictionary or None.
        """
        if coordinate is None:
            return None
        return {
            "X": coordinate.x,
            "Y": coordinate.y,
            "Z": coordinate.z,
        }
