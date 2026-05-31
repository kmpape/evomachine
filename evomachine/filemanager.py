from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np
import skimage.io
import tifffile

from evomachine.config_types import FileNameConfig, FrameMetaData
from evomachine.types import FilterWheelType, LEDType


class FileManager:
    """Manager for frame filenames, save directories, and frame image metadata."""

    TIFF_EXTENSION = "tiff"
    NON_TIFF_IMAGE_SUFFIXES = {".tif", ".png", ".jpg", ".jpeg", ".bmp"}

    def __init__(self, config: FileNameConfig):
        """
        Initialise a file manager from a file name configuration.

        Parameters
        ----------
        config
            FileNameConfig defining the output directory and filename pattern.

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
            frame_metadata: FrameMetaData,
    ) -> Path:
        """
        Return an output path generated from the active filename pattern.

        Parameters
        ----------
        frame_metadata
            FrameMetaData supplying default filename variables.

        Returns
        -------
        Path
            Full TIFF output path under the configured directory.
        """
        if not isinstance(frame_metadata, FrameMetaData):
            raise TypeError(
                f"FileManager.get_filename: frame_metadata must be FrameMetaData, received {type(frame_metadata)}."
            )
        format_values = self._format_values(frame_metadata=frame_metadata)
        filename = self.config.filename_pattern.format(**format_values)
        filename_path = Path(filename)
        if filename_path.is_absolute():
            raise ValueError("FileManager.get_filename: filename_pattern must produce a relative path.")
        return self.config.directory / self._with_tiff_suffix(path=filename_path)

    def save_frame(
            self,
            frame: np.ndarray,
            frame_metadata: FrameMetaData,
    ) -> Path:
        """
        Save one frame image and return the output path.

        Parameters
        ----------
        frame
            Image array to save.
        frame_metadata
            FrameMetaData to include in TIFF metadata and filename variables.

        Returns
        -------
        Path
            Path where the frame was saved.
        """
        if not isinstance(frame, np.ndarray):
            raise TypeError(f"FileManager.save_frame: frame must be np.ndarray, received {type(frame)}.")
        if not isinstance(frame_metadata, FrameMetaData):
            raise TypeError(
                f"FileManager.save_frame: frame_metadata must be FrameMetaData, received {type(frame_metadata)}."
            )
        if frame_metadata.execution_time is None:
            frame_metadata.execution_time = datetime.now()
        filename = self.get_filename(frame_metadata=frame_metadata)
        filename.parent.mkdir(parents=True, exist_ok=True)
        metadata = {"FrameMetaData": frame_metadata.to_metadata_dict()}
        tifffile.imwrite(filename, frame, description=json.dumps(metadata))
        return filename

    @staticmethod
    def load_image(path: Path | str) -> np.ndarray:
        """
        Load an image from a path.

        Parameters
        ----------
        path
            Path to an image file supported by skimage.io.imread.

        Returns
        -------
        np.ndarray
            Loaded image data.
        """
        image_path = FileManager._validate_file_path(path=path, action="FileManager.load_image")
        return skimage.io.imread(image_path)

    @classmethod
    def load_tiff_metadata(cls, path: Path | str) -> dict[str, Any]:
        """
        Load JSON metadata from a TIFF ImageDescription tag.

        Parameters
        ----------
        path
            Path to a TIFF image file written with JSON metadata.

        Returns
        -------
        dict[str, Any]
            Metadata decoded from the TIFF ImageDescription tag.
        """
        image_path = cls._validate_file_path(path=path, action="FileManager.load_tiff_metadata")
        extension = image_path.suffix.lower().lstrip(".")
        if extension != cls.TIFF_EXTENSION:
            raise ValueError(f"FileManager.load_tiff_metadata: path must be a TIFF file, received {image_path}.")
        with tifffile.TiffFile(image_path) as tiff:
            description = tiff.pages[0].tags["ImageDescription"].value
        return json.loads(description)

    @classmethod
    def load_frame(cls, path: Path | str) -> tuple[np.ndarray, dict[str, Any] | None]:
        """
        Load a TIFF image and its metadata.

        Parameters
        ----------
        path
            Path to an image file.

        Returns
        -------
        tuple[np.ndarray, dict[str, Any] | None]
            Loaded image and metadata dictionary.
        """
        image_path = cls._validate_file_path(path=path, action="FileManager.load_frame")
        image = cls.load_image(path=image_path)
        metadata = cls.load_tiff_metadata(path=image_path)
        return image, metadata

    @staticmethod
    def list_filenames(directory: Path | str, filename_pattern: str) -> list[Path]:
        """
        Return files in one directory matching a glob filename pattern.

        Parameters
        ----------
        directory
            Folder to search.
        filename_pattern
            Relative glob pattern, such as "*.tiff" or "LED450NM_*_ref.tiff".

        Returns
        -------
        list[Path]
            Sorted matching file paths. Directories are excluded.
        """
        if isinstance(directory, str):
            directory = Path(directory)
        if not isinstance(directory, Path):
            raise TypeError(f"FileManager.list_filenames: directory must be Path or str, received {type(directory)}.")
        if not directory.exists():
            raise FileNotFoundError(f"FileManager.list_filenames: directory does not exist: {directory}.")
        if not directory.is_dir():
            raise NotADirectoryError(f"FileManager.list_filenames: path is not a directory: {directory}.")
        if not isinstance(filename_pattern, str):
            raise TypeError(
                f"FileManager.list_filenames: filename_pattern must be str, received {type(filename_pattern)}."
            )
        if Path(filename_pattern).is_absolute():
            raise ValueError("FileManager.list_filenames: filename_pattern must be relative.")
        return sorted(path for path in directory.glob(filename_pattern) if path.is_file())

    @staticmethod
    def _validate_file_path(path: Path | str, action: str) -> Path:
        """
        Return a validated existing file path.

        Parameters
        ----------
        path
            Candidate file path.
        action
            Method name used in raised error messages.

        Returns
        -------
        Path
            Validated file path.
        """
        if isinstance(path, str):
            path = Path(path)
        if not isinstance(path, Path):
            raise TypeError(f"{action}: path must be Path or str, received {type(path)}.")
        if not path.exists():
            raise FileNotFoundError(f"{action}: path does not exist: {path}.")
        if not path.is_file():
            raise FileNotFoundError(f"{action}: path is not a file: {path}.")
        return path

    @classmethod
    def _with_tiff_suffix(cls, path: Path) -> Path:
        """
        Return a path ending in the hard-coded TIFF suffix.

        Parameters
        ----------
        path
            Relative filename path generated from the active filename pattern.

        Returns
        -------
        Path
            Filename path ending in .tiff, preserving non-extension periods in
            stems such as multi-LED channel labels.
        """
        if path.suffix.lower() == f".{cls.TIFF_EXTENSION}":
            return path
        if path.suffix.lower() in cls.NON_TIFF_IMAGE_SUFFIXES:
            return path.with_suffix(f".{cls.TIFF_EXTENSION}")
        return path.with_name(f"{path.name}.{cls.TIFF_EXTENSION}")

    def _format_values(self, frame_metadata: FrameMetaData) -> dict[str, Any]:
        """
        Return default filename format values derived from FrameMetaData.

        Parameters
        ----------
        frame_metadata
            FrameMetaData supplying metadata for filename variables.

        Returns
        -------
        dict[str, Any]
            Format variable mapping for filename_pattern.
        """
        coordinate = frame_metadata.coordinate
        coordinate_dict = coordinate.to_dict() if coordinate is not None else {}
        timestamp_source = frame_metadata.execution_time or frame_metadata.creation_time
        format_values = {
            "channel": self._format_channel(frame_metadata=frame_metadata),
            "fov_id": frame_metadata.fov_id,
            "x": coordinate_dict.get("X", ""),
            "y": coordinate_dict.get("Y", ""),
            "z": coordinate_dict.get("Z", "auto"),
            "filter_wheel": self._format_filter_wheel(frame_metadata.filter_wheel),
            "timestamp": timestamp_source.strftime("%Y-%m-%d_%H-%M-%S-%f"),
            "frame_id": frame_metadata.frame_id,
            "callback_id": "" if frame_metadata.callback_id is None else frame_metadata.callback_id,
        }
        format_values.update(frame_metadata.additional_metadata)
        return format_values

    @staticmethod
    def _format_channel(frame_metadata: FrameMetaData) -> str:
        """
        Return a filename-safe channel label for FrameMetaData.

        Parameters
        ----------
        frame_metadata
            FrameMetaData whose LED settings should be represented.

        Returns
        -------
        str
            LED channel label for filenames.
        """
        if not frame_metadata.leds:
            return LEDType.NO_LED.name
        return ".".join(led_type.name.replace("_", "") for led_type in frame_metadata.leds)

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
