from __future__ import annotations

from enum import Enum
import json
import os
from pathlib import Path
import re
from string import Formatter
import threading
from typing import Any

import numpy as np
from pydantic import Field, field_validator
import skimage.io
import tifffile

from evomachine.config_models import EvoConfig
from evomachine.config import (
    TimeConfig,
    ensure_timezone_aware,
    format_timestamp_for_filename,
    format_timezone_id,
    now,
)
from evomachine.frame import FrameMetaData
from evomachine.types import FilterWheelType, LEDType, UNKNOWN_FOV_ID


class FileNameField(str, Enum):
    """Fields available for substitution in file name patterns."""

    CHANNEL = "channel"
    FOV_ID = "fov_id"
    X = "x"
    Y = "y"
    Z = "z"
    FILTER_WHEEL = "filter_wheel"
    TIMESTAMP = "timestamp"
    TIMEZONE = "timezone"
    FRAME_ID = "frame_id"
    CALLBACK_ID = "callback_id"
    FILE_ID = "file_id"
    LIGHT = "light"
    DATETIME = "datetime"
    Z_RANGE = "z_range"


class FileNameConfig(EvoConfig):
    """Configuration object for FileManager output paths and filename patterns."""

    directory: Path
    filename_pattern: str = "FOV{fov_id}_{timestamp}_{callback_id}"
    file_id_digits: int = 6
    "Zero-padded width of the {file_id} filename field."
    create_directory: bool = True
    time_config: TimeConfig = Field(default_factory=TimeConfig)

    @field_validator("directory", mode="before")
    @classmethod
    def _validate_directory_type(cls, value: object) -> object:
        if not isinstance(value, Path | str):
            raise TypeError(f"FileNameConfig: directory must be Path or str, received {type(value)}.")
        return value

    @field_validator("filename_pattern", mode="before")
    @classmethod
    def _validate_filename_pattern_type(cls, value: object) -> object:
        """
        Validate the filename pattern type, syntax, and substitution fields.

        Parameters
        ----------
        value
            Candidate filename pattern.

        Returns
        -------
        object
            The validated filename pattern.
        """
        if not isinstance(value, str):
            raise TypeError(f"FileNameConfig: filename_pattern must be str, received {type(value)}.")
        allowed_fields = {field.value for field in FileNameField}
        invalid_fields: set[str] = set()
        patterns_to_parse = [value]
        formatter = Formatter()
        try:
            while patterns_to_parse:
                pattern = patterns_to_parse.pop()
                for _, field_name, format_spec, conversion in formatter.parse(pattern):
                    if field_name is not None and field_name not in allowed_fields:
                        invalid_fields.add(field_name)
                    if conversion is not None and conversion not in {"a", "r", "s"}:
                        raise ValueError(f"invalid conversion '!{conversion}'")
                    if format_spec:
                        patterns_to_parse.append(format_spec)
        except ValueError as error:
            raise ValueError(f"FileNameConfig: filename_pattern has invalid format syntax: {error}.") from error
        if invalid_fields:
            invalid = ", ".join(repr(field) for field in sorted(invalid_fields))
            allowed = ", ".join(sorted(allowed_fields))
            raise ValueError(
                f"FileNameConfig: filename_pattern contains unsupported fields: {invalid}. "
                f"Allowed fields are: {allowed}."
            )
        return value

    @field_validator("file_id_digits", mode="before")
    @classmethod
    def _validate_file_id_digits(cls, value: object) -> object:
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"FileNameConfig: file_id_digits must be int, received {type(value)}.")
        if not 1 <= value <= 12:
            raise ValueError(f"FileNameConfig: file_id_digits must be in [1, 12], received {value}.")
        return value

    @field_validator("create_directory", mode="before")
    @classmethod
    def _validate_create_directory_type(cls, value: object) -> object:
        if not isinstance(value, bool):
            raise TypeError(f"FileNameConfig: create_directory must be bool, received {type(value)}.")
        return value

    def model_post_init(self, __context) -> None:
        if isinstance(self.directory, str):
            self.directory = Path(self.directory)
        if not isinstance(self.directory, Path):
            raise TypeError(f"FileNameConfig: directory must be Path or str, received {type(self.directory)}.")
        if not isinstance(self.filename_pattern, str):
            raise TypeError(f"FileNameConfig: filename_pattern must be str, received {type(self.filename_pattern)}.")
        if self.filename_pattern == "":
            raise ValueError("FileNameConfig: filename_pattern must not be empty.")
        if not isinstance(self.create_directory, bool):
            raise TypeError(f"FileNameConfig: create_directory must be bool, received {type(self.create_directory)}.")
        if not isinstance(self.time_config, TimeConfig):
            raise TypeError(f"FileNameConfig: time_config must be TimeConfig, received {type(self.time_config)}.")


class FileManager:
    """Manager for frame filenames, save directories, and frame image metadata."""

    TIFF_EXTENSION = "tiff"
    NON_TIFF_IMAGE_SUFFIXES = {".tif", ".png", ".jpg", ".jpeg", ".bmp"}
    FILE_ID_COUNTER_FILENAME = ".evomachine_file_id"
    "Hidden file in each output directory holding the last reserved {file_id}."

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
        self._file_id_lock = threading.Lock()
        self._last_file_ids: dict[Path, int] = {}
        self._ensure_directory()

    def set_output_directory(self, directory: Path | str) -> Path:
        """Use an existing local directory for subsequent saved acquisitions."""
        if not isinstance(directory, Path | str):
            raise TypeError(
                "FileManager.set_output_directory: directory must be Path or str, "
                f"received {type(directory)}."
            )
        output_directory = Path(directory).expanduser()
        if not output_directory.is_dir():
            raise FileNotFoundError(
                f"FileManager.set_output_directory: directory does not exist: {output_directory}."
            )
        self.update_config(directory=output_directory)
        return output_directory

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

    def next_file_id(self) -> int:
        """
        Reserve and return the next {file_id} for the current output directory.

        IDs are one more than the highest ID found in the directory's counter file or in
        existing ID-prefixed TIFF filenames. The counter file is updated before returning,
        so IDs keep increasing across restarts and are not reused if files are deleted.

        Returns
        -------
        int
            Reserved file ID, starting at 1.
        """
        with self._file_id_lock:
            directory = self.config.directory
            file_id = self._current_max_file_id(directory=directory) + 1
            counter_path = directory / self.FILE_ID_COUNTER_FILENAME
            temporary = counter_path.with_name(f"{counter_path.name}.tmp")
            temporary.write_text(f"{file_id}\n")
            os.replace(temporary, counter_path)
            self._last_file_ids[directory] = file_id
            return file_id

    def _current_max_file_id(self, directory: Path) -> int:
        """Return the highest file ID already used in ``directory``, or 0."""
        highest = self._last_file_ids.get(directory)
        if highest is None:
            # First use of this directory: scan once for files written before a restart.
            highest = 0
            id_pattern = re.compile(rf"^(\d{{{self.config.file_id_digits}}})_.*\.tiff?$", re.IGNORECASE)
            if directory.is_dir():
                for entry in directory.iterdir():
                    match = id_pattern.match(entry.name)
                    if match:
                        highest = max(highest, int(match.group(1)))
            self._last_file_ids[directory] = highest
        counter_path = directory / self.FILE_ID_COUNTER_FILENAME
        try:
            highest = max(highest, int(counter_path.read_text().strip() or 0))
        except FileNotFoundError:
            pass
        except ValueError as error:
            raise RuntimeError(f"FileManager: file ID counter {counter_path} is corrupt.") from error
        return highest

    @staticmethod
    def _pattern_uses_field(pattern: str, field_name: str) -> bool:
        """Return whether a str.format pattern references ``field_name``."""
        return any(
            name is not None and re.split(r"[.\[]", name, maxsplit=1)[0] == field_name
            for _, name, _, _ in Formatter().parse(pattern)
        )

    def get_filename(
            self,
            frame_metadata: FrameMetaData,
            file_id: int | None = None,
    ) -> Path:
        """
        Return an output path generated from the active filename pattern.

        Parameters
        ----------
        frame_metadata
            FrameMetaData supplying default filename variables.
        file_id
            ID for the {file_id} field. When None, the next free ID is shown without
            reserving it; save_frame() reserves one.

        Returns
        -------
        Path
            Full TIFF output path under the configured directory.
        """
        if not isinstance(frame_metadata, FrameMetaData):
            raise TypeError(
                f"FileManager.get_filename: frame_metadata must be FrameMetaData, received {type(frame_metadata)}."
            )
        return self._render_filename(
            pattern=self.config.filename_pattern,
            format_values=self._format_values(frame_metadata=frame_metadata, file_id=file_id),
            action="FileManager.get_filename",
        )

    def _render_filename(self, pattern: str, format_values: dict[str, Any], action: str) -> Path:
        """Format ``pattern`` and return a TIFF path under the output directory."""
        filename_path = Path(pattern.format(**format_values))
        if filename_path.is_absolute():
            raise ValueError(f"{action}: filename pattern must produce a relative path.")
        return self.config.directory / self._with_tiff_suffix(path=filename_path)

    def _reserve_file_id(self, pattern: str) -> int | None:
        """Reserve a file ID only when the pattern actually contains {file_id}."""
        return self.next_file_id() if self._pattern_uses_field(pattern, "file_id") else None

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
        self._normalise_execution_time(frame_metadata=frame_metadata)
        # Serialise before reserving an ID so invalid metadata leaves the directory untouched.
        metadata = {"FrameMetaData": frame_metadata.to_metadata_dict()}
        file_id = self._reserve_file_id(pattern=self.config.filename_pattern)
        filename = self.get_filename(frame_metadata=frame_metadata, file_id=file_id)
        filename.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(filename, frame, description=json.dumps(metadata))
        return filename

    def save_z_stack(
            self,
            stack: np.ndarray,
            frame_metadata: list[FrameMetaData],
            centre_z: float | None = None,
            expected_planes: int | None = None,
    ) -> Path:
        """
        Save a z-stack as one multi-page TIFF and return the output path.

        Planes are written sorted by Z (bottom to top). The ImageDescription holds
        {"FrameMetaData": [one dict per plane, in file order], "ZStack": {...}}, where
        "ZStack" records the Z positions, offsets from ``centre_z``, step, acquisition
        order and completeness. Load the file with load_frame() or tifffile.imread().

        Parameters
        ----------
        stack
            Planes with shape (n_planes, height, width), in acquisition order.
        frame_metadata
            One FrameMetaData per plane, aligned with ``stack``.
        centre_z
            Z the stack was taken around. The {z_range} filename field is relative to it
            when given, and absolute otherwise.
        expected_planes
            Number of planes requested, used to mark stopped stacks as incomplete.

        Returns
        -------
        Path
            Path of the saved stack file.
        """
        if not isinstance(stack, np.ndarray) or stack.ndim != 3:
            raise ValueError(
                "FileManager.save_z_stack: stack must be a 3D np.ndarray, "
                f"received {getattr(stack, 'shape', type(stack))}."
            )
        if (not isinstance(frame_metadata, list)
                or not all(isinstance(metadata, FrameMetaData) for metadata in frame_metadata)):
            raise TypeError("FileManager.save_z_stack: frame_metadata must be list[FrameMetaData].")
        if len(frame_metadata) != stack.shape[0]:
            raise ValueError(
                f"FileManager.save_z_stack: {len(frame_metadata)} metadata entries for {stack.shape[0]} planes."
            )
        if len({metadata.fov_id for metadata in frame_metadata}) != 1:
            raise ValueError("FileManager.save_z_stack: all planes must share one fov_id.")
        for metadata in frame_metadata:
            self._normalise_execution_time(frame_metadata=metadata)

        z_values = [None if m.coordinate is None else m.coordinate.z for m in frame_metadata]
        has_z = all(value is not None for value in z_values)
        order = sorted(range(len(z_values)), key=lambda index: z_values[index]) if has_z else list(range(len(z_values)))
        z_positions = [float(z_values[index]) for index in order] if has_z else None
        z_offsets = None if z_positions is None or centre_z is None else [z - centre_z for z in z_positions]
        z_step = None
        if z_positions is not None and len(z_positions) > 1:
            z_step = z_positions[1] - z_positions[0]
        sorted_metadata = [frame_metadata[index] for index in order]
        times = [metadata.execution_time or metadata.creation_time for metadata in frame_metadata]
        description = json.dumps({
            "FrameMetaData": [metadata.to_metadata_dict() for metadata in sorted_metadata],
            "ZStack": {
                "axes": "ZYX",
                "n_planes": len(order),
                "expected_planes": expected_planes,
                "complete": expected_planes is None or expected_planes == len(order),
                "z_positions": z_positions,
                "z_centre": None if centre_z is None else float(centre_z),
                "z_offsets": z_offsets,
                "z_step": z_step,
                # acquisition_order[k] is the file position of the k-th acquired plane.
                "acquisition_order": [order.index(index) for index in range(len(order))],
                "start_time": min(times).isoformat(),
                "end_time": max(times).isoformat(),
            },
        })

        pattern = self.config.filename_pattern
        file_id = self._reserve_file_id(pattern=pattern)
        # Name the file after the first acquired plane (stack start time).
        format_values = self._format_values(frame_metadata=frame_metadata[0], file_id=file_id)
        format_values[FileNameField.Z_RANGE.value] = self._format_z_range(
            z_positions=z_positions, z_step=z_step, centre_z=centre_z,
        )
        filename = self._render_filename(
            pattern=pattern, format_values=format_values, action="FileManager.save_z_stack",
        )
        filename.parent.mkdir(parents=True, exist_ok=True)
        # Write under a hidden temporary name, then rename, so a crash never leaves a
        # truncated stack under the final name.
        temporary = filename.with_name(f".{filename.name}.partial")
        try:
            tifffile.imwrite(
                temporary,
                stack[order],
                photometric="minisblack",  # greyscale planes, never RGB
                description=description,
                metadata=None,  # keep a single ImageDescription
            )
            os.replace(temporary, filename)
        finally:
            if temporary.exists():
                temporary.unlink()
        return filename

    def _normalise_execution_time(self, frame_metadata: FrameMetaData) -> None:
        """Fill a missing execution time and convert it to the configured timezone."""
        if frame_metadata.execution_time is None:
            frame_metadata.execution_time = now(config=self.config.time_config)
        else:
            frame_metadata.execution_time = ensure_timezone_aware(
                value=frame_metadata.execution_time,
                config=self.config.time_config,
            )

    @classmethod
    def _format_z_range(
            cls,
            z_positions: list[float] | None,
            z_step: float | None = None,
            centre_z: float | None = None,
    ) -> str:
        """
        Return the {z_range} filename field as 'min,step,max'.

        Offsets are relative to ``centre_z`` when known (3 decimals) and absolute
        otherwise (2 decimals). One plane gives just its Z; no Z gives 'auto'.
        """
        if not z_positions:
            return "auto"
        reference, decimals = (0.0, 2) if centre_z is None else (float(centre_z), 3)
        low = cls._format_number(z_positions[0] - reference, decimals)
        if len(z_positions) == 1:
            return low
        high = cls._format_number(z_positions[-1] - reference, decimals)
        return f"{low},{cls._format_number(z_step, 3)},{high}"

    @staticmethod
    def _format_number(value: float, decimals: int) -> str:
        """Round for filenames without float noise, trailing zeros or '-0'."""
        rounded = round(float(value), decimals)
        if rounded == 0:
            rounded = 0.0
        return f"{rounded:.{decimals}f}".rstrip("0").rstrip(".")

    @staticmethod
    def load_image(path: Path | str) -> np.ndarray:
        """
        Load an image from a path.

        TIFF files are read with tifffile, so a multi-page z-stack is always returned as
        (n_planes, height, width); other formats use skimage.io.imread.

        Parameters
        ----------
        path
            Path to an image file.

        Returns
        -------
        np.ndarray
            Loaded image data.
        """
        image_path = FileManager._validate_file_path(path=path, action="FileManager.load_image")
        if image_path.suffix.lower() in {".tif", ".tiff"}:
            # skimage would return 3- and 4-plane stacks as (H, W, N) "colour" images.
            return tifffile.imread(image_path)
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
        return sorted(
            (path for path in directory.glob(filename_pattern) if path.is_file()),
            key=lambda path: FileManager._natural_path_key(path=path, directory=directory),
        )

    @staticmethod
    def _natural_path_key(path: Path, directory: Path) -> tuple[tuple[tuple[int, int | str], ...], str]:
        """
        Return a deterministic, case-insensitive natural-sort key for a path.

        Parameters
        ----------
        path
            File path being sorted.
        directory
            Directory against which the path is represented.

        Returns
        -------
        tuple[tuple[tuple[int, int | str], ...], str]
            Numeric-aware key followed by the original relative path as a
            deterministic tie breaker.
        """
        relative_path = path.relative_to(directory).as_posix()
        components: list[tuple[int, int | str]] = []
        for component in re.split(r"(\d+)", relative_path.casefold()):
            if not component:
                continue
            components.append((0, int(component)) if component.isdigit() else (1, component))
        return tuple(components), relative_path

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

    def _format_values(self, frame_metadata: FrameMetaData, file_id: int | None = None) -> dict[str, Any]:
        """
        Return default filename format values derived from FrameMetaData.

        Parameters
        ----------
        frame_metadata
            FrameMetaData supplying metadata for filename variables.
        file_id
            Reserved file ID. When None and the pattern needs it, the next free ID is shown.

        Returns
        -------
        dict[str, Any]
            Format variable mapping for filename_pattern.
        """
        coordinate = frame_metadata.coordinate
        coordinate_dict = coordinate.to_dict() if coordinate is not None else {}
        timestamp_source = ensure_timezone_aware(
            value=frame_metadata.execution_time or frame_metadata.creation_time,
            config=self.config.time_config,
        )
        format_values = {
            FileNameField.CHANNEL.value: self._format_channel(frame_metadata=frame_metadata),
            FileNameField.FOV_ID.value: (
                "" if frame_metadata.fov_id == UNKNOWN_FOV_ID else frame_metadata.fov_id
            ),
            FileNameField.X.value: coordinate_dict.get("X", ""),
            FileNameField.Y.value: coordinate_dict.get("Y", ""),
            FileNameField.Z.value: coordinate_dict.get("Z", "auto"),
            FileNameField.FILTER_WHEEL.value: self._format_filter_wheel(frame_metadata.filter_wheel),
            FileNameField.TIMESTAMP.value: format_timestamp_for_filename(
                value=timestamp_source,
                config=self.config.time_config,
            ),
            FileNameField.TIMEZONE.value: format_timezone_id(
                value=timestamp_source,
                config=self.config.time_config,
            ),
            FileNameField.FRAME_ID.value: frame_metadata.frame_id,
            FileNameField.CALLBACK_ID.value: "" if frame_metadata.callback_id is None else frame_metadata.callback_id,
            FileNameField.FILE_ID.value: self._format_file_id(file_id=file_id),
            FileNameField.LIGHT.value: self._format_light(frame_metadata=frame_metadata),
            FileNameField.DATETIME.value: timestamp_source.strftime("%y%m%d_%H%M%S"),
            FileNameField.Z_RANGE.value: self._format_z_range(
                z_positions=None if "Z" not in coordinate_dict else [float(coordinate_dict["Z"])],
            ),
        }
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

    def _format_file_id(self, file_id: int | None) -> str:
        """Return the zero-padded {file_id}; the next free ID when none was reserved."""
        if file_id is None:
            with self._file_id_lock:
                file_id = self._current_max_file_id(directory=self.config.directory) + 1
        return f"{file_id:0{self.config.file_id_digits}d}"

    @staticmethod
    def _format_light(frame_metadata: FrameMetaData) -> str:
        """
        Return the condensed light label for filenames.

        LED_450_NM becomes 450NM, several LEDs are joined with '+', and no LED is NOLED.
        """
        if not frame_metadata.leds:
            return "NOLED"
        return "+".join(
            "NOLED" if led_type is LEDType.NO_LED else led_type.name.removeprefix("LED_").replace("_", "")
            for led_type in frame_metadata.leds
        )

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
