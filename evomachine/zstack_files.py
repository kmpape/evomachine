"""Write and read z-stacks as single multi-page TIFF files.

One file holds every plane of one z-stack, sorted by Z (lowest first, axes ``ZYX``).
Filenames start with a persistent, monotonically increasing stack ID so that sorting
by name gives acquisition order, and contain no stage coordinates (those are stored
in the metadata)::

    {stack_id:06d}_{light}_FOV{fov_id:02d}_Z{z_min},{z_step},{z_max}_{yymmdd}_{HHMMSS}.tiff
    e.g. 000123_450_FOV03_Z-10,1,10_261108_155754.tiff

The Z part is relative to the stack centre by default (``z_in_filename="relative"``),
or absolute stage Z rounded to 0.1 um (``"absolute"``). The date/time is the local
execution time of the first acquired plane.

The first page's ImageDescription holds JSON::

    {"FrameMetaData": [one FrameMetaData.to_metadata_dict() per plane, in file order],
     "ZStack": {...stack-level information, see ZStackFileWriter.write...}}

This module only depends on FrameMetaData and tifffile and does not change any
existing acquisition or FileManager behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Literal

import numpy as np
import tifffile

from evomachine.config import TimeConfig, ensure_timezone_aware
from evomachine.frame import FrameMetaData
from evomachine.types import FilterWheelType, LEDType

ZSTACK_FORMAT_VERSION = 1
"""Increment when the layout of the "ZStack" metadata block changes incompatibly."""

COUNTER_FILENAME = ".zstack_id_counter"
"""Hidden file in the output directory holding the last stack ID that was handed out."""

# Only files that follow the stack naming scheme count towards the next ID, so an
# unrelated file such as '20261108_overview.tif' cannot inflate the IDs.
_STACK_FILENAME_RE = re.compile(r"^(\d+)_[^_]+_FOV[^_]+_Z[^_]+_\d{6}_\d{6}\.tiff$")


@dataclass(frozen=True)
class ZStackChannel:
    """Illumination settings for one z-stack channel.

    Defined here rather than in a strategy file: strategy files are executed without
    being registered in sys.modules, and on Python 3.11 a dataclass defined in such a
    module fails to build when ``from __future__ import annotations`` is active.
    """

    led: LEDType
    brightness: float
    "LED brightness in [0, 100]."
    exposure_ms: float
    "Camera exposure in milliseconds, in [1, 1000]."
    filter_wheel: FilterWheelType | None = None
    "Filter to select, or None to leave the filter wheel where it is."


@dataclass(frozen=True, kw_only=True)
class ZStackFileConfig:
    """Where and how z-stack files are written."""

    directory: Path
    "Output directory. Created if missing. Stack IDs are unique within this directory."
    z_in_filename: Literal["relative", "absolute"] = "relative"
    "Write Z range relative to the stack centre, or as absolute stage Z."
    id_digits: int = 6
    "Zero-padding width of the stack ID (IDs beyond this width simply get longer)."
    compression: str | None = None
    "Optional tifffile compression, e.g. 'zlib'. None keeps the existing uncompressed format."
    time_config: TimeConfig | None = None
    "Time zone used for the filename date/time. None uses evomachine's default (Europe/London)."

    def __post_init__(self) -> None:
        object.__setattr__(self, "directory", Path(self.directory).expanduser())
        if self.z_in_filename not in ("relative", "absolute"):
            raise ValueError(f"ZStackFileConfig: z_in_filename must be 'relative' or 'absolute', "
                             f"received {self.z_in_filename!r}.")
        if not isinstance(self.id_digits, int) or self.id_digits < 1:
            raise ValueError(f"ZStackFileConfig: id_digits must be a positive int, received {self.id_digits!r}.")


@dataclass(frozen=True)
class ZStackFileRecord:
    """Summary of one written z-stack file."""

    path: Path
    stack_id: int
    n_planes: int
    complete: bool
    z_positions: tuple[float, ...]


class ZStackFileWriter:
    """Allocate stack IDs and write one multi-page TIFF per z-stack."""

    def __init__(self, config: ZStackFileConfig):
        if not isinstance(config, ZStackFileConfig):
            raise TypeError(f"ZStackFileWriter: config must be ZStackFileConfig, received {type(config)}.")
        self.config = config

    # ------------------------------------------------------------------ IDs
    def _existing_max_id(self) -> int:
        """Return the largest stack ID found in filenames or the counter file, or 0."""
        directory = self.config.directory
        highest = 0
        if directory.is_dir():
            for entry in directory.iterdir():
                match = _STACK_FILENAME_RE.match(entry.name)
                if match:
                    highest = max(highest, int(match.group(1)))
        counter_path = directory / COUNTER_FILENAME
        try:
            highest = max(highest, int(counter_path.read_text().strip() or 0))
        except FileNotFoundError:
            pass
        except ValueError as error:
            raise RuntimeError(f"ZStackFileWriter: counter file {counter_path} is corrupt.") from error
        return highest

    def next_stack_id(self) -> int:
        """
        Reserve and return the next stack ID for this directory.

        The ID is one more than the highest ID seen in existing filenames or in the
        counter file, and the counter file is updated before returning. IDs therefore
        keep increasing across restarts, and are never reused even if a file is later
        moved or deleted (as long as the counter file stays).
        """
        self.config.directory.mkdir(parents=True, exist_ok=True)
        stack_id = self._existing_max_id() + 1
        counter_path = self.config.directory / COUNTER_FILENAME
        temporary = counter_path.with_name(counter_path.name + ".tmp")
        temporary.write_text(f"{stack_id}\n")
        os.replace(temporary, counter_path)
        return stack_id

    # ------------------------------------------------------------ filenames
    @staticmethod
    def format_light(frame_metadata: FrameMetaData) -> str:
        """Condensed light label: 'LED_450_NM' -> '450', several LEDs joined by '+'."""
        if not frame_metadata.leds:
            return "NOLED"
        labels = []
        for led in frame_metadata.leds:
            if led is LEDType.NO_LED:
                labels.append("NOLED")
                continue
            name = led.name.removeprefix("LED_").removesuffix("_NM")
            labels.append(name.replace("_", ""))
        return "+".join(labels)

    @staticmethod
    def _format_number(value: float, decimals: int) -> str:
        rounded = round(value, decimals)
        if rounded == 0:
            rounded = 0.0  # avoid '-0'
        return f"{rounded:.{decimals}f}".rstrip("0").rstrip(".")

    def build_filename(
            self,
            stack_id: int,
            frame_metadata: FrameMetaData,
            z_min: float,
            z_step: float | None,
            z_max: float,
            timestamp: Any,
    ) -> str:
        """Return the filename (without directory) for one stack."""
        decimals = 3 if self.config.z_in_filename == "relative" else 1
        step = "x" if z_step is None else self._format_number(z_step, 3)
        z_part = f"Z{self._format_number(z_min, decimals)},{step},{self._format_number(z_max, decimals)}"
        local_time = ensure_timezone_aware(value=timestamp, config=self.config.time_config)
        fov = frame_metadata.fov_id
        fov_part = f"FOV{fov:02d}" if fov >= 0 else "FOVx"
        return (f"{stack_id:0{self.config.id_digits}d}_{self.format_light(frame_metadata)}_{fov_part}_"
                f"{z_part}_{local_time.strftime('%y%m%d_%H%M%S')}.tiff")

    # -------------------------------------------------------------- writing
    def write(
            self,
            array: np.ndarray,
            frame_metadata: list[FrameMetaData],
            *,
            centre_z: float | None = None,
            z_step: float | None = None,
            expected_planes: int | None = None,
            extra: dict[str, Any] | None = None,
    ) -> ZStackFileRecord:
        """
        Write one z-stack to a single multi-page TIFF.

        Parameters
        ----------
        array
            Stack of shape (n_planes, H, W), in acquisition order.
        frame_metadata
            One FrameMetaData per plane, in the same order as ``array``. Each plane's
            ``coordinate.z`` gives its Z position; planes are sorted by it before writing.
        centre_z
            Z the stack was centred on. Required for relative Z in the filename; when None
            the filename falls back to absolute Z.
        z_step
            Requested step size, written into the filename and metadata. When None it is
            inferred from the plane spacing ('x' in the filename if it cannot be).
        expected_planes
            Number of planes that were requested, used to flag incomplete stacks.
        extra
            Additional JSON-serialisable stack-level information (e.g. strategy name, round).

        Returns
        -------
        ZStackFileRecord
            Path, stack ID and plane summary of the written file.
        """
        if not isinstance(array, np.ndarray) or array.ndim != 3:
            raise ValueError(f"ZStackFileWriter.write: array must be a 3D ndarray, received "
                             f"{getattr(array, 'shape', type(array))}.")
        if len(frame_metadata) != array.shape[0]:
            raise ValueError(f"ZStackFileWriter.write: {len(frame_metadata)} metadata entries for "
                             f"{array.shape[0]} planes.")
        if any(m.coordinate is None or m.coordinate.z is None for m in frame_metadata):
            raise ValueError("ZStackFileWriter.write: every plane needs a coordinate with Z.")
        if len({m.fov_id for m in frame_metadata}) != 1:
            raise ValueError("ZStackFileWriter.write: all planes must share one fov_id.")

        acquired_z = [float(m.coordinate.z) for m in frame_metadata]
        order = sorted(range(len(acquired_z)), key=lambda index: acquired_z[index])
        sorted_array = array[order]
        sorted_metadata = [frame_metadata[index] for index in order]
        z_positions = [acquired_z[index] for index in order]
        # acquisition_order[k] = position in the file of the k-th acquired plane
        acquisition_order = [order.index(index) for index in range(len(order))]

        if z_step is None and len(z_positions) > 1:
            spacings = {round(b - a, 6) for a, b in zip(z_positions, z_positions[1:])}
            z_step = spacings.pop() if len(spacings) == 1 else None
        use_relative = self.config.z_in_filename == "relative" and centre_z is not None
        reference = float(centre_z) if use_relative else 0.0
        z_offsets = [z - centre_z for z in z_positions] if centre_z is not None else None
        complete = expected_planes is None or expected_planes == len(z_positions)

        execution_times = [m.execution_time or m.creation_time for m in frame_metadata]
        first_time = min(execution_times)
        stack_id = self.next_stack_id()
        filename = self.build_filename(
            stack_id=stack_id,
            frame_metadata=frame_metadata[0],
            z_min=z_positions[0] - reference,
            z_step=z_step,
            z_max=z_positions[-1] - reference,
            timestamp=first_time,
        )
        path = self.config.directory / filename

        stack_info = {
            "format_version": ZSTACK_FORMAT_VERSION,
            "stack_id": stack_id,
            "axes": "ZYX",
            "n_planes": len(z_positions),
            "expected_planes": expected_planes,
            "complete": complete,
            "fov_id": frame_metadata[0].fov_id,
            "light": self.format_light(frame_metadata[0]),
            "z_positions": z_positions,
            "z_centre": None if centre_z is None else float(centre_z),
            "z_offsets": z_offsets,
            "z_step": z_step,
            "z_in_filename": "relative" if use_relative else "absolute",
            "acquisition_order": acquisition_order,
            "start_time": ensure_timezone_aware(first_time).isoformat(),
            "end_time": ensure_timezone_aware(max(execution_times)).isoformat(),
            **(extra or {}),
        }
        description = json.dumps({
            "FrameMetaData": [m.to_metadata_dict() for m in sorted_metadata],
            "ZStack": stack_info,
        })

        # Write to a hidden temporary file first so a crash never leaves a truncated
        # file under the final name, then move it into place.
        temporary = path.with_name(f".{path.name}.partial")
        try:
            tifffile.imwrite(
                temporary,
                sorted_array,
                # Tells tifffile-based readers these are greyscale planes, not RGB. Note that
                # skimage.io.imread (used by FileManager.load_image) still returns 3- and
                # 4-plane stacks as (H, W, N); use read_zstack() or tifffile.imread instead.
                photometric="minisblack",
                description=description,
                metadata=None,              # no second, tifffile-generated description
                compression=self.config.compression,
            )
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
        return ZStackFileRecord(
            path=path,
            stack_id=stack_id,
            n_planes=len(z_positions),
            complete=complete,
            z_positions=tuple(z_positions),
        )


def read_zstack(path: Path | str) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Read a z-stack file written by ZStackFileWriter.

    Returns
    -------
    tuple[np.ndarray, dict[str, Any]]
        The (n_planes, H, W) array sorted by Z and the decoded metadata
        (keys "FrameMetaData" and "ZStack").
    """
    with tifffile.TiffFile(path) as tiff:
        description = tiff.pages[0].tags["ImageDescription"].value
        array = tiff.asarray()
    if array.ndim == 2:
        array = array[np.newaxis]
    return array, json.loads(description)


def expected_plane_count(relative_range: float, step_size: float) -> int:
    """Number of planes Automaton._execute_z_stack acquires for a range/step pair."""
    ratio = relative_range / step_size
    nearest = round(ratio)
    count = nearest if math.isclose(ratio, nearest, rel_tol=0.0, abs_tol=1e-12) else math.floor(ratio)
    return 2 * count + 1
