from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import ClassVar

import numpy as np
import tifffile

from evomachine.bindings.virtual.peripheralcontroller import VirtualPeripheralController
from evomachine.filemanager import FileManager
from evomachine.peripherals.camera import Camera, CameraReadoutMode
from evomachine.peripherals.camera import ImageConfigType


class VirtualImageSourceMode(str, Enum):
    """Available image sources for a virtual camera."""

    RANDOM = "random"
    FOLDER = "folder"


@dataclass
class VirtualImageSource:
    """
    Store and operate one validated virtual-camera image source.

    Attributes
    ----------
    mode
        Random generation or cyclic folder loading.
    directory
        Folder searched in folder mode, or None in random mode.
    filename_glob
        Relative glob used to snapshot candidate TIFF files.
    usable_files
        Naturally sorted TIFF paths used for cyclic acquisition.
    skipped_files
        Matched paths and reasons for excluding them from acquisition.
    next_index
        Index of the next usable image returned in folder mode.
    """

    mode: VirtualImageSourceMode = VirtualImageSourceMode.RANDOM
    directory: Path | None = None
    filename_glob: str = "*.tiff"
    usable_files: tuple[Path, ...] = ()
    skipped_files: tuple[tuple[Path, str], ...] = ()
    next_index: int = 0

    IMAGE_SHAPE: ClassVar[tuple[int, int]] = (3200, 3200)
    IMAGE_DTYPE: ClassVar[np.dtype] = np.dtype("uint16")
    TIFF_SUFFIXES: ClassVar[frozenset[str]] = frozenset({".tif", ".tiff"})

    def __post_init__(self) -> None:
        """
        Validate and normalise virtual image-source values.

        Returns
        -------
        None
        """
        if not isinstance(self.mode, VirtualImageSourceMode):
            raise TypeError(
                "VirtualImageSource: mode must be VirtualImageSourceMode, "
                f"received {type(self.mode)}."
            )
        if self.directory is not None and not isinstance(self.directory, Path | str):
            raise TypeError(
                "VirtualImageSource: directory must be Path, str, or None, "
                f"received {type(self.directory)}."
            )
        if isinstance(self.directory, str):
            self.directory = Path(self.directory).expanduser()
        if not isinstance(self.filename_glob, str):
            raise TypeError(
                "VirtualImageSource: filename_glob must be str, "
                f"received {type(self.filename_glob)}."
            )
        if not self.filename_glob:
            raise ValueError("VirtualImageSource: filename_glob must not be empty.")
        if Path(self.filename_glob).is_absolute():
            raise ValueError("VirtualImageSource: filename_glob must be relative.")
        if self.mode is VirtualImageSourceMode.FOLDER and self.directory is None:
            raise ValueError("VirtualImageSource: folder mode requires a directory.")
        if not isinstance(self.next_index, int) or isinstance(self.next_index, bool):
            raise TypeError("VirtualImageSource: next_index must be int.")
        if self.next_index < 0 or (self.usable_files and self.next_index >= len(self.usable_files)):
            raise ValueError("VirtualImageSource: next_index is outside the usable file sequence.")

    @classmethod
    def from_folder(cls, directory: Path | str, filename_glob: str = "*.tiff") -> VirtualImageSource:
        """
        Build a validated folder source without changing an active camera.

        Parameters
        ----------
        directory
            Folder containing candidate TIFF files.
        filename_glob
            Relative glob used to select candidate files.

        Returns
        -------
        VirtualImageSource
            Folder source containing usable and skipped file snapshots.
        """
        candidate = cls(
            mode=VirtualImageSourceMode.FOLDER,
            directory=directory,
            filename_glob=filename_glob,
        )
        matches = FileManager.list_filenames(
            directory=candidate.directory,
            filename_pattern=candidate.filename_glob,
        )
        if not matches:
            raise ValueError(
                f"VirtualImageSource: no files match {candidate.filename_glob!r} in {candidate.directory}."
            )
        usable_files: list[Path] = []
        skipped_files: list[tuple[Path, str]] = []
        for path in matches:
            reason = cls.rejection_reason(path=path)
            if reason is None:
                usable_files.append(path)
            else:
                skipped_files.append((path, reason))
        if not usable_files:
            raise ValueError("VirtualImageSource: no matched files are usable 3200 x 3200 uint16 TIFF images.")
        candidate.usable_files = tuple(usable_files)
        candidate.skipped_files = tuple(skipped_files)
        return candidate

    @property
    def next_filename(self) -> Path | None:
        """
        Return the next folder image.

        Returns
        -------
        Path | None
            Next source path, or None when random mode is active.
        """
        if not self.usable_files:
            return None
        return self.usable_files[self.next_index]

    def load_next(self) -> np.ndarray:
        """
        Load and advance past the next folder image.

        Returns
        -------
        np.ndarray
            Next compatible image in the cyclic source sequence.
        """
        path = self.next_filename
        if self.mode is not VirtualImageSourceMode.FOLDER or path is None:
            raise RuntimeError("VirtualImageSource.load_next requires a non-empty folder source.")
        try:
            frame = np.asarray(tifffile.imread(path))
        except Exception as error:
            raise RuntimeError(f"VirtualImageSource: failed to load source image {path}: {error}") from error
        reason = self.rejection_reason(path=path, image=frame)
        if reason is not None:
            raise RuntimeError(f"VirtualImageSource: source image {path} is no longer usable: {reason}")
        self.next_index = (self.next_index + 1) % len(self.usable_files)
        return frame

    @classmethod
    def rejection_reason(cls, path: Path, image: np.ndarray | None = None) -> str | None:
        """
        Return why a source file is unusable, or None when it is valid.

        Parameters
        ----------
        path
            Candidate source image path.
        image
            Optional already-loaded array used for runtime revalidation.

        Returns
        -------
        str | None
            Rejection reason, or None for a compatible TIFF image.
        """
        if path.suffix.casefold() not in cls.TIFF_SUFFIXES:
            return "only .tif and .tiff files are supported"
        try:
            if image is None:
                with tifffile.TiffFile(path) as tiff:
                    if len(tiff.pages) != 1:
                        return "TIFF stacks are not supported"
                    shape = tuple(tiff.series[0].shape)
                    dtype = np.dtype(tiff.series[0].dtype)
            else:
                shape = image.shape
                dtype = image.dtype
        except Exception as error:
            return f"cannot read TIFF metadata: {type(error).__name__}: {error}"
        if shape != cls.IMAGE_SHAPE:
            return f"expected shape {cls.IMAGE_SHAPE}, received {shape}"
        if dtype != cls.IMAGE_DTYPE:
            return f"expected dtype {cls.IMAGE_DTYPE}, received {dtype}"
        return None


class VirtualCamera(Camera):
    """
    In-memory camera implementation for debugging without camera hardware.

    VirtualCamera uses the same public Camera API as hardware-backed cameras,
    but all frames are generated locally.
    """

    DEFAULT_NAME = "Virtual Camera"

    def __init__(
            self,
            peripheral_ctrl: VirtualPeripheralController,
            image: ImageConfigType,
            name: str = DEFAULT_NAME,
            default_exposure_time: float | int = 200,
            readout_mode: CameraReadoutMode | None = None,
            check_initialised: bool = True,
            check_alive: bool = True,
            random_seed: int | None = 0,
    ):
        """
        Initialise an in-memory virtual camera.

        Parameters
        ----------
        peripheral_ctrl
            VirtualPeripheralController that owns the simulated lifecycle.
        image
            Image shape and dtype returned by generated frames.
        name
            Human-readable camera name.
        default_exposure_time
            Exposure time applied during initialise(), in milliseconds.
        readout_mode
            Optional simulated readout mode.
        check_initialised
            If True, inherited public methods require successful initialise().
        check_alive
            If True, inherited public methods require the camera to report alive.
        random_seed
            Seed used for deterministic pseudo-random frame generation. If
            None, NumPy chooses a non-deterministic seed.

        Returns
        -------
        None
        """
        if not isinstance(peripheral_ctrl, VirtualPeripheralController):
            raise TypeError(
                f"VirtualCamera.__init__: peripheral_ctrl must be VirtualPeripheralController, "
                f"received {type(peripheral_ctrl)}."
            )
        self.peripheral_ctrl: VirtualPeripheralController = peripheral_ctrl
        self.random_seed: int | None = random_seed
        self._rng: np.random.Generator = np.random.default_rng(random_seed)
        self._stop_was_called: bool = False
        self._image_source = VirtualImageSource()
        super().__init__(
            image=image,
            name=name,
            default_exposure_time=default_exposure_time,
            readout_mode=readout_mode,
            check_initialised=check_initialised,
            check_alive=check_alive,
        )

    def stop_was_called(self) -> bool:
        """
        Return whether stop has been called.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            True after stop has been called at least once.
        """
        return self._stop_was_called

    def _initialise(self, force: bool = False) -> bool:
        """
        Mark the virtual camera as initialised.

        Parameters
        ----------
        force
            Present for API compatibility. It does not change VirtualCamera
            behavior.

        Returns
        -------
        bool
            True when the virtual peripheral controller is alive.
        """
        return self.peripheral_ctrl.is_alive()

    def _finalise(self, force: bool = False) -> None:
        """
        Finalise the virtual camera.

        Parameters
        ----------
        force
            Present for API compatibility. It does not change VirtualCamera
            behavior.

        Returns
        -------
        None
        """
        return

    def _check_is_alive(self) -> bool:
        """
        Report whether the virtual controller is alive.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            True when the virtual peripheral controller is alive.
        """
        return self.peripheral_ctrl.is_alive()

    def _stop(self) -> None:
        """
        Mark that stop was called.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.peripheral_ctrl.stop()
        self._stop_was_called = True

    def _get_frame(self) -> np.ndarray:
        """
        Return one frame from the active random or folder source.

        Parameters
        ----------
        None

        Returns
        -------
        np.ndarray
            Generated or loaded frame with the configured shape and dtype.
        """
        if self._image_source.mode is VirtualImageSourceMode.FOLDER:
            return self._image_source.load_next()
        if np.issubdtype(self.image.pxl_dtype, np.integer):
            dtype_info = np.iinfo(self.image.pxl_dtype)
            high = min(dtype_info.max, 65535) + 1
            return self._rng.integers(0, high, size=self.image.shape, dtype=self.image.pxl_dtype)
        return self._rng.random(size=self.image.shape).astype(self.image.pxl_dtype)

    def configure_image_source(self, source: VirtualImageSource) -> VirtualImageSource:
        """
        Atomically replace the source used for subsequent virtual frames.

        Parameters
        ----------
        source
            Validated random or folder source object.

        Returns
        -------
        VirtualImageSource
            Status for the newly active source.
        """
        if not isinstance(source, VirtualImageSource):
            raise TypeError(
                "VirtualCamera.configure_image_source: source must be VirtualImageSource, "
                f"received {type(source)}."
            )
        if source.mode is VirtualImageSourceMode.FOLDER:
            if self.image.shape != source.IMAGE_SHAPE or self.image.pxl_dtype != source.IMAGE_DTYPE:
                raise ValueError(
                    "VirtualCamera: folder mode requires a 3200 x 3200 uint16 camera configuration."
                )
            if not source.usable_files:
                raise ValueError("VirtualCamera: folder source must contain at least one usable image.")
        self._image_source = replace(source, next_index=0)
        return self.image_source_status()

    def image_source_status(self) -> VirtualImageSource:
        """
        Return the active virtual image-source state.

        Returns
        -------
        VirtualImageSource
            Copy of the source configuration, files, and next index.
        """
        return replace(self._image_source)

    def _set_exposure(self, exposure_time: float | int) -> None:
        """
        Accept a simulated exposure time.

        Parameters
        ----------
        exposure_time
            Positive exposure time in milliseconds.

        Returns
        -------
        None
        """
        return
