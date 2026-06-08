from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

import numpy as np

from evomachine.peripherals.camera import Camera
from evomachine.coordinates import Coordinate
from evomachine.frame import Frame, FrameMetaData
from evomachine.peripherals.dmd import Dmd
from evomachine.filemanager import FileManager
from evomachine.peripherals.filterwheel import FilterWheel
from evomachine.peripherals.leds import LedManager, LedState
from evomachine.peripherals.stage import Stage
from evomachine.types import LEDType


@dataclass
class FrameAcquisitionSettings:
    """Runtime options controlling how frame acquisition uses peripherals."""

    save: bool = False
    "Save acquired frames with the configured FileManager."
    normalise: bool = False
    "Return camera-normalised image data when the camera supports it."
    illuminate_dmd: bool = True
    "Display DMD illumination before each captured frame when a DMD is available."
    clear_dmd_after: bool = False
    "Blank the DMD after the acquisition call completes."
    restore_leds_after: bool = True
    "Restore cached LED states after the acquisition call completes."
    disable_leds_after: bool = False
    "Disable all LEDs after the acquisition call completes."

    def __post_init__(self) -> None:
        for field_name in ("save", "normalise", "illuminate_dmd", "clear_dmd_after", "restore_leds_after", "disable_leds_after"):
            value = getattr(self, field_name)
            if not isinstance(value, bool):
                raise TypeError(f"FrameAcquisitionSettings: {field_name} must be bool, received {type(value)}.")


class FrameAcquisitionManager:
    """Coordinate low-level frame capture through camera and imaging peripherals."""

    def __init__(
            self,
            camera: Camera,
            led_manager: LedManager,
            filter_wheel: FilterWheel | None = None,
            dmd: Dmd | None = None,
            file_manager: FileManager | None = None,
            stage: Stage | None = None,
            default_settings: FrameAcquisitionSettings | None = None,
    ):
        """
        Initialise a frame acquisition manager.

        Parameters
        ----------
        camera
            Camera used to capture image data.
        led_manager
            LED manager used to actuate illumination.
        filter_wheel
            Optional filter wheel used when FrameMetaData contains a filter setting.
        dmd
            Optional DMD used for full or image illumination.
        file_manager
            Optional file manager used when acquisition settings request saving.
        stage
            Optional stage used by Z-stack acquisition.
        default_settings
            Optional default settings used when take_frame() and take_z_stack()
            are called without per-call settings.

        Returns
        -------
        None
        """
        self.camera = camera
        self.led_manager = led_manager
        self.filter_wheel = filter_wheel
        self.dmd = dmd
        self.file_manager = file_manager
        self.stage = stage
        self.default_settings = self._validate_settings(settings=default_settings)

    def update_settings(
            self,
            settings: FrameAcquisitionSettings | None = None,
            **updates,
    ) -> FrameAcquisitionSettings:
        """
        Replace or update the default frame acquisition settings.

        Parameters
        ----------
        settings
            Optional complete replacement settings. If None, updates are applied
            to the current default settings.
        **updates
            FrameAcquisitionSettings fields to update on the current defaults.

        Returns
        -------
        FrameAcquisitionSettings
            The new default settings object.
        """
        if settings is not None and updates:
            raise ValueError("FrameAcquisitionManager.update_settings: provide settings or updates, not both.")
        if settings is not None:
            self.default_settings = self._validate_settings(settings=settings)
            return self.default_settings
        try:
            self.default_settings = replace(self.default_settings, **updates)
        except TypeError as error:
            raise ValueError("FrameAcquisitionManager.update_settings: unknown settings field.") from error
        return self.default_settings

    def set_camera_live_mode(self, status: bool = False) -> None:
        """
        Set camera live mode when the camera exposes a compatible method.

        Parameters
        ----------
        status
            Desired live-mode state.

        Returns
        -------
        None
        """
        if not isinstance(status, bool):
            raise TypeError(
                f"FrameAcquisitionManager.set_camera_live_mode: status must be bool, received {type(status)}."
            )
        method_name = "enable_live_mode" if status else "disable_live_mode"
        live_mode = getattr(self.camera, method_name, None)
        if callable(live_mode):
            live_mode()

    def take_frame(
            self,
            frame_metadata: FrameMetaData | list[FrameMetaData],
            settings: FrameAcquisitionSettings | None = None,
    ) -> Frame:
        """
        Acquire one or more frames from the configured peripherals.

        Parameters
        ----------
        frame_metadata
            One FrameMetaData object or a list of FrameMetaData objects to
            acquire in order.
        settings
            Optional runtime acquisition settings. When omitted, the manager's
            default settings are used. When supplied, settings replace defaults
            for this call only.

        Returns
        -------
        Frame
            Acquired image stack, metadata, and optional saved paths.
        """
        metadata_items = self._normalise_frame_metadata(frame_metadata=frame_metadata)
        self._validate_frame_metadata_fov(frame_metadata=metadata_items)
        settings = self._normalise_settings(settings=settings)
        previous_led_states = self._capture_led_states() if settings.restore_leds_after else {}
        frames: list[np.ndarray] = []
        saved_paths: list[Path | None] = []
        # Cleanup must run even when DMD display, LED/filter setup, camera
        # capture, or saving raises.
        try:
            for metadata in metadata_items:
                frame, saved_path = self._acquire_single_frame(frame_metadata=metadata, settings=settings)
                frames.append(frame)
                saved_paths.append(saved_path)
        finally:
            self._cleanup(settings=settings, previous_led_states=previous_led_states)
        if not frames:
            raise RuntimeError("FrameAcquisitionManager.take_frame: no frames were acquired.")
        return Frame(
            frame_metadata=metadata_items,
            array=np.stack(frames, axis=0),
            saved_paths=saved_paths,
        )

    def take_z_stack(
            self,
            frame_metadata: FrameMetaData | list[FrameMetaData],
            z_coordinates: list[Coordinate],
            settings: FrameAcquisitionSettings | None = None,
    ) -> Frame:
        """
        Acquire frames at a sequence of Z coordinates and restore the original Z.

        Parameters
        ----------
        frame_metadata
            One FrameMetaData object or a list captured at each Z coordinate.
        z_coordinates
            List of Z-only Coordinates with x=None, y=None, and z set.
        settings
            Optional runtime acquisition settings. When omitted, the manager's
            default settings are used. When supplied, settings replace defaults
            for this call only.

        Returns
        -------
        Frame
            Acquired Z-stack as one leading-axis frame stack.
        """
        if self.stage is None:
            raise RuntimeError("FrameAcquisitionManager.take_z_stack: stage is required for Z-stack acquisition.")
        metadata_items = self._normalise_frame_metadata(frame_metadata=frame_metadata)
        self._validate_frame_metadata_fov(frame_metadata=metadata_items)
        z_coordinates = self._validate_z_coordinates(z_coordinates=z_coordinates)
        settings = self._normalise_settings(settings=settings)
        previous_coordinate = self.stage.get_coordinates(query_hardware=True)
        if previous_coordinate.z is None:
            raise RuntimeError("FrameAcquisitionManager.take_z_stack: current stage coordinate does not contain Z.")
        captured_metadata: list[FrameMetaData] = []
        frames: list[np.ndarray] = []
        saved_paths: list[Path | None] = []
        try:
            for z_coordinate in z_coordinates:
                self.stage.move(target=z_coordinate, block=True)
                stack_coordinate = previous_coordinate.copy()
                stack_coordinate.z = z_coordinate.z
                stack_metadata = [
                    replace(metadata, coordinate=stack_coordinate.copy())
                    for metadata in metadata_items
                ]
                frame = self.take_frame(frame_metadata=stack_metadata, settings=settings)
                captured_metadata.extend(frame.frame_metadata)
                frames.extend(frame.array[index] for index in range(frame.array.shape[0]))
                saved_paths.extend(frame.saved_paths)
        finally:
            self.stage.move(target=Coordinate(None, None, previous_coordinate.z), block=True)
        if not frames:
            raise RuntimeError("FrameAcquisitionManager.take_z_stack: no frames were acquired.")
        return Frame(
            frame_metadata=captured_metadata,
            array=np.stack(frames, axis=0),
            saved_paths=saved_paths,
        )

    def stop(self) -> None:
        """
        Stop acquisition-related peripherals.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.led_manager.disable_led()
        if self.dmd is not None:
            self.dmd.display_none()
        self.camera.stop()
        if self.stage is not None:
            self.stage.stop()

    @staticmethod
    def _normalise_frame_metadata(frame_metadata: FrameMetaData | list[FrameMetaData]) -> list[FrameMetaData]:
        """
        Return frame metadata as a validated non-empty list.

        Parameters
        ----------
        frame_metadata
            One FrameMetaData object or a list of them.

        Returns
        -------
        list[FrameMetaData]
            Validated frame metadata list.
        """
        if isinstance(frame_metadata, FrameMetaData):
            return [frame_metadata]
        if not isinstance(frame_metadata, list):
            raise TypeError(
                f"FrameAcquisitionManager: frame_metadata must be FrameMetaData or list[FrameMetaData], "
                f"received {type(frame_metadata)}."
            )
        if not frame_metadata:
            raise ValueError("FrameAcquisitionManager: frame_metadata list must not be empty.")
        if not all(isinstance(metadata, FrameMetaData) for metadata in frame_metadata):
            raise TypeError("FrameAcquisitionManager: every frame_metadata entry must be FrameMetaData.")
        return list(frame_metadata)

    @staticmethod
    def _validate_frame_metadata_fov(frame_metadata: list[FrameMetaData]) -> int:
        """
        Return the single FoV ID represented by a frame metadata list.

        Parameters
        ----------
        frame_metadata
            Validated non-empty metadata list.

        Returns
        -------
        int
            Shared FoV ID.
        """
        fov_ids = {metadata.fov_id for metadata in frame_metadata}
        if len(fov_ids) > 1:
            raise ValueError(
                "FrameAcquisitionManager: all frame_metadata entries must have the same fov_id, "
                f"received {sorted(fov_ids)}."
            )
        return next(iter(fov_ids))

    def _normalise_settings(self, settings: FrameAcquisitionSettings | None) -> FrameAcquisitionSettings:
        """
        Return per-call settings or manager defaults when omitted.

        Parameters
        ----------
        settings
            Optional FrameAcquisitionSettings.

        Returns
        -------
        FrameAcquisitionSettings
            Validated settings object for this call.
        """
        if settings is None:
            return self.default_settings
        return self._validate_settings(settings=settings)

    @staticmethod
    def _validate_settings(settings: FrameAcquisitionSettings | None) -> FrameAcquisitionSettings:
        """
        Validate a settings object or create defaults when omitted.

        Parameters
        ----------
        settings
            Optional FrameAcquisitionSettings object.

        Returns
        -------
        FrameAcquisitionSettings
            Validated settings object.
        """
        if settings is None:
            return FrameAcquisitionSettings()
        if not isinstance(settings, FrameAcquisitionSettings):
            raise TypeError(
                f"FrameAcquisitionManager: settings must be FrameAcquisitionSettings or None, "
                f"received {type(settings)}."
            )
        return settings

    @staticmethod
    def _validate_z_coordinates(z_coordinates: list[Coordinate]) -> list[Coordinate]:
        """
        Return a validated copy of Z-only coordinates.

        Parameters
        ----------
        z_coordinates
            Candidate list of Z-only Coordinates.

        Returns
        -------
        list[Coordinate]
            Copied Coordinate objects.
        """
        if not isinstance(z_coordinates, list):
            raise TypeError(f"FrameAcquisitionManager.take_z_stack: z_coordinates must be list[Coordinate], received {type(z_coordinates)}.")
        if not z_coordinates:
            raise ValueError("FrameAcquisitionManager.take_z_stack: z_coordinates must not be empty.")
        copied_coordinates: list[Coordinate] = []
        for coordinate in z_coordinates:
            if not isinstance(coordinate, Coordinate):
                raise TypeError("FrameAcquisitionManager.take_z_stack: every z coordinate must be Coordinate.")
            if coordinate.x is not None or coordinate.y is not None or coordinate.z is None:
                raise ValueError("FrameAcquisitionManager.take_z_stack: coordinates must have x=None, y=None, and z set.")
            copied_coordinates.append(coordinate.copy())
        return copied_coordinates

    def _acquire_single_frame(
            self,
            frame_metadata: FrameMetaData,
            settings: FrameAcquisitionSettings,
    ) -> tuple[np.ndarray, Path | None]:
        """
        Acquire one frame for one metadata entry.

        Parameters
        ----------
        frame_metadata
            Metadata and hardware settings for the frame.
        settings
            Runtime acquisition settings.

        Returns
        -------
        tuple[np.ndarray, Path | None]
            Captured image and optional saved path.
        """
        if self.dmd is not None and settings.illuminate_dmd:
            if frame_metadata.dmd_pattern is None:
                self.dmd.display_full()
            else:
                self.dmd.display_image(frame_metadata.dmd_pattern)
        if frame_metadata.filter_wheel is not None:
            if self.filter_wheel is None:
                raise RuntimeError(
                    "FrameAcquisitionManager: frame metadata requested filter wheel control but no filter wheel was provided."
                )
            self.filter_wheel.set_filter_wheel(filter_type=frame_metadata.filter_wheel)
        if frame_metadata.exposure is not None:
            self.camera.set_exposure(frame_metadata.exposure)
        if frame_metadata.leds is not None:
            for led_type, brightness in frame_metadata.leds.items():
                self.led_manager.set_led(led_type=led_type, brightness=brightness)
        frame_metadata.execution_time = datetime.now()
        frame = self.camera.get_frame(normalise=settings.normalise)
        if not settings.save:
            return frame, None
        if self.file_manager is None:
            raise RuntimeError("FrameAcquisitionManager: saving requested but no file manager was provided.")
        saved_path = self.file_manager.save_frame(frame=frame, frame_metadata=frame_metadata)
        return frame, saved_path

    def _capture_led_states(self) -> dict[LEDType, LedState]:
        """
        Return cached LED states when the LED manager exposes state readback.

        Parameters
        ----------
        None

        Returns
        -------
        dict[LEDType, LedState]
            Cached state by LED type, or an empty mapping when unavailable.
        """
        get_available_leds = getattr(self.led_manager, "get_available_leds", None)
        get_led_state = getattr(self.led_manager, "get_led_state", None)
        if not callable(get_available_leds) or not callable(get_led_state):
            return {}
        return {
            led_type: get_led_state(led_type)
            for led_type in get_available_leds()
        }

    def _cleanup(
            self,
            settings: FrameAcquisitionSettings,
            previous_led_states: dict[LEDType, LedState],
    ) -> None:
        """
        Clean up DMD and LED state after acquisition.

        Parameters
        ----------
        settings
            Runtime acquisition settings.
        previous_led_states
            Cached LED states captured before acquisition.

        Returns
        -------
        None
        """
        if settings.clear_dmd_after and self.dmd is not None:
            self.dmd.display_none()
        if settings.disable_leds_after:
            self.led_manager.disable_led()
            return
        if settings.restore_leds_after:
            self._restore_led_states(previous_led_states=previous_led_states)

    def _restore_led_states(self, previous_led_states: dict[LEDType, LedState]) -> None:
        """
        Restore cached LED states or disable all LEDs when no state was captured.

        Parameters
        ----------
        previous_led_states
            Cached LED states by LED type.

        Returns
        -------
        None
        """
        if not previous_led_states:
            self.led_manager.disable_led()
            return
        for led_type, state in previous_led_states.items():
            if state.is_on:
                self.led_manager.set_led(led_type=led_type, brightness=state.brightness)
            else:
                self.led_manager.disable_led(led_type=led_type)
