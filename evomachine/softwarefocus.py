from __future__ import annotations

from dataclasses import dataclass
import threading

import numpy as np
from delta.utils import CroppingBox

from evomachine.acquisition import FrameAcquisitionManager, FrameAcquisitionSettings
from evomachine.bindings.software_focus.software_focus_algorithms import (
    create_software_focus_algorithm,
)
from evomachine.config_types import Frame, SoftwareFocusConfigNew
from evomachine.coordinates import Coordinate
from evomachine.types import FocusCurveType, FocusStatusType, UNKNOWN_FOV_ID


@dataclass
class SoftwareFocusResult:
    """Result returned by one software focus run and stored for one fov."""

    previous_coordinate: Coordinate | None
    "Stage coordinate before the focus run, or None before any run."
    previous_image: np.ndarray | None
    "Image captured before the scan started, or None when unavailable."
    best_coordinate: Coordinate | None
    "Best Z coordinate found by the focus run, or None if no valid best coordinate exists."
    best_frame: np.ndarray | None
    "Frame captured at the best Z coordinate, or None when unavailable."
    focus_stack: np.ndarray
    "Image stack captured during the focus run."
    focus_scores: np.ndarray
    "Focus score for each scanned Z coordinate."
    z_coordinates: np.ndarray
    "Z coordinates scanned during the focus run."
    focus_status: FocusStatusType
    "Overall focus status."
    curve_status: FocusCurveType
    "Focus curve classification."


class SoftwareFocus:
    """Coordinate software focus scoring using a FrameAcquisitionManager."""

    def __init__(
            self,
            acquisition_manager: FrameAcquisitionManager,
            config: SoftwareFocusConfigNew,
    ):
        """
        Initialise a software focus orchestrator.

        Parameters
        ----------
        acquisition_manager
            FrameAcquisitionManager used for all camera and peripheral capture.
        config
            Default SoftwareFocusConfigNew controlling scan range and scoring.

        Returns
        -------
        None
        """
        if not isinstance(acquisition_manager, FrameAcquisitionManager):
            raise TypeError(
                f"SoftwareFocus.__init__: acquisition_manager must be FrameAcquisitionManager, "
                f"received {type(acquisition_manager)}."
            )
        if not isinstance(config, SoftwareFocusConfigNew):
            raise TypeError(f"SoftwareFocus.__init__: config must be SoftwareFocusConfigNew, received {type(config)}.")
        if acquisition_manager.stage is None:
            raise ValueError("SoftwareFocus.__init__: acquisition_manager must have a stage.")
        self.acquisition_manager: FrameAcquisitionManager = acquisition_manager
        self.default_config: SoftwareFocusConfigNew = config
        self._fov_results: dict[int, SoftwareFocusResult] = {}
        self._fov_config: dict[int, SoftwareFocusConfigNew] = {}
        self._stop_requested: bool = False

    def initialise_fovs(
            self,
            fov_ids: list[int],
            fov_configs: dict[int, SoftwareFocusConfigNew] | None = None,
    ) -> None:
        """
        Initialise empty focus results and optional configs for fovs.

        Parameters
        ----------
        fov_ids
            List of integer fov IDs to track.
        fov_configs
            Optional mapping of fov ID to fov-specific config.

        Returns
        -------
        None
        """
        if not isinstance(fov_ids, list):
            raise TypeError(
                f"SoftwareFocus.initialise_fovs: fov_ids must be list[int], received {type(fov_ids)}."
            )
        if not all(isinstance(fov_id, int) and not isinstance(fov_id, bool) for fov_id in fov_ids):
            raise TypeError("SoftwareFocus.initialise_fovs: every fov ID must be int.")
        self._fov_results = {
            fov_id: self._empty_result()
            for fov_id in fov_ids
        }
        self._fov_config = {}
        if fov_configs is None:
            return
        if not isinstance(fov_configs, dict):
            raise TypeError("SoftwareFocus.initialise_fovs: fov_configs must be dict[int, SoftwareFocusConfigNew].")
        for fov_id, config in fov_configs.items():
            if fov_id not in self._fov_results:
                raise KeyError(f"SoftwareFocus.initialise_fovs: unknown config fov ID {fov_id}.")
            self._fov_config[fov_id] = self._validate_config(config=config)

    def update_config(
            self,
            config: SoftwareFocusConfigNew,
            fov_id: int | None = None,
    ) -> None:
        """
        Replace the default or one fov-specific software focus config.

        Parameters
        ----------
        config
            Replacement SoftwareFocusConfigNew.
        fov_id
            Optional fov ID. If None, update the default config.

        Returns
        -------
        None
        """
        config = self._validate_config(config=config)
        if fov_id is None:
            self.default_config = config
            return
        if not isinstance(fov_id, int) or isinstance(fov_id, bool):
            raise TypeError(f"SoftwareFocus.update_config: fov_id must be int or None, received {type(fov_id)}.")
        self._fov_config[fov_id] = config
        self._fov_results.setdefault(fov_id, self._empty_result())

    def score_image(
            self,
            img: np.ndarray,
            config: SoftwareFocusConfigNew | None = None,
            cropping_box: CroppingBox | list[CroppingBox] | None = None,
    ) -> float:
        """
        Return a focus score for one image using a new software focus config.

        Parameters
        ----------
        img
            Image array to score.
        config
            SoftwareFocusConfigNew selecting the algorithm and parameters.
        cropping_box
            Optional crop override. If None, config.cropping_box is used.

        Returns
        -------
        float
            Focus score for the provided image or mean crop score.
        """
        config = self.default_config if config is None else self._validate_config(config=config)
        scorer = create_software_focus_algorithm(
            algorithm=config.algorithm,
            **config.algorithm_kwargs,
        )
        crop_selection = config.cropping_box if cropping_box is None else SoftwareFocusConfigNew._validate_cropping_box(
            cropping_box,
        )
        if crop_selection is None:
            return scorer.score_image(img=img)
        if isinstance(crop_selection, CroppingBox):
            return scorer.score_image(img=crop_selection.crop(img))
        return float(np.mean([scorer.score_image(img=box.crop(img)) for box in crop_selection]))

    def get_fov_result(self, fov_id: int) -> SoftwareFocusResult:
        """
        Return recorded focus result for one fov.

        Parameters
        ----------
        fov_id
            FoV ID to retrieve.

        Returns
        -------
        SoftwareFocusResult
            Recorded result for fov_id.
        """
        if fov_id not in self._fov_results:
            raise KeyError(f"SoftwareFocus.get_fov_result: unknown fov ID {fov_id}.")
        return self._fov_results[fov_id]

    def stop(self) -> None:
        """
        Request that a running software focus scan stops at the next scan boundary.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self._stop_requested = True

    def run(
            self,
            fov_id: int | None = None,
            stop_event: threading.Event | None = None,
    ) -> SoftwareFocusResult:
        """
        Run a software focus Z scan and return the result.

        Parameters
        ----------
        fov_id
            Optional tracked fov ID. If None, the stage's current fov
            ID is used when available, otherwise -1.
        stop_event
            Optional threading.Event checked between scan steps.

        Returns
        -------
        SoftwareFocusResult
            Focus run result, including scores, scanned Z coordinates, and best frame.
        """
        self._stop_requested = False
        resolved_fov_id = self._resolve_fov_id(fov_id=fov_id)
        config = self._fov_config.get(resolved_fov_id, self.default_config)
        self._fov_results.setdefault(resolved_fov_id, self._empty_result())
        stage = self.acquisition_manager.stage
        if stage is None:
            raise RuntimeError("SoftwareFocus.run: acquisition manager has no stage.")
        previous_coordinate = stage.get_coordinates(query_hardware=True)
        if previous_coordinate.z is None:
            raise RuntimeError("SoftwareFocus.run: current stage coordinate does not contain Z.")
        z_coordinates = self._make_z_coordinates(current_z=previous_coordinate.z, config=config)

        frame_metadata_items = list(config.focus_frames)
        if config.acquisition_settings is not None:
            settings = config.acquisition_settings
        else:
            settings = FrameAcquisitionSettings(
                save=False,
                normalise=False,
                illuminate_dmd=True,
                clear_dmd_after=False,
                restore_leds_after=True,
                disable_leds_after=False,
            )
        if self._stop_requested or (stop_event is not None and stop_event.is_set()):
            return self._finalise_result(
                fov_id=resolved_fov_id,
                previous_coordinate=previous_coordinate,
                previous_image=None,
                scanned_z=np.asarray([], dtype=int),
                scores=np.asarray([], dtype=float),
                focus_stack=np.empty((0,), dtype=np.float64),
                early_status=FocusStatusType.UNKNOWN,
            )

        previous_frame = self.acquisition_manager.take_frame(
            frame_metadata=frame_metadata_items,
            settings=settings,
        )
        previous_image = self._mean_frame(frame=previous_frame)
        z_stack_frame = self.acquisition_manager.take_z_stack(
            frame_metadata=frame_metadata_items,
            z_coordinates=[Coordinate(None, None, int(z_coord)) for z_coord in z_coordinates],
            settings=settings,
        )
        scores_array, stack_array = self._score_z_stack(
            frame=z_stack_frame,
            frames_per_z=len(frame_metadata_items),
            config=config,
        )
        scanned_z = z_coordinates[:scores_array.size]
        return self._finalise_result(
            fov_id=resolved_fov_id,
            previous_coordinate=previous_coordinate,
            previous_image=previous_image,
            scanned_z=scanned_z,
            scores=scores_array,
            focus_stack=stack_array,
            early_status=FocusStatusType.UNKNOWN,
        )

    @staticmethod
    def _validate_config(config: SoftwareFocusConfigNew) -> SoftwareFocusConfigNew:
        """
        Return a validated new software focus config.

        Parameters
        ----------
        config
            Candidate SoftwareFocusConfigNew.

        Returns
        -------
        SoftwareFocusConfigNew
            Validated config.
        """
        if not isinstance(config, SoftwareFocusConfigNew):
            raise TypeError(f"SoftwareFocus: config must be SoftwareFocusConfigNew, received {type(config)}.")
        return config

    def _resolve_fov_id(self, fov_id: int | None) -> int:
        """
        Return an explicit or stage-derived fov ID.

        Parameters
        ----------
        fov_id
            Optional caller-provided fov ID.

        Returns
        -------
        int
            Resolved fov ID.
        """
        if fov_id is not None:
            if not isinstance(fov_id, int) or isinstance(fov_id, bool):
                raise TypeError(f"SoftwareFocus.run: fov_id must be int or None, received {type(fov_id)}.")
            return fov_id
        stage = self.acquisition_manager.stage
        if stage is not None:
            return stage.get_fov_id()
        return UNKNOWN_FOV_ID

    @staticmethod
    def _make_z_coordinates(current_z: int | float, config: SoftwareFocusConfigNew) -> np.ndarray:
        """
        Return scan Z coordinates around the current Z coordinate.

        Parameters
        ----------
        current_z
            Current Z coordinate.
        config
            Active software focus configuration.

        Returns
        -------
        np.ndarray
            Integer Z coordinates to scan.
        """
        start = int(current_z - config.rel_range)
        stop = int(current_z + config.rel_range)
        return np.asarray(range(start, stop, config.step_size), dtype=int)

    @staticmethod
    def _empty_result() -> SoftwareFocusResult:
        """
        Return an empty software focus result.

        Parameters
        ----------
        None

        Returns
        -------
        SoftwareFocusResult
            Result object with unknown status and no captured data.
        """
        return SoftwareFocusResult(
            previous_coordinate=None,
            previous_image=None,
            best_coordinate=None,
            best_frame=None,
            focus_stack=np.empty((0,), dtype=np.float64),
            focus_scores=np.asarray([], dtype=float),
            z_coordinates=np.asarray([], dtype=int),
            focus_status=FocusStatusType.UNKNOWN,
            curve_status=FocusCurveType.UNKNOWN,
        )

    def _score_z_stack(
            self,
            frame: Frame,
            frames_per_z: int,
            config: SoftwareFocusConfigNew,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Score a captured Z stack and return per-Z scores and mean frames.

        Parameters
        ----------
        frame
            Captured frame stack from FrameAcquisitionManager.take_z_stack().
        frames_per_z
            Number of metadata captures acquired at each Z coordinate.
        config
            Active software focus configuration.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Per-Z focus scores and height-width-Z focus stack.
        """
        if frames_per_z <= 0:
            raise ValueError("SoftwareFocus._score_z_stack: frames_per_z must be positive.")
        if frame.array.shape[0] % frames_per_z != 0:
            raise ValueError("SoftwareFocus._score_z_stack: frame count is not divisible by frames_per_z.")
        focus_scores: list[float] = []
        focus_stack: list[np.ndarray] = []
        for start in range(0, frame.array.shape[0], frames_per_z):
            frame_group = frame.array[start:start + frames_per_z].astype(np.float64)
            scores = [
                self.score_image(img=frame_group[index], config=config)
                for index in range(frame_group.shape[0])
            ]
            focus_scores.append(float(np.mean(scores)))
            focus_stack.append(np.mean(frame_group, axis=0))
        return np.asarray(focus_scores, dtype=float), self._stack_frames(focus_stack=focus_stack)

    @staticmethod
    def _mean_frame(frame: Frame) -> np.ndarray:
        """
        Return the mean image from a Frame stack.

        Parameters
        ----------
        frame
            Frame object whose array should be averaged over the leading axis.

        Returns
        -------
        np.ndarray
            Mean 2D frame.
        """
        return np.mean(frame.array.astype(np.float64), axis=0)

    @staticmethod
    def _stack_frames(focus_stack: list[np.ndarray]) -> np.ndarray:
        """
        Return captured focus frames as a height-width-z stack.

        Parameters
        ----------
        focus_stack
            List of 2D frames ordered by Z coordinate.

        Returns
        -------
        np.ndarray
            3D focus stack with Z as the final axis, or an empty array.
        """
        if not focus_stack:
            return np.empty((0,), dtype=np.float64)
        return np.stack(focus_stack, axis=-1)

    def _finalise_result(
            self,
            fov_id: int,
            previous_coordinate: Coordinate,
            previous_image: np.ndarray | None,
            scanned_z: np.ndarray,
            scores: np.ndarray,
            focus_stack: np.ndarray,
            early_status: FocusStatusType,
    ) -> SoftwareFocusResult:
        """
        Classify scores, store the result, and move to the best Z when appropriate.

        Parameters
        ----------
        fov_id
            FoV ID whose stored result should be replaced.
        previous_coordinate
            Coordinate from before the focus run.
        previous_image
            Image captured before scanning, or None when no image was captured.
        scanned_z
            Z coordinates that were actually scanned.
        scores
            Focus scores for scanned_z.
        focus_stack
            Captured image stack.
        early_status
            Status set by early stopping or image capture failure.

        Returns
        -------
        SoftwareFocusResult
            Finalised focus result.
        """
        if scores.size < 3:
            curve_status = FocusCurveType.UNKNOWN
            focus_status = early_status if early_status != FocusStatusType.UNKNOWN else FocusStatusType.BAD_FOCUS_CURVE
            best_index = None
        else:
            curve_status = self._get_focus_curve_type(focus_curve=scores)
            focus_status = FocusStatusType.IN_FOCUS if curve_status == FocusCurveType.HAS_GLOBAL_MAXIMUM else FocusStatusType.BAD_FOCUS_CURVE
            best_index = int(np.argmax(scores))

        best_coordinate = None
        best_frame = None
        if best_index is not None:
            best_coordinate = previous_coordinate.copy()
            best_coordinate.z = int(scanned_z[best_index])
            best_frame = focus_stack[:, :, best_index] if focus_stack.ndim == 3 else None
            if focus_status == FocusStatusType.IN_FOCUS:
                stage = self.acquisition_manager.stage
                if stage is None:
                    raise RuntimeError("SoftwareFocus._finalise_result: acquisition manager has no stage.")
                stage.move(target=Coordinate(None, None, int(scanned_z[best_index])), block=True)

        result = SoftwareFocusResult(
            previous_coordinate=previous_coordinate.copy(),
            previous_image=previous_image,
            best_coordinate=best_coordinate,
            best_frame=best_frame,
            focus_stack=focus_stack,
            focus_scores=scores,
            z_coordinates=scanned_z,
            focus_status=focus_status,
            curve_status=curve_status,
        )
        self._fov_results[fov_id] = result
        return result

    @staticmethod
    def _get_focus_curve_type(focus_curve: np.ndarray) -> FocusCurveType:
        """
        Classify the shape of a focus score curve.

        Parameters
        ----------
        focus_curve
            Focus score values ordered by scanned Z fov.

        Returns
        -------
        FocusCurveType
            Classification of the curve maximum pattern.
        """
        if focus_curve.size < 3:
            return FocusCurveType.UNKNOWN

        max_indices = np.where(focus_curve == np.max(focus_curve))[0]
        num_maxima = len(max_indices)

        if 0 in max_indices or len(focus_curve) - 1 in max_indices:
            return FocusCurveType.HAS_BOUNDARY_MAXIMUM
        if num_maxima == 1:
            return FocusCurveType.HAS_GLOBAL_MAXIMUM
        if num_maxima > 1:
            return FocusCurveType.HAS_MAXIMA
        return FocusCurveType.UNKNOWN
