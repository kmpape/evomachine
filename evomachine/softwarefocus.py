from __future__ import annotations

from dataclasses import dataclass
import threading

import numpy as np
from delta.utils import CroppingBox

from evomachine.bindings.software_focus.software_focus_algorithms import (
    DEFAULT_SQUARED_GRAD_THRESHOLD,
    LaplacianVarianceFocusAlgorithm,
    SoftwareFocusAlgorithm,
    SquaredGradientAverageFocusAlgorithm,
    SteelFocusAlgorithm,
    create_software_focus_algorithm as _create_software_focus_algorithm,
)
from evomachine.acquisition import FrameAcquisitionManager, FrameAcquisitionSettings
from evomachine.config_types import Frame, FrameMetaData, SoftwareFocusConfig
from evomachine.coordinates import Coordinate
from evomachine.types import FocusAlgorithmType, FocusCurveType, FocusStatusType, LEDType


# TODO(Codex): can remove positio_id from state
@dataclass
class SoftwareFocusPositionState:
    """State recorded for one software focus position."""

    position_id: int
    "Position identifier associated with the focus run."
    previous_coordinate: Coordinate | None = None
    "Stage coordinate before the most recent focus run."
    z_coordinates: np.ndarray | None = None
    "Z coordinates scanned during the most recent focus run."
    focus_scores: np.ndarray | None = None
    "Focus score for each scanned Z coordinate."
    focus_stack: np.ndarray | None = None
    "Image stack captured during the most recent focus run."
    previous_image: np.ndarray | None = None
    "Image captured before the scan started."
    status: FocusStatusType = FocusStatusType.UNKNOWN
    "Software focus status for the most recent focus run."
    curve_status: FocusCurveType = FocusCurveType.UNKNOWN
    "Focus curve classification for the most recent focus run."


@dataclass
class SoftwareFocusResult:
    """Result returned by one software focus run."""

    best_coordinate: Coordinate | None
    "Best Z coordinate found by the focus run, or None if no valid best coordinate exists."
    best_frame: np.ndarray | None
    "Frame captured at the best Z coordinate, or None when unavailable."
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
            config: SoftwareFocusConfig,
            autofocus=None,
    ):
        """
        Initialise a software focus orchestrator.

        Parameters
        ----------
        acquisition_manager
            FrameAcquisitionManager used for all camera and peripheral capture.
        config
            SoftwareFocusConfig controlling scan range and scoring.
        autofocus
            Optional autofocus-like object exposing unlock().

        Returns
        -------
        None
        """
        if not isinstance(acquisition_manager, FrameAcquisitionManager):
            raise TypeError(
                f"SoftwareFocus.__init__: acquisition_manager must be FrameAcquisitionManager, "
                f"received {type(acquisition_manager)}."
            )
        if not isinstance(config, SoftwareFocusConfig):
            raise TypeError(f"SoftwareFocus.__init__: config must be SoftwareFocusConfig, received {type(config)}.")
        if acquisition_manager.stage is None:
            raise ValueError("SoftwareFocus.__init__: acquisition_manager must have a stage.")
        self.acquisition_manager = acquisition_manager
        self.autofocus = autofocus
        self.config: SoftwareFocusConfig = config
        self._position_states: dict[int, SoftwareFocusPositionState] = {}
        self._stop_requested: bool = False

    @property
    def stage(self):
        """
        Return the stage owned by the acquisition manager.

        Parameters
        ----------
        None

        Returns
        -------
        Stage-like object
            Stage used by the acquisition manager.
        """
        if self.acquisition_manager.stage is None:
            raise RuntimeError("SoftwareFocus.stage: acquisition manager has no stage.")
        return self.acquisition_manager.stage

    def initialise_positions(self, position_ids: list[int]) -> None:
        """
        Initialise empty focus state for each requested position ID.

        Parameters
        ----------
        position_ids
            List of integer position IDs to track.

        Returns
        -------
        None
        """
        if not isinstance(position_ids, list):
            raise TypeError(f"SoftwareFocus.initialise_positions: position_ids must be list[int], received {type(position_ids)}.")
        if not all(isinstance(position_id, int) and not isinstance(position_id, bool) for position_id in position_ids):
            raise TypeError("SoftwareFocus.initialise_positions: every position ID must be int.")
        self._position_states = {
            position_id: SoftwareFocusPositionState(position_id=position_id)
            for position_id in position_ids
        }

    def update_config(self, config: SoftwareFocusConfig | None = None, **updates) -> None:
        """
        Replace or update the active software focus configuration.

        Parameters
        ----------
        config
            Optional replacement SoftwareFocusConfig.
        **updates
            SoftwareFocusConfig field values to update when config is None.

        Returns
        -------
        None
        """
        if config is not None and updates:
            raise ValueError("SoftwareFocus.update_config: provide config or updates, not both.")
        if config is not None:
            if not isinstance(config, SoftwareFocusConfig):
                raise TypeError(f"SoftwareFocus.update_config: config must be SoftwareFocusConfig, received {type(config)}.")
            self.config = config
            return
        self.config = self.config.updated(**updates)

    def score_image(
            self,
            img: np.ndarray,
            algorithm: FocusAlgorithmType | None = None,
            threshold: float | None = None,
            rowshift: int | None = None,
            colshift: int | None = None,
            normalise_score: bool = False,
    ) -> float:
        """
        Return a focus score for one image using this object's configuration.

        Parameters
        ----------
        img
            Image array to score.
        algorithm
            Optional focus algorithm override. If None, self.config.algorithm is used.
        threshold
            Optional squared-gradient threshold.
        rowshift
            Optional row shift override.
        colshift
            Optional column shift override.
        normalise_score
            If True, normalise the Steel score by image area.

        Returns
        -------
        float
            Focus score for the provided image.
        """
        return get_focus_score(
            img=img,
            algorithm=self.config.algorithm if algorithm is None else algorithm,
            threshold=threshold,
            rowshift=self.config.rowshift_px if rowshift is None else rowshift,
            colshift=self.config.colshift_px if colshift is None else colshift,
            normalise_score=normalise_score,
            config=self.config,
        )

    def score_rois(
            self,
            img: np.ndarray,
            boxes: list[CroppingBox],
            algorithm: FocusAlgorithmType | None = None,
            threshold: float | None = None,
            rowshift: int | None = None,
            colshift: int | None = None,
            normalise_score: bool = False,
    ) -> float:
        """
        Return summed ROI focus scores using this object's configuration.

        Parameters
        ----------
        img
            Image array containing all regions.
        boxes
            CroppingBox objects selecting regions to score.
        algorithm
            Optional focus algorithm override. If None, self.config.algorithm is used.
        threshold
            Optional squared-gradient threshold.
        rowshift
            Optional row shift override.
        colshift
            Optional column shift override.
        normalise_score
            If True, normalise the Steel score by image area.

        Returns
        -------
        float
            Summed ROI focus score.
        """
        return get_roi_focus_score(
            img=img,
            algorithm=self.config.algorithm if algorithm is None else algorithm,
            boxes=boxes,
            threshold=threshold,
            rowshift=self.config.rowshift_px if rowshift is None else rowshift,
            colshift=self.config.colshift_px if colshift is None else colshift,
            normalise_score=normalise_score,
            config=self.config,
        )

    def get_position_state(self, position_id: int) -> SoftwareFocusPositionState:
        """
        Return recorded focus state for one position.

        Parameters
        ----------
        position_id
            Position ID to retrieve.

        Returns
        -------
        SoftwareFocusPositionState
            Recorded state for position_id.
        """
        if position_id not in self._position_states:
            raise KeyError(f"SoftwareFocus.get_position_state: unknown position ID {position_id}.")
        return self._position_states[position_id]

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
            position_id: int | None = None,
            stop_event: threading.Event | None = None,
    ) -> SoftwareFocusResult:
        """
        Run a software focus Z scan and return the result.

        Parameters
        ----------
        position_id
            Optional tracked position ID. If None, the stage's current position
            ID is used when available, otherwise -1.
        stop_event
            Optional threading.Event checked between scan steps.

        Returns
        -------
        SoftwareFocusResult
            Focus run result, including scores, scanned Z coordinates, and best frame.
        """
        self._stop_requested = False
        resolved_position_id = self._resolve_position_id(position_id=position_id)
        state = self._position_states.setdefault(
            resolved_position_id,
            SoftwareFocusPositionState(position_id=resolved_position_id),
        )
        previous_coordinate = self.stage.get_coordinates(query_hardware=True)
        if previous_coordinate.z is None:
            raise RuntimeError("SoftwareFocus.run: current stage coordinate does not contain Z.")
        state.previous_coordinate = previous_coordinate.copy()
        z_coordinates = self._make_z_coordinates(current_z=previous_coordinate.z)
        state.z_coordinates = z_coordinates

        self._prepare_hardware()
        frame_metadata_items = self._focus_frame_metadata_items(position_id=resolved_position_id)
        settings = self._focus_acquisition_settings()
        if self._should_stop(stop_event=stop_event):
            result = self._finalise_result(
                state=state,
                previous_coordinate=previous_coordinate,
                scanned_z=np.asarray([], dtype=int),
                scores=np.asarray([], dtype=float),
                focus_stack=np.empty((0,), dtype=np.float64),
                early_status=FocusStatusType.UNKNOWN,
            )
            return result

        previous_frame = self.acquisition_manager.take_frame(
            frame_metadata=frame_metadata_items,
            settings=settings,
        )
        state.previous_image = self._mean_frame(frame=previous_frame)
        z_stack_frame = self.acquisition_manager.take_z_stack(
            frame_metadata=frame_metadata_items,
            z_coordinates=[Coordinate(None, None, int(z_coord)) for z_coord in z_coordinates],
            settings=settings,
        )
        scores_array, stack_array = self._score_z_stack(
            frame=z_stack_frame,
            frames_per_z=len(frame_metadata_items),
        )
        scanned_z = z_coordinates[:scores_array.size]
        result = self._finalise_result(
            state=state,
            previous_coordinate=previous_coordinate,
            scanned_z=scanned_z,
            scores=scores_array,
            focus_stack=stack_array,
            early_status=FocusStatusType.UNKNOWN,
        )
        return result

    def _resolve_position_id(self, position_id: int | None) -> int:
        """
        Return an explicit or stage-derived position ID.

        Parameters
        ----------
        position_id
            Optional caller-provided position ID.

        Returns
        -------
        int
            Resolved position ID.
        """
        if position_id is not None:
            if not isinstance(position_id, int) or isinstance(position_id, bool):
                raise TypeError(f"SoftwareFocus.run: position_id must be int or None, received {type(position_id)}.")
            return position_id
        get_pos = getattr(self.stage, "get_pos", None)
        if callable(get_pos):
            return int(get_pos())
        return -1

    def _make_z_coordinates(self, current_z: int | float) -> np.ndarray:
        """
        Return scan Z coordinates around the current Z position.

        Parameters
        ----------
        current_z
            Current Z coordinate.

        Returns
        -------
        np.ndarray
            Integer Z coordinates to scan.
        """
        start = int(current_z - self.config.rel_range)
        stop = int(current_z + self.config.rel_range)
        return np.asarray(range(start, stop, self.config.step_size), dtype=int)

    def _prepare_hardware(self) -> None:
        """
        Prepare optional autofocus before scanning.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if self.autofocus is not None:
            self.autofocus.unlock()

    def _should_stop(self, stop_event: threading.Event | None) -> bool:
        """
        Return whether scanning should stop.

        Parameters
        ----------
        stop_event
            Optional external stop event.

        Returns
        -------
        bool
            True when an internal or external stop request is active.
        """
        return self._stop_requested or (stop_event is not None and stop_event.is_set())

    def _focus_frame_metadata_items(self, position_id: int) -> list[FrameMetaData]:
        """
        Return focus frame metadata entries as a list.

        Parameters
        ----------
        position_id
            Position ID to attach to generated legacy focus metadata.

        Returns
        -------
        list[FrameMetaData]
            Configured frame metadata entries or one generated legacy entry.
        """
        focus_frames = self.config.focus_frames
        if focus_frames is None:
            return [
                FrameMetaData(
                    frame_id=-1,
                    leds={self.config.focus_channel: self.config.brightness},
                    filter_wheel=None,
                    exposure=self.config.exposure_time,
                    position_id=position_id,
                )
            ]
        if isinstance(focus_frames, FrameMetaData):
            return [focus_frames]
        return list(focus_frames)

    @staticmethod
    def _focus_acquisition_settings() -> FrameAcquisitionSettings:
        """
        Return acquisition settings used by software focus captures.

        Parameters
        ----------
        None

        Returns
        -------
        FrameAcquisitionSettings
            Runtime settings for focus acquisition.
        """
        return FrameAcquisitionSettings(
            save=False,
            normalise=False,
            illuminate_dmd=True,
            clear_dmd_after=False,
            restore_leds_after=True,
            disable_leds_after=False,
        )

    def _score_z_stack(self, frame: Frame, frames_per_z: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Score a captured Z stack and return per-Z scores and mean frames.

        Parameters
        ----------
        frame
            Captured frame stack from FrameAcquisitionManager.take_z_stack().
        frames_per_z
            Number of metadata captures acquired at each Z coordinate.

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
                self.score_image(img=self._crop_for_score(frame=frame_group[index]))
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

    def _crop_for_score(self, frame: np.ndarray) -> np.ndarray:
        """
        Return the frame region used for focus scoring.

        Parameters
        ----------
        frame
            Captured frame.

        Returns
        -------
        np.ndarray
            Cropped or original frame.
        """
        if self.config.cropping_box is None:
            return frame
        return self.config.cropping_box.crop(frame)

    @staticmethod
    def _stack_frames(focus_stack: list[np.ndarray]) -> np.ndarray:
        """
        Return captured focus frames as a height-width-z stack.

        Parameters
        ----------
        focus_stack
            List of 2D frames ordered by Z position.

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
            state: SoftwareFocusPositionState,
            previous_coordinate: Coordinate,
            scanned_z: np.ndarray,
            scores: np.ndarray,
            focus_stack: np.ndarray,
            early_status: FocusStatusType,
    ) -> SoftwareFocusResult:
        """
        Classify scores, update state, and move to the best Z when appropriate.

        Parameters
        ----------
        state
            Position state to update.
        previous_coordinate
            Coordinate from before the focus run.
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
            curve_status = get_focus_curve_type(focus_curve=scores)
            focus_status = FocusStatusType.IN_FOCUS if curve_status == FocusCurveType.HAS_GLOBAL_MAXIMUM else FocusStatusType.BAD_FOCUS_CURVE
            best_index = int(np.argmax(scores))

        best_coordinate = None
        best_frame = None
        if best_index is not None:
            best_coordinate = previous_coordinate.copy()
            best_coordinate.z = int(scanned_z[best_index])
            best_frame = focus_stack[:, :, best_index] if focus_stack.ndim == 3 else None
            if focus_status == FocusStatusType.IN_FOCUS:
                self.stage.move(target=Coordinate(None, None, int(scanned_z[best_index])), block=True)

        state.z_coordinates = scanned_z
        state.focus_scores = scores
        state.focus_stack = focus_stack
        state.status = focus_status
        state.curve_status = curve_status

        return SoftwareFocusResult(
            best_coordinate=best_coordinate,
            best_frame=best_frame,
            focus_scores=scores,
            z_coordinates=scanned_z,
            focus_status=focus_status,
            curve_status=curve_status,
        )


def get_focus_score_is_good(focus_curve: np.ndarray) -> bool:
    """
    Return whether a focus score curve has a single non-boundary maximum.

    Parameters
    ----------
    focus_curve
        Focus score values ordered by scanned Z position.

    Returns
    -------
    bool
        True when the curve is classified as having a global maximum.
    """
    return get_focus_curve_type(focus_curve=focus_curve) == FocusCurveType.HAS_GLOBAL_MAXIMUM


def get_focus_curve_type(focus_curve: np.ndarray) -> FocusCurveType:
    """
    Classify the shape of a focus score curve.

    Parameters
    ----------
    focus_curve
        Focus score values ordered by scanned Z position.

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


def create_software_focus_algorithm(
        algorithm: FocusAlgorithmType,
        config: SoftwareFocusConfig | None = None,
        threshold: float | None = None,
        rowshift: int | None = None,
        colshift: int | None = None,
        normalise_score: bool = False,
) -> SoftwareFocusAlgorithm:
    """
    Create a software focus algorithm using explicit values or config defaults.

    Parameters
    ----------
    algorithm
        FocusAlgorithmType selecting the scoring implementation.
    config
        Optional SoftwareFocusConfig providing algorithm parameters.
    threshold
        Optional squared-gradient threshold.
    rowshift
        Optional row shift for the Steel algorithm. If None, config or legacy
        defaults are used.
    colshift
        Optional column shift for the Steel algorithm. If None, config or
        legacy defaults are used.
    normalise_score
        If True, normalise the Steel score by image area.

    Returns
    -------
    SoftwareFocusAlgorithm
        Focus scoring algorithm instance.
    """
    if config is not None and not isinstance(config, SoftwareFocusConfig):
        raise TypeError(
            f"create_software_focus_algorithm: config must be SoftwareFocusConfig or None, received {type(config)}."
        )
    resolved_rowshift = rowshift if rowshift is not None else (config.rowshift_px if config is not None else 25)
    resolved_colshift = colshift if colshift is not None else (config.colshift_px if config is not None else 50)
    return _create_software_focus_algorithm(
        algorithm=algorithm,
        threshold=threshold,
        rowshift=resolved_rowshift,
        colshift=resolved_colshift,
        normalise=normalise_score,
    )


def get_roi_focus_score(
        img: np.ndarray,
        algorithm: FocusAlgorithmType,
        boxes: list[CroppingBox],
        threshold: float | None = None,
        rowshift: int = 25,
        colshift: int = 50,
        normalise_score: bool = False,
        config: SoftwareFocusConfig | None = None,
) -> float:
    """
    Return a summed focus score for cropped regions of one image.

    Parameters
    ----------
    img
        Source image array containing all regions.
    algorithm
        FocusAlgorithmType selecting the scoring implementation.
    boxes
        CroppingBox objects selecting regions to score.
    threshold
        Optional squared-gradient threshold.
    rowshift
        Row shift for the Steel algorithm.
    colshift
        Column shift for the Steel algorithm.
    normalise_score
        If True, normalise the Steel score by image area.
    config
        Optional SoftwareFocusConfig providing algorithm parameters.

    Returns
    -------
    float
        Sum of focus scores across the provided regions.
    """
    scorer = create_software_focus_algorithm(
        algorithm=algorithm,
        config=config,
        threshold=threshold,
        rowshift=rowshift,
        colshift=colshift,
        normalise_score=normalise_score,
    )
    return scorer.score_rois(img=img, boxes=boxes)


def get_focus_score(
        img: np.ndarray,
        algorithm: FocusAlgorithmType,
        threshold: float | None = None,
        rowshift: int = 25,
        colshift: int = 50,
        normalise_score: bool = False,
        config: SoftwareFocusConfig | None = None,
) -> float:
    """
    Return a focus score for one image.

    Parameters
    ----------
    img
        Image array to score.
    algorithm
        FocusAlgorithmType selecting the scoring implementation.
    threshold
        Optional squared-gradient threshold.
    rowshift
        Row shift for the Steel algorithm.
    colshift
        Column shift for the Steel algorithm.
    normalise_score
        If True, normalise the Steel score by image area.
    config
        Optional SoftwareFocusConfig providing algorithm parameters.

    Returns
    -------
    float
        Focus score for the provided image.
    """
    scorer = create_software_focus_algorithm(
        algorithm=algorithm,
        config=config,
        threshold=threshold,
        rowshift=rowshift,
        colshift=colshift,
        normalise_score=normalise_score,
    )
    return scorer.score_image(img=img)


def get_focus_score_laplacian_var(img: np.ndarray) -> float:
    """
    Return a Laplacian-variance focus score for one image.

    Parameters
    ----------
    img
        Image array to score.

    Returns
    -------
    float
        Variance of the squared inner Laplacian image.
    """
    return LaplacianVarianceFocusAlgorithm().score_image(img=img)


def get_focus_score_squared_gradient(img: np.ndarray, threshold: float | None = None) -> float:
    """
    Return a squared-gradient focus score for one image.

    Parameters
    ----------
    img
        Image array to score.
    threshold
        Optional squared-gradient threshold.

    Returns
    -------
    float
        Mean squared horizontal gradient after thresholding.
    """
    return SquaredGradientAverageFocusAlgorithm(threshold=threshold).score_image(img=img)


def get_focus_score_steel(
        img: np.ndarray,
        rowshift: int,
        colshift: int,
        normalise: bool = False,
) -> float:
    """
    Return a Steel focus score for one image.

    Parameters
    ----------
    img
        Image array to score.
    rowshift
        Pixel shift along image rows.
    colshift
        Pixel shift along image columns.
    normalise
        If True, divide the score by twice the image area.

    Returns
    -------
    float
        Shifted-difference Steel focus score.
    """
    return SteelFocusAlgorithm(rowshift=rowshift, colshift=colshift, normalise=normalise).score_image(img=img)
