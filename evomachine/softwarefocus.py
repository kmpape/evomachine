from __future__ import annotations

from dataclasses import dataclass
import threading

import numpy as np
from delta.utils import CroppingBox

from evomachine.acquisition import FrameAcquisitionManager, FrameAcquisitionSettings
from evomachine.bindings.software_focus.software_focus_algorithms import (
    create_software_focus_algorithm,
)
from evomachine.config_types import Frame, FrameMetaData, SoftwareFocusConfigNew
from evomachine.coordinates import Coordinate
from evomachine.types import FocusCurveType, FocusStatusType


@dataclass
class SoftwareFocusPositionState:
    """State recorded for one software focus position."""

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
        self._position_states: dict[int, SoftwareFocusPositionState] = {}
        self._position_config: dict[int, SoftwareFocusConfigNew] = {}
        self._stop_requested: bool = False

    def initialise_positions(
            self,
            position_ids: list[int],
            position_configs: dict[int, SoftwareFocusConfigNew] | None = None,
    ) -> None:
        """
        Initialise empty focus state and optional configs for positions.

        Parameters
        ----------
        position_ids
            List of integer position IDs to track.
        position_configs
            Optional mapping of position ID to position-specific config.

        Returns
        -------
        None
        """
        if not isinstance(position_ids, list):
            raise TypeError(
                f"SoftwareFocus.initialise_positions: position_ids must be list[int], received {type(position_ids)}."
            )
        if not all(isinstance(position_id, int) and not isinstance(position_id, bool) for position_id in position_ids):
            raise TypeError("SoftwareFocus.initialise_positions: every position ID must be int.")
        self._position_states = {
            position_id: SoftwareFocusPositionState()
            for position_id in position_ids
        }
        self._position_config = {}
        if position_configs is None:
            return
        if not isinstance(position_configs, dict):
            raise TypeError("SoftwareFocus.initialise_positions: position_configs must be dict[int, SoftwareFocusConfigNew].")
        for position_id, config in position_configs.items():
            if position_id not in self._position_states:
                raise KeyError(f"SoftwareFocus.initialise_positions: unknown config position ID {position_id}.")
            self._position_config[position_id] = self._validate_config(config=config)

    def update_config(
            self,
            config: SoftwareFocusConfigNew,
            position_id: int | None = None,
    ) -> None:
        """
        Replace the default or one position-specific software focus config.

        Parameters
        ----------
        config
            Replacement SoftwareFocusConfigNew.
        position_id
            Optional position ID. If None, update the default config.

        Returns
        -------
        None
        """
        config = self._validate_config(config=config)
        if position_id is None:
            self.default_config = config
            return
        if not isinstance(position_id, int) or isinstance(position_id, bool):
            raise TypeError(f"SoftwareFocus.update_config: position_id must be int or None, received {type(position_id)}.")
        self._position_config[position_id] = config
        self._position_states.setdefault(position_id, SoftwareFocusPositionState())

    def score_image(
            self,
            img: np.ndarray,
            config: SoftwareFocusConfigNew,
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
        config = self._validate_config(config=config)
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
        config = self._config_for_position(position_id=resolved_position_id)
        state = self._position_states.setdefault(resolved_position_id, SoftwareFocusPositionState())
        stage = self.acquisition_manager.stage
        if stage is None:
            raise RuntimeError("SoftwareFocus.run: acquisition manager has no stage.")
        previous_coordinate = stage.get_coordinates(query_hardware=True)
        if previous_coordinate.z is None:
            raise RuntimeError("SoftwareFocus.run: current stage coordinate does not contain Z.")
        state.previous_coordinate = previous_coordinate.copy()
        z_coordinates = self._make_z_coordinates(current_z=previous_coordinate.z, config=config)
        state.z_coordinates = z_coordinates

        frame_metadata_items = self._focus_frame_metadata_items(config=config)
        settings = self._focus_acquisition_settings(config=config)
        if self._should_stop(stop_event=stop_event):
            return self._finalise_result(
                state=state,
                previous_coordinate=previous_coordinate,
                scanned_z=np.asarray([], dtype=int),
                scores=np.asarray([], dtype=float),
                focus_stack=np.empty((0,), dtype=np.float64),
                early_status=FocusStatusType.UNKNOWN,
            )

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
            config=config,
        )
        scanned_z = z_coordinates[:scores_array.size]
        return self._finalise_result(
            state=state,
            previous_coordinate=previous_coordinate,
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

    def _config_for_position(self, position_id: int) -> SoftwareFocusConfigNew:
        """
        Return the config for one position.

        Parameters
        ----------
        position_id
            Position ID to resolve.

        Returns
        -------
        SoftwareFocusConfigNew
            Position-specific config when present, otherwise default config.
        """
        return self._position_config.get(position_id, self.default_config)

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
        stage = self.acquisition_manager.stage
        if stage is not None:
            get_pos = getattr(stage, "get_pos", None)
            if callable(get_pos):
                return int(get_pos())
        return -1

    @staticmethod
    def _make_z_coordinates(current_z: int | float, config: SoftwareFocusConfigNew) -> np.ndarray:
        """
        Return scan Z coordinates around the current Z position.

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

    @staticmethod
    def _focus_frame_metadata_items(config: SoftwareFocusConfigNew) -> list[FrameMetaData]:
        """
        Return configured focus frame metadata entries.

        Parameters
        ----------
        config
            Active software focus configuration.

        Returns
        -------
        list[FrameMetaData]
            Configured frame metadata entries.
        """
        return list(config.focus_frames)

    @staticmethod
    def _focus_acquisition_settings(config: SoftwareFocusConfigNew) -> FrameAcquisitionSettings:
        """
        Return acquisition settings used by software focus captures.

        Parameters
        ----------
        config
            Active software focus configuration.

        Returns
        -------
        FrameAcquisitionSettings
            Runtime settings for focus acquisition.
        """
        if config.acquisition_settings is not None:
            return config.acquisition_settings
        return FrameAcquisitionSettings(
            save=False,
            normalise=False,
            illuminate_dmd=True,
            clear_dmd_after=False,
            restore_leds_after=True,
            disable_leds_after=False,
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

    @staticmethod
    def _get_focus_curve_type(focus_curve: np.ndarray) -> FocusCurveType:
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
