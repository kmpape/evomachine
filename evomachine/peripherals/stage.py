from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from evomachine.coordinates import Coordinate, CoordinateBounds
from evomachine.peripherals.peripheralcontrollers import PeripheralController, get_peripheral_controller
from evomachine.peripherals.peripherals import Peripheral
from evomachine.bindings.binding_types import BindingType
from evomachine.types import AxisType, FovDirectionType, PositiveScalingType, UNKNOWN_FOV_ID


@dataclass
class StageConfig:
    """
    Configuration object used by StageFactory to create Stage instances.

    Parameters
    ----------
    binding
        Stage binding type to create.
    fov_step_size
        Positive field-of-view step size in stage coordinate units.
    name
        Optional human-readable stage name.
    check_initialised
        If True, public hardware-querying methods require initialisation.
    check_alive
        If True, public hardware-querying methods require a live stage.
    initial_coordinate
        Optional initial coordinate for virtual stages.
    coordinate_bounds
        Optional software bounds used for movement validation.

    Returns
    -------
    StageConfig
        Validated stage factory configuration.
    """

    binding: BindingType
    fov_step_size: float
    name: str | None = None
    check_initialised: bool = True
    check_alive: bool = True
    initial_coordinate: Coordinate | None = None
    coordinate_bounds: CoordinateBounds | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.binding, BindingType):
            raise TypeError(f"StageConfig: binding must be BindingType, received {type(self.binding)}.")
        if not isinstance(self.fov_step_size, int | float):
            raise TypeError(f"StageConfig: fov_step_size must be numeric, received {type(self.fov_step_size)}.")
        if self.fov_step_size <= 0:
            raise ValueError(f"StageConfig: fov_step_size must be positive, received {self.fov_step_size}.")
        self.fov_step_size = float(self.fov_step_size)
        if self.name is not None and not isinstance(self.name, str):
            raise TypeError(f"StageConfig: name must be str or None, received {type(self.name)}.")
        if not isinstance(self.check_initialised, bool):
            raise TypeError(
                f"StageConfig: check_initialised must be bool, received {type(self.check_initialised)}."
            )
        if not isinstance(self.check_alive, bool):
            raise TypeError(f"StageConfig: check_alive must be bool, received {type(self.check_alive)}.")
        if self.initial_coordinate is not None and not isinstance(self.initial_coordinate, Coordinate):
            raise TypeError(
                f"StageConfig: initial_coordinate must be Coordinate or None, "
                f"received {type(self.initial_coordinate)}."
            )
        if self.coordinate_bounds is not None and not isinstance(self.coordinate_bounds, CoordinateBounds):
            raise TypeError(
                f"StageConfig: coordinate_bounds must be CoordinateBounds or None, "
                f"received {type(self.coordinate_bounds)}."
            )

    def copy(self) -> "StageConfig":
        return StageConfig(**self.__dict__)

    def updated(self, **kwargs) -> "StageConfig":
        unknown_keys = [key for key in kwargs if key not in self.__dict__]
        if unknown_keys:
            raise ValueError(f"StageConfig.updated: unknown fields {unknown_keys}.")
        values = dict(self.__dict__)
        values.update(kwargs)
        return StageConfig(**values)

    def update_from_mapping(self, updates: dict) -> "StageConfig":
        if not isinstance(updates, dict):
            raise TypeError("StageConfig.update_from_mapping: updates must be dict.")
        return self.updated(**updates)


class Stage(Peripheral):
    """
    Base class for microscope stages.

    This class owns movement bookkeeping, fov-ID handling, bounds checking,
    and optional readiness checks. Subclasses implement only the low-level
    hardware operations.
    """

    AXES = (AxisType.X, AxisType.Y, AxisType.Z)
    UNKNOWN_FOV_ID = UNKNOWN_FOV_ID

    def __init__(
            self,
            name: str,
            fov_step_size: float,
            coordinate_bounds: CoordinateBounds | None = None,
            check_initialised: bool = True,
            check_alive: bool = True,
    ):
        """
        Initialise shared stage state.

        Parameters
        ----------
        name
            Human-readable stage name.
        fov_step_size
            Field-of-view step size in stage coordinate units.
        coordinate_bounds
            Optional software movement bounds. If None, hardware-reported limits
            are used when validating moves.
        check_initialised
            If True, public hardware-querying methods raise RuntimeError when the
            stage has not been initialised.
        check_alive
            If True, public hardware-querying methods raise RuntimeError when the
            stage does not report alive.

        Returns
        -------
        None
        """
        if fov_step_size <= 0:
            raise ValueError(f"Stage.__init__: fov_step_size must be positive, received {fov_step_size}.")
        super().__init__(
            name=name,
            check_initialised=check_initialised,
            check_alive=check_alive,
        )
        self._current_coordinate: Coordinate = Coordinate.none_coordinate()
        "Current coordinate as returned by hardware queries and updated after moves. Axes are None when unknown."
        self._current_fov_id: int = self.UNKNOWN_FOV_ID
        "Current fov ID. Becomes available when moving to a registered fov ID."
        self._fov_id_to_coordinate: dict[int, Coordinate] = {}
        "Mapping from FoV ID to Coordinate for registered FoVs. Populated by set_fov_id_to_coordinate."
        self._fov_step_size: float = fov_step_size
        "Field-of-view step size in stage coordinate units. Used for FoV movement targets."
        self._coordinate_bounds: CoordinateBounds | None = coordinate_bounds.copy() if coordinate_bounds else None
        "Software movement bounds. Set via constructor."
        self._fov_direction_to_axis_sign: dict[FovDirectionType, tuple[AxisType, int]] = {
            FovDirectionType.UP: (AxisType.Y, -1),
            FovDirectionType.DOWN: (AxisType.Y, +1),
            FovDirectionType.LEFT: (AxisType.X, -1),
            FovDirectionType.RIGHT: (AxisType.X, +1),
        }
        "Mapping from field-of-view movement direction to affected axis and movement sign. Used for FoV movement targets."

    @classmethod
    def _validate_axes(cls, axes: list[AxisType] | None = None) -> list[AxisType]:
        """
        Validate and de-duplicate a list of stage axes.

        Parameters
        ----------
        axes
            Axes to keep. If None, all stage axes are used.

        Returns
        -------
        list[AxisType]
            Validated axes in the order requested.
        """
        if axes is None:
            return list(cls.AXES)
        axes_norm: list[AxisType] = []
        for axis in axes:
            if not isinstance(axis, AxisType):
                raise TypeError(f"Stage._validate_axes: expected AxisType, received {type(axis)}.")
            if axis not in cls.AXES:
                raise ValueError(f"Stage._validate_axes: unsupported axis {axis}.")
            if axis not in axes_norm:
                axes_norm.append(axis)
        return axes_norm

    def _update_current_coordinate(self, coordinate: Coordinate) -> None:
        """
        Update the cached current coordinate.

        Parameters
        ----------
        coordinate
            Full or partial coordinate returned by hardware or requested by a move.

        Returns
        -------
        None
        """
        self._current_coordinate = self._current_coordinate.merge(update=coordinate)

    def _post_initialise(self, force: bool = False) -> None:
        """Cache current hardware coordinates after initialisation."""
        if self._is_alive:
            self._current_coordinate = self._get_coordinates()

    def stop(self) -> None:
        """
        Stop stage motion.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.halt()

    def get_coordinates(
            self,
            axes: list[AxisType] | None = None,
            query_hardware: bool = True,
    ) -> Coordinate:
        """
        Return current stage coordinates.

        Parameters
        ----------
        axes
            Axes to include. If None, X, Y, and Z are returned.
        query_hardware
            If True, query the device before returning. If False, return the cached
            coordinate only.

        Returns
        -------
        Coordinate
            Coordinate containing the requested axes.
        """
        axes_norm = self._validate_axes(axes=axes)
        if query_hardware:
            self._require_ready(action="get_coordinates")
            self._update_current_coordinate(coordinate=self._get_coordinates())
        return self._current_coordinate.filter_axes(axes=axes_norm)

    def get_fov_id(self) -> int:
        """
        Return the current fov ID.

        Parameters
        ----------
        None

        Returns
        -------
        int
            Current fov ID, or UNKNOWN_FOV_ID when the current coordinate
            does not correspond to a known fov ID.
        """
        return self._current_fov_id

    def get_fov_step_size(self) -> float:
        """
        Return the configured field-of-view movement step.

        Parameters
        ----------
        None

        Returns
        -------
        float
            Field-of-view step size.
        """
        return self._fov_step_size

    def set_fov_step_size(self, fov_step_size: float) -> None:
        """
        Set the field-of-view movement step.

        Parameters
        ----------
        fov_step_size
            Positive movement step size.

        Returns
        -------
        None
        """
        if fov_step_size <= 0:
            raise ValueError(f"Stage.set_fov_step_size: fov_step_size must be positive, received {fov_step_size}.")
        self._fov_step_size = fov_step_size

    def _apply_config(self, config: StageConfig) -> None:
        """Apply stage-specific config fields."""
        self._fov_step_size = config.fov_step_size
        self._coordinate_bounds = config.coordinate_bounds.copy() if config.coordinate_bounds else None
        if config.initial_coordinate is not None:
            self._current_coordinate = config.initial_coordinate.copy()

    def _config_requires_reinitialise(self, current_config: StageConfig, new_config: StageConfig) -> bool:
        """Return True because stage config updates currently reinitialise live stages."""
        return True

    def _after_config_reinitialise(self) -> None:
        """Apply configured initial coordinate after reinitialisation when present."""
        if self.config is not None and self.config.initial_coordinate is not None:
            self._current_coordinate = self.config.initial_coordinate.copy()

    def update_config(self, config: StageConfig | None = None, **updates) -> None:
        """Replace or update stage configuration at runtime."""
        super().update_config(config=config, **updates)

    def get_coordinate_bounds(self) -> CoordinateBounds:
        """
        Return the active software or hardware coordinate bounds.

        Parameters
        ----------
        None

        Returns
        -------
        CoordinateBounds
            Coordinate bounds used for movement validation.
        """
        if self._coordinate_bounds is not None:
            return self._coordinate_bounds.copy()
        self._require_ready(action="get_coordinate_bounds")
        return CoordinateBounds.from_limits(limits=self._get_stage_limits())

    def get_stage_limits(self) -> tuple[Coordinate, Coordinate]:
        """
        Return lower and upper coordinate limits.

        Parameters
        ----------
        None

        Returns
        -------
        tuple[Coordinate, Coordinate]
            Minimum and maximum coordinates. Axes set to None are unchecked.
        """
        return self.get_coordinate_bounds().as_limits()

    def coordinate_is_out_of_bounds(self, coordinate: Coordinate) -> bool:
        """
        Check whether a coordinate is outside the configured stage limits.

        Parameters
        ----------
        coordinate
            Full or partial coordinate to validate. Axes set to None are ignored.

        Returns
        -------
        bool
            True if any provided axis lies outside its stage limit.
        """
        if not isinstance(coordinate, Coordinate):
            raise TypeError(f"Stage.coordinate_is_out_of_bounds: expected Coordinate, received {type(coordinate)}.")
        return self.get_coordinate_bounds().is_out_of_bounds(coordinate=coordinate)

    def set_fov_id_to_coordinate(
            self,
            fov_id_to_coordinate: dict[int, Coordinate],
            use_autofocus: bool,
    ) -> bool:
        """
        Register fov IDs for later movement.

        Parameters
        ----------
        fov_id_to_coordinate
            Mapping from fov ID to stage Coordinate.
        use_autofocus
            If True, registered coordinates must not contain Z. If False, they must
            contain Z.

        Returns
        -------
        bool
            True when the complete mapping is valid and stored, otherwise False.
        """
        coordinates: dict[int, Coordinate] = {}
        for registered_fov_id, coordinate in fov_id_to_coordinate.items():
            if not isinstance(coordinate, Coordinate):
                raise TypeError(f"Stage.set_fov_id_to_coordinate: FoV {registered_fov_id} is not a Coordinate.")
            if (not use_autofocus) and (not coordinate.has_z()):
                return False
            if use_autofocus and coordinate.has_z():
                return False
            if self.coordinate_is_out_of_bounds(coordinate):
                return False
            coordinates[registered_fov_id] = coordinate.copy()
        self._fov_id_to_coordinate = coordinates
        return True

    @staticmethod
    def _fov_multiplier_value(multiplier: PositiveScalingType) -> float:
        """
        Return a validated field-of-view movement multiplier.

        Parameters
        ----------
        multiplier
            Positive numeric multiplier for fov_step_size.

        Returns
        -------
        float
            Validated multiplier.
        """
        if not isinstance(multiplier, int | float) or isinstance(multiplier, bool):
            raise TypeError(f"Stage._fov_multiplier_value: multiplier must be a positive number, received {type(multiplier)}.")
        if multiplier <= 0:
            raise ValueError(f"Stage._fov_multiplier_value: multiplier must be positive, received {multiplier}.")
        return float(multiplier)

    def _coordinate_from_fov_moves(
            self,
            fov_moves: list[tuple[FovDirectionType, PositiveScalingType]],
    ) -> Coordinate:
        """
        Convert a list of field-of-view movements into one absolute Coordinate.

        Parameters
        ----------
        fov_moves
            List of (FovDirectionType, multiplier) movement requests. Multipliers
            must be positive and scale fov_step_size. HOME is not allowed in a list
            because it is an absolute hardware command.

        Returns
        -------
        Coordinate
            Partial absolute Coordinate containing the X/Y target axes.
        """
        if len(fov_moves) == 0:
            return Coordinate.none_coordinate()

        axis_deltas = {AxisType.X: 0, AxisType.Y: 0}
        for fov_move in fov_moves:
            if not isinstance(fov_move, tuple) or len(fov_move) != 2:
                raise TypeError("Stage._coordinate_from_fov_moves: each FoV move must be a tuple(direction, multiplier).")
            direction, multiplier = fov_move
            if not isinstance(direction, FovDirectionType):
                raise TypeError(
                    f"Stage._coordinate_from_fov_moves: direction must be FovDirectionType, received {type(direction)}."
                )
            if direction == FovDirectionType.HOME:
                raise ValueError("Stage._coordinate_from_fov_moves: HOME must be used as a single move target.")
            axis, sign = self._fov_direction_to_axis_sign[direction]
            axis_deltas[axis] += int(sign * self.get_fov_step_size() * self._fov_multiplier_value(multiplier))

        current = self.get_coordinates(axes=[AxisType.X, AxisType.Y], query_hardware=True)
        current_x = current.axis_value(axis=AxisType.X)
        current_y = current.axis_value(axis=AxisType.Y)
        if current_x is None or current_y is None:
            raise RuntimeError("Stage._coordinate_from_fov_moves: current X/Y coordinates are not initialised.")
        return Coordinate(
            x=current_x + axis_deltas[AxisType.X],
            y=current_y + axis_deltas[AxisType.Y],
            z=None,
            channel_id=current.get_channel_id(),
        )

    def _coordinate_from_move_target(
            self,
            target: int | Coordinate | tuple[FovDirectionType, PositiveScalingType] | list[tuple[FovDirectionType, PositiveScalingType]],
    ) -> tuple[int, Coordinate, bool]:
        """
        Convert a public movement target into a fov ID and Coordinate.

        Parameters
        ----------
        target
            FoV ID, Coordinate, single FoV movement tuple, or FoV movement
            list.

        Returns
        -------
        tuple[int, Coordinate, bool]
            FoV ID to cache, Coordinate to move to, and whether to run a
            hardware home command.
        """
        if isinstance(target, int) and not isinstance(target, bool):
            if target not in self._fov_id_to_coordinate:
                raise IndexError(f"Stage.move: fov index {target} out of range.")
            return target, self._fov_id_to_coordinate[target].copy(), False
        if isinstance(target, Coordinate):
            return self.UNKNOWN_FOV_ID, target.copy(), False
        if isinstance(target, tuple):
            if len(target) != 2:
                raise TypeError("Stage.move: FoV tuple target must be tuple(direction, multiplier).")
            direction, multiplier = target
            if not isinstance(direction, FovDirectionType):
                raise TypeError(f"Stage.move: direction must be FovDirectionType, received {type(direction)}.")
            if direction == FovDirectionType.HOME:
                self._fov_multiplier_value(multiplier)
                return 0, Coordinate.none_coordinate(), True
            return self.UNKNOWN_FOV_ID, self._coordinate_from_fov_moves(fov_moves=[target]), False
        if isinstance(target, list):
            return self.UNKNOWN_FOV_ID, self._coordinate_from_fov_moves(fov_moves=target), False
        raise TypeError(
            f"Stage.move: expected int, Coordinate, tuple[FovDirectionType, PositiveScalingType], "
            f"or list[tuple[FovDirectionType, PositiveScalingType]], "
            f"received {type(target)}."
        )

    def move(
            self,
            target: int | Coordinate | tuple[FovDirectionType, PositiveScalingType] | list[tuple[FovDirectionType, PositiveScalingType]],
            block: bool = True,
    ) -> None:
        """
        Move the stage through the single public stage movement interface.

        Parameters
        ----------
        target
            Supported movement target. An int moves to a registered fov ID
            configured by set_fov_id_to_coordinate. A Coordinate is an absolute
            full or partial stage coordinate; axes set to None are not moved. A
            tuple[FovDirectionType, PositiveScalingType] moves one field-of-view
            step in the requested direction by multiplier * fov_step_size. A list of
            those tuples combines all UP/DOWN/LEFT/RIGHT FoV deltas into one
            absolute X/Y move, allowing diagonal or repeated-direction movement
            with a single hardware move. FovDirectionType.HOME is accepted only
            as a single tuple target, runs the hardware home command, and uses an
            implicit positive scaling of 1.0 because home is absolute rather than
            relative. HOME is not valid inside a FoV movement list. Positive
            scaling values must be numeric, non-bool, and greater than zero.
        block
            If True, wait until the hardware move has completed.

        Returns
        -------
        None
        """
        self._require_ready(action="move")
        fov_id, move_coordinate, use_home = self._coordinate_from_move_target(target=target)
        if use_home:
            self._current_coordinate = self._home(block=block)
            self._current_fov_id = fov_id
            return
        if not move_coordinate.has_axis_value():
            return
        if self.coordinate_is_out_of_bounds(coordinate=move_coordinate):
            raise ValueError(f"Stage.move: coordinate is out of bounds: {move_coordinate}.")
        self._update_current_coordinate(coordinate=self._move(coordinate=move_coordinate, block=block))
        self._current_fov_id = fov_id

    @abstractmethod
    def halt(self) -> None:
        """
        Stop stage motion immediately.

        Subclasses should issue the hardware stop/halt command. This method should
        be safe to call when a move is in progress and should raise a standard Python
        exception if the hardware command fails.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        raise NotImplementedError

    def halt_stage(self) -> None:
        """
        Compatibility wrapper for halt.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.halt()

    def zero_coordinates(self) -> None:
        """
        Zero the hardware coordinate system and cache the resulting coordinate.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self._require_ready(action="zero_coordinates")
        self._current_coordinate = self._zero_coordinates()
        self._current_fov_id = self.UNKNOWN_FOV_ID

    @abstractmethod
    def _initialise(self, force: bool = False) -> bool:
        """
        Perform hardware-specific initialisation.

        Implementations should connect to the stage, configure any required state,
        and return True only when the stage is ready for use. They should raise a
        standard Python exception for unrecoverable setup failures.

        Parameters
        ----------
        force
            If True, re-run setup even if the subclass already has an open connection.

        Returns
        -------
        bool
            True when initialisation succeeded.
        """
        raise NotImplementedError

    @abstractmethod
    def _finalise(self, force: bool = False) -> None:
        """
        Perform hardware-specific cleanup.

        Implementations should release connections/resources. They should be
        idempotent where possible and may use force to close partially broken
        connections.

        Parameters
        ----------
        force
            If True, force cleanup even when normal cleanup would be skipped.

        Returns
        -------
        None
        """
        raise NotImplementedError

    @abstractmethod
    def _check_is_alive(self) -> bool:
        """
        Query whether the hardware connection is alive.

        Implementations should perform the lightest reliable health check available.
        They should return False for an unavailable device and raise a standard
        Python exception only when the health check itself fails unexpectedly.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            True when the stage is reachable.
        """
        raise NotImplementedError

    @abstractmethod
    def _get_coordinates(self) -> Coordinate:
        """
        Query the full current hardware coordinate.

        Implementations should return X, Y, and Z in a Coordinate. Missing axes
        should be represented as None only if the hardware genuinely cannot report
        that axis.

        Parameters
        ----------
        None

        Returns
        -------
        Coordinate
            Current hardware coordinate.
        """
        raise NotImplementedError

    @abstractmethod
    def _get_stage_limits(self) -> tuple[Coordinate, Coordinate]:
        """
        Query hardware movement limits.

        Implementations should return the lower and upper Coordinate limits. Both
        Coordinates are expected to include X, Y, and Z values.

        Parameters
        ----------
        None

        Returns
        -------
        tuple[Coordinate, Coordinate]
            Minimum and maximum allowed coordinates.
        """
        raise NotImplementedError

    @abstractmethod
    def _move(self, coordinate: Coordinate, block: bool = True) -> Coordinate:
        """
        Move the hardware stage.

        Implementations should move every axis that is not None in coordinate. They
        should block until idle when block is True. The returned Coordinate should
        describe the stage fov after the move; returning the requested partial
        Coordinate is acceptable when the hardware cannot cheaply report the final
        full fov.

        Parameters
        ----------
        coordinate
            Full or partial target coordinate.
        block
            If True, wait until the hardware move has completed.

        Returns
        -------
        Coordinate
            Full or partial coordinate reached after the move.
        """
        raise NotImplementedError

    @abstractmethod
    def _home(self, block: bool = False) -> Coordinate:
        """
        Move the hardware stage to its home fov.

        Implementations should perform the hardware home command and return the
        coordinate reached after homing. If block is True, they should wait until
        homing has completed.

        Parameters
        ----------
        block
            If True, wait until the home command has completed.

        Returns
        -------
        Coordinate
            Coordinate reached after homing.
        """
        raise NotImplementedError

    @abstractmethod
    def _zero_coordinates(self) -> Coordinate:
        """
        Zero the hardware coordinate system.

        Implementations should issue the device-specific zero command and return
        the coordinate that should be cached afterwards, usually Coordinate(0, 0, 0).

        Parameters
        ----------
        None

        Returns
        -------
        Coordinate
            Coordinate after zeroing.
        """
        raise NotImplementedError


class StageFactory:
    """Factory for creating Stage instances from a typed StageConfig."""

    @staticmethod
    def create(
            config: StageConfig,
            peripheral_controllers: PeripheralController | list[PeripheralController] | None = None,
    ) -> Stage:
        """
        Create a Stage from a StageConfig.

        Parameters
        ----------
        config
            Typed stage configuration describing the desired binding and shared
            construction options.
        peripheral_controllers
            One PeripheralController or a list of available PeripheralController
            instances. The requested stage binding selects the required
            controller type.

        Returns
        -------
        Stage
            A stage instance for the requested binding.
        """
        if not isinstance(config, StageConfig):
            raise TypeError(f"StageFactory.create: expected StageConfig, received {type(config)}.")

        if config.binding == BindingType.VIRTUAL:
            from evomachine.bindings.virtual.peripheralcontroller import VirtualPeripheralController
            from evomachine.bindings.virtual.stage import VirtualStage

            peripheral_ctrl = get_peripheral_controller(
                peripheral_controllers=peripheral_controllers,
                controller_type=VirtualPeripheralController,
                action="StageFactory.create",
            )
            stage = VirtualStage(
                peripheral_ctrl=peripheral_ctrl,
                fov_step_size=config.fov_step_size,
                name=config.name or "Virtual Stage",
                initial_coordinate=config.initial_coordinate,
                coordinate_bounds=config.coordinate_bounds,
                check_initialised=config.check_initialised,
                check_alive=config.check_alive,
            )
            stage.config = config.copy()
            return stage

        if config.binding == BindingType.ASI_TIGER:
            from evomachine.bindings.asitiger.peripheralcontroller import TigerPeripheralController
            from evomachine.bindings.asitiger.stage import TigerStage

            peripheral_ctrl = get_peripheral_controller(
                peripheral_controllers=peripheral_controllers,
                controller_type=TigerPeripheralController,
                action="StageFactory.create",
            )
            stage = TigerStage(
                peripheral_ctrl=peripheral_ctrl,
                fov_step_size=config.fov_step_size,
                name=config.name or "ASI Tiger Stage",
                check_initialised=config.check_initialised,
                check_alive=config.check_alive,
            )
            stage.config = config.copy()
            return stage

        raise ValueError(f"StageFactory.create: unsupported stage binding {config.binding}.")
