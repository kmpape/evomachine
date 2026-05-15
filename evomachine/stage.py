from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from evomachine.coordinates import Coordinate
from evomachine.peripherals import Peripheral, PeripheralController, get_peripheral_controller
from evomachine.types import AxisType, StageBindingType


class Stage(Peripheral):
    """
    Base class for microscope stages.

    This class owns movement bookkeeping, position-ID handling, bounds checking,
    and optional readiness checks. Subclasses implement only the low-level
    hardware operations.
    """

    AXES = (AxisType.X, AxisType.Y, AxisType.Z)
    UNKNOWN_POSITION_ID = -1

    def __init__(
            self,
            name: str,
            delta_fov: float,
            check_initialised: bool = True,
            check_alive: bool = True,
    ):
        """
        Initialise shared stage state.

        Parameters
        ----------
        name
            Human-readable stage name.
        delta_fov
            Field-of-view step size in stage coordinate units.
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
        if delta_fov <= 0:
            raise ValueError(f"Stage.__init__: delta_fov must be positive, received {delta_fov}.")
        self.name: str = name
        self._is_initialised: bool = False
        self._is_alive: bool = False
        self._check_initialised: bool = check_initialised
        self._check_alive: bool = check_alive
        self._current_pos: Coordinate = Coordinate.none_coordinate()
        self._curr_pos: int = self.UNKNOWN_POSITION_ID
        self._pos_id_to_coordinate: dict[int, Coordinate] = {}
        self._delta_fov: float = delta_fov

    @classmethod
    def _normalise_axes(cls, axes: list[AxisType] | None = None) -> list[AxisType]:
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
                raise TypeError(f"Stage._normalise_axes: expected AxisType, received {type(axis)}.")
            if axis not in cls.AXES:
                raise ValueError(f"Stage._normalise_axes: unsupported axis {axis}.")
            if axis not in axes_norm:
                axes_norm.append(axis)
        return axes_norm

    @staticmethod
    def _axis_value(coordinate: Coordinate, axis: AxisType) -> float | int | None:
        """
        Return one axis value from a Coordinate.

        Parameters
        ----------
        coordinate
            Coordinate to read from.
        axis
            Axis to read.

        Returns
        -------
        float | int | None
            Coordinate value for the requested axis, or None when unset.
        """
        if axis == AxisType.X:
            return coordinate.x
        if axis == AxisType.Y:
            return coordinate.y
        if axis == AxisType.Z:
            return coordinate.z
        raise ValueError(f"Stage._axis_value: unsupported axis {axis}.")

    @staticmethod
    def _coordinate_from_axis(
            axis: AxisType,
            value: float | int,
            channel_id: int = 0,
    ) -> Coordinate:
        """
        Build a partial Coordinate containing one axis.

        Parameters
        ----------
        axis
            Axis to populate.
        value
            Target coordinate for the axis.
        channel_id
            Channel ID to preserve on the coordinate.

        Returns
        -------
        Coordinate
            Coordinate with only the selected axis populated.
        """
        return Coordinate(
            x=value if axis == AxisType.X else None,
            y=value if axis == AxisType.Y else None,
            z=value if axis == AxisType.Z else None,
            channel_id=channel_id,
        )

    @classmethod
    def _coordinate_has_value(cls, coordinate: Coordinate) -> bool:
        """
        Check whether any axis in a Coordinate has a value.

        Parameters
        ----------
        coordinate
            Coordinate to inspect.

        Returns
        -------
        bool
            True when at least one of X, Y, or Z is not None.
        """
        return any(cls._axis_value(coordinate=coordinate, axis=axis) is not None for axis in cls.AXES)

    @classmethod
    def _filter_coordinate(cls, coordinate: Coordinate, axes: list[AxisType]) -> Coordinate:
        """
        Return a copy of a Coordinate containing only selected axes.

        Parameters
        ----------
        coordinate
            Coordinate to filter.
        axes
            Axes to include in the returned Coordinate.

        Returns
        -------
        Coordinate
            Partial Coordinate with non-requested axes set to None.
        """
        return Coordinate(
            x=coordinate.x if AxisType.X in axes else None,
            y=coordinate.y if AxisType.Y in axes else None,
            z=coordinate.z if AxisType.Z in axes else None,
            channel_id=coordinate.get_channel_id(),
        )

    @staticmethod
    def _merge_coordinates(base: Coordinate, update: Coordinate) -> Coordinate:
        """
        Merge a partial Coordinate into an existing Coordinate.

        Parameters
        ----------
        base
            Existing coordinate state.
        update
            New coordinate values. None values leave the corresponding base axis
            unchanged.

        Returns
        -------
        Coordinate
            Merged coordinate.
        """
        update_channel_id = update.get_channel_id()
        return Coordinate(
            x=base.x if update.x is None else update.x,
            y=base.y if update.y is None else update.y,
            z=base.z if update.z is None else update.z,
            channel_id=base.get_channel_id() if update_channel_id is None else update_channel_id,
        )

    def _update_current_pos(self, coordinate: Coordinate) -> None:
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
        self._current_pos = self._merge_coordinates(base=self._current_pos, update=coordinate)

    def _require_ready(self, action: str) -> None:
        """
        Raise when a hardware action is not allowed by current readiness checks.

        Parameters
        ----------
        action
            Human-readable action name used in exception messages.

        Returns
        -------
        None
        """
        if self._check_initialised and not self._is_initialised:
            raise RuntimeError(f"Stage.{action}: stage is not initialised.")
        if self._check_alive and not self.is_alive():
            raise RuntimeError(f"Stage.{action}: stage is not alive.")

    def initialise(self, force: bool = False) -> None:
        """
        Initialise the stage and cache its current hardware coordinates.

        Parameters
        ----------
        force
            If True, run initialisation even if the stage is already initialised.

        Returns
        -------
        None
        """
        if self._is_initialised and not force:
            return
        self._is_initialised = self._initialise(force=force)
        if self._check_initialised and not self._is_initialised:
            raise RuntimeError("Stage.initialise: stage failed to initialise.")
        self._is_alive = self._check_is_alive()
        if self._check_alive and not self._is_alive:
            raise RuntimeError("Stage.initialise: stage is not alive after initialisation.")
        if self._is_alive:
            self._current_pos = self._get_coordinates()

    def finalise(self, force: bool = False) -> None:
        """
        Finalise the stage and clear lifecycle flags.

        Parameters
        ----------
        force
            If True, subclass implementations may force cleanup.

        Returns
        -------
        None
        """
        self._finalise(force=force)
        self._is_initialised = False
        self._is_alive = False

    def is_alive(self) -> bool:
        """
        Query whether the stage hardware is alive.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            True when the subclass reports the hardware is alive.
        """
        self._is_alive = self._check_is_alive()
        return self._is_alive

    def is_initialised(self) -> bool:
        """
        Return whether initialise has succeeded.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            True when the stage is marked initialised.
        """
        return self._is_initialised

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
        axes_norm = self._normalise_axes(axes=axes)
        if query_hardware:
            self._require_ready(action="get_coordinates")
            self._update_current_pos(coordinate=self._get_coordinates())
        return self._filter_coordinate(coordinate=self._current_pos, axes=axes_norm)

    def get_pos(self) -> int:
        """
        Return the current position ID.

        Parameters
        ----------
        None

        Returns
        -------
        int
            Current position ID, or UNKNOWN_POSITION_ID when the current coordinate
            does not correspond to a known position ID.
        """
        return self._curr_pos

    def get_delta_fov(self) -> float:
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
        return self._delta_fov

    def set_delta_fov(self, delta_fov: float) -> None:
        """
        Set the field-of-view movement step.

        Parameters
        ----------
        delta_fov
            Positive movement step size.

        Returns
        -------
        None
        """
        if delta_fov <= 0:
            raise ValueError(f"Stage.set_delta_fov: delta_fov must be positive, received {delta_fov}.")
        self._delta_fov = delta_fov

    def get_stage_limits(self) -> tuple[Coordinate, Coordinate]:
        """
        Return lower and upper hardware coordinate limits.

        Parameters
        ----------
        None

        Returns
        -------
        tuple[Coordinate, Coordinate]
            Minimum and maximum coordinates. Both Coordinates must contain X, Y, and Z.
        """
        self._require_ready(action="get_stage_limits")
        low, high = self._get_stage_limits()
        missing_axes = [
            axis for axis in self.AXES
            if self._axis_value(low, axis) is None or self._axis_value(high, axis) is None
        ]
        if missing_axes:
            raise RuntimeError(f"Stage.get_stage_limits: limits must contain X, Y, and Z. Missing {missing_axes}.")
        return low.copy(), high.copy()

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
        low, high = self.get_stage_limits()
        return any(
            self._axis_value(coordinate, axis) is not None and
            (
                self._axis_value(coordinate, axis) < self._axis_value(low, axis) or
                self._axis_value(coordinate, axis) > self._axis_value(high, axis)
            )
            for axis in self.AXES
        )

    def set_pos_id_to_coordinate(
            self,
            pos_id_to_coordinate: dict[int, Coordinate],
            use_autofocus: bool,
    ) -> bool:
        """
        Register position IDs for later movement.

        Parameters
        ----------
        pos_id_to_coordinate
            Mapping from position ID to stage Coordinate.
        use_autofocus
            If True, registered coordinates must not contain Z. If False, they must
            contain Z.

        Returns
        -------
        bool
            True when the complete mapping is valid and stored, otherwise False.
        """
        coordinates: dict[int, Coordinate] = {}
        for i_pos, coordinate in pos_id_to_coordinate.items():
            if not isinstance(coordinate, Coordinate):
                raise TypeError(f"Stage.set_pos_id_to_coordinate: position {i_pos} is not a Coordinate.")
            if (not use_autofocus) and (not coordinate.has_z()):
                return False
            if use_autofocus and coordinate.has_z():
                return False
            if self.coordinate_is_out_of_bounds(coordinate):
                return False
            coordinates[i_pos] = coordinate.copy()
        self._pos_id_to_coordinate = coordinates
        return True

    def move_to_pos(self, i_pos: int, block: bool = True) -> None:
        """
        Move to a registered position ID.

        Parameters
        ----------
        i_pos
            Registered position ID.
        block
            If True, wait until the hardware move has completed.

        Returns
        -------
        None
        """
        self.move_to(coordinate=i_pos, block=block)

    def move_to_id(self, i_pos: int, block: bool = True) -> None:
        """
        Move to a registered position ID.

        Parameters
        ----------
        i_pos
            Registered position ID.
        block
            If True, wait until the hardware move has completed.

        Returns
        -------
        None
        """
        self.move_to(coordinate=i_pos, block=block)

    def move_to(self, coordinate: int | Coordinate, block: bool = True) -> None:
        """
        Move the stage either to a registered position ID or to a Coordinate.

        Parameters
        ----------
        coordinate
            Position ID or full/partial Coordinate. A partial Coordinate moves only
            the axes that are not None.
        block
            If True, wait until the hardware move has completed.

        Returns
        -------
        None
        """
        self._require_ready(action="move_to")
        if isinstance(coordinate, int):
            if coordinate not in self._pos_id_to_coordinate:
                raise IndexError(f"Stage.move_to: position index {coordinate} out of range.")
            pos_id = coordinate
            move_coordinate = self._pos_id_to_coordinate[coordinate].copy()
        elif isinstance(coordinate, Coordinate):
            pos_id = self.UNKNOWN_POSITION_ID
            move_coordinate = coordinate.copy()
        else:
            raise TypeError(f"Stage.move_to: expected int or Coordinate, received {type(coordinate)}.")
        if not self._coordinate_has_value(coordinate=move_coordinate):
            return
        if self.coordinate_is_out_of_bounds(coordinate=move_coordinate):
            raise ValueError(f"Stage.move_to: coordinate is out of bounds: {move_coordinate}.")
        self._update_current_pos(coordinate=self._move(coordinate=move_coordinate, block=block))
        self._curr_pos = pos_id

    def _move_fov(self, axis: AxisType, sign: int, multiplier: float | None = 1.0, block: bool = False) -> None:
        """
        Move by a field-of-view step along one X/Y axis.

        Parameters
        ----------
        axis
            AxisType.X or AxisType.Y.
        sign
            Direction of movement, typically -1 or +1.
        multiplier
            Scale factor for delta_fov. None is treated as 1.0.
        block
            If True, wait until the hardware move has completed.

        Returns
        -------
        None
        """
        if axis not in [AxisType.X, AxisType.Y]:
            raise ValueError(f"Stage._move_fov: FoV moves only support X/Y axes, received {axis}.")
        current = self.get_coordinates(axes=[axis], query_hardware=True)
        current_value = self._axis_value(coordinate=current, axis=axis)
        if current_value is None:
            raise RuntimeError(f"Stage._move_fov: current {axis} coordinate is not initialised.")
        multiplier = 1.0 if multiplier is None else multiplier
        target_value = current_value + int(sign * self.get_delta_fov() * multiplier)
        self.move_to(
            coordinate=self._coordinate_from_axis(
                axis=axis,
                value=target_value,
                channel_id=current.get_channel_id(),
            ),
            block=block,
        )

    def move_home(self, block: bool = False) -> None:
        """
        Move the stage to its hardware home position.

        Parameters
        ----------
        block
            If True, wait until the hardware move has completed.

        Returns
        -------
        None
        """
        self._require_ready(action="move_home")
        self._current_pos = self._home(block=block)
        self._curr_pos = 0

    def move_fov_up(self, multiplier: float | None = 1.0, block: bool = False) -> None:
        """
        Move one field of view up in camera/stage convention.

        Parameters
        ----------
        multiplier
            Scale factor for delta_fov. None is treated as 1.0.
        block
            If True, wait until the hardware move has completed.

        Returns
        -------
        None
        """
        self._move_fov(axis=AxisType.Y, sign=-1, multiplier=multiplier, block=block)

    def move_fov_down(self, multiplier: float | None = 1.0, block: bool = False) -> None:
        """
        Move one field of view down in camera/stage convention.

        Parameters
        ----------
        multiplier
            Scale factor for delta_fov. None is treated as 1.0.
        block
            If True, wait until the hardware move has completed.

        Returns
        -------
        None
        """
        self._move_fov(axis=AxisType.Y, sign=+1, multiplier=multiplier, block=block)

    def move_fov_left(self, multiplier: float | None = 1.0, block: bool = False) -> None:
        """
        Move one field of view left in camera/stage convention.

        Parameters
        ----------
        multiplier
            Scale factor for delta_fov. None is treated as 1.0.
        block
            If True, wait until the hardware move has completed.

        Returns
        -------
        None
        """
        self._move_fov(axis=AxisType.X, sign=-1, multiplier=multiplier, block=block)

    def move_fov_right(self, multiplier: float | None = 1.0, block: bool = False) -> None:
        """
        Move one field of view right in camera/stage convention.

        Parameters
        ----------
        multiplier
            Scale factor for delta_fov. None is treated as 1.0.
        block
            If True, wait until the hardware move has completed.

        Returns
        -------
        None
        """
        self._move_fov(axis=AxisType.X, sign=+1, multiplier=multiplier, block=block)

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
        self._current_pos = self._zero_coordinates()
        self._curr_pos = self.UNKNOWN_POSITION_ID

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
        describe the stage position after the move; returning the requested partial
        Coordinate is acceptable when the hardware cannot cheaply report the final
        full position.

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
        Move the hardware stage to its home position.

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


@dataclass
class StageConfig:
    """Configuration object used by StageFactory to create Stage instances."""

    binding: StageBindingType
    delta_fov: float
    name: str | None = None
    check_initialised: bool = True
    check_alive: bool = True
    initial_coordinate: Coordinate | None = None
    stage_limits: tuple[Coordinate, Coordinate] | None = None
    card_address_crisp: int | None = None

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

        if config.binding == StageBindingType.VIRTUAL:
            from evomachine.bindings.virtual.peripheralcontroller import VirtualPeripheralController
            from evomachine.bindings.virtual.stage import VirtualStage

            peripheral_ctrl = get_peripheral_controller(
                peripheral_controllers=peripheral_controllers,
                controller_type=VirtualPeripheralController,
                action="StageFactory.create",
            )
            return VirtualStage(
                peripheral_ctrl=peripheral_ctrl,
                delta_fov=config.delta_fov,
                name=config.name or "Virtual Stage",
                initial_coordinate=config.initial_coordinate,
                stage_limits=config.stage_limits,
                check_initialised=config.check_initialised,
                check_alive=config.check_alive,
            )

        if config.binding == StageBindingType.ASI_TIGER:
            from evomachine.bindings.asitiger.peripheralcontroller import TigerPeripheralController
            from evomachine.bindings.asitiger.stage import TigerStage

            peripheral_ctrl = get_peripheral_controller(
                peripheral_controllers=peripheral_controllers,
                controller_type=TigerPeripheralController,
                action="StageFactory.create",
            )
            return TigerStage(
                peripheral_ctrl=peripheral_ctrl,
                delta_fov=config.delta_fov,
                name=config.name or "ASI Tiger Stage",
                card_address_crisp=config.card_address_crisp,
                check_initialised=config.check_initialised,
                check_alive=config.check_alive,
            )

        raise ValueError(f"StageFactory.create: unsupported stage binding {config.binding}.")
