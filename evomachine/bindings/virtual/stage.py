from evomachine.bindings.virtual.peripheralcontroller import VirtualPeripheralController
from evomachine.coordinates import Coordinate, CoordinateBounds
from evomachine.stage import Stage


class VirtualStage(Stage):
    """
    In-memory stage implementation for debugging without stage hardware.

    VirtualStage uses the same public Stage API as hardware-backed stages, but all
    coordinates are stored locally and no commands are sent to external devices.
    """

    def __init__(
            self,
            peripheral_ctrl: VirtualPeripheralController,
            delta_fov: float,
            name: str = "Virtual Stage",
            initial_coordinate: Coordinate | None = None,
            coordinate_bounds: CoordinateBounds | None = None,
            check_initialised: bool = True,
            check_alive: bool = True,
    ):
        """
        Initialise an in-memory virtual stage.

        Parameters
        ----------
        peripheral_ctrl
            VirtualPeripheralController that owns the simulated controller lifecycle.
        delta_fov
            Field-of-view movement size in stage coordinate units.
        name
            Human-readable stage name.
        initial_coordinate
            Starting coordinate. If None, Coordinate(0, 0, 0) is used.
        coordinate_bounds
            Movement bounds. If None, broad default bounds are used.
        check_initialised
            If True, inherited public methods require successful initialise().
        check_alive
            If True, inherited public methods require the stage to report alive.

        Returns
        -------
        None
        """
        if not isinstance(peripheral_ctrl, VirtualPeripheralController):
            raise TypeError(
                f"VirtualStage.__init__: peripheral_ctrl must be VirtualPeripheralController, "
                f"received {type(peripheral_ctrl)}."
        )
        self.peripheral_ctrl: VirtualPeripheralController = peripheral_ctrl
        self._virtual_coordinate: Coordinate = initial_coordinate.copy() if initial_coordinate else Coordinate(0, 0, 0)
        self._stage_bounds: CoordinateBounds = coordinate_bounds.copy() if coordinate_bounds else CoordinateBounds(
            low=Coordinate(-1e7, -1e7, -1e7),
            high=Coordinate(1e7, 1e7, 1e7),
        )
        self._halt_was_called: bool = False
        super().__init__(
            name=name,
            delta_fov=delta_fov,
            coordinate_bounds=coordinate_bounds,
            check_initialised=check_initialised,
            check_alive=check_alive,
        )

    def halt_was_called(self) -> bool:
        """
        Return whether halt has been called.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            True after halt has been called at least once.
        """
        return self._halt_was_called

    def _initialise(self, force: bool = False) -> bool:
        """
        Mark the virtual stage as initialised.

        Parameters
        ----------
        force
            Present for API compatibility. It does not change VirtualStage behavior.

        Returns
        -------
        bool
            Always True.
        """
        return self.peripheral_ctrl.is_alive()

    def _finalise(self, force: bool = False) -> None:
        """
        Finalise the virtual stage.

        Parameters
        ----------
        force
            Present for API compatibility. It does not change VirtualStage behavior.

        Returns
        -------
        None
        """
        return

    def _check_is_alive(self) -> bool:
        """
        Report that the in-memory stage is alive.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            Always True.
        """
        return self.peripheral_ctrl.is_alive()

    def _get_coordinates(self) -> Coordinate:
        """
        Return the current in-memory coordinate.

        Parameters
        ----------
        None

        Returns
        -------
        Coordinate
            Current stored coordinate.
        """
        return self._virtual_coordinate.copy()

    def _get_stage_limits(self) -> tuple[Coordinate, Coordinate]:
        """
        Return the configured in-memory stage limits.

        Parameters
        ----------
        None

        Returns
        -------
        tuple[Coordinate, Coordinate]
            Lower and upper coordinate limits.
        """
        return self._stage_bounds.as_limits()

    def _move(self, coordinate: Coordinate, block: bool = True) -> Coordinate:
        """
        Move the in-memory stage by merging the requested coordinate.

        Parameters
        ----------
        coordinate
            Full or partial target coordinate.
        block
            Present for API compatibility. It does not change VirtualStage behavior.

        Returns
        -------
        Coordinate
            New stored coordinate after the move.
        """
        self._virtual_coordinate = self._merge_coordinates(base=self._virtual_coordinate, update=coordinate)
        return self._virtual_coordinate.copy()

    def _home(self, block: bool = False) -> Coordinate:
        """
        Move the in-memory stage to the origin.

        Parameters
        ----------
        block
            Present for API compatibility. It does not change VirtualStage behavior.

        Returns
        -------
        Coordinate
            Coordinate(0, 0, 0).
        """
        self._virtual_coordinate = Coordinate(0, 0, 0)
        return self._virtual_coordinate.copy()

    def halt(self) -> None:
        """
        Mark that halt was called.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self._require_ready(action="halt")
        self.peripheral_ctrl.stop()
        self._halt_was_called = True

    def _zero_coordinates(self) -> Coordinate:
        """
        Zero the in-memory coordinate system.

        Parameters
        ----------
        None

        Returns
        -------
        Coordinate
            Coordinate(0, 0, 0).
        """
        self._virtual_coordinate = Coordinate(0, 0, 0)
        return self._virtual_coordinate.copy()
