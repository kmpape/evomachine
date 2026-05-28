from evomachine.bindings.asitiger.peripheralcontroller import TigerPeripheralController
from evomachine.coordinates import Coordinate
from evomachine.peripherals.stage import Stage


class FakeTigerStageController:
    """Deterministic Tiger-like controller for stage tests and dry runs."""

    def __init__(self):
        """
        Initialise fake Tiger stage state.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.coordinates = {"X": 0, "Y": 0, "Z": 0}
        self.connection = None
        self.move_calls: list[dict[str, float | int]] = []
        self.wait_calls: list[int | None] = []
        self.halt_was_called = False
        self.home_was_called = False
        self.zero_was_called = False

    def status(self) -> bool:
        """Return True to indicate that the fake controller is alive."""
        return True

    def where(self) -> dict[str, float | int]:
        """Return a copy of the current fake coordinates."""
        return self.coordinates.copy()

    def get_stage_limits(self) -> dict[str, tuple[float | int, float | int]]:
        """Return broad fake stage limits for X, Y, and Z."""
        return {"X": (-1000, 1000), "Y": (-1000, 1000), "Z": (-1000, 1000)}

    def move(self, coordinates: dict[str, float | int]) -> None:
        """Record and apply a fake move command."""
        self.move_calls.append(coordinates.copy())
        self.coordinates.update(coordinates)

    def home(self) -> None:
        """Record homing and reset fake coordinates to the origin."""
        self.home_was_called = True
        self.coordinates = {"X": 0, "Y": 0, "Z": 0}

    def wait_until_idle(self, card_address_crisp: int | None = None) -> None:
        """Record a fake wait-until-idle call."""
        self.wait_calls.append(card_address_crisp)

    def halt(self) -> None:
        """Record a fake halt command."""
        self.halt_was_called = True

    def zero(self) -> None:
        """Record zeroing and reset fake coordinates to the origin."""
        self.zero_was_called = True
        self.coordinates = {"X": 0, "Y": 0, "Z": 0}


class TigerStage(Stage):
    """
    Stage implementation backed by an ASI TigerController.

    The class receives an already-created TigerController-like object and uses it
    for all hardware access. It does not create serial connections itself.
    """

    def __init__(
            self,
            peripheral_ctrl: TigerPeripheralController,
            fov_step_size: float,
            name: str = "ASI Tiger Stage",
            check_initialised: bool = True,
            check_alive: bool = True,
    ):
        """
        Initialise a Tiger-backed stage.

        Parameters
        ----------
        peripheral_ctrl
            TigerPeripheralController that owns the shared Tiger connection.
        fov_step_size
            Field-of-view movement size in ASI stage units.
        name
            Human-readable stage name.
        check_initialised
            If True, inherited public methods require successful initialise().
        check_alive
            If True, inherited public methods require a live controller.
        Returns
        -------
        None
        """
        if not isinstance(peripheral_ctrl, TigerPeripheralController):
            raise TypeError(
                f"TigerStage.__init__: peripheral_ctrl must be TigerPeripheralController, "
                f"received {type(peripheral_ctrl)}."
        )
        self.peripheral_ctrl: TigerPeripheralController = peripheral_ctrl
        self.tiger = self.peripheral_ctrl.tiger
        super().__init__(
            name=name,
            fov_step_size=fov_step_size,
            check_initialised=check_initialised,
            check_alive=check_alive,
        )

    @staticmethod
    def _coordinate_from_tiger_dict(coordinates: dict[str, float | int]) -> Coordinate:
        """
        Convert a Tiger coordinate dictionary into a Coordinate.

        Parameters
        ----------
        coordinates
            Dictionary keyed by Tiger axis names X, Y, and Z.

        Returns
        -------
        Coordinate
            Coordinate containing any axes present in the dictionary.
        """
        return Coordinate.from_dict(coordinates)

    @staticmethod
    def _coordinate_to_tiger_dict(coordinate: Coordinate) -> dict[str, float | int]:
        """
        Convert a Coordinate into a Tiger coordinate dictionary.

        Parameters
        ----------
        coordinate
            Full or partial Coordinate.

        Returns
        -------
        dict[str, float | int]
            Dictionary containing only axes that are not None.
        """
        return coordinate.to_dict()

    def _wait_until_idle(self) -> None:
        """
        Wait until the Tiger controller reports idle.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.tiger.wait_until_idle(card_address_crisp=self.peripheral_ctrl.card_address_crisp)

    def _initialise(self, force: bool = False) -> bool:
        """
        Check that the supplied Tiger controller is ready.

        Parameters
        ----------
        force
            Present for the Stage interface. Since this class receives an existing
            controller, force does not recreate the connection.

        Returns
        -------
        bool
            True when the Tiger controller responds to status().
        """
        return self.peripheral_ctrl.is_alive()

    def _finalise(self, force: bool = False) -> None:
        """
        Finalise the Tiger-backed stage.

        Parameters
        ----------
        force
            Present for API compatibility. Tiger connection ownership belongs to
            the peripheral controller.

        Returns
        -------
        None
        """
        return

    def _check_is_alive(self) -> bool:
        """
        Return whether the Tiger controller responds to a status query.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            True when status() succeeds, otherwise False.
        """
        return self.peripheral_ctrl.is_alive()

    def _get_coordinates(self) -> Coordinate:
        """
        Query the current X/Y/Z Tiger stage coordinates.

        Parameters
        ----------
        None

        Returns
        -------
        Coordinate
            Current stage coordinate reported by Tiger.
        """
        return self._coordinate_from_tiger_dict(self.tiger.where())

    def _get_stage_limits(self) -> tuple[Coordinate, Coordinate]:
        """
        Query the configured Tiger stage limits.

        Parameters
        ----------
        None

        Returns
        -------
        tuple[Coordinate, Coordinate]
            Lower and upper coordinate limits for X, Y, and Z.
        """
        limits = self.tiger.get_stage_limits()
        return (
            Coordinate(limits["X"][0], limits["Y"][0], limits["Z"][0]),
            Coordinate(limits["X"][1], limits["Y"][1], limits["Z"][1]),
        )

    def _move(self, coordinate: Coordinate, block: bool = True) -> Coordinate:
        """
        Move the Tiger stage to a full or partial Coordinate.

        Parameters
        ----------
        coordinate
            Full or partial target coordinate. Axes set to None are not sent.
        block
            If True, wait until Tiger is idle and return the queried final
            coordinate. If False, return the requested partial coordinate.

        Returns
        -------
        Coordinate
            Coordinate reached or requested.
        """
        coordinates = self._coordinate_to_tiger_dict(coordinate)
        if not coordinates:
            return Coordinate.none_coordinate()
        self.tiger.move(coordinates=coordinates)
        if block:
            self._wait_until_idle()
            return self._get_coordinates()
        return coordinate.copy()

    def _home(self, block: bool = False) -> Coordinate:
        """
        Home the Tiger stage.

        Parameters
        ----------
        block
            If True, wait until Tiger is idle and return the queried final
            coordinate. If False, cache the expected home coordinate.

        Returns
        -------
        Coordinate
            Coordinate after homing, or expected home coordinate when not blocking.
        """
        self.tiger.home()
        if block:
            self._wait_until_idle()
            return self._get_coordinates()
        return Coordinate(0, 0, 0)

    def halt(self) -> None:
        """
        Halt Tiger stage motion immediately.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self._require_ready(action="halt")
        self.peripheral_ctrl.stop()

    def _zero_coordinates(self) -> Coordinate:
        """
        Zero Tiger's current coordinate system.

        Parameters
        ----------
        None

        Returns
        -------
        Coordinate
            Coordinate cached after zeroing.
        """
        self.tiger.zero()
        return Coordinate(0, 0, 0)
