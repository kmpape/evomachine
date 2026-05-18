from evomachine.bindings.asitiger.peripheralcontroller import TigerPeripheralController
from evomachine.filterwheel import FilterWheel
from evomachine.types import FilterWheelType


class FakeTigerFilterWheelController:
    """Deterministic Tiger-like controller for filter wheel tests."""

    def __init__(self):
        """
        Initialise fake filter wheel command recording.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.connection = None
        self.filter_wheel_calls: list[tuple[int, int]] = []

    def status(self) -> bool:
        """Return True to indicate that the fake controller is alive."""
        return True

    def halt(self) -> None:
        """Accept a fake halt command."""
        return

    def filter_wheel(self, position: int, card_address: int = 8) -> None:
        """Record a fake filter wheel command."""
        self.filter_wheel_calls.append((position, card_address))


class TigerFilterWheel(FilterWheel):
    """
    Filter wheel implementation backed by an ASI TigerController.

    The class receives an already-created TigerController-like object and uses it
    for all hardware access. It does not create serial connections itself.
    """

    DEFAULT_FILTER_WHEEL_SETTINGS: dict[FilterWheelType, int] = {
        FilterWheelType.FILTER: 0,
        FilterWheelType.FILTER_465nm: 1,
        FilterWheelType.FILTER_527nm: 2,
        FilterWheelType.FILTER_592nm: 3,
        FilterWheelType.NO_FILTER: 4,
        FilterWheelType.BLOCKING: 5,
    }

    def __init__(
            self,
            peripheral_ctrl: TigerPeripheralController,
            available_filters: list[FilterWheelType],
            name: str = "ASI Tiger Filter Wheel",
            filter_wheel_settings: dict[FilterWheelType, int] | None = None,
            check_initialised: bool = True,
            check_alive: bool = True,
    ):
        """
        Initialise a Tiger-backed filter wheel.

        Parameters
        ----------
        peripheral_ctrl
            TigerPeripheralController that owns the shared Tiger connection.
        available_filters
            Non-empty list of filters that can be set.
        name
            Human-readable filter wheel name.
        filter_wheel_settings
            Optional mapping from FilterWheelType to ASI Tiger filter wheel position.
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
                f"TigerFilterWheel.__init__: peripheral_ctrl must be TigerPeripheralController, "
                f"received {type(peripheral_ctrl)}."
        )
        self.peripheral_ctrl: TigerPeripheralController = peripheral_ctrl
        self.tiger = self.peripheral_ctrl.tiger
        self.filter_wheel_settings: dict[FilterWheelType, int] = (
            filter_wheel_settings.copy() if filter_wheel_settings else self.DEFAULT_FILTER_WHEEL_SETTINGS.copy()
        )
        self._validate_filter_wheel_settings(filter_wheel_settings=self.filter_wheel_settings)
        super().__init__(
            name=name,
            available_filters=available_filters,
            check_initialised=check_initialised,
            check_alive=check_alive,
        )

    @staticmethod
    def _validate_filter_wheel_settings(filter_wheel_settings: dict[FilterWheelType, int]) -> None:
        """
        Validate the Tiger filter-position mapping.

        Parameters
        ----------
        filter_wheel_settings
            Mapping from FilterWheelType to ASI Tiger integer wheel position.

        Returns
        -------
        None
        """
        if not isinstance(filter_wheel_settings, dict):
            raise TypeError(
                f"TigerFilterWheel._validate_filter_wheel_settings: expected dict, "
                f"received {type(filter_wheel_settings)}."
            )
        for filter_type, position in filter_wheel_settings.items():
            if not isinstance(filter_type, FilterWheelType):
                raise TypeError("TigerFilterWheel._validate_filter_wheel_settings: keys must be FilterWheelType.")
            if not isinstance(position, int):
                raise TypeError("TigerFilterWheel._validate_filter_wheel_settings: positions must be int.")

    def _initialise(self, force: bool = False) -> bool:
        """
        Check that the supplied Tiger controller is ready.

        Parameters
        ----------
        force
            Present for the FilterWheel interface. Since this class receives an
            existing controller, force does not recreate the connection.

        Returns
        -------
        bool
            True when the Tiger controller responds to status().
        """
        return self.peripheral_ctrl.is_alive()

    def _finalise(self, force: bool = False) -> None:
        """
        Finalise the Tiger-backed filter wheel.

        Tiger connection ownership belongs to the peripheral controller.

        Parameters
        ----------
        force
            Present for API compatibility. It does not change TigerFilterWheel behavior.

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

    def _get_filter_wheel(self) -> FilterWheelType:
        """
        Query the current Tiger filter wheel position.

        Parameters
        ----------
        None

        Returns
        -------
        FilterWheelType
            FilterWheelType.UNKNOWN until Tiger readback/parsing is implemented.
        """
        # TODO(Codex): Replace this placeholder once Tiger filter wheel readback
        # is finalised and parsed into FilterWheelType.
        return FilterWheelType.UNKNOWN

    def _set_filter_wheel(self, filter_type: FilterWheelType) -> None:
        """
        Set the Tiger filter wheel position.

        Parameters
        ----------
        filter_type
            Filter wheel type to set.

        Returns
        -------
        None
        """
        if filter_type not in self.filter_wheel_settings:
            raise ValueError(f"TigerFilterWheel._set_filter_wheel: no Tiger position for {filter_type}.")
        self.tiger.filter_wheel(
            position=self.filter_wheel_settings[filter_type],
            card_address=self.peripheral_ctrl.card_address_filter_wheel,
        )
