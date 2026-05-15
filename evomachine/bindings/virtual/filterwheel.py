from evomachine.bindings.virtual.peripheralcontroller import VirtualPeripheralController
from evomachine.filterwheel import FilterWheel
from evomachine.types import FilterWheelType


class VirtualFilterWheel(FilterWheel):
    """
    In-memory filter wheel implementation for debugging without hardware.

    VirtualFilterWheel uses the same public FilterWheel API as hardware-backed
    filter wheels, but all state is stored locally.
    """

    def __init__(
            self,
            peripheral_ctrl: VirtualPeripheralController,
            available_filters: list[FilterWheelType],
            name: str = "Virtual Filter Wheel",
            current_filter_type: FilterWheelType = FilterWheelType.UNKNOWN,
            check_initialised: bool = True,
            check_alive: bool = True,
    ):
        """
        Initialise an in-memory virtual filter wheel.

        Parameters
        ----------
        peripheral_ctrl
            VirtualPeripheralController that owns the simulated controller lifecycle.
        available_filters
            Non-empty list of filters that can be set.
        name
            Human-readable filter wheel name.
        current_filter_type
            Simulated current hardware position returned by initialise().
        check_initialised
            If True, inherited public methods require successful initialise().
        check_alive
            If True, inherited public methods require the filter wheel to report alive.

        Returns
        -------
        None
        """
        if not isinstance(peripheral_ctrl, VirtualPeripheralController):
            raise TypeError(
                f"VirtualFilterWheel.__init__: peripheral_ctrl must be VirtualPeripheralController, "
                f"received {type(peripheral_ctrl)}."
            )
        if not isinstance(current_filter_type, FilterWheelType):
            raise TypeError(
                f"VirtualFilterWheel.__init__: current_filter_type must be FilterWheelType, "
                f"received {type(current_filter_type)}."
            )
        self.peripheral_ctrl: VirtualPeripheralController = peripheral_ctrl
        self._virtual_filter_type: FilterWheelType = current_filter_type
        super().__init__(
            name=name,
            available_filters=available_filters,
            check_initialised=check_initialised,
            check_alive=check_alive,
        )

    def _initialise(self, force: bool = False) -> bool:
        """
        Mark the virtual filter wheel as initialised.

        Parameters
        ----------
        force
            Present for API compatibility. It does not change VirtualFilterWheel behavior.

        Returns
        -------
        bool
            Always True.
        """
        return self.peripheral_ctrl.is_alive()

    def _finalise(self, force: bool = False) -> None:
        """
        Finalise the virtual filter wheel.

        Parameters
        ----------
        force
            Present for API compatibility. It does not change VirtualFilterWheel behavior.

        Returns
        -------
        None
        """
        return

    def _check_is_alive(self) -> bool:
        """
        Report that the in-memory filter wheel is alive.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            Always True.
        """
        return self.peripheral_ctrl.is_alive()

    def _get_filter_wheel(self) -> FilterWheelType:
        """
        Return the current in-memory filter wheel position.

        Parameters
        ----------
        None

        Returns
        -------
        FilterWheelType
            Current stored filter wheel type.
        """
        return self._virtual_filter_type

    def _set_filter_wheel(self, filter_type: FilterWheelType) -> None:
        """
        Set the current in-memory filter wheel position.

        Parameters
        ----------
        filter_type
            Filter wheel type to store.

        Returns
        -------
        None
        """
        self._virtual_filter_type = filter_type
