from evomachine.peripherals import PeripheralController


class VirtualPeripheralController(PeripheralController):
    """
    In-memory peripheral controller for debugging without hardware.
    """

    def __init__(self, name: str = "Virtual Peripheral Controller"):
        self._stop_was_called: bool = False
        super().__init__(name=name)

    def stop_was_called(self) -> bool:
        return self._stop_was_called

    def _initialise(self, force: bool = False) -> bool:
        return True

    def _check_is_alive(self) -> bool:
        return self._is_initialised

    def _stop(self) -> None:
        self._stop_was_called = True

    def _shutdown(self, force: bool = False) -> None:
        return
