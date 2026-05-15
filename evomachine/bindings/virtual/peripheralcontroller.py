from evomachine.peripherals import PeripheralController, PeripheralControllerConfig
from evomachine.types import PeripheralControllerBindingType


class VirtualPeripheralController(PeripheralController):
    """
    In-memory peripheral controller for debugging without hardware.
    """

    DEFAULT_NAME = "Virtual Peripheral Controller"

    def __init__(self, name: str = ""):
        self._stop_was_called: bool = False
        super().__init__(name=name or self.DEFAULT_NAME)

    @classmethod
    def default_config(cls) -> PeripheralControllerConfig:
        return PeripheralControllerConfig(
            binding=PeripheralControllerBindingType.VIRTUAL,
            name=cls.DEFAULT_NAME,
        )

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
