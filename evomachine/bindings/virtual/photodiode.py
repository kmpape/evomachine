from __future__ import annotations

from evomachine.bindings.virtual.peripheralcontroller import VirtualPeripheralController
from evomachine.photodiode import Photodiode, PhotodiodeReadingRange


class VirtualPhotodiode(Photodiode):
    """In-memory photodiode implementation for tests and dry runs."""

    DEFAULT_NAME = "Virtual Photodiode"

    def __init__(
            self,
            peripheral_ctrl: VirtualPeripheralController,
            channel: int = 8,
            reading_range: PhotodiodeReadingRange | None = None,
            name: str = DEFAULT_NAME,
            check_initialised: bool = True,
            check_alive: bool = True,
    ):
        """
        Initialise an in-memory virtual photodiode.

        Parameters
        ----------
        peripheral_ctrl
            VirtualPeripheralController that owns the simulated lifecycle.
        channel
            Simulated photodiode channel identifier.
        reading_range
            Raw reading bounds used to scale raw readings to [0, 100]. If None,
            a default range of 0 to 1 is used.
        name
            Human-readable photodiode name.
        check_initialised
            If True, inherited public methods require successful initialise().
        check_alive
            If True, inherited public methods require the photodiode to report
            alive.

        Returns
        -------
        None
        """
        if not isinstance(peripheral_ctrl, VirtualPeripheralController):
            raise TypeError(
                f"VirtualPhotodiode.__init__: peripheral_ctrl must be VirtualPeripheralController, "
                f"received {type(peripheral_ctrl)}."
            )
        resolved_range = reading_range or PhotodiodeReadingRange(0.0, 1.0)
        self._raw_reading: float = resolved_range.minimum_reading
        super().__init__(
            peripheral_ctrl=peripheral_ctrl,
            channel=channel,
            reading_range=resolved_range,
            name=name,
            check_initialised=check_initialised,
            check_alive=check_alive,
        )

    @property
    def peripheral_ctrl(self) -> VirtualPeripheralController:
        """
        Return the virtual peripheral controller.

        Parameters
        ----------
        None

        Returns
        -------
        VirtualPeripheralController
            Controller backing this virtual photodiode.
        """
        return self._peripheral_ctrl

    @peripheral_ctrl.setter
    def peripheral_ctrl(self, peripheral_ctrl: VirtualPeripheralController) -> None:
        """
        Set the virtual peripheral controller after type validation.

        Parameters
        ----------
        peripheral_ctrl
            VirtualPeripheralController backing this virtual photodiode.

        Returns
        -------
        None
        """
        if not isinstance(peripheral_ctrl, VirtualPeripheralController):
            raise TypeError(
                f"VirtualPhotodiode.peripheral_ctrl: peripheral_ctrl must be VirtualPeripheralController, "
                f"received {type(peripheral_ctrl)}."
            )
        self._peripheral_ctrl = peripheral_ctrl

    def set_raw_reading(self, reading: float) -> None:
        """
        Update the raw reading returned by the virtual photodiode.

        Parameters
        ----------
        reading
            Raw simulated photodiode reading before calibration.

        Returns
        -------
        None
        """
        self._raw_reading = PhotodiodeReadingRange._validate_reading(
            reading=reading,
            name="reading",
        )

    def _initialise(self, force: bool = False) -> bool:
        """
        Mark the virtual photodiode as initialised.

        Parameters
        ----------
        force
            Present for API compatibility. It does not change
            VirtualPhotodiode behavior.

        Returns
        -------
        bool
            True when the virtual peripheral controller is alive.
        """
        return self.peripheral_ctrl.is_alive()

    def _finalise(self, force: bool = False) -> None:
        """
        Finalise the virtual photodiode.

        Parameters
        ----------
        force
            Present for API compatibility. It does not change
            VirtualPhotodiode behavior.

        Returns
        -------
        None
        """
        return

    def _stop(self) -> None:
        """
        Stop the virtual peripheral controller.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.peripheral_ctrl.stop()

    def _read_raw_photodiode(self) -> float:
        """
        Return the current simulated raw photodiode reading.

        Parameters
        ----------
        None

        Returns
        -------
        float
            Simulated raw photodiode reading.
        """
        return self._raw_reading
