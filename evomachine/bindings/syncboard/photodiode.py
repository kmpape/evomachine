from __future__ import annotations

from evomachine.bindings.syncboard.peripheralcontroller import SyncBoardPeripheralController
from evomachine.photodiode import Photodiode, PhotodiodeReadingRange


class SyncBoardPhotodiode(Photodiode):
    """Photodiode implementation backed by a SyncBoard."""

    DEFAULT_NAME = "SyncBoard Photodiode"

    def __init__(
            self,
            peripheral_ctrl: SyncBoardPeripheralController,
            channel: int = 8,
            reading_range: PhotodiodeReadingRange | None = None,
            name: str = DEFAULT_NAME,
            check_initialised: bool = True,
            check_alive: bool = True,
    ):
        """
        Initialise a SyncBoard-backed photodiode.

        Parameters
        ----------
        peripheral_ctrl
            SyncBoardPeripheralController that owns the shared SyncBoard
            connection.
        channel
            SyncBoard photodiode channel identifier.
        reading_range
            Raw reading bounds used to scale raw readings to [0, 100]. If None,
            a default range of 0 to 1 is used.
        name
            Human-readable photodiode name.
        check_initialised
            If True, inherited public methods require successful initialise().
        check_alive
            If True, inherited public methods require the controller to report
            alive.

        Returns
        -------
        None
        """
        if not isinstance(peripheral_ctrl, SyncBoardPeripheralController):
            raise TypeError(
                f"SyncBoardPhotodiode.__init__: peripheral_ctrl must be SyncBoardPeripheralController, "
                f"received {type(peripheral_ctrl)}."
            )
        super().__init__(
            peripheral_ctrl=peripheral_ctrl,
            channel=channel,
            reading_range=reading_range or PhotodiodeReadingRange(0.0, 1.0),
            name=name,
            check_initialised=check_initialised,
            check_alive=check_alive,
        )

    @property
    def peripheral_ctrl(self) -> SyncBoardPeripheralController:
        """
        Return the SyncBoard peripheral controller.

        Parameters
        ----------
        None

        Returns
        -------
        SyncBoardPeripheralController
            Controller backing this SyncBoard photodiode.
        """
        return self._peripheral_ctrl

    @peripheral_ctrl.setter
    def peripheral_ctrl(self, peripheral_ctrl: SyncBoardPeripheralController) -> None:
        """
        Set the SyncBoard peripheral controller after type validation.

        Parameters
        ----------
        peripheral_ctrl
            SyncBoardPeripheralController backing this photodiode.

        Returns
        -------
        None
        """
        if not isinstance(peripheral_ctrl, SyncBoardPeripheralController):
            raise TypeError(
                f"SyncBoardPhotodiode.peripheral_ctrl: peripheral_ctrl must be SyncBoardPeripheralController, "
                f"received {type(peripheral_ctrl)}."
            )
        self._peripheral_ctrl = peripheral_ctrl

    def _initialise(self, force: bool = False) -> bool:
        """
        Mark the SyncBoard photodiode as initialised.

        Parameters
        ----------
        force
            Present for API compatibility. It does not change
            SyncBoardPhotodiode behavior.

        Returns
        -------
        bool
            True when the SyncBoard peripheral controller is alive.
        """
        return self.peripheral_ctrl.is_alive()

    def _finalise(self, force: bool = False) -> None:
        """
        Finalise the SyncBoard photodiode wrapper.

        Parameters
        ----------
        force
            Present for API compatibility. It does not change
            SyncBoardPhotodiode behavior.

        Returns
        -------
        None
        """
        return

    def _stop(self) -> None:
        """
        Stop the SyncBoard photodiode wrapper.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        return

    def _read_raw_photodiode(self) -> float:
        """
        Read one raw photodiode value from the SyncBoard.

        Parameters
        ----------
        None

        Returns
        -------
        float
            Raw SyncBoard photodiode reading.
        """
        raw_reading = self.peripheral_ctrl.syncboard.read_photodiode(channel=self.channel)
        if raw_reading is None:
            raise RuntimeError(
                f"SyncBoardPhotodiode._read_raw_photodiode: "
                f"received no reading for channel {self.channel}."
            )
        try:
            return float(raw_reading)
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                f"SyncBoardPhotodiode._read_raw_photodiode: "
                f"received malformed reading {raw_reading!r} for channel {self.channel}."
            ) from error
