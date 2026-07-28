from __future__ import annotations

from evomachine.bindings.virtual.peripheralcontroller import VirtualPeripheralController
from evomachine.peripherals.autofocus import Autofocus, AutofocusCalibrationConfig
from evomachine.types import AutoFocusStatusType


class VirtualAutofocus(Autofocus):
    """In-memory autofocus implementation for debugging without autofocus hardware."""

    DEFAULT_NAME = "Virtual Autofocus"

    def __init__(
            self,
            peripheral_ctrl: VirtualPeripheralController,
            name: str = DEFAULT_NAME,
            check_initialised: bool = True,
            check_alive: bool = True,
            initial_status: AutoFocusStatusType = AutoFocusStatusType.IDLE,
    ):
        """
        Initialise an in-memory virtual autofocus peripheral.

        Parameters
        ----------
        peripheral_ctrl
            VirtualPeripheralController that owns the simulated lifecycle.
        name
            Human-readable autofocus name.
        check_initialised
            If True, inherited public methods require successful initialise().
        check_alive
            If True, inherited public methods require the peripheral to report alive.
        initial_status
            Initial autofocus status returned before commands change it.

        Returns
        -------
        None
        """
        if not isinstance(peripheral_ctrl, VirtualPeripheralController):
            raise TypeError(
                f"VirtualAutofocus.__init__: peripheral_ctrl must be VirtualPeripheralController, "
                f"received {type(peripheral_ctrl)}."
            )
        if not isinstance(initial_status, AutoFocusStatusType):
            raise TypeError(
                f"VirtualAutofocus.__init__: initial_status must be AutoFocusStatusType, "
                f"received {type(initial_status)}."
            )
        self.peripheral_ctrl: VirtualPeripheralController = peripheral_ctrl
        self.command_history: list[str] = []
        self.is_configured: bool = False
        self._status: AutoFocusStatusType = initial_status
        super().__init__(
            name=name,
            check_initialised=check_initialised,
            check_alive=check_alive,
        )

    def _initialise(self, force: bool = False) -> bool:
        """
        Mark the virtual autofocus peripheral as initialised.

        Parameters
        ----------
        force
            Present for API compatibility. It does not change VirtualAutofocus behavior.

        Returns
        -------
        bool
            True when the virtual peripheral controller is alive.
        """
        return self.peripheral_ctrl.is_alive()

    def _finalise(self, force: bool = False) -> None:
        """
        Finalise the virtual autofocus peripheral.

        Parameters
        ----------
        force
            Present for API compatibility. It does not change VirtualAutofocus behavior.

        Returns
        -------
        None
        """
        return

    def _check_is_alive(self) -> bool:
        """
        Return whether the virtual peripheral controller is alive.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            True when the virtual peripheral controller is alive.
        """
        return self.peripheral_ctrl.is_alive()

    def _apply_config(self, config: AutofocusCalibrationConfig | None = None) -> bool:
        """
        Record a virtual configuration command.

        Parameters
        ----------
        config
            Ignored optional configuration object.

        Returns
        -------
        bool
            Always True.
        """
        self.command_history.append("configure")
        self.is_configured = True
        return True

    def _run_calibration(
            self,
            config: AutofocusCalibrationConfig | None = None,
            lock_after_calibration: bool = False,
    ) -> bool:
        """
        Record a virtual autofocus setup command.

        Parameters
        ----------
        config
            Ignored optional configuration object.
        lock_after_calibration
            If True, lock the virtual autofocus after setup.

        Returns
        -------
        bool
            Always True.
        """
        self.command_history.append("initialise_autofocus")
        self.is_configured = True
        if lock_after_calibration:
            self._lock()
        else:
            self._status = AutoFocusStatusType.READY
        return True

    def _lock(self) -> None:
        """
        Lock virtual autofocus.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.command_history.append("lock")
        self._status = AutoFocusStatusType.IN_FOCUS

    def _unlock(self) -> None:
        """
        Unlock virtual autofocus.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.command_history.append("unlock")
        self._status = AutoFocusStatusType.READY

    def _disable(self) -> None:
        """
        Disable virtual autofocus.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.command_history.append("disable")
        self._status = AutoFocusStatusType.IDLE

    def _get_status(self) -> AutoFocusStatusType:
        """
        Return the virtual autofocus status.

        Parameters
        ----------
        None

        Returns
        -------
        AutoFocusStatusType
            Current virtual status.
        """
        return self._status

    def _is_locked(self) -> bool:
        """
        Return whether virtual autofocus is locked.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            True when virtual status is IN_FOCUS or OUT_OF_FOCUS.
        """
        return self._status in {AutoFocusStatusType.IN_FOCUS, AutoFocusStatusType.OUT_OF_FOCUS}
