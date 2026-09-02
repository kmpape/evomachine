from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from evomachine.bindings.binding_types import BindingType


@dataclass
class PeripheralControllerConfig:
    """Configuration object used by PeripheralControllerFactory."""

    binding: BindingType
    name: str = ""
    initialise: bool = True

    def __post_init__(self) -> None:
        """
        Validate peripheral controller configuration after construction.

        Parameters
        ----------
        None

        Returns
        -------
        None
            The dataclass fields are validated in place.
        """
        if not isinstance(self.binding, BindingType):
            raise TypeError(
                f"PeripheralControllerConfig: binding must be BindingType, "
                f"received {type(self.binding)}."
            )
        if not isinstance(self.name, str):
            raise TypeError(f"PeripheralControllerConfig: name must be str, received {type(self.name)}.")
        if not isinstance(self.initialise, bool):
            raise TypeError(
                f"PeripheralControllerConfig: initialise must be bool, received {type(self.initialise)}."
            )

    def copy(self) -> "PeripheralControllerConfig":
        return type(self)(**self.__dict__)


@dataclass
class SerialPeripheralControllerConfig(PeripheralControllerConfig):
    """Configuration object for serial PeripheralController bindings."""

    port: str | None = None
    hwid: str | None = None
    close_on_shutdown: bool = True

    def __post_init__(self) -> None:
        """
        Validate serial peripheral controller configuration after construction.

        Parameters
        ----------
        None

        Returns
        -------
        None
            The dataclass fields are validated in place.
        """
        super().__post_init__()
        if self.binding not in {
            BindingType.ASI_TIGER,
            BindingType.SYNCBOARD,
            BindingType.KWR103,
        }:
            raise ValueError(
                f"SerialPeripheralControllerConfig: binding must be a serial binding, received {self.binding}."
            )
        if (self.port is None) == (self.hwid is None):
            raise ValueError("SerialPeripheralControllerConfig: exactly one of port or hwid must be provided.")
        if self.port is not None and not isinstance(self.port, str):
            raise TypeError(f"SerialPeripheralControllerConfig: port must be str or None, received {type(self.port)}.")
        if self.hwid is not None and not isinstance(self.hwid, str):
            raise TypeError(f"SerialPeripheralControllerConfig: hwid must be str or None, received {type(self.hwid)}.")
        if not isinstance(self.close_on_shutdown, bool):
            raise TypeError(
                f"SerialPeripheralControllerConfig: close_on_shutdown must be bool, "
                f"received {type(self.close_on_shutdown)}."
            )

    def resolve_port(self, display_name: str = "") -> str:
        """
        Return the configured serial port or resolve the configured HWID.

        Parameters
        ----------
        display_name
            Human-readable device name used in errors raised during HWID
            lookup.

        Returns
        -------
        str
            Serial port path to use when constructing the controller.
        """
        if self.port is not None:
            return self.port
        if self.hwid is None:
            raise ValueError("SerialPeripheralControllerConfig.resolve_port: hwid is required when port is absent.")

        from evomachine.com_ports import get_port

        return get_port(hwid=self.hwid, display_name=display_name)


@dataclass
class SocketPeripheralControllerConfig(PeripheralControllerConfig):
    """Configuration object for socket-backed PeripheralController bindings."""

    host: str = "127.0.0.1"
    port: int = 0
    close_on_shutdown: bool = True

    def __post_init__(self) -> None:
        """
        Validate socket peripheral controller configuration after construction.

        Parameters
        ----------
        None

        Returns
        -------
        None
            The dataclass fields are validated in place.
        """
        super().__post_init__()
        if not isinstance(self.host, str):
            raise TypeError(f"SocketPeripheralControllerConfig: host must be str, received {type(self.host)}.")
        if not isinstance(self.port, int):
            raise TypeError(f"SocketPeripheralControllerConfig: port must be int, received {type(self.port)}.")
        if not isinstance(self.close_on_shutdown, bool):
            raise TypeError(
                f"SocketPeripheralControllerConfig: close_on_shutdown must be bool, "
                f"received {type(self.close_on_shutdown)}."
            )


class PeripheralController(ABC):
    """Base class for objects that own peripheral connection lifecycle."""

    def __init__(self, name: str):
        """
        Initialise common peripheral controller state.

        Parameters
        ----------
        name
            Human-readable controller name used in error messages.

        Returns
        -------
        None
        """
        self.name: str = name
        self._is_initialised: bool = False
        self._is_alive: bool = False
        self.config: PeripheralControllerConfig | None = None

    def initialise(self, force: bool = False) -> None:
        """
        Initialise the underlying controller if needed.

        Parameters
        ----------
        force
            If True, initialise even when the controller is already marked
            initialised.

        Returns
        -------
        None
        """
        if self._is_initialised and not force:
            return
        self._is_initialised = self._initialise(force=force)
        self._is_alive = self._check_is_alive()
        if not self._is_initialised:
            raise RuntimeError(f"PeripheralController.initialise: {self.name} failed to initialise.")
        if not self._is_alive:
            raise RuntimeError(f"PeripheralController.initialise: {self.name} is not alive after initialisation.")

    def reinitialise(self, force: bool = False) -> None:
        """
        Shutdown and initialise the controller again.

        Parameters
        ----------
        force
            If True, force shutdown before reinitialising.

        Returns
        -------
        None
        """
        self.shutdown(force=force)
        self.initialise(force=True)

    def is_alive(self) -> bool:
        """
        Return whether the controller reports an active connection promptly.

        Binding implementations that perform I/O must enforce a transport-level
        timeout so a missing device cannot block application status reporting.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            True when the binding-specific health check succeeds.
        """
        self._is_alive = self._check_is_alive()
        return self._is_alive

    def is_initialised(self) -> bool:
        """
        Return whether the controller is marked initialised.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            True when initialise() has succeeded.
        """
        return self._is_initialised

    def stop(self) -> None:
        """
        Stop controller activity without closing its connection.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self._stop()

    def shutdown(self, force: bool = False) -> None:
        """
        Stop the controller and release its underlying resources.

        Parameters
        ----------
        force
            If True, binding implementations should release resources even when
            configured to keep them open.

        Returns
        -------
        None
        """
        stop_error: Exception | None = None
        try:
            self.stop()
        except Exception as error:
            stop_error = error
        finally:
            try:
                self._shutdown(force=force)
            finally:
                self._is_initialised = False
                self._is_alive = False
        if stop_error is not None:
            raise stop_error

    @abstractmethod
    def _initialise(self, force: bool = False) -> bool:
        """
        Run binding-specific initialisation.

        Parameters
        ----------
        force
            If True, force binding-specific initialisation work.

        Returns
        -------
        bool
            True when initialisation succeeded.
        """
        raise NotImplementedError

    @abstractmethod
    def _check_is_alive(self) -> bool:
        """
        Run a binding-specific, bounded health check.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            True when the controller is alive.
        """
        raise NotImplementedError

    @abstractmethod
    def _stop(self) -> None:
        """
        Run binding-specific stop behavior.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        raise NotImplementedError

    @abstractmethod
    def _shutdown(self, force: bool = False) -> None:
        """
        Run binding-specific resource cleanup.

        Parameters
        ----------
        force
            If True, force cleanup even when normally configured to keep
            resources open.

        Returns
        -------
        None
        """
        raise NotImplementedError


class SerialPeripheralController(PeripheralController):
    """Base class for PeripheralController implementations backed by a serial connection."""

    def __init__(
            self,
            name: str,
            close_on_shutdown: bool = True,
    ):
        """
        Initialise common serial controller lifecycle state.

        Parameters
        ----------
        name
            Human-readable controller name used in error messages.
        close_on_shutdown
            If True, shutdown closes the underlying serial connection. Force
            shutdown closes it regardless of this setting.

        Returns
        -------
        None
        """
        self.close_on_shutdown: bool = close_on_shutdown
        super().__init__(name=name)

    @abstractmethod
    def _get_serial_controller(self) -> Any:
        """
        Return the wrapped binding-specific serial controller object.

        Parameters
        ----------
        None

        Returns
        -------
        Any
            Object that owns or exposes the binding's serial connection.
        """
        raise NotImplementedError

    def _get_connection(self) -> Any:
        """
        Return the object that provides the serial disconnect method.

        Parameters
        ----------
        None

        Returns
        -------
        Any
            Connection-like object with a disconnect method.
        """
        return self._get_serial_controller().connection

    def _before_disconnect(self, force: bool = False) -> None:
        """
        Run binding-specific shutdown work before closing the connection.

        Parameters
        ----------
        force
            True when shutdown should close the connection even if
            close_on_shutdown is False.

        Returns
        -------
        None
        """
        return

    def _disconnect(self) -> None:
        """
        Close the wrapped serial connection.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        try:
            connection = self._get_connection()
            disconnect = connection.disconnect
        except AttributeError as error:
            raise TypeError(
                f"{type(self).__name__}._disconnect: serial connection must expose disconnect()."
            ) from error
        disconnect()

    def _shutdown(self, force: bool = False) -> None:
        """
        Run serial shutdown hooks and optionally close the connection.

        Parameters
        ----------
        force
            True to close the serial connection regardless of
            close_on_shutdown.

        Returns
        -------
        None
        """
        self._before_disconnect(force=force)
        if not (force or self.close_on_shutdown):
            return
        self._disconnect()


class SocketPeripheralController(PeripheralController):
    """Base class for PeripheralController implementations backed by a socket connection."""

    def __init__(
            self,
            name: str,
            close_on_shutdown: bool = True,
    ):
        """
        Initialise common socket controller lifecycle state.

        Parameters
        ----------
        name
            Human-readable controller name used in error messages.
        close_on_shutdown
            If True, shutdown closes the underlying socket. Force shutdown closes
            it regardless of this setting.

        Returns
        -------
        None
        """
        self.close_on_shutdown: bool = close_on_shutdown
        super().__init__(name=name)

    @abstractmethod
    def _get_socket_controller(self) -> Any:
        """
        Return the wrapped binding-specific socket controller object.

        Parameters
        ----------
        None

        Returns
        -------
        Any
            Object that owns or exposes the binding's socket connection.
        """
        raise NotImplementedError

    def _get_socket(self) -> Any:
        """
        Return the socket-like object exposed by the wrapped controller.

        Parameters
        ----------
        None

        Returns
        -------
        Any
            Socket-like object with a close method.
        """
        controller = self._get_socket_controller()
        try:
            return controller.s
        except AttributeError:
            return controller.socket

    def _before_disconnect(self, force: bool = False) -> None:
        """
        Run binding-specific shutdown work before closing the socket.

        Parameters
        ----------
        force
            True when shutdown should close the socket even if
            close_on_shutdown is False.

        Returns
        -------
        None
        """
        return

    def _disconnect(self) -> None:
        """
        Close the wrapped socket.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        try:
            socket = self._get_socket()
            close = socket.close
        except AttributeError as error:
            raise TypeError(f"{type(self).__name__}._disconnect: socket must expose close().") from error
        close()

    def _shutdown(self, force: bool = False) -> None:
        """
        Run socket shutdown hooks and optionally close the socket.

        Parameters
        ----------
        force
            True to close the socket regardless of close_on_shutdown.

        Returns
        -------
        None
        """
        self._before_disconnect(force=force)
        if not (force or self.close_on_shutdown):
            return
        self._disconnect()


class PeripheralControllerFactory:
    """Factory for creating PeripheralController instances from typed configs."""

    @staticmethod
    def create(config: PeripheralControllerConfig, **binding_options: Any) -> PeripheralController:
        """
        Create a peripheral controller for the requested binding.

        Parameters
        ----------
        config
            Typed controller configuration that selects the binding and lifecycle
            behavior.
        **binding_options
            Binding-specific keyword options forwarded to the selected
            controller constructor.

        Returns
        -------
        PeripheralController
            Initialised or uninitialised controller according to config.
        """
        if not isinstance(config, PeripheralControllerConfig):
            raise TypeError(
                f"PeripheralControllerFactory.create: expected PeripheralControllerConfig, received {type(config)}."
            )

        if config.binding == BindingType.VIRTUAL:
            from evomachine.bindings.virtual.peripheralcontroller import VirtualPeripheralController

            controller = VirtualPeripheralController(
                name=config.name or VirtualPeripheralController.DEFAULT_NAME,
                **binding_options,
            )
        elif config.binding == BindingType.ASI_TIGER:
            if not isinstance(config, SerialPeripheralControllerConfig):
                raise TypeError(
                    "PeripheralControllerFactory.create: ASI_TIGER requires SerialPeripheralControllerConfig."
                )
            if "use_thread" in binding_options:
                raise TypeError("PeripheralControllerFactory.create: use_thread is not supported.")
            from evomachine.bindings.asitiger.peripheralcontroller import TigerPeripheralController

            controller = TigerPeripheralController.from_serial_port(
                port=config.resolve_port(display_name="ASI Tiger"),
                name=config.name or TigerPeripheralController.DEFAULT_NAME,
                close_on_shutdown=config.close_on_shutdown,
                **binding_options,
            )
        elif config.binding == BindingType.SYNCBOARD:
            if not isinstance(config, SerialPeripheralControllerConfig):
                raise TypeError(
                    "PeripheralControllerFactory.create: SYNCBOARD requires SerialPeripheralControllerConfig."
                )
            from evomachine.bindings.syncboard.peripheralcontroller import SyncBoardPeripheralController

            controller = SyncBoardPeripheralController.from_serial_port(
                port=config.resolve_port(display_name="SyncBoard"),
                name=config.name or SyncBoardPeripheralController.DEFAULT_NAME,
                close_on_shutdown=config.close_on_shutdown,
                **binding_options,
            )
        elif config.binding == BindingType.KWR103:
            if not isinstance(config, SerialPeripheralControllerConfig):
                raise TypeError(
                    "PeripheralControllerFactory.create: KWR103 requires SerialPeripheralControllerConfig."
                )
            from evomachine.bindings.kwr103.peripheralcontroller import KWR103PeripheralController

            controller = KWR103PeripheralController.from_serial_port(
                port=config.resolve_port(display_name="KWR103"),
                name=config.name or KWR103PeripheralController.DEFAULT_NAME,
                close_on_shutdown=config.close_on_shutdown,
                **binding_options,
            )
        else:
            raise ValueError(f"PeripheralControllerFactory.create: unsupported binding {config.binding}.")

        controller.config = config.copy()
        if config.initialise:
            controller.initialise()
        return controller


def normalise_peripheral_controllers(
        peripheral_controllers: PeripheralController | list[PeripheralController] | None,
) -> list[PeripheralController]:
    """
    Return peripheral controllers as a validated list.

    Parameters
    ----------
    peripheral_controllers
        One controller, a list of controllers, or None.

    Returns
    -------
    list[PeripheralController]
        Validated controller list.
    """
    if peripheral_controllers is None:
        return []
    if isinstance(peripheral_controllers, PeripheralController):
        return [peripheral_controllers]
    if not isinstance(peripheral_controllers, list):
        raise TypeError(
            "normalise_peripheral_controllers: expected PeripheralController, list[PeripheralController], or None."
        )
    if not all(isinstance(controller, PeripheralController) for controller in peripheral_controllers):
        raise TypeError("normalise_peripheral_controllers: all entries must be PeripheralController.")
    return list(peripheral_controllers)


def get_peripheral_controller(
        peripheral_controllers: PeripheralController | list[PeripheralController] | None,
        controller_type: type[PeripheralController],
        action: str,
) -> PeripheralController:
    """
    Return the first controller matching the requested controller type.

    Parameters
    ----------
    peripheral_controllers
        One controller, a list of controllers, or None to search.
    controller_type
        Required PeripheralController subclass.
    action
        Human-readable action used in the error message when no controller is
        found.

    Returns
    -------
    PeripheralController
        First matching controller.
    """
    controllers = normalise_peripheral_controllers(peripheral_controllers=peripheral_controllers)
    for controller in controllers:
        if isinstance(controller, controller_type):
            return controller
    raise ValueError(f"{action}: {controller_type.__name__} is required.")


__all__ = [
    "PeripheralController",
    "PeripheralControllerConfig",
    "PeripheralControllerFactory",
    "SerialPeripheralController",
    "SerialPeripheralControllerConfig",
    "SocketPeripheralController",
    "SocketPeripheralControllerConfig",
    "get_peripheral_controller",
    "normalise_peripheral_controllers",
]
