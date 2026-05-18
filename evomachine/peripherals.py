from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from evomachine.bindings.binding_types import BindingType


class PeripheralController(ABC):

    def __init__(self, name: str):
        self.name: str = name
        self._is_initialised: bool = False
        self._is_alive: bool = False

    def initialise(self, force: bool = False) -> None:
        if self._is_initialised and not force:
            return
        self._is_initialised = self._initialise(force=force)
        self._is_alive = self._check_is_alive()
        if not self._is_initialised:
            raise RuntimeError(f"PeripheralController.initialise: {self.name} failed to initialise.")
        if not self._is_alive:
            raise RuntimeError(f"PeripheralController.initialise: {self.name} is not alive after initialisation.")

    def reinitialise(self, force: bool = False) -> None:
        self.shutdown(force=force)
        self.initialise(force=True)

    def is_alive(self) -> bool:
        self._is_alive = self._check_is_alive()
        return self._is_alive

    def is_initialised(self) -> bool:
        return self._is_initialised

    def stop(self) -> None:
        self._stop()

    def shutdown(self, force: bool = False) -> None:
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
        raise NotImplementedError

    @abstractmethod
    def _check_is_alive(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def _stop(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _shutdown(self, force: bool = False) -> None:
        raise NotImplementedError


@dataclass
class PeripheralControllerConfig:
    """Configuration object used by PeripheralControllerFactory."""

    binding: BindingType
    name: str = ""
    initialise: bool = True

    def __post_init__(self) -> None:
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


@dataclass
class SerialPeripheralControllerConfig(PeripheralControllerConfig):
    """Configuration object for serial PeripheralController bindings."""

    port: str | None = None
    hwid: str | None = None
    close_on_shutdown: bool = True

    def __post_init__(self) -> None:
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
            If True, shutdown closes the underlying serial connection.
            Force shutdown closes it regardless of this setting.
        """
        self.close_on_shutdown: bool = close_on_shutdown
        super().__init__(name=name)

    @abstractmethod
    def _get_serial_controller(self) -> Any:
        """
        Return the wrapped binding-specific serial controller object.

        Returns
        -------
        Any
            Object that owns or exposes the binding's serial connection.
        """
        raise NotImplementedError

    def _get_connection(self) -> Any:
        """
        Return the object that provides the serial disconnect method.

        Returns
        -------
        Any
            Connection-like object with an optional disconnect method.
        """
        return getattr(self._get_serial_controller(), "connection", None)

    def _before_disconnect(self, force: bool = False) -> None:
        """
        Run binding-specific shutdown work before closing the connection.

        Parameters
        ----------
        force
            True when shutdown should close the connection even if
            close_on_shutdown is False.
        """
        return

    def _disconnect(self) -> None:
        """
        Close the wrapped serial connection if it exposes disconnect().

        Parameters
        ----------
        None
        """
        connection = self._get_connection()
        disconnect = getattr(connection, "disconnect", None)
        if callable(disconnect):
            disconnect()

    def _shutdown(self, force: bool = False) -> None:
        """
        Run serial shutdown hooks and optionally close the connection.

        Parameters
        ----------
        force
            True to close the serial connection regardless of
            close_on_shutdown.
        """
        self._before_disconnect(force=force)
        if not (force or self.close_on_shutdown):
            return
        self._disconnect()


@dataclass
class SocketPeripheralControllerConfig(PeripheralControllerConfig):
    """Configuration object for socket-backed PeripheralController bindings."""

    host: str = "127.0.0.1"
    port: int = 0
    close_on_shutdown: bool = True

    def __post_init__(self) -> None:
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


class SocketPeripheralController(PeripheralController):
    """Base class for PeripheralController implementations backed by a socket connection."""

    def __init__(
            self,
            name: str,
            close_on_shutdown: bool = True,
    ):
        self.close_on_shutdown: bool = close_on_shutdown
        super().__init__(name=name)

    @abstractmethod
    def _get_socket_controller(self) -> Any:
        """Return the wrapped binding-specific socket controller object."""
        raise NotImplementedError

    def _get_socket(self) -> Any:
        """Return the socket-like object exposed by the wrapped controller, if any."""
        controller = self._get_socket_controller()
        return getattr(controller, "s", getattr(controller, "socket", None))

    def _before_disconnect(self, force: bool = False) -> None:
        """Run binding-specific shutdown work before closing the socket."""
        return

    def _disconnect(self) -> None:
        """Close the wrapped socket if it exposes close()."""
        socket = self._get_socket()
        close = getattr(socket, "close", None)
        if callable(close):
            close()

    def _shutdown(self, force: bool = False) -> None:
        """Run socket shutdown hooks and optionally close the socket."""
        self._before_disconnect(force=force)
        if not (force or self.close_on_shutdown):
            return
        self._disconnect()


class PeripheralControllerFactory:
    """Factory for creating PeripheralController instances from typed configs."""

    @staticmethod
    def create(config: PeripheralControllerConfig, **binding_options: Any) -> PeripheralController:
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

        if config.initialise:
            controller.initialise()
        return controller


def normalise_peripheral_controllers(
        peripheral_controllers: PeripheralController | list[PeripheralController] | None,
) -> list[PeripheralController]:
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
    controllers = normalise_peripheral_controllers(peripheral_controllers=peripheral_controllers)
    for controller in controllers:
        if isinstance(controller, controller_type):
            return controller
    raise ValueError(f"{action}: {controller_type.__name__} is required.")


class Peripheral(ABC):
    @abstractmethod
    def is_alive(self) -> bool:
        raise NotImplementedError
    
    @abstractmethod
    def stop(self) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def initialise(self, force: bool = False) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def finalise(self, force: bool = False) -> None:
        raise NotImplementedError
