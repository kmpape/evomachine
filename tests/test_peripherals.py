import pytest

from evomachine.bindings.asitiger.peripheralcontroller import TigerPeripheralController
from evomachine.bindings.asitiger.card_addresses import (
    CARD_ADDRESS_CRISP,
    CARD_ADDRESS_FILTER_WHEEL,
    CARD_ADDRESS_LED,
)
from evomachine.bindings.kwr103.KWR103Driver import KWR103
from evomachine.bindings.kwr103.peripheralcontroller import KWR103PeripheralController
from evomachine.bindings.syncboard.peripheralcontroller import SyncBoardPeripheralController
from evomachine.bindings.virtual.peripheralcontroller import VirtualPeripheralController
from evomachine.peripherals.peripheralcontrollers import (
    PeripheralController,
    PeripheralControllerConfig,
    PeripheralControllerFactory,
    SerialPeripheralControllerConfig,
)
from evomachine.peripherals.peripherals import PeripheralController as CompatibilityPeripheralController
from evomachine.bindings.binding_types import BindingType


class FakeInnerConnection:
    def __init__(self):
        self.is_open = True


class FakeConnection:
    def __init__(self):
        self.connection = FakeInnerConnection()
        self.disconnect_was_called = False

    def disconnect(self):
        self.disconnect_was_called = True
        self.connection.is_open = False


class FakeKWR103Serial:
    def __init__(self, port: str, baudrate: int):
        self.port = port
        self.baudrate = baudrate
        self.is_open = True
        self.writes: list[bytes] = []
        self.reply = b"KWR103\n"
        self.reset_output_buffer_was_called = False
        self.reset_input_buffer_was_called = False

    def reset_output_buffer(self):
        self.reset_output_buffer_was_called = True

    def reset_input_buffer(self):
        self.reset_input_buffer_was_called = True

    def write(self, data: bytes):
        self.writes.append(data)

    def readline(self):
        return self.reply

    def close(self):
        self.is_open = False


class FakeTigerController:
    def __init__(self):
        self.connection = FakeConnection()
        self.halt_was_called = False

    def status(self):
        return True

    def halt(self):
        self.halt_was_called = True


class FakeControllerWithoutDisconnect:
    def __init__(self):
        self.connection = object()
        self.halt_was_called = False

    def status(self):
        return True

    def halt(self):
        self.halt_was_called = True


class FakeSyncBoardController:
    def __init__(self):
        self.connection = FakeConnection()
        self.finalise_was_called = False
        self._is_initialised = True

    def initialise(self, force_init=False):
        self._is_initialised = True

    def is_initialised(self):
        return self._is_initialised

    def disable_system(self):
        return

    def finalise(self):
        self.finalise_was_called = True
        self._is_initialised = False


class TrackingPeripheralController(PeripheralController):
    def __init__(self, fail_stop: bool = False):
        self.events: list[str] = []
        self.fail_stop = fail_stop
        super().__init__(name="Tracking")
        self._is_initialised = True
        self._is_alive = True

    def _initialise(self, force: bool = False) -> bool:
        self.events.append("initialise")
        return True

    def _check_is_alive(self) -> bool:
        return True

    def _stop(self) -> None:
        self.events.append("stop")
        if self.fail_stop:
            raise RuntimeError("stop failed")

    def _shutdown(self, force: bool = False) -> None:
        self.events.append(f"shutdown:{force}")


def test_peripheral_controller_config_rejects_non_binding_type():
    with pytest.raises(TypeError):
        PeripheralControllerConfig(binding="virtual")


def test_serial_peripheral_controller_config_requires_port_or_hwid():
    with pytest.raises(ValueError):
        SerialPeripheralControllerConfig(binding=BindingType.ASI_TIGER)

    with pytest.raises(ValueError):
        SerialPeripheralControllerConfig(
            binding=BindingType.ASI_TIGER,
            port="/dev/ttyUSB0",
            hwid="10C4:EA60",
        )


def test_serial_peripheral_controller_config_rejects_non_serial_binding():
    """Check that shared BindingType values are validated for serial controllers."""
    with pytest.raises(ValueError, match="serial binding"):
        SerialPeripheralControllerConfig(
            binding=BindingType.VIRTUAL,
            port="/dev/ttyUSB0",
        )


def test_serial_peripheral_controller_config_accepts_exactly_one_port_source(monkeypatch):
    by_port = SerialPeripheralControllerConfig(
        binding=BindingType.ASI_TIGER,
        port="/dev/ttyUSB0",
    )
    assert by_port.resolve_port() == "/dev/ttyUSB0"

    def fake_get_port(hwid: str, display_name: str = ""):
        assert hwid == "10C4:EA60"
        assert display_name == "ASI Tiger"
        return "/dev/ttyUSB1"

    monkeypatch.setattr("evomachine.com_ports.get_port", fake_get_port)
    by_hwid = SerialPeripheralControllerConfig(
        binding=BindingType.ASI_TIGER,
        hwid="10C4:EA60",
    )

    assert by_hwid.resolve_port(display_name="ASI Tiger") == "/dev/ttyUSB1"


def test_serial_peripheral_controller_config_accepts_kwr103_binding():
    config = SerialPeripheralControllerConfig(
        binding=BindingType.KWR103,
        port="/dev/ttyUSB2",
    )

    assert config.resolve_port() == "/dev/ttyUSB2"


def test_default_configs_return_expected_values():
    virtual_config = VirtualPeripheralController.default_config()
    assert isinstance(virtual_config, PeripheralControllerConfig)
    assert virtual_config.binding == BindingType.VIRTUAL
    assert virtual_config.name == VirtualPeripheralController.DEFAULT_NAME

    tiger_config = TigerPeripheralController.default_config()
    assert isinstance(tiger_config, SerialPeripheralControllerConfig)
    assert tiger_config.binding == BindingType.ASI_TIGER
    assert tiger_config.name == TigerPeripheralController.DEFAULT_NAME
    assert tiger_config.hwid == TigerPeripheralController.DEFAULT_HWID

    syncboard_config = SyncBoardPeripheralController.default_config()
    assert isinstance(syncboard_config, SerialPeripheralControllerConfig)
    assert syncboard_config.binding == BindingType.SYNCBOARD
    assert syncboard_config.name == SyncBoardPeripheralController.DEFAULT_NAME
    assert syncboard_config.hwid == SyncBoardPeripheralController.DEFAULT_HWID

    kwr103_config = KWR103PeripheralController.default_config()
    assert isinstance(kwr103_config, SerialPeripheralControllerConfig)
    assert kwr103_config.binding == BindingType.KWR103
    assert kwr103_config.name == KWR103PeripheralController.DEFAULT_NAME
    assert kwr103_config.hwid == KWR103PeripheralController.DEFAULT_HWID


def test_tiger_peripheral_controller_owns_card_addresses():
    tiger = FakeTigerController()
    controller = TigerPeripheralController(tiger=tiger)

    assert controller.card_address_crisp == CARD_ADDRESS_CRISP
    assert controller.card_address_led == CARD_ADDRESS_LED
    assert controller.card_address_filter_wheel == CARD_ADDRESS_FILTER_WHEEL

    custom = TigerPeripheralController(
        tiger=tiger,
        card_address_crisp=3,
        card_address_led=4,
        card_address_filter_wheel=5,
    )

    assert custom.card_address_crisp == 3
    assert custom.card_address_led == 4
    assert custom.card_address_filter_wheel == 5


def test_peripheral_controller_factory_creates_virtual_controller():
    controller = PeripheralControllerFactory.create(
        PeripheralControllerConfig(binding=BindingType.VIRTUAL)
    )

    assert isinstance(controller, VirtualPeripheralController)
    assert controller.is_initialised()


def test_peripheral_controller_factory_requires_serial_config_for_serial_bindings():
    with pytest.raises(TypeError):
        PeripheralControllerFactory.create(
            PeripheralControllerConfig(binding=BindingType.ASI_TIGER)
        )

    with pytest.raises(TypeError):
        PeripheralControllerFactory.create(
            PeripheralControllerConfig(binding=BindingType.SYNCBOARD)
        )

    with pytest.raises(TypeError):
        PeripheralControllerFactory.create(
            PeripheralControllerConfig(binding=BindingType.KWR103)
        )


def test_peripheral_controller_factory_passes_serial_config_to_asitiger(monkeypatch):
    calls = {}

    def fake_from_serial_port(cls, **kwargs):
        calls.update(kwargs)
        return TigerPeripheralController(tiger=FakeTigerController())

    monkeypatch.setattr(TigerPeripheralController, "from_serial_port", classmethod(fake_from_serial_port))
    controller = PeripheralControllerFactory.create(
        SerialPeripheralControllerConfig(
            binding=BindingType.ASI_TIGER,
            name="Tiger",
            initialise=False,
            port="/dev/ttyUSB0",
            close_on_shutdown=False,
        )
    )

    assert isinstance(controller, TigerPeripheralController)
    assert not controller.is_initialised()
    assert calls == {
        "port": "/dev/ttyUSB0",
        "name": "Tiger",
        "close_on_shutdown": False,
    }


def test_peripheral_controller_factory_rejects_asitiger_use_thread_option():
    with pytest.raises(TypeError):
        PeripheralControllerFactory.create(
            SerialPeripheralControllerConfig(
                binding=BindingType.ASI_TIGER,
                port="/dev/ttyUSB0",
            ),
            use_thread=True,
        )


def test_peripheral_controller_factory_passes_serial_config_to_syncboard(monkeypatch):
    calls = {}

    def fake_from_serial_port(cls, **kwargs):
        calls.update(kwargs)
        return SyncBoardPeripheralController(syncboard=FakeSyncBoardController())

    monkeypatch.setattr(SyncBoardPeripheralController, "from_serial_port", classmethod(fake_from_serial_port))
    controller = PeripheralControllerFactory.create(
        SerialPeripheralControllerConfig(
            binding=BindingType.SYNCBOARD,
            name="Sync",
            initialise=False,
            port="/dev/ttyACM0",
            close_on_shutdown=False,
        )
    )

    assert isinstance(controller, SyncBoardPeripheralController)
    assert not controller.is_initialised()
    assert calls == {
        "port": "/dev/ttyACM0",
        "name": "Sync",
        "close_on_shutdown": False,
    }


def test_peripheral_controller_factory_passes_serial_config_to_kwr103(monkeypatch):
    calls = {}

    def fake_from_serial_port(cls, **kwargs):
        calls.update(kwargs)
        return KWR103PeripheralController(kwr103=KWR103(port=kwargs["port"]))

    monkeypatch.setattr(KWR103PeripheralController, "from_serial_port", classmethod(fake_from_serial_port))
    controller = PeripheralControllerFactory.create(
        SerialPeripheralControllerConfig(
            binding=BindingType.KWR103,
            name="KWR",
            initialise=False,
            port="/dev/ttyUSB2",
            close_on_shutdown=False,
        )
    )

    assert isinstance(controller, KWR103PeripheralController)
    assert not controller.is_initialised()
    assert calls == {
        "port": "/dev/ttyUSB2",
        "name": "KWR",
        "close_on_shutdown": False,
    }


def test_peripheral_controller_factory_accepts_default_serial_configs(monkeypatch):
    ports = {
        (TigerPeripheralController.DEFAULT_HWID, "ASI Tiger"): "/dev/tiger",
        (SyncBoardPeripheralController.DEFAULT_HWID, "SyncBoard"): "/dev/syncboard",
        (KWR103PeripheralController.DEFAULT_HWID, "KWR103"): "/dev/kwr103",
    }
    calls = {}

    def fake_get_port(hwid: str, display_name: str = ""):
        return ports[(hwid, display_name)]

    def fake_tiger_from_serial_port(cls, **kwargs):
        calls["tiger"] = kwargs
        return TigerPeripheralController(tiger=FakeTigerController())

    def fake_syncboard_from_serial_port(cls, **kwargs):
        calls["syncboard"] = kwargs
        return SyncBoardPeripheralController(syncboard=FakeSyncBoardController())

    def fake_kwr103_from_serial_port(cls, **kwargs):
        calls["kwr103"] = kwargs
        return KWR103PeripheralController(kwr103=KWR103(port=kwargs["port"]))

    monkeypatch.setattr("evomachine.com_ports.get_port", fake_get_port)
    monkeypatch.setattr(TigerPeripheralController, "from_serial_port", classmethod(fake_tiger_from_serial_port))
    monkeypatch.setattr(
        SyncBoardPeripheralController,
        "from_serial_port",
        classmethod(fake_syncboard_from_serial_port),
    )
    monkeypatch.setattr(KWR103PeripheralController, "from_serial_port", classmethod(fake_kwr103_from_serial_port))

    tiger_config = TigerPeripheralController.default_config()
    tiger_config.initialise = False
    syncboard_config = SyncBoardPeripheralController.default_config()
    syncboard_config.initialise = False
    kwr103_config = KWR103PeripheralController.default_config()
    kwr103_config.initialise = False

    PeripheralControllerFactory.create(tiger_config)
    PeripheralControllerFactory.create(syncboard_config)
    PeripheralControllerFactory.create(kwr103_config)

    assert calls["tiger"]["port"] == "/dev/tiger"
    assert calls["syncboard"]["port"] == "/dev/syncboard"
    assert calls["kwr103"]["port"] == "/dev/kwr103"


def test_peripheral_controller_shutdown_stops_before_shutdown():
    controller = TrackingPeripheralController()

    controller.shutdown(force=True)

    assert controller.events == ["stop", "shutdown:True"]
    assert not controller.is_initialised()


def test_peripheral_controller_shutdown_runs_shutdown_when_stop_raises():
    controller = TrackingPeripheralController(fail_stop=True)

    with pytest.raises(RuntimeError, match="stop failed"):
        controller.shutdown()

    assert controller.events == ["stop", "shutdown:False"]
    assert not controller.is_initialised()


def test_tiger_shutdown_uses_shared_disconnect_behaviour():
    tiger = FakeTigerController()
    controller = TigerPeripheralController(tiger=tiger)

    controller.shutdown()

    assert tiger.connection.disconnect_was_called


def test_serial_controller_shutdown_requires_disconnect_method():
    controller = TigerPeripheralController(tiger=FakeControllerWithoutDisconnect())

    with pytest.raises(TypeError, match="disconnect"):
        controller.shutdown()


def test_old_peripherals_controller_import_still_works():
    assert CompatibilityPeripheralController is PeripheralController


def test_tiger_shutdown_respects_close_on_shutdown_unless_forced():
    tiger = FakeTigerController()
    controller = TigerPeripheralController(tiger=tiger, close_on_shutdown=False)

    controller.shutdown()
    assert not tiger.connection.disconnect_was_called

    controller.shutdown(force=True)
    assert tiger.connection.disconnect_was_called


def test_syncboard_shutdown_finalises_before_disconnect():
    syncboard = FakeSyncBoardController()
    controller = SyncBoardPeripheralController(syncboard=syncboard)

    controller.shutdown()

    assert syncboard.finalise_was_called
    assert syncboard.connection.disconnect_was_called


def test_syncboard_shutdown_respects_close_on_shutdown_unless_forced():
    syncboard = FakeSyncBoardController()
    controller = SyncBoardPeripheralController(syncboard=syncboard, close_on_shutdown=False)

    controller.shutdown()
    assert syncboard.finalise_was_called
    assert not syncboard.connection.disconnect_was_called

    controller.shutdown(force=True)
    assert syncboard.connection.disconnect_was_called


def test_kwr103_connect_disconnect_and_query(monkeypatch):
    created_serials: list[FakeKWR103Serial] = []

    def fake_serial(port: str, baudrate: int):
        serial = FakeKWR103Serial(port=port, baudrate=baudrate)
        created_serials.append(serial)
        return serial

    monkeypatch.setattr("evomachine.bindings.kwr103.KWR103Driver.serial.Serial", fake_serial)
    kwr103 = KWR103(port="/dev/ttyUSB2")

    kwr103.connect()
    assert kwr103.is_connected()
    assert created_serials[0].port == "/dev/ttyUSB2"
    assert created_serials[0].baudrate == 115200

    assert kwr103.query_serial_no() == "KWR103"
    assert created_serials[0].writes == [b"*IDN?\n"]
    assert created_serials[0].reset_output_buffer_was_called
    assert created_serials[0].reset_input_buffer_was_called

    kwr103.disconnect()
    assert not kwr103.is_connected()
    kwr103.close()
    assert not kwr103.is_connected()


def test_kwr103_command_helpers_preserve_command_strings(monkeypatch):
    created_serials: list[FakeKWR103Serial] = []

    def fake_serial(port: str, baudrate: int):
        serial = FakeKWR103Serial(port=port, baudrate=baudrate)
        serial.reply = b"1.2\n"
        created_serials.append(serial)
        return serial

    monkeypatch.setattr("evomachine.bindings.kwr103.KWR103Driver.serial.Serial", fake_serial)
    kwr103 = KWR103(port="/dev/ttyUSB2")
    kwr103.connect()

    kwr103.set_output(True)
    kwr103.set_output(False)
    kwr103.set_voltage(8)
    kwr103.set_current(0.1)
    assert kwr103.get_voltage_set() == 1.2
    assert kwr103.get_current_set() == 1.2
    assert kwr103.get_voltage_out() == 1.2
    assert kwr103.get_current_out() == 1.2

    assert created_serials[0].writes == [
        b"OUT:1\n",
        b"OUT:0\n",
        b"VSET:8.0\n",
        b"ISET:0.1\n",
        b"VSET?\n",
        b"ISET?\n",
        b"VOUT?\n",
        b"IOUT?\n",
    ]


def test_kwr103_peripheral_controller_initialises_stops_and_shutdowns(monkeypatch):
    created_serials: list[FakeKWR103Serial] = []

    def fake_serial(port: str, baudrate: int):
        serial = FakeKWR103Serial(port=port, baudrate=baudrate)
        created_serials.append(serial)
        return serial

    monkeypatch.setattr("evomachine.bindings.kwr103.KWR103Driver.serial.Serial", fake_serial)
    controller = KWR103PeripheralController.from_serial_port(port="/dev/ttyUSB2")

    controller.initialise()
    assert controller.is_initialised()
    assert controller.is_alive()

    controller.stop()
    controller.shutdown()

    assert created_serials[0].writes == [b"OUT:0\n", b"OUT:0\n", b"OUT:0\n"]
    assert not created_serials[0].is_open


def test_kwr103_shutdown_respects_close_on_shutdown_unless_forced(monkeypatch):
    created_serials: list[FakeKWR103Serial] = []

    def fake_serial(port: str, baudrate: int):
        serial = FakeKWR103Serial(port=port, baudrate=baudrate)
        created_serials.append(serial)
        return serial

    monkeypatch.setattr("evomachine.bindings.kwr103.KWR103Driver.serial.Serial", fake_serial)
    controller = KWR103PeripheralController.from_serial_port(
        port="/dev/ttyUSB2",
        close_on_shutdown=False,
    )
    controller.initialise()

    controller.shutdown()
    assert created_serials[0].writes == [b"OUT:0\n", b"OUT:0\n"]
    assert created_serials[0].is_open

    controller.initialise()
    controller.shutdown(force=True)
    assert created_serials[0].writes == [b"OUT:0\n", b"OUT:0\n", b"OUT:0\n", b"OUT:0\n"]
    assert not created_serials[0].is_open
