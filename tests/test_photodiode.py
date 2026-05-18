import pytest

from evomachine.bindings.binding_types import BindingType
from evomachine.bindings.syncboard.peripheralcontroller import SyncBoardPeripheralController
from evomachine.bindings.syncboard.photodiode import SyncBoardPhotodiode
from evomachine.bindings.virtual.peripheralcontroller import VirtualPeripheralController
from evomachine.bindings.virtual.photodiode import VirtualPhotodiode
from evomachine.photodiode import (
    PhotodiodeConfig,
    PhotodiodeFactory,
    PhotodiodeReadingRange,
)


class FakeInnerConnection:
    """Fake serial connection state used by SyncBoardPeripheralController."""

    def __init__(self):
        """Initialise the fake inner connection as open."""
        self.is_open = True


class FakeConnection:
    """Fake connection wrapper used by SyncBoardPeripheralController."""

    def __init__(self):
        """Initialise fake connection state and disconnect bookkeeping."""
        self.connection = FakeInnerConnection()
        self.disconnect_was_called = False

    def disconnect(self):
        """Record a disconnect call and mark the fake serial connection closed."""
        self.disconnect_was_called = True
        self.connection.is_open = False


class FakeSyncBoardController:
    """Fake SyncBoard controller exposing photodiode reads for tests."""

    def __init__(self, reading=0.5):
        """Initialise fake SyncBoard state with one configurable reading."""
        self.connection = FakeConnection()
        self._is_initialised = True
        self.reading = reading
        self.read_channels = []

    def initialise(self, force_init=False):
        """Mark the fake SyncBoard as initialised."""
        self._is_initialised = True

    def is_initialised(self):
        """Return whether the fake SyncBoard is initialised."""
        return self._is_initialised

    def disable_system(self):
        """Accept a fake disable-system command."""
        return

    def finalise(self):
        """Mark the fake SyncBoard as finalised."""
        self._is_initialised = False

    def read_photodiode(self, channel=8):
        """Record the requested channel and return the configured reading."""
        self.read_channels.append(channel)
        return self.reading


def make_virtual_photodiode(
        reading_range: PhotodiodeReadingRange | None = None,
) -> VirtualPhotodiode:
    """
    Return an initialised virtual photodiode for tests.

    Parameters
    ----------
    reading_range
        Optional raw reading range to use when creating the photodiode.

    Returns
    -------
    VirtualPhotodiode
        Initialised virtual photodiode with an initialised controller.
    """
    peripheral_ctrl = VirtualPeripheralController()
    peripheral_ctrl.initialise()
    photodiode = VirtualPhotodiode(
        peripheral_ctrl=peripheral_ctrl,
        reading_range=reading_range,
    )
    photodiode.initialise()
    return photodiode


def test_photodiode_reading_range_validation():
    """Validate photodiode reading range types and ordering."""
    reading_range = PhotodiodeReadingRange(0, 10)

    assert reading_range.minimum_reading == 0.0
    assert reading_range.maximum_reading == 10.0
    with pytest.raises(TypeError):
        PhotodiodeReadingRange(False, 10)
    with pytest.raises(TypeError):
        PhotodiodeReadingRange(0, "10")
    with pytest.raises(ValueError):
        PhotodiodeReadingRange(10, 10)
    with pytest.raises(ValueError):
        PhotodiodeReadingRange(11, 10)


def test_photodiode_config_validation():
    """Validate photodiode config field types and channel values."""
    config = PhotodiodeConfig(binding=BindingType.VIRTUAL)

    assert config.channel == 8
    with pytest.raises(TypeError):
        PhotodiodeConfig(binding="virtual")
    with pytest.raises(TypeError):
        PhotodiodeConfig(binding=BindingType.VIRTUAL, channel=1.5)
    with pytest.raises(TypeError):
        PhotodiodeConfig(binding=BindingType.VIRTUAL, channel=True)
    with pytest.raises(ValueError):
        PhotodiodeConfig(binding=BindingType.VIRTUAL, channel=0)
    with pytest.raises(TypeError):
        PhotodiodeConfig(binding=BindingType.VIRTUAL, name=123)
    with pytest.raises(TypeError):
        PhotodiodeConfig(binding=BindingType.VIRTUAL, reading_range=(0, 1))
    with pytest.raises(TypeError):
        PhotodiodeConfig(binding=BindingType.VIRTUAL, check_initialised="yes")
    with pytest.raises(TypeError):
        PhotodiodeConfig(binding=BindingType.VIRTUAL, check_alive="yes")


def test_virtual_photodiode_default_reading_is_zero_percent():
    """Check that the virtual photodiode defaults to the calibrated minimum."""
    photodiode = make_virtual_photodiode()

    assert photodiode.read_photodiode() == 0.0


def test_virtual_photodiode_updates_raw_reading_and_clamps():
    """Check virtual raw reading updates, calibration, and output clamping."""
    photodiode = make_virtual_photodiode(PhotodiodeReadingRange(10, 20))

    photodiode.set_raw_reading(15)
    assert photodiode.read_photodiode() == 50.0

    photodiode.set_raw_reading(25)
    assert photodiode.read_photodiode() == 100.0

    photodiode.set_raw_reading(5)
    assert photodiode.read_photodiode() == 0.0

    photodiode.set_raw_reading(20)
    photodiode.set_reading_range(0, 40)
    assert photodiode.read_photodiode() == 50.0

    with pytest.raises(ValueError):
        photodiode.set_reading_range(10, 10)


def test_virtual_photodiode_requires_initialise_by_default():
    """Check that reads fail before photodiode initialisation."""
    peripheral_ctrl = VirtualPeripheralController()
    peripheral_ctrl.initialise()
    photodiode = VirtualPhotodiode(peripheral_ctrl=peripheral_ctrl)

    with pytest.raises(RuntimeError, match="not initialised"):
        photodiode.read_photodiode()


def test_syncboard_photodiode_reads_channel_and_normalises():
    """Check SyncBoard photodiode channel routing and calibration."""
    syncboard = FakeSyncBoardController(reading=75)
    peripheral_ctrl = SyncBoardPeripheralController(syncboard=syncboard)
    peripheral_ctrl.initialise()
    photodiode = SyncBoardPhotodiode(
        peripheral_ctrl=peripheral_ctrl,
        channel=4,
        reading_range=PhotodiodeReadingRange(50, 100),
    )
    photodiode.initialise()

    assert photodiode.read_photodiode() == 50.0
    assert syncboard.read_channels == [4]


def test_syncboard_photodiode_raises_for_missing_or_malformed_reading():
    """Check SyncBoard bad raw readings raise RuntimeError."""
    syncboard = FakeSyncBoardController(reading=None)
    peripheral_ctrl = SyncBoardPeripheralController(syncboard=syncboard)
    peripheral_ctrl.initialise()
    photodiode = SyncBoardPhotodiode(peripheral_ctrl=peripheral_ctrl)
    photodiode.initialise()

    with pytest.raises(RuntimeError, match="received no reading"):
        photodiode.read_photodiode()

    syncboard.reading = "bad"
    with pytest.raises(RuntimeError, match="malformed"):
        photodiode.read_photodiode()


def test_photodiode_factory_creates_virtual_and_syncboard_bindings():
    """Check PhotodiodeFactory creates supported binding implementations."""
    virtual_ctrl = VirtualPeripheralController()
    virtual_ctrl.initialise()
    virtual_photodiode = PhotodiodeFactory.create(
        PhotodiodeConfig(binding=BindingType.VIRTUAL),
        peripheral_controllers=virtual_ctrl,
    )

    assert isinstance(virtual_photodiode, VirtualPhotodiode)

    syncboard_ctrl = SyncBoardPeripheralController(syncboard=FakeSyncBoardController(reading=0.25))
    syncboard_ctrl.initialise()
    syncboard_photodiode = PhotodiodeFactory.create(
        PhotodiodeConfig(binding=BindingType.SYNCBOARD),
        peripheral_controllers=syncboard_ctrl,
    )

    assert isinstance(syncboard_photodiode, SyncBoardPhotodiode)


def test_photodiode_factory_rejects_unsupported_binding():
    """Check PhotodiodeFactory rejects non-photodiode bindings."""
    with pytest.raises(ValueError, match="unsupported photodiode binding"):
        PhotodiodeFactory.create(PhotodiodeConfig(binding=BindingType.ASI_TIGER))
