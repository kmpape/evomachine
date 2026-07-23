from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
import logging

import numpy as np
import pytest

from evomachine.bindings.binding_types import BindingType
from evomachine.bindings.virtual.peripheralcontroller import VirtualPeripheralController
from evomachine.peripherals import camera as camera_module
from evomachine.peripherals.camera import CameraConfig, CameraFactory, CameraReadoutMode, ObjectiveConfigType
from evomachine.peripherals.camera import ImageConfigType


def _image_config() -> ImageConfigType:
    """
    Return a small image configuration for camera tests.

    Parameters
    ----------
    None

    Returns
    -------
    ImageConfigType
        Small uint16 test image configuration.
    """
    return ImageConfigType(pxl_horiz=8, pxl_vert=6, pxl_dtype=np.dtype("uint16"))


def _initialised_virtual_controller() -> VirtualPeripheralController:
    """
    Return an initialised virtual peripheral controller.

    Parameters
    ----------
    None

    Returns
    -------
    VirtualPeripheralController
        Initialised virtual controller for fake camera tests.
    """
    controller = VirtualPeripheralController()
    controller.initialise()
    return controller


def test_camera_config_validation() -> None:
    """
    Check CameraConfig validates binding, image, exposure, and simple flags.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    image = _image_config()
    config = CameraConfig(binding=BindingType.VIRTUAL, image=image)
    assert config.binding == BindingType.VIRTUAL
    assert config.image == image
    assert config.objective_config is None

    objective_config = ObjectiveConfigType(na=0.95, mag=40)
    config = CameraConfig(binding=BindingType.VIRTUAL, image=image, objective_config=objective_config)
    assert config.objective_config == objective_config

    with pytest.raises(TypeError):
        CameraConfig(binding="virtual", image=image)
    with pytest.raises(TypeError):
        CameraConfig(binding=BindingType.VIRTUAL, image="image")
    with pytest.raises(TypeError):
        CameraConfig(binding=BindingType.VIRTUAL, image=image, default_exposure_time=True)
    with pytest.raises(ValueError):
        CameraConfig(binding=BindingType.VIRTUAL, image=image, default_exposure_time=0)
    with pytest.raises(TypeError):
        CameraConfig(binding=BindingType.VIRTUAL, image=image, name=123)
    with pytest.raises(TypeError):
        CameraConfig(binding=BindingType.VIRTUAL, image=image, check_initialised="yes")
    with pytest.raises(TypeError):
        CameraConfig(binding=BindingType.VIRTUAL, image=image, check_alive="yes")
    with pytest.raises(TypeError):
        CameraConfig(binding=BindingType.VIRTUAL, image=image, readout_mode=1)
    with pytest.raises(TypeError):
        CameraConfig(binding=BindingType.VIRTUAL, image=image, objective_config="objective")


def test_virtual_camera_lifecycle_exposure_and_frame() -> None:
    """
    Check virtual camera lifecycle, exposure caching, and frame shape/dtype.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    config = CameraConfig(
        binding=BindingType.VIRTUAL,
        image=_image_config(),
        default_exposure_time=50,
        readout_mode=CameraReadoutMode.SENSITIVITY,
    )
    camera = CameraFactory.create(
        config=config,
        peripheral_controllers=_initialised_virtual_controller(),
        random_seed=1,
    )

    with pytest.raises(RuntimeError):
        camera.get_frame()

    camera.initialise()
    assert camera.is_initialised()
    assert camera.is_alive()
    assert camera.get_exposure() == 50
    assert camera.readout_mode == CameraReadoutMode.SENSITIVITY

    frame = camera.get_frame()
    assert frame.shape == config.image.shape
    assert frame.dtype == config.image.pxl_dtype

    camera.set_exposure(75)
    assert camera.get_exposure() == 75

    camera.stop()
    assert camera.stop_was_called()

    camera.finalise()
    assert not camera.is_initialised()


def test_camera_state_changes_emit_debug_logs() -> None:
    """
    Check camera state-changing operations emit peripheral debug logs.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    camera_module.logger.addHandler(handler)
    try:
        camera = CameraFactory.create(
            config=CameraConfig(binding=BindingType.VIRTUAL, image=_image_config()),
            peripheral_controllers=_initialised_virtual_controller(),
            random_seed=3,
        )

        camera.initialise()
        camera.set_exposure(80)

        log_text = stream.getvalue()
    finally:
        camera_module.logger.removeHandler(handler)

    assert "Camera.initialise" in log_text
    assert "Camera.set_exposure" in log_text


def test_virtual_camera_normalised_frame() -> None:
    """
    Check get_frame(normalise=True) returns a floating image in [0, 1].

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    camera = CameraFactory.create(
        config=CameraConfig(binding=BindingType.VIRTUAL, image=_image_config()),
        peripheral_controllers=_initialised_virtual_controller(),
        random_seed=2,
    )
    camera.initialise()

    frame = camera.get_frame(normalise=True)

    assert frame.shape == _image_config().shape
    assert frame.dtype == np.dtype("float64")
    assert np.min(frame) >= 0
    assert np.max(frame) <= 1


def test_camera_factory_rejects_unsupported_valid_binding() -> None:
    """
    Check camera factory scopes shared BindingType values to camera bindings.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    with pytest.raises(ValueError):
        CameraFactory.create(CameraConfig(binding=BindingType.ASI_TIGER, image=_image_config()))


@dataclass
class FakeTaggedImage:
    """
    Tagged image shape used by FakeMMCCore.

    Parameters
    ----------
    pix
        Flat pixel payload.
    tags
        Tagged image metadata including Height and Width.
    """

    pix: np.ndarray
    tags: dict[str, int]


class FakeMMCCore:
    """Fake pycromanager Core used to test MMCCamera without hardware."""

    def __init__(self, shape: tuple[int, int]):
        """
        Initialise fake Core state.

        Parameters
        ----------
        shape
            Height and width of frames returned by get_tagged_image().

        Returns
        -------
        None
        """
        self.shape = shape
        self.snap_count = 0
        self.exposure: float | int | None = None
        self.properties: list[tuple[str, str, str]] = []

    def snap_image(self) -> None:
        """
        Record that an image was snapped.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.snap_count += 1

    def get_tagged_image(self) -> FakeTaggedImage:
        """
        Return a fake tagged image.

        Parameters
        ----------
        None

        Returns
        -------
        FakeTaggedImage
            Tagged image with a flat uint16 payload.
        """
        frame = np.arange(np.prod(self.shape), dtype=np.uint16)
        return FakeTaggedImage(pix=frame, tags={"Height": self.shape[0], "Width": self.shape[1]})

    def set_exposure(self, exposure_time: float | int) -> None:
        """
        Store the requested exposure.

        Parameters
        ----------
        exposure_time
            Exposure time in milliseconds.

        Returns
        -------
        None
        """
        self.exposure = exposure_time

    def set_property(self, device: str, property_name: str, value: str) -> None:
        """
        Store a requested property value.

        Parameters
        ----------
        device
            Device name.
        property_name
            Property name.
        value
            Property value.

        Returns
        -------
        None
        """
        self.properties.append((device, property_name, value))


class FakeMMCLive:
    """Fake Micro-Manager live mode object."""

    def __init__(self):
        """
        Initialise fake live mode state.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.live_mode: bool | None = None

    def set_live_mode(self, live_mode: bool) -> None:
        """
        Store the requested live mode flag.

        Parameters
        ----------
        live_mode
            True to enable live mode, False to disable it.

        Returns
        -------
        None
        """
        self.live_mode = live_mode


class FakeMMCStudio:
    """Fake pycromanager Studio used to test live-mode disabling."""

    def __init__(self):
        """
        Initialise fake Studio state.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.live_object = FakeMMCLive()

    def live(self) -> FakeMMCLive:
        """
        Return the fake live mode object.

        Parameters
        ----------
        None

        Returns
        -------
        FakeMMCLive
            Fake live mode object.
        """
        return self.live_object


def test_mmc_camera_factory_with_injected_core_and_studio() -> None:
    """
    Check MMCCamera behavior with fake injected pycromanager objects.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    image = _image_config()
    core = FakeMMCCore(shape=image.shape)
    studio = FakeMMCStudio()
    camera = CameraFactory.create(
        CameraConfig(
            binding=BindingType.MMC,
            image=image,
            default_exposure_time=33,
            readout_mode=CameraReadoutMode.DYNAMIC_RANGE,
        ),
        core=core,
        studio=studio,
    )

    camera.initialise()
    frame = camera.get_frame()
    camera.disable_live_mode()

    assert frame.shape == image.shape
    assert core.snap_count == 1
    assert core.exposure == 33
    assert core.properties == [("Camera-1", "Port", "Dynamic Range")]
    assert studio.live_object.live_mode is False


class FakePVCModule:
    """Fake pyvcam.pvc module used to test PVCAMCamera without hardware."""

    def __init__(self):
        """
        Initialise fake PVCAM module state.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.init_count = 0
        self.uninit_count = 0

    def init_pvcam(self) -> None:
        """
        Record PVCAM initialisation.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.init_count += 1

    def uninit_pvcam(self) -> None:
        """
        Record PVCAM finalisation.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.uninit_count += 1


class FakePVCameraDevice:
    """Fake PVCAM camera device."""

    def __init__(self, shape: tuple[int, int]):
        """
        Initialise fake PVCAM device state.

        Parameters
        ----------
        shape
            Height and width of frames returned by get_frame().

        Returns
        -------
        None
        """
        self.shape = shape
        self.open_count = 0
        self.close_count = 0
        self.exp_mode: str | None = None
        self.exp_time: float | int | None = None
        self.readout_port: int | None = None
        self.timeout_ms: int | None = None

    def open(self) -> None:
        """
        Record camera open.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.open_count += 1

    def close(self) -> None:
        """
        Record camera close.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.close_count += 1

    def get_frame(self, timeout_ms: int) -> np.ndarray:
        """
        Return a fake PVCAM frame.

        Parameters
        ----------
        timeout_ms
            Timeout requested by the camera wrapper.

        Returns
        -------
        np.ndarray
            Fake uint16 image.
        """
        self.timeout_ms = timeout_ms
        return np.ones(self.shape, dtype=np.uint16)


class FakePVCameraClass:
    """Fake pyvcam Camera class with detect_camera()."""

    fake_camera: FakePVCameraDevice | None = None

    @classmethod
    def detect_camera(cls):
        """
        Yield the configured fake camera.

        Parameters
        ----------
        None

        Returns
        -------
        Iterator[FakePVCameraDevice]
            Iterator yielding one fake camera.
        """
        yield cls.fake_camera


def test_pvcam_camera_factory_with_injected_pyvcam_objects() -> None:
    """
    Check PVCAMCamera behavior with fake injected pyvcam objects.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    image = _image_config()
    pvc_module = FakePVCModule()
    fake_camera = FakePVCameraDevice(shape=image.shape)
    FakePVCameraClass.fake_camera = fake_camera

    camera = CameraFactory.create(
        CameraConfig(
            binding=BindingType.PVCAM,
            image=image,
            default_exposure_time=44,
            readout_mode=CameraReadoutMode.DYNAMIC_RANGE,
        ),
        pvc_module=pvc_module,
        camera_class=FakePVCameraClass,
        frame_timeout_ms=123,
    )

    camera.initialise()
    frame = camera.get_frame()
    camera.finalise()

    assert frame.shape == image.shape
    assert pvc_module.init_count == 1
    assert pvc_module.uninit_count == 1
    assert fake_camera.open_count == 1
    assert fake_camera.close_count == 1
    assert fake_camera.exp_mode == "Internal Trigger"
    assert fake_camera.exp_time == 44
    assert fake_camera.readout_port == 2
    assert fake_camera.timeout_ms == 123
