from pathlib import Path
import os
from types import SimpleNamespace

import numpy as np
import pytest

from evomachine.bindings.em_dmd_window.dmd import EmDmdWindowDmd
from evomachine.bindings.em_dmd_window.peripheralcontroller import EmDmdWindowPeripheralController
from evomachine.bindings.pygame.dmd import PygameDmd
from evomachine.bindings.pygame.peripheralcontroller import PygameDmdPeripheralController
from evomachine.bindings.virtual.dmd import VirtualDmd, VirtualDmdPeripheralController
from evomachine.peripherals.dmd import Dmd, DmdConfig, DmdFactory
from evomachine.peripherals.peripherals import PeripheralController, SocketPeripheralController
from evomachine.bindings.binding_types import BindingType

# TODO(CODEX): Make these Fake classes import dependent. If some global variable is true, the real classes are imported and the real bindings tested. For security reasons, we need test settings defined somewhere.

class FakeSocket:
    def __init__(self):
        self.closed = False
        self.sent = []

    def sendall(self, data):
        self.sent.append(data)

    def close(self):
        self.closed = True


class FakeSurface:
    def __init__(self):
        self.blits = []

    def blit(self, surface, position):
        self.blits.append((surface, position))


class FakePygame:
    NOFRAME = 1
    FULLSCREEN = 2

    def __init__(self):
        self.updated = False
        self.quit_called = False
        self.modes = []
        self.surfarray = SimpleNamespace(make_surface=lambda img: ("surface", img.copy()))
        self.display = SimpleNamespace(update=self.update, set_mode=self.set_mode)

    def init(self):
        return

    def set_mode(self, size, flags=0):
        self.modes.append((size, flags))
        return FakeSurface()

    def update(self):
        self.updated = True

    def quit(self):
        self.quit_called = True


class NotReadyController(PeripheralController):
    def __init__(self):
        super().__init__(name="not-ready")

    def _initialise(self, force=False):
        return False

    def _check_is_alive(self):
        return False

    def _stop(self):
        return

    def _shutdown(self, force=False):
        return


class RecordingDmd(Dmd):
    def __init__(self, *args, **kwargs):
        self.images = []
        super().__init__(*args, **kwargs)

    def display_image(self, img: np.ndarray, _is_full_display: bool = False) -> None:
        self._check_ready()
        img = self._normalise_display_image(img=img)
        self.images.append(img.copy())
        self._is_full_display = _is_full_display


class SocketTestController(SocketPeripheralController):
    def __init__(self, socket_obj, close_on_shutdown=True):
        self.s = socket_obj
        self.before_disconnect_called = False
        super().__init__(name="socket-test", close_on_shutdown=close_on_shutdown)

    def _get_socket_controller(self):
        return self

    def _initialise(self, force=False):
        return True

    def _check_is_alive(self):
        return True

    def _stop(self):
        return

    def _before_disconnect(self, force=False):
        self.before_disconnect_called = True


def make_calibration_file(tmp_path: Path) -> Path:
    calibration_file = tmp_path / "calibration.pkl"
    calibration_data = [
        ((0, 0), (0, 0), (0, 0)),
        ((0, 9), (0, 9), (0, 0)),
        ((9, 0), (9, 0), (0, 0)),
        ((9, 9), (9, 9), (0, 0)),
    ]
    import pickle as pkl

    with open(calibration_file, "wb") as file:
        pkl.dump(calibration_data, file)
    return calibration_file


def make_recording_dmd(tmp_path, width_height=(10, 10), initialise=True) -> RecordingDmd:
    controller = VirtualDmdPeripheralController()
    if initialise:
        controller.initialise()
    return RecordingDmd(
        peripheral_ctrl=controller,
        width_height_DMD=width_height,
        width_height_CAM=width_height,
        calibration_file=make_calibration_file(tmp_path),
    )


def make_dmd_config(tmp_path: Path, binding: BindingType, **updates) -> DmdConfig:
    """
    Return a DmdConfig with small test dimensions and calibration data.

    Parameters
    ----------
    tmp_path
        Pytest temporary directory fixture.
    binding
        BindingType used for the DMD factory.
    **updates
        DmdConfig fields to override.

    Returns
    -------
    DmdConfig
        Valid DMD configuration for tests.
    """
    values = {
        "binding": binding,
        "width_height_DMD": (10, 10),
        "width_height_CAM": (10, 10),
        "calibration_file": make_calibration_file(tmp_path),
    }
    values.update(updates)
    return DmdConfig(**values)


def test_dmd_config_validates_fields():
    with pytest.raises(TypeError):
        DmdConfig(binding="virtual")
    with pytest.raises(TypeError):
        DmdConfig(binding=BindingType.VIRTUAL, name=123)
    with pytest.raises(TypeError):
        DmdConfig(binding=BindingType.VIRTUAL, check_alive="yes")
    with pytest.raises(TypeError):
        DmdConfig(binding=BindingType.VIRTUAL, width_height_DMD=[10, 10])
    with pytest.raises(TypeError):
        DmdConfig(binding=BindingType.VIRTUAL, width_height_CAM=(10, True))
    with pytest.raises(ValueError):
        DmdConfig(binding=BindingType.VIRTUAL, width_height_DMD=(10, 0))
    with pytest.raises(ValueError):
        DmdConfig(binding=BindingType.VIRTUAL, display_offset=(-1, 0))
    with pytest.raises(TypeError):
        DmdConfig(binding=BindingType.VIRTUAL, monitor_index=True)
    with pytest.raises(ValueError):
        DmdConfig(binding=BindingType.VIRTUAL, monitor_index=-1)
    with pytest.raises(TypeError):
        DmdConfig(binding=BindingType.VIRTUAL, calibration_file=123)


def test_shared_display_helpers_generate_arrays_and_call_display_image(tmp_path):
    dmd = make_recording_dmd(tmp_path)

    dmd.display_full(force_display=True)
    assert np.all(dmd.images[-1] == 255)
    assert dmd.is_full_display()

    dmd.display_none()
    assert np.all(dmd.images[-1] == 0)
    assert not dmd.is_full_display()

    dmd.display_circle(row=5, col=5, radius=2)
    assert dmd.images[-1][5, 5] == 255

    dmd.display_crosshair()
    assert dmd.images[-1][5, :].max() == 255
    assert dmd.images[-1][:, 5].max() == 255


def test_display_calls_reject_when_controller_is_not_ready(tmp_path):
    dmd = RecordingDmd(
        peripheral_ctrl=NotReadyController(),
        width_height_DMD=(10, 10),
        width_height_CAM=(10, 10),
        calibration_file=make_calibration_file(tmp_path),
    )

    with pytest.raises(RuntimeError, match="not initialised"):
        dmd.display_full(force_display=True)


def test_calibration_getters_and_coordinate_transforms(tmp_path):
    dmd = make_recording_dmd(tmp_path)

    assert dmd.is_calibrated()
    assert dmd.get_calibration_filename().name == "calibration.pkl"
    assert dmd.img_to_dmd_coords(3, 4) == (3, 4)
    assert dmd.dmd_to_img_coords(6, 7) == (6, 7)
    assert dmd.img_to_dmd_array(np.ones((10, 10), dtype=np.uint8)).shape == (10, 10)
    assert dmd.dmd_to_img_array(np.ones((10, 10), dtype=np.uint8)).shape == (10, 10)


def test_roi_pattern_helpers_live_on_dmd(tmp_path):
    dmd = make_recording_dmd(tmp_path)
    boxes = [
        SimpleNamespace(xtl=1, ytl=1, xbr=3, ybr=3),
        SimpleNamespace(xtl=6, ytl=1, xbr=8, ybr=3),
    ]

    patches = dmd.patches_from_roi_groups([[0], [1]], boxes)
    pattern = dmd.pattern_from_roi_boxes(boxes=boxes, warp=False, border_px=0)

    assert len(patches) == 3
    assert pattern[1:4, 1:4].max() == 255
    assert pattern[1:4, 6:9].max() == 255


def test_load_image_and_display_loaded_image_use_shared_state(tmp_path):
    dmd = make_recording_dmd(tmp_path)
    image_file = tmp_path / "image.png"
    import skimage.io

    skimage.io.imsave(image_file, np.ones((10, 10), dtype=np.uint8) * 127)

    loaded = dmd.load_image(str(image_file), display_image=False)
    dmd.display_loaded_image()

    assert loaded.shape == (10, 10)
    assert np.array_equal(dmd.images[-1], loaded)


def test_load_constant_non_uint8_image_normalises_without_nan(tmp_path):
    dmd = make_recording_dmd(tmp_path)
    image_file = tmp_path / "image.tif"
    import skimage.io

    skimage.io.imsave(image_file, np.ones((10, 10), dtype=np.uint16) * 5)

    loaded = dmd.load_image(str(image_file), display_image=False)

    assert loaded.dtype == np.uint8
    assert np.all(loaded == 0)


def test_dmd_factory_creates_socket_pygame_and_virtual_wrappers(tmp_path):
    socket_controller = EmDmdWindowPeripheralController(socket_obj=FakeSocket(), debug_mode=True)
    pygame_controller = PygameDmdPeripheralController(debug_mode=True)
    virtual_controller = VirtualDmdPeripheralController()

    assert isinstance(DmdFactory.create(
        make_dmd_config(tmp_path, BindingType.EM_DMD_WINDOW),
        peripheral_controllers=[socket_controller],
    ), EmDmdWindowDmd)
    assert isinstance(DmdFactory.create(
        make_dmd_config(tmp_path, BindingType.PYGAME),
        peripheral_controllers=[pygame_controller],
    ), PygameDmd)
    assert isinstance(DmdFactory.create(
        make_dmd_config(tmp_path, BindingType.VIRTUAL),
        peripheral_controllers=[virtual_controller],
    ), VirtualDmd)


def test_dmd_factory_raises_for_missing_controller():
    with pytest.raises(ValueError, match="EmDmdWindowPeripheralController is required"):
        DmdFactory.create(DmdConfig(binding=BindingType.EM_DMD_WINDOW), peripheral_controllers=[])


def test_dmd_factory_rejects_socket_display_placement_options(tmp_path):
    """
    Check socket-backed DMD rejects unsupported display placement options.

    Parameters
    ----------
    tmp_path
        Pytest temporary directory fixture.

    Returns
    -------
    None
    """
    socket_controller = EmDmdWindowPeripheralController(socket_obj=FakeSocket(), debug_mode=True)

    with pytest.raises(NotImplementedError, match="display_offset"):
        DmdFactory.create(
            make_dmd_config(tmp_path, BindingType.EM_DMD_WINDOW, display_offset=(10, 20)),
            peripheral_controllers=[socket_controller],
        )
    with pytest.raises(NotImplementedError, match="monitor_index"):
        DmdFactory.create(
            make_dmd_config(tmp_path, BindingType.EM_DMD_WINDOW, monitor_index=1),
            peripheral_controllers=[socket_controller],
        )


def test_dmd_factory_rejects_unsupported_shared_binding():
    """Check that shared BindingType values are still scoped per DMD factory."""
    with pytest.raises(ValueError, match="unsupported binding"):
        DmdFactory.create(DmdConfig(binding=BindingType.ASI_TIGER), peripheral_controllers=[])


def test_socket_controller_shutdown_honours_close_on_shutdown():
    fake_socket = FakeSocket()
    controller = SocketTestController(socket_obj=fake_socket, close_on_shutdown=False)

    controller.shutdown()
    assert controller.before_disconnect_called
    assert not fake_socket.closed

    controller.shutdown(force=True)
    assert fake_socket.closed


def test_em_dmd_window_binding_sends_transposed_image_bytes(tmp_path):
    fake_socket = FakeSocket()
    controller = EmDmdWindowPeripheralController(socket_obj=fake_socket, debug_mode=False)
    controller.initialise()
    dmd = EmDmdWindowDmd(
        peripheral_ctrl=controller,
        width_height_DMD=(2, 3),
        width_height_CAM=(2, 3),
        calibration_file=make_calibration_file(tmp_path),
    )
    img = np.arange(6, dtype=np.uint8).reshape((2, 3))

    dmd.display_image(img)

    assert fake_socket.sent[-1] == img.transpose().tobytes()


def test_pygame_binding_displays_through_pygame_controller(tmp_path):
    fake_pygame = FakePygame()
    surface = FakeSurface()
    controller = PygameDmdPeripheralController(surface=surface, pygame_module=fake_pygame)
    controller.initialise()
    dmd = PygameDmd(
        peripheral_ctrl=controller,
        width_height_DMD=(2, 3),
        width_height_CAM=(2, 3),
        calibration_file=make_calibration_file(tmp_path),
    )
    img = np.ones((2, 3), dtype=np.uint8) * 255

    dmd.display_image(img)

    assert surface.blits[-1][1] == (0, 0)
    assert fake_pygame.updated


def test_pygame_factory_configures_controller_display_options(tmp_path):
    """
    Check pygame factory passes size, display offset, and monitor index to the controller.

    Parameters
    ----------
    tmp_path
        Pytest temporary directory fixture.

    Returns
    -------
    None
    """
    controller = PygameDmdPeripheralController(debug_mode=True)
    config = make_dmd_config(
        tmp_path,
        BindingType.PYGAME,
        width_height_DMD=(2, 3),
        width_height_CAM=(2, 3),
        display_offset=(40, 50),
        monitor_index=1,
    )

    dmd = DmdFactory.create(config, peripheral_controllers=[controller])

    assert isinstance(dmd, PygameDmd)
    assert controller.size == (2, 3)
    assert controller.display_offset == (40, 50)
    assert controller.monitor_index == 1


def test_pygame_initialise_sets_sdl_display_environment(tmp_path, monkeypatch):
    """
    Check pygame initialisation sets SDL placement environment variables.

    Parameters
    ----------
    tmp_path
        Pytest temporary directory fixture.
    monkeypatch
        Pytest monkeypatch fixture.

    Returns
    -------
    None
    """
    fake_pygame = FakePygame()
    controller = PygameDmdPeripheralController(
        size=(2, 3),
        display_offset=(40, 50),
        monitor_index=2,
        pygame_module=fake_pygame,
    )
    monkeypatch.delenv("SDL_VIDEO_WINDOW_POS", raising=False)
    monkeypatch.delenv("SDL_VIDEO_FULLSCREEN_DISPLAY", raising=False)

    controller.initialise()

    assert os.environ["SDL_VIDEO_WINDOW_POS"] == "40,50"
    assert os.environ["SDL_VIDEO_FULLSCREEN_DISPLAY"] == "2"
    assert fake_pygame.modes[-1] == ((2, 3), fake_pygame.NOFRAME | fake_pygame.FULLSCREEN)


def test_virtual_binding_stores_displayed_images(tmp_path):
    controller = VirtualDmdPeripheralController()
    controller.initialise()
    dmd = VirtualDmd(
        peripheral_ctrl=controller,
        width_height_DMD=(2, 3),
        width_height_CAM=(2, 3),
        calibration_file=make_calibration_file(tmp_path),
    )
    img = np.ones((2, 3), dtype=np.uint8) * 255

    dmd.display_image(img, _is_full_display=True)

    assert np.array_equal(controller.dmd_control._image, img)
    assert dmd.is_full_display()
