from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from evomachine.bindings.binding_types import BindingType
from evomachine.gui import launchers
from evomachine.gui import napari_app
from evomachine.gui.image_payloads import IMAGE_TRANSPORT_ENV, IMAGE_TRANSPORT_SOCKET_TIFF


def test_launchers_do_not_import_legacy_guidir() -> None:
    assert "guidir" not in inspect.getsource(launchers)


def test_runtime_factory_spec_requires_module_and_function() -> None:
    try:
        launchers._load_runtime_factory("missing_separator")
    except ValueError as error:
        assert "module:function" in str(error)
    else:
        raise AssertionError("expected ValueError")


def test_hardware_gui_runtime_accepts_micromanager_camera() -> None:
    camera = SimpleNamespace(name="camera", config=SimpleNamespace(binding=BindingType.MMC))
    automaton = SimpleNamespace(acq_mngr=SimpleNamespace(camera=camera))

    launchers._require_hardware_gui_mmc_camera(automaton)


def test_hardware_gui_runtime_rejects_pvcam_camera() -> None:
    camera = SimpleNamespace(name="camera", config=SimpleNamespace(binding=BindingType.PVCAM))
    automaton = SimpleNamespace(acq_mngr=SimpleNamespace(camera=camera))

    with pytest.raises(RuntimeError, match="BindingType.MMC"):
        launchers._require_hardware_gui_mmc_camera(automaton)


def test_hardware_gui_has_default_runtime_factory() -> None:
    factory = launchers._load_runtime_factory(launchers.DEFAULT_HARDWARE_RUNTIME)

    assert factory.__name__ == "build_hardware_automaton"


def test_run_napari_passes_forced_image_transport_to_child_environment(monkeypatch) -> None:
    captured = {}

    def fake_run(command, env, check):
        captured["command"] = command
        captured["env"] = env
        captured["check"] = check
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(launchers.subprocess, "run", fake_run)

    result = launchers._run_napari(
        host="127.0.0.1",
        port=8765,
        napari_args=[],
        image_transport=IMAGE_TRANSPORT_SOCKET_TIFF,
    )

    assert result == 0
    assert captured["env"][IMAGE_TRANSPORT_ENV] == IMAGE_TRANSPORT_SOCKET_TIFF


@pytest.mark.parametrize(
    ("window_width", "expected_width"),
    [(1280, 610), (1920, 915), (800, napari_app.MINIMUM_CONTROLS_DOCK_WIDTH)],
)
def test_napari_app_adapts_two_column_controls_to_window_width(
    window_width, expected_width
) -> None:
    class FakeQtWindow:
        def __init__(self, width):
            self.calls = []
            self._width = width

        def width(self):
            return self._width

        def resizeDocks(self, docks, sizes, orientation):  # noqa: N802
            self.calls.append((docks, sizes, orientation))

    qt_window = FakeQtWindow(window_width)
    controls_dock = object()
    viewer = SimpleNamespace(
        reset_count=0,
        window=SimpleNamespace(_qt_window=qt_window),
    )
    viewer.reset_view = lambda: setattr(viewer, "reset_count", viewer.reset_count + 1)

    napari_app._resize_controls_dock(
        viewer,
        controls_dock_widget=controls_dock,
    )

    docks, sizes, _orientation = qt_window.calls[0]
    assert docks == [controls_dock]
    assert sizes == [expected_width]
    assert viewer.reset_count == 1


def test_napari_app_hides_layer_list_and_tabifies_status_with_layer_controls() -> None:
    class FakeDock:
        def __init__(self):
            self.shown = False
            self.hidden = False
            self.raised = False
            self.title = None

        def show(self):
            self.shown = True

        def hide(self):
            self.hidden = True

        def raise_(self):
            self.raised = True

        def setWindowTitle(self, title):  # noqa: N802
            self.title = title

    class FakeQtWindow:
        def __init__(self):
            self.added = []
            self.tabified = []

        def addDockWidget(self, area, dock):  # noqa: N802
            self.added.append((area, dock))

        def tabifyDockWidget(self, first, second):  # noqa: N802
            self.tabified.append((first, second))

    controls = FakeDock()
    layers = FakeDock()
    status = FakeDock()
    qt_window = FakeQtWindow()
    viewer = SimpleNamespace(
        window=SimpleNamespace(
            _qt_window=qt_window,
            _qt_viewer=SimpleNamespace(
                dockLayerControls=controls,
                dockLayerList=layers,
            ),
        )
    )

    napari_app._configure_left_docks(viewer, status_dock_widget=status)

    assert controls.shown
    assert controls.title == "Image Settings"
    assert layers.hidden
    assert qt_window.added[0][1] is controls
    assert qt_window.tabified == [(controls, status)]
    assert status.shown
    assert status.raised


@pytest.mark.parametrize(("window_height", "expected_height"), [(800, 100), (1200, 150)])
def test_napari_app_adapts_bottom_log_dock_to_window_height(window_height, expected_height) -> None:
    class FakeDock:
        def __init__(self):
            self.shown = False
            self.visible = False
            self.raised = False
            self.minimum_height = None
            self.maximum_height = None

        def show(self):
            self.shown = True

        def setVisible(self, visible):  # noqa: N802
            self.visible = visible

        def raise_(self):
            self.raised = True

        def setMinimumHeight(self, height):  # noqa: N802
            self.minimum_height = height

        def setMaximumHeight(self, height):  # noqa: N802
            self.maximum_height = height

    class FakeQtWindow:
        def __init__(self, height):
            self.calls = []
            self._height = height

        def height(self):
            return self._height

        def resizeDocks(self, docks, sizes, orientation):  # noqa: N802
            self.calls.append((docks, sizes, orientation))

    logs_dock = FakeDock()
    qt_window = FakeQtWindow(window_height)
    viewer = SimpleNamespace(window=SimpleNamespace(_qt_window=qt_window))

    napari_app._resize_logs_dock(viewer, logs_dock_widget=logs_dock)

    docks, sizes, _orientation = qt_window.calls[0]
    assert docks == [logs_dock]
    assert sizes == [expected_height]
    assert logs_dock.shown
    assert logs_dock.visible
    assert logs_dock.raised
    assert logs_dock.minimum_height == 0
    assert logs_dock.maximum_height == round(window_height * 0.2)
