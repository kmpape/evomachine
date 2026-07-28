from __future__ import annotations

import inspect
from types import SimpleNamespace

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


def test_napari_app_fits_central_viewer_to_workspace_aspect() -> None:
    class FakeQtWindow:
        def __init__(self):
            self.calls = []

        def resizeDocks(self, docks, sizes, orientation):  # noqa: N802
            self.calls.append((docks, sizes, orientation))

    class FakeWidget:
        def __init__(self, width, height):
            self._width = width
            self._height = height

        def width(self):
            return self._width

        def height(self):
            return self._height

    qt_window = FakeQtWindow()
    qt_viewer = FakeWidget(width=200, height=1000)
    controls_dock = FakeWidget(width=1000, height=1000)
    viewer = SimpleNamespace(
        reset_count=0,
        window=SimpleNamespace(_qt_window=qt_window, _qt_viewer=qt_viewer),
    )
    viewer.reset_view = lambda: setattr(viewer, "reset_count", viewer.reset_count + 1)

    napari_app._fit_central_viewer_to_workspace(
        viewer,
        controls_dock_widget=controls_dock,
    )

    docks, sizes, _orientation = qt_window.calls[0]
    desired_central_width = round(qt_viewer.height() * napari_app.WORKSPACE_SHAPE[1] / napari_app.WORKSPACE_SHAPE[0])
    expected_controls_width = max(
        napari_app.STARTUP_CONTROLS_DOCK_MIN_WIDTH,
        controls_dock.width() + qt_viewer.width() - desired_central_width,
    )
    assert docks == [controls_dock]
    assert sizes == [expected_controls_width]
    assert viewer.reset_count == 1
