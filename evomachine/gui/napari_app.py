from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

from evomachine.gui.central_workspace import fit_central_viewer

REFERENCE_WINDOW_WIDTH = 1280
REFERENCE_WINDOW_HEIGHT = 800
REFERENCE_CONTROLS_DOCK_WIDTH = 610
REFERENCE_LOG_DOCK_HEIGHT = 100
CONTROLS_DOCK_WIDTH_RATIO = REFERENCE_CONTROLS_DOCK_WIDTH / REFERENCE_WINDOW_WIDTH
LOG_DOCK_HEIGHT_RATIO = REFERENCE_LOG_DOCK_HEIGHT / REFERENCE_WINDOW_HEIGHT
MINIMUM_CONTROLS_DOCK_WIDTH = 480
MINIMUM_LOG_DOCK_HEIGHT = 72
MAXIMUM_LOG_DOCK_HEIGHT_RATIO = 0.2


def _configure_left_docks(viewer, *, status_dock_widget) -> None:
    """Show image settings and status as tabs while hiding the layer list."""
    qt_window = getattr(viewer.window, "_qt_window", None)
    qt_viewer = getattr(viewer.window, "_qt_viewer", None)
    if qt_window is None or qt_viewer is None:
        return

    layer_controls = getattr(qt_viewer, "dockLayerControls", None)
    layer_list = getattr(qt_viewer, "dockLayerList", None)
    if layer_list is not None:
        layer_list.hide()
    if layer_controls is None:
        return

    set_title = getattr(layer_controls, "setWindowTitle", None)
    if callable(set_title):
        set_title("Image Settings")
    toggle_action = getattr(layer_controls, "toggleViewAction", None)
    if callable(toggle_action):
        action = toggle_action()
        if action is not None:
            action.setText("Image Settings")

    from PyQt5.QtCore import Qt

    qt_window.addDockWidget(Qt.LeftDockWidgetArea, layer_controls)
    layer_controls.show()
    qt_window.tabifyDockWidget(layer_controls, status_dock_widget)
    status_dock_widget.show()
    status_dock_widget.raise_()


def _schedule_startup_central_viewer_fit(
    viewer,
    *,
    controls_dock_widget,
    logs_dock_widget=None,
    status_dock_widget=None,
) -> None:
    """Fit the central viewer to the workspace aspect ratio once Qt has laid out the docks."""
    from PyQt5.QtCore import QTimer

    QTimer.singleShot(
        500,
        lambda viewer=viewer, controls_dock_widget=controls_dock_widget, logs_dock_widget=logs_dock_widget, status_dock_widget=status_dock_widget: (
            _apply_startup_dock_layout(
                viewer,
                controls_dock_widget=controls_dock_widget,
                logs_dock_widget=logs_dock_widget,
                status_dock_widget=status_dock_widget,
            )
        ),
    )


def _apply_startup_dock_layout(
    viewer,
    *,
    controls_dock_widget,
    logs_dock_widget=None,
    status_dock_widget=None,
) -> None:
    _configure_bottom_dock_corners(viewer)
    if status_dock_widget is not None:
        _configure_left_docks(viewer, status_dock_widget=status_dock_widget)
    if logs_dock_widget is not None:
        _resize_logs_dock(viewer, logs_dock_widget=logs_dock_widget)
    _resize_controls_dock(viewer, controls_dock_widget=controls_dock_widget)


def _resize_logs_dock(viewer, *, logs_dock_widget) -> None:
    show = getattr(logs_dock_widget, "show", None)
    if callable(show):
        show()
    set_visible = getattr(logs_dock_widget, "setVisible", None)
    if callable(set_visible):
        set_visible(True)
    raise_dock = getattr(logs_dock_widget, "raise_", None)
    if callable(raise_dock):
        raise_dock()
    set_minimum_height = getattr(logs_dock_widget, "setMinimumHeight", None)
    if callable(set_minimum_height):
        set_minimum_height(0)
    qt_window = getattr(viewer.window, "_qt_window", None)
    resize_docks = getattr(qt_window, "resizeDocks", None)
    if not callable(resize_docks):
        return
    window_height = _widget_extent(qt_window, "height", REFERENCE_WINDOW_HEIGHT)
    target_height = max(
        MINIMUM_LOG_DOCK_HEIGHT,
        round(window_height * LOG_DOCK_HEIGHT_RATIO),
    )
    maximum_height = max(
        target_height,
        round(window_height * MAXIMUM_LOG_DOCK_HEIGHT_RATIO),
    )
    set_maximum_height = getattr(logs_dock_widget, "setMaximumHeight", None)
    if callable(set_maximum_height):
        set_maximum_height(maximum_height)
    from PyQt5.QtCore import Qt

    resize_docks([logs_dock_widget], [target_height], Qt.Vertical)


def _configure_bottom_dock_corners(viewer) -> None:
    """Allow the application-log dock to span the full window width."""
    qt_window = getattr(viewer.window, "_qt_window", None)
    set_corner = getattr(qt_window, "setCorner", None)
    if not callable(set_corner):
        return
    from PyQt5.QtCore import Qt

    set_corner(Qt.BottomLeftCorner, Qt.BottomDockWidgetArea)
    set_corner(Qt.BottomRightCorner, Qt.BottomDockWidgetArea)


def _resize_controls_dock(viewer, *, controls_dock_widget) -> None:
    """Size the two-column controls in proportion to the current window."""
    qt_window = getattr(viewer.window, "_qt_window", None)
    resize_docks = getattr(qt_window, "resizeDocks", None)
    if not callable(resize_docks):
        return
    window_width = _widget_extent(qt_window, "width", REFERENCE_WINDOW_WIDTH)
    target_width = max(
        MINIMUM_CONTROLS_DOCK_WIDTH,
        round(window_width * CONTROLS_DOCK_WIDTH_RATIO),
    )

    from PyQt5.QtCore import Qt, QTimer

    resize_docks(
        [controls_dock_widget],
        [target_width],
        Qt.Horizontal,
    )
    _reset_view(viewer)
    QTimer.singleShot(0, lambda viewer=viewer: _reset_view(viewer))


def _widget_extent(widget, name: str, fallback: int) -> int:
    """Return a positive Qt widget dimension, or a stable startup fallback."""
    extent = getattr(widget, name, None)
    value = extent() if callable(extent) else extent
    if isinstance(value, int | float) and value > 0:
        return round(value)
    return fallback


def _reset_view(viewer) -> None:
    reset_view = getattr(viewer, "reset_view", None)
    if callable(reset_view):
        fit_central_viewer(viewer)


def main(argv: Sequence[str] | None = None) -> int:
    """Launch Napari with the EvoMachine control and status dock widgets."""
    args = list(sys.argv[1:] if argv is None else argv)
    unsupported_options = [arg for arg in args if arg.startswith("-")]
    if unsupported_options:
        raise ValueError(
            "evomachine.gui.napari_app: source-tree launcher only accepts image paths, "
            f"not Napari CLI options: {unsupported_options}."
        )

    import napari

    from evomachine.gui.plugin import (
        ApplicationLogsDock,
        EvoMachineControlsDock,
        PeripheralControllerStatusDock,
    )

    viewer = napari.Viewer()
    _configure_bottom_dock_corners(viewer)
    controls_dock = EvoMachineControlsDock(napari_viewer=viewer)
    controls_dock_widget = viewer.window.add_dock_widget(
        controls_dock, name="EvoMachine Controls", area="right"
    )
    status_dock = PeripheralControllerStatusDock(controller=controls_dock.controller)
    status_dock_widget = viewer.window.add_dock_widget(
        status_dock,
        name="Controller Status",
        area="left",
    )
    logs_dock = ApplicationLogsDock(controller=controls_dock.controller)
    logs_dock_widget = viewer.window.add_dock_widget(
        logs_dock,
        name="Application Logs",
        area="bottom",
        tabify=True,
    )
    _schedule_startup_central_viewer_fit(
        viewer,
        controls_dock_widget=controls_dock_widget,
        logs_dock_widget=logs_dock_widget,
        status_dock_widget=status_dock_widget,
    )
    for path in args:
        viewer.open(str(Path(path)))
    napari.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
