from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

from evomachine.gui.central_workspace import WORKSPACE_SHAPE

CENTRAL_VIEWER_FIT_TOLERANCE = 8
STARTUP_CONTROLS_DOCK_MIN_WIDTH = 480
STARTUP_LOG_DOCK_HEIGHT = 210


def _configure_left_docks(viewer, *, status_dock_widget) -> None:
    """Show layer controls and status as tabs while hiding the layer list."""
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

    for delay_ms in (0, 150, 500):
        QTimer.singleShot(
            delay_ms,
            lambda viewer=viewer,
            controls_dock_widget=controls_dock_widget,
            logs_dock_widget=logs_dock_widget,
            status_dock_widget=status_dock_widget: _apply_startup_dock_layout(
                viewer,
                controls_dock_widget=controls_dock_widget,
                logs_dock_widget=logs_dock_widget,
                status_dock_widget=status_dock_widget,
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
    _fit_central_viewer_to_workspace(viewer, controls_dock_widget=controls_dock_widget)


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
        set_minimum_height(STARTUP_LOG_DOCK_HEIGHT)
    qt_window = getattr(viewer.window, "_qt_window", None)
    resize_docks = getattr(qt_window, "resizeDocks", None)
    if not callable(resize_docks):
        return
    from PyQt5.QtCore import Qt

    resize_docks([logs_dock_widget], [STARTUP_LOG_DOCK_HEIGHT], Qt.Vertical)


def _configure_bottom_dock_corners(viewer) -> None:
    """Allow the application-log dock to span the full window width."""
    qt_window = getattr(viewer.window, "_qt_window", None)
    set_corner = getattr(qt_window, "setCorner", None)
    if not callable(set_corner):
        return
    from PyQt5.QtCore import Qt

    set_corner(Qt.BottomLeftCorner, Qt.BottomDockWidgetArea)
    set_corner(Qt.BottomRightCorner, Qt.BottomDockWidgetArea)


def _fit_central_viewer_to_workspace(viewer, *, controls_dock_widget) -> None:
    """Resize the right dock so the central viewer matches the workspace aspect ratio."""
    qt_window = getattr(viewer.window, "_qt_window", None)
    qt_viewer = getattr(viewer.window, "_qt_viewer", None)
    resize_docks = getattr(qt_window, "resizeDocks", None)
    if not callable(resize_docks) or qt_viewer is None:
        return

    central_width = _qt_dimension(qt_viewer, "width")
    central_height = _qt_dimension(qt_viewer, "height")
    controls_width = _qt_dimension(controls_dock_widget, "width")
    if central_width is None or central_height is None or controls_width is None or central_height <= 0:
        return

    desired_width = round(central_height * WORKSPACE_SHAPE[1] / WORKSPACE_SHAPE[0])
    width_delta = central_width - desired_width
    if abs(width_delta) <= CENTRAL_VIEWER_FIT_TOLERANCE:
        _reset_view(viewer)
        return

    new_controls_width = max(STARTUP_CONTROLS_DOCK_MIN_WIDTH, controls_width + width_delta)

    from PyQt5.QtCore import Qt

    resize_docks(
        [controls_dock_widget],
        [new_controls_width],
        Qt.Horizontal,
    )
    _reset_view(viewer)


def _qt_dimension(widget, dimension_name: str) -> int | None:
    dimension = getattr(widget, dimension_name, None)
    if callable(dimension):
        return int(dimension())
    if isinstance(dimension, int | float):
        return int(dimension)
    return None


def _reset_view(viewer) -> None:
    reset_view = getattr(viewer, "reset_view", None)
    if callable(reset_view):
        reset_view()


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
    controls_dock_widget = viewer.window.add_dock_widget(controls_dock, name="EvoMachine Controls", area="right")
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
