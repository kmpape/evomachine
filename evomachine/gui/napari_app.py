from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

from evomachine.gui.central_workspace import WORKSPACE_SHAPE

CENTRAL_VIEWER_FIT_TOLERANCE = 8
STARTUP_CONTROLS_DOCK_MIN_WIDTH = 480


def _show_default_napari_left_docks(viewer) -> None:
    """Expose Napari layer controls so camera contrast and gamma remain adjustable."""
    qt_viewer = getattr(viewer.window, "_qt_viewer", None)
    if qt_viewer is None:
        return

    for attr_name in ("dockLayerControls", "dockLayerList"):
        dock = getattr(qt_viewer, attr_name, None)
        if dock is not None:
            dock.show()


def _stack_status_above_layer_controls(viewer, *, status_dock_widget) -> None:
    """Place controller status above Napari's image/contrast controls."""
    qt_window = getattr(viewer.window, "_qt_window", None)
    qt_viewer = getattr(viewer.window, "_qt_viewer", None)
    split_dock_widget = getattr(qt_window, "splitDockWidget", None)
    layer_controls = getattr(qt_viewer, "dockLayerControls", None)
    if not callable(split_dock_widget) or layer_controls is None:
        return

    from PyQt5.QtCore import Qt

    split_dock_widget(status_dock_widget, layer_controls, Qt.Vertical)


def _schedule_left_dock_layout(viewer, *, status_dock_widget) -> None:
    """Apply left-dock ordering after Qt has completed Napari's dock layout."""
    from PyQt5.QtCore import QTimer

    for delay_ms in (0, 150):
        QTimer.singleShot(
            delay_ms,
            lambda viewer=viewer, status_dock_widget=status_dock_widget: (
                _stack_status_above_layer_controls(
                    viewer,
                    status_dock_widget=status_dock_widget,
                )
            ),
        )


def _schedule_startup_central_viewer_fit(viewer, *, controls_dock_widget) -> None:
    """Fit the central viewer to the workspace aspect ratio once Qt has laid out the docks."""
    from PyQt5.QtCore import QTimer

    for delay_ms in (0, 150):
        QTimer.singleShot(
            delay_ms,
            lambda viewer=viewer, controls_dock_widget=controls_dock_widget: _fit_central_viewer_to_workspace(
                viewer,
                controls_dock_widget=controls_dock_widget,
            ),
        )


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

    from evomachine.gui.plugin import EvoMachineControlsDock, PeripheralControllerStatusDock

    viewer = napari.Viewer()
    _show_default_napari_left_docks(viewer)
    controls_dock = EvoMachineControlsDock(napari_viewer=viewer)
    controls_dock_widget = viewer.window.add_dock_widget(controls_dock, name="EvoMachine Controls", area="right")
    status_dock = PeripheralControllerStatusDock(controller=controls_dock.controller)
    status_dock_widget = viewer.window.add_dock_widget(status_dock, name="Controller Status", area="left")
    _schedule_left_dock_layout(
        viewer,
        status_dock_widget=status_dock_widget,
    )
    _schedule_startup_central_viewer_fit(
        viewer,
        controls_dock_widget=controls_dock_widget,
    )
    for path in args:
        viewer.open(str(Path(path)))
    napari.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
