from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path


def _hide_default_napari_left_docks(viewer) -> None:
    """Hide Napari layer controls/list so EvoMachine owns the left dock area."""
    qt_viewer = getattr(viewer.window, "_qt_viewer", None)
    if qt_viewer is None:
        return

    for attr_name in ("dockLayerControls", "dockLayerList"):
        dock = getattr(qt_viewer, attr_name, None)
        if dock is not None:
            dock.hide()


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
    _hide_default_napari_left_docks(viewer)
    controls_dock = EvoMachineControlsDock(napari_viewer=viewer)
    viewer.window.add_dock_widget(controls_dock, name="EvoMachine Controls", area="right")
    status_dock = PeripheralControllerStatusDock(controller=controls_dock.controller)
    viewer.window.add_dock_widget(status_dock, name="Controller Status", area="left")
    for path in args:
        viewer.open(str(Path(path)))
    napari.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
