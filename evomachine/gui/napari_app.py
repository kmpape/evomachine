from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path


def main(argv: Sequence[str] | None = None) -> int:
    """Launch Napari with the EvoMachine GUI dock widget."""
    args = list(sys.argv[1:] if argv is None else argv)
    unsupported_options = [arg for arg in args if arg.startswith("-")]
    if unsupported_options:
        raise ValueError(
            "evomachine.gui.napari_app: source-tree launcher only accepts image paths, "
            f"not Napari CLI options: {unsupported_options}."
        )

    import napari

    from evomachine.gui.plugin import EvoMachineNapariWidget

    viewer = napari.Viewer()
    widget = EvoMachineNapariWidget(napari_viewer=viewer)
    viewer.window.add_dock_widget(widget, name="EvoMachine GUI", area="right")
    for path in args:
        viewer.open(str(Path(path)))
    napari.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
