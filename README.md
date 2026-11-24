# EvoMachine

EvoMachine is a microscope automation application for image acquisition, focus
navigation, DMD projection, experiment strategies, and real-time microscopy
processing. The same orchestration code can run against microscope hardware or
virtual peripherals for development and strategy testing.

## Documentation

Start at the [documentation index](docs/README.md).

- [User guide](docs/user/README.md): run the GUI, acquire images, configure focus,
  use virtual devices, and write Python or AutoStrat strategies.
- [Developer guide](docs/development/README.md): architecture, configuration,
  bindings, GUI integration, runtime failures, image processing, and testing.
- [Configuration reference](docs/development/configuration.md): public configuration
  objects, defaults, validation, units, and update behavior.

### Preview the documentation

Render the documentation index in a browser without adding another project
dependency:

```bash
uvx grip docs/README.md
```

Open `http://localhost:6419`, or ask Grip to open the browser automatically:

```bash
uvx grip --browser docs/README.md
```

Stop the preview server with `Ctrl+C`. In VS Code, open any Markdown file and
press `Ctrl+Shift+V` to use the built-in preview.

## Installation

EvoMachine requires Python 3.11 and uses
[uv](https://docs.astral.sh/uv/) for dependency management.

For local development, clone the sibling repositories beside this repository:

```text
workspace/
├── evomachine/
├── AutoStrat/
├── de-lta-rt/
├── asitiger/
├── sync_board/
└── em_dmd_window/       # required only by that DMD backend
```

Then run from the EvoMachine repository root:

```bash
uv sync
```

The local sibling repositories are installed in editable mode through
`[tool.uv.sources]` in `pyproject.toml`.

For a production installation using the dependency revisions in `uv.lock`:

```bash
uv sync --locked --no-sources --no-dev
```

Optional hardware and strategy-generation dependencies are installed with
extras:

```bash
uv sync --extra pvcam
uv sync --extra autostrat
```

`pyvcam` additionally requires the proprietary PVCAM SDK. The AutoStrat extra
requires access to its repository. See the [developer setup guide](docs/development/testing.md)
for dependency and release details.

## Run the application

Launch the virtual GUI:

```bash
uv run python scripts/launch_virtual_gui.py
```

Launch the configured microscope hardware GUI:

```bash
uv run python scripts/launch_hardware_gui.py
```

The hardware launcher must be used only after its ports, travel limits,
calibration files, and installed devices have been checked for the microscope.
See [Using the GUI](docs/user/gui.md) and
[Application configuration](docs/development/configuration.md).

To smoke-test the virtual backend without opening Napari:

```bash
uv run python scripts/launch_virtual_gui.py --port 0 --no-napari
```

## Test

Run the complete test suite from this repository:

```bash
uv run pytest
```

Some Qt tests are opt-in:

```bash
EVOMACHINE_GUI_RUN_QT_TESTS=1 QT_QPA_PLATFORM=offscreen uv run pytest tests/gui
```

## Runtime data

Acquired images, logs, and GUI-saved AutoStrat files live below one data
directory. Configure it in `/etc/evomachine/config.toml` or
`~/.config/evomachine/config.toml`:

```toml
[paths]
data_dir = "/scratch/$USER/microscope-data"
```

The user file overrides the machine file. If neither exists, the repository
root is used for compatibility. See [System overview](docs/user/overview.md#runtime-data).
