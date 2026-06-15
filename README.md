# Evomachine

Evomachine is the main microscope automation application in this workspace. It
coordinates hardware peripherals, image acquisition, focus navigation,
projection, strategies, and real-time processing workflows.

## Installation

Dependencies are managed with [uv](https://docs.astral.sh/uv/).

### 1. Clone the sibling repositories

*Note: this is currently a bit of a hacky way to install other dependencies. The aim is to clean it up soon.*

Three sibling repositories are installed by `uv` as **editable path
dependencies** (declared in `[tool.uv.sources]`). They must already be cloned
next to `evomachine` in the same parent folder, and checked out on the correct
branch, *before* you run `uv sync` — otherwise the install will fail or pick up
the wrong code:

| Sibling folder | Imported as | Required branch |
|---|---|---|
| `de-lta-rt/`  | `delta`     | `dev_main` |
| `asitiger/`   | `asitiger`  | `master`   |
| `sync_board/` | `syncboard` | `Signals`  |

Expected layout:

```
workspace/
├── evomachine/      ← this repo
├── de-lta-rt/       (dev_main)
├── asitiger/        (master)
└── sync_board/      (Signals)
```

The exact branches matter: e.g. `delta`'s real-time modules (`delta.rttypes`,
`delta.rt`, `delta.imgops`) only exist on `de-lta-rt`'s `dev_main` branch.

### 2. Install the environment

From the `evomachine` repo root:

```bash
# creates .venv and installs: curated runtime deps, the three editable sibling
# packages, the evomachine package (editable), and the dev tools
uv sync

# run anything inside the environment
uv run python scripts/launch_gui.py
uv run pytest
```

`uv sync` reads `pyproject.toml` and pins everything in `uv.lock`. This
application targets the Linux + CUDA microscope machine, including the GPU stack.

### Dev dependencies

The dev tools (`pytest`, `pytest-cov`, `ruff`, `pre-commit`) live in the `dev`
dependency group, which `uv sync` installs **by default**. For a runtime-only
environment (e.g. on the microscope machine), skip them with:

```bash
uv sync --no-dev
```

### Out-of-band dependencies

A couple of runtime dependencies are **not** installed by `uv sync` and must be
provided separately:

- **`em_dmd_window/`** — used by the DMD projection bindings via a compiled
  binary (referenced by path, not pip-installed). Clone it as a sibling
  (branch `master`) and build it.
- **`pyvcam`** (Photometrics PVCAM Python wrapper) — only needed for the PVCAM
  camera backend. Installed from the vendor, not on PyPI.

## Workspace Structure

This project expects several sibling repositories to live next to the main
`evomachine` repository:

- `evomachine/`: main application repository.  
  URL: `https://github.com/kmpape/evomachine`  
  Branches: `dev` (in use) and `refactor`
- `asitiger/`: ASI Tiger controller package used by Tiger hardware bindings.  
  URL: `https://github.com/kmpape/asitiger` (forked from `https://github.com/herophilus/asitiger`)  
  Branches: 
- `sync_board/`: SyncBoard controller package used by SyncBoard bindings.  
  URL: `https://github.com/kmpape/sync_board`  
  Branches: `Signals` (in use), `master` (refactor, differences unclear)
- `de-lta-rt/`: DE-LTA real-time segmentation and tracking package.  
  URL: `https://gitlab.com/kmpape/de-lta-rt`  
  Branches: `dev_main`
- `em_dmd_window/`: DMD display/window helper used by projection bindings.  
  URL: `https://github.com/kmpape/em_dmd_window`
  Branches: `master`  

Keeping these repositories as siblings lets scripts, notebooks, and local
imports resolve the hardware and processing dependencies during development.  

Additionally, a Windows PC runs the microfluidics controls independently:  
- URL: `https://github.com/KSechkar/MM_microfluidics'  

## Code Structure

The main Python package is `evomachine/evomachine`.

- `peripherals/` defines the hardware-facing base classes and typed configs for
  cameras, stages, LEDs, filter wheels, autofocus, photodiodes, DMDs, and shared
  peripheral controllers.
- `bindings/` contains concrete implementations for hardware and software
  backends, including ASI Tiger, SyncBoard, Micro-Manager, PVCAM, pygame, DMD
  window, and virtual devices.
- `acquisition.py`, `navigation.py`, and `projection.py` provide focused
  managers for frame capture, focus/stage navigation, and DMD projection tasks.
- `commands.py`, `strategy.py`, and `automaton.py` describe command objects,
  strategy generation, and high-level experiment orchestration.
- `coordinates.py`, `types.py`, `config_types.py`, and `filemanager.py` contain
  shared data types, metadata, coordinate handling, and file output utilities.
- `softwarefocus.py` and `trackingrt.py` support software focus and real-time
  tracking workflows.
- `gui/` contains a Napari-based GUI that can be used to control the evomachine.

Other useful top-level folders in the main repository:

- `strategies/` contains example strategies and can be populated by users. The
  GUI will scan this folder to list strategy selection options.
- `tests/`: pytest coverage for the package. Run `pytest tests` to execute test suite.
- `scripts/`: runnable hardware and workflow scripts.
- `notebooks/`: interactive notebooks for smoke tests and exploratory work.
- `calibrations/`, `data/`, `delta_models/`, and `images/`:
  runtime inputs, outputs, models, calibration data, and strategy files.  
  Note: some folders are only created when starting the software.

## GUI Demo

After activating the `delta_evomachine` environment, launch the Napari GUI demo
with virtual peripherals from the repository root:

```bash
cd /home/idris/workspace_python/conda_evomachine3.9/evomachine_repo
conda activate delta_evomachine
python scripts/launch_gui.py --port 0
```

To smoke-test the automaton/socket startup without opening Napari:

```bash
python scripts/launch_gui.py --port 0 --no-napari
```
