# Evomachine

Evomachine is the main microscope automation application in this workspace. It
coordinates hardware peripherals, image acquisition, focus navigation,
projection, strategies, and real-time processing workflows.

## Installation

Dependencies are managed with [uv](https://docs.astral.sh/uv/) and specified in the [pyproject.toml](pyproject.toml) file.

### Production / deployment

*Follow this workflow if you're installing `evomachine` on a microscope (see next section for local development).*

From the `evomachine` repo root:
```bash
uv sync --no-sources --no-dev
```
which will create a new `.venv` virtual environment and install all required dependencies.

Depending on the hardware backends in use, you may also need the extra
dependencies, see [Extra dependencies](#extra-dependencies).

### Local development

*Follow this workflow if you're developing new features in this or dependency repos (eg, `sync_board`).*

Start by cloning the sibling repos next to `evomachine` in the same parent folder, and check out the correct respective branches (see [Workspace Structure](#workspace-structure)). 

The expected layout is:
```
workspace/
├── evomachine/      ← this repo
├── AutoStrat/        strategy generation and validation library
├── de-lta-rt/       (dev_main)
├── asitiger/        (master)
└── sync_board/      (Signals)
```

Then, from the `evomachine` repo root:
```bash
uv sync
```
which will create a new `.venv` virtual environment with the local sibling dependencies installed in [editable mode](https://setuptools.pypa.io/en/latest/userguide/development_mode.html) for easier development, as specified in the `[tool.uv.sources]` section of [pyproject.toml](pyproject.toml), as well as all `dev` dependencies.

## Extra dependencies

A couple of runtime dependencies are **not** installed by the plain `uv sync` from the above sections because they're only needed for specific hardware backends:

#### **`pyvcam`**

This is a Photometrics PVCAM Python wrapper, and is only needed for the PVCAM camera backend. It's declared under the optional `pvcam` extra and installed on demand:
```bash
uv sync --extra pvcam   # add to a dev environment
uv sync --no-sources --extra pvcam   # production
```

It builds against the proprietary PVCAM SDK, which must already be installed on the system. See the `pvcam` [repository](https://github.com/Photometrics/PyVCAM) for more information.

#### **`em_dmd_window`**

This is one of our internal libraries used to control the DMD, via a compiled binary, not a Python package. The `EM_DMD_WINDOW` DMD backend launches `../em_dmd_window/Release/evomachine_dmd_window` (expected as a sibling of the repo root) as a subprocess and talks to it over a local socket. This is only required if you use the `EM_DMD_WINDOW` DMD backend.

The compiled binary is committed to the [em_dmd_window](https://github.com/kmpape/em_dmd_window) repo. The easiest way to make it accessible to `evomachine` is by cloning the repo as a sibling of this one:
```bash
git clone --depth 1 git@github.com:kmpape/em_dmd_window.git ../em_dmd_window
```
This places the binary at `<workspace>/em_dmd_window/Release/evomachine_dmd_window`, where the backend eventually looks for it.

## Testing

Run all tests via `pytest`:

```bash
uv run pytest
```

## Running scripts

Anything can be run inside the virtual environment via `uv run ...`, such as:
```bash
uv run python scripts/launch_virtual_gui.py
```

## Release workflow

Let's say you want to make a change to the `sync_board` dependency and have it propagated to `evomachine`. The steps are:

1. Make your changes to `sync_board` and eventually get it merged into the `main` branch (perhaps after making a new branch, PR, and some form of PR review process).
2. Release a new version of `sync_board`: GitHub repo > Releases > Create a new release > Tag `<new_tag>` (eg, `v0.2.0`) > Publish Release.
3. Update the git tag in `evomachine/pyproject.toml` to the new release tag.
4. Run `uv sync --no-sources` to resolve the new dependencies.

## Workspace Structure

This project depends on several sibling repositories:

`evomachine`: 
- main application repository (this one)
- URL: `https://github.com/kmpape/evomachine`  
- Branches: `dev` (in use) and `refactor`

`AutoStrat`:
- domain-independent strategy generation, parsing, validation, and semantic verification library
- URL: `https://github.com/Liam-Metcalf/AutoStrat`
- installed from the sibling checkout in editable mode during EvoMachine development

`asitiger`: 
- ASI Tiger controller package used by Tiger hardware bindings.  
- URL: `https://github.com/kmpape/asitiger` (forked from `https://github.com/herophilus/asitiger`)  
- Branches: 

`sync_board`: 
- SyncBoard controller package used by SyncBoard bindings.  
- URL: `https://github.com/kmpape/sync_board`  
- Branches: `Signals` (in use), `master` (refactor, differences unclear)

`de-lta-rt`:
- DE-LTA real-time segmentation and tracking package.  
- URL: `https://gitlab.com/kmpape/de-lta-rt`  
- Branches: `dev_main`

`em_dmd_window/`: 
- DMD display/window helper used by projection bindings.  
- URL: `https://github.com/kmpape/em_dmd_window`
- Branches: `master`  


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
  strategy execution, and high-level experiment orchestration.
- `strategy_generation/` contains the application-side AutoStrat integration. It provides a
  bridge to AutoStrat's typed expression evaluator, injected command/observation/error interfaces, an
  `AbstractStrategy` wrapper, and a single-worker service. `StrategyGenerationService.build()` is
  explicitly blocking; GUI and event-loop callers must use `submit()` and consume its future
  without blocking their thread. Concrete microscopy command mappings,
  observation calculations, and runtime-error classifications are owned by EvoMachine. Command
  failures stop the remainder of their batch. Before the next strategy step, the host applies the
  domain's authoritative recovery policy using the original exception and command context. A retry re-emits the failed command followed by the
  unexecuted batch tail; `continue` skips the failed command and resumes that tail. Retries are
  bounded by the domain pack, and exhaustion continues, terminates, or aborts according to the
  declared policy. Unexpected
  calculation, observation-contract, or integration failures enter a host-owned fail-safe abort path. Normal strategy
  termination runs finalisation exactly once; abort halts active peripherals and exits without
  running strategy finalisation. The initial microscopy adapter maps `move_fov`, explicit `image`,
  full-field `project`, and `wait` calls onto existing Automaton commands. It exposes lifecycle and
  focus outcomes plus latest-image mean intensity, percentile contrast, saturation fraction, and
  variance-of-Laplacian focus score as strategy observations.
- `domain_packs/` contains EvoMachine-owned strategy declarations and prompting material. The
  `microscopy/` pack is loaded by the separate strategy-generation library during integration.
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

## Running the GUI

### Hardware GUI on the microscope computer

Start Micro-Manager with the microscope configuration loaded. Then run:

```bash
cd /home/hslab/workspace_python/evomachine_refactor/evomachine
.venv/bin/python scripts/launch_hardware_gui.py
```

For an IDE Run button:

- Working directory: `/home/hslab/workspace_python/evomachine_refactor/evomachine`
- Interpreter: `.venv/bin/python`
- Script: `scripts/launch_hardware_gui.py`
- Run it as a Python file, not as a module.

The hardware runtime uses the Micro-Manager camera, SyncBoard LEDs, ASI Tiger
stage/filter/autofocus and overhead LED, KWR103 overhead light, and EM DMD
window. Serial ports are detected from their USB hardware IDs.

The default RAMM optical setup uses a Nikon Plan Apo lambda D 40x/0.95
objective (MRD70470) and a Teledyne Kinetix 3200 x 3200 sensor with 6.5 um
pixels. This gives 0.1625 um per sample pixel and a 520 um square camera field
of view. The separate 60x/1.4 oil-objective preset remains available for other
setups.

The hardware GUI zeroes the ASI Tiger stage at its startup position. Relative
and field-of-view movements then use that Tiger coordinate system and are
checked against the configured software limits. The deployment defaults can be
overridden before launch with:

```bash
export EVOMACHINE_GUI_STAGE_MIN_X_UM=-8000
export EVOMACHINE_GUI_STAGE_MAX_X_UM=8000
export EVOMACHINE_GUI_STAGE_MIN_Y_UM=-19000
export EVOMACHINE_GUI_STAGE_MAX_Y_UM=19000
export EVOMACHINE_GUI_STAGE_MIN_Z_UM=-1000
export EVOMACHINE_GUI_STAGE_MAX_Z_UM=1000
```

These values are in micrometres relative to the startup zero and must be set to
the microscope's confirmed safe travel region before hardware use.

#### Acquisition output folder

Manual images, z-stacks, and strategy images use one local output folder. In
the GUI, select **Choose Output Folder** in the separate **Output Folder**
panel. The same panel is shown in the Strategy tab so the destination is
explicit before starting a strategy. Enable **Save** in Acquisition
Configuration when TIFF output is required. Strategies request that an image
is saved but do not choose a path; they use this shared folder through
`FrameAcquisitionManager`.

The default hardware folder is `images/` and can be changed at startup with
`EVOMACHINE_GUI_OUTPUT_DIR`. The **Load Saved Images** panel has its own
**Choose Loading Folder** control. Browsing a different folder does not change
where new images are saved. Select **Load Selected** to display a listed TIFF.

The `images/` directory is intentionally excluded from Git because microscope
images are generally too large for source control. Notebook `.pkl` image
stacks may be organised locally but should remain uncommitted; the current example stacks are
approximately 801 MiB each.

### Virtual GUI

From the `evomachine` repository root, launch the GUI with virtual peripherals:

```bash
uv run python scripts/launch_virtual_gui.py --port 0
```

To smoke-test the automaton/socket startup without opening Napari:

```bash
uv run python scripts/launch_virtual_gui.py --port 0 --no-napari
```

### AutoStrat schema 8 integration

For a lean prompt → DSL + Python-wrapper test, open
[`notebooks/quick_autostrat.ipynb`](notebooks/quick_autostrat.ipynb) with the EvoMachine `.venv`
kernel. Run setup once, edit the prompt, and run Generate. It uses the same Robin endpoint as the
original notebook and reuses `OPENAI_API_KEY` (or asks privately). Diagnostics are collapsed below
the two outputs. Nothing executes on the microscope. The original
[`explore_autostrat_pipeline.ipynb`](notebooks/explore_autostrat_pipeline.ipynb) remains available
for detailed virtual-hardware testing. The Python output is the actual EvoMachine execution
wrapper, not an independent DSL-to-Python translation.

This branch requires the matching AutoStrat `typed-strategy-expressions` branch. The existing
`uv` editable source points to `../AutoStrat`; switch that checkout to the matching branch before
syncing. The previously released v0.1.1 API is incompatible with these validated programs.

The microscopy pack uses boolean, integer, number, and enum. Exposure numbers are milliseconds;
projection and wait durations and elapsed_time are seconds, as documented in the pack. No numeric
literal unit suffixes are accepted. Each lifecycle section uses one observation snapshot.

Use one-line immutable declarations such as `const number pause = max(1, min(10, observation.elapsed_time))`,
then `wait(duration=pause)`. Inline command calculations and direct observation arguments are
invalid. Constants can also be used in conditions. Only terminate and abort are DSL control actions.
Recovery belongs in runtime_errors.yaml using action, max_retries, and exhausted_action.

The evaluator resolves and validates a complete batch before adapters build commands. Evaluation
failures propagate directly to Automaton's fail-safe boundary. They never enter command recovery.
Retries reuse exact resolved values and the unexecuted batch tail. Failures during finalisation
retain the existing immediate fail-safe behaviour. New experiment-specific observations, ROI
selection, DeLTA integration, and scheduling policies are not introduced by this capability update.
