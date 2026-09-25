# Evomachine

Evomachine is the main microscope automation application in this workspace. It
coordinates hardware peripherals, image acquisition, focus navigation,
projection, strategies, and real-time processing workflows.

## Developer guides

The GUI application log mirrors the backend's configured EvoMachine and hardware
binding log records, including DEBUG device activity and strategy exception
tracebacks, using the terminal formatter. It retains a bounded recent history
(5,000 records on the backend; 5,000 displayed text blocks); rotating log files
remain the longer-term record. This is an application log viewer, not a capture of
arbitrary stdout/stderr from Napari, native libraries or unrelated processes.

- [Extending AutoStrat integration](docs/development/autostrat.md): commands,
  observations, collections, runtime errors, and domain-pack examples.
- [Extending the GUI](docs/development/gui.md): panels, RPC operations, background
  work, lifecycle rules, and tests.

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

### AutoStrat schema 9 integration

In the GUI's Strategy Setup panel, choose **AutoStrat**, enter a prompt and click
**Generate strategy**. Generation and semantic verification run on a background worker,
independently of hardware operations. Review the read-only DSL (or use **Expand DSL**) and
enable **Show generation diagnostics** for the same candidate/revision/error report as
the quick notebook. Internal retry counts not exposed by AutoStrat are explicitly marked
unavailable, not reported as zero.

Then use the shared **Set Strategy**, FoV **Initialise**, and **Start Strategy** buttons;
**Stop Strategy** is shared with fixed strategies too. If FoVs are already initialised,
Set Strategy prepares the generated strategy immediately. Generation alone never installs
or runs a strategy. Editing the prompt or generating again invalidates the preview's Set
eligibility until an accepted result is available. Generated acquisitions save to the
configured Output Folder. Detection/segmentation still require configured DeLTA models
and a suitable segmentation channel; generation itself never loads those models.

At GUI startup, an optional **masked API-key dialog** enables AutoStrat for that session.
Leave it blank and press Enter, or Cancel, to disable AutoStrat; fixed strategies and hardware
controls remain available. Skipping also disables use of any inherited environment key.
The key is held in backend process memory only, not saved to a file or echoed in responses.
Restart the GUI to enter a key after skipping. `OPENAI_BASE_URL` and `AUTOSTRAT_MODEL_ID`
remain optional backend environment settings; defaults match the notebook's Robin endpoint
and `qwen3.6` model. Use the default local GUI connection when supplying credentials.

For a lean prompt → DSL test, open
[`notebooks/quick_autostrat.ipynb`](notebooks/quick_autostrat.ipynb) with the EvoMachine
`.venv` kernel. Run setup once, edit the prompt, and run Generate. It reuses the existing
Robin endpoint and privately supplied API key. Optional diagnostics appear below the DSL.
Nothing executes on the microscope.

Use the matching `collection-loops-state` branch in both repositories. The editable
`../AutoStrat` source supplies the new runtime during development. Production and CI
installs pin AutoStrat commit `c0693bd59155fc4c49f7dfa9b04754938d3bed07`, which supplies
the matching schema-9 API.

Schema 9 replaces `const TYPE name = expression` with `name = expression` and
`observation.NAME` with `observations.NAME`. Revalidate old DSL/programs.
Types are inferred and checked; a variable cannot change its inferred type.
Use `total = 0.0` for a number accumulator. Command arguments remain literals or variables.

Top-level initialise variables persist throughout a run. Other variables are section-local,
reset on each invocation; assignments within branches and loops update the same scope.
A variable must be assigned on every possible path before it is read.
Put persistent counters/defaults before hardware commands if finalisation needs them
after a hardware failure.

Collections are declared in the domain pack. Inject `MicroscopyCollectionProvider()`
alongside the existing command, observation and runtime-error providers:

```text
initialise
    rounds = 0
step
    loop fovs:
        move_fov(target=current_fov)
        image(detect_rois=false, segment=false, exposure=100, led=450nm, led_brightness=10, filter=465nm)
        count = 0
        loop rois:
            count = count + 1
        pause = max(1, min(10, count))
        wait(duration=pause)
    rounds = rounds + 1
    if rounds >= 2:
        terminate
finalise
    move_fov(target=first_fov)
```

Each loop snapshots the registered IDs on entry and starts from the beginning. Empty
collections do nothing. Entering a loop only selects context: explicit commands move or image.
`observations.selected_fov_id` is scoped to the FOV loop; `selected_roi_id` to its ROI loop.
`current_fov_id` remains the actual physical position. Existing image metrics describe the
latest acquisition. Processed images additionally publish scoped mother-cell measurements for each
stable detected trench. See the DeLTA integration below.

Each command is resolved and range-checked just before it is emitted. Successful completion
refreshes observations before execution resumes. Step count measures completed DSL steps,
not hardware callbacks. Retries preserve the failed command's resolved arguments, variables
and loop position; they do not replay completed commands. Runtime policies are handled before
observation refresh or resumed DSL evaluation. Continue invalidates cached global measurements.
Calculation/adapter failures go directly to Automaton's fail-safe boundary; previous physical
effects are not rolled back. Terminate exits all loops and runs finalisation sequentially;
abort skips finalisation. Finalisation failures remain fail-safe.

The generic host interface is `CollectionProvider.items(name, context)` plus
`observe(context)`; providers supply unique stable integer/string IDs and current scoped
values. Missing values fail when read, and invalid supplied values fail on refresh. There are
no DSL lists or implicit hardware actions. Iteration/statement budgets bound each section.

### DeLTA integration (microscopy pack 0.7.0)

AutoStrat itself is unchanged. The microscopy pack adds required `image(detect_rois=..., segment=..., ...)`,
`clear_projection_targets()`, `select_roi()` and `project_selected(...)` for combined ROI projection.
Revalidate older generated strategies after adding `detect_rois=false, segment=false` to acquisition-only images.
Waits remain bounded by the domain pack and interruptible by stop/shutdown events.

For processed imaging, construct the strategy and Automaton with the same explicit configuration:

```python
from evomachine.image_processing_config import ImageProcessorConfigFactory
from evomachine.delta_processing import DeltaProcessor
from evomachine.types import LEDType

cfg = ImageProcessorConfigFactory.default_config(
    channels=[LEDType.LED_450_NM],
    channels_seg=[LEDType.LED_450_NM],
).updated(preproc_enabled=True, roi_enabled=True, seg_enabled=True,
          track_enabled=True, lineage_enabled=True)

# Optional application-supplied selection: IDs refer to the final detected trench order.
processor = DeltaProcessor(cfg, selected_targets={0: {1, 3}, 1: {2}})
# Omit selected_targets to permit all detected trenches.
# Pass cfg_processor=cfg and delta_processor=processor to Automaton.
# Use cfg=cfg, MicroscopyCollectionProvider(), MicroscopyObservationProvider()
# and MicroscopyRuntimeErrorProvider() when constructing AutoStratStrategy.
```

Set `detect_rois=true` to detect trenches and `segment=true` to segment bacteria and update
mother-cell tracking. Detection alone does not load the segmentation model. Segmentation alone
requires existing ROIs. Repeat measurements with `detect_rois=false, segment=true`.
Redetection replaces the FOV's ROIs and resets tracking, measurements, treatment history and
pending projection targets. External target IDs are cleared rather than reassigned to new trenches;
the default all-targets policy continues to permit all detected trenches. Models and supported input channels must be
configured locally; no model loading, imaging or projection happens in the quick notebook.
The current integration accepts one segmentation-channel plane per processed image.

The microscope owns typed per-FOV/per-trench records. The DSL gets `measurement_valid`,
`length`, `measurement_time`, valid previous measurements, and DeLTA's `growth_rate` with
a separate `growth_rate_valid` flag. Invalid/missing values are omitted, not replaced by zero.
Always guard reads with validity observations. No mother found and zero detected trenches are
valid outcomes; model/processing exceptions invalidate the FOV and abort without reprocessing
partially mutated lineage. A failed, skipped or unprocessed acquisition also invalidates current
biological measurements. A new strategy run clears processing, treatments and pending projection targets.

**Growth-rate semantics:** DeLTA's built-in rate is inverse frames, not inverse seconds; its
smoothing assumes equal spacing. Mother-only tracking does not provide daughter-aware division
correction. A strategy can compute a fractional elongation-per-second proxy
from successive lengths and actual times in the DSL. This is not a validated biological
growth estimator and can be affected by division, tracking loss or segmentation noise.

**Projection safety:** `project_selected` requires valid processing and the correct physical
FOV for a nonempty selection. It illuminates all buffered trench boxes in one pattern and one exposure,
not segmented mother-cell outlines. Only confirmed completion increments treatment_count and records
the same last_treatment_time for every included ROI. The shared exposure duration is at most 100 seconds. An interrupted or uncertain exposure
aborts and is never automatically retried; illumination is disabled and the DMD cleared on exit.
This cannot undo a partial delivered dose. The existing full-field `project` retains its old policy.

Post-treatment comparison counters count only valid consecutive processed measurements with both
acquisition times after the last treatment. Treatment resets these counters. They are factual
length-comparison summaries, not a survival or motion decision. The DSL chooses the evidence
threshold, reassessment interval, exposure limit and experiment stopping condition.

Projection selection is per FOV. Clear the buffer, iterate the ROIs and add qualifying targets,
then expose the entire set once after the ROI loop:

```text
clear_projection_targets()
loop rois:
    if observations.target_selected:
        if observations.measurement_valid:
            select_roi()
project_selected(illumination_led=385nm, illumination_brightness=10, duration=0.5)
```

This fragment belongs inside a FOV loop after movement and processed imaging. Its illumination
values are illustrative inputs, not recommended settings. Adding an ROI does not expose it.
Duplicate additions are harmless; an empty projection does nothing. Successful projection clears
the buffer, as does a new acquisition or a new strategy run. Each FOV has its own buffer.
The application-supplied authorised target set remains separate from this temporary selection.

For continuous acquisition, let each step traverse the FOVs, use explicit waits and check the
requested experiment stopping time. There is no per-FOV scheduling API. All timestamps share
one monotonic experiment clock; per-ROI treatment times remain available for DSL calculations.
Experimental thresholds, timings and treatment policies belong in the user's strategy request,
not the domain's general semantic guidance.

**Geometry boundary:** this first version supports horizontal mother-machine layouts using
DeLTA's existing grouping/dead-end conventions. Automatic rotation and drift correction are
disabled to keep ROI boxes in the camera coordinate system used by the DMD transform. Confirm
trench ordering, mother-cell ends, camera/DMD calibration and stage registration on representative
data before enabling UV. Drift compensation and arbitrary trench orientations are not implemented.

The pack includes a generic combined-projection example with explicit user-supplied parameters.
Tests include real PositionRT/lineage code driven by synthetic segmentation outputs, fake-hardware
execution, timing and exposure failures. No real microscope or trained-model performance has been
validated by these tests.
