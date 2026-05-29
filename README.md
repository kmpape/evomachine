# Evomachine

Evomachine is the main microscope automation application in this workspace. It
coordinates hardware peripherals, image acquisition, focus navigation,
projection, strategies, and real-time processing workflows.

## Installation

Use `mamba` to create the conda environment from the pinned environment file:

```bash
cd /home/lady5906/workspace_python/evomachine/evomachine
mamba env create -f environment.yml
mamba activate delta_evomachine
```

To update an existing environment after `environment.yml` changes:

```bash
cd /home/lady5906/workspace_python/evomachine/evomachine
mamba env update -f environment.yml --prune
```

The environment name is defined in `environment.yml` as `delta_evomachine`.

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

Other useful top-level folders in the main repository:

- `tests/`: pytest coverage for the package.
- `scripts/`: runnable hardware and workflow scripts.
- `notebooks/`: interactive notebooks for smoke tests and exploratory work.
- `calibrations/`, `data/`, `delta_models/`, `images/`, and `strategies/`:
  runtime inputs, outputs, models, calibration data, and strategy files.  
  Note: some folders are only created when starting the software.
