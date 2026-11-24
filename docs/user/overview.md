# System overview

[Documentation](../README.md) › [User guide](README.md) › System overview

EvoMachine coordinates microscope peripherals through one `Automaton`. A user
operation originates in the GUI or a strategy, becomes a validated operation or
command, and is executed through a manager or peripheral.

```text
GUI action ──> controller/RPC ──> Automaton ──> acquisition/focus/projection manager
                                           └─> peripheral ──> hardware binding

Python or AutoStrat strategy ──> AutomatonCommand batch ──> same managers/peripherals
```

Virtual mode replaces hardware bindings, not the orchestration layers. Manual
acquisitions and strategies therefore exercise the same command, file-management,
and GUI paths in virtual and hardware modes.

## Main concepts

- A **peripheral controller** owns a shared connection, such as an ASI Tiger or
  SyncBoard serial connection. Several peripherals can use one controller.
- A **peripheral** represents one logical camera, stage, LED source, filter wheel,
  autofocus, photodiode, or DMD.
- A **manager** combines devices into a task. `FrameAcquisitionManager` coordinates
  illumination, filters, DMD state, camera capture, stage information, and saving.
- A **FoV** is a field of view with an integer ID and stage coordinate. Selecting
  an FoV and physically moving to it are distinct operations.
- A **frame** contains a `(planes, height, width)` array and aligned
  `FrameMetaData` records.
- A **strategy** supplies command batches. Python strategies implement
  `AbstractStrategy`; AutoStrat strategies execute validated DSL source.

## Experiment lifecycle

The normal order is:

1. Start the virtual or hardware application.
2. Initialise devices.
3. Configure acquisition and output.
4. Register FoVs and choose focus behavior.
5. Test movement, focus, illumination, and acquisition manually.
6. Select or install a strategy.
7. Initialise the strategy with the registered FoVs.
8. Start it and monitor images, status, and logs.
9. Stop the strategy before changing incompatible configuration.
10. Shut down so peripherals are stopped and finalised.

Device initialisation, FoV initialisation, strategy installation, and strategy
start are separate actions. Selecting or generating a strategy does not execute
hardware.

## Runtime data

`evomachine.settings` resolves one data directory at startup. The machine file
`/etc/evomachine/config.toml` is read first and the user file
`~/.config/evomachine/config.toml` second. The user value wins.

```text
data_dir/
├── images/
├── logs/
└── strategies/autostrat/
```

The GUI can change the acquisition output folder for the current run. The
saved-image loader has a separate input folder. AutoStrat credentials and virtual
camera source choices are session-only.

## Process and hardware boundaries

The Napari GUI and Automaton normally run in separate processes over a local
socket. Closing the GUI requests an orderly backend shutdown. Long-running
hardware work stays in the Automaton process rather than blocking Qt.

The default hardware runtime is a deployment preset, not a portable hardware
description. Review the
[hardware runtime settings](../development/configuration.md#hardware-gui-runtime-settings)
before using it on another microscope.

---

← [User guide](README.md) | [Documentation home](../README.md) | [Using the GUI](gui.md) →
