# Architecture and code structure

[Documentation](../README.md) › [Developer guide](README.md) › Architecture and code structure

## Repository boundary

The main package is `evomachine/`. Sibling repositories provide AutoStrat,
DeLTA-RT, ASI Tiger, SyncBoard, and the optional DMD window. EvoMachine owns the
microscope-specific orchestration and adapters; domain-independent behavior stays
in the relevant sibling library.

Important repository directories are:

| Path | Responsibility |
| --- | --- |
| `evomachine/peripherals/` | Binding-neutral device APIs, configs, common state, validation, and factories. |
| `evomachine/bindings/` | Concrete virtual, hardware, and external-library implementations. |
| `evomachine/gui/` | Qt/Napari UI, controller, protocol, socket transport, facade, and request handlers. |
| `evomachine/strategy_generation/` | EvoMachine AutoStrat adapter, providers, preview, storage, and service. |
| `evomachine/domain_packs/` | Microscopy commands, observations, errors, policies, and generation guidance. |
| `strategies/` | Discoverable user Python strategies. |
| `scripts/` | Launchers, calibration utilities, and focused workflows. |
| `tests/` | Unit, integration, binding-contract, runtime, and opt-in Qt tests. |

Files and directories named `*_bkp` or `trash` are not current architecture and
must not be used as extension examples.

## Runtime ownership

```text
Napari panel
  -> EvoMachineGuiController
  -> GuiSocketClient / protocol command
  -> GuiRpcServer
  -> AutomatonGuiFacade + request handler
  -> Automaton or owned manager
  -> binding-neutral peripheral
  -> concrete binding / external driver
```

The return path carries an authoritative response through the controller signal
to every interested panel. Panels should not assume a submitted value became
active before receiving that response.

Strategy execution enters below the GUI protocol:

```text
AbstractStrategy or AutoStratStrategy
  -> AutomatonCommand batch
  -> Automaton command dispatch
  -> manager/peripheral operation
  -> completed command or CommandExecutionError
  -> strategy callback/recovery
```

## Main components

### Automaton

`Automaton` owns application lifecycle, installed strategy, FoVs, processing
state, device initialisation, command dispatch, stop/shutdown events, command
failure history, and coordination between managers. It is the safety and
lifecycle boundary, not a hardware driver.

### Managers

- `FrameAcquisitionManager` sequences LED/filter/DMD/camera actions, constructs
  `Frame` results, saves through `FileManager`, and implements Z-stacks.
- `FocusNavigator` owns FoV coordinates, current-FoV state, focus mode,
  autofocus locking, recovery, per-FoV focus state, and skip outcomes.
- `ProjectionManager` coordinates calibrated DMD projection behavior.
- `DeltaProcessor` owns processing calls and updates typed `MicroscopyState`.
- `FileManager` owns filenames, output directories, TIFF image/metadata saving,
  and naturally sorted file listing.

Managers depend on peripherals; peripherals do not depend on the GUI or
strategies.

### Peripherals and controllers

A `PeripheralController` owns a connection and its lifecycle. A peripheral owns
logical device behavior and may share a controller with other peripherals.
Binding-neutral base classes contain common readiness checks, state bookkeeping,
validation, and cleanup. Concrete bindings implement low-level operations.

Factories choose a concrete binding from a typed config. See [Bindings](bindings.md).

### Frames and metadata

`FrameMetaData` describes requested and observed properties of one plane.
`FrameMetaDataFactory` assigns process-local frame IDs. Acquisition fills execution
time and coordinate. `Frame` aligns a list of metadata records with the leading
dimension of one image array and with `saved_paths`.

`additional_metadata` must be JSON serialisable. It is stored in TIFF metadata but
is intentionally unavailable to filename formatting.

### Strategies

`AbstractStrategy` validates lifecycle outputs and provides a `CommandFactory`.
The Automaton injects FoVs, ROIs, processing state, DMD access, and processors.
Python strategy discovery loads top-level files from `strategies/`.

`AutoStratStrategy` is an adapter around the external evaluator. Microscopy
command construction, observations, collections, and error classification remain
EvoMachine responsibilities. See [AutoStrat](autostrat.md).

## Process model

The launchers create an Automaton child process with a `GuiRpcServer`, wait for
its bound address, then run Napari with a socket client. Read-only responsive
commands can be serviced while the Automaton is otherwise busy. Hardware and
shared configuration mutations are guarded against incompatible active work.

Image previews use the configured `auto`, `temp_tiff`, `socket_tiff`, or `raw`
transport. Control responses remain protocol objects rather than direct Qt calls.

## Dependency direction

Keep dependencies directed inward:

```text
GUI / strategies / launchers
        -> Automaton and managers
        -> peripheral interfaces and typed configs
        -> concrete bindings and external drivers
```

Do not import GUI code into core managers or peripherals, and do not put
microscopy concepts into domain-independent sibling libraries. Use local imports
only where they intentionally break an otherwise unavoidable optional/circular
dependency.

---

← [Developer guide](README.md) | [Documentation home](../README.md) | [Configuration](configuration.md) →
