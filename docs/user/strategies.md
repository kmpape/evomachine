# Writing and running strategies

[Documentation](../README.md) › [User guide](README.md) › Writing and running strategies

EvoMachine supports Python strategies and AutoStrat strategies. Both ultimately
emit `AutomatonCommand` objects and use the same Automaton, managers, peripherals,
stop events, output configuration, and logs.

## Python strategies

Place a `.py` file directly in `strategies/`. Discovery ignores `__init__.py`,
imports each top-level file, and lists concrete `AbstractStrategy` subclasses
defined by that file. Imported classes and abstract classes are not listed.

### Minimal repeating strategy

```python
from evomachine.commands import AutomatonCommand
from evomachine.frame import FrameMetaDataFactory
from evomachine.strategy import AbstractStrategy
from evomachine.types import AutomatonCommandType, LEDType


class ExampleImagingStrategy(AbstractStrategy):
    """Acquire one saved image at every registered FoV on each cycle."""

    def register_automaton_commands(self) -> set[AutomatonCommandType]:
        """Declare every command type this strategy can emit."""
        return {
            AutomatonCommandType.MOVE,
            AutomatonCommandType.IMAGE,
            AutomatonCommandType.WAIT,
        }

    def _cycle(self) -> list[AutomatonCommand]:
        """Build one imaging cycle for the currently registered FoVs."""
        commands: list[AutomatonCommand] = []
        for fov_id in self.fovs:
            commands.append(self.command_factory.command_move(fov_id=fov_id))
            metadata = FrameMetaDataFactory.default(
                leds={LEDType.LED_450_NM: 10},
                exposure=100,
                fov_id=fov_id,
            )
            commands.append(
                self.command_factory.command_image(
                    frame_metadata=metadata,
                    segment=False,
                    save=True,
                )
            )
        commands.append(
            self.command_factory.command_wait(duration=60, set_live_mode=False)
        )
        return commands

    def _initialise(self) -> list[AutomatonCommand]:
        """Return commands for the first lifecycle batch."""
        return self._cycle()

    def _callback(
        self,
        fov_id: int,
        data: list[AutomatonCommand],
        errors: list[Exception],
    ) -> list[AutomatonCommand]:
        """Handle the completed batch and return the next one."""
        if errors:
            raise errors[0]
        return self._cycle()

    def finalise(self) -> list[AutomatonCommand]:
        """Return cleanup commands for normal termination."""
        return []
```

The GUI constructs the class with the active `ImageProcessorConfig`. The base
class creates `self.command_factory`; the Automaton later injects registered
FoVs, ROIs, processing state, processors, and optional DMD access through
`initialise()`.

### Lifecycle contract

- `register_automaton_commands()` must contain every type the strategy might
  emit in initialisation, callbacks, or finalisation. Missing declarations fail
  before unsafe execution; declare only operations the runtime supports.
- `_initialise()` runs after FoVs and runtime state are injected and returns the
  first batch.
- `_callback(fov_id, data, errors)` receives the completed commands and failures
  and returns the next batch. The base class increments `callback_counter` after
  a valid callback result.
- `finalise()` runs once after normal completion or termination. Abort and
  process failure do not guarantee normal strategy finalisation.
- Every lifecycle method must return `list[AutomatonCommand]`, including an empty
  list when no operation is needed.

Do not call hardware directly from a strategy. Construct commands so the
Automaton can validate requirements, stop safely, record context, and update
observations.

### Command categories

`CommandFactory` in `evomachine/commands.py` is authoritative. Important groups
include:

- movement and per-FoV focus updates;
- image acquisition and Z-stacks;
- interruptible waits and live-mode changes;
- ROI detection, segmentation, and processing requests attached to images;
- full-field and selected-ROI DMD projection;
- state saving and device-specific operations where supported.

An image command can contain one `FrameMetaData` or a list for multiple channels.
`save=True` uses the shared `FileManager`; a strategy does not choose its own
path. `segment=True` and `detect_rois=True` require compatible processing config
and segmentation channels.

A Z-stack uses one metadata template across its planes. `relative_range` is a
positive half-range and `step_size` is a positive step no larger than the range.
Capture order is centre, positive offsets, then negative offsets. Only complete
steps inside the range are used. With `save=True` the stack is saved as one
multi-page TIFF per channel, planes sorted by Z, and every plane's
`saved_paths` entry points to that file.

Projection changes physical illumination. Validate DMD calibration, selection,
brightness, duration, and occupied FoV manually. Never encode automatic retry for
an exposure whose completion is uncertain.

### Results and errors

Completed command objects contain `command_data` and execution time. Image and
Z-stack results contain frame arrays, metadata, and saved paths through the
existing acquisition result shape. Inspect the command type rather than assuming
all entries contain images.

If a command fails, the rest of its batch stops and `errors` contains contextual
exceptions. A Python strategy owns its callback response: it may raise, return a
safe next batch, or arrange normal termination according to its design. Do not
silently ignore hardware errors.

## FoVs, ROIs, and identity

`self.fovs` maps stable integer FoV IDs to stage `Coordinate` objects.
`self.region_of_interests` maps FoVs to stable ROI IDs after detection. The
currently selected loop item, requested move target, and physical stage location
are distinct until movement succeeds.

ROI redetection replaces ROI identities and resets associated processing,
tracking, treatment history, and pending projection targets. Do not redetect a
collection while iterating it.

## Focus behavior

Focus is configured at three levels:

1. `FocusNavigatorConfig` sets the run-wide mode, autofocus use, recovery policy,
   waits, retry limit, and default FoV policy.
2. `FovConfig` overrides arrival/departure locks, software focus, and focus
   configuration for one FoV.
3. `SoftwareFocusConfig` controls shared focus frames, acquisition, typed scoring,
   and crop boxes. Its typed selection pair chooses either maximum-score scanning
   or `ReferenceFocusSelectionConfig` reference-profile matching.

A strategy may populate `self.fov_configs` during `_initialise()` because
`self.fovs` is available then:

```python
from evomachine.navigation import FovConfig

self.fov_configs = {
    fov_id: FovConfig(
        lock_autofocus_on_fov=True,
        run_software_focus_on_arrival=False,
    )
    for fov_id in self.fovs
}
```

GUI-supplied per-FoV configs take precedence over strategy-provided configs for
the same IDs. Reference-profile mode requires `use_autofocus=False`. A reference
must be calibrated, visually reviewed, and accepted before it can correct focus.
See the [configuration reference](../development/configuration.md#focus-and-navigation).

## AutoStrat strategies

AutoStrat is an optional typed DSL and generation system. In the Strategy panel:

1. Configure a provider/model at startup. An API key is needed only for AI
   generation, not for editing or running saved DSL.
2. Generate from a prompt or edit DSL directly.
3. Select **Validate**. Validation checks types, commands, argument ranges,
   observation scope, and control flow; it does not touch hardware.
4. Select **Set Strategy** to install the accepted source.
5. Initialise FoVs, initialise the strategy, then start it.

GUI **Save** and **Save As** write UTF-8 `.strat` files under
`data_dir/strategies/autostrat/`. Credentials remain in backend process memory and
are omitted from status responses.

### Sections and state

An AutoStrat program has `initialise`, `step`, and `finalise` sections.
Top-level initialise variables persist. Other variables are section-local and
must be assigned on every path before use. The evaluator pauses at commands,
refreshes observations after successful completion, and resumes from the same
logical position.

Collections such as FoVs and ROIs snapshot their IDs on loop entry. Loop selection
does not move hardware: use explicit movement and image commands. Guard optional
measurements with their validity observations.

### Recovery

Declared runtime errors use domain-owned policies: retry, continue, terminate, or
abort. Retry preserves the failed command's resolved arguments and loop position;
it does not replay already completed commands. Unexpected evaluator or integration
errors abort through the host fail-safe boundary.

See [Extending the AutoStrat integration](../development/autostrat.md) for the
complete developer model and domain-pack extension workflow.

## Testing a strategy

1. Unit-test command construction with fake configuration.
2. Run it in the virtual GUI with random images.
3. Use a folder source for deterministic recorded images.
4. Confirm output names, saved metadata, stop behavior, and failure logs.
5. If processing is used, test with representative images and matching models.
6. Perform conservative manual hardware checks before unattended execution.

Virtual success does not validate mechanics, optics, processing quality, or
experimental safety.

---

← [Using the GUI](gui.md) | [User guide](README.md) | [Virtual mode](virtual-mode.md) →
