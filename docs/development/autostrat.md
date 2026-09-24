# Extending the AutoStrat integration

This guide covers AutoStrat **inside EvoMachine**, not developing a new language
feature in the sibling AutoStrat repository. See also [GUI extensions](gui.md).

## Who owns what?

| Layer | Responsibility | Starting point |
| --- | --- | --- |
| AutoStrat library | Parsing, expression evaluation, validation, generation and collection iteration | Sibling `AutoStrat` repository |
| Microscopy domain pack | Available commands, typed arguments, observations, collections and error policies | `evomachine/domain_packs/microscopy/` |
| EvoMachine adapter/providers | Translate commands and expose application state | `evomachine/strategy_generation/microscopy.py` |
| Strategy runtime | Resume evaluation after command completion; handle lifecycle and errors | `evomachine/strategy_generation/strategy.py` |
| Hardware/processing | Execute commands and produce measurements | `evomachine/automaton.py`, `evomachine/delta_processing.py` |

Ordinary microscopy extensions should not require changing AutoStrat's parser or
adding microscopy-specific concepts to that library. A YAML declaration alone
does not implement an operation or calculate a measurement.

## Execution model

```text
prompt + domain pack -> generation and validation -> VerifiedStrategy
 -> AutoStratStrategy -> evaluated command -> MicroscopyCommandAdapter
 -> AutomatonCommand -> Automaton execution -> callback/observations -> resume
```

The evaluator pauses at commands so later statements can use updated observations.
`observations.step_count` counts completed `step` sections, not commands or ROIs.
Initialise variables persist; step-local values are recreated on each step. Keep
strategy calculations in the language where possible; retain acquisition results,
tracking state and treatment history in the application.

The GUI uses `gui/strategy_generation.py` and `GenerationPreview` to generate in a
background worker, then installs an `AutoStratStrategy` only when Set Strategy is
pressed. `StrategyGenerationService` is a separate programmatic entry point; its
providers are shared and it is intended for one active strategy at a time. Do not
assume it is the GUI's generation path.

## Add an observation

1. Decide whether it describes the latest/global state, a selected FOV, or a
   selected ROI. Use the existing `MicroscopyState` dataclasses for persistent
   processing state rather than a parallel cache in the GUI.
2. Declare its type, meaning, units and bounds in `observations.yaml`.
   Global values go under `observations`; scoped ones under
   `collections.<collection>.observations`.
3. Populate global values in `MicroscopyObservationProvider.observe` (or its
   acquisition/movement helpers). Populate selected-record values in
   `MicroscopyCollectionProvider.observe(context)`.
4. Define availability and invalidation: before the first image, after acquisition
   failure, after ROI redetection and after a new strategy is installed. Use a
   validity observation when a measurement may be unavailable; do not invent zero
   measurements or return NaN to mean missing.
5. Add guidance/examples only where needed to explain correct use; test both the
   supplied value and the missing/invalid case.

For example, a **proposed** per-trench area measurement would add these entries
under `collections.rois.observations` (these are not existing observations):

```yaml
area_valid:
  description: Whether the selected trench's latest mother-cell area is available.
  type: boolean
area:
  description: Area of the selected trench's mother cell. Unit is square pixels.
  type: number
  minimum_inclusive: 0
```

The provider must return `area_valid` and supply `area` only when available. The
measurement producer must calculate/store it from the segmentation result. The
strategy can then guard access with `if observations.area_valid:`.

Types currently include `boolean`, `integer`, `number` and `enum`; use the domain
schema, not Python's incidental acceptance of booleans as integers. Units belong
in descriptions and adapter conversions, not new types such as `seconds`.

## Add a command or argument

Use `wait` as the small existing example, and `image` for a command with hardware
effects and processing flags.

1. Add the command/argument to `commands.yaml`: description, type, enum choices or
   numeric bounds, units, and possible `runtime_errors`. Describe side effects and
   prerequisites, including required collection scope.
2. Update `_COMMAND_TYPES` and dispatch/building in `MicroscopyCommandAdapter`.
   Convert validated arguments to application values. Validate selected versus
   physically occupied FOV when the operation requires them to match.
3. Reuse a `CommandFactory` operation when possible. For genuinely new execution
   behavior, add an `AutomatonCommandType` in `types.py`, a factory method in
   `commands.py`, and execution in `Automaton._execute_strategy_batch` or a helper.
   Register the type in any fixed strategy that emits it.
4. Return command results needed by providers; do not read hardware from the LLM
   generator or execute hardware in the adapter's `build` method.
5. Classify failures and declare their policies (see below). Keep argument
   validation at the execution boundary as well as generation-time validation.
6. Update semantic guidance and relevant few-shot examples. If an argument becomes
   required, migrate every example and caller; accepted older source may no longer
   validate. Review the domain version in `domain.yaml`; `schema_version` describes
   the supported schema, not a counter to bump for every new command.

`image(detect_rois=..., segment=...)` deliberately separates trench detection from
cell segmentation. Redetection replaces trench identities and resets associated
tracking/treatment state. Do not redetect while iterating those ROIs. Image saving
is an execution setting: GUI-generated strategies currently use
`MicroscopyCommandAdapter(..., save_images=True)`.

## Collections and observation scope

Declare collections in `observations.yaml`, including `parent` for nested ones.
Implement `MicroscopyCollectionProvider.items(collection, context)` to return
stable IDs in the intended order, and `observe(context)` for scoped measurements.
Bind the backing records through `bind`/`bind_processing_state`.

```text
loop fovs:
    move_fov(target=current_fov)
    loop rois:
        if observations.measurement_valid:
            length = observations.length
```

Iteration selects context; it **does not move hardware**. `selected_fov_id` is the
loop selection; `current_fov_id` describes the occupied FOV. Keep those distinct.
No new language syntax or public list implementation is needed for a new host
collection. Test parent scope, empty collections, ordering, unavailable values and
observation refresh after commands. Avoid mutating a collection's identities while
it is being traversed.

## Runtime errors and safety

Execution failures are wrapped with command identity and lifecycle information in
the Automaton. `MicroscopyRuntimeErrorProvider` classifies them;
`AutoStratStrategy._recover` applies the domain pack's policy.

For a new error:

1. Raise a distinguishable application exception; preserve the original cause.
2. Map it in `MicroscopyRuntimeErrorProvider`, retaining command origin.
3. Declare it in `runtime_errors.yaml` and on affected commands in `commands.yaml`.
4. Choose a safe policy: retry only if repeating the physical operation is safe.
   The current pack supports bounded retries with an exhausted action, continue,
   terminate and abort. Strategies do not need explicit error checks each step.
5. Test partial failure, not just failure before any side effect.

`terminate` runs finalisation; `abort` halts without normal finalisation. An uncertain
UV exposure must not be automatically repeated: `targeted_projection_failed`
and `projection_exposure_uncertain` abort. A full-field `projection_failed` is
retryable only when illumination was never attempted. Processing failures also
abort because lineage may have partially advanced.
Arithmetic/evaluator failures are separate from device errors; do not disguise
them as observations or retry hardware to fix an invalid expression.

## Generation guidance and examples

Keep reusable semantics in `semantic_guidance.yaml`, demonstrations in
`few_shot_examples.yaml`, and experiment-specific durations/selection rules in the
user prompt. The GUI pipeline includes the pack's few-shot examples. Keep the
notebook and GUI aligned through existing preview/pipeline helpers rather than
duplicating retry reporting. Do not report unavailable retry counts as zero.

Only change the sibling AutoStrat repository for domain-independent language,
schema, evaluator or pipeline capabilities. Such a change needs AutoStrat's own
tests plus EvoMachine integration tests, and a compatible dependency version for
non-editable installs. Local editable sources can hide packaging/version mistakes.

## Test and deployment checklist

From the EvoMachine repository root:

```bash
.venv/bin/python -m pytest tests/test_strategy_generation_integration.py tests/test_delta_processing.py -q
EVOMACHINE_GUI_RUN_QT_TESTS=1 QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/gui/test_strategy_generation.py -q
```

Cover domain loading/validation, adapter translation, observation scope and
refresh, failure policy, and initialise/step/finalise behavior. Use fake devices
and injected processing backends for deterministic tests; no API key or real UV
exposure should be needed. Add a focused test at each changed boundary, not a new
wrapper just for testing.

Generation acceptance is not evidence of hardware success. Check runtime failures,
saved outputs and expected command execution separately. DeLTA-RT's Python package
does not include EvoMachine's custom trained model files: configure matching model
paths and input sizes in `image_processing_config.py`. The virtual camera supplies
random pixels; use recorded microscopy frames for meaningful processing validation.

Current diagnostic limitation: runtime histories exist on the Automaton/strategy,
but GUI strategy status does not yet expose their full error details. Do not treat
the GUI's stopped label as proof of successful completion.
