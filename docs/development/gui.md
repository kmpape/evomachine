# Extending the EvoMachine GUI

[Documentation](../README.md) › [Developer guide](README.md) › Extending the GUI

See [architecture](architecture.md), [runtime behavior](runtime.md), and
[AutoStrat integration](autostrat.md) for the boundaries surrounding the GUI.
This guide covers the current PyQt5/Napari GUI and its application boundary.

## Request and response path

```text
Panel -> EvoMachineGuiController -> RPC worker/socket -> AutomatonGuiFacade
 -> GUI_REQUEST_HANDLERS -> application operation
 -> GuiResponse -> controller signal -> panel update
```

The GUI and Automaton run in separate processes. Widgets belong to the GUI thread;
hardware state belongs to the application process. Do not import a hardware
singleton into a panel or make blocking socket/model calls from a button handler.

| File | What to change there |
| --- | --- |
| `evomachine/gui/panels/` | Widgets, input collection, display and control enablement |
| `gui/docks/controls.py` | Compose device tabs and Acquisition/Strategy/Focus workflow tabs |
| `gui/controller.py` | Request methods, Qt signals and response routing |
| `gui/protocol.py` | Allowlisted command enum, request/response dataclasses and mutation policy |
| `gui/request_map.py` | Validated backend handlers and `GUI_REQUEST_HANDLERS` registration |
| `gui/facade.py` | Shared application access, status payloads and safety gates |
| `gui/operations.py` | Background operation state, progress and cooperative cancellation |
| `gui/focus.py` | Focus configuration transport, independent manual state and strategy reference review |
| `gui/runtime.py` | Virtual and hardware application construction |
| `gui/central_workspace.py` | Mutually exclusive Microscope/DMD layers and compact metadata presentation |

Paths beginning `gui/` in this table are relative to `evomachine/`.

## Current layout composition

The right **Device Controls** group presents Stage, LEDs, Camera, DMD, and Filter
Wheel as independently scrollable tabs. The device panel classes are flat
`QWidget` contents because the tab already names the device; internal headings
identify meaningful subsections. A trailing stretch keeps short Camera and
Filter Wheel content compact at the top of tall tabs. Acquisition, Strategy, and Focus remain
separate workflow tabs. Automaton lifecycle actions do not belong to a device
tab: the upper-left Controller Status dock renders one flat, scrollable list with
lifecycle actions first, Automaton status below them, and peripheral controllers
after it. The single Refresh Status
action uses `PING`, whose response contains both lifecycle and peripheral status.
Napari Image Settings occupies the lower-left dock row, approximately 40%
of the usable height; the status row receives approximately 60%. The native Layer
List stays hidden.

The central workspace owns exactly two image layers, selected by **Microscope**
and **DMD** tabs. Camera frames and stacks update only the microscope layer, while
DMD previews retain the existing hardware-to-display orientation. A fixed-height
strip beneath the canvas formats existing response fields and follows Napari's
selected stack plane. The first real image and each explicit tab switch are fit
to the canvas; later image updates preserve the user's zoom. Do not reintroduce a
composited dashboard or derive extra metadata by expanding the wire protocol.

## Add a panel using an existing operation

Start with `CameraPanel` for a small status/configuration panel, or `StagePanel`
for asynchronous operations.

1. Create a `QGroupBox`/`QWidget` in `gui/panels/`, accepting the shared controller
   and optional parent. Store only UI state there.
2. Connect buttons to existing controller methods and controller signals to
   display/update methods. Keep rendering callbacks free of backend writes.
3. Reflect device initialisation, active strategy and active operation state in
   control enablement. Backend validation remains required even when disabled
   buttons appear to prevent invalid requests.
4. Instantiate the panel in the relevant builder in `EvoMachineControlsDock` and
   add it to the layout. Reuse the dock's controller, not a new socket per panel.
5. Add a panel test with the existing `FakeController` pattern in
   `tests/gui/test_panels.py`.

Minimal illustrative panel using existing camera RPC (not a new backend API):

```python
from PyQt5.QtWidgets import QGroupBox, QLabel, QPushButton, QVBoxLayout

class CameraSummaryPanel(QGroupBox):
    def __init__(self, controller, parent=None):
        super().__init__("Camera summary", parent)
        self.label = QLabel("Not refreshed")
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(controller.refresh_camera)
        controller.camera_status_received.connect(self.update_status)
        layout = QVBoxLayout(self)
        layout.addWidget(self.label)
        layout.addWidget(refresh)

    def update_status(self, payload):
        self.label.setText(str(payload["camera"]))  # Format selected fields in real UI.
```

Check the controller's emitted payload shape: some signals receive a nested value,
others the entire response payload. Do not assume every signal unwraps the same way.

## Configuration forms and manual acquisition

`ConfigFieldSpec` describes each simple configuration value. Supply `help_text`
with units, effects, ranges, interactions, and the meaning of read-only values;
`ConfigForm` applies it to both the label and input so disabled widgets do not hide
their explanation. Numeric bounds and choices remain validation hints in the GUI,
not replacements for backend validation. Dock-panel composition normalises action
buttons to expanding horizontal and fixed vertical policies using a font-derived
minimum height.

`FrameAcquisitionSettingsPanel` retains the existing wire key
`use_current_main_controls` for compatibility, but presents it as **Use current
device settings**. Its general dialog owns exposure and acquisition cleanup
options. Separate LED, DMD, and filter dialogs own only their existing override
values and are disabled while live device state is selected. The shared
`ManualAcquisitionSaveConfig` is an explicit UI-state object linking the
Acquisition-tab Output Configuration checkbox to manual-frame and Z-stack
payloads. Do not attach it to the Strategy-tab output panel. `ZStackSettingsPanel`
owns relative start/end/step values and merges them with the shared acquisition
payload when submitting a stack.

`FilterWheelPanel` builds checkable position buttons from authoritative status.
A click sends the existing set command, while only a subsequent status response
changes the authoritative checked position. Retain lifecycle and strategy gates
when changing this presentation.

## Add a new GUI operation end to end

Use `CAMERA_SET_EXPOSURE` as the existing small example:

1. Add a unique enum value to `GuiCommandType` in `protocol.py`.
2. Decide whether it mutates application/hardware state. Add mutations to
   `MUTATING_COMMANDS`; use `ALWAYS_ALLOWED_MUTATING_COMMANDS` only for actions
   genuinely safe during strategy execution. Review the facade's
   `_OPERATION_SAFE_COMMANDS` separately for concurrent background operations.
3. Write a handler in `request_map.py` and register it in `GUI_REQUEST_HANDLERS`.
   Validate types, finite numeric ranges, enum choices, device readiness and
   operation prerequisites before acting. Reuse application methods.
4. Return serializable results/status in the established envelope. Keep typed
   configuration/dataclass objects internally; dictionaries are the wire format,
   not a reason to duplicate application state as arbitrary dictionaries.
5. Add a controller method calling `_send`, then route any new response key in
   `_handle_response` to a typed-purpose Qt signal.
6. Connect the panel and test the request payload, handler behavior, response
   routing, error case and safety gate.

Never pass a `VerifiedStrategy`, device, NumPy array or arbitrary Python object
directly over the JSON RPC boundary. For images, reuse `image_payloads.py` and the
existing negotiated image transport instead of adding another encoding.

## Work that takes more than a moment

Moving network I/O to the controller worker keeps Qt responsive, but a long
synchronous backend handler can still monopolise the Automaton request loop.
Use `GuiOperationManager` for suitable long-running operations:

- A start handler launches a runner and immediately returns its operation ID/state.
- The runner receives a cancellation event and progress reporter. Check cancellation
  between safe units of work; cancellation does not forcibly interrupt a device call.
- A status handler returns operation state; a `QTimer` polls with at most one
  outstanding request. Stop polling on terminal states and clear pending flags on
  transport errors as well as successful responses.
- Never update Qt widgets from a worker. Route completion through controller signals.
- Decide which concurrent operations are safe. Do not bypass the hardware operation
  gate simply to make a button responsive.

Use `gui_operations` for hardware operations. AutoStrat generation uses a separate
`strategy_generation` manager because it must not lock stage/camera controls and
does not execute hardware. Preserve that separation. Generation status exposes
serializable preview data; the accepted strategy object stays in the backend.

## AutoStrat UI and credentials

`StrategySetupPanel` switches between Python and AutoStrat sources but shares
Set/Start/Stop. AutoStrat includes an editable code area and a named `.strat` file library
under `strategies/autostrat`. Loaded or hand-written code must pass deterministic parser and
domain validation before Set can install it. Generation only prepares a candidate; Set
installs it; Start is an explicit separate action. A changed prompt invalidates the prior
candidate for installation, and generation IDs prevent installing stale results.

The startup dialog sends a provider, model, optional base URL and masked API key through
`AUTOSTRAT_CONFIGURE`. The Robin Office 2 preset contains only non-secret connection
metadata; custom OpenAI-compatible and Anthropic sessions are supported. Blank Apply or
Continue without AI disables generation even if an environment key was inherited; local
`.strat` workflows remain enabled. The key is held in backend process memory, passed only
to the disposable generation process, and is not saved or returned in status. Do not log
request payloads containing credentials. This RPC connection is not a credential-security
boundary: use the default local connection; do not expose it on an untrusted network.

Model/endpoint configuration and generation pipeline construction live in
`gui/strategy_generation.py`. Reuse `strategy_generation/preview.py` for notebook-
compatible diagnostics. UI labels say “strategy code”; internal DSL names need
not be renamed. GUI generation runs in a disposable process so Cancel and the
`AUTOSTRAT_GENERATION_TIMEOUT_SECONDS` limit can stop a stalled model request.

## Focus panels and configuration

Manual CRISP and software-focus panels live in the RHS Focus tab, alongside the
reference-profile panel. The Strategy tab owns global and per-FoV mode selection.
Use `ConfigForm` inside larger editors and `ConfigDialog` for simple modal forms.
A mode change selects a whole editor page; retain inactive page values.
FoVs can inherit the global mode or select their own, with optional settings
overrides. The manual Focus tab does not change these strategy settings.

`panels/focus_settings.py` owns the mode-specific editors. Reference sampling uses
`SamplingEditor`: generated outer range/step, optional denser centre, exact-position
preview, and Custom mode for irregular lists. Image sampling controls acquisition;
the matching range/step controls numerical Pearson candidates, not additional images.
Preserve custom positions and fractional shift bounds when round-tripping settings.

Applying unchanged configuration or editing scan/search settings must retain an
accepted reference. `ReferenceProfileFocus.update_config` owns compatibility:
smoothing/metric changes refit cached images for review; changed imaging settings
keep the old curve visible but require a new capture. Panels should display the
backend's acceptance, calibration-needed flag and explanatory message.

`panels/reference_focus.py` embeds the dark Matplotlib plots. Latest samples appear
as crosses over the reference; alignment and score scaling affect display only.
Show sparse/recovery outcomes and rejection reasons, retaining rejected correlations
for inspection. Explain undefined correlations instead of presenting a silent empty
plot. Numeric curves cross JSON as lists, with unsupported correlations represented
by null; hardware objects and image stacks remain in the application process.

Manual configuration/results are independent of strategy state. FoV setup must not
subscribe to manual software-focus status as a source of strategy defaults. Load
strategy defaults using the focus-defaults endpoint instead. Strategy references
are captured at saved FoV Zs on first use, with acceptance before imaging. The
review command accepts only the current candidate ID. Preserve its limited
mutation gate. Manual scans use `GuiOperationManager`; strategy-owned scans service
GUI requests between completed planes, and reference-review waits service requests
on the strategy thread. Keep status and Stop responsive without running hardware
from the socket thread. Cancellation checks run at safe boundaries; an active
device call may need to finish before cancellation takes effect.

## Lifecycle rules to preserve

- Keep FOV setup separate from physical movement. Capturing a position requests
  fresh stage coordinates; iterating strategy collections is not movement.
- Set Strategy prepares a fresh instance and can reuse configured FOVs. It is
  blocked during active execution, command unwinding and after shutdown.
- Following stop/completion, users press Set Strategy again, then Start Strategy.
  Start validates the prepared strategy before explicitly clearing the halt flag.
  Installing a strategy must not itself resume hardware.
- Long strategy waits service GUI requests. Retain the re-entrancy guard and the
  checks that prevent execution continuing after stop; a new strategy must not
  replace one halfway through an old command batch.
- A successful Start RPC only means the request was accepted. Do not present it
  as proof of completed acquisition or successful processing.

## Verification

From the repository root:

```bash
EVOMACHINE_GUI_RUN_QT_TESTS=1 QT_QPA_PLATFORM=offscreen uv run pytest tests/gui -q
EVOMACHINE_GUI_RUN_QT_TESTS=1 QT_QPA_PLATFORM=offscreen uv run pytest -q
uv run python scripts/launch_virtual_gui.py
```

Qt tests are opt-in; a run that skips them is not GUI verification. Use panel tests
for wiring/layout, controller tests for routing, facade/request-map tests for
validation, socket tests for transport/responsiveness, and Automaton tests for
lifecycle behavior. Prefer extending those tests over building parallel mocks or
new abstraction layers for each feature.

Manually check narrow layouts, long labels, pending/error states, stop during a
wait, and Set/Start after completion. Virtual tests do not establish physical
projection accuracy or DeLTA model performance. Do not trigger real hardware or
live paid model requests as part of the default test suite.

---

← [Virtual implementations](virtual-mode.md) | [Developer guide](README.md) | [AutoStrat](autostrat.md) →
