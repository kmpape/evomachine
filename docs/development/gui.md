# Extending the EvoMachine GUI

See [AutoStrat integration](autostrat.md) for generated commands and observations.
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
| `gui/docks/controls.py` | Compose panels into the main controls/acquisition/strategy tabs |
| `gui/controller.py` | Request methods, Qt signals and response routing |
| `gui/protocol.py` | Allowlisted command enum, request/response dataclasses and mutation policy |
| `gui/request_map.py` | Validated backend handlers and `GUI_REQUEST_HANDLERS` registration |
| `gui/facade.py` | Shared application access, status payloads and safety gates |
| `gui/operations.py` | Background operation state, progress and cooperative cancellation |
| `gui/runtime.py` | Virtual and hardware application construction |
| `gui/central_workspace.py` | Image/plot presentation in the central workspace |

Paths beginning `gui/` in this table are relative to `evomachine/`.

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

`StrategySetupPanel` switches between fixed and AutoStrat sources but shares
Set/Start/Stop. Generation only prepares a candidate; Set installs it; Start is an
explicit separate action. A changed prompt invalidates the prior candidate for
installation, and generation IDs prevent installing stale results.

The startup prompt in `napari_app.py` sends an optional masked API key through
`AUTOSTRAT_CONFIGURE`. Blank Enter/Cancel disables AutoStrat even if an environment
key was inherited. The key is held in backend process memory, not saved or returned
in status. Do not log request payloads containing credentials. This RPC connection
is not a credential-security boundary: use the default local connection; do not
expose it on an untrusted network.

Model/endpoint configuration and generation pipeline construction live in
`gui/strategy_generation.py`. Reuse `strategy_generation/preview.py` for notebook-
compatible diagnostics. UI labels say “strategy code”; internal DSL names need
not be renamed.

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
EVOMACHINE_GUI_RUN_QT_TESTS=1 QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/gui -q
EVOMACHINE_GUI_RUN_QT_TESTS=1 QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q
.venv/bin/python scripts/launch_virtual_gui.py
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
