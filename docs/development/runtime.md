# Commands, runtime lifecycle, and errors

[Documentation](../README.md) › [Developer guide](README.md) › Commands, lifecycle, and errors

## Command construction

`AutomatonCommand` carries a unique ID, `AutomatonCommandType`, validated command
arguments, creation/execution timestamps, output data, and optional FoV context.
Strategies should create commands only through `CommandFactory`. The factory owns
the command counter and validates processing channels, FoVs, ROIs, ranges, and
argument types close to construction.

Each strategy declares all possible command types through
`register_automaton_commands()`. Before execution, the Automaton validates that
required devices exist and that emitted command types were declared.

## Lifecycle

1. `set_strategy()` installs a strategy only while execution is inactive and
   resets lifecycle flags.
2. `initialise()` initialises devices and FoV/processing state, calls the
   strategy's initialise boundary, merges strategy and GUI focus configs, and
   registers FoVs in `FocusNavigator`.
3. A start event begins initial commands and callback batches.
4. Successful commands are returned with execution data. The callback supplies
   the next batch.
5. Normal completion or terminate enters finalisation once.
6. Abort skips normal strategy finalisation. Shutdown independently stops work
   and finalises devices.

GUI per-FoV focus configuration overrides the strategy's config for matching
IDs. A new strategy run resets processing, treatment, and pending projection
state.

## Validation failures

Prefer validation at typed configuration and command boundaries. Factories and
GUI handlers must revalidate server-side; client validation improves feedback but
is not authoritative. Invalid updates must be atomic: retain the active object
when construction of the candidate fails.

## Command failures

The Automaton wraps an execution failure in `CommandExecutionError`, preserving:

- command ID, type, and resolved arguments;
- lifecycle section (`initialise`, `step`, or `finalise`);
- original exception and causal traceback;
- occurrence time and FoV/logging context.

It stops the current batch, records failure history, logs detailed context, and
passes the failure to the strategy boundary. `UnexpectedRuntimeError` represents
failures outside normal command execution.

Python strategies decide how their callback treats received errors. AutoStrat
uses `MicroscopyRuntimeErrorProvider` to classify them and `_recover()` to apply
the domain policy.

## Recovery actions

| Action | Runtime effect |
| --- | --- |
| `retry` | Re-emit the failed resolved command, followed by the unexecuted tail. Already completed commands are not replayed. |
| `continue` | Skip the failed command and resume the tail; invalidate measurements where required. |
| `terminate` | Exit active loops and run finalisation sequentially. |
| `abort` | Stop strategy execution without normal finalisation. |

Retry count is policy-owned and bounded. On exhaustion the declared exhausted
action is selected. Classification, policy action, retries used/allowed, and the
exhausted action are logged.

Only retry operations whose physical repetition is safe. Projection with an
uncertain exposure aborts because the delivered dose cannot be known or undone.
Processing failures abort when partially advanced lineage could make continuing
ambiguous.

## Stop and shutdown

Stop and shutdown are separate events. Stops are checked by waits and supported
operations and lead out of strategy execution. Shutdown is application-wide and
must stop peripherals even if strategy finalisation fails or is skipped. Cleanup
should be idempotent and defensive because a device can fail between an action
and cleanup.

## Logging and histories

Use `get_logger(__name__)` so records share terminal, rotating-file, and GUI
handlers. Peripheral and external-binding loggers have independently configured
levels. Reconfiguration removes existing handlers before installing exactly one
of each application handler.

The GUI status is intentionally compact. Complete command and recovery details
belong in logs and runtime histories. Never reduce a caught exception to a status
string before recording its original context and traceback.

---

← [Bindings](bindings.md) | [Developer guide](README.md) | [Virtual implementations](virtual-mode.md) →
