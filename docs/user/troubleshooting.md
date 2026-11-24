# Failures, logs, and troubleshooting

[Documentation](../README.md) › [User guide](README.md) › Failures, logs, and troubleshooting

The application log is the primary diagnostic record. It appears in the terminal,
a rotating log file below the data directory, and the GUI log buffer according to
configured levels. Command failures include strategy, command type and ID,
lifecycle section, FoV, original error, and traceback.

## Validation errors

Configurations validate when constructed or updated. A rejected GUI update does
not replace the active configuration. Common causes include an invalid filename
field, a focus step larger than its range, no usable virtual TIFF, an out-of-bounds
stage coordinate, or segmentation requested without its prerequisites.

## Command and strategy failures

When a strategy command fails, the remaining commands in that batch stop. The
failure is wrapped with command context and added to runtime history. Python
strategies receive errors through their callback. AutoStrat classifies the error
and applies its domain policy:

- **retry** repeats the failed command and then the unexecuted batch tail;
- **continue** skips the failed command and resumes the tail;
- **terminate** exits normally and runs finalisation;
- **abort** stops without normal finalisation.

Retries are bounded. Exhaustion uses the declared action, which is also logged.
Unexpected evaluator, adapter, and integration errors take a fail-safe abort path.

Physical effects cannot be rolled back. In particular, uncertain projection
exposure is not automatically retried because that could deliver a second dose.

## Stop and shutdown

**Stop Strategy** requests interruption; it is not an undo operation. A physical
operation may already have completed or partially completed. Closing the GUI
requests backend shutdown, stops active work, clears or disables supported
peripherals, and finalises connections.

## Diagnostic checklist

1. Select the log for the correct run and find its earliest error.
2. Read the command type, ID, lifecycle section, and FoV around it.
3. Separate the original exception from later recovery and shutdown messages.
4. Determine whether execution retried, continued, terminated, or aborted.
5. Confirm active output, focus, camera-source, and FoV configuration.
6. Reproduce through virtual devices or a minimal manual operation where useful.
7. Never repeatedly retry uncertain illumination or movement on hardware.

The compact GUI status is not complete failure history. Use the full log and
Automaton/strategy histories for diagnosis.

---

← [Virtual mode](virtual-mode.md) | [User guide](README.md)
