# Peripherals, controllers, and bindings

[Documentation](../README.md) › [Developer guide](README.md) › Peripherals, controllers, and bindings

## Separation of responsibilities

- `PeripheralController` owns a connection, initialisation, liveness, and closing.
- A peripheral base class owns binding-neutral validation, readiness checks,
  cached state, and public behavior.
- A concrete binding translates protected low-level operations to a controller or
  third-party driver.
- A typed config describes construction; a factory selects the binding.

Shared behavior belongs in the base class. Do not repeat readiness, range, state,
or lifecycle logic across bindings.

## Existing binding identifiers

`BindingType` currently includes `VIRTUAL`, `ASI_TIGER`, `SYNCBOARD`, `KWR103`,
`EM_DMD_WINDOW`, `PYGAME`, `MMC`, and `PVCAM`. Not every peripheral supports every
binding. The enum docstrings and each factory's dispatch are authoritative.

## Add a binding

1. **Choose the interface.** Extend the relevant base peripheral. If the device
   uses a new shared connection type, implement a `PeripheralController` first.
2. **Reuse or extend typed config.** Put common construction fields in the
   binding-neutral config. Add binding-specific calibration/config only when it
   cannot be expressed by an existing type. Do not pass an unstructured option
   dictionary across the factory boundary.
3. **Add a binding identifier only when necessary.** Reuse an identifier when the
   new class is another implementation detail of the same backend. Otherwise add
   `BindingType` and document which peripherals support it.
4. **Implement low-level methods.** Let the base class retain public lifecycle,
   readiness checks, validation, and cached state. Convert external-driver values
   and units at the binding boundary.
5. **Update the factory.** Map the typed config's binding to the concrete class and
   retrieve its controller with `get_peripheral_controller()` where appropriate.
   Fail clearly for unsupported combinations.
6. **Translate errors.** Preserve the original exception with chaining. Raise a
   distinguishable application/binding error when recovery policy depends on the
   failure type. Never report a device command as successful before confirmation.
7. **Implement cleanup.** `stop()` should halt active behavior without destroying
   the connection; `finalise()` should leave a safe state and release owned
   resources. Respect shared-controller ownership and `close_on_shutdown`.
8. **Wire a runtime explicitly.** Update a launcher/runtime factory only when that
   deployment should use the binding. Adding a factory option must not silently
   change existing hardware.

## Controllers

`PeripheralControllerConfig` selects binding, name, and whether to initialise on
construction. `SerialPeripheralControllerConfig` accepts exactly one of explicit
`port` or `hwid`; `SocketPeripheralControllerConfig` accepts host and port.

When several logical peripherals use one connection, construct the controller
once and inject it into their factories. Do not let each peripheral independently
open the same serial device.

## Lifecycle contract

All high-level peripherals implement:

- `initialise(force=False)`;
- `is_alive()`;
- `stop()`;
- `finalise(force=False)`.

Actions that query or mutate hardware normally enforce `check_initialised` and
`check_alive`. Virtual/test construction may disable these checks explicitly;
hardware defaults should keep them enabled. Forced calls must not weaken
validation or conceal an unsafe state.

## Logging

Create binding loggers with `get_logger(..., is_binding=True)` and peripheral
loggers with `is_peripheral=True`. Log device identity, operation, important
arguments, and returned status without exposing secrets. Raise the exception
after logging only when the caller must handle failure; avoid duplicate traceback
logs at every layer.

## Tests

Follow `evomachine/bindings/README.md`:

- fake bindings belong in normal deterministic tests;
- shared behavior is tested against the base contract and representative
  implementations;
- real-device tests are opt-in and clearly marked;
- no default test may require connected hardware, an API key, or UV exposure.

Test construction and validation, lifecycle transitions, readiness guards,
argument/unit translation, state updates, controller sharing, external exceptions,
partial failures, stop/finalise cleanup, and factory dispatch.

---

← [Configuration](configuration.md) | [Developer guide](README.md) | [Runtime](runtime.md) →
