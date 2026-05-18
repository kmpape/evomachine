# Binding Test and Fake Hardware Policy

Bindings adapt evomachine's typed device interfaces to concrete hardware or
debug backends. Tests should exercise shared behavior through the public device
APIs and use binding-specific fake controllers when real hardware is not
explicitly enabled.

## Fake Bindings

- Fake classes live next to the binding they simulate, for example
  `bindings/asitiger/stage.py` contains `FakeTigerStageController`.
- Fake classes must not import pytest and must not require real serial, socket,
  display, or hardware resources.
- Fake classes should record commands and expose enough state for tests to
  assert behavior through public APIs.
- Fake classes are deterministic test/debug helpers; production code should
  continue to use the real controller classes.

## Real Binding Tests

The checked-in test configuration defaults to fake bindings only. Real hardware
tests must be enabled through the test binding configuration file and should be
skipped unless all required ports, HWIDs, or executable paths are configured.

## Shared Behavior Tests

Tests should be written around behavior, not implementation names. Prefer a
public test such as `test_movement` that loops over configured stage cases and
calls a helper such as `_test_movement(stage_case)`. Apply the same pattern for
peripheral controllers, LEDs, filter wheels, and DMDs.
