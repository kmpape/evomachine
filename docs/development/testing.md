# Development, testing, and releases

[Documentation](../README.md) › [Developer guide](README.md) › Development, testing, and releases

## Development environment

Use Python 3.11 and run commands from the main repository:

```bash
uv sync
uv run pytest
```

Development `uv sync` installs sibling repositories in editable mode through
`[tool.uv.sources]`. This can modify `uv.lock` to reflect local sources; do not
commit that incidental lock change. Production resolution uses pinned git
dependencies:

```bash
uv lock --no-sources
uv sync --locked --no-sources --no-dev
```

Optional extras are `pvcam` and `autostrat`. PVCAM additionally requires the
vendor SDK. The EM DMD window is an external sibling executable rather than a
Python dependency.

## Test layers

| Layer | Purpose |
| --- | --- |
| Unit | Config validation, data types, calculations, command construction, and isolated state transitions. |
| Manager/peripheral | Shared device behavior with fake or virtual dependencies. |
| Binding contract | Factory dispatch, driver translation, lifecycle, and controlled external failures. |
| Automaton integration | Command batches, strategy lifecycle, stop/shutdown, failure history, and safety behavior. |
| GUI facade/controller | Protocol validation, routing, mutation guards, and authoritative responses. |
| Qt panel | Visible controls, signals, validation feedback, and panel refresh. |
| Virtual end-to-end | Launcher, socket, acquisition, saving, and strategy smoke behavior. |

Run focused tests while developing, then the complete suite. Qt tests are opt-in:

```bash
EVOMACHINE_GUI_RUN_QT_TESTS=1 QT_QPA_PLATFORM=offscreen uv run pytest tests/gui
uv run pytest
```

Default tests must not require hardware, an API key, a paid model request, trained
model files, or real illumination. Use injected clocks, sleeps, processing
backends, and failure objects for deterministic behavior.

## Test expectations by change

- Config changes test valid construction, each important invalid dependency,
  validated updates, and atomic retention after failed GUI updates.
- Manager changes test normal behavior, stop, partial failure, restoration, and
  cleanup.
- Binding changes follow the fake/real/shared policy in
  `evomachine/bindings/README.md`.
- Strategy changes test declaration, initialise/step/finalise, output data,
  command failures, and stop behavior.
- GUI operations test panel, controller, facade/request map, and socket boundaries
  rather than only one layer.
- Processing changes test state invalidation and partial mutation, not only a
  successful model call.

Every new function, class, and test function requires a docstring under the
repository instructions.

## Release dependency workflow

For a sibling dependency change:

1. merge and release the sibling repository;
2. update its pinned tag or revision in `pyproject.toml`;
3. run `uv lock --no-sources`;
4. test a non-editable install so local checkouts cannot hide packaging errors;
5. commit the intentional `pyproject.toml` and production lock updates together.

AutoStrat changes that alter language/schema behavior require tests in AutoStrat
and EvoMachine plus a compatible pinned revision. Microscopy-only commands and
observations normally belong in EvoMachine's domain pack and adapter.

## Documentation checks

When public behavior changes, update the user workflow, configuration reference,
and relevant extension guide in the same change. Verify relative links, imported
names, launch commands, GUI labels, defaults, units, and validation limits against
source. Do not document backup files or experimental scripts as current APIs.

---

← [Image processing](image-processing.md) | [Developer guide](README.md)
