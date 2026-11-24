# Using the GUI

[Documentation](../README.md) › [User guide](README.md) › Using the GUI

## Launch

For development and strategy testing:

```bash
uv run python scripts/launch_virtual_gui.py
```

For the configured microscope:

```bash
uv run python scripts/launch_hardware_gui.py
```

Both launchers start the Automaton backend in a child process and Napari in the
GUI process. Options common to both include `--host`, `--port`,
`--image-transport`, and `--no-napari`. The hardware launcher additionally accepts
`--runtime module:function` for an alternative Automaton factory.

## Layout and recommended order

**Main Controls** contains Automaton lifecycle, stage, LEDs, camera, DMD, and
filter controls. The workflow area contains **Acquisition**, **Strategy**, and
**Focus** tabs.

1. Initialise devices in Main Controls.
2. Verify stage position and software travel bounds.
3. Configure acquisition, output, and focus.
4. Test one manual image.
5. Register FoVs and test movement between them.
6. Only then initialise and start a strategy.

Controls that mutate hardware or shared experiment configuration are rejected
while an incompatible operation or strategy is active.

## Acquisition

### Acquisition Configuration

The acquisition settings control saving, image normalisation, DMD illumination,
DMD clearing, and LED restoration/disable behavior. These are defaults for GUI
manual acquisitions; a strategy command may supply its own save behavior.

Virtual mode adds **Virtual Image Source**. See [Virtual mode](virtual-mode.md).

### Output Configuration

Choose the output directory and filename pattern. The current pattern is shown in
the panel. **Modify Pattern** opens a dialog that validates before applying.
`.tiff` is added automatically.

Available fields are:

| Field | Meaning |
| --- | --- |
| `{channel}` | First configured LED channel, or an empty value when absent. |
| `{fov_id}` | FoV ID; the unknown sentinel `-1` renders as an empty string. |
| `{x}`, `{y}`, `{z}` | Stage coordinate components when known. |
| `{filter_wheel}` | Filter identifier when configured. |
| `{timestamp}` | Filename-safe acquisition timestamp. |
| `{timezone}` | Filename-safe timezone identifier. |
| `{frame_id}` | ID assigned by `FrameMetaDataFactory`. |
| `{callback_id}` | Strategy callback ID when one is supplied. |

Literal text, escaped braces (`{{` and `}}`), conversions, and Python format
specifications such as `{frame_id:04d}` are accepted. Include `timestamp`,
`frame_id`, or `callback_id` to reduce accidental overwrites. Additional TIFF
metadata is saved but cannot introduce filename fields.

### Manual acquisition and Z-stacks

Manual acquisition constructs `FrameMetaData` from the selected channel,
brightness, filter, exposure, FoV, and DMD settings. The result appears in Napari
and is saved only when saving is enabled.

A Z-stack reads the starting Z, captures the centre followed by positive and then
negative complete steps within the requested half-range, and normally restores
the starting Z. An interrupted stop does not promise restoration. Check stage
bounds before acquisition.

### Saved-image loader

The loader lists TIFF files from its own chosen directory and displays the
selected image. Its directory is independent from Output Configuration.

## FoVs and strategies

The Strategy tab registers linear or explicit FoV coordinates. Each FoV receives
an integer ID. Initialising FoVs stores coordinates and prepares the configured
focus policy; it does not start the selected strategy.

**Python Strategies** lists concrete `AbstractStrategy` subclasses discovered in
top-level `strategies/*.py` files. AutoStrat source can be generated, loaded,
edited, validated, and installed separately. **Set Strategy** installs the chosen
strategy, **Initialise** prepares it with FoVs, and **Start Strategy** begins
execution. See [Writing and running strategies](strategies.md).

## Focus

The Focus tab separates hardware autofocus, software focus, and reference-profile
focus.

- **Autofocus** manages the binding-specific lock and calibration process.
- **Software Focus** captures a Z scan and selects the best score using its
  configured algorithm.
- **Reference Focus** calibrates a whole-image reference profile and compares
  later scans to estimate Z shift. Calibration must be reviewed and explicitly
  accepted.

Strategy focus configuration has three levels: global navigation behavior,
per-FoV policy, and the software/reference algorithm configuration. Per-FoV
overrides replace the corresponding defaults. Reference-profile mode is
software-only and cannot enable hardware autofocus simultaneously.

See the [strategy focus section](strategies.md#focus-behavior) and the
[configuration reference](../development/configuration.md#focus-and-navigation).

## Logs and shutdown

Watch the application log during initialisation and strategy execution. A stopped
status alone does not prove success. On close, allow the backend to complete its
shutdown request. If the process exits unexpectedly, confirm illumination, DMD,
autofocus, and stage state before restarting.

---

← [System overview](overview.md) | [User guide](README.md) | [Writing strategies](strategies.md) →
