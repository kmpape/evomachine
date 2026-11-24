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

The left **Controller Status** area is one flat list: lifecycle buttons appear
first, followed by Automaton status and then peripheral-controller status. Red
and green dots show whether each component is responding. **Refresh Status**
pings the Automaton and refreshes both lifecycle and peripheral information.
**Device Controls** contains separate Stage, LEDs, Camera, DMD, and Filter Wheel
tabs. The workflow area retains the **Acquisition**, **Strategy**, and **Focus**
tabs.

Each device tab uses the tab name as its only outer title; redundant device
frames are omitted. Controls are packed at the top rather than stretched over the
available height. Stage and DMD retain bold internal section headings, while the
LED tab groups channels by controller. Camera status includes the current and
default exposure alongside image shape, dtype, and readout mode.

The central canvas has **Microscope** and **DMD** display tabs. Only the selected
image is visible. A compact row beneath it shows the metadata already available
for the selected frame, stack plane, or DMD pattern. Napari **Image Settings**
remains in the lower-left area; the Layer List is hidden. The DMD image has a red
outline marking the full pattern extent without changing the underlying pattern.

1. Initialise devices in Controller Status.
2. Verify stage position and software travel bounds.
3. Configure acquisition, output, and focus.
4. Test one manual image.
5. Register FoVs and test movement between them.
6. Only then initialise and start a strategy.

Controls that mutate hardware or shared experiment configuration are rejected
while an incompatible operation or strategy is active.

## Acquisition

### Acquisition Configuration

Select **Use current device settings** to acquire with the exposure, LED, DMD,
and filter-wheel state already selected in Device Controls. Clear it to enable
the separate **LEDs**, **DMD**, and **Filter Wheel** configuration buttons.

The general **Configure** dialog contains exposure, normalisation, DMD
illumination/cleanup, and LED restoration/disable behavior. If both restoration
and disabling are selected, disabling LEDs takes precedence. These settings are
defaults for GUI manual acquisitions; strategy commands provide their own values.
Hover over a configuration label or input to see its units, accepted range,
effect, or read-only meaning. Help remains available when an input is disabled.

Virtual mode adds **Virtual Image Source**. See [Virtual mode](virtual-mode.md).

### Output Configuration

In the Acquisition workflow, **Save manual acquisitions** controls saving for
the GUI's manual frames and Z-stacks. It does not affect Python or AutoStrat
strategies, which control their own saving. The Strategy workflow therefore does
not show this checkbox.

Choose the output directory and filename pattern. The destination and pattern
remain shared with strategy saves even though the manual-save checkbox is not.
The current pattern is shown in the panel. **Modify Pattern** opens a dialog that
validates before applying. `.tiff` is added automatically.

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

A Z-stack owns its relative start, end, and step values. Use **Configure** in the
Z Stack Acquisition section to edit them; cancelling keeps the active values.
The request combines that range with the shared acquisition settings. The stack
reads the starting Z, captures the centre followed by positive and then negative
complete steps within the requested range, and normally restores the starting Z.
An interrupted stop does not promise restoration. Check stage bounds first.

### Filter wheel

The Filter Wheel device tab shows one checkable button per reported position.
Clicking a button submits the existing filter command immediately. The checked
button follows the position confirmed by the backend; Refresh reads it again.
Manual position changes remain locked while a strategy is running. Positions use
full-width, wavelength-coloured buttons. The five-colour
`BANDPASS_405_445_514_561_640nm` position is the Chroma
[ZET405/445/514/561/640m](https://www.chroma.com/products/parts/zet405-445-514-561-640m/)
multi-bandpass filter. `FILTER` remains accepted only as a compatibility name for
older configurations.

The DMD Calibration section reports `calibration status`, which is the state of
the asynchronous calibration task—not a separate device action. It changes from
idle/starting to running and then to a terminal state such as completed, failed,
or cancelled.

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

The Focus tab separates hardware autofocus from one unified software-focus engine.

- **Autofocus** manages the binding-specific lock and calibration process.
- **Software Focus** offers **Maximum** selection or **Reference profile**
  matching. Both use the same focus images, typed scoring method, acquisition
  settings, and optional crop rectangles. Maximum selects the best Z score.
  Reference profile compares later scans to an accepted calibrated curve.
  Calibration must be reviewed and explicitly
  accepted. Strategies acquire an independent curve when each FoV is first
  visited by default. Enable **Use one reference curve for all FoVs** only for
  samples whose focus-score profiles are expected to be comparable.

Use **Configure Crop** to score the whole image or one GUI-defined rectangle.
Python configurations may supply several rectangles. Crop coordinates include
the left/top pixels and exclude the right/bottom pixels; the microscope view
shows authoritative crops as a read-only cyan overlay.

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
