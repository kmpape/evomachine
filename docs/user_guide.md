# EvoMachine user guide

This guide is for experiment operators and authors of experiment strategies. It
covers the normal setup path, focus modes, configuration, commands, and the main
code structure. Hardware installation and microscope commissioning are
site-specific and are not covered here.

## Quick start

EvoMachine is normally started through:

- [`scripts/launch_gui.py`](../scripts/launch_gui.py) uses the microscope
  hardware.

Before launch, edit the script's `__main__` section to select the save directory,
camera configuration, image-processing configuration, and strategy. If you would like to use `pycharm` as IDE, open a terminal, then type `pycharm`. Open `scripts/launch_gui.py` and press the green play button on the top right. Wait a few seconds until the GUI finishes loading.


In the experiment panel:

1. Record one or more `From`/`To` paths. A path generates a grid of fields of
   view (FoVs).
2. Select the focus mode before initialization.
3. Use **Initialise all**, or initialize **FoVs**, **Focus**, references, and
   image processing in that order.
4. Check the save path, selected strategy, illumination, and focus state.
5. Select **Start**. The strategy then supplies commands to the Automaton.

Focus selections are captured during initialization. Reinitialize the
experiment after changing them. **Focus Curves** is available only when an
initial software-focus scan produced curve data.

## How the application fits together

```text
configuration + selected strategy
              |
              v
             GUI  <------ status and command results
              |
              v
          Automaton ------> camera, stage, CRISP, LEDs, filter wheel, DMD
              |
              +-----------> saved files
              |
              +-----------> optional DeLTA image processing
```

- Configuration dataclasses describe the camera, focus hardware, software
  focus, and processing pipeline.
- A strategy creates commands in initialization and callbacks.
- The Automaton executes each returned command in order and returns the same
  command objects, populated with results, to the next strategy callback.
- The camera classes contain device-specific acquisition, movement, focus, and
  file-naming behavior.
- GUI and worker queues keep hardware execution outside the GUI process.

See [Code map](#code-map) for the relevant modules.

## Configuration

The factory classes provide a useful starting point. Customize the returned
objects before constructing the strategy or starting the GUI:

```python
from pathlib import Path

from evomachine.config import (
    ConfigCameraFactory,
    ConfigImageProcessorFactory,
)
from evomachine.evotypes import FocusAlgorithmType, LEDType

camera_config = ConfigCameraFactory.default_air_config(
    path_to_save=Path("/absolute/existing/output/directory"),
)
camera_config.focus.focus_channel = LEDType.LED_565_NM
camera_config.focus.exposure_time = 100
camera_config.focus.algorithm = FocusAlgorithmType.BANDPASS_FFT

processor_config = ConfigImageProcessorFactory.default_config(
    channels=[LEDType.LED_450_NM, LEDType.LED_565_NM],
    channels_seg=[LEDType.LED_565_NM],
)
```

Configuration objects validate most values when constructed. If fields are
changed later, use values within the same constraints.

### `ConfigCamera`

`ConfigCamera` collects all camera-facing configuration:

| Setting | Meaning |
| --- | --- |
| `objective` | Objective magnification and numerical aperture configuration. |
| `image` | Image dimensions and pixel data type. Images must currently be square. |
| `focus` | `ConfigFocus` used for software focus and Z-stack defaults. |
| `autofocus` | `ConfigCRISP` used when CRISP autofocus is enabled. |
| `leds`, `filters` | Hardware options exposed to the application. |
| `path_to_save` | Existing output directory used by image and Z-stack saving. |
| `default_exposure_time` | Default camera exposure in milliseconds; factory default is `200`. |
| `default_focus_channel_id` | Index into `leds`; factory default is `0`. |
| `cam_pxl_size` | Physical camera-pixel size in micrometres; default is `6.5`. |

Use `ConfigCameraFactory.default_air_config()` or
`ConfigCameraFactory.default_oil_config()`. The latter selects the oil-objective
and oil-CRISP defaults. Always verify the objective, available LEDs/filters, and
save path against the installed microscope.

### `ConfigFocus`

The default from `ConfigFocusFactory.default_config()` is:

| Setting | Default | Constraint and effect |
| --- | ---: | --- |
| `exposure_time` | `200` | Focus exposure in ms; at least `0.01`. |
| `focus_channel` | `LED_450_NM` | Illumination channel used during a focus scan. |
| `brightness` | `29` | Focus illumination in the range `0`–`29`. |
| `filter_wheel` | `FILTER` | Filter wheel of `FilterWheelType`. |
| `rel_range` | `50` | Scan distance on either side of the center, in 0.1 µm stage units (`50` = ±5 µm); positive and below `2000`. |
| `step_size` | `5` | Scan spacing in 0.1 µm units (`5` = 0.5 µm); positive and no larger than `rel_range`. |
| `algorithm` | `STEEL` | Focus-score algorithm. |
| `rowshift_px` | `25` | Vertical pixel shift used by `STEEL`. |
| `colshift_px` | `0` | Horizontal pixel shift used by `STEEL`. |
| `cropping_box` | camera-specific box | Region scored during focus; the factory box must be reviewed for each chip and camera. |
| `user_input` | `True` | Allow confirmation prompts during interactive software focus. |
| `cell_radius` | `20.0` | Approximate cell radius in full-resolution pixels for `BANDPASS_FFT`; positive. |
| `downsample` | `4` | FFT input stride; positive integer. |
| `order` | `4` | Butterworth filter order; positive integer. |
| `saturation` | `None` | Optional positive upper intensity clip for `BANDPASS_FFT`. |
| `normalize_brightness` | `True` | Divide the FFT score by squared mean intensity. |

The software-focus algorithms are:

- `LAPLACIAN_VAR`: variance-based Laplacian sharpness score.
- `SQUARED_GRAD_AVG`: mean squared horizontal-gradient score with a threshold.
- `STEEL`: the default shifted row/column difference score. Its tunable settings
  are `rowshift_px` and `colshift_px`.
- `BANDPASS_FFT`: variance after a cell-scale frequency bandpass. Its tunable
  settings are `cell_radius`, `downsample`, `order`, `saturation`, and
  `normalize_brightness`.

Higher scores indicate sharper images. The focus scan chooses and returns to the
best-scoring Z plane. A crop containing representative cells and little empty
background usually gives a more useful curve.

### `ConfigCRISP`

`ConfigCRISP` controls hardware autofocus. Air and oil defaults differ mainly in
objective NA.

| Setting | Air default | Meaning |
| --- | ---: | --- |
| `led_intensity` | `70` | CRISP LED intensity; integer greater than `1` and at most `100`. |
| `loop_gain` | `10` | Feedback responsiveness; integer from `1` to `100`. |
| `averaging` | `5` | Number of samples averaged; integer from `0` to `99`. |
| `update_rate` | `10` | CRISP trajectory update interval in ms. |
| `objective_na` | `0.9` | NA used to calculate dither steps; oil default is `1.4`. |
| `lock_range` | `0.1` | Maximum permitted lock movement in mm; positive and below `1`. |
| `user_input` | `True` | Ask before CRISP configuration and locking. |
| `min_snr`, `min_error` | `2`, `100` | Minimum acceptable calibration signal and error. |
| `pause_long`, `pause_short` | `5`, `1` | Configuration pauses in seconds. |

The CRISP settings do not by themselves enable autofocus. The experiment
panel's **Use autofocus** selection determines the focus policy for a run.

### `ConfigImageProcessor`

The defaults below are produced by
`ConfigImageProcessorFactory.default_config()`. This is the recommended way to
construct the configuration. The raw dataclass constructor requires
`cfg_delta`, `channels`, and `channels_seg`, and its `preproc_enabled` and
`roi_enabled` defaults are `True`; the factory deliberately changes both to
`False`.

| Setting | Factory default | Meaning |
| --- | --- | --- |
| `cfg_delta` | DeLTA `mothermachine` configuration | DeLTA model paths, input sizes, and processing options. Confirm that the configured model files exist before enabling processing. |
| `channels` | 450, 515, 565, and 645 nm | Channels used for reference imaging and processing. Do not include UV illumination. |
| `channels_seg` | 450, 515, 565, and 645 nm | Channels averaged for segmentation. Every entry must also occur in `channels`. |
| `preproc_enabled` | `False` | Enable image preprocessing. This is required by ROI identification and segmentation. |
| `roi_enabled` | `False` | Enable ROI identification. Preprocessing must also be enabled. |
| `roi_min_area` | `1500` | Minimum accepted ROI area in original-image pixels; `None` disables this limit. |
| `roi_max_area` | `7000` | Maximum accepted ROI area in original-image pixels; `None` disables this limit. |
| `roi_max_height` | `40` | Maximum accepted ROI height in original-image pixels; `None` disables this limit. |
| `seg_enabled` | `False` | Enable image segmentation. Preprocessing must also be enabled. |
| `track_enabled` | `False` | Enable cell tracking. Preprocessing and segmentation must also be enabled. |
| `lineage_enabled` | `False` | Enable lineage calculations. Preprocessing, segmentation, and tracking must all be enabled. |
| `tracking_setting` | `TrackingSetting.MOTHERONLY` | Select the DeLTA tracking algorithm. |
| `delta_roi_preprocess_target_size` | `(3200, 3200)` | Image size immediately before DeLTA ROI processing; this can differ from `cfg_delta.target_size_rois`. |
| `image_processing_verbosity` | `0` | Image-processing verbosity; must be a non-negative integer. |
| `refocus` | `True` | Attempt recovery after CRISP loses lock instead of stopping immediately. Detailed behavior is described below. |
| `refocus_using_software_focus` | `True` | Use a software-focus scan during lost-lock recovery. Detailed behavior is described below. |
| `refocus_on_all_positions` | `False` | Permit software-focus recovery to try recorded FoVs in turn. Detailed behavior is described below. |
| `max_refocus_trials` | `10` | Recovery-counter threshold before the run is stopped. Detailed behavior is described below. |
| `chamber_orientation` | `ChamberOrientationType.HORIZONTAL` | Orientation of the sample chambers. |

## Focus behavior

Autofocus has precedence. Disabled checkboxes retain their values in the GUI,
but values that are irrelevant to the selected mode are ignored.

| `use_autofocus` | `use_software_focus` | `use_software_focus_initialisation` | Effective behavior |
| --- | --- | --- | --- |
| `True` | either | either | CRISP autofocus. Both software-focus flags are ignored. |
| `False` | `True` | either | Scan every FoV during initialization and run software focus again after each position move. |
| `False` | `False` | `True` | Scan every FoV once during initialization, retain each resulting Z, and do not refocus on later moves. |
| `False` | `False` | `False` | Read the stage Z once at FoV initialization, assign that same Z to every FoV, and perform no automatic focus scans. |

When autofocus is disabled, stage movement does not wait for CRISP status.
Manual software-focus operations remain available independently of the
automatic experiment policy.

With CRISP enabled, moving between FoVs configured for different autofocus
channels can require a software-focus scan while CRISP is reinitialized. Lost
lock recovery can also invoke software focus according to `ConfigImageProcessor`.
Those are CRISP-management operations, not recurring software-focus mode.

### CRISP lost-lock recovery (`manage_autofocus`)

After a position move, `Automaton.manage_autofocus()` checks CRISP when hardware
autofocus is enabled. If CRISP initially reports `OUT_OF_FOCUS`, it waits 10
seconds and checks again. Recovery begins when CRISP is not locked or still
reports `OUT_OF_FOCUS`.

The four `ConfigImageProcessor` recovery settings interact as follows:

- If `refocus=False`, a lost lock stops the run without attempting recovery.
- If `refocus=True` and `refocus_using_software_focus=False`, the Automaton
  unlocks CRISP, moves to the preceding recorded FoV and Z coordinate,
  reinitializes and locks CRISP there, and then returns to the current FoV.
- If both `refocus` and `refocus_using_software_focus` are `True`, the Automaton
  returns to the stored coordinate, runs software focus, and then reinitializes
  and locks CRISP. With `refocus_on_all_positions=False`, it tries only the
  current FoV. With `True`, it can try each recorded FoV in sequence until one
  succeeds, then returns to the original FoV if necessary.

The recovery counter is reset when the strategy run starts. The current
implementation checks whether the counter is greater than `max_refocus_trials`
before incrementing it. Consequently, the default value `10` permits attempts
numbered 1 through 11 and stops on the next lost-lock event. Failed software
focus, failed CRISP reinitialization, or exceeding that threshold stops the run.

After the check, the current Z coordinate and observed lock state are recorded
for the FoV. When hardware autofocus is disabled, `manage_autofocus()` only
updates the stored Z coordinate and performs no CRISP recovery. These settings
therefore do not select recurring or initialization-only software focus; those
modes are controlled by the three experiment focus flags in the table above.

### Software focus

Software focus scans Z positions around the current stage position. `rel_range`
sets the distance scanned on either side and `step_size` sets the spacing
between planes. At each plane, the camera acquires an image using the configured
`focus_channel`, `exposure_time`, `brightness`, and `filter_wheel`. The
`cropping_box` is then extracted and the selected algorithm assigns it a
sharpness score. Higher scores mean that the algorithm considers the image
sharper, so the sampled Z position with the highest score is selected.

A focus curve is accepted only when it has one global maximum and that maximum
is not the first or last sampled plane. A maximum at a scan boundary suggests
that the real peak may be outside the scanned range. Equal highest scores at
multiple planes do not identify a unique focus position, while fewer than three
measurements or an indeterminate curve provide too little evidence. These cases
produce `BAD_FOCUS_CURVE`. The camera currently moves to the highest-scoring
sampled plane during finalization even when the curve is bad, but recurring
software focus and CRISP lost-lock recovery reject that result and shut down.
This fail-safe prevents continued acquisition at an unverified Z position and
prevents an unreliable focus coordinate from being reused on later moves.

Software focus requires visible image structure. If nothing fluoresces, it
cannot reliably determine the cell focal plane: the curve will usually be flat
or noise-driven and focus fails. Weak, noisy, or saturated fluorescence can
cause the same problem. The scoring algorithms detect sharpness rather than
recognizing cells, so they may instead focus on strong sample features such as
microfluidic channel edges when those features occur inside the scored region.

To improve a poor focus curve:

- Choose a `focus_channel` containing fluorescent objects and adjust
  `exposure_time`, `brightness`, and `filter_wheel` to obtain a visible but
  unsaturated signal.
- Set `cropping_box` around representative cells and exclude empty regions,
  channel edges, and other structures that should not determine focus.
- For a boundary maximum, start closer to the expected focal plane or increase
  `rel_range` within the stage limits. Reduce `step_size` when finer Z sampling
  is needed; this increases the number of images and scan time.
- Select an algorithm that responds to the intended features. `LAPLACIAN_VAR`,
  `SQUARED_GRAD_AVG`, and `STEEL` score spatial edges or intensity differences
  and can favor sharp channel structures. Tune `STEEL` with `rowshift_px` and
  `colshift_px`. `BANDPASS_FFT` can emphasize cell-sized structure; tune it with
  `cell_radius`, `downsample`, `order`, `saturation`, and
  `normalize_brightness`.

Inspect the **Focus Curves** display after initialization when it is available.
A boundary peak, flat curve, noisy curve, or competing peaks indicates whether
the scan range, fluorescence settings, crop, or scoring algorithm should be
changed.

## Strategies and commands

A strategy subclasses `AbstractStrategy` and implements:

- `_initialise()`: reset strategy-owned state and return the first command list.
- `_callback(fov_id, data, errors)`: inspect the previously executed commands
  and return the next list. `callback_counter` starts at zero and is incremented
  after each callback.
- `finalise()`: return a final command list. No callback follows it.

Commands in a returned list run sequentially. In the next callback, `data`
contains those same `AutomatonCommand` objects. Inspect `command_type`,
`command_data`, `fov_id`, and `command_execution_time` rather than relying only
on list positions.

```python
from evomachine.evotypes import AutomatonCommandType, LEDType
from evomachine.strategy import AbstractStrategy


class ImagingStrategy(AbstractStrategy):
    """Acquire and save one image from the current FoV, then stop."""

    def _initialise(self):
        """Return the first acquisition commands."""
        return [
            self.command_factory.command_move(fov_id=0),
            self.command_factory.command_image(
                channels=[LEDType.LED_565_NM],
                exposure_time=100,
                segment=False,
                save=True,
                filename_suffix="_example",
            ),
        ]

    def _callback(self, fov_id, data, errors):
        """Inspect acquisition results and stop the experiment."""
        image_commands = [
            command for command in data
            if command.command_type == AutomatonCommandType.IMAGE
        ]
        if image_commands:
            image_data = image_commands[0].command_data["img"]
            _ = image_data  # Use the returned normalized image as needed.
        return [self.command_factory.command_stop()]

    def finalise(self):
        """Return no additional commands during finalization."""
        return []
```

Call `strategy.test_strategy()` before launching, as the supplied launchers do.
It is a lightweight structural check and does not replace a test-camera or
hardware validation run.

### Strategy-facing command reference

| Factory method | Purpose and result in `command_data` |
| --- | --- |
| `command_move(fov_id)` | Move to an FoV; `-1` selects the next FoV and `None` does not move. Automatic focus policy is applied. |
| `command_image(...)` | Acquire channels, optionally process/segment and save them; returns a dictionary containing `img` and, when configured, segmentation/cell data. |
| `command_z_stack(...)` | Acquire and save a multi-channel stack around current Z; returns `ZStackResult`, never image arrays. |
| `command_project(...)` | Project a supplied DMD pattern with one LED for a duration in seconds. |
| `command_project_roi(...)` | Build and project a pattern from known ROI boxes. It does not move to `pos_id`. |
| `command_wait(...)` | Wait in seconds, optionally with live mode and illumination; no data result. |
| `command_live_mode(status)` | Enable or disable camera live mode. |
| `command_save_state(suffix)` | Save current Automaton state. |
| `command_stop()` | Stop the run; no later strategy callback occurs. |
| `command_magnet(...)` | Set magnet enable state, current, or field when supported. |
| `command_calibrate_magnet()` | Calibrate the supported magnet hardware. |
| `command_calibrate_hall(hall_id)` | Calibrate a Hall sensor. |
| `command_read_hall(hall_id)` | Read a Hall sensor; the current implementation logs the value. |

`command_focus_data`, `command_fov_data`, `command_autofocus`,
`command_info_text`, `command_ref_data`, `command_roi_data`, and
`command_seg_data` are Automaton/GUI communication helpers, not strategy
commands.

### Image acquisition

```python
command = self.command_factory.command_image(
    channels=[LEDType.LED_450_NM, LEDType.LED_565_NM],
    exposure_time=100,             # ms
    segment=False,
    brightness=[10, 20],           # one value per channel, 0-100
    save=True,
    filename_suffix="_measurement",
    filter_wheel=None,             # leave the current setting unchanged
)
```

An `IMAGE` command updates the experiment image buffers and runs the configured
preprocessing. Setting `segment=True` requests segmentation/tracking and
requires preprocessing plus every configured segmentation channel in
`channels`. A DMD `pattern` can be supplied; patterned images are returned but
do not go through the normal position-processing path.

The returned `command_data["img"]` contains normalized/rotated images. Saving
uses the corresponding raw camera frames. A scalar integer brightness is shared
by all channels; per-channel brightness and filter-wheel lists must match
`channels`.

### Z-stack acquisition

```python
command = self.command_factory.command_z_stack(
    channels=[LEDType.LED_450_NM, LEDType.LED_565_NM],
    exposure_time=100,             # ms
    brightness=[10, 20],
    rel_range=20,                  # ±2.0 µm
    step_size=5,                   # 0.5 µm
    filename_suffix="_sample",
)
```

`rel_range` and `step_size` use 0.1 µm stage units. If either is `None`, its
value comes from `ConfigFocus`. Planes are generated symmetrically from the
largest negative in-range step through zero to the largest positive in-range
step. Zero is always included; a range that is not divisible by the step stops
at the nearest in-range multiple.

Before acquisition, the complete Z range and output directory are validated.
The command then:

1. records the current stage Z and disables live mode;
2. unlocks CRISP if autofocus is active;
3. visits every Z plane and immediately saves every requested channel;
4. disables illumination, clears the DMD, restores the recorded Z, and only
   then relocks CRISP.

The command attempts to restore the starting Z even when autofocus is off or
acquisition fails; a restoration failure is treated as a command error. Acquired
stack images do not enter experiment image buffers, reference frames,
segmentation, tracking, or other image processing.

On success, `command_data` is an immutable `ZStackResult`:

```python
result = z_stack_command.command_data
center_z = result.center_z
plane_coordinates = result.z_coordinates
saved_paths = result.filenames
```

Coordinates are in acquisition order. Filenames are ordered by plane and then
channel. The number of files is therefore normally `planes × channels`.

## Saved files

Images are written to `ConfigCamera.path_to_save`. The directory must already
exist. Hardware-camera filenames include channel, position, stored coordinates,
filter wheel, and timestamp. An `IMAGE` command then appends the strategy
callback counter and optional `filename_suffix`.

Every Z-stack filename also ends in a signed offset such as
`_zstack_dz-0010`, `_zstack_dz+0000`, or `_zstack_dz+0010` before `.tiff`.
This `dz` value is the authoritative plane offset. The base filename's Z value
may describe the stored FoV coordinate rather than the temporary stack plane.

## DMD and DMD calibration

The digital micromirror device (DMD) controls where LED illumination reaches
the sample. White pattern pixels project illumination and black pixels block
it. Strategy commands that accept a pattern expect a two-dimensional `uint8`
array in DMD dimensions. ROI patterns are instead constructed in camera
coordinates and warped into DMD coordinates using the calibration homography.

In the DMD panel, **Initialise** starts the DMD and loads its configured
calibration. **NONE** and **FULL** display blank and fully illuminated patterns;
**IMG**, **CHECKB**, and **CALIMG** display a loaded image or test patterns.
**Calibrate** runs calibration, while **Show Loaded Calibration** plots the
matched DMD and camera points.

Calibration projects a grid of points and records the brightest corresponding
camera position for each visible point. The active `DMDCalibConfigType` sets the
LED channel, brightness, exposure, point size, grid spacing and bounds, and the
delay between projections. Accepted DMD/camera point pairs are saved as a
dated, versioned pickle file; forward and inverse homographies are calculated
from them and loaded immediately for the current session.

Use a fluorescent calibration sample that gives clear, unsaturated spots, and
inspect **Show Loaded Calibration** for sensible coverage across both coordinate
systems. Recalibrate after changing the camera-to-DMD alignment, objective, or
relevant optical path. At startup the configured calibration pickle is loaded,
so check the calibration filename in the log after recalibration to confirm
that later sessions use the intended file.

## Operating checklist and troubleshooting

- Confirm that `path_to_save` exists and has sufficient free space. Z stacks
  create one TIFF per plane and channel.
- Confirm the objective configuration and stage limits before initialization.
  Software-focus and Z-stack ranges are rejected if they leave those limits.
- Choose a focus channel and exposure that reveal useful structure without
  saturation. Review the focus cropping box whenever the chip or camera setup
  changes.
- If software focus gives a boundary maximum or irregular curve, inspect the
  crop, illumination, range, step size, and sample contrast before widening the
  scan.
- If CRISP is selected, verify its calibration, signal-to-noise ratio, error,
  and lock state. The software-focus checkboxes do not replace CRISP while
  autofocus is enabled.
- If processing fails to initialize, check the DeLTA model paths and the
  preprocessing → segmentation → tracking → lineage dependency chain.
- Use `TestCamera` with representative saved frames to exercise GUI and strategy
  flow without controlling the microscope. It cannot validate physical stage,
  CRISP, illumination, filter-wheel, or DMD behavior.
- Review logs for the first exception. Automaton command errors normally stop
  the run to protect experiment and hardware state.

## Code map

| Module | Responsibility |
| --- | --- |
| [`evomachine/config.py`](../evomachine/config.py) | Configuration dataclasses, factories, logging, and hardware-selection constants. |
| [`evomachine/evotypes.py`](../evomachine/evotypes.py) | Command, focus, LED, filter-wheel, image, and objective enums/types. |
| [`evomachine/strategy.py`](../evomachine/strategy.py) | Strategy lifecycle and basic example strategy. |
| [`evomachine/commands.py`](../evomachine/commands.py) | `AutomatonCommand`, `ZStackResult`, and `CommandFactory`. |
| [`evomachine/automaton.py`](../evomachine/automaton.py) | Initialization, focus policy, command execution, processing, and recovery. |
| [`evomachine/acquisition.py`](../evomachine/acquisition.py) | Camera abstractions and implementations, stage movement, acquisition, focusing, and saving. |
| [`evomachine/software_focus.py`](../evomachine/software_focus.py) | Focus-score algorithms. |
| [`evomachine/guidir/experiments.py`](../evomachine/guidir/experiments.py) | Experiment initialization controls and focus-mode selection. |
