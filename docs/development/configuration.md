# Configuration conventions and reference

[Documentation](../README.md) › [Developer guide](README.md) › Configuration reference

This reference covers objects intentionally constructed or updated by operators,
strategies, launchers, and backend developers. Runtime results, GUI transport
payloads, and internal state records are documented with their owning subsystem
rather than treated as user configuration.

## Conventions

### `EvoConfig`

Most application configs inherit Pydantic `EvoConfig`. Unknown fields are
forbidden and arbitrary external types are permitted where explicitly declared.
Construction performs validation. `updated(**fields)` returns a new validated
same-type object and rejects unknown fields; `update_from_mapping(mapping)` does
the same for GUI-style mappings. `copy()` returns a shallow same-type copy.

Do not mutate nested fields and assume validation reruns. Construct an updated
object and replace it at the owning component's boundary.

### Dataclass configs

Some established configs are typed dataclasses. Their `__post_init__` methods
validate and may normalise values. Frozen dataclasses use `dataclasses.replace()`;
mutable configs should still be replaced atomically when they govern active
runtime behavior. Result and state dataclasses are not configuration simply
because they use the same Python feature.

### Defaults, lifetime, and GUI column

“Required” means the caller or a factory must supply a value. Factory presets may
set more specific values than the class declaration. The **Editable** column uses:

- **GUI**: exposed through a current GUI control;
- **startup**: set by TOML, environment, or runtime construction before launch;
- **strategy**: commonly constructed by strategy/application code;
- **internal**: filled by acquisition/runtime rather than manually edited.

GUI changes are session-only unless a section explicitly describes persistence.

## Experiment configuration files

`ExperimentConfig` (`evomachine/experiment_config.py`) is the shareable,
versioned description of how an experiment is run. It is stored as a TOML file
and is intended to be loaded, saved, and reused between users and microscopes.
`experiments/example.toml` is a commented example.

An experiment file deliberately excludes:

- microscope hardware settings such as serial ports, camera size, objective, and
  stage limits, which belong to the machine configuration;
- FoV positions, which are selected in the GUI before every initialisation;
- runtime state such as counters, focus results, and reference profiles.

Experiment configs are frozen Pydantic models. Construction and loading run full
validation, unknown keys are rejected at every level, and changes are made with
`updated(**fields)`, which returns a new validated object. Keys omitted from a
file take their defaults from the runtime classes they describe, so each default
is defined in one place.

| API | Behavior |
| --- | --- |
| `ExperimentConfig.load(path)` | Read, version-check, and validate a TOML file. |
| `ExperimentConfig.from_toml(text, source=...)` | The same for TOML text; `source` names the file in error messages. |
| `config.to_toml()` | Return TOML text that loads back into an equal config. |
| `config.save(path)` | Write via a temporary file so an interrupted save never leaves a partial file. |

Every read, parse, version, and validation failure raises
`ExperimentConfigError` with the file name and, where available, the dotted
location of each problem (for example `output.filename_pattern`). Saved files
contain no comments.

### Schema version

Every file must start with `schema_version`. The current version is
`SCHEMA_VERSION = 1`. A file with a newer version is rejected with a request to
update EvoMachine. When the version is increased, older supported versions
should be upgraded in `_check_schema_version()` before validation.

### Sections

| Key or section | Type | Default | Meaning |
| --- | --- | --- | --- |
| `schema_version` | `int` | required in files | File-format version. |
| `name` | `str` | `""` | Short experiment name. |
| `description` | `str` | `""` | Free-text description. |
| `[acquisition]` | `FrameAcquisitionSettings` | class defaults | Default behavior of every image acquisition; see below. |
| `[output]` | `OutputConfig` | required | Where saved images go and how they are named. |

`OutputConfig` fields:

| Field | Type/default | Meaning and validation |
| --- | --- | --- |
| `directory` | `Path`, required | Full output path. `~` expands to the running user's home directory. A relative path is accepted with a logged warning because it resolves against the process working directory. |
| `filename_pattern` | `FileNameConfig` default | Validated by `FileNameConfig`; see [`FileNameConfig`](#filenameconfig). |

`OutputConfig.to_file_name_config(time_config=None)` returns a new
`FileNameConfig` with `~` expanded, for constructing a `FileManager`.

## Application paths, time, and logging

### `EvoSettings`

Resolved once when `evomachine.settings` is imported.

| Field | Type/default | Meaning and validation | Editable |
| --- | --- | --- | --- |
| `data_dir` | `Path`, required | Root for images, logs, and GUI-saved AutoStrat files. Machine TOML is applied before user TOML. | startup |
| `source_files` | `tuple[Path, ...] = ()` | Existing configuration files, in applied order. Informational. | internal |

The only TOML key is `[paths].data_dir`. `~` and environment variables expand for
the running user. Unknown sections/keys and empty/non-string values fail startup.

### `TimeConfig`

| Field | Type/default | Meaning and validation | Editable |
| --- | --- | --- | --- |
| `timezone_name` | `str = "Europe/London"` | IANA timezone used by generated/displayed timestamps; must be known to `zoneinfo`. | startup/code |

### `EvoLoggingConfig`

| Field | Type/default | Meaning | Editable |
| --- | --- | --- | --- |
| `level` | `int = INFO` | Core application logger level. | startup/code |
| `gui_level` | `int = INFO` | GUI logger level. | startup/code |
| `peripheral_level` | `int = DEBUG` | Binding-neutral peripheral logger level. | startup/code |
| `binding_level` | `int = DEBUG` | External binding package logger level. | startup/code |
| `binding_logger_names` | `tuple[str, ...] = ("syncboard", "asitiger")` | Non-empty package logger names configured at binding level. | startup/code |
| `time_config` | `TimeConfig()` | Timezone used by log formatting. | startup/code |

Levels must be integers (not booleans); logger names must be non-empty strings.

## Hardware GUI runtime settings

`HardwareGuiRuntimeSettings.from_env()` builds the default microscope deployment.
Fields are startup-only. An explicit serial port wins; otherwise one matching HWID
must be found. The three stage bound pairs must be finite, ordered, and include
startup zero.

| Field | Default | Meaning / environment suffix |
| --- | --- | --- |
| `syncboard_port` | `None` | Exact SyncBoard port; `SYNCBOARD_PORT`. |
| `syncboard_hwid` | `"16C0:0483"` | USB HWID fragment; `SYNCBOARD_HWID`. |
| `tiger_port` | `None` | Exact ASI Tiger port; `TIGER_PORT`. |
| `tiger_hwid` | `"10C4:EA60"` | Tiger HWID fragment; `TIGER_HWID`. |
| `kwr103_port` | `None` | Exact KWR103 port; `KWR103_PORT`. |
| `kwr103_hwid` | `"0416:5011"` | KWR103 HWID fragment; `KWR103_HWID`. |
| `camera_device` | `"Camera-1"` | Micro-Manager camera device; `MICROMANAGER_CAMERA_DEVICE`. |
| `readout_mode_property` | `"Port"` | Micro-Manager readout property; `MICROMANAGER_READOUT_PROPERTY`. |
| `camera_width`, `camera_height` | `3200`, `3200` | Positive sensor dimensions; `CAMERA_WIDTH`, `CAMERA_HEIGHT`. |
| `camera_exposure_ms` | `200.0` | Default exposure in milliseconds; `CAMERA_EXPOSURE_MS`. |
| `stage_fov_step_size` | `100.0` | FoV step in µm; `STAGE_FOV_STEP_SIZE`. |
| `stage_min_x_um`, `stage_max_x_um` | `-8000`, `8000` | X travel bounds relative to startup zero. |
| `stage_min_y_um`, `stage_max_y_um` | `-19000`, `19000` | Y travel bounds relative to startup zero. |
| `stage_min_z_um`, `stage_max_z_um` | `-1000`, `1000` | Z travel bounds relative to startup zero. |
| `filter_card_address` | `8` | Tiger filter-wheel card address; `FILTER_CARD_ADDRESS`. |
| `dmd_width`, `dmd_height` | `2716`, `1600` | DMD pattern dimensions; `DMD_WIDTH`, `DMD_HEIGHT`. |
| `dmd_calibration_file` | deployment calibration path | Camera/DMD transform file; `DMD_CALIBRATION_FILE`. |
| `output_directory` | resolved `DATA_DIR` | Initial acquisition directory; `OUTPUT_DIR`. |
| `use_dmd` | `True` | Construct the DMD path; `USE_DMD`. |
| `use_kwr103` | `True` | Construct KWR103 overhead illumination; `USE_KWR103`. |

All names above use the `EVOMACHINE_GUI_` prefix. Boolean values recognise
`1`, `true`, `yes`, and `on` (case-insensitive) as true.

## Acquisition, frames, and files

### `FrameAcquisitionSettings`

All fields are booleans and editable in the GUI acquisition configuration.
Instances are immutable; use `updated(**fields)` for a changed copy.

| Field | Default | Effect |
| --- | --- | --- |
| `save` | `False` | Save planes through the active `FileManager`. |
| `normalise` | `False` | Return camera-normalised data where supported. |
| `illuminate_dmd` | `True` | Display DMD illumination before capture when available. |
| `clear_dmd_after` | `False` | Blank the DMD after the acquisition call. |
| `restore_leds_after` | `True` | Restore cached LED states after the call. |
| `disable_leds_after` | `False` | Disable all LEDs after the call. This can supersede restoration. |

Per-call settings replace manager defaults for that call; they are not merged.

### `FrameMetaData`

This is request metadata plus runtime-populated metadata for one image plane.

| Field | Type/default | Meaning and validation | Editable |
| --- | --- | --- | --- |
| `frame_id` | `int`, required | Plane ID. `FrameMetaDataFactory` assigns increasing process-local IDs unless supplied. | strategy |
| `leds` | `dict[LEDType, number] \| None`, required | Logical channels and brightness in `[0, 100]`. | GUI/strategy |
| `filter_wheel` | `FilterWheelType \| None`, required | Requested filter position. | GUI/strategy |
| `exposure` | `number \| None`, required | Exposure in milliseconds, constrained to `[1, 1000]` when set. | GUI/strategy |
| `dmd_pattern` | `np.ndarray \| None = None` | Optional illumination pattern. | GUI/strategy |
| `fov_id` | `int = UNKNOWN_FOV_ID` | Associated FoV; unknown is `-1`. | GUI/strategy |
| `coordinate` | `Coordinate \| None = None` | Stage coordinate, normally filled/resolved by runtime. | internal/strategy |
| `creation_time` | current aware time | Metadata creation time; naive inputs are interpreted in configured timezone. | internal |
| `execution_time` | `datetime \| None = None` | Actual acquisition time. | internal |
| `callback_id` | `int \| None = None` | Strategy callback identity assigned where available. | internal/strategy |
| `additional_metadata` | `dict[str, JSON value] = {}` | Saved inside TIFF metadata; never used as filename fields. | strategy |
| `acquisition_monotonic` | finite number or `None` | Capture-start monotonic clock for runtime calculations; not meaningful after reload. | internal |

`FilterWheelType.BANDPASS_405_445_514_561_640nm` identifies wheel position 0,
which contains the Chroma ZET405/445/514/561/640m multi-bandpass filter. The old
member name `FILTER` is retained as an alias for loading older configurations;
new code, GUI payloads, and saved enum metadata use the descriptive canonical
name.

### `FileNameConfig`

| Field | Type/default | Meaning and validation | Editable |
| --- | --- | --- | --- |
| `directory` | `Path \| str`, required | Output directory. Created when requested. | GUI/startup |
| `filename_pattern` | `str = "FOV{fov_id}_{timestamp}_{callback_id}"` | Non-empty Python format pattern using only `FileNameField`. | GUI/startup |
| `create_directory` | `bool = True` | Create a missing output directory. | startup/code |
| `time_config` | `TimeConfig()` | Timestamp/timezone formatting. | startup/code |

Allowed fields are `channel`, `fov_id`, `x`, `y`, `z`, `filter_wheel`,
`timestamp`, `timezone`, `frame_id`, and `callback_id`. Positional fields,
compound access, unknown names, and malformed syntax are rejected. `.tiff` is
added automatically. Unknown FoV `-1` renders as an empty string.

## Peripheral configuration

### Common `PeripheralConfig`

Every Pydantic peripheral config includes:

| Field | Type/default | Effect | Editable |
| --- | --- | --- | --- |
| `binding` | `BindingType`, required | Concrete factory implementation. | startup |
| `name` | `str \| None = None` | Optional display/logging identity. | startup |
| `check_initialised` | `bool = True` | Guard public hardware operations before initialisation. | startup |
| `check_alive` | `bool = True` | Guard operations when the device is not alive. | startup |

Disable readiness checks only for controlled virtual/test construction.

### Controller configs

| Config/field | Default | Meaning and validation |
| --- | --- | --- |
| `PeripheralControllerConfig.binding` | required | Controller binding. |
| `.name` | `""` | Human-readable identity. |
| `.initialise` | `True` | Initialise during factory construction. |
| `SerialPeripheralControllerConfig.port` | `None` | Exact port. Exactly one of `port` and `hwid` is required. |
| `.hwid` | `None` | USB HWID fragment resolved through port discovery. |
| `.close_on_shutdown` | `True` | Close the connection on controller shutdown. |
| `SocketPeripheralControllerConfig.host` | `"127.0.0.1"` | Socket host. |
| `.port` | `0` | Socket port integer. |
| `.close_on_shutdown` | `True` | Close socket resources on shutdown. |

Controller configs are startup-only and should be shared by all peripherals on
the same physical connection.

### Camera configs

| Config/field | Default | Meaning and validation |
| --- | --- | --- |
| `ImageConfigType.pxl_horiz` | required | Positive horizontal pixels. |
| `.pxl_vert` | required | Positive vertical pixels; array shape is `(vert, horiz)`. |
| `.pxl_dtype` | required | Any value accepted by `np.dtype`. |
| `ObjectiveConfigType.na` | required | Positive numerical aperture. |
| `.mag` | required | Positive integer magnification. |
| `.descr` | `"UNKNOWN OBJECTIVE"` | Optional descriptive text. |
| `CameraConfig.image` | required | `ImageConfigType`. |
| `.default_exposure_time` | `200` ms | Positive numeric default. |
| `.sensor_pixel_size_um` | `6.5` µm | Positive physical sensor pixel size. |
| `.readout_mode` | `None` | Optional `CameraReadoutMode`: Dynamic Range, Sensitivity, Speed, or Sub-Electron. |
| `.objective_config` | `None` | Optional objective used for sample pixel/FoV calculations. |

Factories supply Photometrics/DeLTA image presets and hardware, oil, and air
objective presets. Camera construction is startup-only; exposure is also editable
per acquisition through `FrameMetaData`.

### Stage, LEDs, filters, autofocus, and photodiode

| Config/field | Default | Meaning and validation |
| --- | --- | --- |
| `StageConfig.fov_step_size` | required | Positive stage-coordinate step used for FoV movement. |
| `.initial_coordinate` | `None` | Optional virtual/start coordinate. |
| `.coordinate_bounds` | `None` | Optional software movement bounds. |
| `.zero_on_initialise` | `False` | Zero the binding-supported stage during initialisation. |
| `LedConfig.available_leds` | required | Non-empty unique logical LED list. |
| `.led_to_internal` | `None` | Complete mapping to binding-native channels; a binding may provide defaults. |
| `FilterWheelConfig.available_filters` | required | Non-empty supported logical filter positions. |
| `AutofocusConfig` | common fields only | Selects the autofocus binding and readiness behavior. |
| `PhotodiodeReadingRange.minimum_reading` | required | Raw value representing 0%; must be below maximum. |
| `.maximum_reading` | required | Raw value representing 100%. |
| `PhotodiodeConfig.channel` | `8` | Non-negative binding input channel. |
| `.reading_range` | `0.0 .. 1.0` | Scaling bounds for normalised percentage output. |

These are startup configs. LED brightness, filter position, and camera exposure
are runtime GUI/strategy values. Stage bounds and device availability must not be
weakened through runtime controls.

### DMD configuration

| Config/field | Default | Meaning and validation |
| --- | --- | --- |
| `DmdConfig.width_height_DMD` | `(2716, 1600)` | Positive `(width, height)` pattern size. |
| `.width_height_CAM` | `(3200, 3200)` | Positive camera size used by transforms. |
| `.display_offset` | `(0, 0)` | Non-negative display offset. |
| `.monitor_index` | `None` | Optional non-negative display index. |
| `.calibration_file` | `None` | Optional camera/DMD calibration path. |
| `DmdShapeConfig.rectangle_row`, `.rectangle_col` | `None` | Non-negative rectangle origin. |
| `.rectangle_height`, `.rectangle_width` | `None` | Positive rectangle size. |
| `.checkerboard_box_size` | `None` | Positive checker size. |
| `.crosshair_row`, `.crosshair_col` | `None` | Non-negative crosshair centre. |
| `.crosshair_width` | `1` | Positive line width. |
| `.circle_row`, `.circle_col` | `None` | Non-negative circle centre. |
| `.circle_radius` | `None` | Positive radius. |

`DmdShapeConfig` is editable in the GUI for the selected built-in pattern. Shape
requirements also depend on the pattern (`empty`, `full`, `rectangle`,
`checkerboard`, `crosshair`, or `circle`). `DmdConfig` is startup-only.

`DmdCalibrationConfig` is used by calibration workflows:

| Field | Factory default | Meaning |
| --- | --- | --- |
| `channel` | `LED_450_NM` | One or more illumination channels. |
| `brightness` | `29` | Illumination brightness. |
| `exposure` | `100` ms | Camera exposure. |
| `line_width` | `5` | Calibration line width. |
| `step` | `150` | Pattern sampling step. |
| `delay` | `0.75` s | Delay between calibration operations. |
| `start_row`, `end_row` | `200`, `2500` | DMD row interval; ordered inside `[0, 2716)`. |
| `start_col`, `end_col` | `0`, `1599` | DMD column interval; ordered inside `[0, 1600)`. |
| `on_mothermachine` | `True` | Select mother-machine calibration behavior. |

The `thin_fluo_slide()` and `fluo_slide()` factories provide different values.
These are calibration starting points, not universally safe experimental values.

### ASI Tiger autofocus calibration

`TigerAutofocusConfig` is edited through autofocus configuration/calibration UI
or supplied at startup/per FoV.

| Field | Air preset | Validation / meaning |
| --- | --- | --- |
| `averaging` | `5` | Integer `[0, 100)`. |
| `led_intensity` | `70` | Integer `(1, 100]`. |
| `lock_range` | `0.1` | Numeric `(0, 1)`. |
| `loop_gain` | `10` | Integer `[1, 100]`. |
| `update_rate` | `10` | Non-negative integer. |
| `objective_na` | `0.9` | Numeric `(0, 10)`. |
| `min_snr` | `2` | Non-negative minimum calibration signal-to-noise ratio. |
| `min_error` | `100` | Non-negative minimum error-signal criterion. |

Use `TigerAutofocusConfigFactory.default_config()` or
`default_oil_config()` and validate the result on the installed objective.

## Focus and navigation

### `FocusNavigatorConfig`

The GUI can edit this complete session configuration. Strategy code can supply it
directly when using the Automaton programmatically.

| Field | Default | Meaning and interaction |
| --- | --- | --- |
| `use_autofocus` | `False` | Manage hardware autofocus during movement and checks. Must be false in reference-profile mode. |
| `refocus` | `True` | Attempt recovery after lock/focus loss. |
| `refocus_using_software_focus` | `True` | Run software focus before autofocus reinitialisation during recovery. |
| `refocus_on_all_fovs` | `False` | Permit software recovery across every registered FoV. |
| `max_refocus_trials` | `10` | Positive maximum recovery attempts. |
| `out_of_focus_wait_s` | `10.0` | Non-negative delay after OUT_OF_FOCUS before checking lock. |
| `post_autofocus_wait_s` | `3.0` | Non-negative delay after autofocus initialise/lock. |
| `post_move_wait_s` | `1.0` | Non-negative delay after move-time software focus. |
| `toggle_autofocus_on_channel_change` | `True` | Unlock/refocus when consecutive FoVs use different focus channels. |
| `autofocus_initialise_config` | `None` | Binding-specific calibration config, normally `TigerAutofocusConfig`. |
| `default_fov_config` | `FovConfig()` | Policy for FoVs without an override. |
| `initialise_from_first_fov_z` | `False` | Use the first FoV Z as initial focus basis where supported. |
| `focus_mode` | `FocusMode.STANDARD` | Select standard hardware/software behavior or reference-profile registration. |

### `FovConfig`

GUI per-FoV values override strategy values for the same ID.

| Field | Default | Meaning |
| --- | --- | --- |
| `lock_autofocus_during_arrival_move` | `True` | Keep hardware autofocus locked while arriving. |
| `lock_autofocus_on_fov` | `True` | Lock after arrival and requested setup. |
| `lock_autofocus_during_departure_move` | `True` | Keep lock while leaving. |
| `run_software_focus_on_arrival` | `False` | Run software focus after arrival before final lock. |
| `software_focus_config` | `None` | Optional FoV-specific scan/scoring config. |
| `autofocus_initialise_config` | `None` | Optional binding-specific FoV calibration config. |
| `focus_mode` | `None` | Optional per-FoV mode override. |
| `navigator_config` | `None` | Optional per-FoV navigation behavior. |

### `SoftwareFocusConfig`

| Field | Factory default | Meaning and validation |
| --- | --- | --- |
| `focus_frames` | one 450 nm, brightness 29, 200 ms frame | Non-empty metadata list acquired at each Z. |
| `acquisition_settings` | `FrameAcquisitionSettings()` | Capture and saving behavior shared by maximum and reference scans. |
| `scoring` | `(FocusScoringType.STEEL, SteelScoringConfig(...))` | Scoring enum paired with its required typed config. |
| `selection` | `(FocusSelectionType.MAXIMUM, MaximumFocusSelectionConfig())` | Maximum or reference-profile selection paired with its required typed config. |
| `cropping_boxes` | DeLTA box `(200, 300, 3000, 2900)` | Tuple of left/top-inclusive, right/bottom-exclusive boxes; an empty tuple scores the whole image. |

The GUI can edit software-focus acquisition, scan, algorithm, and cropping
settings. The factory values are microscope-specific starting points.

### Focus scoring and selection

`FocusScoringConfig` subclasses hold scorer-specific parameters: squared-gradient
threshold; Steel row/column shifts and normalization; or bandpass FFT radius,
downsampling, order, saturation and brightness normalization. Laplacian variance
has no parameters. The scoring enum and concrete config type must agree.

`MaximumFocusSelectionConfig` contains `rel_range` (default `50` µm) and
`step_size` (default `5` µm). `ReferenceFocusSelectionConfig` contains the
reference-only settings below. Both policies use the same focus frames,
acquisition settings, scorer, and crops.

### Reference-profile selection

`ReferenceFocusSelectionConfig.sampling_offsets()` produces symmetric image offsets
for defaults and the GUI editor:

| Field | Default | Meaning and validation |
| --- | --- | --- |
| `half_range` | `20.0` µm | Positive outer half-range. |
| `step` | `5.0` µm | Positive outer acquisition step. |
| `central_range` | `None` | Optional positive dense centre no larger than half-range. |
| `central_step` | `0.5` µm | Positive dense step, smaller than outer step when used. |

The result includes both endpoints and zero, must contain at least five positions,
and is limited to prevent thousands of captures.

The flat `sparse_*` and `recovery_*` fields control two independent numerical
matching policies, not image acquisition:

| Field | Default | Meaning and validation |
| --- | --- | --- |
| `<prefix>_shift_min`, `<prefix>_shift_max` | `-20.0`, `20.0` µm | Ordered finite candidate range. |
| `<prefix>_shift_step` | `0.1` µm | Positive grid step, at most half the span. |
| `<prefix>_min_correlation` | `0.90` | Accepted Pearson correlation in `[-1, 1]`. |
| `<prefix>_min_peak_margin` | `0.05` | Required separation from another peak, `[0, 2]`. |
| `<prefix>_boundary_margin` | `1.0` µm | Reject candidates too near a search boundary. |
| `<prefix>_min_relative_variation` | `0.01` | Non-negative score-variation confidence threshold. |

`ReferenceFocusSelectionConfig` fields are GUI/session or strategy editable:

| Field | Default | Meaning |
| --- | --- | --- |
| `calibration_range` | `40.0` µm | Calibration half-range: images span `-40` to `+40` µm; positive and sufficient for five planes. |
| `calibration_step` | `1.0` µm | Regular spacing used to measure the reference profile. |
| `sample_offsets` | generated `±20`, step `5` | Increasing finite routine-scan offsets. |
| `recovery_offsets` | dense-centre sampling offsets | Wider recovery sampling offsets. |
| `sparse_*` | `-20..20`, step `0.1` plus thresholds above | Sparse shift matching fields. |
| `recovery_*` | `-20..20`, step `0.1` plus thresholds above | Recovery shift matching fields. |
| `smoothing` | `0.0` | Non-negative spline lambda; `None` requests GCV. |
| `allow_peak_reset` | `False` | Permit the workflow's explicit reference-peak reset behavior. |
| `share_reference_across_fovs` | `False` | Reuse one measured reference curve across FoVs; by default each FoV acquires its own curve. |

Calibration remains a candidate until reviewed and explicitly accepted. Editing
scan/search settings does not implicitly accept a reference.

Calibration uses a range and step because its grid is always regular and
symmetric. Sparse and recovery matching support irregular and mixed-density
sampling, so their resolved positions are stored as explicit offset tuples. The
calibration half-range must cover the largest absolute acquisition offset plus
the largest absolute candidate shift; matching does not extrapolate the spline.

## Image-processing configuration

`ImageProcessorConfigFactory.default_config()` supplies a DeLTA mother-machine
config and local model paths. Model availability is deployment-specific.

| Field | Class default | Meaning and constraints |
| --- | --- | --- |
| `cfg_delta` | required | `delta.config.Config` with model paths and algorithm settings. |
| `channels` | required | Non-empty acquired LED channels; cannot contain `NO_LED`. |
| `channels_seg` | required | Non-empty segmentation channels, all present in `channels`. |
| `preproc_enabled` | `True` | Enable preprocessing. Factory currently sets false. |
| `roi_enabled` | `True` | Enable ROI detection. Factory currently sets false. |
| `roi_min_area` | `1500` px² or `None` | Minimum detected ROI area. |
| `roi_max_area` | `7000` px² or `None` | Maximum detected ROI area. |
| `roi_max_height` | `40` px or `None` | Maximum detected ROI height. |
| `seg_enabled` | `False` | Enable cell segmentation; requires preprocessing. |
| `track_enabled` | `False` | Enable tracking; requires preprocessing and segmentation. |
| `lineage_enabled` | `False` | Enable lineage; also requires tracking. |
| `tracking_setting` | `MOTHERONLY` | DeLTA tracking mode. |
| `delta_roi_preprocess_target_size` | `(3200, 3200)` | Whole-frame target size for ROI preprocessing. |
| `image_processing_verbosity` | `0` | Non-negative processing verbosity. |
| `refocus` | `True` | Legacy processing-related focus recovery flag. |
| `refocus_using_software_focus` | `True` | Legacy software-focus recovery selection. |
| `refocus_on_all_fovs` | `False` | Legacy all-FoV recovery selection. |
| `max_refocus_trials` | `10` | Positive when refocus is enabled. |
| `chamber_orientation` | `HORIZONTAL` | Mother-machine chamber orientation assumption. |

The runtime/strategy setup owns this config; only selected behavior is surfaced
in the GUI. Prefer `updated()` rather than mutating fields after construction.

## AutoStrat model configuration

`AutoStratModelConfig` is session-only and configured at GUI startup.

| Field | Default | Meaning and validation |
| --- | --- | --- |
| `provider` | required | `openai-compatible` or `anthropic`. |
| `model` | required | Non-empty model identifier without provider prefix. |
| `api_key` | `""` | Secret held in backend memory; empty disables generation. Never returned in public status. |
| `base_url` | `None` | Optional valid HTTP(S) endpoint. |
| `preset_id` | `"custom"` | Known preset ID only while all preset values still match. |

The Robin Office 2 preset selects the openAI-compatible provider, `qwen3.6`, and
its configured endpoint. Saved DSL editing, validation, and execution do not need
an API key.

## Virtual image-source configuration

`VirtualImageSource` is public session configuration for a virtual camera.

| Field | Default | Meaning and validation |
| --- | --- | --- |
| `mode` | `RANDOM` | `RANDOM` or `FOLDER`. |
| `directory` | `None` | Required folder path in folder mode. |
| `filename_glob` | `"*.tiff"` | Non-empty relative glob; absolute patterns are rejected. |
| `usable_files` | `()` | Naturally sorted validated snapshot; produced by `from_folder()`. |
| `skipped_files` | `()` | Matched paths paired with rejection reasons. |
| `next_index` | `0` | Index into usable files; resets on reconfiguration and wraps after loading. |

The GUI edits mode, directory, and glob. Snapshot and index fields are runtime
status. Folder files must be readable 2D `3200 × 3200 uint16` TIFF images.

---

← [Architecture](architecture.md) | [Developer guide](README.md) | [Bindings](bindings.md) →
