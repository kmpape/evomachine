# Focus workflows

The right-hand workflow area has Acquisition, Strategy and Focus tabs. Focus
contains the manual CRISP controls, manual software autofocus, and manual
reference-profile autofocus. The stage, camera and illumination hardware remain
shared; the manual software/reference configurations and their results are
independent of strategy configurations and reference profiles.

## Manual reference-profile experiments

1. Initialise devices and manually find the desired focal plane using the camera
   preview and stage controls.
2. In Focus → Reference Profile Focus, configure the imaging, scoring and scan
   settings. Whole images are always scored. The default image metric is squared
   gradient average; all existing metrics remain selectable.
3. Choose **Calibrate at current Z**. The initial scan defaults to ±40 µm at
   1 µm steps. The starting Z defines reference zero, even if the score curve has
   a higher peak elsewhere. The stage returns to the starting Z after capture.
4. Inspect the measured points and fitted spline. The default lambda is zero;
   enable automatic smoothing to use GCV. **Refit cached reference** updates the fit without
   acquiring more images. Refitting requires acceptance again.
5. **Accept reference**, then use **Test match (no correction)** to acquire
   samples and inspect correlation without applying the estimated correction.
   **Run focus** applies a correction only when the match passes its checks.
   The latest sample scores appear as pink crosses on the reference. The plot
   controls can align crosses to the candidate shift and rescale scores using a
   positive gain and baseline fit, so brightness differences do not obscure shape.
   Turn these off to inspect raw scores at offsets relative to the scan centre.
   Plot alignment never moves the stage or changes acceptance.
6. The plot toolbar can save the figure. Acquisition settings can save raw focus
   images through the existing output directory/file manager.

Manual experiments do not replace the strategy's references or settings. Manual
hardware actions are disabled during strategy execution. Read-only inspection
remains available. Stop/Cancel is cooperative between acquisition steps.

Opening Configure or editing sampling, matching, saving, or the next calibration's
range/step preserves the accepted reference. A smoothing/metric change refits
cached calibration images and asks for acceptance again without reacquisition.
Exposure, illumination, filter, DMD, or camera-normalisation changes retain the
old curve for inspection but require a new capture before it can be used again.

Image sampling and the Pearson search are separate. **Generated positions** uses
an outer range and image step, with an optional denser central range/step. The
preview shows the exact positions and image count. **Custom positions** preserves
irregular lists. The central region is optional; broader coverage of the curve's
slopes is still needed to distinguish shifts. Matching range/step controls only
the candidate corrections evaluated against the spline; it acquires no images.

## Strategy configuration

In Strategy → FoV Setup, **Global Focus Settings** and **Configure Selected FoV**
use the same mode-specific editor. Selecting a mode replaces the whole settings
page. Values are retained when switching between the pages.

- **Standard** retains CRISP lock/recovery policies, optional software focus on
  arrival, timing, retry limits, image metrics and acquisition parameters.
- **Reference profile** exposes calibration range/step, smoothing, sparse/recovery
  offsets, both shift searches and acceptance thresholds, image metric parameters,
  illumination/exposure/filter settings and acquisition options.
- A FoV can inherit the global mode, choose a different mode, or override the
  complete settings for its selected mode. An override uses its own configuration
  snapshot; subsequent edits to global settings do not change that snapshot.

Apply the FoV table/settings using **Initialise**, then Set Strategy and Start.
GUI-specified per-FoV policies survive Set Strategy. Future explicit strategy
commands can still update a FoV policy. Existing programmatic callers default to
`FocusMode.STANDARD`.

### Initial focal plane

In reference mode, first use of a reference context moves to that FoV's saved XYZ
and captures a strategy-owned reference around its saved Z. The strategy waits
for **Accept and continue** in the curve-review dialog before imaging proceeds.
**Reject / stop** stops execution. The manual Focus-tab reference is not imported.
The initial shared reference is reused by FoVs inheriting its settings. A FoV
with reference-settings overrides is calibrated separately at its saved Z.
Starting a fresh GUI strategy run prepares fresh strategy references.

Standard mode has an optional **Initialise from first FoV Z** setting. It positions
at saved XYZ before configuring CRISP. If software focus on arrival is also
enabled for that FoV, software focus may refine/change that Z before CRISP is
calibrated and locked. To preserve the manually chosen first plane, disable
software focus on arrival for that FoV using its override.

### Subsequent movement and recovery

Reference-mode arrivals move XY while preserving current Z. Sparse sampling is
followed by denser sampling when matching is rejected. The defaults are nine
sparse images at −20…+20 µm in 5 µm increments. Recovery uses the same outer
positions plus a ±5 µm centre at 0.5 µm spacing (27 images total). Both shift
searches cover ±20 µm at 0.1 µm increments, with minimum correlation 0.9.
Sampling and matching settings remain provisional and editable.

The whole shifted sample set must fit inside the measured reference: a ±40 µm
reference with ±20 µm samples supports shifts up to ±20 µm. Unsupported candidates
are not extrapolated. Errors report actual reference/sample ranges, candidate
counts, correlation thresholds, competing peaks, or boundary failures.

Status identifies sparse success, recovery success, or both attempts failing.
The correlation plot retains both attempts, including rejected ones. Low score
variation still permits diagnostic correlations, but blocks correction. Constant
scores have undefined Pearson correlation; the graph explains this instead of
remaining silently empty. Sample crosses remain visible on rejected attempts.

If both attempts fail, the FoV is skipped and the existing imaging skip path
suppresses its images. Device failures propagate instead of becoming bad matches.
Failed matches do not overwrite the stored accepted focal Z.

**Allow highest-peak reset** is off by default. When enabled, there is one final
attempt: acquire a dense scan, require a unique interior maximum with sufficient
score variation, then capture a new reference centred on that peak. This explicitly
changes the definition of the focal plane. The replacement belongs only to the
failed FoV; the shared reference and other FoVs remain unchanged. If rebuilding
also fails, skip the FoV. Bounds are checked before scanning, and cancellation
never applies a later correction.

CRISP is unlocked/disabled during reference focusing. Standard software Z scans
also run unlocked; CRISP is recalibrated at the resulting plane before relocking.
FoV coordinates retain full precision even when their table display is rounded.

## Developer notes

- `focus_registration.py` is the numerical layer: image scoring, smoothing spline,
  Pearson shift search and plotting support. `referencefocus.py` owns acquisition,
  explicit reference acceptance, refitting and optional peak rebuilding.
- `FocusNavigator` resolves global/per-FoV policies while keeping its public
  `move()` interface and `FocusNavigatorFovRecord` result compatible. Runtime
  builders provide the reference component alongside standard focus.
- GUI widgets stay on the Qt thread. `gui/focus.py` contains configuration transport,
  isolated manual state and strategy review state; no hardware objects cross JSON
  RPC. Curves are finite numeric lists; unsupported correlation points are null.
- Manual acquisition uses `GuiOperationManager`. During strategy-owned scans the
  acquisition manager services GUI requests between completed planes. Acceptance
  waits also service requests, so review, status polling and Stop remain usable.
  Only the current review ID can be accepted. Acceptance mutates review state;
  it never executes a hardware operation on the socket thread.
- Add future modes as separate editor pages and explicit navigator workflows.
  Do not reconnect manual status/configuration signals to strategy settings.

Focused checks:

```bash
EVOMACHINE_GUI_RUN_QT_TESTS=1 QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest \
  tests/test_focus_registration.py tests/test_referencefocus.py tests/test_focusnavigator.py \
  tests/gui/test_focus.py tests/gui/test_panels.py -q
```

Run the GUI suite and virtual launcher as described in [GUI development](gui.md).
Synthetic tests establish software behaviour; microscope data is still needed to
validate profile-shape assumptions and tune sampling and confidence thresholds.
