# Image-processing integration

[Documentation](../README.md) › [Developer guide](README.md) › Image-processing integration

This page describes the current integration boundary. Detailed model preparation,
algorithm validation, and biological interpretation are intentionally reserved
for future documentation.

## Current data flow

```text
image command
  -> FrameAcquisitionManager
  -> Frame
  -> DeltaProcessor (when detection/segmentation is requested)
  -> per-FoV PositionRT + typed MicroscopyState
  -> strategy observations / ROI collections / projection selection
```

`ImageProcessorConfig` selects channels and enables preprocessing, ROI detection,
segmentation, tracking, and lineage stages. The nested `delta.config.Config` owns
DeLTA model paths and algorithm-specific settings. `DeltaProcessor` performs the
application integration and updates EvoMachine-owned state.

## Detection, segmentation, and identity

`detect_rois=True` detects trench regions and replaces the FoV's ROI identities.
It resets tracking, measurements, treatment history, and pending selections for
that FoV. `segment=True` requires existing/detected ROIs, preprocessing, and the
configured segmentation channels. Tracking and lineage require their preceding
stages.

The current integration accepts one segmentation-channel plane for a processed
image. Stable trench records expose optional measurements with explicit validity
flags; missing measurements are not converted to zero or NaN.

DeLTA's built-in growth rate is per frame and assumes equal spacing. A strategy
may use actual acquisition times for another proxy, but that does not by itself
create a validated biological estimator.

## Failure and invalidation

Failed, skipped, or unprocessed acquisitions invalidate current biological
measurements where appropriate. No detected trench and no mother cell are valid
outcomes. Processing exceptions invalidate affected state and abort when lineage
may have partially advanced; the application does not replay a partially mutated
pipeline automatically.

## Geometry and projection boundary

The current ROI/projection workflow assumes horizontal mother-machine geometry
and preserves camera coordinates for DMD transformation. Automatic rotation and
drift correction are disabled in this path. Verify trench ordering, mother ends,
camera/DMD calibration, and stage registration before projection.

## Future documentation

The following sections are intentionally reserved:

- preparing and versioning ROI, segmentation, and tracking models;
- image normalisation and preprocessing algorithms;
- representative datasets and quantitative validation;
- ROI geometry, rotation, and drift correction;
- lineage quality control and biological measurement semantics;
- performance, GPU memory, batching, and model-loading diagnostics;
- calibration and acceptance criteria for treatment/projection experiments.

Until these are documented and validated, source validators, DeLTA documentation,
model-specific records, and experiment protocols remain authoritative.

---

← [AutoStrat](autostrat.md) | [Developer guide](README.md) | [Testing and releases](testing.md) →
