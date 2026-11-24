# Virtual mode and recorded images

[Documentation](../README.md) › [User guide](README.md) › Virtual mode and recorded images

Virtual mode supports GUI development, command-flow tests, and early strategy
tests without microscope hardware.

```bash
uv run python scripts/launch_virtual_gui.py
```

The runtime supplies virtual stage, camera, LEDs, filters, autofocus, photodiode,
and DMD bindings. Its camera uses `3200 × 3200 uint16` images.

## Camera image sources

Open **Acquisition Configuration**. In virtual mode it includes a
**Virtual Image Source** tab:

- **Random** returns deterministic pseudo-random images. It tests acquisition and
  transport, not meaningful focus or segmentation.
- **Folder** snapshots TIFF files selected by a folder and relative glob such as
  `P0_*.tiff`. Each capture returns the next file in natural order and wraps.

Folder images must be two-dimensional `3200 × 3200 uint16` TIFF files. Files with
the wrong type, dimensions, stack depth, or unreadable content are reported as
skipped. An update is rejected if its directory is invalid, nothing matches, or
no usable files remain; the old source stays active. Reapply the configuration to
include newly added files.

Selection deliberately ignores FoV, channel, exposure, and Z. Every camera
capture advances the same cyclic sequence.

## What virtual mode proves

It can test:

- Python or AutoStrat lifecycle and command ordering;
- FoV registration and virtual movement;
- metadata, image transport, filename patterns, and saving;
- source ordering and wraparound;
- GUI mutation guards, stops, and ordinary failure handling.

It cannot establish safe stage travel, illumination, UV dose, DMD alignment,
device communication, physical timing, autofocus behavior, optical image quality,
or biological validity. Model behavior is meaningful only with representative
images and matching models.

After virtual testing, perform conservative manual hardware checks before an
unattended strategy.

---

← [Writing strategies](strategies.md) | [User guide](README.md) | [Troubleshooting](troubleshooting.md) →
