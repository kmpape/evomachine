# Virtual implementations

[Documentation](../README.md) › [Developer guide](README.md) › Virtual implementations

Virtual bindings implement the same high-level peripheral contracts without
external hardware. `build_virtual_automaton()` composes them into the same
`Automaton`, managers, FileManager, focus navigation, GUI facade, and socket
transport used by hardware mode.

## Runtime composition

The virtual runtime currently creates:

- a virtual controller and separate virtual DMD controller;
- stage starting at `(0, 0, 0)` with a 100 µm FoV step;
- `3200 × 3200 uint16` camera;
- configured logical filters and LED channels;
- virtual DMD and autofocus;
- acquisition/file managers and software/reference focus;
- a default `NoStrategy` and processing configuration.

Changing the virtual runtime preset must not change explicitly constructed
virtual devices in tests or other callers.

## Virtual camera source

`VirtualImageSourceMode` has `RANDOM` and `FOLDER`. `VirtualImageSource` owns the
mode, directory, relative glob, naturally sorted snapshot, skipped-file reasons,
and cyclic next index.

`VirtualImageSource.from_folder()` validates the candidate without changing an
active camera. Only readable two-dimensional TIFF images with shape
`(3200, 3200)` and dtype `uint16` enter the usable snapshot. The camera replaces
its source only after construction succeeds, which makes updates atomic.

Folder acquisition loads the next file and advances modulo the snapshot length.
Reconfiguration resets the index. If a snapshotted file later becomes invalid,
loading raises a runtime error rather than silently switching data.

The source ignores FoV, channel, Z, exposure, and filter. FoV-aware dataset
routing is deliberately outside the current interface.

## GUI protocol

Virtual-source status and update requests pass through the normal controller,
protocol, facade, and request-handler path. Status includes mode, folder, glob,
usable/skipped files, and next filename. Updates are session-only, server
validated, rejected during incompatible active work, and unavailable for a
non-virtual camera.

## Testing expectations

Virtual tests should verify deterministic random output, natural file order,
cyclic wrap, reset after reconfiguration, mixed valid/invalid folders, atomic
failure, status payloads, hardware rejection, and strategy mutation guards.

Virtual bindings test software behavior, not realistic device physics. Do not add
random timing or intermittent failure to default virtual behavior; inject
controlled failures in focused tests instead.

---

← [Runtime](runtime.md) | [Developer guide](README.md) | [Extending the GUI](gui.md) →
