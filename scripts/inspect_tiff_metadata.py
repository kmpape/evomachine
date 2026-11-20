"""Inspect evomachine TIFF files: metadata, filename tokens, z-stack grouping and viewing.

Reads TIFF tags only (no pixel data) unless an image is displayed, so large folders
are scanned quickly. Does not import the ``evomachine`` package (avoids the slow
TensorFlow import), and works with both the per-plane files written by
``FileManager.save_frame`` and multi-page stack files whose first-page description
holds ``{"FrameMetaData": [per-plane dicts], ...}``.

Examples
--------
Summary of a folder::

    .venv/bin/python scripts/inspect_tiff_metadata.py ~/Desktop/Z_stacks_new_gui_v2

Write one CSV row per file and check filename tokens against the metadata::

    .venv/bin/python scripts/inspect_tiff_metadata.py ~/Desktop/Z_stacks_new_gui_v2 \
        --csv metadata.csv --check

Print the full metadata of one file::

    .venv/bin/python scripts/inspect_tiff_metadata.py path/to/file.tiff --json

View a single file, or stack number 3 from the summary (planes sorted by Z, with a slider)::

    .venv/bin/python scripts/inspect_tiff_metadata.py path/to/file.tiff --show
    .venv/bin/python scripts/inspect_tiff_metadata.py ~/Desktop/Z_stacks_new_gui_v2 --show-stack 3
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tifffile

# Matches FileManager's default pattern:
# '{channel}_FOV{fov_id}_X{x}_Y{y}_Z{z}_F{filter_wheel}_{timestamp}' + '.tiff'
DEFAULT_FILENAME_RE = re.compile(
    r"^(?P<channel>.+?)_FOV(?P<fov_id>-?\d+)"
    r"_X(?P<x>[^_]*)_Y(?P<y>[^_]*)_Z(?P<z>[^_]*)"
    r"_F(?P<filter_wheel>[^_]+)"
    r"_(?P<timestamp>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}-\d{6}[+-]\d{4})$"
)
TIFF_SUFFIXES = (".tiff", ".tif")


@dataclass
class PlaneMetadata:
    """Flattened evomachine FrameMetaData for one image plane."""

    frame_id: int | None
    callback_id: int | None
    fov_id: int | None
    leds: dict[str, Any] | None
    filter_wheel: Any
    exposure: float | None
    dmd_pattern: Any
    x: float | None
    y: float | None
    z: float | None
    creation_time: str | None
    creation_timezone: str | None
    execution_time: str | None
    execution_timezone: str | None
    additional_metadata: dict[str, Any]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlaneMetadata":
        coordinate = data.get("coordinate") or {}
        return cls(
            frame_id=data.get("frame_id"),
            callback_id=data.get("callback_id"),
            fov_id=data.get("fov_id"),
            leds=data.get("leds"),
            filter_wheel=data.get("filter_wheel"),
            exposure=data.get("exposure"),
            dmd_pattern=data.get("dmd_pattern"),
            x=coordinate.get("X"),
            y=coordinate.get("Y"),
            z=coordinate.get("Z"),
            creation_time=data.get("creation_time"),
            creation_timezone=data.get("creation_timezone"),
            execution_time=data.get("execution_time"),
            execution_timezone=data.get("execution_timezone"),
            additional_metadata=data.get("additional_metadata") or {},
        )


@dataclass
class TiffRecord:
    """Everything read from one TIFF file without loading pixel data."""

    path: Path
    n_pages: int
    page_shape: tuple[int, ...]
    dtype: str
    filename_tokens: dict[str, str] | None
    planes: list[PlaneMetadata] = field(default_factory=list)
    file_metadata: dict[str, Any] = field(default_factory=dict)
    """Top-level description keys other than FrameMetaData (e.g. stack-level info)."""
    error: str | None = None

    @property
    def is_stack_file(self) -> bool:
        return self.n_pages > 1


def find_tiffs(paths: list[Path], recursive: bool) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        path = path.expanduser()
        if path.is_dir():
            pattern = "**/*" if recursive else "*"
            files.extend(p for p in path.glob(pattern) if p.is_file() and p.suffix.lower() in TIFF_SUFFIXES)
        elif path.is_file():
            files.append(path)
        else:
            print(f"warning: {path} does not exist", file=sys.stderr)
    return sorted(set(files))


def parse_filename(path: Path) -> dict[str, str] | None:
    match = DEFAULT_FILENAME_RE.match(path.stem)
    return match.groupdict() if match else None


def _evomachine_description(tiff: tifffile.TiffFile) -> dict[str, Any] | None:
    """Return the first ImageDescription JSON that holds evomachine metadata."""
    for tag in tiff.pages[0].tags.values():
        if tag.name != "ImageDescription":
            continue
        try:
            value = json.loads(tag.value)
        except (TypeError, ValueError):
            continue
        if isinstance(value, dict) and "FrameMetaData" in value:
            return value
    return None


def read_record(path: Path) -> TiffRecord:
    tokens = parse_filename(path)
    try:
        with tifffile.TiffFile(path) as tiff:
            page = tiff.pages[0]
            record = TiffRecord(
                path=path,
                n_pages=len(tiff.pages),
                page_shape=tuple(page.shape),
                dtype=str(page.dtype),
                filename_tokens=tokens,
            )
            description = _evomachine_description(tiff)
    except Exception as error:  # noqa: BLE001 - report and keep scanning
        return TiffRecord(path=path, n_pages=0, page_shape=(), dtype="", filename_tokens=tokens,
                          error=f"{type(error).__name__}: {error}")
    if description is None:
        record.error = "no evomachine FrameMetaData in ImageDescription"
        return record
    frame_metadata = description.pop("FrameMetaData")
    items = frame_metadata if isinstance(frame_metadata, list) else [frame_metadata]
    record.planes = [PlaneMetadata.from_dict(item) for item in items]
    record.file_metadata = description
    return record


@dataclass
class Stack:
    """Planes belonging to one z-stack, either from one stack file or grouped per-plane files."""

    key: tuple[Any, ...]
    entries: list[tuple[TiffRecord, int]]
    """(record, page index) pairs, sorted by Z."""

    @property
    def fov_id(self) -> int | None:
        return self.entries[0][0].planes[self.entries[0][1]].fov_id

    @property
    def z_values(self) -> list[float]:
        return [record.planes[index].z for record, index in self.entries]


def group_stacks(records: list[TiffRecord]) -> tuple[list[Stack], list[TiffRecord]]:
    """Group planes into stacks.

    Per-plane files of one stack share fov_id, frame_id and creation_time (take_z_stack copies
    one FrameMetaData per plane), so that triple is the grouping key. frame_id alone is not
    enough because it restarts at 0 when the backend restarts. Files without a Z coordinate
    (e.g. '..._X_Y_Zauto_...') are returned separately.
    """
    groups: dict[tuple[Any, ...], list[tuple[TiffRecord, int]]] = defaultdict(list)
    no_z: list[TiffRecord] = []
    for record in records:
        if not record.planes:
            continue
        if record.is_stack_file:
            groups[("file", str(record.path))] = [(record, i) for i in range(len(record.planes))]
            continue
        plane = record.planes[0]
        if plane.z is None:
            no_z.append(record)
            continue
        groups[(plane.fov_id, plane.frame_id, plane.creation_time)].append((record, 0))
    stacks = []
    for key, entries in groups.items():
        entries.sort(key=lambda item: (item[0].planes[item[1]].z is None, item[0].planes[item[1]].z or 0.0))
        stacks.append(Stack(key=key, entries=entries))
    stacks.sort(key=lambda s: min((r.planes[i].execution_time or "") for r, i in s.entries))
    return stacks, no_z


def _float_or_none(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def check_filename_against_metadata(record: TiffRecord) -> list[str]:
    """Return mismatches between filename tokens and the stored metadata (per-plane files only)."""
    tokens = record.filename_tokens
    if tokens is None or not record.planes or record.is_stack_file:
        return []
    plane = record.planes[0]
    problems: list[str] = []
    if int(tokens["fov_id"]) != plane.fov_id:
        problems.append(f"fov_id filename={tokens['fov_id']} metadata={plane.fov_id}")
    for axis, value in (("x", plane.x), ("y", plane.y), ("z", plane.z)):
        token = tokens[axis]
        if value is None:
            if token not in ("", "auto"):
                problems.append(f"{axis} filename={token!r} metadata=None")
            continue
        parsed = _float_or_none(token)
        if parsed is None or not math.isclose(parsed, value, rel_tol=0, abs_tol=1e-9):
            problems.append(f"{axis} filename={token!r} metadata={value}")
    led_names = ".".join(name.replace("_", "") for name in (plane.leds or {})) or "NOLED"
    if tokens["channel"] != led_names:
        problems.append(f"channel filename={tokens['channel']} metadata leds={list((plane.leds or {}).keys())}")
    filter_value = "None" if plane.filter_wheel is None else str(plane.filter_wheel.get("value"))
    if tokens["filter_wheel"] != filter_value:
        problems.append(f"filter filename={tokens['filter_wheel']} metadata={plane.filter_wheel}")
    timestamp_source = plane.execution_time or plane.creation_time or ""
    # '2026-11-08T15:57:42.353247+00:00' -> '2026-11-08_15-57-42-353247+0000'
    expected = re.sub(r"[T:]", "-", timestamp_source).replace(".", "-")
    expected = expected[:10] + "_" + expected[11:]
    expected = re.sub(r"([+-]\d{2})-(\d{2})$", r"\1\2", expected)
    if tokens["timestamp"] != expected:
        problems.append(f"timestamp filename={tokens['timestamp']} metadata={timestamp_source}")
    return problems


def flat_row(record: TiffRecord, plane_index: int = 0) -> dict[str, Any]:
    plane = record.planes[plane_index] if record.planes else None
    tokens = record.filename_tokens or {}
    row: dict[str, Any] = {
        "file": record.path.name,
        "page": plane_index,
        "n_pages": record.n_pages,
        "page_shape": "x".join(map(str, record.page_shape)),
        "dtype": record.dtype,
        **{f"name_{key}": value for key, value in tokens.items()},
        "error": record.error or "",
    }
    if plane is not None:
        row.update({
            "frame_id": plane.frame_id,
            "callback_id": plane.callback_id,
            "fov_id": plane.fov_id,
            "leds": json.dumps(plane.leds),
            "filter_wheel": json.dumps(plane.filter_wheel),
            "exposure": plane.exposure,
            "dmd_pattern": json.dumps(plane.dmd_pattern),
            "x": plane.x,
            "y": plane.y,
            "z": plane.z,
            "creation_time": plane.creation_time,
            "creation_timezone": plane.creation_timezone,
            "execution_time": plane.execution_time,
            "execution_timezone": plane.execution_timezone,
            "additional_metadata": json.dumps(plane.additional_metadata),
            "file_metadata": json.dumps(record.file_metadata) if record.file_metadata else "",
        })
    return row


def write_csv(records: list[TiffRecord], path: Path) -> None:
    rows = [flat_row(r, i) for r in records for i in range(max(len(r.planes), 1))]
    fieldnames: list[str] = []
    for row in rows:
        fieldnames.extend(key for key in row if key not in fieldnames)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {path}")


def print_summary(records: list[TiffRecord], stacks: list[Stack], no_z: list[TiffRecord]) -> None:
    errors = [r for r in records if r.error]
    print(f"{len(records)} TIFF files, {len(errors)} without readable evomachine metadata")
    shapes = defaultdict(int)
    for record in records:
        shapes[(record.n_pages, record.page_shape, record.dtype)] += 1
    for (pages, shape, dtype), count in sorted(shapes.items(), key=lambda item: -item[1]):
        print(f"  {count:5d} file(s): {pages} page(s) of {shape} {dtype}")
    for record in errors[:10]:
        print(f"  ! {record.path.name}: {record.error}")
    print(f"\n{len(stacks)} z-stack(s); {len(no_z)} file(s) without a Z coordinate")
    print(f"{'#':>4} {'fov':>4} {'frame':>6} {'cb':>3} {'planes':>6} {'z_min':>9} {'z_max':>9} {'step':>6}  first execution_time")
    for index, stack in enumerate(stacks):
        z = stack.z_values
        steps = sorted({round(b - a, 6) for a, b in zip(z, z[1:])})
        step = f"{steps[0]:g}" if len(steps) == 1 else ("var" if steps else "-")
        first_record, first_index = stack.entries[0]
        plane = first_record.planes[first_index]
        times = sorted(r.planes[i].execution_time or "" for r, i in stack.entries)
        print(f"{index:>4} {str(stack.fov_id):>4} {str(plane.frame_id):>6} {str(plane.callback_id):>3} "
              f"{len(z):>6} {min(z):>9.3f} {max(z):>9.3f} {step:>6}  {times[0]}")


def show_planes(entries: list[tuple[TiffRecord, int]], title: str) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    cache: dict[int, Any] = {}

    def load(index: int):
        if index not in cache:
            record, page = entries[index]
            cache[index] = tifffile.imread(record.path, key=page)
        return cache[index]

    def label(index: int) -> str:
        record, page = entries[index]
        plane = record.planes[page] if record.planes else None
        z = f"Z={plane.z}" if plane is not None and plane.z is not None else ""
        return f"{title}\nplane {index + 1}/{len(entries)} {z}\n{record.path.name}"

    figure, axes = plt.subplots(figsize=(8, 8.5))
    image = axes.imshow(load(0), cmap="gray")
    axes.set_title(label(0), fontsize=8)
    axes.axis("off")
    figure.colorbar(image, ax=axes, fraction=0.046)
    if len(entries) > 1:
        slider_axes = figure.add_axes((0.2, 0.03, 0.6, 0.03))
        slider = Slider(slider_axes, "plane", 0, len(entries) - 1, valinit=0, valstep=1)

        def update(value: float) -> None:
            index = int(value)
            data = load(index)
            image.set_data(data)
            image.set_clim(data.min(), data.max())
            axes.set_title(label(index), fontsize=8)
            figure.canvas.draw_idle()

        slider.on_changed(update)
        figure._slider = slider  # keep a reference so the widget stays responsive
    plt.show()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", type=Path, help="TIFF files and/or folders")
    parser.add_argument("--recursive", action="store_true", help="search folders recursively")
    parser.add_argument("--csv", type=Path, help="write one row per plane to this CSV file")
    parser.add_argument("--check", action="store_true", help="compare filename tokens with stored metadata")
    parser.add_argument("--json", action="store_true", help="print the full metadata of each given file")
    parser.add_argument("--show", action="store_true", help="display the given file (all pages, with a slider)")
    parser.add_argument("--show-stack", type=int, metavar="N", help="display stack N from the summary table")
    args = parser.parse_args(argv)

    files = find_tiffs(args.paths, recursive=args.recursive)
    if not files:
        print("no TIFF files found", file=sys.stderr)
        return 1
    records = [read_record(path) for path in files]

    if args.json:
        for record in records:
            print(f"== {record.path}")
            print(json.dumps({
                "n_pages": record.n_pages, "page_shape": record.page_shape, "dtype": record.dtype,
                "filename_tokens": record.filename_tokens, "error": record.error,
                "file_metadata": record.file_metadata,
                "FrameMetaData": [vars(plane) for plane in record.planes],
            }, indent=2, default=str))
        return 0

    if args.show:
        for record in records:
            show_planes([(record, i) for i in range(max(record.n_pages, 1))], title=record.path.name)
        return 0

    stacks, no_z = group_stacks(records)
    if args.show_stack is not None:
        if not 0 <= args.show_stack < len(stacks):
            print(f"stack index must be 0..{len(stacks) - 1}", file=sys.stderr)
            return 1
        stack = stacks[args.show_stack]
        show_planes(stack.entries, title=f"stack {args.show_stack} (FOV {stack.fov_id})")
        return 0

    print_summary(records, stacks, no_z)
    if args.check:
        mismatched = [(r, problems) for r in records if (problems := check_filename_against_metadata(r))]
        unparsed = [r for r in records if r.filename_tokens is None]
        print(f"\nfilename check: {len(mismatched)} file(s) disagree with their metadata; "
              f"{len(unparsed)} filename(s) do not follow the default pattern")
        for record, problems in mismatched[:20]:
            print(f"  {record.path.name}: {'; '.join(problems)}")
    if args.csv:
        write_csv(records, args.csv.expanduser())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
