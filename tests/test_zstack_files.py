"""Tests for single-file z-stack writing (issue #96) and SingleFileZStackStrategy."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
import re

import numpy as np
import pytest
import tifffile

from evomachine.commands import AutomatonCommand
from evomachine.coordinates import Coordinate
from evomachine.filemanager import FileManager
from evomachine.frame import FrameMetaData, FrameMetaDataFactory
from evomachine.image_processing_config import ImageProcessorConfigFactory
from evomachine.strategy import create_strategy_from_definition, list_strategy_definitions
from evomachine.types import AutomatonCommandType, LEDType
from evomachine.zstack_files import (
    COUNTER_FILENAME,
    ZStackChannel,
    ZStackFileConfig,
    ZStackFileWriter,
    expected_plane_count,
    read_zstack,
)

STRATEGY_NAME = "SingleFileZStackStrategy"
FILENAME_RE = re.compile(r"^(\d{6})_450_FOV03_Z-2,1,2_(\d{6})_(\d{6})\.tiff$")


def _planes(centre: float = 7.0, offsets=(0, 1, 2, -1, -2), fov_id: int = 3,
            base: FrameMetaData | None = None) -> list[FrameMetaData]:
    """Per-plane metadata in Automaton Z_STACK acquisition order (centre, +, -)."""
    if base is None:
        base = FrameMetaDataFactory.default(leds={LEDType.LED_450_NM: 10}, exposure=100, fov_id=fov_id)
    start = datetime(2026, 11, 8, 15, 57, 54, 123456, tzinfo=timezone.utc)
    return [
        replace(base, coordinate=Coordinate(1.5, 2.5, centre + offset),
                execution_time=start + timedelta(seconds=0.5 * index))
        for index, offset in enumerate(offsets)
    ]


def _stack(planes: list[FrameMetaData]) -> np.ndarray:
    # Each plane is filled with its own Z so the sort order can be checked from the pixels.
    return np.stack([np.full((4, 5), int(round(m.coordinate.z * 10)), dtype=np.uint16) for m in planes])


# ----------------------------------------------------------------------------- writer
def test_expected_plane_count_matches_automaton_rule() -> None:
    assert expected_plane_count(10, 1) == 21
    assert expected_plane_count(500, 50) == 21
    assert expected_plane_count(0.3, 0.1) == 7   # exact endpoint despite float division
    assert expected_plane_count(10, 3) == 7      # 0, +-3, +-6, +-9


def test_write_single_file_sorted_with_metadata(tmp_path: Path) -> None:
    writer = ZStackFileWriter(ZStackFileConfig(directory=tmp_path))
    planes = _planes()
    record = writer.write(_stack(planes), planes, centre_z=7.0, z_step=1.0, expected_planes=5,
                          extra={"strategy": "test"})

    assert record.complete and record.n_planes == 5 and record.stack_id == 1
    match = FILENAME_RE.match(record.path.name)
    assert match, record.path.name
    assert match.group(2, 3) == ("261108", "155754")  # local (Europe/London = UTC in November)
    assert list(tmp_path.glob("*.tiff")) == [record.path]
    assert not list(tmp_path.glob(".*.partial"))

    array, metadata = read_zstack(record.path)
    assert array.shape == (5, 4, 5)
    assert [int(plane[0, 0]) for plane in array] == [50, 60, 70, 80, 90]  # sorted by Z
    stack = metadata["ZStack"]
    assert stack["z_positions"] == [5.0, 6.0, 7.0, 8.0, 9.0]
    assert stack["z_offsets"] == [-2.0, -1.0, 0.0, 1.0, 2.0]
    assert stack["acquisition_order"] == [2, 3, 4, 1, 0]
    assert stack["axes"] == "ZYX" and stack["strategy"] == "test"
    frames = metadata["FrameMetaData"]
    assert [p["coordinate"]["Z"] for p in frames] == stack["z_positions"]
    assert frames[0]["coordinate"]["X"] == 1.5 and frames[0]["exposure"] == 100
    assert frames[0]["execution_time"] == planes[4].execution_time.isoformat()  # Z=5 was acquired last

    # Only one ImageDescription, and the existing FileManager metadata loader still reads it.
    with tifffile.TiffFile(record.path) as tiff:
        descriptions = [t for t in tiff.pages[0].tags.values() if t.name == "ImageDescription"]
        assert len(descriptions) == 1 and len(tiff.pages) == 5
    assert FileManager.load_tiff_metadata(record.path)["ZStack"]["stack_id"] == 1


def test_three_plane_stack_reads_back_as_planes(tmp_path: Path) -> None:
    writer = ZStackFileWriter(ZStackFileConfig(directory=tmp_path))
    planes = _planes(offsets=(0, 1, -1))
    record = writer.write(_stack(planes), planes, centre_z=7.0, z_step=1.0)
    assert read_zstack(record.path)[0].shape == (3, 4, 5)
    assert tifffile.imread(record.path).shape == (3, 4, 5)


def test_stack_ids_increase_across_writers_and_survive_deleted_files(tmp_path: Path) -> None:
    planes = _planes()
    first = ZStackFileWriter(ZStackFileConfig(directory=tmp_path)).write(_stack(planes), planes, centre_z=7.0)
    second = ZStackFileWriter(ZStackFileConfig(directory=tmp_path)).write(_stack(planes), planes, centre_z=7.0)
    assert (first.stack_id, second.stack_id) == (1, 2)
    second.path.unlink()
    third = ZStackFileWriter(ZStackFileConfig(directory=tmp_path)).write(_stack(planes), planes, centre_z=7.0)
    assert third.stack_id == 3  # the counter file prevents reusing ID 2
    assert (tmp_path / COUNTER_FILENAME).read_text().strip() == "3"
    # Unrelated digit-prefixed files are ignored ...
    (tmp_path / "20261108_overview.tif").write_bytes(b"")
    assert ZStackFileWriter(ZStackFileConfig(directory=tmp_path)).next_stack_id() == 4
    # ... but a stack file with a higher ID copied in from elsewhere moves the counter on.
    (tmp_path / "000041_450_FOV00_Z-1,1,1_261108_000000.tiff").write_bytes(b"")
    assert ZStackFileWriter(ZStackFileConfig(directory=tmp_path)).next_stack_id() == 42


def test_absolute_z_and_incomplete_stack(tmp_path: Path) -> None:
    writer = ZStackFileWriter(ZStackFileConfig(directory=tmp_path, z_in_filename="absolute"))
    planes = _planes(centre=0.6799999999999999, offsets=(0, 1))
    record = writer.write(_stack(planes), planes, centre_z=planes[0].coordinate.z, z_step=1.0,
                          expected_planes=5)
    assert "_Z0.7,1,1.7_" in record.path.name  # rounded, no float noise
    assert not record.complete
    assert read_zstack(record.path)[1]["ZStack"]["complete"] is False


def test_write_rejects_inconsistent_input(tmp_path: Path) -> None:
    writer = ZStackFileWriter(ZStackFileConfig(directory=tmp_path))
    planes = _planes()
    with pytest.raises(ValueError):
        writer.write(_stack(planes)[:4], planes)
    with pytest.raises(ValueError):
        writer.write(_stack(planes), [replace(planes[0], coordinate=None), *planes[1:]])
    assert not list(tmp_path.glob("*.tiff"))


# --------------------------------------------------------------------------- strategy
def _load_strategy():
    cfg = ImageProcessorConfigFactory.default_config(
        channels=[LEDType.LED_450_NM],
        channels_seg=[LEDType.LED_450_NM],
    )
    definitions = {item.name: item for item in list_strategy_definitions()}
    assert STRATEGY_NAME in definitions  # discoverable by the GUI strategy list
    return create_strategy_from_definition(
        name=STRATEGY_NAME, file_path=definitions[STRATEGY_NAME].file_path, cfg=cfg,
    )


def _initialised_strategy(tmp_path: Path, **settings):
    strategy = _load_strategy()
    strategy.output_directory = tmp_path
    strategy.relative_range_um = 2.0
    strategy.step_size_um = 1.0
    for name, value in settings.items():
        setattr(strategy, name, value)
    commands = strategy.initialise(
        fovs={3: Coordinate(0, 0, 7), 4: Coordinate(10, 0, 7)},
        region_of_interests={}, fov_processors={}, dmd=None,
    )
    return strategy, commands


def _completed(command: AutomatonCommand, planes: list[FrameMetaData]) -> AutomatonCommand:
    """Fill a Z_STACK command as Automaton._execute_z_stack would."""
    command.command_data = {"img": [_stack(planes)], "frame_metadata": planes,
                            "saved_paths": [None] * len(planes)}
    command.fov_id = planes[0].fov_id
    return command


def test_strategy_is_discovered_and_registers_its_commands() -> None:
    assert _load_strategy().register_automaton_commands() == {
        AutomatonCommandType.MOVE,
        AutomatonCommandType.Z_STACK,
        AutomatonCommandType.WAIT,
        AutomatonCommandType.TERMINATE_STRATEGY,
        AutomatonCommandType.ABORT_STRATEGY,
    }


def test_strategy_surfaces_errors_like_the_bundled_strategies() -> None:
    strategy = _load_strategy()
    error = RuntimeError("Camera acquisition failed")
    with pytest.raises(RuntimeError, match="Camera acquisition failed") as caught:
        strategy.callback(fov_id=0, data=[], errors=[error])
    assert caught.value is error and strategy.callback_counter == 0


def test_strategy_saves_completed_stack_before_raising_error(tmp_path: Path) -> None:
    strategy, commands = _initialised_strategy(tmp_path)
    assert [c.command_type for c in commands] == [AutomatonCommandType.MOVE, AutomatonCommandType.Z_STACK]
    assert commands[1].command_args["save"] is False
    error = RuntimeError("stage failed")
    with pytest.raises(RuntimeError, match="stage failed"):
        strategy.callback(fov_id=3, data=[commands[0], _completed(commands[1], _planes())], errors=[error])
    assert len(list(tmp_path.glob("*.tiff"))) == 1


def test_strategy_skipped_stack_and_write_failure(tmp_path: Path, monkeypatch) -> None:
    strategy, commands = _initialised_strategy(tmp_path)
    commands[1].command_data = {"skipped": True, "skip_reason": "move failed",
                                "frame_metadata": [], "saved_paths": [None]}
    next_commands = strategy.callback(fov_id=3, data=commands, errors=[])
    assert not list(tmp_path.glob("*.tiff"))
    assert [c.command_type for c in next_commands] == [AutomatonCommandType.MOVE, AutomatonCommandType.Z_STACK]

    def broken_write(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(strategy._writer, "write", broken_write)
    result = strategy.callback(fov_id=4, data=[_completed(next_commands[1], _planes(fov_id=4))], errors=[])
    assert [c.command_type for c in result] == [AutomatonCommandType.ABORT_STRATEGY]


def test_strategy_one_stack_per_batch_and_waits_between_rounds(tmp_path: Path) -> None:
    channels = (ZStackChannel(LEDType.LED_450_NM, 10, 100), ZStackChannel(LEDType.LED_565_NM, 20, 50))
    strategy, commands = _initialised_strategy(tmp_path, channels=channels, n_rounds=2, round_period_s=60.0)
    types_per_batch = []
    for _ in range(4):  # 2 FoVs x 2 channels
        types_per_batch.append([c.command_type for c in commands])
        stack_command = commands[-1]
        planes = _planes(base=stack_command.command_args["frame_metadata"])  # keeps the channel LED
        commands = strategy.callback(fov_id=planes[0].fov_id, data=[_completed(stack_command, planes)], errors=[])
    move, zstack, wait = AutomatonCommandType.MOVE, AutomatonCommandType.Z_STACK, AutomatonCommandType.WAIT
    assert types_per_batch == [[move, zstack], [zstack], [move, zstack], [zstack]]
    assert [c.command_type for c in commands] == [wait, move, zstack]  # next round after a wait
    assert 0 < commands[0].command_args["duration"] <= 60.0
    names = sorted(p.name for p in tmp_path.glob("*.tiff"))
    assert [n.split("_")[1] for n in names] == ["450", "565", "450", "565"]


def test_strategy_end_to_end_on_virtual_microscope(tmp_path: Path) -> None:
    from evomachine.gui.runtime import build_virtual_automaton

    strategy = _load_strategy()
    strategy.output_directory = tmp_path / "stacks"
    strategy.relative_range_um = 2.0
    strategy.step_size_um = 1.0
    strategy.n_rounds = 2
    strategy.channels = (ZStackChannel(LEDType.LED_450_NM, 10, 100), ZStackChannel(LEDType.LED_565_NM, 20, 50))

    from evomachine.gui.facade import AutomatonGuiFacade

    automaton = build_virtual_automaton()
    AutomatonGuiFacade(automaton).gui_initialise_controllers()  # as the GUI Initialise button does
    per_plane_dir = tmp_path / "per_plane"
    per_plane_dir.mkdir()
    automaton.acq_mngr.file_manager.set_output_directory(per_plane_dir)
    automaton.set_strategy(strategy)
    automaton.initialise(fovs={0: Coordinate(0, 0, 0), 1: Coordinate(50, 0, 3)})
    automaton.start_strategy()
    for _ in range(100):
        if automaton.strategy_has_stopped() or automaton.stopped():
            break
        automaton._process()
    assert automaton.strategy_has_stopped() or automaton.stopped()

    files = sorted((tmp_path / "stacks").glob("*.tiff"))
    assert len(files) == 8  # 2 rounds x 2 FoVs x 2 channels
    assert [f.name[:6] for f in files] == [f"{i:06d}" for i in range(1, 9)]
    assert [re.search(r"_FOV(\d+)_", f.name).group(1) for f in files] == ["00", "00", "01", "01"] * 2
    assert [f.name.split("_")[1] for f in files] == ["450", "565"] * 4
    assert all("_Z-2,1,2_" in f.name for f in files)
    for path in files:
        array, metadata = read_zstack(path)
        stack = metadata["ZStack"]
        assert array.ndim == 3 and array.shape[0] == 5
        assert stack["complete"] and stack["strategy"] == STRATEGY_NAME
        assert stack["z_offsets"] == pytest.approx([-2, -1, 0, 1, 2])
        assert stack["z_positions"] == sorted(stack["z_positions"])
        assert len(metadata["FrameMetaData"]) == 5
    assert [r.stack_id for r in strategy.saved_records] == list(range(1, 9))
    assert not any(per_plane_dir.iterdir())  # save=False: no per-plane FileManager files
