from __future__ import annotations

import pytest

from evomachine.coordinates import Coordinate
from evomachine.image_processing_config import ImageProcessorConfigFactory
from evomachine.strategy import create_strategy_from_definition, list_strategy_definitions
from evomachine.types import AutomatonCommandType, LEDType


def _timelapse_strategy():
    definitions = {item.name: item for item in list_strategy_definitions()}
    return create_strategy_from_definition(
        name="TimelapseStrategy",
        file_path=definitions["TimelapseStrategy"].file_path,
        cfg=ImageProcessorConfigFactory.default_config(
            channels=[LEDType.LED_OVERHEAD_TIGER],
            channels_seg=[LEDType.LED_OVERHEAD_TIGER],
        ),
    )


def _initialise(strategy, fov_count: int):
    return strategy.initialise(
        fovs={fov_id: Coordinate(float(fov_id), 0.0, 0.0) for fov_id in range(fov_count)},
        region_of_interests={},
        fov_processors={},
        dmd=None,
    )


def test_timelapse_strategy_is_discovered_once_with_its_constants() -> None:
    names = [item.name for item in list_strategy_definitions()]

    assert names.count("TimelapseStrategy") == 1

    strategy = _timelapse_strategy()

    assert strategy.register_automaton_commands() == {
        AutomatonCommandType.MOVE,
        AutomatonCommandType.IMAGE,
        AutomatonCommandType.WAIT,
    }
    assert strategy.period_s == pytest.approx(5.0 * 60.0)
    assert strategy.led is LEDType.LED_OVERHEAD_TIGER


def test_timelapse_round_moves_and_images_every_fov_then_waits_one_period() -> None:
    strategy = _timelapse_strategy()
    strategy.period_s = 120.0

    commands = _initialise(strategy, fov_count=3)

    assert [command.command_type for command in commands] == [
        AutomatonCommandType.MOVE, AutomatonCommandType.IMAGE,
        AutomatonCommandType.MOVE, AutomatonCommandType.IMAGE,
        AutomatonCommandType.MOVE, AutomatonCommandType.IMAGE,
        AutomatonCommandType.WAIT,
    ]
    assert [command.command_args for command in commands[0::2][:3]] == [0, 1, 2]
    image = commands[1]
    assert image.command_args["save"] is True
    assert image.command_args["segment"] is False
    [metadata] = image.command_args["frame_metadata"]
    assert metadata.leds == {LEDType.LED_OVERHEAD_TIGER: strategy.brightness}
    assert metadata.exposure == strategy.exposure_ms
    assert commands[-1].command_args["duration"] == pytest.approx(120.0)
    assert commands[-1].command_args["set_live_mode"] is False


def test_timelapse_next_round_shortens_the_wait_by_the_time_the_round_took(monkeypatch) -> None:
    strategy = _timelapse_strategy()
    strategy.period_s = 120.0
    clock = {"now": 1000.0}
    monkeypatch.setattr("time.monotonic", lambda: clock["now"])

    first = _initialise(strategy, fov_count=2)
    assert first[-1].command_args["duration"] == pytest.approx(120.0)

    clock["now"] += 45.0  # the moves, images and wait of round one took 45 s in total
    second = strategy.callback(fov_id=1, data=first, errors=[])
    assert [command.command_type for command in second].count(AutomatonCommandType.IMAGE) == 2
    assert second[-1].command_args["duration"] == pytest.approx(75.0)

    clock["now"] += 300.0  # a round that overran the period: next one starts immediately
    third = strategy.callback(fov_id=1, data=second, errors=[])
    assert third[-1].command_args["duration"] == 0.0


def test_timelapse_with_no_fovs_only_waits() -> None:
    strategy = _timelapse_strategy()

    commands = _initialise(strategy, fov_count=0)

    assert [command.command_type for command in commands] == [AutomatonCommandType.WAIT]
