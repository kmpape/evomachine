"""Tests for parsing, validating, and interpreting the strategy DSL."""

import pytest

from evomachine.coordinates import Coordinate
from evomachine.dsl import (
    CaptureImage,
    DSLSyntaxError,
    DSLValidationError,
    Move,
    MoveTarget,
    Wait,
    parse_dsl,
    validate_strategy,
)
from evomachine.dsl.strategy import DSLStrategy
from evomachine.types import AutomatonCommandType, LEDType


VALID_DSL = """initialise
    move first_fov
    image exposure 100ms
    wait 3s

callback
    move next_fov
    image exposure 100ms led 565nm
    wait 3s

finalise
"""

def test_parse_dsl_builds_expected_intermediate_strategy() -> None:
    parsed = parse_dsl(VALID_DSL)

    assert parsed.initialise == (
        Move(target=MoveTarget.FIRST_FOV),
        CaptureImage(exposure_ms=100.0),
        Wait(duration_seconds=3.0),
    )
    assert parsed.callback[0] == Move(target=MoveTarget.NEXT_FOV)
    assert parsed.finalise == ()


@pytest.mark.parametrize(
    ("wavelength", "led_type"),
    [
        ("385nm", LEDType.LED_385_NM),
        ("450nm", LEDType.LED_450_NM),
        ("515nm", LEDType.LED_515_NM),
        ("565nm", LEDType.LED_565_NM),
        ("645nm", LEDType.LED_645_NM),
    ],
)
def test_parse_dsl_supports_every_image_led_channel(
        wavelength: str,
        led_type: LEDType,
) -> None:
    dsl_text = VALID_DSL.replace("led 565nm", f"led {wavelength}", 1)

    image = parse_dsl(dsl_text).callback[1]
    assert isinstance(image, CaptureImage)
    assert image.led_type is led_type


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("100ms", "0ms", "image exposure"),
        ("3s", "10.1s", "wait duration"),
    ],
)
def test_validate_strategy_rejects_invalid_parameters(old: str, new: str, message: str) -> None:
    with pytest.raises(DSLValidationError, match=message):
        validate_strategy(parse_dsl(VALID_DSL.replace(old, new)))


def test_parse_dsl_wraps_lark_syntax_errors() -> None:
    with pytest.raises(DSLSyntaxError, match="line"):
        parse_dsl(VALID_DSL.replace("image exposure", "capture exposure"))


@pytest.mark.parametrize(
    ("old", "upper_bound"),
    [
        ("100ms", "1000ms"),
        ("3s", "10.0s"),
    ],
)
def test_validate_strategy_accepts_inclusive_upper_bounds(
        old: str,
        upper_bound: str,
) -> None:
    parsed = parse_dsl(VALID_DSL.replace(old, upper_bound))

    assert validate_strategy(parsed) is parsed


def test_dsl_strategy_accepts_ten_second_wait() -> None:
    dsl_text = VALID_DSL.replace("3s", "10.0s")

    strategy = DSLStrategy(cfg=object(), dsl_text=dsl_text)

    assert strategy.validated_strategy.initialise[-1] == Wait(duration_seconds=10.0)


def test_strategy_maps_lifecycle_sections_to_fresh_automaton_commands() -> None:
    strategy = DSLStrategy(cfg=object(), dsl_text=VALID_DSL)
    fovs = {
        10: Coordinate(x=0, y=0, z=0),
        20: Coordinate(x=100, y=100, z=0),
    }
    initialise_commands = strategy.initialise(
        fovs=fovs,
        region_of_interests={10: [], 20: []},
        fov_processors={},
        dmd=None,
    )
    callback_commands = strategy.callback(
        fov_id=10,
        data=initialise_commands,
        errors=[],
    )

    assert [command.command_type for command in initialise_commands] == [
        AutomatonCommandType.MOVE,
        AutomatonCommandType.IMAGE,
        AutomatonCommandType.WAIT,
    ]
    assert initialise_commands[0].command_args == 10
    assert callback_commands[0].command_args == -1
    assert initialise_commands[1].command_args["frame_metadata"].exposure == 100.0
    assert initialise_commands[1].command_args["frame_metadata"].leds == {
        LEDType.LED_450_NM: 10
    }
    assert callback_commands[1].command_args["frame_metadata"].leds == {
        LEDType.LED_565_NM: 10
    }
    assert callback_commands[0].command_id > initialise_commands[-1].command_id
    assert strategy.finalise() == []
