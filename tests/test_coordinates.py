import pytest

from evomachine.coordinates import Coordinate, CoordinateBounds, CoordinateFactory


def test_coordinate_copy_and_equality_are_value_based():
    coordinate = Coordinate(1, 2, 3, channel_id=4)

    copied = coordinate.copy()

    assert copied == coordinate
    assert copied is not coordinate
    assert copied.get_channel_id() == 4


def test_coordinate_to_dict_omits_none_axes():
    coordinate = Coordinate(1, None, 3)

    assert coordinate.to_dict() == {"X": 1, "Z": 3}


def test_coordinate_from_dict_accepts_upper_and_lower_case_keys():
    assert Coordinate.from_dict({"X": 1, "Y": 2, "Z": 3, "channel_id": 4}) == Coordinate(1, 2, 3, 4)
    assert Coordinate.from_dict({"x": 5, "y": 6, "z": 7}) == Coordinate(5, 6, 7)


def test_coordinate_none_coordinate_has_no_axes_and_no_channel():
    coordinate = Coordinate.none_coordinate()

    assert coordinate.x is None
    assert coordinate.y is None
    assert coordinate.z is None
    assert coordinate.get_channel_id() is None
    assert coordinate.to_dict() == {}


def test_coordinate_has_z_tracks_only_z_presence():
    assert Coordinate(1, 2, 3).has_z()
    assert not Coordinate(1, 2, None).has_z()


def test_coordinate_sign_preserves_channel_and_none_z():
    coordinate = Coordinate(-4, 0, None, channel_id=2)

    assert coordinate.sign() == Coordinate(-1, 0, None, channel_id=2)


def test_coordinate_arithmetic_with_coordinates():
    left = Coordinate(10, 20, None, channel_id=3)
    right = Coordinate(1, 2, 3, channel_id=7)

    assert left + right == Coordinate(11, 22, None, channel_id=3)
    assert left - right == Coordinate(9, 18, None, channel_id=3)
    assert left * right == Coordinate(10, 40, None, channel_id=3)


def test_coordinate_arithmetic_with_scalars():
    coordinate = Coordinate(10, 20, None, channel_id=3)

    assert coordinate + 5 == Coordinate(15, 25, None, channel_id=3)
    assert coordinate - 5 == Coordinate(5, 15, None, channel_id=3)
    assert coordinate * 2 == Coordinate(20, 40, None, channel_id=3)


def test_coordinate_arithmetic_rejects_unsupported_types():
    coordinate = Coordinate(1, 2, 3)

    with pytest.raises(TypeError):
        _ = coordinate + "bad"
    with pytest.raises(TypeError):
        _ = coordinate - "bad"
    with pytest.raises(TypeError):
        _ = coordinate * "bad"


def test_coordinate_ordering_ignores_none_z():
    low = Coordinate(1, 2, None)
    high = Coordinate(2, 3, 4)

    assert low < high
    assert low <= high
    assert high > low
    assert high >= low


def test_coordinate_bounds_skip_none_bounds():
    bounds = CoordinateBounds(
        low=Coordinate(None, -5, None),
        high=Coordinate(10, None, 5),
    )

    assert bounds.contains(Coordinate(-1e6, -5, 5))
    assert bounds.contains(Coordinate(10, 1e6, None))
    assert bounds.is_out_of_bounds(Coordinate(11, 0, 0))
    assert bounds.is_out_of_bounds(Coordinate(0, -6, 0))


def test_coordinate_bounds_from_limits_and_as_limits_copy_values():
    low = Coordinate(0, 1, 2)
    high = Coordinate(3, 4, 5)

    bounds = CoordinateBounds.from_limits((low, high))
    low.x = -100

    assert bounds.as_limits() == (Coordinate(0, 1, 2), Coordinate(3, 4, 5))


def test_coordinate_factory_make_grid_returns_single_copy_for_identical_points():
    start = Coordinate(1, 2, 3, channel_id=4)

    grid = CoordinateFactory(dfov=10).make_grid(start=start, stop=start)

    assert grid == [start]
    assert grid[0] is not start


def test_coordinate_factory_make_grid_steps_along_longer_axis():
    grid = CoordinateFactory(dfov=10).make_grid(
        start=Coordinate(0, 0, 0, channel_id=2),
        stop=Coordinate(20, 10, 4, channel_id=9),
    )

    assert grid == [
        Coordinate(0, 0, 0, channel_id=2),
        Coordinate(10, 5, 2, channel_id=2),
        Coordinate(20, 10, 4, channel_id=2),
    ]
