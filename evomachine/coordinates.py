from dataclasses import dataclass
from math import ceil

from evomachine.exceptions import ConfigError, ErrorCode
from evomachine.types import AxisType

COORD_PRINT_PRECISION = 1

class Coordinate:
    """
    Coordinate value object containing X, Y, optional Z, and a channel ID.

    Parameters
    ----------
    x
        X-axis coordinate value, or None when the X axis is unset.
    y
        Y-axis coordinate value, or None when the Y axis is unset.
    z
        Optional Z-axis coordinate value, or None when the Z axis is unset.
    channel_id
        Channel ID associated with the coordinate.

    Returns
    -------
    Coordinate
        Coordinate object storing axis values and channel metadata.
    """

    def __init__(
            self,
            x: float | int | None,
            y: float | int | None,
            z: float | int | None = None,
            channel_id: int = 0,
    ):
        self.x = x
        self.y = y
        self.z = z
        self._channel_id = channel_id

    def copy(self) -> 'Coordinate':
        """
        Return a copy of this coordinate.

        Parameters
        ----------
        None

        Returns
        -------
        Coordinate
            Coordinate with the same axis values and channel ID.
        """
        return Coordinate(x=self.x, y=self.y, z=self.z, channel_id=self._channel_id)

    def get_channel_id(self) -> int:
        """
        Return the coordinate channel ID.

        Parameters
        ----------
        None

        Returns
        -------
        int
            Channel ID stored on the coordinate.
        """
        return self._channel_id

    def has_z(self) -> bool:
        """
        Return whether the Z axis has a value.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            True when z is not None.
        """
        return self.z is not None

    def axis_value(self, axis: AxisType) -> float | int | None:
        """
        Return the value for one axis.

        Parameters
        ----------
        axis
            AxisType value selecting X, Y, or Z.

        Returns
        -------
        float | int | None
            Axis value, or None when the coordinate does not contain that axis.
        """
        if axis == AxisType.X:
            return self.x
        if axis == AxisType.Y:
            return self.y
        if axis == AxisType.Z:
            return self.z
        raise ValueError(f"Coordinate.axis_value: unsupported axis {axis}.")

    @staticmethod
    def from_axis(axis: AxisType, value: float | int, channel_id: int = 0) -> 'Coordinate':
        """
        Build a partial coordinate containing one axis value.

        Parameters
        ----------
        axis
            AxisType value selecting the axis to populate.
        value
            Coordinate value for the selected axis.
        channel_id
            Channel ID to store on the returned coordinate.

        Returns
        -------
        Coordinate
            Coordinate with the selected axis populated and other axes set to None.
        """
        if axis == AxisType.X:
            return Coordinate(x=value, y=None, z=None, channel_id=channel_id)
        if axis == AxisType.Y:
            return Coordinate(x=None, y=value, z=None, channel_id=channel_id)
        if axis == AxisType.Z:
            return Coordinate(x=None, y=None, z=value, channel_id=channel_id)
        raise ValueError(f"Coordinate.from_axis: unsupported axis {axis}.")

    def has_axis_value(self) -> bool:
        """
        Return whether any coordinate axis has a value.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            True when X, Y, or Z is not None.
        """
        return self.x is not None or self.y is not None or self.z is not None

    def filter_axes(self, axes: list[AxisType]) -> 'Coordinate':
        """
        Return a coordinate containing only the selected axes.

        Parameters
        ----------
        axes
            AxisType values to preserve in the returned coordinate.

        Returns
        -------
        Coordinate
            Coordinate with non-selected axes set to None and channel ID preserved.
        """
        return Coordinate(
            x=self.x if AxisType.X in axes else None,
            y=self.y if AxisType.Y in axes else None,
            z=self.z if AxisType.Z in axes else None,
            channel_id=self.get_channel_id(),
        )

    def merge(self, update: 'Coordinate') -> 'Coordinate':
        """
        Merge a partial coordinate into this coordinate.

        Parameters
        ----------
        update
            Coordinate whose non-None axis values replace this coordinate's values.
            A non-None channel ID replaces this coordinate's channel ID.

        Returns
        -------
        Coordinate
            Merged coordinate.
        """
        if not isinstance(update, Coordinate):
            raise TypeError(f"Coordinate.merge: update must be Coordinate, received {type(update)}.")
        update_channel_id = update.get_channel_id()
        return Coordinate(
            x=self.x if update.x is None else update.x,
            y=self.y if update.y is None else update.y,
            z=self.z if update.z is None else update.z,
            channel_id=self.get_channel_id() if update_channel_id is None else update_channel_id,
        )

    def sign(self) -> 'Coordinate':
        sign_x = 1 if self.x > 0 else (-1 if self.x < 0 else 0)
        sign_y = 1 if self.y > 0 else (-1 if self.y < 0 else 0)
        sign_z = None if self.z is None else (1 if self.z > 0 else (-1 if self.z < 0 else 0))
        return Coordinate(x=sign_x, y=sign_y, z=sign_z, channel_id=self._channel_id)

    def to_dict(self):
        return {key: val for key, val in zip(['X', 'Y', 'Z'], [self.x, self.y, self.z]) if val is not None}

    def __abs__(self):
        return Coordinate(
            x=abs(self.x),
            y=abs(self.y),
            z=None if self.z is None else abs(self.z),
            channel_id=self._channel_id
        )

    def __add__(self, other):
        if isinstance(other, Coordinate):
            new_x = self.x + other.x
            new_y = self.y + other.y
            new_z = None if (self.z is None or other.z is None) else self.z + other.z
            return Coordinate(new_x, new_y, new_z, self.get_channel_id())
        elif isinstance(other, int) or isinstance(other, float):
            new_x = self.x + other
            new_y = self.y + other
            new_z = None if self.z is None else self.z + other
            return Coordinate(new_x, new_y, new_z, self.get_channel_id())
        else:
            raise TypeError("Unsupported operand type(s) for +: '{}' and '{}'".format(type(self), type(other)))

    def __sub__(self, other):
        if isinstance(other, Coordinate):
            new_x = self.x - other.x
            new_y = self.y - other.y
            new_z = None if (self.z is None or other.z is None) else self.z - other.z
            return Coordinate(new_x, new_y, new_z, self.get_channel_id())
        elif isinstance(other, int) or isinstance(other, float):
            new_x = self.x - other
            new_y = self.y - other
            new_z = None if self.z is None else self.z - other
            return Coordinate(new_x, new_y, new_z, self.get_channel_id())
        else:
            raise TypeError("Unsupported operand type(s) for -: '{}' and '{}'".format(type(self), type(other)))

    def __str__(self):
        x_str = f"{self.x:.{COORD_PRINT_PRECISION}f}" if self.x is not None else "None"
        y_str = f"{self.y:.{COORD_PRINT_PRECISION}f}" if self.y is not None else "None"
        z_str = f"{self.z:.{COORD_PRINT_PRECISION}f}" if self.z is not None else "None"
        return f"(x={x_str}, y={y_str}, z={z_str}, channel_id={self._channel_id})"

    def __repr__(self):
        return f"Coordinate{self}"

    def __ge__(self, other):
        if isinstance(other, Coordinate):
            return self.x >= other.x and self.y >= other.y and (self.z is None or other.z is None or self.z >= other.z)
        else:
            raise TypeError("Unsupported operand type(s) for >=: '{}' and '{}'".format(type(self), type(other)))

    def __le__(self, other):
        if isinstance(other, Coordinate):
            return self.x <= other.x and self.y <= other.y and (self.z is None or other.z is None or self.z <= other.z)
        else:
            raise TypeError("Unsupported operand type(s) for <=: '{}' and '{}'".format(type(self), type(other)))

    def __gt__(self, other):
        if isinstance(other, Coordinate):
            return self.x > other.x and self.y > other.y and (self.z is None or other.z is None or self.z > other.z)
        else:
            raise TypeError("Unsupported operand type(s) for >: '{}' and '{}'".format(type(self), type(other)))

    def __lt__(self, other):
        if isinstance(other, Coordinate):
            return self.x < other.x and self.y < other.y and (self.z is None or other.z is None or self.z < other.z)
        else:
            raise TypeError("Unsupported operand type(s) for <: '{}' and '{}'".format(type(self), type(other)))

    def __mul__(self, other):
        if isinstance(other, Coordinate):
            new_x = self.x * other.x
            new_y = self.y * other.y
            new_z = None if (self.z is None or other.z is None) else self.z * other.z
            return Coordinate(new_x, new_y, new_z, self.get_channel_id())
        elif isinstance(other, int) or isinstance(other, float):
            new_x = self.x * other
            new_y = self.y * other
            new_z = None if self.z is None else self.z * other
            return Coordinate(new_x, new_y, new_z, self.get_channel_id())
        else:
            raise TypeError("Unsupported operand type(s) for *: '{}' and '{}'".format(type(self), type(other)))

    def __eq__(self, other):
        if isinstance(other, Coordinate):
            return self.x == other.x and self.y == other.y and self.z == other.z and \
                self.get_channel_id() == other.get_channel_id()
        else:
            return False

    @staticmethod
    def from_dict(coord_dict: dict[str, float | int]) -> 'Coordinate':
        x = coord_dict.get('x') or coord_dict.get('X')
        y = coord_dict.get('y') or coord_dict.get('Y')
        z = coord_dict.get('z') or coord_dict.get('Z')
        channel_id = coord_dict['channel_id'] if 'channel_id' in coord_dict.keys() else 0
        return Coordinate(x=x, y=y, z=z, channel_id=channel_id)

    @staticmethod
    def from_coordinate(other: 'Coordinate') -> 'Coordinate':
        return Coordinate(other.x, other.y, other.z, other.get_channel_id())

    @staticmethod
    def none_coordinate() -> 'Coordinate':
        return Coordinate(None, None, None, None)


@dataclass
class CoordinateBounds:
    """
    Coordinate lower and upper bounds with optional per-axis checks.

    Parameters
    ----------
    low
        Optional lower coordinate bounds. Axes set to None are not checked from
        below.
    high
        Optional upper coordinate bounds. Axes set to None are not checked from
        above.

    Returns
    -------
    CoordinateBounds
        Bounds object that can validate partial Coordinates.
    """

    low: Coordinate | None = None
    high: Coordinate | None = None

    def __post_init__(self) -> None:
        """
        Validate coordinate bounds after construction.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if self.low is not None and not isinstance(self.low, Coordinate):
            raise TypeError(f"CoordinateBounds: low must be Coordinate or None, received {type(self.low)}.")
        if self.high is not None and not isinstance(self.high, Coordinate):
            raise TypeError(f"CoordinateBounds: high must be Coordinate or None, received {type(self.high)}.")

    @classmethod
    def from_limits(cls, limits: tuple[Coordinate, Coordinate]) -> 'CoordinateBounds':
        """
        Create CoordinateBounds from a two-coordinate limit tuple.

        Parameters
        ----------
        limits
            Tuple containing lower and upper Coordinate objects.

        Returns
        -------
        CoordinateBounds
            Bounds object with copied lower and upper coordinates.
        """
        if not isinstance(limits, tuple) or len(limits) != 2:
            raise TypeError("CoordinateBounds.from_limits: limits must be tuple[Coordinate, Coordinate].")
        low, high = limits
        if not isinstance(low, Coordinate) or not isinstance(high, Coordinate):
            raise TypeError("CoordinateBounds.from_limits: limits entries must be Coordinate.")
        return cls(low=low.copy(), high=high.copy())

    def copy(self) -> 'CoordinateBounds':
        """
        Return a deep copy of the coordinate bounds.

        Parameters
        ----------
        None

        Returns
        -------
        CoordinateBounds
            Copied bounds.
        """
        return CoordinateBounds(
            low=None if self.low is None else self.low.copy(),
            high=None if self.high is None else self.high.copy(),
        )

    @staticmethod
    def _axis_value(coordinate: Coordinate | None, axis: str) -> float | int | None:
        """
        Return one axis value from a coordinate.

        Parameters
        ----------
        coordinate
            Coordinate to inspect, or None.
        axis
            One of X, Y, or Z.

        Returns
        -------
        float | int | None
            Axis value, or None when unchecked.
        """
        if coordinate is None:
            return None
        if axis == "X":
            return coordinate.x
        if axis == "Y":
            return coordinate.y
        if axis == "Z":
            return coordinate.z
        raise ValueError(f"CoordinateBounds._axis_value: unsupported axis {axis}.")

    def contains(self, coordinate: Coordinate) -> bool:
        """
        Check whether a coordinate is inside the configured bounds.

        Parameters
        ----------
        coordinate
            Full or partial Coordinate to validate. Axes set to None are ignored.

        Returns
        -------
        bool
            True when all provided coordinate axes satisfy checked bounds.
        """
        if not isinstance(coordinate, Coordinate):
            raise TypeError(f"CoordinateBounds.contains: coordinate must be Coordinate, received {type(coordinate)}.")
        for axis, value in (("X", coordinate.x), ("Y", coordinate.y), ("Z", coordinate.z)):
            if value is None:
                continue
            low_value = self._axis_value(coordinate=self.low, axis=axis)
            high_value = self._axis_value(coordinate=self.high, axis=axis)
            if low_value is not None and value < low_value:
                return False
            if high_value is not None and value > high_value:
                return False
        return True

    def is_out_of_bounds(self, coordinate: Coordinate) -> bool:
        """
        Check whether a coordinate is outside the configured bounds.

        Parameters
        ----------
        coordinate
            Full or partial Coordinate to validate.

        Returns
        -------
        bool
            True when any checked axis is outside bounds.
        """
        return not self.contains(coordinate=coordinate)

    def as_limits(self) -> tuple[Coordinate, Coordinate]:
        """
        Return lower and upper Coordinate objects.

        Parameters
        ----------
        None

        Returns
        -------
        tuple[Coordinate, Coordinate]
            Lower and upper bounds. Missing sides are represented by
            Coordinate.none_coordinate().
        """
        return (
            Coordinate.none_coordinate() if self.low is None else self.low.copy(),
            Coordinate.none_coordinate() if self.high is None else self.high.copy(),
        )


class CoordinateFactory:
    def __init__(
            self,
            dfov: float,
    ):
        self.dfov: float = float(dfov)
        "Size of (square) field of view > 0."

        assert dfov > 0

    def make_grid(
            self,
            start: Coordinate,
            stop: Coordinate,
    ) -> list[Coordinate]:
        """
        Creates a grid of field of views from start to stop coordinate. Moves in steps of dfov in either x or y
        direction, and linearly in the other and the z direction (if not None).

        Parameters
        ----------
        start
        stop

        Returns
        -------

        """
        diff = stop - start
        num_pos_x = ceil(abs(diff.x) / self.dfov) + 1
        num_pos_y = ceil(abs(diff.y) / self.dfov) + 1
        if (num_pos_x == 1) and (num_pos_y == 1):
            return [Coordinate.from_coordinate(start)]
        elif num_pos_x > num_pos_y:
            delta = Coordinate(
                x=diff.sign().x * self.dfov,
                y=diff.y / (float(num_pos_x) - 1),
                z=diff.z / (float(num_pos_x) - 1),
                channel_id=start.get_channel_id(),
            )
        else:
            delta = Coordinate(
                x=diff.x / (float(num_pos_y) - 1),
                y=diff.sign().y * self.dfov,
                z=diff.z / (float(num_pos_y) - 1),
                channel_id=start.get_channel_id(),
            )
        return [start + (delta * i) for i in range(max(num_pos_x, num_pos_y))]
