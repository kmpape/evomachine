from dataclasses import dataclass
from math import ceil
from typing import Dict, List, Optional, Union

from evomachine.exceptions import ConfigError, ErrorCode
from evomachine.utils import EvoCroppingBox

COORD_PRINT_PRECISION = 1


class Coordinate:

    def __init__(
            self,
            x: Union[float, int, None],
            y: Union[float, int, None],
            z: Union[float, int, None] = None,
    ):
        # if not ((isinstance(x, float) or isinstance(x, int)) and (isinstance(y, float) or isinstance(y, int))):
        #     raise TypeError(f"Coordinate: both x and y must be float or int. Received x={x} and y={y}.")
        self.x = x
        self.y = y
        self.z = z

    def copy(self) -> 'Coordinate':
        return Coordinate(self.x, self.y, self.z)

    def has_z(self) -> bool:
        return self.z is not None

    def sign(self) -> 'Coordinate':
        sign_x = 1 if self.x > 0 else (-1 if self.x < 0 else 0)
        sign_y = 1 if self.y > 0 else (-1 if self.y < 0 else 0)
        sign_z = None if self.z is None else (1 if self.z > 0 else (-1 if self.z < 0 else 0))
        return Coordinate(sign_x, sign_y, sign_z)

    def to_dict(self):
        return {key: val for key, val in zip(['X', 'Y', 'Z'], [self.x, self.y, self.z]) if val is not None}

    def __abs__(self):
        return Coordinate(abs(self.x), abs(self.y), None if self.z is None else abs(self.z))

    def __add__(self, other):
        if isinstance(other, Coordinate):
            new_x = self.x + other.x
            new_y = self.y + other.y
            new_z = None if (self.z is None or other.z is None) else self.z + other.z
            return Coordinate(new_x, new_y, new_z)
        elif isinstance(other, int) or isinstance(other, float):
            new_x = self.x + other
            new_y = self.y + other
            new_z = None if self.z is None else self.z + other
            return Coordinate(new_x, new_y, new_z)
        else:
            raise TypeError("Unsupported operand type(s) for +: '{}' and '{}'".format(type(self), type(other)))

    def __sub__(self, other):
        if isinstance(other, Coordinate):
            new_x = self.x - other.x
            new_y = self.y - other.y
            new_z = None if (self.z is None or other.z is None) else self.z - other.z
            return Coordinate(new_x, new_y, new_z)
        elif isinstance(other, int) or isinstance(other, float):
            new_x = self.x - other
            new_y = self.y - other
            new_z = None if self.z is None else self.z - other
            return Coordinate(new_x, new_y, new_z)
        else:
            raise TypeError("Unsupported operand type(s) for -: '{}' and '{}'".format(type(self), type(other)))

    def __str__(self):
        x_str = f"{self.x:.{COORD_PRINT_PRECISION}f}" if self.x is not None else "None"
        y_str = f"{self.y:.{COORD_PRINT_PRECISION}f}" if self.y is not None else "None"
        z_str = f"{self.z:.{COORD_PRINT_PRECISION}f}" if self.z is not None else "None"
        return f"(x={x_str}, y={y_str}, z={z_str})"

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
            return Coordinate(new_x, new_y, new_z)
        elif isinstance(other, int) or isinstance(other, float):
            new_x = self.x * other
            new_y = self.y * other
            new_z = None if self.z is None else self.z * other
            return Coordinate(new_x, new_y, new_z)
        else:
            raise TypeError("Unsupported operand type(s) for *: '{}' and '{}'".format(type(self), type(other)))

    def __eq__(self, other):
        if isinstance(other, Coordinate):
            return self.x == other.x and self.y == other.y and self.z == other.z
        else:
            return False

    @staticmethod
    def from_dict(coord_dict: Dict[str, Union[float, int]]) -> 'Coordinate':
        x = coord_dict.get('x') or coord_dict.get('X')
        y = coord_dict.get('y') or coord_dict.get('Y')
        z = coord_dict.get('z') or coord_dict.get('Z')
        return Coordinate(x, y, z)

    @staticmethod
    def from_coordinate(other: 'Coordinate') -> 'Coordinate':
        return Coordinate(other.x, other.y, other.z)

    @staticmethod
    def none_coordinate() -> 'Coordinate':
        return Coordinate(None, None, None)


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
    ) -> List[Coordinate]:
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
            )
        else:
            delta = Coordinate(
                x=diff.x / (float(num_pos_y) - 1),
                y=diff.sign().y * self.dfov,
                z=diff.z / (float(num_pos_y) - 1),
            )
        return [start + (delta * i) for i in range(max(num_pos_x, num_pos_y))]


@dataclass
class FieldOfView:
    fov_id: int
    "ID of the field of view."
    coordinate: Coordinate
    "Coordinate of the field of view."
    cropping_boxes: Dict[int, EvoCroppingBox]
    "List of cropping boxes for the field of view."

    def get_position_ids(self) -> List[int]:
        return list(self.cropping_boxes.keys())