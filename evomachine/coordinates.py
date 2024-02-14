from math import ceil
from typing import Dict, List, Optional, Union

COORD_PRINT_PRECISION = 1


class Coordinate:

    def __init__(
            self,
            x: Union[float, int],
            y: Union[float, int],
            z: Union[float, int, None] = None,
    ):
        if not ((isinstance(x, float) or isinstance(x, int)) and (isinstance(y, float) or isinstance(y, int))):
            raise TypeError(f"Coordinate: both x and y must be float or int. Received x={x} and y={y}.")
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
        return f"(x={self.x:.{COORD_PRINT_PRECISION}f}, " \
               f"y={self.y:.{COORD_PRINT_PRECISION}f}, " \
               f"z={self.z:.{COORD_PRINT_PRECISION}f})"

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


class CoordinateFactory:
    def __init__(
            self,
            dfov_x: float,
            dfov_y: Optional[float] = None,
    ):
        self.dfov_x: float = float(dfov_x)
        "Length of field of view in x direction. Must be positive."
        self.dfov_y: float = float(dfov_y) if dfov_y is not None else dfov_x
        "Length of field of view in y direction."

        assert dfov_x > 0
        assert dfov_y is None or dfov_y > 0

    def make_grid_xy(
            self,
            start: Coordinate,
            stop: Coordinate,
    ) -> List[Coordinate]:
        tmp_start = Coordinate(start.x, start.y)
        tmp_stop = Coordinate(stop.x, stop.y)
        tmp_delta = tmp_stop - tmp_start
        num_pos_x = ceil(abs(tmp_delta).x / self.dfov_x) + 1
        num_pos_y = ceil(abs(tmp_delta).y / self.dfov_y) + 1
        if (num_pos_x == 1) and (num_pos_y == 1):
            return [Coordinate.from_coordinate(tmp_start)]
        elif num_pos_x > num_pos_y:
            delta = Coordinate(tmp_delta.sign().x * self.dfov_x, tmp_delta.y / (float(num_pos_x) - 1), 0.0)
        else:
            delta = Coordinate(tmp_delta.x / (float(num_pos_y) - 1), tmp_delta.sign().y * self.dfov_y, 0.0)
        return [tmp_start + (delta * i) for i in range(max(num_pos_x, num_pos_y))]
