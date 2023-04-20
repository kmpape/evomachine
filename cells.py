from typing import Tuple, List


class CellPosition:
    def __init__(self, frame_id: int, trench_id: int, x_coord: float, y_coord: float):
        self._frame_id: int = frame_id
        self._trench_id: int = trench_id
        self._x_coord: float = x_coord
        self._y_coord: float = y_coord

    def set_x_y_coord(self, x_coord: float, y_coord: float):
        self._x_coord: float = x_coord
        self._y_coord: float = y_coord

    def get_x_y_coord(self) -> Tuple[float, float]:
        return self._x_coord, self._y_coord

    def get_frame_id(self) -> int:
        return self._frame_id

    def get_trench_id(self) -> int:
        return self._trench_id

    def __str__(self):
        return f"frame={self._frame_id}, trench={self._trench_id}, (x,y)=({self._x_coord:.2f}, {self._y_coord:.2f})"

    def __eq__(self, other):
        return ((self._frame_id == other.get_frame_id()) and (self._trench_id == other.get_trench_id()) and
                (self._x_coord == other.get_x_y_coord()[0]) and (self._y_coord == other.get_x_y_coord()[1]))


class CellProperties:
    def __init__(self, area: float, age: float, is_alive: bool,
                 fluo_chan1: float, fluo_chan2: float, fluo_chan3: float):
        self.area: float = area
        self.age: float = age
        self.is_alive: bool = is_alive
        self.fluo_chan1: float = fluo_chan1
        self.fluo_chan2: float = fluo_chan2
        self.fluo_chan3: float = fluo_chan3

    def __str__(self):
        return f"area={self.area}, age={self.age}, alive={self.is_alive}, fluo=({self.fluo_chan1:.2f}, "\
               f"{self.fluo_chan2:.2f}, {self.fluo_chan3:.2f})"


class Cell:
    def __init__(self, cell_id: int, mask_id: int, position: 'CellPosition', properties: 'CellProperties'):
        self._cell_id: int = cell_id
        self._mask_id: int = mask_id
        self._position: 'CellPosition' = position
        self._properties: 'CellProperties' = properties

    def set_area(self, area: float):
        self._properties.area = area

    def set_age(self, age: float):
        self._properties.age = age

    def set_is_alive(self, is_alive: bool):
        self._properties.is_alive = is_alive

    def set_fluo(self, fluo_chan1: float, fluo_chan2: float, fluo_chan3: float):
        self._properties.fluo_chan1 = fluo_chan1
        self._properties.fluo_chan2 = fluo_chan2
        self._properties.fluo_chan3 = fluo_chan3

    def get_cell_id(self) -> int:
        return self._cell_id

    def get_mask_id(self) -> int:
        return self._mask_id

    def get_properties(self) -> 'CellProperties':
        return self._properties

    def get_position(self) -> 'CellPosition':
        return self._position

    def __str__(self):
        return f"Cell ID={self._cell_id}, mask ID={self._mask_id}"

    def __eq__(self, other):
        return self._cell_id == other.get_cell_id()


class CellFactory:
    def __init__(self):
        # Cell
        self._cell_id: int = 0  # unique ID, automatically incremented
        self._mask_id: int = 0
        # CellPosition
        self._frame_id: int = 0
        self._trench_id: int = 0
        self._x_coord: float = 0.0
        self._y_coord: float = 0.0
        # CellProperties
        self._area: float = 0.0
        self._age: float = 0.0
        self._is_alive: bool = False
        self._fluo_chan1: float = 0.0
        self._fluo_chan2: float = 0.0
        self._fluo_chan3: float = 0.0

    def set_mask_id(self, mask_id: int):
        self._mask_id = mask_id

    def set_frame_id(self, frame_id: int):
        self._frame_id = frame_id

    def set_trench_id(self, trench_id: int):
        self._trench_id = trench_id

    def set_x_y_coord(self, x_coord: float, y_coord: float):
        self._x_coord = x_coord
        self._y_coord = y_coord

    def set_properties(self, area: float, age: float, is_alive: bool,
                       fluo_chan1: float, fluo_chan2: float, fluo_chan3: float):
        self._area = area
        self._age = age
        self._is_alive = is_alive
        self._fluo_chan1 = fluo_chan1
        self._fluo_chan2 = fluo_chan2
        self._fluo_chan3 = fluo_chan3

    def make_cell(self) -> 'Cell':
        position: 'CellPosition' = CellPosition(frame_id=self._frame_id, trench_id=self._trench_id,
                                                x_coord=self._x_coord, y_coord=self._y_coord)
        properties: 'CellProperties' = CellProperties(area=self._area, age=self._age, is_alive=self._is_alive,
                                                      fluo_chan1=self._fluo_chan1, fluo_chan2=self._fluo_chan2,
                                                      fluo_chan3=self._fluo_chan3)
        cell = Cell(cell_id=self._cell_id, mask_id=self._mask_id, position=position, properties=properties)
        self._cell_id += 1
        return cell


class MotherMachine:
    def __init__(self):
        # TODO: probably need to index this via frame_id->trench_id
        self.cells: List['Cell'] = []
        self.cell_factory: 'CellFactory' = CellFactory()
