from typing import List
from cells import Cell, CellFactory


class MotherMachine:
    def __init__(self, num_frames: int):
        self.num_frames = num_frames
        self.cells: List[List['Cell']] = [[] for _ in range(0, num_frames)]
        self.cell_factory: 'CellFactory' = CellFactory()
