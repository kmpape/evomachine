import unittest

from cells import CellFactory


class TestCells(unittest.TestCase):
    def test_cell_factory(self):
        # TODO: Test assignements and operators
        cell_factory = CellFactory()
        cell0 = cell_factory.make_cell()
        cell1 = cell_factory.make_cell()
        self.assertNotEqual(cell0.get_cell_id(), cell1.get_cell_id(), "IDs are not unique")


if __name__ == '__main__':
    unittest.main()
