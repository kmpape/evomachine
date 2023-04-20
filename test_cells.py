import unittest

from cells import CellFactory


class TestCells(unittest.TestCase):
    def test_cell_factory_id(self):
        # Check Unique IDs
        factory = CellFactory()
        cell0 = factory.make_cell()
        cell1 = factory.make_cell()
        self.assertNotEqual(cell0.get_cell_id(), cell1.get_cell_id(), "IDs are not unique")
        self.assertFalse(cell0 == cell1)
        self.assertTrue(cell0 == cell0)

    def test_cell_factory_properties(self):
        factory = CellFactory()
        mask_id = 22
        frame_id = 103
        trench_id = 55
        x_coord = 2.6
        y_coord = 3.9
        area = 2.5
        age = 1.1
        is_alive = True
        fluo = [1.0, 2.0, 3.0]
        factory.set_mask_id(mask_id)
        factory.set_frame_id(frame_id)
        factory.set_trench_id(trench_id)
        factory.set_x_y_coord(x_coord, y_coord)
        factory.set_properties(area, age, is_alive, fluo[0], fluo[1], fluo[2])
        cell3 = factory.make_cell()
        self.assertEqual(cell3.get_mask_id(), mask_id)
        self.assertEqual(cell3.get_position().get_frame_id(), frame_id)
        self.assertEqual(cell3.get_position().get_trench_id(), trench_id)
        self.assertEqual(cell3.get_position().get_x_y_coord(), (x_coord, y_coord))
        self.assertEqual(cell3.get_properties().area, area)
        self.assertEqual(cell3.get_properties().age, age)
        self.assertEqual(cell3.get_properties().is_alive, is_alive)
        self.assertEqual(cell3.get_properties().fluo_chan1, fluo[0])
        self.assertEqual(cell3.get_properties().fluo_chan2, fluo[1])
        self.assertEqual(cell3.get_properties().fluo_chan3, fluo[2])


if __name__ == '__main__':
    unittest.main()
