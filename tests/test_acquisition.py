import numpy as np
import pytest
import unittest

from delta import utils

from evomachine.acquisition import DeltaCamera
from evomachine.config import ConfigDevice, DEVICE_CONFIG_DELTA_SIM, EVOMACHINE_DIR
from evomachine.exceptions import ConfigError, StageError


class TestDeltaCamera(unittest.TestCase):
    def test_load_images(self):
        delta_reader: utils.XPReader = utils.XPReader(DEVICE_CONFIG_DELTA_SIM.path_to_images /
                                                      "Position{p}Channel{c}Frames{t}.tif")
        all_images = [delta_reader.getframes(position=i) for i in delta_reader.positions]
        delta_camera: DeltaCamera = DeltaCamera(cfg_device=DEVICE_CONFIG_DELTA_SIM)

        for i_pos in range(DEVICE_CONFIG_DELTA_SIM.num_pos):
            for i_chan in range(DEVICE_CONFIG_DELTA_SIM.num_chan):
                for t in range(DEVICE_CONFIG_DELTA_SIM.num_periods):
                    delta_camera.move_to_pos(i_pos=i_pos)
                    self.assertEqual(i_pos, delta_camera.get_pos())
                    im_cam = delta_camera.get_frame(i_chan=i_chan, i_period=t)
                    im_delta = all_images[i_pos][t, i_chan, :, :]
                    self.assertTrue(np.array_equal(im_cam, im_delta))

    def test_move_pos_exception(self):
        delta_camera: DeltaCamera = DeltaCamera(cfg_device=DEVICE_CONFIG_DELTA_SIM)
        with pytest.raises(StageError):
            delta_camera.move_to_pos(i_pos=-1)

    def test_config_exception(self):
        faulty_cfg = ConfigDevice(
                        num_pos=2,
                        coord_pos=[(0, 0)],
                        num_chan=2,
                        num_periods=10,
                        read_from_disk=True,
                        path_to_images=EVOMACHINE_DIR.parent / "tests/data/movie_mothermachine_tif",
                        image_processing_verbosity=1,
                        tiger_port=None,
                    )
        with pytest.raises(ConfigError):
            delta_camera = DeltaCamera(cfg_device=faulty_cfg)


if __name__ == '__main__':
    unittest.main()
