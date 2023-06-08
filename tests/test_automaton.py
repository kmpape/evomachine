import logging
import matplotlib.pyplot as plt
import numpy as np
import unittest

import delta
from delta import utils
from delta.config import DEFAULT_CONFIG_MOTHERMACHINE
from delta.pipeline import TIMER_ROI

from evomachine.acquisition import DeltaCamera
from evomachine.automaton import Automaton
from evomachine.config import IMAGE_CONFIG_DELTA_SIM, DEVICE_CONFIG_DELTA_SIM, EVOMACHINE_DIR
from evomachine.positionrt import TIMER_POSITION

TEST_VERBOSITY = logging.INFO
logger = logging.getLogger(__name__)
logger.setLevel(TEST_VERBOSITY)
handler = logging.StreamHandler()
handler.setLevel(TEST_VERBOSITY)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

EPS_REL = 10**(-6)
EPS_REL_APPROX = 10**(-2)
APPROX_EQUAL = True  # allow for assertions with approximate equality
NUM_PXL_DIFF = 10  # number of admissible pixel difference in an image/mask
ABS_PXL_DIFF = 2  # number of admissible pixel difference for features


def features_approx_equal(exp: delta.lineage.CellFeatures, res: delta.lineage.CellFeatures) -> bool:
    new_pole = ((abs(exp.new_pole[0]-res.new_pole[0]) <= ABS_PXL_DIFF) and
                (abs(exp.new_pole[1]-res.new_pole[1]) <= ABS_PXL_DIFF))
    if not new_pole:
        logger.warning("X exp new_pole={}, res new_pole={}".format(exp.new_pole, res.old_pole))
    old_pole = ((abs(exp.old_pole[0]-res.old_pole[0]) <= ABS_PXL_DIFF) and
                (abs(exp.old_pole[1]-res.old_pole[1]) <= ABS_PXL_DIFF))
    if not old_pole:
        logger.warning("X exp old_pole={}, res old_pole={}".format(exp.old_pole, res.old_pole))
    length = abs(exp.length - res.length) < EPS_REL_APPROX * exp.length
    if not length:
        logger.warning("X exp length={}, res length={}".format(exp.length, res.length))
    width = abs(exp.width - res.width) < EPS_REL_APPROX * exp.width
    if not width:
        logger.warning("X exp width={}, res width={}".format(exp.width, res.width))
    area = abs(exp.area - res.area) < EPS_REL_APPROX * exp.area
    if not area:
        logger.warning("X exp area={}, res area={}".format(exp.area, res.area))
    perimeter = abs(exp.perimeter - res.perimeter) < EPS_REL_APPROX * exp.perimeter
    if not perimeter:
        logger.warning("X exp perimeter={}, res perimeter={}".format(exp.perimeter, res.perimeter))
    fluo = abs(exp.fluo[0] - res.fluo[0]) < EPS_REL_APPROX * exp.fluo[0]
    if not fluo:
        logger.warning("X exp fluo={}, res fluo={}".format(exp.fluo[0], res.fluo[0]))
    return new_pole and old_pole and length and width and area and perimeter and fluo


def features_equal(exp: delta.lineage.CellFeatures, res: delta.lineage.CellFeatures) -> bool:
    new_pole = all(exp.new_pole == res.new_pole)
    if not new_pole:
        logger.warning("exp new_pole={}, res new_pole={}".format(exp.new_pole, res.old_pole))
    old_pole = all(exp.old_pole == res.old_pole)
    if not old_pole:
        logger.warning("exp old_pole={}, res old_pole={}".format(exp.old_pole, res.old_pole))
    length = exp.length == res.length
    if not length:
        logger.warning("exp length={}, res length={}".format(exp.length, res.length))
    width = exp.width == res.width
    if not width:
        logger.warning("exp width={}, res width={}".format(exp.width, res.width))
    area = exp.area == res.area
    if not area:
        logger.warning("exp area={}, res area={}".format(exp.area, res.area))
    perimeter = (abs(exp.perimeter - res.perimeter) / exp.perimeter) < EPS_REL
    if not perimeter:
        logger.warning("exp perimeter={}, res perimeter={}".format(exp.perimeter, res.perimeter))
    fluo = (abs(exp.fluo[0] - res.fluo[0]) / exp.fluo[0]) < EPS_REL
    if not fluo:
        logger.warning("exp fluo={}, res fluo={}".format(exp.fluo[0], res.fluo[0]))
    is_eq = new_pole and old_pole and length and width and area and perimeter and fluo
    if (not APPROX_EQUAL) or is_eq:
        return is_eq
    else:
        logger.info("features not equal. checking approximate equality.")
        return features_approx_equal(exp, res)


def arrays_eq_or_approx_eq(exp, res, msg="") -> bool:
    is_eq = np.array_equal(exp, res)
    if (not APPROX_EQUAL) or is_eq:
        return is_eq
    else:
        logger.info("arrays not equal. checking approximate equality.")
        num_nnz = np.count_nonzero(exp != res)
        is_approx_eq = num_nnz <= NUM_PXL_DIFF
        if not is_approx_eq:
            logger.warning(f"{msg}: {np.count_nonzero(exp != res)} / {exp.size} differ.")
        elif num_nnz > 0:
            logger.info(f"{msg}: {np.count_nonzero(exp != res)} / {exp.size} differ.")
        return is_approx_eq


class TestAutomaton(unittest.TestCase):
    @unittest.skip("skipping test_initialisation")
    def test_initialisation(self):
        automaton = Automaton(
            cfg_device=DEVICE_CONFIG_DELTA_SIM,
            cfg_image=IMAGE_CONFIG_DELTA_SIM,
            cfg_delta=DEFAULT_CONFIG_MOTHERMACHINE,
            camera=DeltaCamera(cfg_device=DEVICE_CONFIG_DELTA_SIM),
        )
        automaton.initialise()
        self.assertTrue(automaton._curr_period == 1)
        self.assertTrue(all([all([(roi._frame_id == 1)
                                  for roi in pos_proc.rois])
                             for pos_proc in automaton._pos_processor]
                            )
                        )

        for i_pos in range(DEVICE_CONFIG_DELTA_SIM.num_pos):
            cmap = plt.cm.get_cmap('tab10', 10)
            pos_rt_i = automaton._pos_processor[i_pos]
            for roi in pos_rt_i.rois:
                for i_frame in range(2):
                    img = roi.get_img(frame=i_frame)
                    seg = roi.get_seg(frame=i_frame)
                    lab = roi.get_labels(frame=i_frame)
                    fig, axs = plt.subplots(1, 3)
                    axs[0].imshow(img, cmap='gray', vmin=0, vmax=1)
                    axs[0].set_title('Img')
                    axs[1].imshow(seg, cmap='gray', vmin=0, vmax=1)
                    axs[1].set_title('Seg')
                    axs[2].imshow(lab, cmap=cmap, vmin=0, vmax=10)
                    axs[2].set_title('Lab')
                    fig_name = "out/test_initialisation/unittest_pos_{}_roi_{}_frame_{}.png".format(
                        i_pos, roi.roi_nb, i_frame
                    )
                    fig.savefig(fig_name)

    @unittest.skip("skipping test_stepping")
    def test_stepping(self):
        delta_reader: utils.XPReader = utils.XPReader(
            DEVICE_CONFIG_DELTA_SIM.path_to_images / "Position{p}Channel{c}Frames{t}.tif"
        )
        all_images = [delta_reader.getframes(position=i) for i in delta_reader.positions]
        delta_camera: DeltaCamera = DeltaCamera(cfg_device=DEVICE_CONFIG_DELTA_SIM)
        automaton: Automaton = Automaton(
            cfg_device=DEVICE_CONFIG_DELTA_SIM,
            cfg_image=IMAGE_CONFIG_DELTA_SIM,
            cfg_delta=DEFAULT_CONFIG_MOTHERMACHINE,
            camera=delta_camera
        )
        automaton.initialise()
        self.assertTrue(automaton._curr_period == 1)
        self.assertTrue(all([all([(roi._frame_id == 1)
                                  for roi in pos_proc.rois])
                             for pos_proc in automaton._pos_processor]
                            )
                        )

        while automaton.get_period() < DEVICE_CONFIG_DELTA_SIM.num_periods:
            i_pos = automaton.get_pos()
            i_period = automaton.get_period()
            automaton._take_image()
            img_automaton = automaton.get_frame(i_pos=i_pos, i_chan=0)
            img_delta = all_images[i_pos][i_period, 0, :, :]
            self.assertTrue(np.array_equal(img_automaton, img_delta))
            automaton.increment_pos()

    def test_process(self):
        path_exp_results = EVOMACHINE_DIR.parent / "tests/data/movie_mothermachine_tif"
        pos0_exp = delta.pipeline.Position.load_netcdf(path_exp_results / "expected_results/Position000001.nc")
        pos1_exp = delta.pipeline.Position.load_netcdf(path_exp_results / "expected_results/Position000002.nc")
        this_cfg_device = DEVICE_CONFIG_DELTA_SIM
        this_cfg_device.image_processing_verbosity = 0
        automaton: Automaton = Automaton(
            cfg_device=this_cfg_device,
            cfg_image=IMAGE_CONFIG_DELTA_SIM,
            cfg_delta=DEFAULT_CONFIG_MOTHERMACHINE,
            camera=DeltaCamera(cfg_device=DEVICE_CONFIG_DELTA_SIM),
        )
        automaton.initialise()
        self.assertTrue(automaton._curr_period == 1)
        self.assertTrue(all([all([(roi._frame_id == 1)
                                  for roi in pos_proc.rois])
                             for pos_proc in automaton._pos_processor]
                            )
                        )

        for i_pos in range(2):
            if i_pos == 0:
                exp_result = pos0_exp
            else:
                exp_result = pos1_exp
            for i_roi, roi in enumerate(automaton._pos_processor[i_pos].rois):
                i_period = 0
                msg = "img_stack mismatch at pos={}, roi={}, i_frame={}".format(i_pos, i_roi, i_period)
                self.assertTrue(arrays_eq_or_approx_eq(exp_result.rois[i_roi].img_stack[0], roi.img_stack[1], msg), msg)
                msg = "seg_stack mismatch at pos={}, roi={}, i_frame={}".format(i_pos, i_roi, i_period)
                self.assertTrue(arrays_eq_or_approx_eq(exp_result.rois[i_roi].seg_stack[0], roi.seg_stack[1], msg), msg)
                msg = "label_stack mismatch at pos={}, roi={}, i_frame={}".format(i_pos, i_roi, i_period)
                self.assertTrue(arrays_eq_or_approx_eq(exp_result.rois[i_roi].label_stack[0], roi.label_stack[1], msg), msg)

        for i_period in range(1, DEVICE_CONFIG_DELTA_SIM.num_periods):
            # print("PROCESS AT POS {} and PERIOD {}".format(automaton.get_pos(), automaton.get_period()))
            # Process an additional step at position 0
            self.assertTrue(automaton.get_pos() == 0)
            self.assertTrue(automaton.get_period() == i_period)
            automaton.process()
            # Process an additional step at position 1
            self.assertTrue(automaton.get_pos() == 1)
            self.assertTrue(automaton.get_period() == i_period)
            automaton.process()
            for i_pos in range(2):
                if i_pos == 0:
                    exp_result = pos0_exp
                else:
                    exp_result = pos1_exp
                for i_roi, roi in enumerate(automaton._pos_processor[i_pos].rois):
                    exp_roi = exp_result.rois[i_roi]
                    msg = "img_stack mismatch at pos={}, roi={}, i_frame={}".format(i_pos, i_roi, i_period)
                    self.assertTrue(arrays_eq_or_approx_eq(exp_roi.img_stack[i_period], roi.img_stack[1], msg), msg)
                    msg = "seg_stack mismatch at pos={}, roi={}, i_frame={}".format(i_pos, i_roi, i_period)
                    self.assertTrue(arrays_eq_or_approx_eq(exp_roi.seg_stack[i_period], roi.seg_stack[1], msg), msg)
                    msg = "label_stack mismatch at pos={}, roi={}, i_frame={}".format(i_pos, i_roi, i_period)
                    self.assertTrue(arrays_eq_or_approx_eq(exp_roi.label_stack[i_period], roi.label_stack[1], msg), msg)

        for i_pos in range(2):
            if i_pos == 0:
                exp_result = pos0_exp
            else:
                exp_result = pos1_exp
            for i_roi, roi in enumerate(automaton._pos_processor[i_pos].rois):
                exp_cells = exp_result.rois[i_roi].lineage.cells
                cells = roi.lineage.cells
                assert set(exp_cells.keys()) == set(cells.keys())
                for i_cell, cell in cells.items():
                    exp_cell = exp_cells[i_cell]
                    self.assertTrue(exp_cell.motherid == cell.motherid,
                                    "motherid mismatch at pos={}, roi={}, i_cell={}".format(i_pos, i_roi, i_cell))
                    if cell.first_frame == 0:
                        self.assertTrue(exp_cell._daughterids == cell._daughterids[1:],
                                        "_daughterids mismatch at pos={}, roi={}, i_cell={}".format(
                                            i_pos, i_roi, i_cell))
                    else:
                        self.assertTrue(exp_cell._daughterids == cell._daughterids,
                                        "_daughterids mismatch at pos={}, roi={}, i_cell={}".format(
                                            i_pos, i_roi, i_cell))
                    for i_frame in range(len(exp_cell._features)):
                        exp_features = exp_cell._features[i_frame]
                        if cell.first_frame == 0:
                            res_features = cell._features[i_frame + 1]
                        else:
                            res_features = cell._features[i_frame]
                        self.assertTrue(features_equal(exp_features, res_features),
                                        "features mismatch at pos={}, roi={}, i_cell={}, i_frame".format(
                                            i_pos, i_roi, i_cell, i_frame))

        print(f"Total number of positions {len(automaton._pos_processor)}")
        print(f"Number of ROI per position {[len(pos.rois) for pos in automaton._pos_processor]}")
        TIMER_POSITION.display_timings()
        timings_pos = TIMER_POSITION.get_timings_per_call()
        for key, val in timings_pos.items():
            print(f"{key}: {val}")
        TIMER_ROI.display_timings()


if __name__ == '__main__':
    unittest.main()
