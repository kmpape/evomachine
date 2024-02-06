import copy

import cv2
import logging
import matplotlib.pyplot as plt
import numpy as np
import unittest
import sys
import os

sys.path.append(os.path.expanduser('~') + '/workspace_python/conda_evomachine3.9/evomachine_repo')
sys.path.append(os.path.expanduser('~') + '/workspace_python/conda_evomachine3.9/de-lta-rt')
sys.path.append(os.path.expanduser('~') + '/workspace_python/conda_evomachine3.9/asitiger')
import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle

import delta
from delta import utils
from delta.config import DEFAULT_CONFIG_MOTHERMACHINE
from delta.pipeline import TIMER_ROI

from evomachine.strategy import NoStrategy
from evomachine.acquisition import DeltaCamera, EvoCamera
from evomachine.automaton import Automaton
from evomachine.config import IMAGE_CONFIG_DELTA_SIM, DEVICE_CONFIG_DELTA_SIM, DEVICE_CONFIG_EVO_TEST, EVOMACHINE_DIR
import evomachine.trackingrt as trackingrt
from evomachine.positionrt import TIMER_POSITION



EPS_REL = 10**(-6)
EPS_REL_APPROX = 10**(-2)
APPROX_EQUAL = False  # allow for assertions with approximate equality
NUM_PXL_DIFF = 10  # number of admissible pixel difference in an image/mask
ABS_PXL_DIFF = 2  # number of admissible pixel difference for features


def features_approx_equal(exp: delta.lineage.CellFeatures, res: delta.lineage.CellFeatures) -> bool:
    new_pole = ((abs(exp.new_pole[0]-res.new_pole[0]) <= ABS_PXL_DIFF) and
                (abs(exp.new_pole[1]-res.new_pole[1]) <= ABS_PXL_DIFF))
    old_pole = ((abs(exp.old_pole[0]-res.old_pole[0]) <= ABS_PXL_DIFF) and
                (abs(exp.old_pole[1]-res.old_pole[1]) <= ABS_PXL_DIFF))
    length = abs(exp.length - res.length) < EPS_REL_APPROX * exp.length
    width = abs(exp.width - res.width) < EPS_REL_APPROX * exp.width
    area = abs(exp.area - res.area) < EPS_REL_APPROX * exp.area
    perimeter = abs(exp.perimeter - res.perimeter) < EPS_REL_APPROX * exp.perimeter
    fluo = abs(exp.fluo[0] - res.fluo[0]) < EPS_REL_APPROX * exp.fluo[0]
    return new_pole and old_pole and length and width and area and perimeter and fluo


def features_equal(exp: delta.lineage.CellFeatures, res: delta.lineage.CellFeatures) -> bool:
    new_pole = all(exp.new_pole == res.new_pole)
    old_pole = all(exp.old_pole == res.old_pole)
    length = exp.length == res.length
    width = exp.width == res.width
    area = exp.area == res.area
    perimeter = (abs(exp.perimeter - res.perimeter) / exp.perimeter) < EPS_REL
    fluo = (abs(exp.fluo[0] - res.fluo[0]) / exp.fluo[0]) < EPS_REL
    is_eq = new_pole and old_pole and length and width and area and perimeter and fluo
    if (not APPROX_EQUAL) or is_eq:
        return is_eq
    else:
        return features_approx_equal(exp, res)


def arrays_eq_or_approx_eq(exp, res, msg="") -> bool:
    is_eq = np.array_equal(exp, res)
    if (not APPROX_EQUAL) or is_eq:
        return is_eq
    else:
        num_nnz = np.count_nonzero(exp != res)
        is_approx_eq = num_nnz <= NUM_PXL_DIFF
        return is_approx_eq


logging.basicConfig(level=logging.INFO, format='%(message)s')

path_exp_results = EVOMACHINE_DIR.parent / "tests/data/movie_mothermachine_tif"
pos0_exp = delta.pipeline.Position.load_netcdf(path_exp_results / "expected_results/Position000001.nc")
pos1_exp = delta.pipeline.Position.load_netcdf(path_exp_results / "expected_results/Position000002.nc")
this_cfg_device = DEVICE_CONFIG_DELTA_SIM
this_cfg_device.image_processing_verbosity = 0
this_cfg_image = IMAGE_CONFIG_DELTA_SIM
this_cfg_image.crop_out_ROI = False
this_cfg_image.use_track_RT = True
automaton: Automaton = Automaton(
    cfg_device=this_cfg_device,
    cfg_image=this_cfg_image,
    cfg_delta=DEFAULT_CONFIG_MOTHERMACHINE,
    camera=DeltaCamera(cfg_device=DEVICE_CONFIG_DELTA_SIM),
    strategy=NoStrategy(),
)
automaton.initialise()

all_states = [[[] for _ in automaton._pos_processor[0].rois],
              [[] for _ in automaton._pos_processor[1].rois]]
for i_pos in range(2):
    if i_pos == 0:
        exp_result = pos0_exp
    else:
        exp_result = pos1_exp
    for i_roi, roi in enumerate(automaton._pos_processor[i_pos].rois):
        i_period = 0
        print(f"i_period={i_period}\n")
        msg = "img_stack mismatch at pos={}, roi={}, i_frame={}".format(i_pos, i_roi, i_period)
        assert arrays_eq_or_approx_eq(exp_result.rois[i_roi].img_stack[0], roi.img_stack[1], msg)
        msg = "seg_stack mismatch at pos={}, roi={}, i_frame={}".format(i_pos, i_roi, i_period)
        assert arrays_eq_or_approx_eq(exp_result.rois[i_roi].seg_stack[0], roi.seg_stack[1], msg)
        msg = "label_stack mismatch at pos={}, roi={}, i_frame={}".format(i_pos, i_roi, i_period)
        assert arrays_eq_or_approx_eq(exp_result.rois[i_roi].label_stack[0], roi.label_stack[1], msg)
        for i_period in range(1, DEVICE_CONFIG_DELTA_SIM.num_periods):
            print(f"i_period={i_period}\n")
            # print("PROCESS AT POS {} and PERIOD {}".format(automaton.get_pos(), automaton.get_period()))
            # Process an additional step at position 0
            assert automaton.get_pos() == 0
            assert automaton.get_period() == i_period
            automaton.process()
            # Process an additional step at position 1
            assert automaton.get_pos() == 1
            assert automaton.get_period() == i_period
            automaton.process()
            for i_pos in range(2):
                if i_pos == 0:
                    exp_result = pos0_exp
                else:
                    exp_result = pos1_exp
                for i_roi, roi in enumerate(automaton._pos_processor[i_pos].rois):
                    do_check = not (this_cfg_image.use_track_RT and (i_pos == 1) and (i_roi == 17)
                                    and (i_period >= 3))
                    do_check = do_check and (not (this_cfg_image.use_track_RT and (i_pos == 0) and (i_roi == 3))
                                             and (i_period >= 4))
                    do_check = do_check and (not (this_cfg_image.use_track_RT and (i_pos == 1) and (i_roi == 1)
                                                  and (i_period >= 4)))
                    do_check = do_check and (not (this_cfg_image.use_track_RT and (i_pos == 0) and (i_roi == 7)
                                                  and (i_period >= 7)))
                    do_check = False
                    exp_roi = exp_result.rois[i_roi]
                    all_states[i_pos][i_roi].append(roi.state_old)
                    if do_check:
                        msg = "img_stack mismatch at pos={}, roi={}, i_frame={}".format(i_pos, i_roi, i_period)
                        assert arrays_eq_or_approx_eq(exp_roi.img_stack[i_period], roi.img_stack[1], msg)
                        msg = "seg_stack mismatch at pos={}, roi={}, i_frame={}".format(i_pos, i_roi, i_period)
                        assert arrays_eq_or_approx_eq(exp_roi.seg_stack[i_period], roi.seg_stack[1], msg)
                        exp_cell_ids = [_id for (_id, x) in exp_roi.lineage.cells.items() if x.first_frame <= i_period <= x.last_frame]
                        cell_ids = [_id for (_id, x) in roi.lineage.cells.items() if x.first_frame <= i_period <= x.last_frame]
                        assert exp_cell_ids == cell_ids
                        # msg = "label_stack mismatch at pos={}, roi={}, i_frame={}".format(i_pos, i_roi, i_period)
                        # assert arrays_eq_or_approx_eq(exp_roi.label_stack[i_period], roi.label_stack[1], msg)

with open("evomachine_repo/tests/data/pos0_states_v1.pickle", "wb") as file:
    pickle.dump(all_states[0], file)

exp_cell_ids = [_id for (_id, x) in exp_roi.lineage.cells.items() if x.first_frame <= i_period <= x.last_frame]
cell_ids = [_id for (_id, x) in roi.lineage.cells.items() if x.first_frame <= i_period <= x.last_frame]

exp_cell_ids_l = [_id for (_id, x) in exp_roi.lineage.cells.items() if x.first_frame <= i_period-1 <= x.last_frame]
cell_ids_l = [_id for (_id, x) in roi.lineage.cells.items() if x.first_frame <= i_period-1 <= x.last_frame]

cmap = plt.cm.get_cmap('tab20', 20)
fig, axs = plt.subplots(3, 2)
axs[0, 0].imshow(exp_roi.label_stack[i_period], cmap=cmap, vmin=0, vmax=20)
axs[0, 0].set_title(f"POS={i_pos}, ROI={i_roi}, t={i_period-1}")
axs[0, 1].imshow(exp_roi.label_stack[i_period+1], cmap=cmap, vmin=0, vmax=20)
axs[0, 1].set_title(f"POS={i_pos}, ROI={i_roi}, t={i_period}")
axs[1, 0].imshow(exp_roi.img_stack[i_period], cmap='gray', vmin=0, vmax=1)
axs[1, 0].set_title(f"POS={i_pos}, ROI={i_roi}, t={i_period-1}")
axs[1, 1].imshow(exp_roi.img_stack[i_period+1], cmap='gray', vmin=0, vmax=1)
axs[1, 1].set_title(f"POS={i_pos}, ROI={i_roi}, t={i_period}")
axs[2, 0].imshow(exp_roi.seg_stack[i_period], cmap='gray', vmin=0, vmax=1)
axs[2, 0].set_title(f"POS={i_pos}, ROI={i_roi}, t={i_period-1}")
axs[2, 1].imshow(exp_roi.seg_stack[i_period+1], cmap='gray', vmin=0, vmax=1)
axs[2, 1].set_title(f"POS={i_pos}, ROI={i_roi}, t={i_period}")

fig.savefig('tracking_error3.png')

T = 9
fig, axs = plt.subplots(2, T)
for i in range(T):
    axs[0, i].imshow(exp_roi.label_stack[i], cmap=cmap, vmin=0, vmax=20)
    axs[0, i].set_title(f"POS={i_pos}, ROI={i_roi}, t={i}")
    axs[1, i].imshow(exp_roi.img_stack[i], cmap='gray', vmin=0, vmax=1)

plt.show()


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
            assert exp_cell.motherid == cell.motherid
            if cell.first_frame == 0:
                assert exp_cell._daughterids == cell._daughterids[1:]
            else:
                assert exp_cell._daughterids == cell._daughterids
            for i_frame in range(len(exp_cell._features)):
                exp_features = exp_cell._features[i_frame]
                if cell.first_frame == 0:
                    res_features = cell._features[i_frame + 1]
                else:
                    res_features = cell._features[i_frame]
                assert features_equal(exp_features, res_features)

