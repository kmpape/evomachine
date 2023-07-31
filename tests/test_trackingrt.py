import logging
from pathlib import Path
import pickle
import unittest

import numba as nb
import numpy as np

import delta
from delta import utils
from delta.config import DEFAULT_CONFIG_MOTHERMACHINE
from delta.pipeline import TIMER_ROI

from evomachine.acquisition import DeltaCamera
from evomachine.automaton import Automaton
from evomachine.config import IMAGE_CONFIG_DELTA_SIM, IMAGE_CONFIG_DELTA_BENCH, DEVICE_CONFIG_DELTA_SIM, EVOMACHINE_DIR
from evomachine.exceptions import ErrorCode
from evomachine.positionrt import TIMER_POSITION
import evomachine.trackingrt as trackingrt
import evomachine.trackingrt_jit as trackingrt_jit

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
APPROX_EQUAL = False  # allow for assertions with approximate equality
NUM_PXL_DIFF = 10  # number of admissible pixel difference in an image/mask
ABS_PXL_DIFF = 2  # number of admissible pixel difference for features

THIS_DIR = Path(__file__).parent


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


class TestTracking(unittest.TestCase):
    def test_tracking_jit_init(self):
        x_old = []
        y_new = [1.0, 2.0, 3.0]
        area_new = [1.0, 1.0, 1.0]
        u_new = [{"y": y, "area":  area} for (y, area) in zip(y_new, area_new)]
        state_new_jit, attributions_matrix_jit, image_processing_error_jit = trackingrt_jit.track_trench_rt(
            x_old, u_new, 0,
        )
        self.assertEqual(image_processing_error_jit, ErrorCode.ERROR_TRACK_NO_PREV_STATE.value)
        x_new = [{"y": u_i["y"], "area": u_i["area"], "id": id_new+1, "div": False,
                  "ErrorCode.value": trackingrt_jit.ERROR_MAP[trackingrt_jit.ERROR_NO_PREV_STATE]}
                 for id_new, u_i in enumerate(u_new)]
        for (x_new_i, state_new_jit_i) in zip(x_new, state_new_jit):
            self.assertDictEqual(x_new_i, state_new_jit_i)

    def test_tracking_jit_no_inputs(self):
        x_old = [{"y": 1.0, "area": 1.0, "id": 1, "div": False, "ErrorCode.value": ErrorCode.NO_ERROR.value}]
        u_new = []
        state_new_jit, attributions_matrix_jit, image_processing_error_jit = trackingrt_jit.track_trench_rt(
            x_old, u_new, 0,
        )
        self.assertEqual(len(state_new_jit), 0)
        self.assertEqual(image_processing_error_jit, ErrorCode.ERROR_TRACK_NO_INPUTS.value)

    def test_tracking_jit(self):
        with open(THIS_DIR / "data/pos0_states_v1.pickle", "rb") as file:
            pos0_states = pickle.load(file)
        for i_roi in range(10):
            roi_states = pos0_states[i_roi]
            # Inputs for JIT
            areas_old = [np.array([nb.float32(s['area']) for s in roi_state]) for roi_state in roi_states]
            ids_old = [np.array([nb.int32(s['id']) for s in roi_state]) for roi_state in roi_states]
            divs_old = [np.array([nb.bool_(s['div']) for s in roi_state]) for roi_state in roi_states]
            max_ids = []
            for id_list in ids_old:
                this_max = max(id_list)
                all_max = max(max_ids) if max_ids else -1
                max_ids.append(max(this_max, all_max))

            max_ids = np.array([nb.int32(id_) for id_ in max_ids])
            areas_new = [np.array([np.float32(_) for _ in area], dtype=np.float32) for area in areas_old]
            ids_new = [np.array([np.int32(-1) for _ in id_], dtype=np.int32) for id_ in ids_old]
            divs_new = [np.array([np.bool_(False) for _ in div], dtype=np.bool_) for div in divs_old]
            error_codes = [np.array([np.int32(0) for _ in e], dtype=np.int32) for e in areas_old]
            # Inputs for non-JIT
            u_new = [[{'area': s['area'], 'y': s['y']} for s in roi_state] for roi_state in roi_states]
            x_old = [[{'area': s['area'], 'y': s['y'], 'id': s['id']} for s in roi_state] for roi_state in roi_states]
            t = 0
            attributions_matrix = np.zeros((len(areas_old[t]), len(areas_old[t + 1])), dtype=np.bool_)
            error_code = trackingrt_jit.track_trench_rt_jit(areas_old[t], ids_old[t],
                                                            areas_new[t + 1], ids_new[t + 1],
                                                            divs_new[t + 1], error_codes[t + 1],
                                                            attributions_matrix, nb.int32(max_ids[t]))
            for t in range(len(areas_old) - 1):
                attributions_matrix = np.zeros((len(areas_old[t]), len(areas_old[t + 1])), dtype=np.bool_)
                error_code = trackingrt_jit.track_trench_rt_jit(areas_old[t], ids_old[t],
                                                                areas_new[t + 1], ids_new[t + 1],
                                                                divs_new[t + 1], error_codes[t + 1],
                                                                attributions_matrix, nb.int32(max_ids[t]))
                # assert ids_new[t + 1].tolist() == ids_old[t + 1].tolist()
                self.assertListEqual(ids_new[t + 1].tolist(), ids_old[t + 1].tolist())
                if not (ids_new[t + 1].tolist() == ids_old[t + 1].tolist()):
                    print(
                        f"t={t}, roi={i_roi}, ids_new[t+1].tolist()={ids_new[t + 1].tolist()}, ids_old[t+1].tolist()={ids_old[t + 1].tolist()}")
                    print(
                        f"areas_old[t]={areas_old[t]}, ids_old[t]={ids_old[t]}, areas_new[t+1]={areas_new[t + 1]}, ids_new[t+1]={ids_new[t + 1]}, max_ids={max_ids[t]}")
                state_new_jit, attributions_matrix_jit, image_processing_error_jit = trackingrt_jit.track_trench_rt(
                    x_old[t], u_new[t + 1], max_ids[t],
                )
                self.assertListEqual([s['id'] for s in state_new_jit], ids_old[t + 1].tolist())
                # assert [s['id'] for s in state_new_jit] == ids_old[t + 1].tolist()


if __name__ == '__main__':
    unittest.main()
