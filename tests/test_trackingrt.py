import logging
from pathlib import Path
import pickle
import unittest

import numba as nb
import numpy as np

from evomachine.exceptions import ErrorCode
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

THIS_DIR = Path(__file__).parent


class TestTracking(unittest.TestCase):
    def test_tracking_jit_init(self):
        x_old = []
        y_new = [1.0, 2.0, 3.0]
        area_new = [1.0, 1.0, 1.0]
        u_new = [{"y": y, "area":  area} for (y, area) in zip(y_new, area_new)]
        state_new_jit, attributions_matrix_jit, image_processing_error_jit = trackingrt_jit.track_trench_rt(
            x_old, u_new, 0,
        )
        self.assertEqual(image_processing_error_jit.error_code.value, ErrorCode.ERROR_TRACK_NO_PREV_STATE.value)
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
        self.assertEqual(image_processing_error_jit.error_code.value, ErrorCode.ERROR_TRACK_NO_INPUTS.value)

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
            # Inputs for JIT wrapper
            u_new = [[{'area': s['area'], 'y': s['y']} for s in roi_state] for roi_state in roi_states]
            x_old = [[{'area': s['area'], 'y': s['y'], 'id': s['id']} for s in roi_state] for roi_state in roi_states]
            for t in range(len(areas_old) - 1):
                attributions_matrix = np.zeros((len(areas_old[t]), len(areas_old[t + 1])), dtype=np.bool_)
                error_code = trackingrt_jit.track_trench_rt_jit(areas_old[t], ids_old[t],
                                                                areas_new[t + 1], ids_new[t + 1],
                                                                divs_new[t + 1], error_codes[t + 1],
                                                                attributions_matrix, nb.int32(max_ids[t]))
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

    def test_tracking_no_jit(self):
        with open(THIS_DIR / "data/pos0_states_v1.pickle", "rb") as file:
            pos0_states = pickle.load(file)
        for i_roi in range(10):
            roi_states = pos0_states[i_roi]
            # Inputs for JIT
            ids_old = [np.array([nb.int32(s['id']) for s in roi_state]) for roi_state in roi_states]
            max_ids = []
            for id_list in ids_old:
                this_max = max(id_list)
                all_max = max(max_ids) if max_ids else -1
                max_ids.append(max(this_max, all_max))

            # Inputs for non-JIT
            u_new = [[{'area': s['area'], 'y': s['y']} for s in roi_state] for roi_state in roi_states]
            x_old = [[{'area': s['area'], 'y': s['y'], 'id': s['id']} for s in roi_state] for roi_state in roi_states]
            for t in range(len(ids_old) - 1):
                state_new_no_jit, _, _ = trackingrt.track_trench_rt(
                    x_old[t], u_new[t + 1], max_ids[t],
                )
                self.assertListEqual([s['id'] for s in state_new_no_jit], ids_old[t + 1].tolist())

    def test_tracking_no_jit_init(self):
        x_old = []
        y_new = [1.0, 2.0, 3.0]
        area_new = [1.0, 1.0, 1.0]
        u_new = [{"y": y, "area":  area} for (y, area) in zip(y_new, area_new)]
        state_new_jit, attributions_matrix_jit, image_processing_error_jit = trackingrt.track_trench_rt(
            x_old, u_new, 0,
        )
        self.assertEqual(image_processing_error_jit.error_code.value, ErrorCode.ERROR_TRACK_NO_PREV_STATE.value)
        x_new = [{"y": u_i["y"], "area": u_i["area"], "id": id_new+1, "div": False,
                  "ErrorCode.value": ErrorCode.ERROR_TRACK_NO_PREV_STATE.value}
                 for id_new, u_i in enumerate(u_new)]
        for (x_new_i, state_new_jit_i) in zip(x_new, state_new_jit):
            self.assertDictEqual(x_new_i, state_new_jit_i)

    def test_tracking_no_jit_no_inputs(self):
        x_old = [{"y": 1.0, "area": 1.0, "id": 1, "div": False, "ErrorCode.value": ErrorCode.NO_ERROR.value}]
        u_new = []
        state_new_jit, attributions_matrix_jit, image_processing_error_jit = trackingrt.track_trench_rt(
            x_old, u_new, 0,
        )
        self.assertEqual(len(state_new_jit), 0)
        self.assertEqual(image_processing_error_jit.error_code.value, ErrorCode.ERROR_TRACK_NO_INPUTS.value)


if __name__ == '__main__':
    unittest.main()
