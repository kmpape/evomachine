import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numba as nb
import numpy as np

import delta

from evomachine.exceptions import ErrorCode, ImageProcessingError

logger = logging.getLogger(__name__)

# Custom error codes to allow for JIT compilation
NO_ERROR = np.int32(0)
ERROR_TRACK_NO_INPUTS = np.int32(1)
ERROR_TRACK_DIV_NOT_DETECTED = np.int32(2)
ERROR_NO_PREV_STATE = np.int32(3)
ERROR = np.int32(4)
ERROR_MAP = {
    NO_ERROR: ErrorCode.NO_ERROR.value,
    ERROR: ErrorCode.ERROR.value,
    ERROR_TRACK_NO_INPUTS: ErrorCode.ERROR_TRACK_NO_INPUTS.value,
    ERROR_TRACK_DIV_NOT_DETECTED: ErrorCode.ERROR_TRACK_DIV_NOT_DETECTED.value,
    ERROR_NO_PREV_STATE: ErrorCode.ERROR_TRACK_NO_PREV_STATE.value,
}


@nb.njit(nb.boolean(nb.float32, nb.float32))
def is_division(
        area_old: float,
        area_new: float,
) -> bool:
    """
    Decides based on the formulate "area_new <= diff_param * area_old" whether the cell divided or not.

    Parameters
    ----------
    area_old: float
        Area of cell at previous time step.
    area_new: float
        Area of cell at current time step.

    Returns
    -------
    cell_division_occurred: bool
        Returns true if a cell division occurred.

    """
    diff_param: float = 0.9
    cell_division_occurred = area_new <= diff_param * area_old
    return cell_division_occurred


def get_tracking_inputs_rt(seg_mask: np.typing.NDArray[np.uint8]) -> List[Dict[str, float]]:
    """
    Computes the inputs for track_trench_rt. Assumes that the cells grow from top to bottom.

    Parameters
    ----------
    seg_mask: np.typing.NDArray[np.uint8])
        Output from ROI.get_frame().

    Returns
    -------
    tracking_inputs: [{"y": float, "y_min": float, "y_max": float, "area": float}, {...}, ...] sorted by "y" value.
        y is the y-coordinate of the "center", i.e. y_new+y_old (Note: multiply by 0.5 to get the real coordinate).
    """
    contours = delta.utils.find_contours(seg_mask)
    areas = [cv2.contourArea(contour) for contour in contours]
    y_min_max = [(contour[:, 0, 1].min(), contour[:, 0, 1].max()) for contour in contours]
    tracking_inputs = [{"y": y[0] + y[1], "y_min": y[0], "y_max": y[1], "area":  area}
                       for (y, area) in zip(y_min_max, areas)]
    tracking_inputs.sort(key=lambda x: x["y"], reverse=False)  # assuming that the mother cell is on top
    return tracking_inputs


def track_trench_rt(
        x_old: List[Dict[str, Union[float, int, bool]]],
        u_new: List[Dict[str, float]],
        max_id: int,
) -> Tuple[List[Dict[str, Union[float, int, bool]]], np.typing.NDArray[bool], ImageProcessingError]:
    area_old = np.array([x['area'] for x in x_old], dtype=np.float32)
    id_old = np.array([x['id'] for x in x_old], dtype=np.int32)
    area_new = np.array([u['area'] for u in u_new], dtype=np.float32)
    id_new = np.array([-1 for u in u_new], dtype=np.int32)
    div_new = np.array([False for u in u_new], dtype=np.bool_)
    error_codes = np.array([NO_ERROR for u in u_new], dtype=np.int32)
    attributions_matrix = np.zeros((len(x_old), len(u_new)), dtype=np.bool_)
    error_code = track_trench_rt_jit(
        area_old, id_old, area_new, id_new, div_new, error_codes, attributions_matrix, np.int32(max_id),
    )
    x_new = [{**u_i, **{"id": id_i, "div": div_i, "ErrorCode.value": ERROR_MAP[e_i]}}
             for (u_i, id_i, div_i, e_i) in zip(u_new, id_new, div_new, error_codes)]
    image_processing_error = ImageProcessingError(message="", error_code=ErrorCode(ERROR_MAP[error_code]))
    return x_new, attributions_matrix, image_processing_error


@nb.njit(nb.int32(nb.float32[:], nb.int32[:], nb.float32[:], nb.int32[:], nb.boolean[:], nb.int32[:], nb.boolean[:, :],
                  nb.int32))
def track_trench_rt_jit(
        area_old: np.typing.NDArray[nb.float32],
        id_old: np.typing.NDArray[nb.int32],
        area_new: np.typing.NDArray[nb.float32],
        id_new: np.typing.NDArray[nb.int32],
        div_new: np.typing.NDArray[nb.bool_],
        error_codes: np.typing.NDArray[nb.int32],
        attributions_matrix: np.typing.NDArray[np.bool_],
        max_id: np.int32,
) -> np.int32:
    len_old = len(id_old)
    len_new = len(id_new)
    new_id = max_id + 1

    # Handle special cases
    if len_old == 0:
        for i_new in range(len_new):
            id_new[i_new] = new_id + i_new
            error_codes[i_new] = ERROR_NO_PREV_STATE
        return ERROR_NO_PREV_STATE
    elif len_new == 0:
        return ERROR_TRACK_NO_INPUTS

    # Initialise mother cell
    id_new[0] = id_old[0]
    div_new[0] = is_division(area_old[0], area_new[0])
    attributions_matrix[0] = True

    # Loop over remaining cells
    i_old = 1
    for i_new in range(1, len_new):
        if div_new[i_new-1]:
            id_new[i_new] = new_id
            new_id = new_id + 1
            attributions_matrix[i_old-1, i_new] = True
        elif i_old >= len_old:
            id_new[i_new] = new_id
            new_id = new_id + 1
            error_codes[i_new] = ERROR_TRACK_DIV_NOT_DETECTED
        else:
            id_new[i_new] = id_old[i_old]
            div_new[i_new] = is_division(area_old[i_old], area_new[i_new])
            attributions_matrix[i_old, i_new] = True
            i_old = i_old + 1

    if sum(error_codes):
        return ERROR
    else:
        return NO_ERROR
