import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numba as nb
import numpy as np

import delta

from evomachine.exceptions import ErrorCode, ImageProcessingError

logger = logging.getLogger(__name__)

# Custom error codes to allow for JIT compilation
NO_ERROR = 0
ERROR_TRACK_NO_INPUTS = 1
ERROR_TRACK_DIV_NOT_DETECTED = 2
ERROR_NO_PREV_STATE = 3
ERROR = 4
ERROR_MAP = {
    NO_ERROR: ErrorCode.NO_ERROR,
    ERROR: ErrorCode.ERROR,
    ERROR_TRACK_NO_INPUTS: ErrorCode.ERROR_TRACK_NO_INPUTS,
    ERROR_TRACK_DIV_NOT_DETECTED: ErrorCode.ERROR_TRACK_DIV_NOT_DETECTED,
    ERROR_NO_PREV_STATE: ErrorCode.ERROR_TRACK_NO_PREV_STATE,
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
    areas = [cv2.contourArea(contour) for contour in contours]  # *0.5 not needed
    centers = [contour[:, 0, 1].min() + contour[:, 0, 1].max() for contour in contours]
    tracking_inputs = [{"y": y, "area":  area} for (y, area) in zip(centers, areas)]
    tracking_inputs.sort(key=lambda x: x["y"], reverse=False)  # assuming that the mother cell is on top
    return tracking_inputs


def init_track_trench_rt(
        seg_mask: np.typing.NDArray[np.uint8]
) -> Tuple[List[Dict[str, Union[float, int]]], np.typing.NDArray[bool]]:
    """
    Computes the output of track_trench_rt for the first frame.

    Parameters
    ----------
    seg_mask: np.typing.NDArray[np.uint8])
        Output from ROI.get_frame().

    Returns
    -------
    x_new: [{"y":float, "y_new":float, "y_old":float, "area": float, "id": int}, {...}, ...] sorted by "y"
        y is the y-coordinate of the "center", i.e. y_new+y_old (Note: multiply by 0.5 to get the real coordinate).
    attributions_matrix: np.typing.NDArray[bool]
        Delta-style attributions matrix.
    """
    x_new = get_tracking_inputs_rt(seg_mask)
    for cell_id, x_i in enumerate(x_new, start=1):  # modifies the list in-place
        x_i["id"] = cell_id
        x_i["div"] = False
    attributions_matrix = np.empty((0, len(x_new)), dtype=bool)
    logger.debug(f"x_new={x_new}")
    return x_new, attributions_matrix


@nb.njit(nb.int32[:](nb.float32[:], nb.float32[:], nb.int32[:],
                     nb.float32[:], nb.float32[:], nb.int32[:],
                     nb.boolean[:],
                     nb.int32[:],
                     nb.boolean[:, :],
                     np.int32))
def track_trench_rt(
        y_old: List[float],
        area_old: List[float],
        id_old: List[int],
        y_new: List[float],
        area_new: List[float],
        id_new: List[int],
        div_new: List[bool],
        error_codes: List[int],
        attributions_matrix: np.typing.NDArray[(Any, Any), np.bool_],
        max_id: int,
) -> int:
    len_old = len(id_old)
    len_new = len(id_new)
    new_id = max_id + 1

    # Handle special cases, TODO: move elsewhere
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
    attributions_matrix[0, 0] = True

    # Loop over remaining cells
    i_old = 1
    for i_new in range(1, len_new):
        if div_new[i_new]:
            id_new[i_new] = new_id
            new_id = new_id + 1
            attributions_matrix[i_old - 1, i_new] = True
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
