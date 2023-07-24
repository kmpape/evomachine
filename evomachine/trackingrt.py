import cv2
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

import delta

from evomachine.exceptions import ErrorCode, ImageProcessingError

logger = logging.getLogger(__name__)

def is_division(
        area_old: float,
        area_new: float,
        diff_param: float = 0.9,
) -> bool:
    """
    Decides based on the formulate "area_new <= diff_param * area_old" whether the cell divided or not.

    Parameters
    ----------
    area_old: float
        Area of cell at previous time step.
    area_new: float
        Area of cell at current time step.
    diff_param: float
        Parameter for cell division formula. Should be chosen as 0 < diff_param < 1 to give meaningful results.

    Returns
    -------
    cell_division_occurred: bool
        Returns true if a cell division occurred.

    """
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


def track_trench_rt(
        x_old: List[Dict[str, Union[float, int, bool]]],
        u_new: List[Dict[str, float]],
        max_id: int,
) -> Tuple[List[Dict[str, Union[float, int, bool]]], np.typing.NDArray[bool], "ImageProcessingError"]:
    """
    Function for tracking cells in mother machine trenches.

    Parameters
    ----------
    x_old: [{"y":float, "area": float, "id": int}, {...}, ...] sorted by "y" value.
        State of trench at time t-1. First cell of list is expected to be the mother cell.
    u_new: [{"y":float, "area": float}, {...}, ...] sorted by "y" value.
        Output from segmentation at time t. First cell of list is expected to be the mother cell.
    max_id: int
        Highest ID ever used in this trench. New IDs are created as max_id+1, max_id+2 etc.

    Returns
    -------
    x_new: [{"y":float, "area": float, "id": int, "div": bool}, {...}, ...] sorted by "y" value.
        State of trench at time t.
    attributions_matrix: np.typing.NDArray[bool]
        Delta-style attributions matrix.
    image_processing_error: Union[None, ImageProcessingError]
        Returns a non-zero error if a cell division was not detected.

    TODO:
    - what if a new cell disappears?
    - what if we haven't detected division events?
    - what if max_id overflows?
    """
    len_old = len(x_old)
    len_new = len(u_new)
    attributions_matrix = np.zeros((len_old, len_new), dtype=bool)
    x_new = [{"y": u_i["y"], "area": u_i["area"], "id": -1, "div": False}
             for u_i in u_new]
    new_id = max_id + 1
    image_processing_error = ImageProcessingError("", ErrorCode.NO_ERROR)
    logger.debug(f"x_old={x_old}\nu_new={u_new}")

    # Initialise with mother cell
    if x_old and x_new:
        x_new[0]["id"] = x_old[0]["id"]
        x_new[0]["div"] = is_division(x_old[0]["area"], x_new[0]["area"])
        attributions_matrix[0, 0] = True
    else:
        image_processing_error = ImageProcessingError("u_new or x_old is empty.", ErrorCode.ERROR_TRACK_NO_INPUTS)
        return x_new, attributions_matrix, image_processing_error

    # Iterate over remaining cells
    i_old = 1
    for i_new in range(1, len_new):
        if x_new[i_new-1]["div"]:
            x_new[i_new]["id"] = new_id
            attributions_matrix[i_old-1, i_new] = True
            new_id = new_id + 1
        elif i_old >= len_old:
            x_new[i_new]["id"] = new_id
            new_id = new_id + 1
            image_processing_error = ImageProcessingError("Divisions not detected. Assigning new IDs instead.",
                                                          ErrorCode.ERROR_TRACK_DIV_NOT_DETECTED)
        else:
            x_new[i_new]["id"] = x_old[i_old]["id"]
            x_new[i_new]["div"] = is_division(x_old[i_old]["area"], x_new[i_new]["area"])
            attributions_matrix[i_old, i_new] = True
            i_old = i_old + 1
    logger.debug(f"x_new={x_new}")

    return x_new, attributions_matrix, image_processing_error

