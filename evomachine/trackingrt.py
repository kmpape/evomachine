import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

import delta

from evomachine.exceptions import ErrorCode, ImageProcessingError

def is_division(
        area_old: float,
        area_new: float,
        diff_param: float=0.9,
) -> bool:
    return area_new <= diff_param * area_old


def get_tracking_inputs_rt(seg_mask: np.typing.NDArray[np.uint8]) -> List[Dict[str, float]]:
    """
    Computes the inputs for track_trench_mothermachine.

    Parameters
    ----------
    seg_mask: Output from ROI.get_frame()

    Returns
    -------
    tracking_inputs: [{"y":float, "area": float}, {...}, ...] sorted by "y" value.
    """
    contours = delta.utils.find_contours(seg_mask)
    areas = [cv2.contourArea(contour) for contour in contours]
    centers = [contour[:, 0, 1].min() + contour[:, 0, 1].max() for contour in contours]  # *0.5 not needed
    tracking_inputs = [{"y": center, "area":  area} for (center, area) in zip(centers, areas)]
    tracking_inputs.sort(key=lambda x: x["y"], reverse=False)  # assuming that the mother cell is on top
    return tracking_inputs


def initialise_tracking(seg_mask: np.typing.NDArray[np.uint8]) -> List[Dict[str, Union[float, int]]]:
    """
    Computes the output of track_trench_mothermachine for the first frame.

    Parameters
    ----------
    seg_mask: Output from ROI.get_frame()

    Returns
    -------
    tracking_state: [{"y":float, "area": float, "id": int}, {...}, ...] sorted by "y" value.
    """
    tracking_inputs = get_tracking_inputs_rt(seg_mask)
    for cell_id, tracking_input in enumerate(tracking_inputs):
        tracking_input["id"] = cell_id  # modifies the list in-place
    return tracking_inputs


def track_trench_mothermachine(
        x_old: List[Dict[str, Union[float, int]]],
        u_new: List[Dict[str, float]],
        max_id: int,
) -> Tuple[List[Dict[str, Union[float, int]]], Union[None, "ImageProcessingError"]]:
    """
    Parameters
    ----------
    x_old: [{"y":float, "area": float, "id": int}, {...}, ...] sorted by "y" value.
    State of trench at time t-1. First cell of list is expected to be the mother cell.
    u_new: [{"y":float, "area": float}, {...}, ...] sorted by "y" value.
    Output from segmentation at time t.
    max_id: int
    Highest ID ever used in this trench. New IDs are created as max_id+1, max_id+2 etc.

    Returns
    -------
    x_new: [{"y":float, "area": float, "id": int, "div": bool}, {...}, ...] sorted by "y" value.
    State of trench at time t.

    TODO:
    - what if a new cell disappears?
    - what if we haven't detected division events?
    - what if max_id overflows?
    """
    len_old = len(x_old)
    x_new = [{"y": u_i["y"], "area": u_i["area"], "id": -1, "div": False} for u_i in u_new]
    new_id = max_id + 1
    image_processing_error = None

    # Initialise with mother cell
    x_new[0]["id"] = x_old[0]["id"]
    x_new[0]["div"] = is_division(x_old[0]["div"], x_new[0]["div"])

    # Iterate over remaining cells
    i_old = 2
    for i_new, u_i in enumerate(x_new[2:], start=2):
        if x_new[i_new-1]["div"]:
            x_new[i_new]["id"] = new_id
            new_id = new_id + 1
        elif i_old > len_old:
            x_new[i_new]["id"] = new_id
            new_id = new_id + 1
            image_processing_error = ImageProcessingError("Divisions not detected. Assigning new IDs instead.",
                                                          ErrorCode.ERROR_TRACK_DIV_NOT_DETECTED)
        else:
            x_new[i_new]["id"] = x_old[i_old]["id"]
            x_new[i_new]["div"] = is_division(x_old[i_old]["div"], x_new[i_new]["div"])
            i_old = i_old + 1

    return x_new, image_processing_error

