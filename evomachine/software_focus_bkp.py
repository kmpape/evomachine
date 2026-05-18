import cv2
import numpy as np

from delta.utils import CroppingBox

from evomachine.exceptions import ConfigError, ErrorCode
from evomachine.types import FocusAlgorithmType, FocusCurveType


# TODO(CODEX) START
# Following comments apply to refactor this file and include functions from acquisition.py.
# - First, create a backup-copy software_focus_bkp.py before starting the refactor, so we can easily refer back to the original code and make sure we don't miss any important functions or logic. Fix the existing imports of software_focus.py to use software_focus_bkp.py, so we can make sure the existing code continues to work while we refactor.
# - Rename this file as softwarefocus.py.
# - The SoftwareFocus class will need to take the DMD, LEDManager, Camera, Autofocus, and Stage instances as constructor parameters, as it will need to control all of these components during the focus routines. We can pass these instances from the main code when we create the SoftwareFocus instance.
# - Make a SoftwareFocus class that encapsulates all focus-related functions and state.
# - Similar to stage.py, it should offer to initialise with a list of position IDs, and keep the software focus state for each of those.
# - Refactor ConfigFocus and call it SoftwareFocusConfig. Leave it in config_types.py, make it a constructor parameter.
# - Implement a function to update the SoftwareFocusConfig as this can happen through the GUI.
# - Make an abstract class SoftwareFocusAlgorithm as a template for the focus routines, and implement the current software focus routines as a child classes of it. This will allow us to easily add new focus routines in the future.
# - Move all specific focus computation routines/SoftwareFocusAlgorithms to bindings/software_focus/software_focus_algorithms.py. 
# - Define a function in the bindings file that takes a SoftwareFocusAlgorithmType and returns the corresponding SoftwareFocusAlgorithm instance. This will allow us to easily call the focus algorithms from the main code without having to import all the specific algorithm classes. Allow it to take a SoftwareFocusConfig as an argument to pass any necessary parameters to the algorithm instance.
# - Move get_focus_score and get_roi_focus_score to this class, and make them instance
# - Make sure that after refactor there is a function that one can use to compute the focus score for a given image and algorithm, and that it can be easily called from the main code and GUI.
# - Already start pulling a Stop event through the focus routines to allow them to be stopped early if needed.
# - Remove the evomachine-specific exceptions. Use standard exceptions.
# - Add type hints to all functions and classes. Also add doc strings.
# - Extend/Refactor SoftwareFocusConfig to use ConfigFrame or a list of ConfigFrames, in which case it averages the images.
# - Allow SoftwareFocusConfig to be updated during runtime.
# TODO (CODEX) END

DEFAULT_SQUARED_GRAD_THRESHOLD = 0

def get_focus_score_is_good(focus_curve: np.array) -> bool:  # noqa
    return get_focus_curve_type(focus_curve=focus_curve) == FocusCurveType.HAS_GLOBAL_MAXIMUM


def get_focus_curve_type(focus_curve: np.array) -> FocusCurveType:
    if focus_curve.size < 3:
        return FocusCurveType.UNKNOWN  # Not enough data to analyze properly

    max_indices = np.where(focus_curve == np.max(focus_curve))[0]
    num_maxima = len(max_indices)

    # Check for boundary maximum
    if 0 in max_indices or len(focus_curve) - 1 in max_indices:
        return FocusCurveType.HAS_BOUNDARY_MAXIMUM

    # Check for single global maximum
    if num_maxima == 1:
        return FocusCurveType.HAS_GLOBAL_MAXIMUM

    # Check for multiple maxima
    if num_maxima > 1:
        return FocusCurveType.HAS_MAXIMA

    return FocusCurveType.UNKNOWN


def get_roi_focus_score(
        img: np.array,
        algorithm: FocusAlgorithmType,
        boxes: list[CroppingBox],
        threshold: float | None = None,
        rowshift: int = 25,
        colshift: int = 50,
        normalise_score: bool = False,
) -> float:
    total_score = 0
    for box in boxes:
        if algorithm == FocusAlgorithmType.LAPLACIAN_VAR:
            total_score += get_focus_score_laplacian_var(img=box.crop(img))
        elif algorithm == FocusAlgorithmType.SQUARED_GRAD_AVG:
            total_score += get_focus_score_squared_gradient(img=box.crop(img), threshold=threshold)
        elif algorithm == FocusAlgorithmType.STEEL:
            total_score += get_focus_score_steel(img=box.crop(img), rowshift=rowshift, colshift=colshift, normalise=normalise_score)
        else:
            raise ConfigError(f"get_focus_score: Unknown focus algorithm: {algorithm}.", ErrorCode.ERROR_FOCUS_CONFIG)
    return total_score


def get_focus_score(
        img: np.array,
        algorithm: FocusAlgorithmType,
        threshold: float | None = None,
        rowshift: int = 25,
        colshift: int = 50,
        normalise_score: bool = False,
) -> float:
    if algorithm == FocusAlgorithmType.LAPLACIAN_VAR:
        return get_focus_score_laplacian_var(img=img)
    elif algorithm == FocusAlgorithmType.SQUARED_GRAD_AVG:
        return get_focus_score_squared_gradient(img=img, threshold=threshold)
    elif algorithm == FocusAlgorithmType.STEEL:
        return get_focus_score_steel(img=img, rowshift=rowshift, colshift=colshift, normalise=normalise_score)
    else:
        raise ConfigError(f"get_focus_score: Unknown focus algorithm: {algorithm}.", ErrorCode.ERROR_FOCUS_CONFIG)


def get_focus_score_laplacian_var(img: np.array) -> float:
    lap = cv2.Laplacian(img, cv2.CV_64F)
    return (lap[1:-1, 1:-1]**2).var()


def get_focus_score_squared_gradient(img: np.array, threshold: float | None = None) -> float:
    threshold = DEFAULT_SQUARED_GRAD_THRESHOLD if threshold is None else threshold
    tmp = abs(img[:, 1:] - img[:, :-1])
    tmp[tmp < threshold] = 0
    return (tmp**2).mean()


def get_focus_score_steel(img: np.array, rowshift: int, colshift: int, normalise=False):
    img_float = img.astype(float)
    img_row = np.multiply(
        np.abs(img_float - np.roll(img_float, -rowshift, axis=0)),
        np.abs(img_float - np.roll(img_float, +rowshift, axis=0)),
    )
    img_col = np.multiply(
        np.abs(img_float - np.roll(img_float, -colshift, axis=1)),
        np.abs(img_float - np.roll(img_float, +colshift, axis=1)),
    )
    if normalise:
        return (img_row.sum() + img_col.sum()) / float(2 * img.shape[0] * img.shape[1])
    else:
        return img_row.sum() + img_col.sum()
