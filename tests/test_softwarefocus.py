import numpy as np
import pytest
from delta.utils import CroppingBox

from evomachine import software_focus_bkp
from evomachine.bindings.software_focus.software_focus_algorithms import (
    LaplacianVarianceFocusAlgorithm,
    SoftwareFocusAlgorithm,
    SquaredGradientAverageFocusAlgorithm,
    SteelFocusAlgorithm,
)
from evomachine.config_types import (
    ConfigFocus,
    ConfigFocusFactory,
    ConfigFrame,
    SoftwareFocusConfig,
    SoftwareFocusConfigFactory,
)
from evomachine.softwarefocus import (
    create_software_focus_algorithm,
    get_focus_curve_type,
    get_focus_score,
    get_focus_score_is_good,
    get_roi_focus_score,
)
from evomachine.types import FocusAlgorithmType, FocusCurveType, LEDType


def _image() -> np.ndarray:
    """
    Return a deterministic image for software focus score tests.

    Parameters
    ----------
    None

    Returns
    -------
    np.ndarray
        Small deterministic image with non-zero gradients.
    """
    return np.arange(36, dtype=np.float64).reshape(6, 6)


def _config(**kwargs) -> SoftwareFocusConfig:
    """
    Return a valid SoftwareFocusConfig with optional field overrides.

    Parameters
    ----------
    **kwargs
        SoftwareFocusConfig field values to override.

    Returns
    -------
    SoftwareFocusConfig
        Valid software focus configuration.
    """
    values = {
        "exposure_time": 200,
        "focus_channel": LEDType.LED_450_NM,
        "brightness": 29,
        "rel_range": 50,
        "step_size": 5,
    }
    values.update(kwargs)
    return SoftwareFocusConfig(**values)


def test_backup_module_exposes_legacy_focus_helpers() -> None:
    """
    Check that the backup module preserves the legacy helper functions.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    img = _image()

    assert software_focus_bkp.get_focus_score_laplacian_var(img) >= 0
    assert software_focus_bkp.get_focus_score_squared_gradient(img) >= 0
    assert software_focus_bkp.get_focus_score_steel(img, rowshift=1, colshift=2) >= 0


def test_algorithm_classes_match_backup_scores() -> None:
    """
    Check that new algorithm classes match the backed-up implementations.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    img = _image()

    assert LaplacianVarianceFocusAlgorithm().score_image(img) == software_focus_bkp.get_focus_score_laplacian_var(img)
    assert SquaredGradientAverageFocusAlgorithm(threshold=0).score_image(img) == software_focus_bkp.get_focus_score_squared_gradient(img, threshold=0)
    assert SteelFocusAlgorithm(rowshift=1, colshift=2).score_image(img) == software_focus_bkp.get_focus_score_steel(img, rowshift=1, colshift=2)


def test_top_level_focus_score_helpers_match_backup_scores() -> None:
    """
    Check that new top-level helpers preserve score behavior.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    img = _image()

    assert get_focus_score(img, FocusAlgorithmType.LAPLACIAN_VAR) == software_focus_bkp.get_focus_score(img, FocusAlgorithmType.LAPLACIAN_VAR)
    assert get_focus_score(img, FocusAlgorithmType.SQUARED_GRAD_AVG, threshold=2) == software_focus_bkp.get_focus_score(img, FocusAlgorithmType.SQUARED_GRAD_AVG, threshold=2)
    assert get_focus_score(img, FocusAlgorithmType.STEEL, rowshift=1, colshift=2) == software_focus_bkp.get_focus_score(img, FocusAlgorithmType.STEEL, rowshift=1, colshift=2)


def test_roi_focus_score_matches_backup_score() -> None:
    """
    Check that ROI scoring sums the same crop scores as the backup helper.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    img = _image()
    boxes = [
        CroppingBox(xtl=0, xbr=3, ytl=0, ybr=3),
        CroppingBox(xtl=2, xbr=6, ytl=2, ybr=6),
    ]

    assert get_roi_focus_score(
        img=img,
        algorithm=FocusAlgorithmType.STEEL,
        boxes=boxes,
        rowshift=1,
        colshift=2,
    ) == software_focus_bkp.get_roi_focus_score(
        img=img,
        algorithm=FocusAlgorithmType.STEEL,
        boxes=boxes,
        rowshift=1,
        colshift=2,
    )


def test_algorithm_factory_selects_algorithms_and_uses_config() -> None:
    """
    Check algorithm factory type selection and config parameter use.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    config = _config(rowshift_px=1, colshift_px=2)

    assert isinstance(create_software_focus_algorithm(FocusAlgorithmType.LAPLACIAN_VAR), SoftwareFocusAlgorithm)
    assert isinstance(create_software_focus_algorithm(FocusAlgorithmType.SQUARED_GRAD_AVG), SquaredGradientAverageFocusAlgorithm)
    steel = create_software_focus_algorithm(FocusAlgorithmType.STEEL, config=config)

    assert isinstance(steel, SteelFocusAlgorithm)
    assert steel.rowshift == 1
    assert steel.colshift == 2
    with pytest.raises(TypeError):
        create_software_focus_algorithm("STEEL")
    with pytest.raises(TypeError):
        create_software_focus_algorithm(FocusAlgorithmType.STEEL, config="bad")


def test_focus_curve_helpers_match_backup_behavior() -> None:
    """
    Check focus curve classification helpers.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    focus_curve = np.array([1.0, 3.0, 2.0])

    assert get_focus_curve_type(focus_curve) == FocusCurveType.HAS_GLOBAL_MAXIMUM
    assert get_focus_score_is_good(focus_curve)
    assert get_focus_curve_type(focus_curve) == software_focus_bkp.get_focus_curve_type(focus_curve)


def test_software_focus_config_alias_factory_and_updates() -> None:
    """
    Check SoftwareFocusConfig validation, aliases, factories, and updates.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    config = _config()
    frame = ConfigFrame(leds={LEDType.LED_450_NM: 50}, filter_wheel=None, exposure=100)
    updated = config.updated(brightness=10, focus_frames=[frame])

    assert ConfigFocus is SoftwareFocusConfig
    assert isinstance(ConfigFocusFactory.default_config(), SoftwareFocusConfig)
    assert isinstance(SoftwareFocusConfigFactory.default_config(), SoftwareFocusConfig)
    assert updated.brightness == 10
    assert updated.focus_frames == [frame]
    assert config.brightness == 29
    assert config.update_from_mapping({"step_size": 10}).step_size == 10
    with pytest.raises(ValueError):
        config.updated(unknown=1)
    with pytest.raises(TypeError):
        config.update_from_mapping([("brightness", 10)])
    with pytest.raises(TypeError):
        _config(focus_frames=["bad"])
