import numpy as np
import pytest

from evomachine.bindings.binding_types import BindingType
from evomachine.coordinates import Coordinate
from evomachine.peripherals.autofocus import AutofocusConfig
from evomachine.peripherals.camera import CameraConfig, ImageConfigType
from evomachine.peripherals.dmd import DmdConfig
from evomachine.peripherals.filterwheel import FilterWheelConfig
from evomachine.peripherals.leds import LedConfig
from evomachine.peripherals.peripherals import PeripheralConfig
from evomachine.peripherals.photodiode import PhotodiodeConfig
from evomachine.peripherals.stage import StageConfig
from evomachine.types import FilterWheelType, LEDType


def _image_config() -> ImageConfigType:
    """
    Return a small valid image configuration.

    Parameters
    ----------
    None

    Returns
    -------
    ImageConfigType
        Valid image configuration for camera config tests.
    """
    return ImageConfigType(pxl_horiz=8, pxl_vert=6, pxl_dtype=np.dtype("uint16"))


CONFIG_CASES = (
    (
        StageConfig,
        {"binding": BindingType.VIRTUAL, "fov_step_size": 100.0},
        (BindingType.VIRTUAL, 100.0),
    ),
    (
        AutofocusConfig,
        {"binding": BindingType.VIRTUAL},
        (BindingType.VIRTUAL,),
    ),
    (
        CameraConfig,
        {"binding": BindingType.VIRTUAL, "image": _image_config()},
        (BindingType.VIRTUAL, _image_config()),
    ),
    (
        DmdConfig,
        {"binding": BindingType.VIRTUAL},
        (BindingType.VIRTUAL,),
    ),
    (
        LedConfig,
        {"binding": BindingType.VIRTUAL, "available_leds": [LEDType.LED_450_NM]},
        (BindingType.VIRTUAL, [LEDType.LED_450_NM]),
    ),
    (
        FilterWheelConfig,
        {"binding": BindingType.VIRTUAL, "available_filters": [FilterWheelType.FILTER]},
        (BindingType.VIRTUAL, [FilterWheelType.FILTER]),
    ),
    (
        PhotodiodeConfig,
        {"binding": BindingType.VIRTUAL},
        (BindingType.VIRTUAL,),
    ),
)


@pytest.mark.parametrize(("config_type", "kwargs", "_positional_args"), CONFIG_CASES)
def test_device_configs_inherit_peripheral_config(config_type, kwargs, _positional_args):
    """
    Check every device factory config inherits PeripheralConfig.

    Parameters
    ----------
    config_type
        Config dataclass type under test.
    kwargs
        Keyword arguments used to construct the config.
    _positional_args
        Positional argument values used by the keyword-only test.

    Returns
    -------
    None
    """
    config = config_type(**kwargs)

    assert isinstance(config, PeripheralConfig)


@pytest.mark.parametrize(("config_type", "_kwargs", "positional_args"), CONFIG_CASES)
def test_device_configs_are_keyword_only(config_type, _kwargs, positional_args):
    """
    Check device configs reject positional construction.

    Parameters
    ----------
    config_type
        Config dataclass type under test.
    _kwargs
        Keyword arguments used by sibling parametrized tests.
    positional_args
        Positional arguments that should be rejected.

    Returns
    -------
    None
    """
    with pytest.raises(TypeError):
        config_type(*positional_args)


@pytest.mark.parametrize(("config_type", "kwargs", "_positional_args"), CONFIG_CASES)
def test_device_configs_use_inherited_copy(config_type, kwargs, _positional_args):
    """
    Check PeripheralConfig.copy returns a same-type validated copy.

    Parameters
    ----------
    config_type
        Config dataclass type under test.
    kwargs
        Keyword arguments used to construct the config.
    _positional_args
        Positional argument values used by sibling parametrized tests.

    Returns
    -------
    None
    """
    config = config_type(**kwargs)

    copied = config.copy()

    assert type(copied) is config_type
    assert copied == config


def test_device_config_common_validation_comes_from_peripheral_config():
    """
    Check common config fields are validated by PeripheralConfig.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    with pytest.raises(TypeError, match="StageConfig: binding must be BindingType"):
        StageConfig(binding="virtual", fov_step_size=100.0)

    with pytest.raises(TypeError, match="CameraConfig: name must be str or None"):
        CameraConfig(binding=BindingType.VIRTUAL, image=_image_config(), name=123)

    with pytest.raises(TypeError, match="FilterWheelConfig: check_alive must be bool"):
        FilterWheelConfig(
            binding=BindingType.VIRTUAL,
            available_filters=[FilterWheelType.FILTER],
            check_alive="yes",
        )


def test_device_specific_validation_still_runs_after_base_validation():
    """
    Check device-specific config validation still runs.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    with pytest.raises(ValueError, match="StageConfig: fov_step_size must be positive"):
        StageConfig(binding=BindingType.VIRTUAL, fov_step_size=0)

    with pytest.raises(TypeError, match="CameraConfig: image must be ImageConfigType"):
        CameraConfig(binding=BindingType.VIRTUAL, image="image")

    with pytest.raises(ValueError, match="available_filters must not be empty"):
        FilterWheelConfig(binding=BindingType.VIRTUAL, available_filters=[])
