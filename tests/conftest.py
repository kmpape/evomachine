"""Pytest configuration for evomachine binding tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests import binding_cases
from tests.binding_test_config import BindingTestConfig


TEST_CACHE_DIR = Path(os.environ.get("EVOMACHINE_TEST_CACHE_DIR", "/tmp/evomachine_test_cache"))
TEST_CACHE_DIR.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("XDG_CACHE_HOME", str(TEST_CACHE_DIR))


@pytest.fixture(scope="session")
def binding_test_config() -> BindingTestConfig:
    """
    Return the session binding test configuration.

    Parameters
    ----------
    None

    Returns
    -------
    BindingTestConfig
        Loaded fake/real binding test configuration.
    """
    return BindingTestConfig.load()


@pytest.fixture(scope="session")
def stage_cases(binding_test_config: BindingTestConfig) -> list[binding_cases.StageBindingCase]:
    """
    Return configured stage binding cases.

    Parameters
    ----------
    binding_test_config
        Session binding test configuration.

    Returns
    -------
    list[binding_cases.StageBindingCase]
        Stage cases selected by the config file.
    """
    return binding_cases.stage_cases(binding_test_config)


@pytest.fixture(scope="session")
def filterwheel_cases(binding_test_config: BindingTestConfig) -> list[binding_cases.FilterWheelBindingCase]:
    """
    Return configured filter wheel binding cases.

    Parameters
    ----------
    binding_test_config
        Session binding test configuration.

    Returns
    -------
    list[binding_cases.FilterWheelBindingCase]
        Filter wheel cases selected by the config file.
    """
    return binding_cases.filterwheel_cases(binding_test_config)


@pytest.fixture(scope="session")
def peripheral_cases(binding_test_config: BindingTestConfig) -> list[binding_cases.FactoryBindingCase]:
    """
    Return configured peripheral controller binding cases.

    Parameters
    ----------
    binding_test_config
        Session binding test configuration.

    Returns
    -------
    list[binding_cases.FactoryBindingCase]
        Peripheral controller cases selected by the config file.
    """
    return binding_cases.peripheral_cases(binding_test_config)


@pytest.fixture(scope="session")
def led_cases(binding_test_config: BindingTestConfig) -> list[binding_cases.BindingCase]:
    """
    Return configured LED binding cases.

    Parameters
    ----------
    binding_test_config
        Session binding test configuration.

    Returns
    -------
    list[binding_cases.BindingCase]
        LED cases selected by the config file.
    """
    return binding_cases.simple_cases(binding_test_config, binding_test_config.led_bindings, "led")


@pytest.fixture(scope="session")
def dmd_cases(binding_test_config: BindingTestConfig) -> list[binding_cases.BindingCase]:
    """
    Return configured DMD binding cases.

    Parameters
    ----------
    binding_test_config
        Session binding test configuration.

    Returns
    -------
    list[binding_cases.BindingCase]
        DMD cases selected by the config file.
    """
    return binding_cases.simple_cases(binding_test_config, binding_test_config.dmd_bindings, "dmd")


@pytest.fixture(scope="session")
def camera_cases(binding_test_config: BindingTestConfig) -> list[binding_cases.BindingCase]:
    """
    Return configured camera binding cases.

    Parameters
    ----------
    binding_test_config
        Session binding test configuration.

    Returns
    -------
    list[binding_cases.BindingCase]
        Camera cases selected by the config file.
    """
    return binding_cases.simple_cases(binding_test_config, binding_test_config.camera_bindings, "camera")


@pytest.fixture(scope="session")
def autofocus_cases(binding_test_config: BindingTestConfig) -> list[binding_cases.BindingCase]:
    """
    Return configured autofocus binding cases.

    Parameters
    ----------
    binding_test_config
        Session binding test configuration.

    Returns
    -------
    list[binding_cases.BindingCase]
        Autofocus cases selected by the config file.
    """
    return binding_cases.simple_cases(binding_test_config, binding_test_config.autofocus_bindings, "autofocus")


@pytest.fixture(scope="session")
def photodiode_cases(binding_test_config: BindingTestConfig) -> list[binding_cases.BindingCase]:
    """
    Return configured photodiode binding cases.

    Parameters
    ----------
    binding_test_config
        Session binding test configuration.

    Returns
    -------
    list[binding_cases.BindingCase]
        Photodiode cases selected by the config file.
    """
    return binding_cases.simple_cases(binding_test_config, binding_test_config.peripheral_bindings, "photodiode")
