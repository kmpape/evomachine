"""Pytest configuration for evomachine binding tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

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
