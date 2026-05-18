"""Typed loader for evomachine binding test configuration."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path


@dataclass
class BindingTestConfig:
    """Configuration for fake and real binding tests."""

    use_real_bindings: bool
    stage_bindings: list[str]
    led_bindings: list[str]
    filterwheel_bindings: list[str]
    dmd_bindings: list[str]
    camera_bindings: list[str]
    autofocus_bindings: list[str]
    peripheral_bindings: list[str]

    def __post_init__(self) -> None:
        """
        Validate binding test configuration.

        Parameters
        ----------
        None

        Returns
        -------
        None
            The dataclass fields are validated in place.
        """
        if not isinstance(self.use_real_bindings, bool):
            raise TypeError("BindingTestConfig: use_real_bindings must be bool.")
        for field_name in [
            "stage_bindings",
            "led_bindings",
            "filterwheel_bindings",
            "dmd_bindings",
            "camera_bindings",
            "autofocus_bindings",
            "peripheral_bindings",
        ]:
            value = getattr(self, field_name)
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                raise TypeError(f"BindingTestConfig: {field_name} must be list[str].")

    @classmethod
    def load(cls, path: Path | None = None) -> "BindingTestConfig":
        """
        Load binding test configuration from JSON.

        Parameters
        ----------
        path
            Optional config file path. If None, EVOMACHINE_TEST_CONFIG or the
            checked-in default config is used.

        Returns
        -------
        BindingTestConfig
            Validated binding test configuration.
        """
        config_path = path or Path(
            os.environ.get("EVOMACHINE_TEST_CONFIG", Path(__file__).with_name("binding_test_config.json"))
        )
        with open(config_path, encoding="utf-8") as file:
            return cls(**json.load(file))
