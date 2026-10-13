from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse


OPENAI_COMPATIBLE_PROVIDER = "openai-compatible"
ANTHROPIC_PROVIDER = "anthropic"
SUPPORTED_AUTOSTRAT_PROVIDERS = (
    OPENAI_COMPATIBLE_PROVIDER,
    ANTHROPIC_PROVIDER,
)
CUSTOM_PRESET_ID = "custom"


@dataclass(frozen=True, slots=True)
class AutoStratConnectionPreset:
    """Non-secret model connection values offered by the startup dialog."""

    preset_id: str
    label: str
    provider: str
    model: str
    base_url: str | None = None


ROBIN_OFFICE_2_PRESET = AutoStratConnectionPreset(
    preset_id="robin-office-2",
    label="Robin Office 2",
    provider=OPENAI_COMPATIBLE_PROVIDER,
    model="qwen3.6",
    base_url="https://robin-office-2.tail32bb7.ts.net/v1",
)

AUTOSTRAT_CONNECTION_PRESETS = (ROBIN_OFFICE_2_PRESET,)


def autostrat_preset(preset_id: str) -> AutoStratConnectionPreset | None:
    return next(
        (
            preset
            for preset in AUTOSTRAT_CONNECTION_PRESETS
            if preset.preset_id == preset_id
        ),
        None,
    )


@dataclass(frozen=True, slots=True)
class AutoStratModelConfig:
    """Session-only credentials and provider settings for strategy generation."""

    provider: str
    model: str
    api_key: str = field(default="", repr=False)
    base_url: str | None = None
    preset_id: str = CUSTOM_PRESET_ID

    def __post_init__(self) -> None:
        provider = self.provider.strip().lower()
        model = self.model.strip()
        api_key = self.api_key.strip()
        base_url = self.base_url.strip() if isinstance(self.base_url, str) else None
        base_url = base_url or None
        if provider not in SUPPORTED_AUTOSTRAT_PROVIDERS:
            raise ValueError(
                "AutoStrat provider must be openai-compatible or anthropic."
            )
        if not model:
            raise ValueError("AutoStrat model must be non-empty.")
        if ":" in model:
            raise ValueError(
                "Enter the model identifier without a provider prefix; select the provider separately."
            )
        if base_url is not None:
            parsed = urlparse(base_url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError("AutoStrat base URL must be a valid HTTP(S) URL.")

        preset_id = self.preset_id.strip() or CUSTOM_PRESET_ID
        preset = autostrat_preset(preset_id)
        if preset is None or (
            provider != preset.provider
            or model != preset.model
            or base_url != preset.base_url
        ):
            preset_id = CUSTOM_PRESET_ID

        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "api_key", api_key)
        object.__setattr__(self, "base_url", base_url)
        object.__setattr__(self, "preset_id", preset_id)

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    @property
    def qualified_model(self) -> str:
        prefix = (
            "openai-chat"
            if self.provider == OPENAI_COMPATIBLE_PROVIDER
            else "anthropic"
        )
        return f"{prefix}:{self.model}"

    def public_payload(self) -> dict[str, Any]:
        """Return connection metadata with credentials deliberately omitted."""
        return {
            "enabled": self.enabled,
            "preset": self.preset_id,
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "AutoStratModelConfig":
        """Build a config while keeping key-only legacy clients on the Robin preset."""
        preset_id = payload.get("preset", ROBIN_OFFICE_2_PRESET.preset_id)
        if not isinstance(preset_id, str):
            raise TypeError("AutoStrat preset must be text.")
        preset = autostrat_preset(preset_id) or ROBIN_OFFICE_2_PRESET
        values = {
            "provider": payload.get("provider", preset.provider),
            "model": payload.get("model", preset.model),
            "api_key": payload.get("api_key", ""),
            "base_url": payload.get("base_url", preset.base_url),
            "preset_id": preset_id,
        }
        for name in ("provider", "model", "api_key"):
            if not isinstance(values[name], str):
                raise TypeError(f"AutoStrat {name.replace('_', ' ')} must be text.")
        if values["base_url"] is not None and not isinstance(values["base_url"], str):
            raise TypeError("AutoStrat base URL must be text or null.")
        return cls(**values)
