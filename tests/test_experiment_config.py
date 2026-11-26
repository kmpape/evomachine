from __future__ import annotations

import logging
from pathlib import Path

import pytest
from pydantic import ValidationError

from evomachine.acquisition import FrameAcquisitionSettings
from evomachine.config import TimeConfig
from evomachine.experiment_config import (
    SCHEMA_VERSION,
    ExperimentConfig,
    ExperimentConfigError,
    OutputConfig,
)
from evomachine.filemanager import FileNameConfig

EXAMPLE_FILE = Path(__file__).resolve().parents[1] / "experiments" / "example.toml"

MINIMAL_TOML = """
schema_version = 1

[output]
directory = "/data/experiment"
"""


@pytest.fixture
def experiment_log(caplog: pytest.LogCaptureFixture):
    """Capture records from the experiment_config logger, which does not propagate to root."""
    logger = logging.getLogger("evomachine.experiment_config")
    logger.addHandler(caplog.handler)
    yield caplog
    logger.removeHandler(caplog.handler)


def _full_config() -> ExperimentConfig:
    return ExperimentConfig(
        name="mCherry timelapse",
        description="Two channels, imaged every 3 minutes.",
        acquisition=FrameAcquisitionSettings(
            save=True, clear_dmd_after=True, disable_leds_after=True,
        ),
        output=OutputConfig(
            directory="~/data/mcherry", filename_pattern="FOV{fov_id}_{channel}_{timestamp}",
        ),
    )


# Loading and defaults


def test_minimal_file_uses_defaults_from_the_runtime_classes() -> None:
    config = ExperimentConfig.from_toml(MINIMAL_TOML)

    assert config.schema_version == SCHEMA_VERSION
    assert config.name == ""
    assert config.acquisition == FrameAcquisitionSettings()
    assert config.output.directory == Path("/data/experiment")
    assert config.output.filename_pattern == FileNameConfig.model_fields["filename_pattern"].default


def test_example_file_in_repository_loads() -> None:
    config = ExperimentConfig.load(EXAMPLE_FILE)

    assert config.name == "Example experiment"
    assert config.output.directory == Path("~/evomachine_data/example_experiment")


# Saving and round trips


def test_toml_text_round_trip_preserves_every_value() -> None:
    config = _full_config()

    assert ExperimentConfig.from_toml(config.to_toml()) == config


def test_file_round_trip_preserves_every_value(tmp_path: Path) -> None:
    config = _full_config()

    written = config.save(tmp_path / "experiment.toml")

    assert written == tmp_path / "experiment.toml"
    assert ExperimentConfig.load(written) == config
    assert [path.name for path in tmp_path.iterdir()] == ["experiment.toml"]


def test_saved_file_is_readable_toml_with_names_and_sections() -> None:
    text = _full_config().to_toml()

    assert text.startswith(f"schema_version = {SCHEMA_VERSION}\n")
    assert "[acquisition]" in text
    assert "[output]" in text
    assert 'directory = "~/data/mcherry"' in text


def test_save_reports_unwritable_destination(tmp_path: Path) -> None:
    with pytest.raises(ExperimentConfigError, match="Cannot write"):
        _full_config().save(tmp_path / "missing_folder" / "experiment.toml")


# Schema version


def test_missing_schema_version_explains_how_to_fix_it() -> None:
    with pytest.raises(ExperimentConfigError, match=f"Add 'schema_version = {SCHEMA_VERSION}'"):
        ExperimentConfig.from_toml('[output]\ndirectory = "/data"\n')


def test_newer_schema_version_is_rejected_with_update_hint() -> None:
    text = MINIMAL_TOML.replace("schema_version = 1", f"schema_version = {SCHEMA_VERSION + 1}")

    with pytest.raises(ExperimentConfigError, match="Update EvoMachine"):
        ExperimentConfig.from_toml(text)


@pytest.mark.parametrize("value", ['"1"', "true", "1.0", "0"])
def test_invalid_schema_version_is_rejected(value: str) -> None:
    text = MINIMAL_TOML.replace("schema_version = 1", f"schema_version = {value}")

    with pytest.raises(ExperimentConfigError, match="schema"):
        ExperimentConfig.from_toml(text)


# Invalid files


def test_unknown_key_is_rejected_with_its_location() -> None:
    text = MINIMAL_TOML + 'filename_patern = "FOV{fov_id}"\n'

    with pytest.raises(ExperimentConfigError, match=r"output\.filename_patern"):
        ExperimentConfig.from_toml(text)


def test_unknown_section_is_rejected() -> None:
    with pytest.raises(ExperimentConfigError, match="imagin"):
        ExperimentConfig.from_toml(MINIMAL_TOML + "[imagin]\ninterval_s = 1\n")


def test_wrong_value_type_is_rejected() -> None:
    with pytest.raises(ExperimentConfigError, match="save must be bool"):
        ExperimentConfig.from_toml(MINIMAL_TOML + '[acquisition]\nsave = "yes"\n')


def test_missing_output_directory_is_rejected() -> None:
    with pytest.raises(ExperimentConfigError, match=r"output\.directory: Field required"):
        ExperimentConfig.from_toml("schema_version = 1\n[output]\n")


def test_invalid_filename_pattern_uses_file_name_config_validation() -> None:
    text = MINIMAL_TOML + 'filename_pattern = "{bogus}"\n'

    with pytest.raises(ExperimentConfigError, match="unsupported fields: 'bogus'") as raised:
        ExperimentConfig.from_toml(text)
    assert "Value error" not in str(raised.value)


def test_invalid_toml_syntax_names_the_source() -> None:
    with pytest.raises(ExperimentConfigError, match="broken.toml: not a valid TOML file"):
        ExperimentConfig.from_toml("schema_version = \n", source="broken.toml")


def test_missing_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(ExperimentConfigError, match="Cannot read"):
        ExperimentConfig.load(tmp_path / "does_not_exist.toml")


# Output directory


def test_relative_output_directory_is_accepted_with_a_warning(experiment_log) -> None:
    config = ExperimentConfig.from_toml(MINIMAL_TOML.replace("/data/experiment", "data/experiment"))

    assert config.output.directory == Path("data/experiment")
    assert "is not a full path" in experiment_log.text


@pytest.mark.parametrize("directory", ["/data/experiment", "~/data/experiment"])
def test_full_or_home_relative_directory_does_not_warn(experiment_log, directory: str) -> None:
    OutputConfig(directory=directory)

    assert "is not a full path" not in experiment_log.text


def test_empty_output_directory_is_rejected() -> None:
    with pytest.raises(ValidationError, match="must not be empty"):
        OutputConfig(directory="  ")


def test_output_builds_independent_file_name_configs_with_expanded_home() -> None:
    output = OutputConfig(directory="~/data", filename_pattern="FOV{fov_id}")
    time_config = TimeConfig(timezone_name="UTC")

    first = output.to_file_name_config(time_config=time_config)
    second = output.to_file_name_config()

    assert first.directory == Path("~/data").expanduser()
    assert first.filename_pattern == "FOV{fov_id}"
    assert first.time_config == time_config
    assert first is not second


# Immutability


def test_config_and_nested_sections_cannot_be_modified() -> None:
    config = _full_config()

    with pytest.raises(ValidationError):
        config.name = "changed"
    with pytest.raises(ValidationError):
        config.acquisition.save = False
    with pytest.raises(ValidationError):
        config.output.filename_pattern = "changed"


def test_updated_returns_validated_copy_and_leaves_original_unchanged() -> None:
    config = _full_config()

    changed = config.updated(name="renamed")

    assert changed.name == "renamed"
    assert config.name == "mCherry timelapse"
    with pytest.raises(ValueError, match="unknown fields"):
        config.updated(nmae="typo")


def test_frame_acquisition_settings_are_immutable() -> None:
    settings = FrameAcquisitionSettings()

    with pytest.raises(ValidationError):
        settings.save = True
    assert settings.updated(save=True).save is True
