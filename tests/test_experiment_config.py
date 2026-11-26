from __future__ import annotations

import logging
import tomllib
from pathlib import Path

import pytest
from delta.utils import CroppingBox
from pydantic import ValidationError

from evomachine.acquisition import FrameAcquisitionSettings
from evomachine.config import TimeConfig
from evomachine.experiment_config import (
    SCHEMA_VERSION,
    CropBoxConfig,
    ExperimentConfig,
    ExperimentConfigError,
    ExperimentFocusConfig,
    ExperimentOutputConfig,
    ExperimentSoftwareFocusConfig,
    FocusFrameConfig,
    all_options,
    all_options_toml,
)
from evomachine.filemanager import FileNameConfig
from evomachine.softwarefocus import (
    BandpassFFTScoringConfig,
    MaximumFocusSelectionConfig,
    ReferenceFocusSelectionConfig,
    SoftwareFocusConfig,
    SteelScoringConfig,
)
from evomachine.types import FilterWheelType, FocusScoringType, FocusSelectionType, LEDType

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1] / "experiments"
EXAMPLE_FILE = EXPERIMENTS_DIR / "example.toml"
ALL_OPTIONS_FILE = EXPERIMENTS_DIR / "all_options.toml"

FRAME_TOML = """
[[focus.software.frames]]
leds = { LED_450_NM = 29 }
exposure_ms = 200
filter = "NO_FILTER"
"""

MINIMAL_TOML = """
schema_version = 1

[output]
directory = "/data/experiment"
""" + FRAME_TOML


@pytest.fixture
def experiment_log(caplog: pytest.LogCaptureFixture):
    """Capture records from the experiment_config logger, which does not propagate to root."""
    logger = logging.getLogger("evomachine.experiment_config")
    logger.addHandler(caplog.handler)
    yield caplog
    logger.removeHandler(caplog.handler)


def _frame(**overrides) -> FocusFrameConfig:
    fields = dict(leds={"LED_565_NM": 10, "LED_515_NM": 5}, exposure_ms=150, filter="FILTER_592nm")
    return FocusFrameConfig(**{**fields, **overrides})


def _full_config() -> ExperimentConfig:
    """Return a configuration where every section differs from its defaults."""
    return ExperimentConfig(
        name="mCherry timelapse",
        description="Two channels, imaged every 3 minutes.",
        acquisition=FrameAcquisitionSettings(save=True),
        output=ExperimentOutputConfig(
            directory="~/data/mcherry", filename_pattern="FOV{fov_id}_{channel}_{timestamp}",
        ),
        focus=ExperimentFocusConfig(
            software=ExperimentSoftwareFocusConfig(
                frames=(_frame(), _frame(leds={"LED_OVERHEAD": 50}, filter="NO_FILTER")),
                acquisition=FrameAcquisitionSettings(save=True),
                scoring=(
                    FocusScoringType.BANDPASS_FFT,
                    BandpassFFTScoringConfig(cell_radius=12.0, saturation=None),
                ),
                matching=(
                    FocusSelectionType.REFERENCE_PROFILE,
                    ReferenceFocusSelectionConfig(smoothing=None, allow_peak_reset=True),
                ),
                crops=(CropBoxConfig(xtl=200, ytl=300, xbr=3000, ybr=2900),),
            )
        ),
    )


def _with_software_focus(extra: str) -> str:
    return MINIMAL_TOML + extra


# Loading and defaults


def test_minimal_file_uses_the_agreed_defaults() -> None:
    config = ExperimentConfig.from_toml(MINIMAL_TOML)

    assert config.schema_version == SCHEMA_VERSION
    assert config.name == ""
    assert config.acquisition == FrameAcquisitionSettings.experiment_defaults()
    assert config.output.directory == Path("/data/experiment")
    assert config.output.filename_pattern == FileNameConfig.model_fields["filename_pattern"].default
    software = config.focus.software
    assert software.acquisition == FrameAcquisitionSettings.experiment_defaults()
    assert software.scoring == (FocusScoringType.STEEL, SteelScoringConfig())
    assert software.matching == (
        FocusSelectionType.REFERENCE_PROFILE,
        ReferenceFocusSelectionConfig(),
    )
    assert software.crops == ()


def test_experiment_acquisition_defaults_switch_everything_off_after_each_image() -> None:
    defaults = FrameAcquisitionSettings.experiment_defaults()

    assert (defaults.clear_dmd_after, defaults.restore_leds_after, defaults.disable_leds_after) == (
        True,
        False,
        True,
    )
    assert FrameAcquisitionSettings().restore_leds_after is True


def test_example_file_in_repository_loads() -> None:
    config = ExperimentConfig.load(EXAMPLE_FILE)

    assert config.name == "Example experiment"
    assert config.output.directory == Path("~/evomachine_data/example_experiment")
    assert config.focus.software.frames[0].filter == FilterWheelType.BANDPASS_405_445_514_561_640nm


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


def test_saved_file_writes_options_by_name() -> None:
    text = _full_config().to_toml()

    assert text.startswith(f"schema_version = {SCHEMA_VERSION}\n")
    assert 'directory = "~/data/mcherry"' in text
    assert 'filter = "FILTER_592nm"' in text
    assert "LED_565_NM = 10.0" in text
    assert 'type = "BANDPASS_FFT"' in text
    assert 'smoothing = "auto"' in text
    assert "saturation" not in text


def test_save_reports_unwritable_destination(tmp_path: Path) -> None:
    with pytest.raises(ExperimentConfigError, match="Cannot write"):
        _full_config().save(tmp_path / "missing_folder" / "experiment.toml")


# Schema version


def test_missing_schema_version_explains_how_to_fix_it() -> None:
    text = MINIMAL_TOML.replace("schema_version = 1", "")

    with pytest.raises(ExperimentConfigError, match=f"Add 'schema_version = {SCHEMA_VERSION}'"):
        ExperimentConfig.from_toml(text)


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
    text = MINIMAL_TOML.replace("[output]", '[output]\nfilename_patern = "FOV{fov_id}"')

    with pytest.raises(ExperimentConfigError, match=r"output\.filename_patern"):
        ExperimentConfig.from_toml(text)


def test_unknown_section_is_rejected() -> None:
    with pytest.raises(ExperimentConfigError, match="imagin"):
        ExperimentConfig.from_toml(MINIMAL_TOML + "[imagin]\ninterval_s = 1\n")


def test_wrong_value_type_is_rejected() -> None:
    with pytest.raises(ExperimentConfigError, match="save must be bool"):
        ExperimentConfig.from_toml(MINIMAL_TOML + '[acquisition]\nsave = "yes"\n')


def test_missing_output_directory_is_rejected() -> None:
    text = MINIMAL_TOML.replace('directory = "/data/experiment"', "")

    with pytest.raises(ExperimentConfigError, match=r"output\.directory: Field required"):
        ExperimentConfig.from_toml(text)


def test_invalid_filename_pattern_uses_file_name_config_validation() -> None:
    text = MINIMAL_TOML.replace("[output]", '[output]\nfilename_pattern = "{bogus}"')

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
    ExperimentOutputConfig(directory=directory)

    assert "is not a full path" not in experiment_log.text


def test_empty_output_directory_is_rejected() -> None:
    with pytest.raises(ValidationError, match="must not be empty"):
        ExperimentOutputConfig(directory="  ")


def test_output_builds_independent_file_name_configs_with_expanded_home() -> None:
    output = ExperimentOutputConfig(directory="~/data", filename_pattern="FOV{fov_id}")
    time_config = TimeConfig(timezone_name="UTC")

    first = output.to_file_name_config(time_config=time_config)
    second = output.to_file_name_config()

    assert first.directory == Path("~/data").expanduser()
    assert first.filename_pattern == "FOV{fov_id}"
    assert first.time_config == time_config
    assert first is not second


# Software focus: required imaging settings


def test_missing_focus_section_is_rejected() -> None:
    text = MINIMAL_TOML.replace(FRAME_TOML, "")

    with pytest.raises(ExperimentConfigError, match="focus: Field required"):
        ExperimentConfig.from_toml(text)


def test_software_focus_without_frames_is_rejected() -> None:
    text = MINIMAL_TOML.replace(FRAME_TOML, "[focus.software]\ncrops = []\n")

    with pytest.raises(ExperimentConfigError, match=r"focus\.software\.frames: Field required"):
        ExperimentConfig.from_toml(text)


def test_empty_frame_list_is_rejected() -> None:
    with pytest.raises(ValidationError, match="at least one focus frame"):
        ExperimentSoftwareFocusConfig(frames=())


@pytest.mark.parametrize("missing", ["leds", "exposure_ms", "filter"])
def test_every_focus_frame_field_is_required(missing: str) -> None:
    lines = [line for line in FRAME_TOML.splitlines() if not line.startswith(missing)]
    text = MINIMAL_TOML.replace(FRAME_TOML, "\n".join(lines) + "\n")

    with pytest.raises(ExperimentConfigError, match=rf"frames\.0\.{missing}: Field required"):
        ExperimentConfig.from_toml(text)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("LED_450_NM", "LED_451_NM", "unknown LED 'LED_451_NM'. Allowed: LED_385_NM"),
        ("LED_450_NM", "NO_LED", "unknown LED 'NO_LED'"),
        ('"NO_FILTER"', '"UNKNOWN"', "unknown filter 'UNKNOWN'"),
        ("leds = { LED_450_NM = 29 }", "leds = {}", "at least one LED"),
        ("= 29", "= 150", r"brightness must be in \[0, 100\]"),
        ("exposure_ms = 200", "exposure_ms = 5000", r"exposure must be in \[1, 1000\]"),
    ],
)
def test_invalid_focus_frame_values_are_rejected(old: str, new: str, message: str) -> None:
    text = MINIMAL_TOML.replace(old, new)

    with pytest.raises(ExperimentConfigError, match=message):
        ExperimentConfig.from_toml(text)


def test_old_filter_alias_still_loads() -> None:
    config = ExperimentConfig.from_toml(MINIMAL_TOML.replace('"NO_FILTER"', '"FILTER"'))

    assert config.focus.software.frames[0].filter == FilterWheelType.BANDPASS_405_445_514_561_640nm


# Software focus: scoring and matching


def test_scoring_and_matching_parameters_are_read_for_the_chosen_type() -> None:
    config = ExperimentConfig.from_toml(_with_software_focus(
        '[focus.software.scoring]\ntype = "STEEL"\ncolshift = 0\n'
        '[focus.software.matching]\ntype = "MAXIMUM"\nrel_range = 30\nstep_size = 2\n'
    ))

    assert config.focus.software.scoring == (
        FocusScoringType.STEEL,
        SteelScoringConfig(rowshift=25, colshift=0),
    )
    assert config.focus.software.matching == (
        FocusSelectionType.MAXIMUM,
        MaximumFocusSelectionConfig(rel_range=30, step_size=2),
    )


def test_optional_parameter_without_default_can_be_set() -> None:
    config = ExperimentConfig.from_toml(_with_software_focus(
        '[focus.software.scoring]\ntype = "SQUARED_GRAD_AVG"\nthreshold = 0.5\n'
    ))

    assert config.focus.software.scoring[1].threshold == 0.5


def test_auto_smoothing_and_offset_lists_are_converted() -> None:
    config = ExperimentConfig.from_toml(_with_software_focus(
        '[focus.software.matching]\ntype = "REFERENCE_PROFILE"\nsmoothing = "auto"\n'
        "sample_offsets = [-10.0, -5.0, 0.0, 5.0, 10.0]\n"
    ))

    parameters = config.focus.software.matching[1]
    assert parameters.smoothing is None
    assert parameters.sample_offsets == (-10.0, -5.0, 0.0, 5.0, 10.0)


@pytest.mark.parametrize(
    ("table", "message"),
    [
        ('[focus.software.scoring]\nrowshift = 3\n', "scoring is missing 'type'"),
        ('[focus.software.scoring]\ntype = "SHARPEST"\n', "unknown scoring type 'SHARPEST'"),
        (
            '[focus.software.scoring]\ntype = "STEEL"\ncolshfit = 3\n',
            "unknown STEEL parameter.*colshfit.*Allowed: rowshift, colshift, normalise",
        ),
        (
            '[focus.software.scoring]\ntype = "LAPLACIAN_VAR"\nthreshold = 1\n',
            "unknown LAPLACIAN_VAR parameter.*Allowed: none",
        ),
        (
            '[focus.software.matching]\ntype = "MAXIMUM"\nrel_range = 5\nstep_size = 10\n',
            r"invalid MAXIMUM parameters: step_size must lie in \(0, rel_range\]",
        ),
        (
            '[focus.software.matching]\ntype = "REFERENCE_PROFILE"\nrel_range = 50\n',
            "unknown REFERENCE_PROFILE parameter.*rel_range",
        ),
    ],
)
def test_invalid_scoring_or_matching_tables_are_rejected(table: str, message: str) -> None:
    with pytest.raises(ExperimentConfigError, match=message):
        ExperimentConfig.from_toml(_with_software_focus(table))


def test_crops_are_read_and_validated_by_software_focus_config() -> None:
    config = ExperimentConfig.from_toml(_with_software_focus(
        "[focus.software]\ncrops = [{ xtl = 200, ytl = 300, xbr = 3000, ybr = 2900 }]\n"
    ))
    assert config.focus.software.crops == (CropBoxConfig(xtl=200, ytl=300, xbr=3000, ybr=2900),)

    with pytest.raises(ExperimentConfigError, match="positive area"):
        ExperimentConfig.from_toml(_with_software_focus(
            "[focus.software]\ncrops = [{ xtl = 10, ytl = 0, xbr = 5, ybr = 100 }]\n"
        ))


def test_software_focus_config_is_built_from_new_objects_every_time() -> None:
    software = _full_config().focus.software

    first = software.to_software_focus_config()
    second = software.to_software_focus_config()

    assert isinstance(first, SoftwareFocusConfig)
    assert first.focus_frames[0].leds == {LEDType.LED_565_NM: 10, LEDType.LED_515_NM: 5}
    assert first.focus_frames[0].exposure == 150
    assert first.focus_frames[0].filter_wheel == FilterWheelType.FILTER_592nm
    assert first.scoring == software.scoring
    assert first.selection == software.matching
    assert first.cropping_boxes == (CroppingBox(xtl=200, ytl=300, xbr=3000, ybr=2900),)
    assert first.focus_frames[0] is not second.focus_frames[0]
    assert first.cropping_boxes[0] is not second.cropping_boxes[0]


# Reference list of options


def test_all_options_file_is_up_to_date_with_the_code() -> None:
    assert tomllib.loads(ALL_OPTIONS_FILE.read_text(encoding="utf-8")) == all_options(), (
        "experiments/all_options.toml is out of date; regenerate it with the command in its header."
    )
    assert ALL_OPTIONS_FILE.read_text(encoding="utf-8") == all_options_toml()


def test_all_options_lists_every_choice_and_no_placeholders() -> None:
    options = all_options()

    assert options["leds"] == [led.name for led in LEDType if led != LEDType.NO_LED]
    assert "UNKNOWN" not in options["filters"]
    assert options["scoring_types"] == [member.name for member in FocusScoringType]
    assert options["matching_types"] == ["MAXIMUM", "REFERENCE_PROFILE"]
    assert options["scoring_parameters"]["STEEL"] == {
        "rowshift": 25,
        "colshift": 50,
        "normalise": False,
    }
    assert options["optional_parameters"] == {
        "SQUARED_GRAD_AVG": ["threshold"],
        "BANDPASS_FFT": ["saturation"],
    }


# Immutability


def test_config_and_nested_sections_cannot_be_modified() -> None:
    config = _full_config()

    with pytest.raises(ValidationError):
        config.name = "changed"
    with pytest.raises(ValidationError):
        config.acquisition.save = False
    with pytest.raises(ValidationError):
        config.output.filename_pattern = "changed"
    with pytest.raises(ValidationError):
        config.focus.software.frames[0].exposure_ms = 1
    assert isinstance(config.focus.software.frames, tuple)
    assert isinstance(config.focus.software.frames[0].leds, tuple)


def test_updated_returns_validated_copy_and_leaves_original_unchanged() -> None:
    config = _full_config()

    changed = config.updated(name="renamed")

    assert changed.name == "renamed"
    assert config.name == "mCherry timelapse"
    assert changed.focus == config.focus
    with pytest.raises(ValueError, match="unknown fields"):
        config.updated(nmae="typo")


def test_frame_acquisition_settings_are_immutable() -> None:
    settings = FrameAcquisitionSettings()

    with pytest.raises(ValidationError):
        settings.save = True
    assert settings.updated(save=True).save is True
