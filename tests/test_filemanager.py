from datetime import datetime
import json

import numpy as np
import pytest
import skimage.io
import tifffile

from evomachine.config_types import ConfigFrame, ConfigFrameFactory, FileNameConfig
from evomachine.coordinates import Coordinate
from evomachine.filemanager import FileManager
from evomachine.stage import Stage
from evomachine.types import FilterWheelType, LEDType, UNKNOWN_POSITION_ID


def _frame_config() -> ConfigFrame:
    """
    Return a deterministic ConfigFrame for file manager tests.

    Parameters
    ----------
    None

    Returns
    -------
    ConfigFrame
        Frame configuration with metadata-friendly values.
    """
    return ConfigFrame(
        leds={LEDType.LED_450_NM: 25},
        filter_wheel=FilterWheelType.FILTER_465nm,
        exposure=100,
        position_id=3,
        coordinate=Coordinate(10, 20, 30),
        creation_time=datetime(2026, 1, 2, 3, 4, 5, 6000),
    )


def test_filename_config_validation_and_updates(tmp_path) -> None:
    """
    Check FileNameConfig validation, normalization, and update helpers.

    Parameters
    ----------
    tmp_path
        Pytest temporary directory fixture.

    Returns
    -------
    None
    """
    config = FileNameConfig(directory=tmp_path, extension=".TIF")

    assert config.extension == "tif"
    assert config.updated(extension="png").extension == "png"
    assert config.update_from_mapping({"extension": "jpg"}).extension == "jpg"
    with pytest.raises(ValueError):
        FileNameConfig(directory=tmp_path, extension="bmp")
    with pytest.raises(ValueError):
        FileNameConfig(directory=tmp_path, filename_pattern="")
    with pytest.raises(TypeError):
        config.update_from_mapping([("extension", "png")])
    with pytest.raises(ValueError):
        config.updated(missing=True)


def test_file_manager_creates_directory_and_updates_config(tmp_path) -> None:
    """
    Check FileManager creates configured directories and accepts config updates.

    Parameters
    ----------
    tmp_path
        Pytest temporary directory fixture.

    Returns
    -------
    None
    """
    directory = tmp_path / "frames"
    manager = FileManager(FileNameConfig(directory=directory))

    assert directory.is_dir()
    manager.update_config(extension="png")
    assert manager.config.extension == "png"
    with pytest.raises(FileNotFoundError):
        FileManager(FileNameConfig(directory=tmp_path / "missing", create_directory=False))


def test_config_frame_new_fields_and_factory() -> None:
    """
    Check ConfigFrame metadata fields, defaults, and factory behavior.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    config_frame = ConfigFrameFactory.default(
        leds={LEDType.LED_450_NM: 50},
        filter_wheel=FilterWheelType.NO_FILTER,
        exposure=100,
    )

    assert config_frame.position_id == UNKNOWN_POSITION_ID
    assert Stage.UNKNOWN_POSITION_ID == UNKNOWN_POSITION_ID
    assert config_frame.coordinate is None
    assert isinstance(config_frame.creation_time, datetime)
    with pytest.raises(TypeError):
        ConfigFrame(leds=None, filter_wheel=None, exposure=None, position_id=True)
    with pytest.raises(TypeError):
        ConfigFrame(leds=None, filter_wheel=None, exposure=None, coordinate=(1, 2, 3))
    with pytest.raises(TypeError):
        ConfigFrame(leds=None, filter_wheel=None, exposure=None, creation_time="now")
    with pytest.raises(TypeError):
        ConfigFrame(leds=None, filter_wheel=None, exposure=None, execution_time="now")


def test_default_and_override_filename_generation(tmp_path) -> None:
    """
    Check default and custom filename pattern rendering.

    Parameters
    ----------
    tmp_path
        Pytest temporary directory fixture.

    Returns
    -------
    None
    """
    manager = FileManager(FileNameConfig(directory=tmp_path))
    config_frame = _frame_config()

    filename = manager.get_filename(config_frame=config_frame, suffix="_raw")
    assert filename.parent == tmp_path
    assert filename.name.startswith("LED450NM_P3_X10_Y20_Z30_F1_")
    assert filename.name.endswith("_raw.tiff")

    custom = manager.get_filename(
        config_frame=config_frame,
        filename_pattern="{experiment}_{channel}.{extension}",
        experiment="demo",
    )
    assert custom == tmp_path / "demo_LED450NM.tiff"


def test_missing_filename_values_use_stable_placeholders(tmp_path) -> None:
    """
    Check missing ConfigFrame values render stable filename placeholders.

    Parameters
    ----------
    tmp_path
        Pytest temporary directory fixture.

    Returns
    -------
    None
    """
    manager = FileManager(FileNameConfig(directory=tmp_path))
    config_frame = ConfigFrame(leds=None, filter_wheel=None, exposure=None)

    filename = manager.get_filename(config_frame=config_frame)

    assert filename.name.startswith("NO_LED_P-1_X_Y_Zauto_FNone_")


def test_save_tiff_frame_writes_config_frame_metadata(tmp_path) -> None:
    """
    Check TIFF saves include JSON ConfigFrame metadata.

    Parameters
    ----------
    tmp_path
        Pytest temporary directory fixture.

    Returns
    -------
    None
    """
    manager = FileManager(FileNameConfig(directory=tmp_path))
    config_frame = _frame_config()
    frame = np.arange(16, dtype=np.uint16).reshape(4, 4)

    filename = manager.save_frame(frame=frame, config_frame=config_frame, suffix="_meta")

    assert filename.exists()
    with tifffile.TiffFile(filename) as tiff:
        saved = tiff.asarray()
        description = tiff.pages[0].tags["ImageDescription"].value
    metadata = json.loads(description)

    assert np.array_equal(saved, frame)
    assert metadata["ConfigFrame"]["leds"]["LED_450_NM"]["brightness"] == 25.0
    assert metadata["ConfigFrame"]["filter_wheel"] == {"name": "FILTER_465nm", "value": 1}
    assert metadata["ConfigFrame"]["position_id"] == 3
    assert metadata["ConfigFrame"]["coordinate"] == {"X": 10, "Y": 20, "Z": 30}
    assert config_frame.execution_time is not None


def test_save_png_frame_without_metadata_assertions(tmp_path) -> None:
    """
    Check non-TIFF image saves work without metadata expectations.

    Parameters
    ----------
    tmp_path
        Pytest temporary directory fixture.

    Returns
    -------
    None
    """
    manager = FileManager(FileNameConfig(directory=tmp_path, extension="png"))
    frame = np.arange(16, dtype=np.uint8).reshape(4, 4)

    filename = manager.save_frame(frame=frame, config_frame=_frame_config())

    assert filename.suffix == ".png"
    assert filename.exists()
    assert skimage.io.imread(filename).shape == frame.shape
