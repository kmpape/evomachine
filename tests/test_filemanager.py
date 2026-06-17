from datetime import datetime
import importlib
import json
import logging
import logging.handlers
from zoneinfo import ZoneInfo

import numpy as np
import pytest
import skimage.io
import tifffile

from evomachine import config as config_module
from evomachine.config import (
    EvoLoggingConfig,
    EvoFormatter,
    TimeConfig,
    configure_binding_loggers,
    ensure_timezone_aware,
    format_timestamp_for_display,
    format_timestamp_for_filename,
    format_timezone_id,
    get_logger,
    now,
)
from evomachine.filemanager import FileNameConfig
from evomachine.frame import FrameMetaData, FrameMetaDataFactory
from evomachine.coordinates import Coordinate
from evomachine.filemanager import FileManager
from evomachine.peripherals.stage import Stage
from evomachine.types import FilterWheelType, LEDType, UNKNOWN_FOV_ID


def _frame_metadata(**updates) -> FrameMetaData:
    """
    Return deterministic FrameMetaData for file manager tests.

    Parameters
    ----------
    **updates
        FrameMetaData field values to override.

    Returns
    -------
    FrameMetaData
        Frame metadata with stable test values.
    """
    values = {
        "frame_id": 7,
        "leds": {LEDType.LED_450_NM: 25},
        "filter_wheel": FilterWheelType.FILTER_465nm,
        "exposure": 100,
        "fov_id": 3,
        "coordinate": Coordinate(10, 20, 30),
        "creation_time": datetime(2026, 1, 2, 3, 4, 5, 6000),
    }
    values.update(updates)
    return FrameMetaData(**values)


def test_time_config_helpers_use_default_timezone() -> None:
    """
    Check timezone helpers normalise and format timestamps consistently.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    time_config = TimeConfig()
    naive = datetime(2026, 1, 2, 3, 4, 5, 6000)

    aware = ensure_timezone_aware(value=naive, config=time_config)
    current = now(config=time_config)

    assert aware.tzinfo == ZoneInfo("Europe/London")
    assert current.tzinfo == ZoneInfo("Europe/London")
    assert format_timestamp_for_filename(value=aware, config=time_config) == "2026-01-02_03-04-05-006000+0000"
    assert format_timestamp_for_display(value=aware, config=time_config) == "2026-01-02 03:04:05"
    assert format_timezone_id(value=aware, config=time_config) == "Europe-London"
    with pytest.raises(ValueError):
        TimeConfig(timezone_name="Not/AZone")


def test_evo_formatter_uses_timezone_identifier() -> None:
    """
    Check log formatter timestamps use configured local display time.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    formatter = EvoFormatter("%(asctime)s - %(message)s")
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello",
        args=(),
        exc_info=None,
    )
    record.created = datetime(2026, 1, 2, 3, 4, 5, tzinfo=ZoneInfo("Europe/London")).timestamp()

    rendered = formatter.format(record)

    assert "2026-01-02 03:04:05" in rendered
    assert "Europe-London" not in rendered
    assert "+0000" not in rendered


def test_default_log_filename_includes_timezone_identifier() -> None:
    """
    Check generated log filenames include a timezone identifier.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    assert config_module.filename.startswith("evom_")
    assert "Europe-London" in config_module.filename
    assert config_module.filename.endswith(".log")


def test_logging_config_accepts_binding_level() -> None:
    """
    Check EvoLoggingConfig validates binding logging options.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    logging_config = EvoLoggingConfig(
        binding_level=logging.WARNING,
        binding_logger_names=("syncboard", "asitiger"),
    )

    assert logging_config.binding_level == logging.WARNING
    assert logging_config.binding_logger_names == ("syncboard", "asitiger")
    with pytest.raises(TypeError):
        EvoLoggingConfig(binding_level=True)
    with pytest.raises(TypeError):
        EvoLoggingConfig(binding_logger_names=["syncboard"])
    with pytest.raises(ValueError):
        EvoLoggingConfig(binding_logger_names=("syncboard", ""))


def test_get_logger_uses_binding_level() -> None:
    """
    Check binding loggers use the configured binding level.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    logging_config = EvoLoggingConfig(
        level=logging.ERROR,
        gui_level=logging.ERROR,
        peripheral_level=logging.ERROR,
        binding_level=logging.DEBUG,
    )

    logger = get_logger(
        name="test.binding.logger",
        logging_config=logging_config,
        is_binding=True,
    )

    assert logger.level == logging.DEBUG
    assert config_module.file_handler.level == logging.DEBUG


def test_configure_binding_loggers_configures_external_packages() -> None:
    """
    Check external binding package loggers receive Evo logging handlers.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    logging_config = EvoLoggingConfig(
        level=logging.ERROR,
        gui_level=logging.ERROR,
        peripheral_level=logging.ERROR,
        binding_level=logging.DEBUG,
        binding_logger_names=("syncboard", "asitiger"),
    )

    loggers = configure_binding_loggers(logging_config=logging_config)

    assert tuple(logger.name for logger in loggers) == ("syncboard", "asitiger")
    assert all(logger.level == logging.DEBUG for logger in loggers)
    assert all(config_module.file_handler in logger.handlers for logger in loggers)
    assert all(logger.propagate is False for logger in loggers)


def test_binding_logger_output_uses_evo_display_timestamp() -> None:
    """
    Check binding log output uses Evo display timestamps without inline timezone text.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    logger = get_logger(name="test.binding.rendered", is_binding=True)
    record = logging.LogRecord(
        name="test.binding.rendered",
        level=logging.DEBUG,
        pathname=__file__,
        lineno=1,
        msg="binding message",
        args=(),
        exc_info=None,
    )
    record.created = datetime(2026, 1, 2, 3, 4, 5, tzinfo=ZoneInfo("Europe/London")).timestamp()
    formatter = next(handler.formatter for handler in logger.handlers if handler.formatter is not None)

    rendered = formatter.format(record)

    assert "2026-01-02 03:04:05 - DEBUG - test.binding.rendered - binding message" == rendered
    assert "Europe-London" not in rendered
    assert "+0000" not in rendered


def test_syncboard_import_does_not_install_own_handlers() -> None:
    """
    Check importing syncboard leaves handler configuration to applications.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    syncboard_logger = logging.getLogger("syncboard")
    syncboard_logger.handlers.clear()
    syncboard = importlib.import_module("syncboard")
    serialconnection = importlib.import_module("syncboard.serialconnection")
    syncboard = importlib.reload(syncboard)
    serialconnection = importlib.reload(serialconnection)

    assert all(isinstance(handler, logging.NullHandler) for handler in syncboard_logger.handlers)
    assert logging.getLogger("syncboard.serialconnection").handlers == []
    assert not hasattr(serialconnection, "LOG_DIR")
    assert not hasattr(serialconnection, "filename")


def test_binding_child_logs_reach_configured_evo_package_logger() -> None:
    """
    Check child binding loggers propagate records into configured Evo package loggers.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    logging_config = EvoLoggingConfig(
        binding_level=logging.DEBUG,
        binding_logger_names=("syncboard", "asitiger"),
    )
    configure_binding_loggers(logging_config=logging_config)
    syncboard_handler = logging.handlers.BufferingHandler(capacity=10)
    asitiger_handler = logging.handlers.BufferingHandler(capacity=10)
    syncboard_logger = logging.getLogger("syncboard")
    asitiger_logger = logging.getLogger("asitiger")
    syncboard_logger.addHandler(syncboard_handler)
    asitiger_logger.addHandler(asitiger_handler)

    logging.getLogger("syncboard.serialconnection").debug("syncboard binding debug")
    logging.getLogger("asitiger.tigercontroller").debug("asitiger binding debug")

    assert [record.getMessage() for record in syncboard_handler.buffer] == ["syncboard binding debug"]
    assert [record.getMessage() for record in asitiger_handler.buffer] == ["asitiger binding debug"]
    syncboard_logger.removeHandler(syncboard_handler)
    asitiger_logger.removeHandler(asitiger_handler)


def test_filename_config_validation_and_updates(tmp_path) -> None:
    """
    Check FileNameConfig validation and update helpers.

    Parameters
    ----------
    tmp_path
        Pytest temporary directory fixture.

    Returns
    -------
    None
    """
    config = FileNameConfig(directory=tmp_path)

    assert config.directory == tmp_path
    assert config.time_config.timezone_name == "Europe/London"
    assert config.updated(filename_pattern="{frame_id}").filename_pattern == "{frame_id}"
    assert config.update_from_mapping({"filename_pattern": "{channel}"}).filename_pattern == "{channel}"
    with pytest.raises(ValueError):
        FileNameConfig(directory=tmp_path, filename_pattern="")
    with pytest.raises(TypeError):
        config.update_from_mapping([("filename_pattern", "{frame_id}")])
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
    manager.update_config(filename_pattern="{frame_id}")
    assert manager.config.filename_pattern == "{frame_id}"
    with pytest.raises(FileNotFoundError):
        FileManager(FileNameConfig(directory=tmp_path / "missing", create_directory=False))


def test_frame_metadata_fields_and_factory_counter() -> None:
    """
    Check FrameMetaData fields, validation, and factory counter behavior.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    FrameMetaDataFactory.reset_counter()
    first = FrameMetaDataFactory.default(
        leds={LEDType.LED_450_NM: 50},
        filter_wheel=FilterWheelType.NO_FILTER,
        exposure=100,
        dmd_pattern=np.ones((2, 3), dtype=np.uint8),
    )
    explicit = FrameMetaDataFactory.default(leds=None, filter_wheel=None, exposure=None, frame_id=99)
    second = FrameMetaDataFactory.default(leds=None, filter_wheel=None, exposure=None)

    assert first.frame_id == 0
    assert explicit.frame_id == 99
    assert second.frame_id == 1
    assert first.fov_id == UNKNOWN_FOV_ID
    assert Stage.UNKNOWN_FOV_ID == UNKNOWN_FOV_ID
    assert first.coordinate is None
    assert first.to_metadata_dict()["dmd_pattern"] == {
        "present": True,
        "shape": [2, 3],
        "dtype": "uint8",
    }
    assert "array" not in str(first)
    assert "dmd_pattern" in str(first)
    assert isinstance(first.creation_time, datetime)
    assert first.creation_time.tzinfo == ZoneInfo("Europe/London")
    with pytest.raises(TypeError):
        FrameMetaDataFactory.reset_counter(start=True)
    with pytest.raises(TypeError):
        FrameMetaData(frame_id=True, leds=None, filter_wheel=None, exposure=None)
    with pytest.raises(TypeError):
        FrameMetaData(frame_id=0, leds=None, filter_wheel=None, exposure=None, fov_id=True)
    with pytest.raises(TypeError):
        FrameMetaData(frame_id=0, leds=None, filter_wheel=None, exposure=None, coordinate=(1, 2, 3))
    with pytest.raises(TypeError):
        FrameMetaData(frame_id=0, leds=None, filter_wheel=None, exposure=None, creation_time="now")
    with pytest.raises(TypeError):
        FrameMetaData(frame_id=0, leds=None, filter_wheel=None, exposure=None, execution_time="now")
    with pytest.raises(TypeError):
        FrameMetaData(frame_id=0, leds=None, filter_wheel=None, exposure=None, callback_id=True)
    with pytest.raises(TypeError):
        FrameMetaData(frame_id=0, leds=None, filter_wheel=None, exposure=None, additional_metadata={1: "bad"})
    with pytest.raises(TypeError):
        FrameMetaData(frame_id=0, leds=None, filter_wheel=None, exposure=None, dmd_pattern="bad")


def test_old_frame_symbols_are_removed() -> None:
    """
    Check the hard rename removed old frame metadata public symbols.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    with pytest.raises(TypeError):
        FrameMetaData(
            frame_id=0,
            leds=None,
            filter_wheel=None,
            exposure=None,
            position_id=0,
        )


def test_default_and_configured_filename_generation(tmp_path) -> None:
    """
    Check default and configured filename pattern rendering.

    Parameters
    ----------
    tmp_path
        Pytest temporary directory fixture.

    Returns
    -------
    None
    """
    manager = FileManager(FileNameConfig(directory=tmp_path))
    frame_metadata = _frame_metadata(additional_metadata={"experiment": "demo"})

    filename = manager.get_filename(frame_metadata=frame_metadata)
    assert filename.parent == tmp_path
    assert filename.name.startswith("LED450NM_FOV3_X10_Y20_Z30_F1_")
    assert filename.suffix == ".tiff"

    manager.update_config(
        filename_pattern="{experiment}_{channel}_{timezone}_F{frame_id}_C{callback_id}.png",
    )
    frame_metadata.callback_id = 12
    custom = manager.get_filename(frame_metadata=frame_metadata)
    assert custom == tmp_path / "demo_LED450NM_Europe-London_F7_C12.tiff"


def test_multiple_led_filename_uses_period_separator(tmp_path) -> None:
    """
    Check multiple LED channel labels use periods instead of plus signs.

    Parameters
    ----------
    tmp_path
        Pytest temporary directory fixture.

    Returns
    -------
    None
    """
    manager = FileManager(FileNameConfig(directory=tmp_path, filename_pattern="{channel}"))
    frame_metadata = _frame_metadata(leds={LEDType.LED_450_NM: 25, LEDType.LED_565_NM: 30})

    filename = manager.get_filename(frame_metadata=frame_metadata)

    assert filename.name == "LED450NM.LED565NM.tiff"
    assert "+" not in filename.name


def test_missing_filename_values_use_stable_placeholders(tmp_path) -> None:
    """
    Check missing FrameMetaData values render stable filename placeholders.

    Parameters
    ----------
    tmp_path
        Pytest temporary directory fixture.

    Returns
    -------
    None
    """
    manager = FileManager(FileNameConfig(directory=tmp_path))
    frame_metadata = FrameMetaData(frame_id=0, leds=None, filter_wheel=None, exposure=None)

    filename = manager.get_filename(frame_metadata=frame_metadata)

    assert filename.name.startswith("NO_LED_FOV-1_X_Y_Zauto_FNone_")


def test_save_tiff_frame_writes_frame_metadata(tmp_path) -> None:
    """
    Check TIFF saves include JSON FrameMetaData metadata.

    Parameters
    ----------
    tmp_path
        Pytest temporary directory fixture.

    Returns
    -------
    None
    """
    manager = FileManager(FileNameConfig(directory=tmp_path))
    frame_metadata = _frame_metadata(callback_id=11, additional_metadata={"experiment": "alpha"})
    frame = np.arange(16, dtype=np.uint16).reshape(4, 4)

    filename = manager.save_frame(frame=frame, frame_metadata=frame_metadata)

    assert filename.suffix == ".tiff"
    assert filename.exists()
    with tifffile.TiffFile(filename) as tiff:
        saved = tiff.asarray()
        description = tiff.pages[0].tags["ImageDescription"].value
    metadata = json.loads(description)

    assert np.array_equal(saved, frame)
    assert metadata["FrameMetaData"]["frame_id"] == 7
    assert metadata["FrameMetaData"]["callback_id"] == 11
    assert metadata["FrameMetaData"]["leds"]["LED_450_NM"]["brightness"] == 25.0
    assert metadata["FrameMetaData"]["filter_wheel"] == {"name": "FILTER_465nm", "value": 1}
    assert metadata["FrameMetaData"]["dmd_pattern"] is None
    assert metadata["FrameMetaData"]["fov_id"] == 3
    assert metadata["FrameMetaData"]["coordinate"] == {"X": 10, "Y": 20, "Z": 30}
    assert metadata["FrameMetaData"]["creation_time"].endswith("+00:00")
    assert metadata["FrameMetaData"]["creation_timezone"] == "Europe-London"
    assert metadata["FrameMetaData"]["execution_time"][-6] in "+-"
    assert metadata["FrameMetaData"]["execution_time"][-3] == ":"
    assert metadata["FrameMetaData"]["execution_timezone"] == "Europe-London"
    assert metadata["FrameMetaData"]["additional_metadata"] == {"experiment": "alpha"}
    assert "force" + "_settings" not in metadata["FrameMetaData"]
    assert frame_metadata.execution_time is not None


def test_non_json_serializable_metadata_raises_before_write(tmp_path) -> None:
    """
    Check non-JSON-serializable additional metadata raises before writing.

    Parameters
    ----------
    tmp_path
        Pytest temporary directory fixture.

    Returns
    -------
    None
    """
    manager = FileManager(FileNameConfig(directory=tmp_path))
    frame_metadata = _frame_metadata(additional_metadata={"bad": object()})
    frame = np.arange(16, dtype=np.uint16).reshape(4, 4)

    with pytest.raises(TypeError):
        manager.save_frame(frame=frame, frame_metadata=frame_metadata)

    assert not list(tmp_path.iterdir())


def test_load_helpers_read_tiff_image_and_metadata(tmp_path) -> None:
    """
    Check load helpers read saved TIFF image data and metadata.

    Parameters
    ----------
    tmp_path
        Pytest temporary directory fixture.

    Returns
    -------
    None
    """
    manager = FileManager(FileNameConfig(directory=tmp_path))
    frame = np.arange(16, dtype=np.uint16).reshape(4, 4)
    filename = manager.save_frame(frame=frame, frame_metadata=_frame_metadata())

    image = FileManager.load_image(filename)
    metadata = FileManager.load_tiff_metadata(filename)
    loaded_image, loaded_metadata = FileManager.load_frame(filename)

    assert np.array_equal(image, frame)
    assert np.array_equal(loaded_image, frame)
    assert metadata["FrameMetaData"]["frame_id"] == 7
    assert loaded_metadata == metadata


def test_load_frame_rejects_non_tiff_files(tmp_path) -> None:
    """
    Check load_frame is TIFF-only while load_image remains generic.

    Parameters
    ----------
    tmp_path
        Pytest temporary directory fixture.

    Returns
    -------
    None
    """
    path = tmp_path / "image.png"
    frame = np.arange(16, dtype=np.uint8).reshape(4, 4)
    skimage.io.imsave(path, frame, check_contrast=False)

    image = FileManager.load_image(path)

    with pytest.raises(ValueError):
        FileManager.load_frame(path)
    assert image.shape == frame.shape


def test_list_filenames_returns_sorted_matching_files_only(tmp_path) -> None:
    """
    Check filename listing returns sorted matching files and skips directories.

    Parameters
    ----------
    tmp_path
        Pytest temporary directory fixture.

    Returns
    -------
    None
    """
    first = tmp_path / "LED450NM_a_ref.tiff"
    second = tmp_path / "LED450NM_b_ref.tiff"
    other = tmp_path / "LED565NM_c_ref.tiff"
    match_directory = tmp_path / "LED450NM_dir_ref.tiff"
    second.write_text("second")
    first.write_text("first")
    other.write_text("other")
    match_directory.mkdir()

    filenames = FileManager.list_filenames(directory=tmp_path, filename_pattern="LED450NM_*_ref.tiff")

    assert filenames == [first, second]


def test_list_filenames_rejects_invalid_inputs(tmp_path) -> None:
    """
    Check filename listing validates directory and pattern inputs.

    Parameters
    ----------
    tmp_path
        Pytest temporary directory fixture.

    Returns
    -------
    None
    """
    file_path = tmp_path / "file.tiff"
    file_path.write_text("data")

    with pytest.raises(FileNotFoundError):
        FileManager.list_filenames(directory=tmp_path / "missing", filename_pattern="*.tiff")
    with pytest.raises(NotADirectoryError):
        FileManager.list_filenames(directory=file_path, filename_pattern="*.tiff")
    with pytest.raises(TypeError):
        FileManager.list_filenames(directory=tmp_path, filename_pattern=1)
    with pytest.raises(ValueError):
        FileManager.list_filenames(directory=tmp_path, filename_pattern="/absolute/*.tiff")
