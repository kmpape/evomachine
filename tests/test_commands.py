import pytest

from evomachine.commands import CommandFactory
from evomachine.frame import FrameMetaData
from evomachine.image_processing_config import ImageProcessorConfigFactory
from evomachine.types import LEDType


def make_factory() -> CommandFactory:
    """
    Return a command factory with a small default image-processing config.

    Parameters
    ----------
    None

    Returns
    -------
    CommandFactory
        Command factory configured for tests.
    """
    cfg = ImageProcessorConfigFactory.default_config(
        channels=[LEDType.LED_450_NM, LEDType.LED_565_NM],
        channels_seg=[LEDType.LED_450_NM],
    )
    cfg.preproc_enabled = True
    return CommandFactory(cfg=cfg)


def make_metadata(frame_id: int = 0, led_type: LEDType = LEDType.LED_450_NM) -> FrameMetaData:
    """
    Return deterministic frame metadata for command tests.

    Parameters
    ----------
    frame_id
        Frame ID to assign.
    led_type
        LEDType used in metadata.

    Returns
    -------
    FrameMetaData
        Valid frame metadata.
    """
    return FrameMetaData(frame_id=frame_id, leds={led_type: 10}, filter_wheel=None, exposure=50)


def test_command_image_accepts_frame_metadata() -> None:
    """
    Check IMAGE commands carry FrameMetaData directly.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    factory = make_factory()
    metadata = make_metadata()

    command = factory.command_image(frame_metadata=metadata, segment=False, save=True)

    assert command.command_args["frame_metadata"] is metadata
    assert command.command_args["segment"] is False
    assert command.command_args["save"] is True


def test_command_image_accepts_metadata_list_for_segmentation() -> None:
    """
    Check segmentation validation uses FrameMetaData LED channels.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    factory = make_factory()
    metadata = [make_metadata(0, LEDType.LED_450_NM), make_metadata(1, LEDType.LED_565_NM)]

    command = factory.command_image(frame_metadata=metadata, segment=True)

    assert command.command_args["frame_metadata"] == metadata


def test_command_image_rejects_old_dict_style_arguments() -> None:
    """
    Check old image command fields are no longer accepted.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    factory = make_factory()

    with pytest.raises(TypeError):
        factory.command_image(
            channels=[LEDType.LED_450_NM],
            exposure_time=50,
            segment=False,
        )


def test_command_image_rejects_missing_segmentation_channel() -> None:
    """
    Check segmentation requires configured segmentation channels in metadata.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    factory = make_factory()

    with pytest.raises(TypeError, match="channels_seg"):
        factory.command_image(frame_metadata=make_metadata(0, LEDType.LED_565_NM), segment=True)


def test_command_project_roi_rejects_old_pos_id_argument() -> None:
    """
    Check PROJECT_ROI commands require fov_id instead of pos_id.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    factory = make_factory()
    factory.update_region_of_interests({2: [0]})

    with pytest.raises(TypeError):
        factory.command_project_roi(
            channel=LEDType.LED_450_NM,
            pos_id=2,
            roi_ids=[0],
            duration=1.0,
        )
