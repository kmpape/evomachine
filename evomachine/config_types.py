"""Deprecated compatibility exports for the old central config module.

Active evomachine code should import config/data types from their domain
modules instead of this file.
"""

from __future__ import annotations

from evomachine.bindings.asitiger.autofocus import TigerAutofocusConfig as ConfigCRISP
from evomachine.bindings.asitiger.autofocus import TigerAutofocusConfigFactory as ConfigCRISPFactory
from evomachine.filemanager import FileNameConfig
from evomachine.frame import Frame, FrameMetaData, FrameMetaDataFactory
from evomachine.image_processing_config import ImageProcessorConfig, ImageProcessorConfigFactory
from evomachine.peripherals.camera import (
    CameraSystemConfig,
    CameraSystemConfigFactory,
    ImageConfigType,
    ImageConfigTypeFactory,
    ObjectiveConfigType,
    ObjectiveConfigTypeFactory,
)
from evomachine.peripherals.dmd import DmdCalibrationConfig, DmdCalibrationConfigFactory
from evomachine.softwarefocus import SoftwareFocusConfig, SoftwareFocusConfigFactory

ConfigImageProcessor = ImageProcessorConfig
ConfigImageProcessorFactory = ImageProcessorConfigFactory
ConfigCamera = CameraSystemConfig
ConfigCameraFactory = CameraSystemConfigFactory
DMDCalibConfigType = DmdCalibrationConfig
DMDCalibConfigTypeFactory = DmdCalibrationConfigFactory
ConfigFocus = SoftwareFocusConfig
ConfigFocusFactory = SoftwareFocusConfigFactory
SoftwareFocusConfigNew = SoftwareFocusConfig
SoftwareFocusConfigNewFactory = SoftwareFocusConfigFactory

__all__ = [
    "CameraSystemConfig",
    "CameraSystemConfigFactory",
    "ConfigCRISP",
    "ConfigCRISPFactory",
    "ConfigCamera",
    "ConfigCameraFactory",
    "ConfigFocus",
    "ConfigFocusFactory",
    "ConfigImageProcessor",
    "ConfigImageProcessorFactory",
    "DMDCalibConfigType",
    "DMDCalibConfigTypeFactory",
    "DmdCalibrationConfig",
    "DmdCalibrationConfigFactory",
    "FileNameConfig",
    "Frame",
    "FrameMetaData",
    "FrameMetaDataFactory",
    "ImageConfigType",
    "ImageConfigTypeFactory",
    "ImageProcessorConfig",
    "ImageProcessorConfigFactory",
    "ObjectiveConfigType",
    "ObjectiveConfigTypeFactory",
    "SoftwareFocusConfig",
    "SoftwareFocusConfigFactory",
    "SoftwareFocusConfigNew",
    "SoftwareFocusConfigNewFactory",
]
