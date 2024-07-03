import os
from datetime import datetime
import numpy as np
from pathlib import Path
import pickle
from typing import Dict, List, Tuple, Type, Union

from evomachine.commands import AutomatonCommand
from evomachine.config import get_logger, ConfigCameraFactory, ConfigImageProcessorFactory, USE_DMD_SOCKET, \
    ConfigImageProcessor
if USE_DMD_SOCKET:
    from evomachine.dmd_socket import DMD_WIDTH_HEIGHT
else:
    from evomachine.dmd import DMD_WIDTH_HEIGHT
from evomachine.exceptions import EvoMachineError
from evomachine.evotypes import LEDType, MagnetModeType
from evomachine.strategy import AbstractStrategy
import delta

from datetime import datetime

logger = get_logger(name=__name__)

def preprocessor_config():
    default_channels = [LEDType.LED_450_NM, LEDType.LED_515_NM, LEDType.LED_565_NM, LEDType.LED_645_NM]
    cfg_delta = delta.config.Config.default("mothermachine")
    cfg_delta.whole_frame_drift = True
    cfg_delta.target_size_rois = (1024, 1024)
    cfg_delta.tolerable_resizing_rois = 0
    cfg_delta.model_file_rois = Path("/home/hslab/workspace_python/delta3.0/de-lta-rt/"
                                        "evomodels/evo_roi_2024-05-08.keras")  # TODO relative paths
    return ConfigImageProcessor(
        cfg_delta=cfg_delta,
        channels=default_channels,
        channel_seg=LEDType.LED_450_NM,
        channel_rot=LEDType.LED_450_NM,
        channel_roi=LEDType.LED_450_NM,
        preproc_enabled=False,
        seg_enabled=False,
        roi_enabled=False,
    )

# Define configuration objects
CAMERA_CONFIG = ConfigCameraFactory.default_air_config()
PROCESSOR_CONFIG = preprocessor_config()

class MagnetOnOffStrategy(AbstractStrategy):
    """
    Example strategy that projects light on half of the FoV for <timer_interval> seconds and then switches to the other
    half. The strategy images every <imaging_interval> seconds and saves the images. Additionally, the strategy saves
    the time at which the changes happened and saves them in a pickle file.
    """
    def __init__(self, cfg: ConfigImageProcessor):
        super().__init__(cfg=cfg)

        self.path_to_save = Path("/media/hslab/Data/ImageData/Gabi/"+datetime.today().strftime('%Y_%m_%d')+"/")
        if not os.path.exists(self.path_to_save):
            os.mkdir(self.path_to_save)

        self.exposure_time: int = 80  # in ms
        self.imaging_channel: LEDType = LEDType.LED_450_NM
        self.imaging_interval: float = 200 / 1000  # in seconds

        self.magnet_on = False
        self.images_taken = 0
        self.images_per_magnet = 10 # switch magnet every 10 images

    def _initialise(self) -> List[AutomatonCommand]:
        """
        Initialise the strategy. Following attributes are available:

        self.field_of_views : Dict[int, Coordinate]
            Dictionary with fov_id as key and Coordinate as value.
        self.positions: Dict[int, List[int]]
            Dictionary with fov_id as key and list of pos_id as value.
        self.region_of_interests: Dict[int, List[int]]
            Dictionary with pos_id as key and list of roi_id as value.
        self.config_camera: ConfigCamera
            Object defining camera configuration.

        Returns
        -------
        List[AutomatonCommand]
            List of commands to be executed by the automaton at the first iteration.
        """
        self.magnet_on = False
        self.images_taken = 0
        logger.info("Initialising MagnetOnOffStrategy.")
        
        cmds = [
            self.command_factory.command_calibrate_magnet(),
            self.command_factory.command_wait(
                duration=5,
                set_live_mode=False,
                brightness=0,
            ),
            self.command_factory.command_calibrate_hall(
                hall_id=0,    
            ),
            self.command_factory.command_wait(
                duration=5,
                set_live_mode=False,
                brightness=0,
            ),
            self.command_factory.command_magnet(
                enable=True,
                value=0.0,
                mode=MagnetModeType.FIELD_SET
            ),
        ]
        
        return cmds

    def _callback(
            self,
            fov_id: int,
            data: List[AutomatonCommand],
            errors: List[EvoMachineError],
    ) -> List[AutomatonCommand]:
        """
        Callback function for the strategy. This function is called by the
        automaton when new data is available.

        Parameters
        ----------
        `fov_id` : int
            The id of the current field of view.
        `t` : int
            The time of the data.
        `data` : dict
            Processed image data such as cell positions.

        Returns
        -------
        List[AutomatonCommand]
            List of commands to be executed at the next iteration.
        """
        logger.info("FOV {} with data =\n{} \nand errors = {}.".format(
            fov_id,
            '    \n'.join(str(d) for d in data),
            errors,
        ))
        
        self.images_taken += 1
        if self.images_taken >= self.images_per_magnet:
            self.images_taken = 0
            self.magnet_on = not self.magnet_on
        
        cmds = [
            self.command_factory.command_magnet(
                enable=self.magnet_on,
                value=5.0,
                mode=MagnetModeType.FIELD_SET
            ),
            self.command_factory.command_read_hall(
                hall_id=0,
            ),
            self.command_factory.command_image(
                channels=[self.imaging_channel],
                exposure_time=self.exposure_time,
                segment=False,
                save=True,
                brightness=10,
                force_led=True,
                reset_led=False,
            ),
            self.command_factory.command_wait(
                duration=self.imaging_interval - self.exposure_time / 1000,
                set_live_mode=False,
                brightness=10,
            )
        ]
        
        return cmds

    def finalise(self) -> List[AutomatonCommand]:
        """
        Save everything else than images here.

        Returns
        -------
        List[AutomatonCommand]
            List of commands to be executed by the automaton at the last iteration.
        """
        logger.info("Finalising strategy and saving data.")
        # current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # filename = self.path_to_save / f"strategy_MagnetOnOff_{current_time}.pkl"
        # with open(str(filename), "wb") as file:
        #     pickle.dump({'time_changes': self.time_changes}, file)

        cmds = [
            self.command_factory.command_magnet(
                enable=False,
                value=0.0,
                mode=MagnetModeType.CURRENT_SET
            )
        ]
        
        return cmds
