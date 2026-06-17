import glob
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import random
import sys
import time

sys.path.append("/home/hslab/workspace_python/conda_evomachine3.9/asitiger")
sys.path.append("/home/hslab/workspace_python/conda_evomachine3.9/evomachine_repo")
from asitiger.command import CRISPSetState, Command
from evomachine.acquisition_bkp import EvoCamera
from evomachine.config import DEVICE_CONFIG_EVO_TEST, CRISP_CONFIG_DEFAULT, OBJECTIVE_CONFIG_OIL, \
    OBJECTIVE_CONFIG_AIR, IMAGE_CONFIG_DEFAULT, ConfigDevice, ConfigFocus, ConfigLED, ConfigCRISP
from evomachine.dmd_pygame import DMDControl, DMDColor

from skimage.io import imread
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from delta.config import _DELTA_DIR, DEFAULT_CONFIG_2D, DEFAULT_CONFIG_MOTHERMACHINE
from delta.data import load_training_dataset_seg, trainGenerator_track, saveResult_seg, predictGenerator_seg, postprocess, readreshape
from delta.model import unet_rois, unet_seg, unet_track

from delta.data import saveResult_seg, predictGenerator_seg, postprocess, readreshape
from delta.model import unet_seg
import delta.utils as utils


symbac_gen_data_path = Path('/mnt/EvomachineData/Symbac_Training_Data_2023-12-22_ovalcells')
symbac_filenames = sorted(glob.glob(str(symbac_gen_data_path/"img/*.tif")))

im = imread(symbac_filenames[0])



