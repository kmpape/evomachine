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
from asitiger.command import CRISPState, Command
from evomachine.acquisition import EvoCamera
from evomachine.config import DEVICE_CONFIG_EVO_TEST, CRISP_CONFIG_DEFAULT, OBJECTIVE_CONFIG_OIL, \
    OBJECTIVE_CONFIG_AIR, IMAGE_CONFIG_DEFAULT, ConfigDevice, ConfigFocus, ConfigLED, ConfigCRISP
from evomachine.dmd import DMDControl, DMDColor

from skimage.io import imread
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from delta.config import _DELTA_DIR, DEFAULT_CONFIG_2D, DEFAULT_CONFIG_MOTHERMACHINE
from delta.data import load_training_dataset_seg, trainGenerator_track, saveResult_seg, predictGenerator_seg, postprocess, readreshape
from delta.model import unet_rois, unet_seg, unet_track

from delta.data import saveResult_seg, predictGenerator_seg, postprocess, readreshape
from delta.model import unet_seg
import delta.utils as utils


# Load config
presets, batch_size = ("mothermachine", 3)

if presets == "2D":
    config = DEFAULT_CONFIG_2D
else:
    config = DEFAULT_CONFIG_MOTHERMACHINE

# Files
training_set_delta = config.training_set_path("seg")
training_set = Path('/mnt/EvomachineData/Symbac_Training_Data_2023-12-21_widecells')
standard_model = "/home/hslab/.cache/delta/models/unet_moma_seg.hdf5"
model_20131221 = "/home/hslab/workspace_python/conda_evomachine3.9/evomachine_repo/delta_models/2023-12-21_widecells_seg.hdf5"
model_20131222 = "/home/hslab/workspace_python/conda_evomachine3.9/evomachine_repo/delta_models/2023-12-22_ovalcells_seg.hdf5"


# Input image sequence (change to whatever images sequence you want to evaluate):
# List files in inputs folder:
num_test = 3
unprocessed = sorted(
    glob.glob(str(training_set) + "/img/*.tif")
)
unprocessed = random.sample(unprocessed, num_test)
processed = [p.replace("/img/", "/seg/") for p in unprocessed]

# Load up model:
model_new = unet_seg(input_size=config.target_size_seg + (1,))
model_new.load_weights(model_20131221)
model_old = unet_seg(input_size=config.target_size_seg + (1,))
model_old.load_weights(model_20131221)

# Input data generator:
predGene_new = predictGenerator_seg(
    Path(training_set),
    files_list=[Path(p) for p in unprocessed],
    target_size=config.target_size_seg,
    crop_windows=False,
)
predGene_old = predictGenerator_seg(
    Path(training_set),
    files_list=[Path(p) for p in unprocessed],
    target_size=config.target_size_seg,
    crop_windows=False,
)

# mother machine: Don't crop images into windows
results_new = model_new.predict(predGene_new, verbose=1)[:, :, :, 0]
results_old = model_old.predict(predGene_old, verbose=1)[:, :, :, 0]

# Post process results (binarize + light morphology-based cleaning):
results_new = postprocess(results_new, crop=False)
results_old = postprocess(results_old, crop=False)

predGene_new = predictGenerator_seg(
    Path(training_set),
    files_list=[Path(p) for p in unprocessed],
    target_size=config.target_size_seg,
    crop_windows=False,
)
inputs = [x for x in predGene_new]

predGene_new = predictGenerator_seg(
    Path(training_set),
    files_list=[Path(p) for p in processed],
    target_size=config.target_size_seg,
    crop_windows=False,
)
outputs_symbac = [x for x in predGene_new]

nsamples = 3
fig, ax = plt.subplots(nsamples, 4)
for i in range(nsamples):
    ax[i, 0].imshow(inputs[i][0])
    ax[i, 0].set_title('Input')
    ax[i, 1].imshow(results_old[i])
    ax[i, 1].set_title('Old')
    ax[i, 2].imshow(results_new[i])
    ax[i, 2].set_title('New')
    ax[i, 3].imshow(outputs_symbac[i][0])
    ax[i, 3].set_title('Symbac')

plt.show()
