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
from evomachine.config import EVOMACHINE_DIR

from skimage.io import imread
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from delta.config import _DELTA_DIR, DEFAULT_CONFIG_2D, DEFAULT_CONFIG_MOTHERMACHINE
from delta.data import load_training_dataset_seg, trainGenerator_track, saveResult_seg, predictGenerator_seg, postprocess, readreshape
from delta.model import unet_rois, unet_seg, unet_track

from delta.data import saveResult_seg, predictGenerator_seg, postprocess, readreshape
from delta.model import unet_seg
import delta.utils as utils


# Load config
config = DEFAULT_CONFIG_MOTHERMACHINE
config.model_file_rois
print(f"Original target_size_seg: {config.target_size_seg}")
print(f"Original target_size_rois: {config.target_size_rois}")
config.target_size_rois = (3200, 3200)
print(f"New target_size_rois: {config.target_size_rois}")
savefile = EVOMACHINE_DIR.parent / "delta_models/evo_roi_v0.hdf5"
epochs = 3
steps_per_epoch = 30
patience = 50
# training_set_delta = config.training_set_path("rois")
training_set = "/home/lady5906/workspace_python/chambers/ImageData/roi_data_v2_LED450NM"

# Data generator parameters:
data_gen_args = dict(
    rotation=3,
    shiftX=0.1,
    shiftY=0.1,
    zoom=0.25,
    horizontal_flip=True,
    vertical_flip=True,
    rotations_90d=True,
    histogram_voodoo=True,
    illumination_voodoo=True,
    gaussian_noise=0.03,
    gaussian_blur=1,
)

# Generator init:
print("Loading training dataset")
ds_train, ds_val = load_training_dataset_seg(
    dataset_path=Path(training_set),
    target_size=config.target_size_rois,
    crop=False,
    kw_data_aug=data_gen_args,
    validation_split=0.05,
    multiply=None,
)

# Define model:
model = unet_seg(input_size=config.target_size_rois + (1,))
model.summary()

# Callbacks:
model_checkpoint = ModelCheckpoint(
    savefile, monitor="loss", verbose=2, save_best_only=True
)
early_stopping = EarlyStopping(
    monitor="loss", mode="min", verbose=2, patience=patience
)

# Train:
print("Starting training")
model.fit(
    ds_train,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=ds_val,
    callbacks=[model_checkpoint, early_stopping],
)

print("Training finished")


print("Producing example images")
num_test = 3
unprocessed = sorted(
    glob.glob(str(training_set) + "/img/*.tiff")
)
unprocessed = random.sample(unprocessed, num_test)
processed = [p.replace("/img/", "/seg/") for p in unprocessed]

# Load up model:
model_new = unet_seg(input_size=config.target_size_seg + (1,))
model_new.load_weights(savefile)

# Input data generator:
predGene_new = predictGenerator_seg(
    Path(training_set),
    files_list=[Path(p) for p in unprocessed],
    target_size=config.target_size_seg,
    crop_windows=False,
)

# mother machine: Don't crop images into windows
results_new = model_new.predict(predGene_new, verbose=1)[:, :, :, 0]

# Post process results (binarize + light morphology-based cleaning):
results_new = postprocess(results_new, crop=False)

predGene_new = predictGenerator_seg(
    Path(training_set),
    files_list=[Path(p) for p in unprocessed],
    target_size=config.target_size_seg,
    crop_windows=False,
)
inputs = [x for x in predGene_new]

nsamples = 3
fig, ax = plt.subplots(nsamples, 2)
for i in range(nsamples):
    ax[i, 0].imshow(inputs[i][0])
    ax[i, 0].set_title('Input')
    ax[i, 1].imshow(results_new[i])
    ax[i, 1].set_title('New')

plt.show()
