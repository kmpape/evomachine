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


# Load config
presets, batch_size = ("mothermachine", 3)

if presets == "2D":
    config = DEFAULT_CONFIG_2D
else:
    config = DEFAULT_CONFIG_MOTHERMACHINE

# Files
training_set_delta = config.training_set_path("seg")  # '/home/hslab/.cache/delta/training_sets/mothermachine/training/chambers_seg_set/train'
training_set = Path('/mnt/EvomachineData/Symbac_Training_Data_2023-12-22_ovalcells')
standard_model = "/home/hslab/.cache/delta/models/unet_moma_seg.hdf5"
model_20131221 = "/home/hslab/workspace_python/conda_evomachine3.9/evomachine_repo/delta_models/2023-12-21_widecells_seg.hdf5"
model_20131222 = "/home/hslab/workspace_python/conda_evomachine3.9/evomachine_repo/delta_models/2023-12-22_ovalcells_seg.hdf5"

path_old_model = model_20131221
path_new_model = model_20131222

# Get example images to compare training sets

# Training parameters:
# batch_size = 2
epochs = 10
# steps_per_epoch = 30
patience = 50

# Data generator parameters:
data_gen_args = dict(
    rotation=2,
    rotations_90d=presets == "2D",
    zoom=0.15,
    horizontal_flip=True,
    vertical_flip=True,
    illumination_voodoo=True,
    gaussian_noise=0.03,
    gaussian_blur=1,
)

# Generator init:
ds_train, ds_val = load_training_dataset_seg(
    dataset_path=training_set,
    target_size=config.target_size_seg,
    crop=presets == "2D",
    kw_data_aug=data_gen_args,
    validation_split=0.05,
    stack=True,
    multiply=255,
)

# Define model:
model = unet_seg(input_size=config.target_size_seg + (1,), pretrained_weights=path_old_model)
model.summary()

# Callbacks:
model_checkpoint = ModelCheckpoint(
    path_new_model, monitor="loss", verbose=2, save_best_only=True
)
early_stopping = EarlyStopping(
    monitor="loss", mode="min", verbose=2, patience=patience
)

# Train:
# model.fit(
#     ds_train,
#     steps_per_epoch=steps_per_epoch,
#     epochs=epochs,
#     validation_data=ds_val,
#     callbacks=[model_checkpoint, early_stopping],
# )
model.fit(
    ds_train,
    epochs=epochs,
    validation_data=ds_val,
    callbacks=[model_checkpoint, early_stopping],
)

num_test = 3
unprocessed = sorted(
    glob.glob(str(training_set) + "/img/*.tif")
)
unprocessed = random.sample(unprocessed, num_test)
processed = [p.replace("/img/", "/seg/") for p in unprocessed]

# Load up model:
model_new = unet_seg(input_size=config.target_size_seg + (1,))
model_new.load_weights(path_new_model)
model_old = unet_seg(input_size=config.target_size_seg + (1,))
model_old.load_weights(path_old_model)

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


# # Input image sequence (change to whatever images sequence you want to evaluate):
# inputs_folder_delta = "/home/hslab/workspace_python/symbac_daoxin/tests/testdata/testdata_evomachine/testimg2"
# inputs_folder_evo = "/home/hslab/workspace_python/symbac_daoxin/tests/testdata/testdata_evomachine/testimg"
#
# # List files in inputs folder:
# num_test = 3
# unprocessed_delta = sorted(
#     glob.glob(inputs_folder_delta + "/*.tif") + glob.glob(inputs_folder_delta + "/*.png")
# )
# unprocessed_evo = sorted(
#     glob.glob(inputs_folder_evo + "/*.tif") + glob.glob(inputs_folder_evo + "/*.png")
# )
# unprocessed_delta = unprocessed_delta[0:num_test]
# unprocessed_evo = unprocessed_evo[0:num_test]
# validation = [x.replace("testimg", "seg") for x in unprocessed_evo]
#
# # Load up model:
# model = unet_seg(input_size=config.target_size_seg + (1,))
# model.load_weights(new_model)
#
# # Input data generator:
# predGene_delta = predictGenerator_seg(
#     Path(inputs_folder_delta),
#     files_list=[Path(p) for p in unprocessed_delta],
#     target_size=config.target_size_seg,
#     crop_windows=False,
# )
# predGene_evo = predictGenerator_seg(
#     Path(inputs_folder_evo),
#     files_list=[Path(p) for p in unprocessed_evo],
#     target_size=config.target_size_seg,
#     crop_windows=False,
# )
#
# # mother machine: Don't crop images into windows
# results_delta = model.predict(predGene_delta, verbose=1)[:, :, :, 0]
# results_evo = model.predict(predGene_evo, verbose=1)[:, :, :, 0]
#
# # Post process results (binarize + light morphology-based cleaning):
# results_delta = postprocess(results_delta, crop=False)
# results_evo = postprocess(results_evo, crop=False)
#
# predGene_delta = predictGenerator_seg(
#     Path(inputs_folder_delta),
#     files_list=[Path(p) for p in unprocessed_delta],
#     target_size=config.target_size_seg,
#     crop_windows=False,
# )
# inputs_delta = [x for x in predGene_delta]
# predGene_evo = predictGenerator_seg(
#     Path(inputs_folder_evo),
#     files_list=[Path(p) for p in unprocessed_evo],
#     target_size=config.target_size_seg,
#     crop_windows=False,
# )
# inputs_evo = [x for x in predGene_evo]
#
# nsamples = 3
# fig, ax = plt.subplots(nsamples, 4)
# for i in range(nsamples):
#     ax[i, 0].imshow(inputs_delta[i][0])
#     ax[i, 1].imshow(results_delta[i])
#     ax[i, 2].imshow(inputs_evo[i][0])
#     ax[i, 3].imshow(results_evo[i])
#
# plt.show()
