from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, Optional, Union, cast

import cv2
import numpy as np

import delta
from delta.config import Config  # TODO: ask about putting config into init

from evomachine.config import ConfigImage
from evomachine.exceptions import ImageProcessingError, ErrorCode


class PositionRT(delta.pipeline.Position):
    def __init__(
        self,
        position_nb: int,
        config: delta.config.Config,
        cfg_image: ConfigImage,
        verbose: Optional[int] = 0,
    ) -> None:
        super().__init__(position_nb=position_nb, config=config)

        self.roi_boxes: list[delta.utils.CroppingBox] = []
        "List of CroppingBox indexed by i_roi"
        self.drifttemplate: delta.utils.Image = np.empty((cfg_image.pxl_vert, cfg_image.pxl_horiz), cfg_image.pxl_dtype)
        "Drift template obtained from reference image"
        self.driftcorbox: delta.utils.CroppingBox = delta.utils.CroppingBox(0, 0, 0, 0)
        "Cropping box used to correct drift"
        self._is_initialised: bool = False
        "Flag set to true after calling initialise()"

        self.verbose = verbose

    def initialise(
        self,
        reference: np.ndarray[(int, int, int), ConfigImage.pxl_dtype],
    ) -> None:
        self._msg("Starting initialisation")

        # Rotation correction
        if self.config.rotation_correction:
            self.rotate = delta.utils.deskew(reference[0, :, :])
            self._msg(f"Rotation correction: {self.rotate} degrees")
            reference[0, :, :] = delta.utils.imrotate(reference[0, :, :], self.rotate)
            for i_chan in range(reference.shape[0]):
                reference[i_chan, :, :] = delta.utils.imrotate(reference[i_chan, :, :], self.rotate)

        # Find ROIs
        if "rois" in self.config.models:
            self.roi_boxes = delta.pipeline.Position.find_roi_boxes(reference[0, :, :], self.config)
        else:
            self.roi_boxes = [delta.utils.CroppingBox.full(reference[0, :, :])]

        # Get drift correction template and box
        if self.config.drift_correction:
            self.drifttemplate = delta.utils.getDriftTemplate(
                self.roi_boxes,
                reference[0, :, :],
                whole_frame=self.config.whole_frame_drift,
            )
            self.driftcorbox = delta.utils.CroppingBox.full(reference[0, :, :])
            if not self.config.whole_frame_drift:
                self.driftcorbox.ybr = max(box.ytl for box in self.roi_boxes)

        # Instantiate ROIs with 2x reference
        self.rois = [
            delta.pipeline.ROI(
                img_stack=[box.crop(reference[0, :, :]), box.crop(reference[0, :, :])],
                fluo_stack=[[box.crop(img) for img in reference[1:, :, :]],
                            [box.crop(img) for img in reference[1:, :, :]]],
                roi_nb=i_roi,
                first_frame=0,
                box=box,
                config=self.config,
            )
            for i_roi, box in enumerate(self.roi_boxes)
        ]

        # Run pipeline after init
        self.segment(frames=range(2))
        self.track(frames=range(2))
        self.compute_growthrates(frames=range(2))

        self._is_initialised = True

    def process_new_frame(self, new_frame: np.ndarray[(int, int, int), ConfigImage.pxl_dtype]):
        if not self._is_initialised:
            raise ImageProcessingError("Position {} not initialised.".format(self.position_nb),
                                       ErrorCode.ERROR_NOT_INITIALISED)

        self._preprocess_new_frame(new_frame=new_frame)
        self.segment(frames=range(1, 2))
        self.track(frames=range(1, 2))
        self.compute_growthrates(frames=range(1, 2))

    def _preprocess_new_frame(
        self,
        new_frame: np.ndarray[(int, int, int), ConfigImage.pxl_dtype],
    ) -> None:
        self._msg("Starting pre-processing of new frame")

        # Rotation correction
        if self.config.rotation_correction:  # TODO: remove conditional statement
            for i_chan in range(new_frame.shape[0]):
                new_frame[i_chan, :, :] = delta.utils.imrotate(new_frame[i_chan, :, :], self.rotate)

        # Drift correction
        if self.config.drift_correction:  # TODO: remove conditional statement
            int_frame = delta.utils.to_integer_values(new_frame[0, :, :], np.uint8)
            drift_corr_frame = self.driftcorbox.crop(int_frame)
            res = cv2.matchTemplate(drift_corr_frame, self.drifttemplate, cv2.TM_CCOEFF_NORMED)
            _, _, _, max_loc = cv2.minMaxLoc(res)
            y_corr = max_loc[0] - res.shape[1] / 2
            x_corr = max_loc[1] - res.shape[0] / 2
            transformation = np.array([[1, 0, -y_corr], [0, 1, -x_corr]], dtype=np.float32)
            self.drift_values[0].append(x_corr)
            self.drift_values[1].append(y_corr)

            for i_chan in range(new_frame.shape[0]):
                new_frame[i_chan, :, :] = cv2.warpAffine(new_frame[i_chan, :, :], transformation,
                                                         new_frame.shape[2:0:-1])

        # Swap images and assign new frame
        for i_roi, box in enumerate(self.roi_boxes):
            new_roi = box.crop(new_frame[0, :, :])
            (
                self.rois[i_roi].img_stack[0],
                self.rois[i_roi].img_stack[1]
            ) = (
                self.rois[i_roi].img_stack[1],
                self.rois[i_roi].img_stack[0]
            )
            self.rois[i_roi].img_stack[1] = (new_roi - new_roi.min()) / new_roi.ptp()
            (
                self.rois[i_roi].fluo_stack[0, :, :, :],
                self.rois[i_roi].fluo_stack[1, :, :, :]
            ) = (
                self.rois[i_roi].fluo_stack[1, :, :, :],
                self.rois[i_roi].fluo_stack[0, :, :, :]
            )
            self.rois[i_roi].fluo_stack[1, :, :, :] = np.array([box.crop(img) for img in new_frame[1:, :, :]])
