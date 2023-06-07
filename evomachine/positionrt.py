from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, Optional, Union, cast

import cv2
import numpy as np

import delta
from delta.config import Config  # TODO: ask about putting config into init

from evomachine.config import ConfigImage
from evomachine.exceptions import ImageProcessingError, ErrorCode
from evomachine.utils import Timer

TIMER_POSITION = Timer(timer_level=0, name="PositionRT", enabled=True)


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

        self.segmentation_model = self.config.model("seg")
        "Preloading segmentation model."
        self.tracking_model = self.config.model("track")
        "Preloading tracking model."

        self.verbose = verbose

    def initialise(
        self,
        reference: np.ndarray[(int, int, int), 'ConfigImage.pxl_dtype'],
    ) -> None:
        self._msg("Starting initialisation")
        TIMER_POSITION.start("initialise", 0)

        # Rotation correction
        if self.config.rotation_correction:
            self.rotate = delta.utils.deskew(reference[0, :, :])
            self._msg(f"Rotation correction: {self.rotate} degrees")
            for i_chan in range(reference.shape[0]):
                reference[i_chan, :, :] = delta.utils.imrotate(reference[i_chan, :, :], self.rotate)

        # Find ROIs
        if "rois" in self.config.models:
            self.roi_boxes = delta.pipeline.Position.find_roi_boxes(reference[0, :, :], self.config)
        else:
            self.roi_boxes = [delta.utils.CroppingBox.full(reference[0, :, :])]

        # Get drift correction template and box
        if self.config.drift_correction:
            self.drifttemplate = delta.utils.to_integer_values(
                delta.utils.getDriftTemplate(
                    self.roi_boxes,
                    reference[0, :, :],
                    whole_frame=self.config.whole_frame_drift,
                ),
                np.uint8
            )
            self.driftcorbox = delta.utils.CroppingBox.full(reference[0, :, :])
            if not self.config.whole_frame_drift:
                self.driftcorbox.ybr = max(box.ytl for box in self.roi_boxes)
            # Need to apply drift correction to match the output of DeLTA
            int_frame = delta.utils.to_integer_values(reference[0, :, :], np.uint8)
            drift_corr_frame = self.driftcorbox.crop(int_frame)
            res = cv2.matchTemplate(drift_corr_frame, self.drifttemplate, cv2.TM_CCOEFF_NORMED)
            _, _, _, max_loc = cv2.minMaxLoc(res)
            y_corr = max_loc[0] - res.shape[1] / 2
            x_corr = max_loc[1] - res.shape[0] / 2
            transformation = np.array([[1, 0, -y_corr], [0, 1, -x_corr]], dtype=np.float32)
            self.drift_values[0].append(x_corr)
            self.drift_values[1].append(y_corr)

            for i_chan in range(reference.shape[0]):
                reference[i_chan, :, :] = cv2.warpAffine(reference[i_chan, :, :], transformation,
                                                         reference.shape[2:0:-1])

        # Instantiate ROIs with 2x reference
        self.rois = [
            ROIRT(
                img_stack=[box.crop(reference[0, :, :]), box.crop(reference[0, :, :])],
                fluo_stack=[[box.crop(img) for img in reference[1:, :, :]],
                            [box.crop(img) for img in reference[1:, :, :]]],
                roi_nb=i_roi,
                first_frame=0,
                box=box,
                config=self.config,
                verbose=self.verbose,
            )
            for i_roi, box in enumerate(self.roi_boxes)
        ]

        # Run pipeline after init
        self.segment(frames=range(2))
        self.track(frames=range(2))
        self.compute_growthrates(frames=range(2))

        self._is_initialised = True
        TIMER_POSITION.stop("initialise", 0)

    def process_new_frame(
            self,
            new_frame: np.ndarray[(int, int, int), 'ConfigImage.pxl_dtype']
    ) -> None:
        if not self._is_initialised:
            raise ImageProcessingError("Position {} not initialised.".format(self.position_nb),
                                       ErrorCode.ERROR_NOT_INITIALISED)
        TIMER_POSITION.start("process_new_frame:_preprocess_new_frame", 0)
        self._preprocess_new_frame(new_frame=new_frame)
        TIMER_POSITION.stop("process_new_frame:_preprocess_new_frame", 0)

        TIMER_POSITION.start("process_new_frame:segment", 0)
        self.segment(frames=range(1, 2))  # This does not affect lineages
        TIMER_POSITION.stop("process_new_frame:segment", 0)

        TIMER_POSITION.start("process_new_frame:track", 0)
        self.track(frames=range(1, 2))  # This does affect lineages
        TIMER_POSITION.stop("process_new_frame:track", 0)

        TIMER_POSITION.start("process_new_frame:compute_growthrates", 0)
        self.compute_growthrates(frames=range(1, 2))  # TODO
        TIMER_POSITION.stop("process_new_frame:compute_growthrates", 0)

    def _preprocess_new_frame(
        self,
        new_frame: np.ndarray[(int, int, int), 'ConfigImage.pxl_dtype'],
    ) -> None:
        self._msg("Starting pre-processing of new frame")

        # Rotation correction
        TIMER_POSITION.start("_preprocess_new_frame:rotation_correction", 1)
        if self.config.rotation_correction:  # TODO: remove conditional statement
            for i_chan in range(new_frame.shape[0]):
                new_frame[i_chan, :, :] = delta.utils.imrotate(new_frame[i_chan, :, :], self.rotate)
        TIMER_POSITION.stop("_preprocess_new_frame:rotation_correction", 1)

        # Drift correction
        TIMER_POSITION.start("_preprocess_new_frame:drift_correction", 1)
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
        TIMER_POSITION.stop("_preprocess_new_frame:drift_correction", 1)

        # Swap images and assign new frame
        TIMER_POSITION.start("_preprocess_new_frame:swap", 1)
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
            # Swap seg and label stacks
            (
                self.rois[i_roi].seg_stack[0],
                self.rois[i_roi].seg_stack[1]
            ) = (
                self.rois[i_roi].seg_stack[1],
                self.rois[i_roi].seg_stack[0]
            )
            (
                self.rois[i_roi].label_stack[0],
                self.rois[i_roi].label_stack[1]
            ) = (
                self.rois[i_roi].label_stack[1],
                self.rois[i_roi].label_stack[0]
            )
        TIMER_POSITION.stop("_preprocess_new_frame:swap", 1)

    def segment(self, frames: range) -> None:
        """
        Segment cells in all ROIs in position.

        Parameters
        ----------
        frames : range
            Frames to run.

        Returns
        -------
        None.

        """
        self._msg(f"Starting segmentation ({len(frames)} frames)")

        for iroi, roi in enumerate(self.rois, start=1):
            self._msg(f"Segmentation - ROI {iroi}/{len(self.rois)}")
            roi.segment(frames, self.segmentation_model)

    def track(self, frames: range) -> None:
        """
        Track cells in all ROIs in position.

        Parameters
        ----------
        frames : range
            Frames to track.

        Returns
        -------
        None.

        """
        self._msg(f"Starting tracking ({len(frames)} frames)")

        for iroi, roi in enumerate(self.rois, start=1):
            self._msg(f"Tracking - ROI {iroi}/{len(self.rois)}")
            roi.track(frames, self.tracking_model)


class ROIRT(delta.pipeline.ROI):
    def __init__(
        self,
        img_stack: list[delta.utils.Image],
        fluo_stack: list[list[delta.utils.Image]],
        roi_nb: int,
        first_frame: int,
        box: delta.utils.CroppingBox,
        config: delta.config.Config,
        verbose: Optional[int] = 0,
    ) -> None:
        super().__init__(
            img_stack=img_stack,
            fluo_stack=fluo_stack,
            roi_nb=roi_nb,
            first_frame=first_frame,
            box=box,
            config=config,
        )
        self._frame_id: int = 0
        "This flag keeps track of the actual time index, which might be different from the position index in img_stack"\
        "It is automatically incremented in process_segmentation_outputs() and assumes that things are called in order."

        assert len(self.img_stack) == 2  # Otherwise, change code in process_XXX_outputs

        # Initialise seg_stack and label_stack lists using length of img_stack
        self.seg_stack = [
            np.empty(shape=self.config.target_size_seg, dtype=np.dtype("uint8"))
            for _ in range(len(self.img_stack))
        ]
        self.label_stack = [
            np.empty(shape=self.config.target_size_track, dtype=np.dtype("uint8"))
            for _ in range(len(self.img_stack))
        ]

    def process_segmentation_outputs(
        self,
        logits: np.typing.NDArray[np.float32],
        frame: int,
        windows: Optional[tuple[list, list]] = None,
    ) -> None:
        """
        Process outputs after they have been segmented.

        Parameters
        ----------
        logits : 4D array
            Segmentation output array. Dimensions are
            (windows, *self.config.target_size_seg, 1).
        frame : int
            Frame index.
        windows : tuple of 2 lists
            y and x coordinates of crop windows if any, or None.

        Returns
        -------
        None.

        """
        # Stitch windows back together (if needed):
        if windows is None:
            logits = logits[0, :, :, 0]
        else:
            logits = delta.utils.stitch_pic(logits[..., 0], windows[0], windows[1])

        # Binarize:
        seg = delta.data.binarizerange(logits, threshold=0).astype(np.uint8)
        # Crop out segmentation if image was smaller than target_size
        if self.config.crop_windows:
            seg = seg[: self.img_stack[0].shape[0], : self.img_stack[0].shape[1]]
        # Area filtering:
        seg = delta.utils.opencv_areafilt(seg, min_area=self.config.min_cell_area)

        # Append to segmentation results stack:
        # assert len(self.seg_stack) == frame - self.first_frame
        self.seg_stack[frame - self.first_frame] = seg

    def process_tracking_outputs(
        self,
        logits: np.typing.NDArray[np.float32],
        frame: int,
        boxes: list[tuple[delta.utils.CroppingBox, delta.utils.CroppingBox]],
    ) -> None:
        """
        Process output from tracking U-Net.

        Get poles, update lineage and create label_stack.

        Parameters
        ----------
        logits : 4D array
            Tracking output array. Dimensions are
            (previous_cells, *self.config.target_size_track, 1).
        frame : int
            The frame to process for.
        boxes : List of tuples of 2 dicts
            Crop and fill boxes to re-place outputs in the ROI.

        Returns
        -------
        None.

        """
        if frame > 0:  # Changed
            self._frame_id += 1

        # Get scores and attributions:
        # Label frame but numbered 1, 2, 3, 4, etc. (temporary labels)
        labels = delta.utils.label_seg(self.get_seg(frame))
        scores = delta.utils.getTrackingScores(labels, logits[:, :, :, 0], boxes=boxes)

        attributions = delta.utils.getAttributions(scores)
        previous_cell_nbs = (
            delta.utils.getcellsinframe(self.get_labels(frame - 1)[::-1, :])[::-1]
            if frame > self.first_frame
            else []
        )
        assert len(previous_cell_nbs) == attributions.shape[0]
        cell_nbs = [None] * attributions.shape[1]

        # Get poles:
        poles = delta.utils.getpoles(self.get_seg(frame), labels, scaling=self.scaling)

        # Resize labels if not using crop windows:
        if not self.config.crop_windows:
            resize = (
                self.box.xbr - self.box.xtl,
                self.box.ybr - self.box.ytl,
            )
            labels = cv2.resize(labels, resize, interpolation=cv2.INTER_NEAREST)

        # Extract features for all cells in the ROI:
        extracted_features = delta.utils.roi_features(
            labels,
            fluo_frames=self.get_fluo(frame),
        )

        # Make sure the same cell_ids are present in both dicts
        assert poles.keys() == extracted_features.keys()

        # Assign poles to extracted features:
        for cellid, (old_pole, new_pole) in poles.items():
            extracted_features[cellid].old_pole = old_pole
            extracted_features[cellid].new_pole = new_pole

        # Go through old cells
        for cellid, attribs in zip(previous_cell_nbs, attributions):
            if self.lineage.cells[cellid].last_frame != self._frame_id - 1:
                print("frame={}, last_frame={}, _frame_id={}".format(
                    frame, self.lineage.cells[cellid].last_frame, self._frame_id
                ))
            assert self.lineage.cells[cellid].last_frame == self._frame_id - 1  # Changed
            attrib = attribs.nonzero()[0]
            previous_poles = self.lineage.cells[cellid].poles(self._frame_id - 1)  # Changed
            if len(attrib) == 1:
                # Simple tracking event
                [n] = attrib
                features = delta.utils.track_poles(extracted_features[n + 1], *previous_poles)
                self.lineage.extend(cellid, features)
                cell_nbs[n] = cellid
            elif len(attrib) == 2:
                # Division event
                n0, n1 = attrib
                (
                    mother_features,
                    daughter_features,
                    first_cell_is_mother,
                ) = delta.utils.division_poles(
                    extracted_features[n0 + 1],
                    extracted_features[n1 + 1],
                    *previous_poles,
                )
                if not first_cell_is_mother:
                    # mother_features, daughter_features = daughter_features, mother_features
                    n0, n1 = n1, n0
                self.lineage.extend(cellid, mother_features)
                newcellid = self.lineage.create(
                    self._frame_id, daughter_features, motherid=cellid  # Changed
                )
                cell_nbs[n0] = cellid
                cell_nbs[n1] = newcellid

        # Go through new cells
        for n, attribs in enumerate(attributions.T):
            attrib = attribs.nonzero()[0]
            if len(attrib) == 1:
                # Case already treated
                continue
            # Brand new cell event: attribute poles arbitrarily
            if (
                extracted_features[n + 1].old_pole[0]
                >= extracted_features[n + 1].new_pole[0]
            ):
                extracted_features[n + 1].swap_poles()
            cellid = self.lineage.create(
                self._frame_id, extracted_features[n + 1], motherid=None  # Changed
            )
            cell_nbs[n] = cellid

        assert None not in cell_nbs
        # Recompile label frame with new labels
        labels = delta.utils.label_seg(self.get_seg(frame), cell_nbs)

        # Resize image:
        if not self.config.crop_windows:
            resize = (
                self.box.xbr - self.box.xtl,
                self.box.ybr - self.box.ytl,
            )
            labels = cv2.resize(labels, resize, interpolation=cv2.INTER_NEAREST)

        # assert len(self.label_stack) == frame - self.first_frame
        self.label_stack[frame - self.first_frame] = labels
