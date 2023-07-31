import copy
from collections.abc import Sequence
import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union, cast

import cv2
import numpy as np
import tqdm

import delta
from delta.config import Config  # TODO: ask about putting config into init
from delta.pipeline import TIMER_ROI

from evomachine.config import ConfigImage
from evomachine.exceptions import ImageProcessingError, ErrorCode, ErrorContainer
import evomachine.trackingrt as trackingrt
from evomachine.utils import Timer

TIMER_POSITION = Timer(timer_level=0, name="PositionRT", enabled=True)

logger = logging.getLogger(__name__)


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
        self.drifttemplate: delta.utils.Image = np.empty((cfg_image.pxl_vert*cfg_image.tile_image[0],
                                                          cfg_image.pxl_horiz*cfg_image.tile_image[1]),
                                                         cfg_image.pxl_dtype)
        "Drift template obtained from reference image"
        self.driftcorbox: delta.utils.CroppingBox = delta.utils.CroppingBox(0, 0, 0, 0)
        "Cropping box used to correct drift"
        self._is_initialised: bool = False
        "Flag set to true after calling initialise()"

        self.segmentation_model = self.config.model("seg")
        "Preloading segmentation model."
        self.tracking_model = self.config.model("track")
        "Preloading tracking model."

        self.cfg_image = cfg_image
        "Image configuration."

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

        if any(val != 1 for val in self.cfg_image.tile_image):
            reference = np.tile(reference, (1, *self.cfg_image.tile_image))

        # Find ROIs
        if "rois" in self.config.models:
            self.roi_boxes = self.find_roi_boxes(reference[0, :, :], self.config)
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
        if self.cfg_image.use_track_RT:
            self.init_track_rt()
        else:
            self.track(frames=range(2))
        self.compute_growthrates(frames=range(2))

        # Remove redundant cell lineage object
        #for roi in self.rois:
        #    for cell in roi.lineage.cells.values():
        #        cell._features.pop()  # TODO: ask for delete function

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
        self.segment_at_once()
        TIMER_POSITION.stop("process_new_frame:segment", 0)

        TIMER_POSITION.start("process_new_frame:track", 0)
        if self.cfg_image.use_track_RT:
            self.track_rt()
        else:
            self.track_at_once()
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
        if any(val != 1 for val in self.cfg_image.tile_image):
            new_frame = np.tile(new_frame, (1, *self.cfg_image.tile_image))

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
        # For debugging
        self.new_frame = copy.deepcopy(new_frame)

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
            self.rois[i_roi].img_stack[1] = (new_roi - new_roi.min()) / new_roi.ptp()  # noqa
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

    def segment_at_once(self) -> None:
        self._msg(f"Starting segmentation for {len(self.rois)} ROIs")

        TIMER_POSITION.start("segment_at_once:prepare", 1)
        inputs = np.concatenate([roi.get_segmentation_inputs(1)[0] for roi in self.rois])
        TIMER_POSITION.stop("segment_at_once:prepare", 1)

        TIMER_POSITION.start("segment_at_once:predict", 1)
        logits = self.segmentation_model.predict(inputs, batch_size=4, verbose=0)
        TIMER_POSITION.stop("segment_at_once:predict", 1)

        TIMER_POSITION.start("segment_at_once:process", 1)
        for iroi, roi in enumerate(self.rois):
            roi.process_segmentation_outputs(
                logits[iroi: iroi + 1],
                frame=1,
                windows=None,
            )
        TIMER_POSITION.stop("segment_at_once:process", 1)

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

    def track_at_once(self) -> None:
        """
        Track cells in all ROIs in position.
        """
        self._msg(f"Starting tracking for {len(self.rois)} ROIs")

        TIMER_POSITION.start("track_at_once:prepare", 1)
        inputs_with_cells = []
        iroi_with_cells = []
        all_boxes = []
        num_cells_iroi = [0]
        for iroi, roi in enumerate(self.rois):
            inputs, boxes = roi.get_tracking_inputs(frame=1)
            if inputs.shape[0] > 0:  # noqa
                inputs_with_cells.append(inputs)
                iroi_with_cells.append(iroi)
                num_cells_iroi.append(inputs.shape[0])  # noqa
            all_boxes.append(boxes)
        iroi_without_cells = set(range(len(self.rois))) - set(iroi_with_cells)
        num_cells_iroi = np.cumsum(num_cells_iroi)
        TIMER_POSITION.stop("track_at_once:prepare", 1)

        TIMER_POSITION.start("track_at_once:predict", 1)
        all_inputs = np.concatenate(inputs_with_cells)
        logits = self.tracking_model.predict(
            all_inputs,
            batch_size=4,
            verbose=self.verbose,
        )
        TIMER_POSITION.stop("track_at_once:predict", 1)

        TIMER_POSITION.start("track_at_once:process", 1)
        for i, iroi in enumerate(iroi_with_cells):
            self.rois[iroi].process_tracking_outputs(
                logits[num_cells_iroi[i]: num_cells_iroi[i+1]],
                frame=1,
                boxes=all_boxes[iroi]
            )

        logits_empty = np.empty(shape=(0, *self.config.target_size_track, 1))
        for iroi in iroi_without_cells:
            self.rois[iroi].process_tracking_outputs(
                logits_empty,
                frame=1,
                boxes=all_boxes[iroi],
            )
        TIMER_POSITION.stop("track_at_once:process", 1)

    def init_track_rt(self) -> None:
        """
        Track cells in all ROIs in position. Uses the faster tracking algorithm for mother-machine devices.
        """

        for i_roi, roi in enumerate(self.rois):
            logging.debug(f"{self}:init_track_rt for ROI {i_roi}")
            roi.init_track_rt()
            roi.track_rt()

    def track_rt(self) -> None:
        """
        Track cells in all ROIs in position. Uses the faster tracking algorithm for mother-machine devices.
        """
        for i_roi, roi in enumerate(self.rois):
            logging.debug(f"{self}:track_rt for ROI {i_roi}")
            roi.track_rt()

    def find_roi_boxes(self, reference: delta.utils.Image, config: Config) -> list[delta.utils.CroppingBox]:
        """
        Use U-Net to detect ROIs (chambers etc...).

        Parameters
        ----------
        reference : utils.Image
            Reference image to use to detect ROIs.
        config : Config
            DeLTA configuration object.

        Returns
        -------
        boxes : List[CroppingBox]
            List of ROI boxes.

        """
        # Rescale pixel values between 0 and 1 for the old model
        reference = (reference - reference.min()) / reference.ptp()  # noqa

        if self.cfg_image.crop_out_ROI:
            reference_shape = reference.shape
            frac = 0.5
            res_width = abs(reference.shape[1]-self.config.target_size_rois[0]) < frac*self.config.target_size_rois[0]
            res_height = abs(reference.shape[0]-self.config.target_size_rois[1]) < frac*self.config.target_size_rois[1]
            if res_width and res_height:
                reference = cv2.resize(reference, self.config.target_size_rois)
            elif res_width:
                reference = cv2.resize(reference, (self.config.target_size_rois[0], reference.shape[0]))
            elif res_height:
                reference = cv2.resize(reference, (reference.shape[1], self.config.target_size_rois[1]))
            # Crop out windows
            inputs, win_y, win_x = delta.utils.create_windows(image=reference, target_size=self.config.target_size_rois)
            # Predict
            logits = self.config.model("rois").predict(inputs[:, :, :, np.newaxis], verbose=0)
            rois_pred = delta.utils.stitch_pic(logits[..., 0], win_y, win_x)
            # Clean up
            rois_mask = delta.data.postprocess(
                cv2.resize(np.squeeze(rois_pred), reference_shape[::-1]),
                min_size=self.config.min_roi_area,
            )
        else:
            # Predict
            rois_pred = self.config.model("rois").predict(
                cv2.resize(reference, self.config.target_size_rois)[
                    np.newaxis, :, :, np.newaxis
                ],
                verbose=0,
            )
            # Clean up
            rois_mask = delta.data.postprocess(
                cv2.resize(np.squeeze(rois_pred), reference.shape[::-1]),
                min_size=config.min_roi_area,
            )

        # Get boxes
        # Implementation note: cv2.findContours (even including
        # cv2.boundingRect) is about twice as fast as
        # cv2.connectedComponentsWithStats here.
        roi_boxes = []
        contours = delta.utils.find_contours(rois_mask)
        for chamber in contours:
            xtl, ytl, boxwidth, boxheight = cv2.boundingRect(chamber)
            roi_boxes.append(
                delta.utils.CroppingBox(
                    xtl=xtl,
                    # -10% of height to make sure the top is not cropped
                    ytl=ytl - int(0.1 * boxheight),
                    xbr=xtl + boxwidth,
                    ybr=ytl + boxheight,
                )
            )

        # Sorting by top-left X (normally sorted by top-left Y)
        roi_boxes.sort(key=lambda box: box.xtl)

        return roi_boxes

    def __str__(self):
        return f"POS_{self.position_nb:03}"


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
        "It is automatically incremented in process_segmentation_outputs() and assumes that things are called in order."  # noqa

        self.error_container: ErrorContainer = ErrorContainer()
        "Deque object for recording errors."

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

        self.state_old: List[Dict[str, Union[float, int, bool]]] = []
        "Variable for tracking algorithm at previous time step."
        self.max_id: int = 0
        "Keeps track of the maximum ID ever assigned to a cell in the ROI."

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
        self.tmp_labels = copy.deepcopy(labels)  # TODO
        scores = delta.utils.getTrackingScores(labels, logits[:, :, :, 0], boxes=boxes)
        self.tmp_scores = copy.deepcopy(scores)  # TODO

        attributions = delta.utils.getAttributions(scores)
        self.tmp_attributions = copy.deepcopy(attributions)  # TODO
        previous_cell_nbs = (
            delta.utils.getcellsinframe(self.get_labels(frame - 1)[::-1, :])[::-1]
            if frame > self.first_frame
            else []
        )
        self.tmp_previous_cell_nbs = copy.deepcopy(previous_cell_nbs)  # TODO
        assert len(previous_cell_nbs) == attributions.shape[0]
        cell_nbs = [None] * attributions.shape[1]

        # Get poles:
        poles = delta.utils.getpoles(self.get_seg(frame), labels, scaling=self.scaling)
        self.tmp_poles = copy.deepcopy(poles)  # TODO

        # Resize labels if not using crop windows:
        if not self.config.crop_windows:
            resize = (
                self.box.xbr - self.box.xtl,
                self.box.ybr - self.box.ytl,
            )
            labels = cv2.resize(labels, resize, interpolation=cv2.INTER_NEAREST)
        self.tmp_labels_resized = copy.deepcopy(labels)  # TODO

        # Extract features for all cells in the ROI:
        extracted_features = delta.utils.roi_features(
            labels,
            fluo_frames=self.get_fluo(frame),
        )
        self.tmp_extracted_features = copy.deepcopy(extracted_features)  # TODO

        # Make sure the same cell_ids are present in both dicts
        assert poles.keys() == extracted_features.keys()

        # Assign poles to extracted features:
        for cellid, (old_pole, new_pole) in poles.items():
            extracted_features[cellid].old_pole = old_pole
            extracted_features[cellid].new_pole = new_pole

        # Go through old cells
        for cellid, attribs in zip(previous_cell_nbs, attributions):
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

    def init_track_rt(self) -> None:
        """
        """
        seg_mask = self.get_seg(self.first_frame)
        state_new, attributions_matrix = trackingrt.init_track_trench_rt(seg_mask=seg_mask)

        # Update tracking algorithm variables
        self.state_old = state_new
        if state_new:
            self.max_id = max(self.max_id, max([state["id"] for state in state_new]))

        # Dispatch tracking outputs
        self.process_tracking_outputs_rt(attributions=attributions_matrix, frame=self.first_frame)

    def track_rt(self) -> None:
        """
        """

        # Run through frames and compile inputs and references
        TIMER_ROI.start("track_rt", 0)
        seg_mask = self.get_seg(frame=self.first_frame+1)

        TIMER_ROI.start("track_rt:prepare", 1)
        tracking_inputs = trackingrt.get_tracking_inputs_rt(seg_mask)
        TIMER_ROI.stop("track_rt:prepare", 1)

        # Track
        TIMER_ROI.start("track_rt:predict", 1)
        if not self.state_old:
            state_new, attributions_matrix = trackingrt.init_track_trench_rt(seg_mask=seg_mask)
        else:
            state_new, attributions_matrix, image_processing_error = trackingrt.track_trench_rt(  # noqa
                self.state_old,
                tracking_inputs,
                self.max_id,
            )
            if image_processing_error.error_code.value:
                logging.warning(f"{self}: {image_processing_error}")
                self.error_container.add_error(new_error=image_processing_error)
        TIMER_ROI.stop("track_rt:predict", 1)

        # Update tracking algorithm variables
        self.state_old = state_new
        if state_new:
            self.max_id = max(self.max_id, max([state["id"] for state in state_new]))

        # Dispatch tracking outputs
        TIMER_ROI.start("track_rt:process", 1)
        self.process_tracking_outputs_rt(attributions=attributions_matrix, frame=self.first_frame+1)
        TIMER_ROI.stop("track_rt:process", 1)
        TIMER_ROI.stop("track_rt", 0)

    def process_tracking_outputs_rt(
        self,
        attributions: np.typing.NDArray[bool],
        frame: int,
    ) -> None:
        """
        Process output from tracking algorithm.

        Get poles, update lineage and create label_stack.

        Parameters
        ----------
        attributions: np.typing.NDArray[bool]
            Delta-style attributions matrix.
        frame: int
            Frame to process.

        Returns
        -------
        None.

        """
        if frame > 0:  # Changed
            self._frame_id += 1

        # Get scores and attributions:
        # Label frame but numbered 1, 2, 3, 4, etc. (temporary labels)
        labels = delta.utils.label_seg(self.get_seg(frame))

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

        self.label_stack[frame - self.first_frame] = labels

    def __str__(self):
        return f"ROI_{self.roi_nb:03}"
