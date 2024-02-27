import copy
import cv2
from enum import Enum
import glob
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import queue
import re
from serial import SerialException
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from PyQt5.QtCore import pyqtSignal, pyqtSlot, QEventLoop, QThread, QTimer, QObject, QRegExp, Qt
from PyQt5 import QtGui
from PyQt5.QtGui import QRegExpValidator, QDoubleValidator, QFont, QPalette, QColor, QValidator
from PyQt5.QtWidgets import (
    QWidget,
    QMainWindow, QApplication,
    QLabel, QLineEdit, QPushButton, QComboBox, QMessageBox,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QSizePolicy, QScrollArea, QFileDialog, QCheckBox
)

import delta.utils

from evomachine.acquisition import AbstractCamera, EvoCamera
from evomachine.automaton import Automaton, AutomatonQueueDataType
from evomachine.commands import AutomatonCommand, AutomatonCommandType
from evomachine.config import ConfigCRISP, ConfigFocus, get_logger
from evomachine.coordinates import Coordinate, CoordinateFactory
from evomachine.dmd import DMDControl
from evomachine.exceptions import ConfigError, TigerError
from evomachine.evotypes import LEDType, FocusAlgorithmType
from evomachine.guidir.figures import FigureMultiWindow, FigureWidget
from evomachine.guidir.guitemplates import EvoGUIThread, EvoPanelTemplate, EvoWorkerTemplate, QueueManager
from evomachine.guidir.guitypes import Direction, DMDModes, DisplayMode, SMALL, NORMAL, LEFT, AXES
from evomachine.guidir.position import PositionPanel

logger = get_logger(name=__name__)

class ExperimentWorker(EvoWorkerTemplate):
    def __init__(
            self,
            cam: AbstractCamera,
            automaton: Automaton,
    ):
        super().__init__()
        self.cam = cam
        self.automaton = automaton
        self.valid_coordinates = False
        self.factory: CoordinateFactory = CoordinateFactory(dfov=self.cam.get_delta_fov())
        self._field_of_views: Union[None, Dict[int, Coordinate]] = None
        self._cropping_boxes: Union[None, List[delta.utils.CroppingBox]] = None
        self.pause_time = 1

    @ staticmethod
    def make_delta_cropping_boxes(
            cropping_inds: Union[None, List[Tuple[Tuple[int, int], Tuple[int, int]]]],
    ) -> Union[None, List[delta.utils.CroppingBox]]:
        if cropping_inds is None or not cropping_inds:
            return None
        # cropping_indices = ((box0.xtl, box0.xbr), (box0.ytl, box0.ybr))
        return [
            delta.utils.CroppingBox(xtl=c[0][0], xbr=c[0][1], ytl=c[1][0], ybr=c[1][1])
            for c in cropping_inds
        ]

    def get_positions_from_dict(
            self,
            positions: Dict[int, Dict[str, Union[None, Dict[str, Union[float, int]]]]]
    ) -> List[Coordinate]:
        valid_all = [v for key, val in positions.items() for v in val.values()
                     if all([val1 is not None for val1 in val.values()])]
        valid_dict = [val for key, val in positions.items() if all([val1 is not None for val1 in val.values()])]
        if any(self.cam.coordinate_is_out_of_bounds(v) for v in valid_all):
            out = [v for v in valid_all if self.cam.coordinate_is_out_of_bounds(v)]
            self.valid_coordinates = False
            logger.warning(f"ThreadExperiment: Following coordinates are out of bounds: {out}")
            return None
        elif (not valid_all) or (len(valid_all) % 2 != 0):
            self.valid_coordinates = False
            logger.warning(f"ThreadExperiment: No valid coordinates: {valid_all}")
            return None
        else:
            self.valid_coordinates = True
            logger.info(f"Provided points: {valid_all}")
        coordinates = []
        for from_to in valid_dict:
            grid = self.factory.make_grid(
                start=Coordinate.from_dict(from_to["from"]),
                stop=Coordinate.from_dict(from_to["to"]),
            )
            coordinates.extend(grid)
        logger.info(f"Extracted {len(coordinates)} coordinates: {coordinates}")
        return coordinates

    def initialise_automaton_focus(self):
        try:
            self.automaton.initialise_fov_focus()
        except ConfigError as e:
            logger.warning(f"ExperimentWorker.initialise_automaton_focus: {e}")

    def initialise_automaton_image_processors(self):
        try:
            self.automaton.initialise_position_processor()
        except ConfigError as e:
            logger.warning(f"ExperimentWorker.initialise_automaton_image_processors: {e}")

    def initialise_automaton_references(self):
        try:
            self.automaton.initialise_reference_frames()
        except ConfigError as e:
            logger.warning(f"ExperimentWorker.initialise_automaton_image_processors: {e}")

    def initialise_automaton_field_of_views(
            self,
            read_in_positions: Dict[int, Dict[str, Union[None, Dict[str, Union[float, int]]]]],
            cropping_indices: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
    ):
        coordinates = self.get_positions_from_dict(read_in_positions)
        cropping_boxes_one_fov = ExperimentWorker.make_delta_cropping_boxes(cropping_indices)
        if not self.valid_coordinates or coordinates is None:
            logger.warning("No valid coordinates provided. Aborting.")
            return
        self._field_of_views = {fov_id: coord for fov_id, coord in enumerate(coordinates)}
        self._cropping_boxes = {fov_id: cropping_boxes_one_fov for fov_id in self._field_of_views.keys()} if\
            cropping_boxes_one_fov is not None else None
        self.automaton.initialise_field_of_view_list(
            field_of_views=self._field_of_views,
            cropping_boxes=self._cropping_boxes,
        )


class ExperimentPanel(EvoPanelTemplate):
    def __init__(
            self,
            cam: AbstractCamera,
            automaton: Automaton,
            queue_manager: QueueManager,
    ):
        super().__init__(cam=cam, automaton=automaton)

        queue_manager.register(func=self.read_focus_data, msg_type=AutomatonQueueDataType.FOCUS_DATA)
        queue_manager.register(func=self.read_processor_data, msg_type=AutomatonQueueDataType.PROCESSOR_INIT_DATA)

        self.worker = ExperimentWorker(cam=self.cam, automaton=self.automaton)
        self.workers.append(self.worker)
        thread = EvoGUIThread()
        self.worker.moveToThread(thread)
        thread.start()
        self.threads.append(thread)

        self.num_read_ins = 3
        self.read_in_buttons = {i: {
            "from": self.make_button(text="From", func=self.record_param, font=SMALL, which=(i, "from")),
            "to": self.make_button(text="To", func=self.record_param, font=SMALL, which=(i, "to"))
        } for i in range(self.num_read_ins)}
        self.read_in_display = {i: {
            "from": self.make_label(text=self.make_pos_str(None), font=SMALL),
            "to": self.make_label(text=self.make_pos_str(None), font=SMALL)
        } for i in range(self.num_read_ins)}
        self.read_in_positions = {i: {"from": None, "to": None} for i in range(self.num_read_ins)}
        self.read_in_label = {i: self.make_label(text=f"Path {i}", font=SMALL) for i in range(self.num_read_ins)}
        self.init_positions_button = self.make_button(text="Initialise Positions", func=self.init_positions, font=SMALL)
        self.init_focus_button = self.make_button(text="Initialise Focus", func=self.init_focus, font=SMALL)
        self.init_references_button = self.make_button(text="Take Refs", func=self.init_references, font=SMALL)
        self.init_processors_button = self.make_button(text="Initialise IP", func=self.init_processors, font=SMALL)
        self.read_in_clear_button = self.make_button(text="Clear Paths", func=self.clear_param, font=SMALL)
        self.focus_curves_button = self.make_button(text="Focus Curves", func=self.show_focus_curves, font=SMALL)
        self.start_button = self.make_button(text="Start", func=self.start_acquisition, font=SMALL)
        self.stop_button = self.make_button(text="Stop", func=self.stop_acquisition, font=SMALL)

        self.layout = QGridLayout()
        self.layout.addWidget(self.make_label(text="Experiment", font=NORMAL), 0, 0, 1, 3, LEFT)
        self.layout.addWidget(self.read_in_clear_button, 1, 0, 1, 1)
        self.layout.addWidget(self.init_positions_button, 2, 0, 1, 1)
        self.layout.addWidget(self.init_focus_button, 3, 0, 1, 1)
        self.layout.addWidget(self.init_references_button, 4, 0, 1, 1)
        self.layout.addWidget(self.init_processors_button, 5, 0, 1, 1)
        self.layout.addWidget(self.focus_curves_button, 6, 0, 1, 1)
        for i in range(self.num_read_ins):
            self.layout.addWidget(self.read_in_label[i], 2*i+1, 1, 1, 1)
            self.layout.addWidget(self.read_in_buttons[i]["from"], 2*i+1, 2, 1, 1)
            self.layout.addWidget(self.read_in_buttons[i]["to"], 2*i+1, 3, 1, 1)
            self.layout.addWidget(self.read_in_display[i]["from"], 2*i+2, 2, 1, 1)
            self.layout.addWidget(self.read_in_display[i]["to"], 2*i+2, 3, 1, 1)
        self.layout.addWidget(self.start_button, min(7, 1+2*self.num_read_ins), 0, 1, 1)
        self.layout.addWidget(self.stop_button, min(7, 1+2*self.num_read_ins), 1, 1, 1)
        self.widget = QWidget()
        self.widget.setLayout(self.layout)

        self.focus_data: Union[None, Tuple[np.typing.Array, np.typing.Array, np.typing.Array, np.typing.Array]] = None
        self.focus_curves: Union[None, np.typing.Array] = None
        self.focus_prev_curr_stack: Union[None, Tuple[np.typing.Array, np.typing.Array]] = None
        self.focus_curves_window: Union[None, FigureMultiWindow] = None

        self.processor_data: Dict[int, Tuple[float, List[delta.utils.CroppingBox]]] = {}

    def record_param(self, which: Tuple[int, str]):
        i, from_to = which
        try:
            pos_dict = self.cam.get_coordinates(AXES)
            pos_str = f"{self.make_pos_str(pos_dict['X'])[:-2]}, " \
                      f"{self.make_pos_str(pos_dict['Y'])[:-2]}, " \
                      f"{self.make_pos_str(pos_dict['Z'])[:-2]}"
            self.read_in_display[i][from_to].setText(pos_str)
            self.read_in_positions[i][from_to] = pos_dict
        except (SerialException, KeyError) as e:
            logger.warning(f"EvoGUI.exp_record_param: {e}")
            self.read_in_display[i][from_to].setText("?")
            self.read_in_positions[i][from_to] = None

    def clear_param(self):
        self.read_in_positions = {i: {"from": None, "to": None} for i in range(self.num_read_ins)}
        self.worker.set_labels(labels=self.read_in_display, text="?")

    def init_focus(self):
        self.worker.initialise_automaton_focus()

    def init_processors(self):
        self.worker.initialise_automaton_image_processors()

    def init_references(self):
        self.worker.initialise_automaton_references()

    def init_positions(self):
        tmp = None  # self.pic_get_cropping_boxes()  # TODO this needs to come from somewhere
        if FigureWidget.cropping_boxes_are_valid(tmp):
            cropping_indices = FigureWidget.get_cropping_indices(tmp)
        else:
            cropping_indices = None
        self.worker.initialise_automaton_field_of_views(
            read_in_positions=self.read_in_positions,
            cropping_indices=cropping_indices,
        )

    def start_acquisition(self):
        if not self.automaton.is_initialised():
            self.automaton.initialise()
        if not self.automaton.is_alive():
            self.automaton.start()
        else:
            self.automaton.restart()
        self.start_button.setStyleSheet("background-color: green;")

    def stop_acquisition(self):
        if not self.automaton.stopped():
            self.automaton.stop()
        self.start_button.setStyleSheet("background-color: white;")

    def read_focus_data(self, data):
        self.focus_curves_button.setStyleSheet("background-color: green;")
        self.focus_data = data

    def read_processor_data(self, data):
        self.processor_data[data[0]] = (data[1], data[2])

    def show_focus_curves(self):
        if self.focus_data is None:
            logger.error("exp_show_curve: missing data. Returning.")
            return
        focus_curves, focus_prev_stack, focus_stack, focus_prev_z_coords = self.focus_data

        all_figs = {i: None for i in focus_curves.keys()}
        for i in all_figs.keys():
            fig = plt.figure()
            gs = fig.add_gridspec(2, 2, width_ratios=[1, 1])
            ax1 = fig.add_subplot(gs[0, :])
            z_coords = np.array(list(focus_curves[i][0])) / 10
            ax1.plot(z_coords, focus_curves[i][1], marker='x')
            ax1.set_xticks(z_coords.tolist())
            ax1.set_xticklabels([f'{x:.1f}' for x in z_coords.tolist()])
            ax1.set_xlabel("Z position [um]")
            ax1.set_ylabel("Sharpness Scores")
            ax1.set_title(f"Focus Curve at pos_id = {i}")
            best_index = np.argmax(focus_curves[i][1])
            ax1.plot(z_coords[best_index], focus_curves[i][1][best_index], marker='o')
            ax1.grid(True)
            best_image = self.cam.normalise_frame(focus_stack[:, :, i])
            prev_image = self.cam.normalise_frame(focus_prev_stack[:, :, i])
            ax2 = fig.add_subplot(gs[1, 0])
            ax2.imshow(prev_image)
            ax2.set_title(f"Before Focus\n Z={focus_prev_z_coords[i]/10:.1f}")
            ax3 = fig.add_subplot(gs[1, 1])
            ax3.imshow(best_image)
            ax3.set_title(f"After Focus\n Z={z_coords[best_index]:.1f}")
            fig.tight_layout()
            all_figs[i] = fig
        self.focus_curves_window = FigureMultiWindow(all_figs)
        self.focus_curves_window.show()
