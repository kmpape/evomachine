import copy
import cv2

import matplotlib.pyplot as plt
from multiprocessing import Event
import numpy as np
from serial import SerialException
from typing import Any
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt
from PyQt5.QtWidgets import QWidget, QPushButton, QDialog, QComboBox, QLabel, QGridLayout, QCheckBox

import delta.utils
from delta.utils import CroppingBox as DeltaCroppingBox

from evomachine.acquisition import AbstractCamera
from evomachine.commands import AutomatonCommand, AutomatonCommandType
from evomachine.config import ConfigCamera, ConfigImageProcessor, get_logger
from evomachine.coordinates import Coordinate, CoordinateFactory
from evomachine.guidir.figures import FigureMultiWindow, ChannelPlotter
from evomachine.guidir.guitemplates import EvoGUIThread, EvoPanelTemplate, EvoWorkerTemplate
from evomachine.guidir.guitypes import SMALL, NORMAL, LEFT, AXES, CENTER, RIGHT, VERYSMALL
from evomachine.guidir.queuemanager import QueueManager
from evomachine.utils import EvoCroppingBox

logger = get_logger(name=__name__, is_gui=True)


class ExperimentWorker(EvoWorkerTemplate):
    def __init__(
            self,
            queue_manager: QueueManager,
            config_camera: ConfigCamera,
            signal_enable_button: pyqtSignal,
            signal_disable_button: pyqtSignal,
            signal_set_button_color: pyqtSignal,
    ):
        super().__init__()
        self.queue_manager = queue_manager
        self.cfg_camera = config_camera
        self.signal_enable_button = signal_enable_button
        self.signal_disable_button = signal_disable_button
        self.signal_set_button_color = signal_set_button_color

        self.valid_coordinates = False
        self.factory: CoordinateFactory = CoordinateFactory(dfov=self.cfg_camera.fov_size * 10)
        self._field_of_views: None | dict[int, Coordinate] = None
        self._cropping_boxes: None | list[delta.utils.CroppingBox] = None
        self._stage_limits: None | dict[str, tuple[float, float]] = None
        self.queue_manager.request(
            req_str='self.cam.get_stage_limits',
            kwargs_dict={},
            callback=self.update_limits,
        )
        self.pause_time = 1

    @ staticmethod
    def make_delta_cropping_boxes(
            cropping_inds: None | list[tuple[tuple[int, int], tuple[int, int]]],
    ) -> None | list[delta.utils.CroppingBox]:
        if cropping_inds is None or not cropping_inds:
            return None
        # cropping_indices = ((box0.xtl, box0.xbr), (box0.ytl, box0.ybr))
        return [
            delta.utils.CroppingBox(xtl=c[0][0], xbr=c[0][1], ytl=c[1][0], ybr=c[1][1])
            for c in cropping_inds
        ]

    def coordinate_is_out_of_bounds(self, coordinates: dict[str, float]) -> bool:
        return False if self._stage_limits is None else any((key not in self._stage_limits) or
                                                            (val < self._stage_limits[key][0]) or
                                                            (val > self._stage_limits[key][1])
                                                            for key, val in coordinates.items())

    def update_limits(self, data: tuple[Coordinate, Coordinate] | Exception):
        if isinstance(data, Exception):
            logger.error("ExperimentWorker.update_limits: received exception. Returning.")
            return
        logger.info(f"Stage limits: {data}")
        self._stage_limits = {'X': (data[0].x, data[1].x), 'Y': (data[0].y, data[1].y), 'Z': (data[0].z, data[1].z)}

    def get_positions_from_dict(
            self,
            positions: dict[int, dict[str, None | dict[str, float | int]]]
    ) -> list[Coordinate] | None:
        logger.info(f"get_positions_from_dict DEBUG: Recorded positions = {positions}")
        valid_all = [v for key, val in positions.items() for v in val.values()
                     if all([val1 is not None for val1 in val.values()])]
        valid_dict = [val for key, val in positions.items() if all([val1 is not None for val1 in val.values()])]
        if any(self.coordinate_is_out_of_bounds(v) for v in valid_all):
            out = [v for v in valid_all if self.coordinate_is_out_of_bounds(v)]
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
        for i_channel, from_to in enumerate(valid_dict):
            from_to["from"]["channel_id"] = i_channel
            from_to["to"]["channel_id"] = i_channel
            start_coord = Coordinate.from_dict(from_to["from"])
            stop_coord = Coordinate.from_dict(from_to["to"])
            grid = self.factory.make_grid(
                start=start_coord,
                stop=stop_coord,
            )
            coordinates.extend(grid)
        logger.info(f"Extracted {len(coordinates)} coordinates: {coordinates}")
        return coordinates

    def initialise_automaton_focus(self, data: Any = None, is_init_all: bool = False, use_autofocus: bool = False):
        if isinstance(data, Exception):
            logger.error(f"initialise_automaton_focus: Exception encountered. Aborting.")
            self.enable_disable_next_buttons(None, [0, 8], [1, 2, 3, 4, 5, 6, 7])
            return
        if not is_init_all:
            self.queue_manager.request(
                req_str='self.initialise_fov_focus',
                kwargs_dict={'use_autofocus': use_autofocus},
                callback=self.enable_disable_next_buttons,
                callback_args=([2, 6], [3, 4, 5, 7],),
            )
        else:
            self.queue_manager.request(
                req_str='self.initialise_fov_focus',
                kwargs_dict={'use_autofocus': use_autofocus},
                callback=self.initialise_automaton_references,
                callback_args=(True,),
            )

    def initialise_automaton_image_processors(self, data: Any = None, is_init_all: bool = False):
        if isinstance(data, Exception):
            logger.error("ExperimentWorker.initialise_automaton_image_processors: received exception. Returning.")
            self.enable_disable_next_buttons(None, [0, 1, 2, 8], [3, 4, 5])
            return
        logger.debug("ExperimentWorker.initialise_automaton_image_processors: Requesting initialisation of IP.")
        if not is_init_all:
            self.queue_manager.request(
                req_str='self.initialise_position_processor',
                kwargs_dict={},
                callback=self.enable_disable_next_buttons,
                callback_args=([4, 7], [],),
            )
        else:
            self.queue_manager.request(
                req_str='self.initialise_position_processor',
                kwargs_dict={'rotation': 0},  # TODO remove me
                callback=self.enable_after_init_all,
            )

    def initialise_automaton_references(self, data: Any = None, is_init_all: bool = False):
        if isinstance(data, Exception):
            logger.error("ExperimentWorker.initialise_automaton_references: received exception. Returning.")
            self.enable_disable_next_buttons(None, [0, 1], [2, 3, 4, 5])
            return
        logger.debug("ExperimentWorker.initialise_automaton_references: Requesting initialisation of references.")
        if not is_init_all:
            self.queue_manager.request(
                req_str='self.initialise_reference_frames',
                kwargs_dict={},
                callback=self.enable_disable_next_buttons,
                callback_args=([3], [4, 5, 7],),
            )
        else:
            self.queue_manager.request(
                req_str='self.initialise_reference_frames',
                kwargs_dict={},
                callback=self.initialise_automaton_image_processors,
                callback_args=(True,),
            )

    def initialise_automaton_field_of_views(
            self,
            read_in_positions: dict[int, dict[str, None | dict[str, float | int]]],
            cropping_boxes: list[EvoCroppingBox] | None = None,
            is_init_all: bool = False,
            use_autofocus: bool = False,
    ):
        logger.debug(f"ExperimentWorker.initialise_automaton_field_of_views: {read_in_positions}.")
        coordinates = self.get_positions_from_dict(read_in_positions)
        # cropping_boxes_one_fov = ExperimentWorker.make_delta_cropping_boxes(cropping_indices)
        if not self.valid_coordinates or coordinates is None:
            logger.warning("No valid coordinates provided. Aborting.")
            return
        if not all(isinstance(b, EvoCroppingBox) for b in cropping_boxes):
            raise TypeError(f"Expected list of EvoCroppingBox, got {cropping_boxes} instead. Aborting.")
        self._field_of_views = {fov_id: coord for fov_id, coord in enumerate(coordinates)}
        self._cropping_boxes = {fov_id: cropping_boxes for fov_id in self._field_of_views.keys()} if \
            (cropping_boxes is not None and len(cropping_boxes) > 0) else None
        if not is_init_all:
            self.queue_manager.request(
                req_str='self.initialise_field_of_view_list',
                kwargs_dict={
                    'field_of_views': self._field_of_views,
                    'cropping_boxes': self._cropping_boxes,
                    'use_autofocus': use_autofocus,
                },
                callback=self.enable_disable_next_buttons,
                callback_args=([1], [2, 3, 4, 5, 6, 7, 9],),
            )
        else:
            self.queue_manager.request(
                req_str='self.initialise_field_of_view_list',
                kwargs_dict={
                    'field_of_views': self._field_of_views,
                    'cropping_boxes': self._cropping_boxes,
                    'use_autofocus': use_autofocus,
                },
                callback=self.initialise_automaton_focus,
                callback_args=(True, use_autofocus),
            )

    def initialise_all(
            self,
            read_in_positions: dict[int, dict[str, None | dict[str, float | int]]],
            cropping_boxes: list[EvoCroppingBox] | None = None,
            use_autofocus: bool = False,
    ):
        self.initialise_automaton_field_of_views(
            read_in_positions=read_in_positions,
            cropping_boxes=cropping_boxes,
            is_init_all=True,
            use_autofocus=use_autofocus,
        )

    def enable_after_init_all(self, data: Any):
        if isinstance(data, Exception):
            logger.error("ExperimentWorker.enable_after_init_all: received exception. Returning.")
            return
        self.queue_manager.request(
            req_str='self.save_state',
            kwargs_dict={'filename_suffix': 'GUI'},
        )
        self.signal_enable_button.emit([4, 6, 7])
        self.signal_set_button_color.emit([0, 1, 2, 3], "green")
        self.signal_set_button_color.emit([8], "lightgray")

    def enable_disable_next_buttons(self, data: Any, enable: list[int], disable: list[int]):
        if isinstance(data, Exception):
            logger.error("ExperimentWorker.enable_disable_next_buttons: received exception. Returning.")
            return
        self.signal_enable_button.emit(enable)
        self.signal_disable_button.emit(disable)


class ButtonWorker(EvoWorkerTemplate):
    signal_update_strategy_label = pyqtSignal(str)  # noqa

    def __init__(
            self,
            queue_manager: QueueManager,
            button_list: list[QPushButton | QCheckBox],
            strategy_label: QLabel
    ):
        super().__init__()
        self.queue_manager = queue_manager
        self.button_list = button_list
        self.strategy_label = strategy_label
        self.signal_update_strategy_label.connect(self._update_strategy_label)

    @pyqtSlot(list, str)  # noqa
    def set_color(self, button_indices: list[int], color_str: str):
        for i in button_indices:
            if i >= len(self.button_list):
                logger.error(f"ButtonWorker: Cannot set color for button {i}.")
            else:
                self.button_list[i].setStyleSheet(f"background-color: {color_str};")

    @pyqtSlot(list)  # noqa
    def disable_button(self, indices: list[int]):
        for i in indices:
            if i >= len(self.button_list):
                logger.error(f"ButtonWorker: Cannot disable button {i}.")
            else:
                self.button_list[i].setEnabled(False)

    @pyqtSlot(list)  # noqa
    def enable_button(self, indices: list[int]):
        for i in indices:
            if i >= len(self.button_list):
                logger.error(f"ButtonWorker: Cannot enable button {i}.")
            else:
                self.button_list[i].setEnabled(True)

    @pyqtSlot()  # noqa
    def update_strategy_label(self):
        self.queue_manager.request(
            req_str="self.get_strategy_name",
            kwargs_dict={},
            callback=self.update_strategy_label_callback,
        )

    def update_strategy_label_callback(self, data: Any):
        if isinstance(data, Exception):
            logger.error("ExperimentWorker.update_strategy_label_callback: received exception. Returning.")
            return
        self.signal_update_strategy_label.emit(data)

    @pyqtSlot(str)  # noqa
    def _update_strategy_label(self, data: str):
        self.strategy_label.setText(data)


class ExperimentPanel(EvoPanelTemplate):
    signal_set_button_color = pyqtSignal(list, str)  # noqa
    signal_enable_button = pyqtSignal(list)  # noqa
    signal_disable_button = pyqtSignal(list)  # noqa
    signal_update_strategy = pyqtSignal()  # noqa
    signal_set_labels = pyqtSignal(dict, str)  # noqa

    def __init__(
            self,
            queue_manager: QueueManager,
            camera_config: ConfigCamera,
            processor_config: ConfigImageProcessor,
            start_strategy_event: Event,
            stop_strategy_event: Event,
            stop_event: Event,
            shutdown_event: Event,
    ):
        super().__init__(
            queue_manager=queue_manager,
            camera_config=camera_config,
            processor_config=processor_config,
            start_strategy_event=start_strategy_event,
            stop_strategy_event=stop_strategy_event,
            stop_event=stop_event,
            shutdown_event=shutdown_event,
        )

        queue_manager.register(func=self.read_focus_data, msg_type=AutomatonCommandType.FOCUS_DATA)
        queue_manager.register(func=self.read_roi_data, msg_type=AutomatonCommandType.ROI_DATA)
        queue_manager.register(func=self.read_fov_data, msg_type=AutomatonCommandType.FOV_DATA)
        queue_manager.register(func=self.read_ref_data, msg_type=AutomatonCommandType.REF_DATA)

        self.worker = ExperimentWorker(
            queue_manager=queue_manager,
            config_camera=camera_config,
            signal_enable_button=self.signal_enable_button,
            signal_disable_button=self.signal_disable_button,
            signal_set_button_color=self.signal_set_button_color,
        )
        self.signal_set_labels.connect(self.worker.set_labels)
        self.workers.append(self.worker)
        thread = EvoGUIThread()
        self.worker.moveToThread(thread)
        thread.start()
        self.threads.append(thread)

        self.num_read_ins = 6
        self.read_in_buttons = {i: {
            "from": self.make_button(text="From", func=self.record_param, font=VERYSMALL, which=(i, "from")),
            "to": self.make_button(text="To", func=self.record_param, font=VERYSMALL, which=(i, "to"))
        } for i in range(self.num_read_ins)}
        self.read_in_display = {i: {
            "from": self.make_label(text="X=??????\nY=??????\nZ=??????", font=SMALL, align=Qt.AlignLeft),
            "to": self.make_label(text="X=??????\nY=??????\nZ=??????", font=SMALL, align=Qt.AlignLeft)
        } for i in range(self.num_read_ins)}
        self.read_in_positions = {i: {"from": None, "to": None} for i in range(self.num_read_ins)}
        self.read_in_label = {i: self.make_label(text=f"Path {i}", font=SMALL) for i in range(self.num_read_ins)}
        self.init_all_button = self.make_button(text="Initialise all", func=self.init_all, font=VERYSMALL)
        self.init_positions_button = self.make_button(text="Initialise FoVs", func=self.init_positions, font=VERYSMALL)
        self.init_focus_button = self.make_button(text="Initialise Focus", func=self.init_focus, font=VERYSMALL)
        self.init_references_button = self.make_button(text="Take Refs", func=self.init_references, font=VERYSMALL)
        self.init_processors_button = self.make_button(text="Initialise IP", func=self.init_processors, font=VERYSMALL)
        self.read_in_clear_button = self.make_button(text="Clear Paths", func=self.clear_param, font=VERYSMALL)
        self.focus_curves_button = self.make_button(text="Focus Curves", func=self.show_focus_curves, font=VERYSMALL)
        self.use_autofocus: bool = True
        self.autofocus_checkbox = self.make_checkbox(text="Use autofocus", font=SMALL, set_true=self.use_autofocus,
                                                     func=self.toggle_autofocus)
        self.position_dialog_button = self.make_button(text="FoVs", func=self.show_position_dialog, font=VERYSMALL)
        self.strategy_label = self.make_label(text="???", font=VERYSMALL)
        self._automaton_is_initialised: bool = False
        self.start_button = self.make_button(text="Start", func=self.start_acquisition, font=VERYSMALL)
        #  self.pause_button = self.make_button(text="Pause", func=self.pause_acquisition, font=SMALL)
        self.stop_button = self.make_button(text="Stop", func=self.stop_acquisition, font=VERYSMALL)
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

        self.layout.addWidget(self.make_label(text="Strategy", font=NORMAL), 0, 0, 1, 1, LEFT)
        self.layout.addWidget(self.strategy_label, 0, 1, 1, 2, CENTER)
        self.layout.addWidget(self.read_in_clear_button, 1, 0, 1, 1)
        self.layout.addWidget(self.init_positions_button, 2, 0, 1, 1)
        self.layout.addWidget(self.init_focus_button, 3, 0, 1, 1)
        self.layout.addWidget(self.autofocus_checkbox, 3, 1, 1, 1)
        self.layout.addWidget(self.init_references_button, 4, 0, 1, 1)
        self.layout.addWidget(self.init_processors_button, 5, 0, 1, 1)
        self.layout.addWidget(self.init_all_button, 6, 0, 1, 1)
        for i in range(self.num_read_ins):
            # self.layout.addWidget(self.read_in_label[i], 2*i+1, 2, 1, 1)
            # _shift = 1
            # self.layout.addWidget(self.read_in_buttons[i]["from"], 2*i+1, 3+_shift, 1, 1)
            # self.layout.addWidget(self.read_in_buttons[i]["to"], 2*i+1, 4+_shift, 1, 1)
            # self.layout.addWidget(self.read_in_display[i]["from"], 2*i+2, 3+_shift, 1, 1)
            # self.layout.addWidget(self.read_in_display[i]["to"], 2*i+2, 4+_shift, 1, 1)
            _shift = 1
            self.layout.addWidget(self.read_in_buttons[i]["from"], 2*i+1, 3+_shift, 1, 1)
            self.layout.addWidget(self.read_in_buttons[i]["to"], 2*i+1, 5+_shift, 1, 1)
            self.layout.addWidget(self.read_in_display[i]["from"], 2*i+1, 4+_shift, 1, 1)
            self.layout.addWidget(self.read_in_display[i]["to"], 2*i+1, 6+_shift, 1, 1)

        this_row = min(7, 1+2*self.num_read_ins)
        self.layout.addWidget(self.start_button, this_row, 0, 1, 1)
        self.layout.addWidget(self.stop_button, this_row, 1, 1, 1)
        self.layout.addWidget(self.focus_curves_button, this_row, 2, 1, 1)
        self.layout.addWidget(self.position_dialog_button, this_row, 3, 1, 1)
        self.widget = QWidget()  # noqa
        self.widget.setLayout(self.layout)

        # FIXME any changes of <buttons> will bug below
        buttons = [self.init_positions_button,      # 0
                   self.init_focus_button,          # 1
                   self.init_references_button,     # 2
                   self.init_processors_button,     # 3
                   self.start_button,               # 4
                   self.stop_button,                # 5
                   self.focus_curves_button,        # 6
                   self.position_dialog_button,     # 7
                   self.init_all_button,            # 8
                   self.autofocus_checkbox]         # 9
        self.worker_buttons = ButtonWorker(
            queue_manager=queue_manager,
            button_list=buttons,
            strategy_label=self.strategy_label
        )
        self.signal_set_button_color.connect(self.worker_buttons.set_color)
        self.signal_disable_button.connect(self.worker_buttons.disable_button)
        self.signal_enable_button.connect(self.worker_buttons.enable_button)
        self.signal_update_strategy.connect(self.worker_buttons.update_strategy_label)
        self.workers.append(self.worker_buttons)
        thread = EvoGUIThread()
        self.worker_buttons.moveToThread(thread)
        thread.start()
        self.threads.append(thread)

        self.signal_disable_button.emit([0, 1, 2, 3, 4, 5, 6, 7, 8])
        self.signal_update_strategy.emit()

        self.cropping_boxes: dict[int, None | EvoCroppingBox] = {0: None, 1: None}

        self.fov_data: dict[str, Any] = {}
        self.ref_data: dict[int, np.ndarray] = {}
        self.roi_data: dict[int, dict[str, Any]] = {}
        self.focus_data: dict[str, Any] = {}

        self.pos_dialog: PositionDialog | None = None

    def record_param(self, which: tuple[int, str]):
        logger.debug(f"ExperimentPanel.record_param: Recording {which}.")
        self.queue_manager.request(
            req_str='self.cam.get_coordinates',
            kwargs_dict={'axes': AXES},
            callback=self._record_param,
            callback_args=(which,),
        )

    def _record_param(self, data, which: tuple[int, str]):
        if isinstance(data, Exception):
            logger.error("ExperimentWorker._record_param: received exception. Returning.")
            return
        logger.debug("ExperimentPanel._record_param: Recording coordinates.")
        i, from_to = which
        try:
            pos_dict = data
            pos_str = "\n".join([f"X={self.make_pos_str(pos_dict['X'])[:-2]}",
                                 f"Y={self.make_pos_str(pos_dict['Y'])[:-2]}",
                                 f"Z={self.make_pos_str(pos_dict['Z'])[:-2]}"])
            self.read_in_display[i][from_to].setText(pos_str)
            self.read_in_positions[i][from_to] = pos_dict
        except (SerialException, KeyError) as e:
            logger.warning(f"EvoGUI.exp_record_param: {e}")
            self.read_in_display[i][from_to].setText("X=??????\nY=??????\nZ=??????")
            self.read_in_positions[i][from_to] = None

        if self.read_in_positions[i]["from"] is not None and self.read_in_positions[i]["to"] is not None:
            self.signal_enable_button.emit([0, 8])

    def clear_param(self):
        self.read_in_positions = {i: {"from": None, "to": None} for i in range(self.num_read_ins)}
        self.signal_set_labels.emit(self.read_in_display, "X=??????\nY=??????\nZ=??????")
        if not any(r["from"] is not None and r["to"] is not None for r in self.read_in_positions.values()):
            self.signal_disable_button.emit([0, 8])

    def init_focus(self):
        self._automaton_is_initialised = False
        self.init_focus_button.setStyleSheet("background-color: orange;")
        self.init_references_button.setStyleSheet("background-color: lightgray;")
        self.init_processors_button.setStyleSheet("background-color: lightgray;")
        self.signal_disable_button.emit([9])
        self.worker.initialise_automaton_focus(use_autofocus=self.use_autofocus)
        self.signal_set_button_color.emit([1], "orange")
        self.signal_set_button_color.emit([2, 3], "lightgray")

    def init_processors(self):
        self.signal_set_button_color.emit([3], "orange")
        self.worker.initialise_automaton_image_processors()
        self._automaton_is_initialised = True

    def init_references(self):
        self._automaton_is_initialised = False
        self.signal_set_button_color.emit([2], "orange")
        self.signal_set_button_color.emit([3], "lightgray")
        self.worker.initialise_automaton_references()

    def init_all(self):
        self._automaton_is_initialised = True
        self.signal_set_button_color.emit([0, 1, 2, 3, 8], "orange")
        self.signal_disable_button.emit([9])
        self.worker.initialise_all(
            read_in_positions=self.read_in_positions,
            cropping_boxes=[b for b in self.cropping_boxes.values() if b is not None],
            use_autofocus=self.use_autofocus,
        )

    def init_positions(self):
        self._automaton_is_initialised = False
        self.signal_set_button_color.emit([0], "orange")
        self.signal_disable_button.emit([9])
        self.worker.initialise_automaton_field_of_views(
            read_in_positions=self.read_in_positions,
            cropping_boxes=[b for b in self.cropping_boxes.values() if b is not None],
            use_autofocus=self.use_autofocus,
        )

    def start_acquisition(self):
        if not self._automaton_is_initialised:
            self.queue_manager.request(
                req_str='self.initialise',
                kwargs_dict={'axes': AXES},
                callback=self._start_acquisition,
            )
            self._automaton_is_initialised = True
        else:
            self._start_acquisition(None)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def _start_acquisition(self, data):
        if isinstance(data, Exception):
            logger.error("ExperimentWorker._start_acquisition: received exception. Returning.")
            self._automaton_is_initialised = False
            return
        logger.info("ExperimentPanel._start_acquisition: Starting acquisition.")
        self.stop_strategy_event.clear()
        self.stop_event.clear()
        self.start_strategy_event.set()
        self.start_button.setStyleSheet("background-color: green;")

    def pause_acquisition(self):
        # TODO
        logger.info("ExperimentPanel.pause_acquisition: Pausing acquisition.")
        self.stop_event.set()

    def stop_acquisition(self):
        logger.info("ExperimentPanel.stop_acquisition: Ending acquisition.")
        self.start_strategy_event.clear()
        self.stop_strategy_event.set()
        self.start_button.setStyleSheet("background-color: green;")
        self.start_button.setEnabled(True)
        # self.stop_button.setEnabled(False)

    def read_focus_data(self, data: AutomatonCommand):
        if not self.use_autofocus:
            self.signal_set_button_color.emit([1, 6], "green")
        self.focus_data = data.command_args
        logger.debug("Received focus data.")

    def read_fov_data(self, data: AutomatonCommand):
        self.fov_data = data.command_args
        self.roi_data = {fov_id: {} for fov_id in data.command_args['fovs']}
        self.signal_set_button_color.emit([0], "green")
        self.signal_set_button_color.emit([1, 2, 3], "lightgray")

    def read_ref_data(self, data: AutomatonCommand):
        self.ref_data = data.command_args
        self.signal_set_button_color.emit([2], "green")

    def read_roi_data(self, data: AutomatonCommand):
        if not data.command_args['fov_id'] in self.roi_data:
            logger.warning(f"read_roi_data: fov_id {data.command_args['fov_id']} not in {self.roi_data.keys()}")
        self.signal_set_button_color.emit([3], "green")
        self.roi_data[data.command_args['fov_id']] = {
            'rotation': data.command_args['rotation'],
            'roi_boxes': data.command_args['roi_boxes'],
        }

    def show_focus_curves(self):
        if self.use_autofocus:
            logger.error("show_focus_curves: using autofocus. Returning.")
            return
        if self.focus_data is None:
            logger.error("show_focus_curves: missing data. Returning.")
            return
        focus_curves = self.focus_data['focus_curves']
        focus_prev_stack = self.focus_data['focus_prev_stack']
        focus_stack = self.focus_data['focus_stack']
        focus_prev_z_coords = self.focus_data['focus_prev_z_coords']

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
            best_image = AbstractCamera.normalise_frame(focus_stack[:, :, i])
            prev_image = AbstractCamera.normalise_frame(focus_prev_stack[:, :, i])
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

    def show_position_dialog(self):
        if any(x is None for x in [self.fov_data, self.ref_data, self.focus_data, self.roi_data]):
            tmp = list(x is None for x in [self.fov_data, self.ref_data, self.focus_data, self.roi_data])
            logger.warning(f"Some data for PositionDialog is None: {tmp}.")
        self.pos_dialog = PositionDialog(
            queue_manager=self.queue_manager,
            fov_data=self.fov_data,
            ref_data=self.ref_data,
            focus_data=self.focus_data,
            roi_data=self.roi_data,
            processor_config=self.processor_config,

        )
        self.pos_dialog.open()

    def toggle_autofocus(self, state):
        if state == Qt.Checked:
            self.use_autofocus = True
        else:
            self.use_autofocus = False

    @pyqtSlot(int, int, EvoCroppingBox)  # noqa
    def update_cropping_boxes(self, fov_id: int, box_id: int, cropping_box: EvoCroppingBox | None):
        logger.debug(f"ExperimentPanel.update_cropping_boxes: fov_id={fov_id}, box_id={box_id}, cropping_box={cropping_box}")
        self.cropping_boxes[box_id] = None if cropping_box.is_none else cropping_box


class PositionDialogWorker(EvoWorkerTemplate):
    def __init__(
            self,
            curr_fov_id: int,
            fov_coordinates: dict[int, Coordinate],
            actives: dict[int, dict[str, Any]],
            overrides: dict[int, dict[str, Any]],
            info_labels: dict[str, QLabel],
            value_labels: dict[str, QLabel],
            override_labels: dict[str, QLabel],
            edits: dict[str, QLabel],
            buttons: dict[str, QLabel],
            combo_box: QComboBox,
    ):
        super().__init__()
        self.fov_id = curr_fov_id
        self.fov_coordinates = fov_coordinates
        self.actives = actives
        self.overrides = overrides
        self.info_labels = info_labels
        self.value_labels = value_labels
        self.override_labels = override_labels
        self.edits = edits
        self.buttons = buttons
        self.combo_box = combo_box

    @staticmethod
    def _format(val: float | int) -> str:
        return f"{val:.2f}" if isinstance(val, float) else str(val)

    @pyqtSlot(int)  # noqa
    def update_fov_id(self, fov_id: int):
        self.fov_id = fov_id
        self.update_display()

    @pyqtSlot(bool)  # noqa
    def update_buttons(self, val: bool):
        for k in self.value_labels.keys():
            self.buttons[k].setEnabled(val)
        self.combo_box.setEnabled(val)

    @pyqtSlot()  # noqa
    def update_display(self):
        self.info_labels["fov"].setText(f"FoV {self.fov_id} at {str(self.fov_coordinates[self.fov_id])}")
        for k in self.value_labels.keys():
            self.value_labels[k].setText(f"{self._format(self.actives[self.fov_id][k])}")
            self.override_labels[k].setText("None" if self.overrides[self.fov_id][k] is None
                                            else f"{self._format(self.overrides[self.fov_id][k])}")
            self.edits[k].setText("None")
            self.buttons[k].setEnabled(True)
        self.combo_box.setEnabled(True)

    @pyqtSlot(str, float)  # noqa
    def update_override(self, param: str, new_val: float):
        self.overrides[self.fov_id][param] = new_val
        self.override_labels[param].setText(f"{self._format(new_val)}")

    @pyqtSlot(str, float)  # noqa
    def update_active(self, param: str, new_val: float):
        self.actives[self.fov_id][param] = new_val
        self.value_labels[param].setText(f"{self._format(new_val)}")
        self.edits[param].setText("None")
        self.overrides[self.fov_id][param] = None
        self.override_labels[param].setText(f"None")
        if param == 'z_pos':
            self.fov_coordinates[self.fov_id].z = new_val
            self.info_labels["fov"].setText(f"FoV {self.fov_id} at {str(self.fov_coordinates[self.fov_id])}")


class PositionDialog(QDialog):
    signal_update_override = pyqtSignal(str, float)  # noqa
    signal_update_active = pyqtSignal(str, float)  # noqa
    signal_update_buttons = pyqtSignal(bool)  # noqa
    signal_update_fov_id = pyqtSignal(int)  # noqa
    signal_update_display = pyqtSignal()  # noqa

    def __init__(
            self,
            queue_manager: QueueManager,
            fov_data: dict[str, Any],
            ref_data: dict[int, np.ndarray],
            focus_data: dict[str, Any],
            roi_data: dict[int, dict[str, Any]],
            processor_config: ConfigImageProcessor,
     ):
        super().__init__()  # noqa
        self.setWindowTitle("Field of Views")
        self.setStyleSheet("""
        background-color: #262626;
        color: #FFFFFF;
        font-family: Titillium;
        font-size: 18px;
        """)
        self.queue_manager: QueueManager = queue_manager
        self.fov_data: dict[str, Any] = fov_data
        self.ref_data: dict[int, np.ndarray] = ref_data
        self.focus_data: dict[str, Any] = focus_data
        self.roi_data: dict[int, dict[str, Any]] = roi_data
        self.processor_config: ConfigImageProcessor = processor_config
        self.channel_to_index = self.processor_config.channel_to_index

        self.fovs: dict[int, Coordinate] = fov_data["fovs"]
        self.fov_cropping_boxes: dict[int, None | EvoCroppingBox] = fov_data["cropping_boxes"]
        self.rotations: dict[int, float] = {k: d["rotation"] for k, d in self.roi_data.items()}

        self.combo_box = QComboBox()  # noqa
        self.combo_box.addItems(["FoV "+str(key) for key in self.fovs.keys()])
        self.combo_box.currentIndexChanged.connect(self.update_display)  # noqa
        self.curr_fov = list(self.fovs.keys())[0]

        self.actives = {k: {"z_pos": self.fovs[k].z, "rotation": self.rotations[k]}
                        for k in self.fovs.keys()}
        self.overrides = {k: {"z_pos": None, "rotation": None} for k in self.fovs.keys()}
        self.param_types = {"z_pos": float, "rotation": float}
        self.info_labels = {
            "fov": EvoPanelTemplate.make_label(f"FoV {self.curr_fov} at {str(self.fovs[self.curr_fov])}", font=SMALL),
        }
        self.labels = {
            "z_pos": EvoPanelTemplate.make_label("Z Position: ", font=SMALL),
            "rotation": EvoPanelTemplate.make_label("Rotation: ", font=SMALL),
        }
        self.value_labels = {
            "z_pos": EvoPanelTemplate.make_label(self._frmt_z(self.actives[self.curr_fov]['z_pos']), font=SMALL),
            "rotation": EvoPanelTemplate.make_label(self._frmt_z(self.actives[self.curr_fov]['rotation']), font=SMALL),
        }
        self.override_labels = {
            "z_pos": EvoPanelTemplate.make_label(f"{str(self.overrides[self.curr_fov]['z_pos'])}", font=SMALL),
            "rotation": EvoPanelTemplate.make_label(f"{str(self.overrides[self.curr_fov]['rotation'])}", font=SMALL),
        }
        self.edits = {
            "z_pos": EvoPanelTemplate.make_lineedit(text="None", func=self.update_override, param="z_pos"),
            "rotation": EvoPanelTemplate.make_lineedit(text="None", func=self.update_override, param="rotation"),
        }
        self.override_buttons = {
            "z_pos": EvoPanelTemplate.make_button("Override", func=self.send_override, font=SMALL, field="z_pos"),
            "rotation": EvoPanelTemplate.make_button("Override", func=self.send_override, font=SMALL, field="rotation"),
        }

        self.worker = PositionDialogWorker(
            curr_fov_id=self.curr_fov,
            fov_coordinates=self.fovs,
            actives=self.actives,
            overrides=self.overrides,
            info_labels=self.info_labels,
            value_labels=self.value_labels,
            override_labels=self.override_labels,
            edits=self.edits,
            buttons=self.override_buttons,
            combo_box=self.combo_box,
        )
        self.signal_update_fov_id.connect(self.worker.update_fov_id)
        self.signal_update_override.connect(self.worker.update_override)
        self.signal_update_active.connect(self.worker.update_active)
        self.signal_update_buttons.connect(self.worker.update_buttons)
        self.signal_update_display.connect(self.worker.update_display)
        self.thread = EvoGUIThread()
        self.worker.moveToThread(self.thread)
        self.thread.start()

        self.orig_plot = ChannelPlotter(
            img=self.ref_data[self.curr_fov],
            channel_to_index=self.channel_to_index,
            height=6,
            width=6,
            title_prefix="Raw - ",
        )
        self.rot_data = copy.deepcopy(self.ref_data)
        for i in range(len(self.rot_data)):
            for j in range(self.rot_data[i].shape[0]):
                self.rot_data[i][j, :, :] = delta.imgops.affine_transform(
                    image=self.rot_data[i][j, :, :],
                    angle=self.rotations[i],
                    order=1,
                    border_mode=cv2.BORDER_CONSTANT,
                )
        self.rot_plot = ChannelPlotter(
            img=self.rot_data[self.curr_fov],
            channel_to_index=self.channel_to_index,
            height=6,
            width=6,
            title_prefix="Rotated - ",
            roi_boxes=self.roi_data[self.curr_fov]['roi_boxes']
        )

        self.layout = QGridLayout()
        this_row = 0
        self.layout.addWidget(self.info_labels["fov"], this_row, 0, 1, 4, LEFT)
        self.layout.addWidget(self.combo_box, this_row, 4, 1, 1, RIGHT)

        this_row += 1
        self.layout.addWidget(EvoPanelTemplate.make_label(f"Current", font=VERYSMALL), this_row, 1, 1, 1, CENTER)
        self.layout.addWidget(EvoPanelTemplate.make_label(f"Send", font=VERYSMALL), this_row, 2, 1, 1, CENTER)

        this_row += 1
        for i, k in enumerate(self.param_types.keys(), start=this_row):
            self.layout.addWidget(self.value_labels[k], i, 1, 1, 1, CENTER)
            self.layout.addWidget(self.override_labels[k], i, 2, 1, 1, CENTER)
            self.layout.addWidget(self.edits[k], i, 3, 1, 1, LEFT)
            self.layout.addWidget(self.override_buttons[k], i, 4, 1, 1, RIGHT)
            self.layout.addWidget(self.labels[k], i, 0, 1, 1, LEFT)

        this_row += len(self.param_types.keys())
        self.layout.addWidget(self.orig_plot.widget, this_row, 0, 2, 2, LEFT)
        self.layout.addWidget(self.rot_plot.widget, this_row, 2, 2, 2, LEFT)

        self.setLayout(self.layout)

        # Initialize display with the first index
        self.update_display()

    @staticmethod
    def _frmt_z(z: float | int | None) -> str:
        return "None" if z is None else f"{z:.2f}"

    def update_display(self):
        self.curr_fov = self.combo_box.currentIndex()
        self.signal_update_fov_id.emit(self.curr_fov)
        self.orig_plot.update_image(self.ref_data[self.curr_fov], [])
        self.rot_plot.update_image(self.rot_data[self.curr_fov], self.roi_data[self.curr_fov]['roi_boxes'])

    def update_override(self, param: str):
        logger.debug(f"Updating override for {param}.")
        try:
            new_val = self.param_types[param](self.edits[param].text())
            self.overrides[self.curr_fov][param] = new_val
        except ValueError as e:
            logger.error(f"Cannot parse <{self.edits[param].text()}> for parameter {param}.")
            return
        if isinstance(new_val, float):
            self.signal_update_override.emit(param, new_val)

    def send_override(self, field: str):
        logger.debug(f"Sending override {self.overrides[self.curr_fov][field]} for {field}.")
        if self.overrides[self.curr_fov][field] is None:
            logger.error(f"Cannot send None for {field} and FoV {self.curr_fov}.")
            return
        self.signal_update_buttons.emit(False)
        self.queue_manager.request(
            req_str='self.override_parameter',
            kwargs_dict={
                'fov_id': self.curr_fov,
                'pos_id': self.curr_fov,  # FIXME
                'param_name': field,
                'param_value': self.overrides[self.curr_fov][field],
            },
            callback=self.receive_override_response,
            callback_args=(self.curr_fov, self.curr_fov, field, self.overrides[self.curr_fov][field]),
        )

    def receive_override_response(self, data: Any, fov_id: int, pos_id: int, param_name: str, param_value: Any):
        if isinstance(data, Exception):
            logger.error("PositionDialog.receive_override_response: received exception. Returning.")
            return
        logger.debug(f"Received override response for {param_name}={param_value} for FoV {fov_id}.")
        if fov_id != self.curr_fov:
            logger.error(f"Received response for FoV {fov_id}, but currently at {self.curr_fov}.")
            return
        self.actives[self.curr_fov][param_name] = param_value
        self.overrides[self.curr_fov][param_name] = None
        if param_name == 'rotation':
            for j in range(self.rot_data[fov_id].shape[0]):
                self.rot_data[fov_id][j, :, :] = delta.imgops.affine_transform(
                    image=self.rot_data[fov_id][j, :, :],
                    angle=self.actives[fov_id][param_name],
                    order=1,
                    border_mode=cv2.BORDER_CONSTANT,
                )
                self.rot_plot.update_image(self.rot_data[fov_id], self.roi_data[self.curr_fov]['roi_boxes'])
        elif param_name == 'z_pos':
            self.fovs[self.fov_id].z = float(param_value)
        self.signal_update_active.emit(param_name, param_value)
        self.signal_update_buttons.emit(True)

