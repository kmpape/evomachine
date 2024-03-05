import matplotlib.pyplot as plt
from multiprocessing import Event
import numpy as np
from serial import SerialException
from typing import Dict, List, Optional, Tuple, Union
from PyQt5.QtCore import pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QWidget, QPushButton

import delta.utils

from evomachine.acquisition import AbstractCamera
from evomachine.commands import AutomatonCommand, AutomatonCommandType
from evomachine.config import ConfigCamera, ConfigImageProcessor, get_logger
from evomachine.coordinates import Coordinate, CoordinateFactory
from evomachine.guidir.figures import FigureMultiWindow
from evomachine.guidir.guitemplates import EvoGUIThread, EvoPanelTemplate, EvoWorkerTemplate
from evomachine.guidir.guitypes import SMALL, NORMAL, LEFT, AXES
from evomachine.guidir.queuemanager import QueueManager
from evomachine.utils import EvoCroppingBox

logger = get_logger(name=__name__, is_gui=True)


class ExperimentWorker(EvoWorkerTemplate):
    def __init__(
            self,
            queue_manager: QueueManager,
            config_camera: ConfigCamera,
    ):
        super().__init__()
        self.queue_manager = queue_manager
        self.cfg_camera = config_camera
        self.valid_coordinates = False
        self.factory: CoordinateFactory = CoordinateFactory(dfov=self.cfg_camera.fov_size)
        self._field_of_views: Union[None, Dict[int, Coordinate]] = None
        self._cropping_boxes: Union[None, List[delta.utils.CroppingBox]] = None
        self._stage_limits: Union[None, Dict[str, Tuple[float, float]]] = None
        self.queue_manager.request(
            req_str='self.cam.get_stage_limits',
            kwargs_dict={},
            callback=self.update_limits,
        )
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

    def coordinate_is_out_of_bounds(self, coordinates: Dict[str, float]) -> bool:
        return False if self._stage_limits is None else any((key not in self._stage_limits) or
                                                            (val < self._stage_limits[key][0]) or
                                                            (val > self._stage_limits[key][1])
                                                            for key, val in coordinates.items())

    def update_limits(self, data: Tuple[Coordinate, Coordinate]):
        logger.info(f"Stage limits: {data}")
        self._stage_limits = {'X': (data[0].x, data[1].x), 'Y': (data[0].y, data[1].y), 'Z': (data[0].z, data[1].z)}

    def get_positions_from_dict(
            self,
            positions: Dict[int, Dict[str, Union[None, Dict[str, Union[float, int]]]]]
    ) -> Union[List[Coordinate], None]:
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
        for from_to in valid_dict:
            grid = self.factory.make_grid(
                start=Coordinate.from_dict(from_to["from"]),
                stop=Coordinate.from_dict(from_to["to"]),
            )
            coordinates.extend(grid)
        logger.info(f"Extracted {len(coordinates)} coordinates: {coordinates}")
        return coordinates

    def initialise_automaton_focus(self):
        self.queue_manager.request(
            req_str='self.initialise_fov_focus',
            kwargs_dict={},
            callback=None,
        )

    def initialise_automaton_image_processors(self):
        logger.debug("ExperimentWorker.initialise_automaton_image_processors: Requesting initialisation of IP.")
        self.queue_manager.request(
            req_str='self.initialise_position_processor',
            kwargs_dict={},
            callback=None,
        )

    def initialise_automaton_references(self):
        logger.debug("ExperimentWorker.initialise_automaton_references: Requesting initialisation of references.")
        self.queue_manager.request(
            req_str='self.initialise_reference_frames',
            kwargs_dict={},
            callback=None,
        )

    def initialise_automaton_field_of_views(
            self,
            read_in_positions: Dict[int, Dict[str, Union[None, Dict[str, Union[float, int]]]]],
            cropping_boxes: Optional[List[EvoCroppingBox]] = None,
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
        self.queue_manager.request(
            req_str='self.initialise_field_of_view_list',
            kwargs_dict={'field_of_views': self._field_of_views, 'cropping_boxes': self._cropping_boxes},
            callback=None,
        )


class ButtonWorker(EvoWorkerTemplate):
    def __init__(self, button_list: List[QPushButton]):
        super().__init__()
        self.button_list = button_list

    @pyqtSlot(list, str)
    def set_color(self, button_indices: List[int], color_str: str):
        for i in button_indices:
            self.button_list[i].setStyleSheet(f"background-color: {color_str};")


class ExperimentPanel(EvoPanelTemplate):
    signal_set_button_color = pyqtSignal(list, str)

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
        queue_manager.register(func=self.read_processor_data, msg_type=AutomatonCommandType.ROI_DATA)
        queue_manager.register(func=self.init_positions_done, msg_type=AutomatonCommandType.FOV_DATA)
        queue_manager.register(func=self.read_ref_data, msg_type=AutomatonCommandType.REF_DATA)

        self.worker = ExperimentWorker(queue_manager=queue_manager, config_camera=camera_config)
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
        self.init_positions_button = self.make_button(text="Initialise FoVs", func=self.init_positions, font=SMALL)
        self.init_focus_button = self.make_button(text="Initialise Focus", func=self.init_focus, font=SMALL)
        self.init_references_button = self.make_button(text="Take Refs", func=self.init_references, font=SMALL)
        self.init_processors_button = self.make_button(text="Initialise IP", func=self.init_processors, font=SMALL)
        self.read_in_clear_button = self.make_button(text="Clear Paths", func=self.clear_param, font=SMALL)
        self.focus_curves_button = self.make_button(text="Focus Curves", func=self.show_focus_curves, font=SMALL)
        self._automaton_is_initialised: bool = False
        self.start_button = self.make_button(text="Start", func=self.start_acquisition, font=SMALL)
        self.stop_button = self.make_button(text="Stop", func=self.stop_acquisition, font=SMALL)
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

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

        # FIXME any changes of <buttons> will bug below
        buttons = [self.init_positions_button, self.init_focus_button, self.init_references_button,
                   self.init_processors_button, self.start_button, self.stop_button, self.focus_curves_button]
        self.worker_buttons = ButtonWorker(button_list=buttons)
        self.workers.append(self.worker_buttons)
        thread = EvoGUIThread()
        self.worker_buttons.moveToThread(thread)
        thread.start()
        self.threads.append(thread)
        self.signal_set_button_color.connect(self.worker_buttons.set_color)

        self.cropping_boxes: Dict[int, Union[None, EvoCroppingBox]] = {0: None, 1: None}

        self.focus_data: Union[None, AutomatonCommand] = None
        self.focus_curves: Union[None, np.typing.Array] = None
        self.focus_prev_curr_stack: Union[None, Tuple[np.typing.Array, np.typing.Array]] = None
        self.focus_curves_window: Union[None, FigureMultiWindow] = None

        self.processor_data: Dict[int, AutomatonCommand] = {}

    def record_param(self, which: Tuple[int, str]):
        logger.debug(f"ExperimentPanel.record_param: Recording {which}.")
        self.queue_manager.request(
            req_str='self.cam.get_coordinates',
            kwargs_dict={'axes': AXES},
            callback=self._record_param,
            callback_args=(which,),
        )

    def _record_param(self, data, which: Tuple[int, str]):
        logger.debug("ExperimentPanel._record_param: Recording coordinates.")
        i, from_to = which
        try:
            pos_dict = data
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
        self._automaton_is_initialised = False
        self.init_focus_button.setStyleSheet("background-color: orange;")
        self.init_references_button.setStyleSheet("background-color: lightgray;")
        self.init_processors_button.setStyleSheet("background-color: lightgray;")
        self.worker.initialise_automaton_focus()
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

    def init_positions(self):
        self._automaton_is_initialised = False
        self.signal_set_button_color.emit([0], "orange")
        self.worker.initialise_automaton_field_of_views(
            read_in_positions=self.read_in_positions,
            cropping_boxes=[b for b in self.cropping_boxes.values() if b is not None],
        )

    def init_positions_done(self, data: AutomatonCommand):
        self.signal_set_button_color.emit([0], "green")
        self.signal_set_button_color.emit([1, 2, 3], "lightgray")

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
        logger.info("ExperimentPanel._start_acquisition: Starting acquisition.")
        self.stop_strategy_event.clear()
        self.start_strategy_event.set()
        self.start_button.setStyleSheet("background-color: green;")

    def stop_acquisition(self):
        logger.info("ExperimentPanel.stop_acquisition: Stopping acquisition.")
        self.start_strategy_event.clear()
        self.stop_strategy_event.set()
        self.start_button.setStyleSheet("background-color: lightgray;")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def read_focus_data(self, data: AutomatonCommand):
        self.signal_set_button_color.emit([1, 6], "green")
        self.focus_data = data

    def read_ref_data(self, data: AutomatonCommand):
        self.signal_set_button_color.emit([2], "green")

    def read_processor_data(self, data: AutomatonCommand):
        self.signal_set_button_color.emit([3], "green")
        self.processor_data[data.fov_id] = data.command_args

    def show_focus_curves(self):
        if self.focus_data is None:
            logger.error("exp_show_curve: missing data. Returning.")
            return
        focus_curves = self.focus_data.command_args['focus_curves']
        focus_prev_stack = self.focus_data.command_args['focus_prev_stack']
        focus_stack = self.focus_data.command_args['focus_stack']
        focus_prev_z_coords = self.focus_data.command_args['focus_prev_z_coords']

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

    @pyqtSlot(int, int, EvoCroppingBox)
    def update_cropping_boxes(self, fov_id: int, box_id: int, cropping_box: Union[EvoCroppingBox, None]):
        logger.debug(f"ExperimentPanel.update_cropping_boxes: fov_id={fov_id}, box_id={box_id}, cropping_box={cropping_box}")
        self.cropping_boxes[box_id] = None if cropping_box.is_none else cropping_box
