import matplotlib.pyplot as plt
from multiprocessing import Event
import numpy as np
from serial import SerialException
import threading
from typing import Any, Dict, List, Optional, Tuple, Union
from PyQt5.QtCore import pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QWidget, QPushButton, QDialog, QVBoxLayout, QComboBox, QLabel, QGridLayout

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
            signal_enable_button: pyqtSignal,
            signal_disable_button: pyqtSignal,
    ):
        super().__init__()
        self.queue_manager = queue_manager
        self.cfg_camera = config_camera
        self.signal_enable_button = signal_enable_button
        self.signal_disable_button = signal_disable_button

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
            callback=self.enable_disable_next_buttons,
            callback_args=([2, 6], [3, 4, 5],),
        )

    def initialise_automaton_image_processors(self):
        logger.debug("ExperimentWorker.initialise_automaton_image_processors: Requesting initialisation of IP.")
        self.queue_manager.request(
            req_str='self.initialise_position_processor',
            kwargs_dict={},
            callback=self.enable_disable_next_buttons,
            callback_args=([4], [],),
        )

    def initialise_automaton_references(self):
        logger.debug("ExperimentWorker.initialise_automaton_references: Requesting initialisation of references.")
        self.queue_manager.request(
            req_str='self.initialise_reference_frames',
            kwargs_dict={},
            callback=self.enable_disable_next_buttons,
            callback_args=([3], [4, 5],),
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
            callback=self.enable_disable_next_buttons,
            callback_args=([1], [2, 3, 4, 5, 6],),
        )

    def enable_disable_next_buttons(self, data: Any, enable: List[int], disable: List[int]):
        self.signal_enable_button.emit(enable)
        self.signal_disable_button.emit(disable)


class ButtonWorker(EvoWorkerTemplate):
    def __init__(self, button_list: List[QPushButton]):
        super().__init__()
        self.button_list = button_list

    @pyqtSlot(list, str)
    def set_color(self, button_indices: List[int], color_str: str):
        for i in button_indices:
            self.button_list[i].setStyleSheet(f"background-color: {color_str};")

    @pyqtSlot(list)
    def disable_button(self, indices: List[int]):
        for i in indices:
            self.button_list[i].setEnabled(False)

    @pyqtSlot(list)
    def enable_button(self, indices: List[int]):
        for i in indices:
            self.button_list[i].setEnabled(True)


class FoVPanel(EvoPanelTemplate):
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
        self.worker = ExperimentWorker(queue_manager=queue_manager, config_camera=camera_config)
        self.workers.append(self.worker)
        thread = EvoGUIThread()
        self.worker.moveToThread(thread)
        thread.start()
        self.threads.append(thread)


class ExperimentPanel(EvoPanelTemplate):
    signal_set_button_color = pyqtSignal(list, str)
    signal_enable_button = pyqtSignal(list)
    signal_disable_button = pyqtSignal(list)

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
        )
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
        self.signal_set_button_color.connect(self.worker_buttons.set_color)
        self.signal_disable_button.connect(self.worker_buttons.disable_button)
        self.signal_enable_button.connect(self.worker_buttons.enable_button)
        self.workers.append(self.worker_buttons)
        thread = EvoGUIThread()
        self.worker_buttons.moveToThread(thread)
        thread.start()
        self.threads.append(thread)

        self.signal_disable_button.emit([0, 1, 2, 3, 4, 5, 6])

        self.cropping_boxes: Dict[int, Union[None, EvoCroppingBox]] = {0: None, 1: None}

        self.fov_data: Dict[str, Any] = {}
        self.ref_data: Dict[str, Any] = {}
        self.roi_data: Dict[int, Dict[str, Any]] = {}
        self.focus_data: Dict[str, Any] = {}

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

        if self.read_in_positions[i]["from"] is not None and self.read_in_positions[i]["to"] is not None:
            self.signal_enable_button.emit([0])

    def clear_param(self):
        self.read_in_positions = {i: {"from": None, "to": None} for i in range(self.num_read_ins)}
        self.worker.set_labels(labels=self.read_in_display, text="?")
        if not any(r["from"] is not None and r["to"] is not None for r in self.read_in_positions.values()):
            self.signal_disable_button.emit([0])

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

    def read_fov_data(self, data: AutomatonCommand):
        # command_args = {
        #     'fovs': fovs,
        #     'cropping_boxes': cropping_boxes,
        #     'fov_to_pos': fov_to_pos,
        #     'pos_to_fov_index': pos_to_fov_index,
        # }
        self.fov_data = data.command_args
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
        self.focus_data = data.command_args

    def read_ref_data(self, data: AutomatonCommand):
        self.ref_data = data.command_data
        self.signal_set_button_color.emit([2], "green")

    def read_roi_data(self, data: AutomatonCommand):
        self.signal_set_button_color.emit([3], "green")
        self.roi_data[data.fov_id] = data.command_args

    def show_focus_curves(self):
        if self.focus_data is None:
            logger.error("exp_show_curve: missing data. Returning.")
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

    @pyqtSlot(int, int, EvoCroppingBox)
    def update_cropping_boxes(self, fov_id: int, box_id: int, cropping_box: Union[EvoCroppingBox, None]):
        logger.debug(f"ExperimentPanel.update_cropping_boxes: fov_id={fov_id}, box_id={box_id}, cropping_box={cropping_box}")
        self.cropping_boxes[box_id] = None if cropping_box.is_none else cropping_box


class PositionDialog(QDialog):
    def __init__(
            self,
            queue_manager: QueueManager,
            fov_data: Dict[str, Any],
            ref_data: Dict[str, Any],
            focus_data: Dict[str, Any],
            roi_data: Dict[int, Dict[str, Any]],
     ):
        super().__init__()
        self.setWindowTitle("Field of Views")
        self.setStyleSheet("""
        background-color: #262626;
        color: #FFFFFF;
        font-family: Titillium;
        font-size: 18px;
        """)
        self.queue_manager: QueueManager = queue_manager
        self.fov_data: Dict[str, Any] = fov_data
        self.ref_data: Dict[str, Any] = ref_data
        self.focus_data: Dict[str, Any] = focus_data
        self.roi_data: Dict[int, Dict[str, Any]] = roi_data

        self.fovs: Dict[int, Coordinate] = fov_data["fovs"]
        self.fov_cropping_boxes: Dict[int, Union[None, EvoCroppingBox]] = fov_data["cropping_boxes"]
        self.rotations: Dict[int, float] = {k: d["rotation"] for k, d in self.roi_data.items()}

        self.combo_box = QComboBox()
        self.combo_box.addItems(["FoV "+str(key) for key in self.fovs.keys()])
        self.combo_box.currentIndexChanged.connect(self.update_display)
        self.curr_fov = list(self.fovs.keys())[0]

        self.actives = {k: {"z_pos": self.fovs[k].z, "rotation": self.rotations[k]}
                        for k in self.fovs.keys()}
        self.overrides = {k: {"z_pos": None, "rotation": None} for k in self.fovs.keys()}
        self.param_types = {"z_pos": int, "rotation": float}
        self.info_labels = {
            "fov": self.make_label(f"Field of view {self.curr_fov}", font=NORMAL),
            "coordinate": self.make_label(f"Coordinate {str(self.fovs[self.curr_fov])}", font=NORMAL),
        }
        self.labels = {
            "z_pos": EvoPanelTemplate.make_label("Z Position: ", font=SMALL),
            "rotation": EvoPanelTemplate.make_label("Rotation: ", font=SMALL),
        }
        self.value_labels = {
            "z_pos": EvoPanelTemplate.make_label(f"{self.actives[self.curr_fov]['z_pos']:.2f}", font=SMALL),
            "rotation": EvoPanelTemplate.make_label(f"{self.actives[self.curr_fov]['rotation']:.2f}", font=SMALL),
        }
        self.override_labels = {
            "z_pos": EvoPanelTemplate.make_label(f"{str(self.overrides[self.curr_fov]['z_pos'])}", font=SMALL),
            "rotation": EvoPanelTemplate.make_label(f"{str(self.overrides[self.curr_fov]['rotation'])}", font=SMALL),
        }
        self.edits = {
            "z_pos": EvoPanelTemplate.make_lineedit(text="None", func=self.update_overrides, param="z_pos"),
            "rotation": EvoPanelTemplate.make_lineedit(text="None", func=self.update_overrides, param="rotation"),
        }
        self.override_buttons = {
            "z_pos": EvoPanelTemplate.make_button("Override", func=self.send_override, font=SMALL, field="z_pos"),
            "rotation": EvoPanelTemplate.make_button("Override", func=self.send_override, font=SMALL, field="rotation"),
        }

        self.layout = QGridLayout()
        self.layout.addWidget(self.info_labels["fov"], 0, 0, 1, 1, LEFT)
        self.layout.addWidget(self.info_labels["coordinate"], 0, 1, 1, 1, LEFT)
        self.layout.addWidget(self.combo_box, 0, 2, 1, 1, LEFT)

        self.layout.addWidget(self.make_label(f"Active", font=NORMAL), 1, 0, 1, 1, LEFT)
        self.layout.addWidget(self.make_label(f"To overwrite", font=NORMAL), 1, 0, 1, 1, LEFT)
        for i, k in enumerate(self.overrides, start=2):
            self.layout.addWidget(self.labels[k], i, 0, 1, 1, LEFT)
            self.layout.addWidget(self.value_labels[k], i, 1, 1, 1, LEFT)
            self.layout.addWidget(self.override_labels[k], i, 2, 1, 1, LEFT)
            self.layout.addWidget(self.edits[k], i, 3, 1, 1, LEFT)
            self.layout.addWidget(self.override_buttons[k], i, 4, 1, 1, LEFT)

        self.setLayout(self.layout)

        # Initialize display with the first index
        self.update_display(0)

    def update_display(self):
        self.curr_fov = self.combo_box.currentIndex()
        thread = threading.Thread(
            target=self._update_display,
            args=(
                self.curr_fov,
                self.fovs[self.curr_fov],
                self.actives[self.curr_fov],
                self.overrides[self.curr_fov],
                self.info_labels,
                self.value_labels,
                self.override_labels
            ),
        )
        thread.start()

    @staticmethod
    def _update_display(
            fov_id: int,
            fov_coordinate: Coordinate,
            actives: Dict[str, Any],
            overrides: Dict[str, Any],
            info_labels: Dict[str, QLabel],
            value_labels: Dict[str, QLabel],
            override_labels: Dict[str, QLabel],
    ):
        info_labels["fov"].setText(f"Field of view {fov_id}")
        info_labels["coordinate"].setText(f"Field of view {str(fov_coordinate)}")
        for k in overrides.keys():
            value_labels[k].setText(f"{actives[k]:.2f}")
            override_labels[k].setText("None" if overrides[k] is None else f"{overrides[k]:.2f}")

    def update_override(self, param: str):
        try:
            new_val = self.param_types[param](self.edits[param].text())
        except ValueError as e:
            logger.error(f"Cannot parse <{self.edits[param].text()}> for parameter {param}.")
            return
        if isinstance(new_val, float):
            self.override_labels[param].setText(f"{new_val:.2f}")
        else:
            self.override_labels[param].setText(str(new_val))

    def send_override(self, field: str):
        logger.debug(f"Sending override for {field}.")
        if self.overrides[self.curr_fov][field] is None:
            logger.error(f"Cannot send None for {field} and FoV {self.curr_fov}.")
            return
        self.override_buttons[field].setEnabled(False)
        self.combo_box.setEnabled(False)
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
        logger.debug(f"Received override response for {param_name}={param_value} for FoV {fov_id}.")
        if fov_id != self.curr_fov:
            logger.error(f"Received response for FoV {fov_id}, but currently at {self.curr_fov}.")
            return
        self.actives[self.curr_fov][param_name] = param_value
        self.overrides[self.curr_fov][param_name] = None
        self.update_display()
        self.override_buttons[param_name].setEnabled(True)
        self.combo_box.setEnabled(True)

