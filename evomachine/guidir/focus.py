from multiprocessing import Event, Queue
import numpy as np
import matplotlib.pyplot as plt
from typing import Any
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

from evomachine.acquisition import AbstractCamera
from evomachine.config import ConfigCamera, ConfigCRISP, ConfigFocus, ConfigImageProcessor, get_logger
from evomachine.types import FocusAlgorithmType, LEDType
from evomachine.guidir.figures import FigureWindow
from evomachine.guidir.guitemplates import EvoPanelTemplate, EvoWorkerTemplate, EvoGUIThread
from evomachine.guidir.guitypes import SMALL, CENTER, LEFT, RIGHT, NORMAL
from evomachine.guidir.queuemanager import QueueManager
from evomachine.utils import EvoCroppingBox


logger = get_logger(name=__name__, is_gui=True)


class FocusWorker(EvoWorkerTemplate):
    def __init__(
            self,
            labels_values: dict[str, list[QLabel | QLineEdit]],
            this_cfg: ConfigFocus,
            parent: QObject | None = None,
    ):
        super().__init__(parent)
        self.this_cfg: ConfigFocus | ConfigCRISP = this_cfg
        self.labels_values: dict[str, list[QLabel, QLineEdit]] = labels_values

        # Available after calling the focus routine
        self.focus_curve_window: FigureWindow | None = None
        self.focus_z_coords: np.ndarray | None = None
        self.focus_scores: np.ndarray | None = None
        self.focus_stack: np.ndarray | None = None
        self.prev_image: np.ndarray | None = None
        self.prev_z: float | None = None
        self.new_z: float | None = None

    @pyqtSlot()
    def clear_config(self):
        for param_name in self.labels_values.keys():
            if param_name == 'algorithm':
                dropdown = self.labels_values[param_name][1]
                index = dropdown.findText(FocusAlgorithmType.get_name(self.this_cfg.algorithm.value))
                self.labels_values[param_name][1].setCurrentIndex(index)
            else:
                self.labels_values[param_name][1].setText(str(getattr(self.this_cfg, param_name)))

    def read_focus_data(self, data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]):
        self.focus_z_coords, self.focus_scores, self.focus_stack, self.prev_image, self.prev_z, self.new_z = data

    @pyqtSlot()
    def show_curve(self):
        if self.focus_z_coords is None:
            logger.error("Cannot show focus curves. No data available.")
            return

        fig = plt.figure()
        gs = fig.add_gridspec(2, 2, width_ratios=[1, 1])
        ax1 = fig.add_subplot(gs[0, :])
        z_coords = np.array(list(self.focus_z_coords)) / 10
        ax1.plot(z_coords, self.focus_scores, marker='x')
        ax1.set_xticks(z_coords.tolist())
        ax1.set_xticklabels([f'{x:.1f}' for x in z_coords.tolist()])
        ax1.set_xlabel("Z position [um]")
        ax1.set_ylabel("Sharpness Scores")
        ax1.set_title("Focus Curve")
        best_index = np.argmax(self.focus_scores)
        best_image = AbstractCamera.normalise_frame(self.focus_stack[:, :, best_index])
        prev_image = AbstractCamera.normalise_frame(self.prev_image)
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.imshow(prev_image)
        ax2.set_title(f"Before Focus\n Z={self.prev_z/10:.1f}")
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.imshow(best_image)
        ax3.set_title(f"After Focus\n Z={z_coords[best_index]:.1f}")
        fig.tight_layout()
        self.focus_curve_window = FigureWindow(fig, title="mothermachine_gui: Focus curve.")
        # self.focus_curve_window.move(self.geometry().x() + self.geometry().width() + 10, self.geometry().y())
        self.focus_curve_window.show()


class FocusPanel(EvoPanelTemplate):
    signal_clear_cfg = pyqtSignal()
    signal_show_curve = pyqtSignal()

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
        self.cfg_focus = self.camera_config.focus
        self.cfg_focus_default = self.camera_config.focus.copy()
        self.cropping_box: None | EvoCroppingBox = None
        self.focus_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float] | None = None

        self.algorithm_dropdown_options = FocusAlgorithmType.get_all_names()
        curr_algorithm_name = FocusAlgorithmType.get_name(self.cfg_focus.algorithm.value)
        self.algorithm_dropdown_options.remove(curr_algorithm_name)
        self.algorithm_dropdown_options.insert(0, curr_algorithm_name)
        self.algorithm_current_option = FocusAlgorithmType.from_string(
            self.algorithm_dropdown_options[0]
        )
        self.algorithm_dropdown = self.make_dropdown(items=self.algorithm_dropdown_options,
                                                     func=self.update_algorithm_option)
        self.labels_values = {
            'exposure_time': [self.make_label(text="Exposure [ms]", font=SMALL),
                              self.make_lineedit(text=str(int(self.cfg_focus.exposure_time)),
                                                 func=self.update_param, param='exposure_time')],
            'focus_channel': [self.make_label(text="Channel number [0,...,3]", font=SMALL),
                              self.make_lineedit(text=str(int(self.cfg_focus.focus_channel.value)),
                                                 func=self.update_param, param='focus_channel')],
            'rel_range': [self.make_label(text="Relative range [um*10]", font=SMALL),
                          self.make_lineedit(text=str(int(self.cfg_focus.rel_range)),
                                             func=self.update_param, param='rel_range')],
            'step_size': [self.make_label(text="Step Size [um*10]", font=SMALL),
                          self.make_lineedit(text=str(int(self.cfg_focus.step_size)),
                                             func=self.update_param, param='step_size')],
            'algorithm': [self.make_label(text="Algorithm", font=SMALL), self.algorithm_dropdown],
        }
        self.start_button = self.make_button(text="Start", font=SMALL, func=self.stop_crisp)
        self.start_button.setEnabled(True)
        self.stop_button = self.make_button(text="Stop", font=SMALL, func=self.stop_focus)
        self.stop_button.setEnabled(False)
        self.reset_button = self.make_button(text="Reset", font=SMALL, func=self.reset_focus)
        self.curve_button = self.make_button(text="Focus Curve", font=SMALL, func=self.show_curve)
        self.curve_button.setEnabled(False)
        self.layout = QGridLayout()
        self.layout.addWidget(self.make_label(text="Software Focus", font=NORMAL), 0, 0, 1, 4, LEFT)
        for i, lab_val in enumerate(self.labels_values.values(), start=1):
            self.layout.addWidget(lab_val[0], i, 0, CENTER)
            self.layout.addWidget(lab_val[1], i, 1, CENTER)
        self.layout.addWidget(self.start_button, len(self.labels_values)+2, 0, CENTER)
        self.layout.addWidget(self.stop_button, len(self.labels_values)+2, 1, CENTER)
        self.layout.addWidget(self.reset_button, len(self.labels_values)+2, 2, CENTER)
        self.layout.addWidget(self.curve_button, len(self.labels_values)+2, 3, CENTER)

        self.widget = QWidget()
        self.widget.setLayout(self.layout)

        self.worker = FocusWorker(labels_values=self.labels_values, this_cfg=self.cfg_focus_default)
        self.signal_clear_cfg.connect(self.worker.clear_config)
        self.signal_show_curve.connect(self.worker.show_curve)
        self.workers.append(self.worker)
        thread = EvoGUIThread()
        self.worker.moveToThread(thread)
        thread.start()
        self.threads.append(thread)

    def get_param(self, param_name: str):
        if param_name == 'algorithm':
            val = self.algorithm_current_option
        else:
            val = ConfigFocus.get_attr_from_str(
                attr_name=param_name,
                attr_value_str=self.labels_values[param_name][1].text(),
            )
        if not self.cfg_focus.attr_is_valid(attr_name=param_name, attr_value=val):
            raise ValueError("Check parameter range and type in evomachine.config.")
        return val

    def reset_focus(self):
        self.cfg_focus = self.cfg_focus_default.copy()
        self.signal_clear_cfg.emit()

    def show_curve(self):
        if self.focus_data is None:
            logger.error("No focus data available. Returning...")
            return
        self.signal_show_curve.emit()

    def update_algorithm_option(self):
        self.algorithm_current_option = FocusAlgorithmType.from_string(
            self.algorithm_dropdown.currentText()
        )
        setattr(self.cfg_focus, 'algorithm', self.algorithm_current_option)

    @pyqtSlot(int, int, EvoCroppingBox)
    def update_cropping_box(self, fov_id: int, box_id: int, cropping_box: EvoCroppingBox):
        if box_id == 0:
            self.cropping_box = None if cropping_box is None else cropping_box

    def read_focus_data(self, data: Any):
        logger.info("Focus done.")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.curve_button.setEnabled(True)
        self.focus_data = data
        self.worker.read_focus_data(data)

    @pyqtSlot()
    def stop_crisp(self):
        logger.info("Starting focus and stopping autofocus.")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.stop_event.clear()
        self.queue_manager.request(
            req_str='self.cam.autofocus_unlock',
            kwargs_dict={},
            callback=self.start_focus,
        )

    @pyqtSlot()
    def stop_focus(self):
        logger.info("Stopping focus.")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.stop_event.set()

    def start_focus(self, data: Any):
        try:
            for key in self.labels_values.keys():
                self.get_param(param_name=key)
        except ValueError as e:
            logger.warning(f"start: invalid parameter provided. Aborting. {e}")
            return
        self.queue_manager.request(
            req_str='self.software_focus',
            kwargs_dict={
                'cfg_focus': self.cfg_focus,
                'cropping_box': self.cropping_box,
                'user_input_override': True,
            },
            callback=self.read_focus_data,
        )

    def update_param(self, param_name: str):
        try:
            val = self.get_param(param_name=param_name)
            if param_name == 'focus_channel':
                val = LEDType(val)
            setattr(self.cfg_focus, param_name, val)
            logger.debug(f"Changed {param_name} to {val}")
        except ValueError as e:
            logger.warning(f"update: invalid parameter for key {param_name}: {e}")
            self.labels_values[param_name][1].setText(str(getattr(self.cfg_focus_default, param_name)))
