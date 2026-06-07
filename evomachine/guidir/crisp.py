from multiprocessing import Event, Queue
import numpy as np
import matplotlib.pyplot as plt
import time
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

from evomachine.bindings.asitiger.autofocus import TigerAutofocusConfig
from evomachine.config import ImageProcessorConfig, get_logger
from evomachine.types import FocusAlgorithmType, LEDType
from evomachine.guidir.figures import FigureWindow
from evomachine.guidir.guitemplates import EvoPanelTemplate, EvoWorkerTemplate, EvoGUIThread, EVO_STYLE
from evomachine.guidir.guitypes import SMALL, CENTER, LEFT, RIGHT, NORMAL
from evomachine.guidir.queuemanager import QueueManager
from evomachine.utils import EvoCroppingBox


logger = get_logger(name=__name__, is_gui=True)


class CrispWorker(EvoWorkerTemplate):
    def __init__(
            self,
            queue_manager: QueueManager,
            labels_values: dict[str, list[QLabel | QLineEdit]],
            init_button: QPushButton,
            lock_button: QPushButton,
            unlock_button: QPushButton,
            locked_value: QLabel,
            parent=None,
    ):
        super().__init__(parent)
        self.queue_manager: QueueManager = queue_manager
        self.labels_values: dict[str, list[QLabel | QLineEdit]] = labels_values
        self.init_button: QPushButton = init_button
        self.lock_button: QPushButton = lock_button
        self.unlock_button: QPushButton = unlock_button
        self.locked_value: QLabel = locked_value
        self.is_locked = False

    @pyqtSlot(TigerAutofocusConfig)
    def init_crisp(self, cfg_crisp: TigerAutofocusConfig):
        self.show_processing()
        self.queue_manager.request(
            req_str='self.cam.autofocus_initialise',
            kwargs_dict={'this_cfg_crisp': cfg_crisp, 'user_input': False},
            callback=self.check_crisp,
        )

    @pyqtSlot()
    def init_crisp_values(self):
        self.check_crisp(data=None)

    def check_crisp(self, data: Any):
        time.sleep(1)  # Need to be sure that we get the updated value here
        self.queue_manager.request(
            req_str='self.cam.autofocus_is_locked',
            kwargs_dict={},
            callback=self.show_done,
        )

    def show_processing(self):
        self.init_button.setText("In progress")
        # self.init_button.setStyleSheet("background-color: orange;")

    def show_done(self, data: bool):
        self.is_locked = data
        self.init_button.setText("Init")
        # self.init_button.setStyleSheet(EVO_STYLE)
        time.sleep(1)
        if self.is_locked:
            self.locked_value.setText("Yes")
            self.locked_value.setStyleSheet("background-color: green;")
        else:
            self.locked_value.setStyleSheet("background-color: red;")
            self.locked_value.setText("No")
        self.lock_button.setEnabled(True)

    @pyqtSlot()
    def unlock_crisp(self):
        self.queue_manager.request(
            req_str='self.cam.autofocus_unlock',
            kwargs_dict={},
            callback=self.check_crisp,
        )

    @pyqtSlot()
    def lock_crisp(self):
        self.queue_manager.request(
            req_str='self.cam.autofocus_lock',
            kwargs_dict={},
            callback=self.check_crisp,
        )

    @pyqtSlot()
    def reset_config(self):
        for param_name in self.labels_values.keys():
            if param_name == 'algorithm':
                dropdown = self.labels_values[param_name][1]
                index = dropdown.findText(FocusAlgorithmType.get_name(self.this_cfg.algorithm.value))
                self.labels_values[param_name][1].setCurrentIndex(index)
            else:
                self.labels_values[param_name][1].setText(str(getattr(self.this_cfg, param_name)))
            time.sleep(0.1)


class CRISPPanel(EvoPanelTemplate):
    signal_init_crisp = pyqtSignal(TigerAutofocusConfig)
    signal_init_crisp_values = pyqtSignal()
    signal_lock_crisp = pyqtSignal()
    signal_unlock_crisp = pyqtSignal()
    signal_reset_config = pyqtSignal()

    def __init__(
            self,
            queue_manager: QueueManager,
            camera_config: Any,
            processor_config: ImageProcessorConfig,
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
        self.cfg_crisp = self.camera_config.autofocus.copy()
        self.cfg_default = self.camera_config.autofocus.copy()
        self.labels_values = {
            'averaging': [self.make_label(text="#Samples averaged [ms]", font=SMALL),
                          self.make_lineedit(text=str(int(self.cfg_crisp.averaging)),
                                             func=self.update_param, param='averaging')],
            'led_intensity': [self.make_label(text="LED intensity [1,100]", font=SMALL),
                              self.make_lineedit(text=str(int(self.cfg_crisp.led_intensity)),
                                                 func=self.update_param, param='led_intensity')],
            'lock_range': [self.make_label(text="Lock range [mm]", font=SMALL),
                           self.make_lineedit(text=str(float(self.cfg_crisp.lock_range)),
                                              func=self.update_param, param='lock_range')],
            'loop_gain': [self.make_label(text="Loop gain [1,100]", font=SMALL),
                          self.make_lineedit(text=str(int(self.cfg_crisp.loop_gain)),
                                             func=self.update_param, param='loop_gain')],
            'objective_na': [self.make_label(text="NA (0,Inf)", font=SMALL),
                             self.make_lineedit(text=str(float(self.cfg_crisp.objective_na)),
                                                func=self.update_param, param='objective_na')],
            'update_rate': [self.make_label(text="Update rate [ms]]", font=SMALL),
                            self.make_lineedit(text=str(int(self.cfg_crisp.update_rate)),
                                               func=self.update_param, param='update_rate')]
        }
        self.labels_values['objective_na'][1].setEnabled(False)
        self.locked_label = self.make_label(text="Locked", font=SMALL, width_px=100)
        self.locked_value = self.make_label(
            text="No",
            font=SMALL,
            width_px=100,
            stylesheet="background-color: red;",
        )
        self.init_button = self.make_button(text="Init", font=SMALL, func=self.init_crisp)
        self.lock_button = self.make_button(text="Lock", font=SMALL, func=self.signal_lock_crisp.emit)
        self.lock_button.setEnabled(False)
        self.unlock_button = self.make_button(text="Unlock", font=SMALL, func=self.signal_unlock_crisp.emit)
        self.reset_button = self.make_button(text="Reset Cfg", font=SMALL, func=self.signal_reset_config.emit)

        self.layout = QGridLayout()
        self.layout.addWidget(self.make_label(text="CRISP Control", font=NORMAL), 0, 0, 1, 3, LEFT)
        for i, lab_val in enumerate(self.labels_values.values(), start=1):
            self.layout.addWidget(lab_val[0], i, 0, CENTER)
            self.layout.addWidget(lab_val[1], i, 1, CENTER)
        self.layout.addWidget(self.locked_label, len(self.labels_values)+2, 0, CENTER)
        self.layout.addWidget(self.locked_value, len(self.labels_values)+2, 1, CENTER)
        self.layout.addWidget(self.init_button, len(self.labels_values)+3, 0, CENTER)
        self.layout.addWidget(self.lock_button, len(self.labels_values)+3, 1, CENTER)
        self.layout.addWidget(self.unlock_button, len(self.labels_values)+3, 2, CENTER)
        self.layout.addWidget(self.reset_button, len(self.labels_values)+3, 3, CENTER)
        self.init_widget()

        self.worker = CrispWorker(
            queue_manager=self.queue_manager,
            labels_values=self.labels_values,
            init_button=self.init_button,
            lock_button=self.lock_button,
            unlock_button=self.unlock_button,
            locked_value=self.locked_value,
        )
        self.signal_init_crisp.connect(self.worker.init_crisp)
        self.signal_lock_crisp.connect(self.worker.lock_crisp)
        self.signal_unlock_crisp.connect(self.worker.unlock_crisp)
        self.signal_reset_config.connect(self.worker.reset_config)
        self.signal_init_crisp_values.connect(self.worker.init_crisp_values)
        self.workers.append(self.worker)
        thread = EvoGUIThread()
        self.worker.moveToThread(thread)
        thread.start()
        self.threads.append(thread)
        self.signal_init_crisp_values.emit()


    def init_crisp(self):
        self.signal_init_crisp.emit(self.cfg_crisp)

    def update_param(self, param_name: str):
        try:
            val = self.get_param(param_name=param_name)
            setattr(self.cfg_crisp, param_name, val)
        except ValueError as e:
            logger.warning(f"focus_update: invalid parameter for key {param_name}: {e}")
            self.labels_values[param_name][1].setText(str(getattr(self.cfg_default, param_name)))

    def get_param(self, param_name: str):
        val = TigerAutofocusConfig.get_attr_from_str(
            attr_name=param_name,
            attr_value_str=self.labels_values[param_name][1].text()
        )
        if not self.cfg_crisp.attr_is_valid(attr_name=param_name, attr_value=val):
            raise ValueError("Check parameter range and type in evomachine.config.")
        return val
