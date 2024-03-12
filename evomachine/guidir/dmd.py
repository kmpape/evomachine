from multiprocessing import Event, Queue
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
from serial import SerialException

from asitiger.errors import Errors as ASIErrors

from evomachine.acquisition import AbstractCamera, EvoCamera
from evomachine.automaton import Automaton
from evomachine.commands import AutomatonCommand, AutomatonCommandType
from evomachine.config import ConfigCamera, ConfigCRISP, ConfigFocus, ConfigImageProcessor, get_logger
from evomachine.coordinates import Coordinate, CoordinateFactory
from evomachine.dmd import DMDControl, DMDColor
from evomachine.exceptions import ConfigError, TigerError
from evomachine.evotypes import LEDType, FocusAlgorithmType
from evomachine.guidir.guitemplates import EvoPanelTemplate, EvoWorkerTemplate, EvoGUIThread
from evomachine.guidir.guitypes import DMDModes, SMALL, CENTER, LEFT, RIGHT, NORMAL
from evomachine.guidir.queuemanager import QueueManager


logger = get_logger(name=__name__, is_gui=True)


class DMDWorker(EvoWorkerTemplate):
    def __init__(
            self,
            buttons: Dict[int, QPushButton],
            parent: Optional[QObject] = None,
    ):
        super().__init__(parent)
        self.buttons = buttons

    @pyqtSlot(int)
    def set_dmd_states(self, i_active: int):
        for i, button in self.buttons.items():
            if i == i_active:
                button.setStyleSheet("background-color: green;")
            elif i in [DMDModes.DISPLAY_NONE.value, DMDModes.DISPLAY_FULL.value]:
                button.setStyleSheet("background-color: red;")

    @pyqtSlot()
    def dmd_click_start(self):
        for button in self.buttons.values():
            button.setEnabled(False)

    @pyqtSlot()
    def dmd_click_stop(self):
        for button in self.buttons.values():
            button.setEnabled(True)


class DMDPanel(EvoPanelTemplate):
    signal_set_dmd = pyqtSignal(int)
    signal_dmd = pyqtSignal()
    signal_dmd_done = pyqtSignal()

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
        self.dmd_buttons = {i: self.make_button(
            text=txt,
            func=self.set_dmd,
            font=SMALL,
            mode=i,
            stylesheet="QPushButton {background-color: red;}",
        ) for i, txt in zip([DMDModes.DISPLAY_NONE.value, DMDModes.DISPLAY_FULL.value], ["NONE", "FULL"])}
        self.dmd_buttons[DMDModes.DISPLAY_NONE.value].setStyleSheet("background-color: green;")
        self.dmd_init_button = self.make_button(
            text="Initialise",
            func=self.initialise_dmd,
            font=SMALL,
        )
        self.dmd_finalise_button = self.make_button(
            text="Finalise",
            func=self.finalise_dmd,
            font=SMALL,
        )
        self.layout = QGridLayout()
        self.layout.addWidget(self.make_label(text="DMD Control", font=NORMAL), 0, 0, 1, 1, LEFT)
        _ = [self.layout.addWidget(button, 1, i, CENTER) for i, button in enumerate(self.dmd_buttons.values())]
        self.layout.addWidget(self.dmd_init_button, 1, 2, 1, 1, CENTER)
        self.layout.addWidget(self.dmd_finalise_button, 1, 3, 1, 1, CENTER)
        self.widget = QWidget()
        self.widget.setLayout(self.layout)

        self.dmd_buttons[max(list(self.dmd_buttons.keys()))+1] = self.dmd_init_button
        self.dmd_buttons[max(list(self.dmd_buttons.keys()))+1] = self.dmd_finalise_button
        self.worker = DMDWorker(buttons=self.dmd_buttons)
        self.signal_set_dmd.connect(self.worker.set_dmd_states)
        self.signal_dmd.connect(self.worker.dmd_click_start)
        self.signal_dmd_done.connect(self.worker.dmd_click_stop)
        self.workers.append(self.worker)
        thread = EvoGUIThread()
        self.worker.moveToThread(thread)
        thread.start()
        self.threads.append(thread)

    def set_dmd(self, mode: int):
        self.queue_manager.request(
            req_str='self._dmd.display_none' if mode == DMDModes.DISPLAY_NONE.value else 'self._dmd.display_full',
            kwargs_dict={},
            callback=self.show_dmd_done,
        )
        self.signal_dmd.emit()
        self.signal_set_dmd.emit(mode)

    def initialise_dmd(self):
        self.queue_manager.request(
            req_str='self._dmd.initialise',
            kwargs_dict={},
            callback=self.show_dmd_done,
        )
        self.signal_dmd.emit()

    def finalise_dmd(self):
        self.queue_manager.request(
            req_str='self._dmd.finalise',
            kwargs_dict={},
            callback=self.show_dmd_done,
        )
        self.signal_dmd.emit()

    def show_dmd_done(self, data: Any):
        self.signal_dmd_done.emit()
