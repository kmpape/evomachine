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
            else:
                button.setStyleSheet("background-color: red;")


class DMDPanel(EvoPanelTemplate):
    signal_set_dmd = pyqtSignal(int)

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
        self.layout = QGridLayout()
        self.layout.addWidget(self.make_label(text="DMD Control", font=NORMAL), 0, 0, 1, 2, LEFT)
        _ = [self.layout.addWidget(button, 1, i, CENTER) for i, button in enumerate(self.dmd_buttons.values())]
        self.widget = QWidget()
        self.widget.setLayout(self.layout)

        self.worker = DMDWorker(buttons=self.dmd_buttons)
        self.signal_set_dmd.connect(self.worker.set_dmd_states)
        self.workers.append(self.worker)
        thread = EvoGUIThread()
        self.worker.moveToThread(thread)
        thread.start()
        self.threads.append(thread)

    def set_dmd(self, mode: int):
        self.queue_manager.request(
            req_str='self._dmd.display_none' if mode == DMDModes.DISPLAY_NONE.value else 'self._dmd.display_full',
            kwargs_dict={},
        )
        self.signal_set_dmd.emit(mode)
