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

sys.path.append(os.path.expanduser('~') + '/workspace_python/conda_evomachine3.9/asitiger')
sys.path.append(os.path.expanduser('~') + '/workspace_python/conda_evomachine3.9/evomachine_repo')

from asitiger.errors import Errors as ASIErrors
from asitiger.tigercontroller import SAFE_STAGE_LIMITS

from evomachine.acquisition import AbstractCamera, EvoCamera
from evomachine.automaton import Automaton, AutomatonQueueDataType
from evomachine.commands import AutomatonCommand, AutomatonCommandType
from evomachine.config import ConfigCRISP, ConfigFocus, get_logger
from evomachine.coordinates import Coordinate, CoordinateFactory
from evomachine.dmd import DMDControl
from evomachine.exceptions import ConfigError, TigerError
from evomachine.evotypes import LEDType, FocusAlgorithmType
from evomachine.guidir.experiments import ExperimentPanel
from evomachine.guidir.figures import FigureMultiWindow, FigureWidget, ImagePlotter
from evomachine.guidir.guitemplates import EvoGUIThread, EvoPanelTemplate, EvoWorkerTemplate, QueueManager
from evomachine.guidir.guitypes import Direction, DMDModes, DisplayMode, SMALL, NORMAL
from evomachine.guidir.position import PositionPanel


logger = get_logger(name=__name__)

AXES = ['X', 'Y', 'Z']
LEFT = Qt.AlignLeft
CENTER = Qt.AlignCenter
RIGHT = Qt.AlignRight

ARROW_LEFT = "\u2190"
ARROW_RIGHT = "\u2192"
ARROW_UP = "\u2191"
ARROW_DOWN = "\u2193"
LRUD = [Direction.LEFT.value, Direction.RIGHT.value, Direction.UP.value, Direction.DOWN.value]
ARROWS = [ARROW_LEFT, ARROW_RIGHT, ARROW_LEFT, ARROW_RIGHT]

stylesheet_led = """
    QPushButton {background-color: red;}
"""


class EvoGUI(QMainWindow):
    def __init__(
            self,
            cam: AbstractCamera,
            dmd: DMDControl,
            data_queue: queue.Queue,
            automaton: Automaton,
            is_testmode: bool = False,
            *args,
            **kwargs
    ):
        super(EvoGUI, self).__init__(*args, **kwargs)
        self.panels: List[EvoPanelTemplate] = []

        # Evomachine Objects
        self.cam: AbstractCamera = cam
        self.dmd: DMDControl = dmd
        self.queue_manager: QueueManager = QueueManager(data_queue=data_queue)
        self.automaton: Automaton = automaton

        self.is_testmode = is_testmode

        # Main Widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Position Panel
        self.pos_panel = PositionPanel(cam=self.cam)
        self.panels.append(self.pos_panel)

        # Experiment Panel
        self.exp_panel = ExperimentPanel(cam=self.cam, automaton=self.automaton, queue_manager=self.queue_manager)
        self.panels.append(self.exp_panel)

        # Figure Panel
        self.fig_panel = ImagePlotter(cam=self.cam, automaton=self.automaton, queue_manager=self.queue_manager)
        self.panels.append(self.exp_panel)

        # Main Layout
        main_layout = QGridLayout()
        main_layout.addWidget(self.pos_panel.widget, 0, 0)
        main_layout.addWidget(self.exp_panel.widget, 0, 1)
        central_widget.setLayout(main_layout)

        self.queue_manager.start()

    def closeEvent(self, event):
        result = QMessageBox.question(self, "Confirm Exit...", "Are you sure you want to exit ?",
                                      QMessageBox.Yes | QMessageBox.No)
        event.ignore()

        if result == QMessageBox.Yes:
            logger.info("closing threads")
            for panel in self.panels:
                panel.close_threads()
            logger.info("closing dmd")
            self.dmd.finalise()
            logger.info("closing automaton")
            self.automaton.stop()
            if self.automaton.is_alive():
                self.automaton.join()
            logger.info("closing cam")
            self.cam.finalise()

            event.accept()


