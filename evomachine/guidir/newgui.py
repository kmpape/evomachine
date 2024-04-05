import copy
import cv2
from enum import Enum
import glob
import logging
import matplotlib.pyplot as plt
from multiprocessing import Event, Lock, Process, Queue
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


sys.path.append(os.path.expanduser('~') + '/workspace_python/conda_evomachine3.9/de-lta-rt')
sys.path.append(os.path.expanduser('~') + '/workspace_python/conda_evomachine3.9/asitiger')
sys.path.append(os.path.expanduser('~') + '/workspace_python/conda_evomachine3.9/evomachine_repo')


from evomachine.config import ConfigCamera, ConfigImageProcessor, get_logger
from evomachine.guidir.crisp import CRISPPanel
from evomachine.guidir.dmd import DMDPanel
from evomachine.guidir.experiments import ExperimentPanel
from evomachine.guidir.figures import FigureMultiWindow, ImagePlotter
from evomachine.guidir.filterwheel import FilterWheelPanel
from evomachine.guidir.focus import FocusPanel
from evomachine.guidir.guitemplates import EvoGUIThread, EvoPanelTemplate, EvoWorkerTemplate, EVO_STYLE
from evomachine.guidir.guitypes import Direction, DMDModes, DisplayMode, SMALL, NORMAL
from evomachine.guidir.position import PositionPanel
from evomachine.guidir.leds import LEDPanel
from evomachine.guidir.queuemanager import QueueManager


logger = get_logger(name=__name__, is_gui=True)

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
            queue_manager: QueueManager,
            camera_config: ConfigCamera,
            processor_config: ConfigImageProcessor,
            start_strategy_event: Event,
            stop_strategy_event: Event,
            stop_event: Event,
            shutdown_event: Event,
            is_testmode: bool = False,
            *args,
            **kwargs
    ):
        super(EvoGUI, self).__init__(*args, **kwargs)
        self.panels: List[EvoPanelTemplate] = []

        self.queue_manager: QueueManager = queue_manager
        self.camera_config: ConfigCamera = camera_config
        self.processor_config: ConfigImageProcessor = processor_config
        self.start_strategy_event: Event = start_strategy_event
        self.stop_strategy_event: Event = stop_strategy_event
        self.stop_event: Event = stop_event
        self.shutdown_event: Event = shutdown_event

        self.is_testmode = is_testmode

        # Main Widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Position Panel
        self.pos_panel = PositionPanel(
            queue_manager=queue_manager,
            camera_config=camera_config,
            processor_config=processor_config,
            start_strategy_event=start_strategy_event,
            stop_strategy_event=stop_strategy_event,
            stop_event=stop_event,
            shutdown_event=shutdown_event,
        )
        self.panels.append(self.pos_panel)

        # Experiment Panel
        self.exp_panel = ExperimentPanel(
            queue_manager=queue_manager,
            camera_config=camera_config,
            processor_config=processor_config,
            start_strategy_event=start_strategy_event,
            stop_strategy_event=stop_strategy_event,
            stop_event=stop_event,
            shutdown_event=shutdown_event,
        )
        self.panels.append(self.exp_panel)

        # Figure Panel
        self.fig_panel = ImagePlotter(
            queue_manager=queue_manager,
            camera_config=camera_config,
            processor_config=processor_config,
            start_strategy_event=start_strategy_event,
            stop_strategy_event=stop_strategy_event,
            stop_event=stop_event,
            shutdown_event=shutdown_event,
        )
        self.panels.append(self.fig_panel)

        # Figure Panel
        self.focus_panel = FocusPanel(
            queue_manager=queue_manager,
            camera_config=camera_config,
            processor_config=processor_config,
            start_strategy_event=start_strategy_event,
            stop_strategy_event=stop_strategy_event,
            stop_event=stop_event,
            shutdown_event=shutdown_event,
        )
        self.panels.append(self.focus_panel)

        # LED Panel
        self.led_panel = LEDPanel(
            queue_manager=queue_manager,
            camera_config=camera_config,
            processor_config=processor_config,
            start_strategy_event=start_strategy_event,
            stop_strategy_event=stop_strategy_event,
            stop_event=stop_event,
            shutdown_event=shutdown_event,
        )
        self.panels.append(self.led_panel)

        # FilterWheelPanel Panel
        self.fw_panel = FilterWheelPanel(
            queue_manager=queue_manager,
            camera_config=camera_config,
            processor_config=processor_config,
            start_strategy_event=start_strategy_event,
            stop_strategy_event=stop_strategy_event,
            stop_event=stop_event,
            shutdown_event=shutdown_event,
        )
        self.panels.append(self.fw_panel)

        # DMD Panel
        self.dmd_panel = DMDPanel(
            queue_manager=queue_manager,
            camera_config=camera_config,
            processor_config=processor_config,
            start_strategy_event=start_strategy_event,
            stop_strategy_event=stop_strategy_event,
            stop_event=stop_event,
            shutdown_event=shutdown_event,
        )
        self.panels.append(self.dmd_panel)

        # CRISP Panel
        self.crisp_panel = CRISPPanel(
            queue_manager=queue_manager,
            camera_config=camera_config,
            processor_config=processor_config,
            start_strategy_event=start_strategy_event,
            stop_strategy_event=stop_strategy_event,
            stop_event=stop_event,
            shutdown_event=shutdown_event,
        )
        self.panels.append(self.crisp_panel)

        # Connect Signals
        self.fig_panel.worker.cropping_box_drawn.connect(self.exp_panel.update_cropping_boxes)
        self.led_panel.signal_set_led.connect(self.fig_panel.update_led)

        # Main Layout
        main_layout = QGridLayout()

        r, c = 0, 0
        main_layout.addWidget(self.pos_panel.widget, r, c, 2, 5)
        r, c = r+2, c
        main_layout.addWidget(self.led_panel.widget, r, c, 2, 5)
        r, c = r+2, c
        main_layout.addWidget(self.dmd_panel.widget, r, c, 2, 5)
        r, c = r+2, c
        main_layout.addWidget(self.crisp_panel.widget, r, c, 2, 5)

        r, c = 0, 5
        main_layout.addWidget(self.fig_panel.widget, r, c, 10, 4)

        r, c = 0, c+4
        main_layout.addWidget(self.exp_panel.widget, 0, 9, 4, 4)
        r, c = r+4, c
        main_layout.addWidget(self.fw_panel.widget, 4, 9, 2, 5)
        r, c = r+2, c
        main_layout.addWidget(self.focus_panel.widget, 6, 9, 3, 5)

        central_widget.setLayout(main_layout)

        self.setStyleSheet(EVO_STYLE)

    def closeEvent(self, event):
        result = QMessageBox.question(
            self,
            "Confirm Exit...",
            "Are you sure you want to exit?\nShutting down will take about 5s, so don't panic on the titanic!",
            QMessageBox.Yes | QMessageBox.No,
        )
        event.ignore()

        if result == QMessageBox.Yes:
            # self.crisp_panel.signal_unlock_crisp.emit()
            # time.sleep(1)
            self.stop_event.set()
            self.stop_strategy_event.set()
            self.start_strategy_event.set()
            self.shutdown_event.set()
            time.sleep(5)  # This is needed, otherwise, peripherals are not shut down properly
            logger.debug("closing threads")
            for panel in self.panels:
                panel.close_threads()
            event.accept()


