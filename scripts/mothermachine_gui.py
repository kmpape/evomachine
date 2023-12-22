import copy
import cv2
from enum import Enum
import glob
import logging
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patches as patches
import numpy as np
import os
import PIL
import re
from serial import SerialException
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from PyQt5.QtCore import pyqtSignal, pyqtSlot, QEventLoop, QThread, QTimer, QObject, QRegExp, Qt
from PyQt5.QtGui import QRegExpValidator, QDoubleValidator, QFont, QPalette, QColor
from PyQt5.QtWidgets import (
    QWidget,
    QMainWindow, QApplication,
    QLabel, QLineEdit, QPushButton, QComboBox,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QSizePolicy, QScrollArea, QFileDialog, QCheckBox
)

import sys
sys.path.append("/home/hslab/workspace_python/conda_evomachine3.9/asitiger")
sys.path.append("/home/hslab/workspace_python/conda_evomachine3.9/evomachine_repo")

from asitiger.errors import Errors as ASIErrors
from asitiger.tigercontroller import SAFE_STAGE_LIMITS
from evomachine.acquisition import EvoCamera, TestCamera
from evomachine.config import ConfigCRISP, ConfigFocus, ConfigFocusAlgorithm, DEVICE_CONFIG_EVO_TEST, \
    CRISP_CONFIG_DEFAULT, FOCUS_CONFIG_DEFAULT, IMAGE_CONFIG_DEFAULT, \
    OBJECTIVE_CONFIG_AIR, ConfigLED, EVO_FORMATTER, OBJECTIVE_CONFIG_OIL, CRISP_CONFIG_OIL
from evomachine.dmd import DMDControl
from evomachine.exceptions import ConfigError, TigerError


logger = logging.getLogger(__name__)
for handler in logger.handlers:
    logger.removeHandler(handler)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(EVO_FORMATTER)
logger.addHandler(handler)
logger.propagate = False

AXES = ['X', 'Y', 'Z']
SMALL = QFont("Arial", 10)
NORMAL = QFont("Arial", 12)
LEFT = Qt.AlignLeft
CENTER = Qt.AlignCenter
RIGHT = Qt.AlignRight

ARROW_LEFT = "\u2190"
ARROW_RIGHT = "\u2192"
ARROW_UP = "\u2191"
ARROW_DOWN = "\u2193"


class Direction(Enum):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3
    HOME = 4
    MOVETO = 5

    @classmethod
    def get_all_values(cls) -> List[int]:
        return [member.value for member in cls]


class DMDModes(Enum):
    DISPLAY_NONE = 0
    DISPLAY_FULL = 1


class DisplayMode(Enum):
    NO_CROP = 0
    SHOW_FRAME = 1
    CROP = 2
    UNKNOWN = 3

    @classmethod
    def get_all_values(cls) -> List[int]:
        return [member.value for member in cls]

    def get_string(self) -> str:
        if self == DisplayMode.NO_CROP:
            return "No Crop"
        elif self == DisplayMode.SHOW_FRAME:
            return "Show Frame(s)"
        elif self == DisplayMode.CROP:
            return "Crop"
        else:
            return "Unknown"

    @classmethod
    def from_string(cls, s: str) -> 'DisplayMode':
        for member in cls:
            if member.get_string() == s:
                return member
        return cls.UNKNOWN



LRUD = [Direction.LEFT.value, Direction.RIGHT.value, Direction.UP.value, Direction.DOWN.value]
ARROWS = [ARROW_LEFT, ARROW_RIGHT, ARROW_LEFT, ARROW_RIGHT]

stylesheet_led = """
    QPushButton {background-color: red;}
"""

class EvoGUI(QMainWindow):
    update_signal_pic_clear_readin = pyqtSignal(str)
    update_signal_pic_show_boxes = pyqtSignal()

    def __init__(self, cam: EvoCamera, dmd: DMDControl, is_testmode: bool = False, *args, **kwargs):
        super(EvoGUI, self).__init__(*args, **kwargs)

        # Evomachine Objects
        self.cam = cam
        self.dmd = dmd

        self.is_testmode = is_testmode

        # Main Widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Position Panel
        self.pos_widget = self.make_position_panel()

        # DMD Panel
        self.dmd_widget = self.make_dmd_panel()

        # LED Panel
        self.led_widget = self.make_led_panel()

        # CRISP Panel
        self.crisp_widget = self.make_crisp_panel()

        # Picture Panel
        self.pic_widget = self.make_picture_panel(central_widget)

        # Multi Acquisition
        self.multi_param_widget = self.make_multi_acquisition_panel()

        # Experiment panel readin buttons
        self.exp_widget = self.make_experiment_panel()

        # Save path configuration
        self.savecfg_widget = self.make_save_config_panel()

        # Software Focus Panel
        self.swfocus_widget = self.make_software_focus_panel()

        # Main Layout
        main_layout = QGridLayout()
        main_layout.addWidget(self.pos_widget, 0, 0)
        main_layout.addWidget(self.dmd_widget, 1, 0)
        main_layout.addWidget(self.led_widget, 2, 0)
        main_layout.addWidget(self.crisp_widget, 3, 0)
        main_layout.addWidget(self.savecfg_widget, 4, 0)
        main_layout.addWidget(self.pic_widget, 0, 1, 4, 1)
        main_layout.addWidget(self.multi_param_widget, 4, 1)
        main_layout.addWidget(self.swfocus_widget, 5, 0)
        main_layout.addWidget(self.exp_widget, 5, 1)

        self.mpl_canvas.plot_image()

        central_widget.setLayout(main_layout)

    def make_button(
            self,
            text: str,
            func: Callable,
            font: QFont = NORMAL,
            param: Any = None,
            stylesheet: str = None,
    ) -> QPushButton:
        button = QPushButton(text, self)
        if param is None:
            button.clicked.connect(func)
        else:
            button.clicked.connect(lambda: func(param))
        button.setFont(font)
        button.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        button.setMinimumSize(button.sizeHint())
        if stylesheet is not None:
            button.setStyleSheet(stylesheet)
        return button

    def make_dropdown(
            self,
            items: List[str],
            func: Optional[Union[Callable, None]]=None,
    ):
        dropdown = QComboBox(self)
        dropdown.addItems(items)
        if func:
            dropdown.currentIndexChanged.connect(func)
        return dropdown

    @staticmethod
    def make_lineedit(
            text: Union[str, None],
            func: Optional[Callable] = None,
            param: Optional[Any] = None,
    ) -> QLineEdit:
        lineedit = QLineEdit()
        if func is not None:
            if param is None:
                lineedit.returnPressed.connect(func)
            else:
                lineedit.returnPressed.connect(lambda: func(param))
        if text is not None:
            lineedit.setText(text)
        return lineedit

    @staticmethod
    def make_label(
            text: str,
            width_px: Union[int, None] = None,
            font: QFont = NORMAL,
            align: int = Qt.AlignCenter,
            stylesheet: Union[str, None] = None,
    ) -> QLabel:
        label = QLabel()
        label.setText(text)
        label.setAlignment(align)
        label.setFont(font)
        if width_px is not None:
            label.setFixedWidth(width_px)
        if stylesheet is not None:
            label.setStyleSheet(stylesheet)
        return label

    def make_checkbox(
            self,
            text: str,
            func: Callable,
            font: QFont = NORMAL,
            param: Any = None,
            set_true: bool = False,
            stylesheet: str = None,
    ):
        checkbox = QCheckBox(text, self)
        checkbox.setChecked(set_true)
        if param is None:
            checkbox.stateChanged.connect(func)
        else:
            checkbox.stateChanged.connect(lambda: func(param))
        checkbox.setFont(font)
        if stylesheet is not None:
            checkbox.setStyleSheet(stylesheet)
        return checkbox

    @staticmethod
    def make_pos_str(value: Union[int, None]) -> str:
        try:
            return f"+{abs(value):06}" if value > 0 else f"-{abs(value):06}"
        except TypeError as e:
            return "?"*7

    def make_crisp_panel(self) -> QWidget:
        self.cfg_crisp = copy.copy(self.cam.cfg_crisp)
        self.cfg_crisp_default = copy.copy(self.cam.cfg_crisp)
        self.crisp_labels_values = {
            'averaging': [EvoGUI.make_label(text="#Samples averaged [ms]", font=SMALL),
                          EvoGUI.make_lineedit(text=str(int(self.cfg_crisp.averaging)),
                                               func=self.crisp_update, param='averaging')],
            'led_intensity': [EvoGUI.make_label(text="LED intensity [1,100]", font=SMALL),
                              EvoGUI.make_lineedit(text=str(int(self.cfg_crisp.led_intensity)),
                                                   func=self.crisp_update, param='led_intensity')],
            'lock_range': [EvoGUI.make_label(text="Lock range [mm]", font=SMALL),
                           EvoGUI.make_lineedit(text=str(float(self.cfg_crisp.lock_range)),
                                                func=self.crisp_update, param='lock_range')],
            'loop_gain': [EvoGUI.make_label(text="Loop gain [1,100]", font=SMALL),
                          EvoGUI.make_lineedit(text=str(int(self.cfg_crisp.loop_gain)),
                                               func=self.crisp_update, param='loop_gain')],
            'objective_na': [EvoGUI.make_label(text="NA (0,Inf)", font=SMALL),
                             EvoGUI.make_lineedit(text=str(float(self.cfg_crisp.objective_na)),
                                                  func=self.crisp_update, param='objective_na')],
            'update_rate': [EvoGUI.make_label(text="Update rate [ms]]", font=SMALL),
                            EvoGUI.make_lineedit(text=str(int(self.cfg_crisp.update_rate)),
                                                 func=self.crisp_update, param='update_rate')]
        }
        self.crisp_locked_label = EvoGUI.make_label(text="Locked", font=SMALL, width_px=100)
        self.crisp_locked_value = EvoGUI.make_label(
            text="No",
            font=SMALL,
            width_px=100,
            stylesheet="background-color: red;",
        )
        self.crisp_enable_crisp_button = self.make_button(text="Enable", font=SMALL, func=self.start_crisp)
        self.crisp_disable_crisp_button = self.make_button(text="Disable", font=SMALL, func=self.end_crisp)
        self.crisp_reset_button = self.make_button(text="Reset", font=SMALL, func=self.crisp_reset)
        self.crisp_thread: Union[ThreadStartCRISP, None] = None
        self.crisp_layout = QGridLayout()
        self.crisp_layout.addWidget(EvoGUI.make_label(text="CRISP Control", font=NORMAL), 0, 0, 1, 3, LEFT)
        for i, lab_val in enumerate(self.crisp_labels_values.values(), start=1):
            self.crisp_layout.addWidget(lab_val[0], i, 0, CENTER)
            self.crisp_layout.addWidget(lab_val[1], i, 1, CENTER)
        self.crisp_layout.addWidget(self.crisp_locked_label, len(self.crisp_labels_values)+2, 0, CENTER)
        self.crisp_layout.addWidget(self.crisp_locked_value, len(self.crisp_labels_values)+2, 1, CENTER)
        self.crisp_layout.addWidget(self.crisp_enable_crisp_button, len(self.crisp_labels_values)+3, 0, CENTER)
        self.crisp_layout.addWidget(self.crisp_disable_crisp_button, len(self.crisp_labels_values)+3, 1, CENTER)
        self.crisp_layout.addWidget(self.crisp_reset_button, len(self.crisp_labels_values)+3, 2, CENTER)
        crisp_widget = QWidget()
        crisp_widget.setLayout(self.crisp_layout)
        self.crisp_thread: Union[ThreadStartCRISP, None] = None
        self.crisp_reset_thread: Union[None, ThreadConfigReset] = None
        return crisp_widget

    def make_dmd_panel(self) -> QWidget:
        self.dmd_buttons = {i: self.make_button(
            text=txt,
            func=self.set_dmd,
            font=SMALL,
            param=i,
            stylesheet=stylesheet_led,
        ) for i, txt in zip([DMDModes.DISPLAY_NONE.value, DMDModes.DISPLAY_FULL.value], ["NONE", "FULL"])}
        self.dmd_buttons[DMDModes.DISPLAY_NONE.value].setStyleSheet("background-color: green;")
        self.dmd_layout = QGridLayout()
        self.dmd_layout.addWidget(EvoGUI.make_label(text="DMD Control", font=NORMAL), 0, 0, 1, 2, LEFT)
        _ = [self.dmd_layout.addWidget(button, 1, i, CENTER) for i, button in enumerate(self.dmd_buttons.values())]
        dmd_widget = QWidget()
        dmd_widget.setLayout(self.dmd_layout)
        self.dmd_thread: Union[ThreadDMD, None] = None
        return dmd_widget

    def make_experiment_panel(self):
        self.exp_num_read_ins = 3
        self.exp_readin_buttons = {i: {
            "from": self.make_button(text="From", func=self.exp_record_param, font=SMALL, param=(i, "from")),
            "to": self.make_button(text="To", func=self.exp_record_param, font=SMALL, param=(i, "to"))
        } for i in range(self.exp_num_read_ins)}
        self.exp_readin_display = {i: {
            "from": self.make_label(text="?", font=SMALL),
            "to": self.make_label(text="?", font=SMALL)
        } for i in range(self.exp_num_read_ins)}
        self.exp_readin_positions = {i: {"from": None, "to": None} for i in range(self.exp_num_read_ins)}
        self.exp_readin_label = {i: self.make_label(text=f"Path {i}", font=SMALL)
                                   for i in range(self.exp_num_read_ins)}
        self.exp_readin_clear_button = self.make_button(text="Clear all", func=self.exp_clear_param, font=SMALL)
        self.exp_clear_thread: Union[ThreadClearReadin, None] = None
        self.exp_start_button = self.make_button(text="Start", func=self.exp_start_acquisition, font=SMALL)
        self.exp_stop_button = self.make_button(text="Stop", func=self.exp_stop_acquisition, font=SMALL)
        self.exp_layout = QGridLayout()
        self.exp_layout.addWidget(EvoGUI.make_label(text="Experiment", font=NORMAL), 0, 0, 1, 3, LEFT)
        for i in range(self.exp_num_read_ins):
            self.exp_layout.addWidget(self.exp_readin_label[i], 2*i+1, 0, 1, 1)
            self.exp_layout.addWidget(self.exp_readin_buttons[i]["from"], 2*i+1, 1, 1, 1)
            self.exp_layout.addWidget(self.exp_readin_buttons[i]["to"], 2*i+1, 2, 1, 1)
            self.exp_layout.addWidget(self.exp_readin_display[i]["from"], 2*i+2, 1, 1, 1)
            self.exp_layout.addWidget(self.exp_readin_display[i]["to"], 2*i+2, 2, 1, 1)
        self.exp_layout.addWidget(self.exp_start_button, 1+2*self.exp_num_read_ins, 0, 1, 1)
        self.exp_layout.addWidget(self.exp_stop_button, 1+2*self.exp_num_read_ins, 1, 1, 1)
        self.exp_layout.addWidget(self.exp_readin_clear_button, 1+2*self.exp_num_read_ins, 2, 1, 1)
        exp_widget = QWidget()
        exp_widget.setLayout(self.exp_layout)
        self.exp_thread: Union[ThreadExperiment, None] = None
        return exp_widget

    def make_led_panel(self) -> QWidget:
        self.current_led_id: int = ConfigLED.LED_NO_LED.value
        self.led_labels = [EvoGUI.make_label(
            text=ConfigLED.get_name(value_to_find=i),
            font=SMALL,
            width_px=100,
        ) for i in self.cam.channel_settings.keys()]
        self.led_buttons = {i: self.make_button(
            text="OFF",
            func=self.set_led,
            font=SMALL,
            param=i,
            stylesheet=stylesheet_led,
        ) for i in self.cam.channel_settings.keys()}
        self.led_buttons[ConfigLED.LED_NO_LED.value].setStyleSheet("background-color: green;")
        self.led_buttons[ConfigLED.LED_NO_LED.value].setText("ON")
        self.led_layout = QGridLayout()
        self.led_layout.addWidget(EvoGUI.make_label(text="LED Control", font=NORMAL), 0, 0, 1, 2, LEFT)
        _ = [self.led_layout.addWidget(label, i, 0, CENTER) for i, label in enumerate(self.led_labels, start=1)]
        _ = [self.led_layout.addWidget(button, i, 1, CENTER) for i, button in enumerate(self.led_buttons.values(), start=1)]
        led_widget = QWidget()
        led_widget.setLayout(self.led_layout)
        self.led_thread: Union[ThreadLED, None] = None
        return led_widget

    def make_multi_acquisition_panel(self) -> QWidget:
        self.current_multi_param = {'X': 0, 'Y': 0, 'LED': [-1]}
        self.multi_param_labels = [EvoGUI.make_label(
            text=txt,
            font=SMALL,
            width_px=100,
        ) for txt in self.current_multi_param.keys()]
        dummy_params = ["1", "-2", "1,2"]
        self.multi_param_lineedits = {i: self.make_lineedit(
            text=t,
            func=self.on_enter_pressed_multi_param,
            param=i,
        ) for i, t in zip(self.current_multi_param.keys(), dummy_params)}
        self.multi_param_button = self.make_button(text="Acquire", func=self.multi_dim_acquisition, font=SMALL)
        self.multi_param_layout = QGridLayout()
        self.multi_param_layout.addWidget(EvoGUI.make_label(text="Multi Acquisition", font=NORMAL), 0, 0, 1, 2, LEFT)
        _ = [self.multi_param_layout.addWidget(lab, 1+i, 0, CENTER) for i, lab in enumerate(self.multi_param_labels)]
        _ = [self.multi_param_layout.addWidget(ed, 1+i, 1, CENTER)
             for i, ed in enumerate(self.multi_param_lineedits.values())]
        self.multi_param_layout.addWidget(self.multi_param_button, 4, 0, 1, 2)
        multi_param_widget = QWidget()
        multi_param_widget.setLayout(self.multi_param_layout)
        self.multi_param_thread: Union[ThreadMultiParam, None] = None
        return multi_param_widget

    def make_picture_panel(self, central_widget: QWidget) -> QWidget:
        self.pic_num_read_ins = 4
        self.pic_readins = {i: {
            "tl": self.make_lineedit(text="[0,0]", func=self.pic_record_param, param=(i, "tl")),
            "br": self.make_lineedit(text="[0,0]", func=self.pic_record_param, param=(i, "br"))
        } for i in range(self.pic_num_read_ins)}
        self.pic_readin_positions = {i: {"tl": [0, 0], "br": [0, 0]} for i in range(self.pic_num_read_ins)}
        self.pic_readin_label = {i: self.make_label(text=f"Cropping Box {i}", font=SMALL)
                                 for i in range(self.pic_num_read_ins)}
        self.pic_readin_clear_button = self.make_button(text="Clear all", func=self.pic_clear_param, font=SMALL)
        self.update_signal_pic_clear_readin.connect(self.on_update_signal_pic_clear_readin)
        self.pic_show_boxes_button = self.make_button(text="Show boxes", func=self.pic_show_boxes, font=SMALL)
        self.update_signal_pic_show_boxes.connect(self.on_update_signal_pic_show_boxes)
        self.pic_crop_dropdown_options = [
            DisplayMode.NO_CROP.get_string(), DisplayMode.SHOW_FRAME.get_string(), DisplayMode.CROP.get_string()
        ]
        self.pic_crop_current_option = DisplayMode.NO_CROP
        self.pic_crop_dropdown = self.make_dropdown(items=self.pic_crop_dropdown_options,
                                                    func=self.pic_update_crop_option)

        self.current_exposure: int = int(self.cam.cfg_focus.exposure_time)
        self.current_normalise_frame: bool = True
        self.pic_exposure_label = EvoGUI.make_label(text="Exposure [ms]", font=SMALL, width_px=100)
        self.pic_exposure_value = EvoGUI.make_label(text=f"{self.cam.cfg_focus.exposure_time}", font=SMALL, width_px=100)
        self.pic_exposure_edit = QLineEdit()
        self.pic_exposure_edit.returnPressed.connect(self.on_enter_pressed_exposure)
        self.pic_take_frame_label = self.make_label(text="Single frame", font=SMALL)
        self.pic_take_frame_button = self.make_button(text="Take Frame", font=SMALL, func=self.take_frame)
        self.pic_live_frame_label = self.make_label(text="Live mode", font=SMALL)
        self.pic_live_frame_start_button = self.make_button(text="Start", font=SMALL, func=self.start_live_mode)
        self.pic_live_frame_stop_button = self.make_button(text="Stop", font=SMALL, func=self.stop_live_mode)
        self.pic_normalise_checkbox = self.make_checkbox(
            text="Normalise",
            font=SMALL,
            set_true=self.current_normalise_frame,
            func=self.toggle_normalise_frame,
        )
        self.mpl_canvas = FigureWidget(central_widget, width=10, height=10, dpi=100)
        self.mpl_canvas.setMinimumSize(400, 300)
        self.pic_layout = QGridLayout()
        self.pic_layout.addWidget(EvoGUI.make_label(text="Frame Control", font=NORMAL), 0, 0, 1, 3, LEFT)
        self.pic_layout.addWidget(self.pic_exposure_label, 1, 0, CENTER)
        self.pic_layout.addWidget(self.pic_exposure_value, 1, 1, CENTER)
        self.pic_layout.addWidget(self.pic_exposure_edit, 1, 2, CENTER)
        self.pic_layout.addWidget(self.pic_take_frame_label, 2, 0, CENTER)
        self.pic_layout.addWidget(self.pic_take_frame_button, 2, 1, CENTER)
        self.pic_layout.addWidget(self.pic_normalise_checkbox, 2, 2, CENTER)
        self.pic_layout.addWidget(self.pic_live_frame_label, 3, 0, CENTER)
        self.pic_layout.addWidget(self.pic_live_frame_start_button, 3, 1, CENTER)
        self.pic_layout.addWidget(self.pic_live_frame_stop_button, 3, 2, CENTER)
        self.pic_layout.addWidget(self.pic_crop_dropdown, 4, 0, 1, 1)
        self.pic_layout.addWidget(self.pic_readin_clear_button, 5, 0, 1, 1)
        self.pic_layout.addWidget(self.pic_show_boxes_button, 6, 0, 1, 1)
        for i in range(self.pic_num_read_ins):
            self.pic_layout.addWidget(self.pic_readin_label[i], 4+i, 1, CENTER)
            self.pic_layout.addWidget(self.pic_readins[i]["tl"], 4+i, 2, CENTER)
            self.pic_layout.addWidget(self.pic_readins[i]["br"], 4+i, 3, CENTER)
        self.pic_layout.addWidget(self.mpl_canvas, 4 + self.pic_num_read_ins, 0, 1, 4)
        pic_widget = QWidget()
        pic_widget.setLayout(self.pic_layout)
        self.live_mode_thread: Union[ThreadLiveMode, None] = None
        return pic_widget

    def make_position_panel(self) -> QWidget:
        self.current_moveto = {'X': 0, 'Y': 0}
        self.pos_labels = [EvoGUI.make_label(
            text=f"{ax}",
            font=SMALL,
            width_px=20,
        ) for ax in AXES]
        self.pos_values = [EvoGUI.make_label(
            text=EvoGUI.make_pos_str(None),
            font=SMALL,
            width_px=80,
            align=RIGHT,
        ) for _ in AXES]
        self.pos_update_button = self.make_button(text="Update", func=self.update_position, font=SMALL)
        self.pos_halt_button = self.make_button(text="Halt", func=self.halt_position, font=SMALL)
        self.pos_home_button = self.make_button(
            text="Home",
            func=self.move_thread,
            font=SMALL,
            param=Direction.HOME.value,
        )
        self.pos_zero_button = self.make_button(text="Zero", func=self.zero_position, font=SMALL)
        self.pos_lrud_buttons = [
            self.make_button(text=ARROWS[i], func=self.move_thread, font=SMALL, param=i) for i in LRUD
        ]
        self.pos_move_lineedits = {key: self.make_lineedit(text=str(self.current_moveto[key])) for key in ['X', 'Y']}
        self.pos_move_button = self.make_button(
            text="Move to",
            func=self.move_thread,
            font=SMALL,
            param=Direction.MOVETO.value,
        )
        self.pos_layout = QGridLayout()
        self.pos_layout.addWidget(EvoGUI.make_label(text="Stage Control", font=NORMAL), 0, 0, 1, 4, LEFT)
        _ = [self.pos_layout.addWidget(pos_label, i, 0, CENTER) for i, pos_label in enumerate(self.pos_labels, start=1)]
        _ = [self.pos_layout.addWidget(pos_value, i, 1, CENTER) for i, pos_value in enumerate(self.pos_values, start=1)]
        self.pos_layout.addWidget(self.pos_update_button, 4, 0, 1, 1)
        self.pos_layout.addWidget(self.pos_home_button, 4, 1, 1, 1)
        self.pos_layout.addWidget(self.pos_halt_button, 4, 2, 1, 1)
        self.pos_layout.addWidget(self.pos_zero_button, 4, 3, 1, 1)
        _ = [self.pos_layout.addWidget(self.pos_lrud_buttons[i], int(i/2) + 1, (i % 2) + 2) for i in LRUD]
        _ = [self.pos_layout.addWidget(self.pos_move_lineedits[key], i + 1, 4) for i, key in enumerate(['X', 'Y'])]
        self.pos_layout.addWidget(self.pos_move_button, 4, 4, 1, 1)
        _ = [self.pos_layout.setColumnMinimumWidth(i, 0) for i in range(self.pos_layout.columnCount())]
        _ = [self.pos_layout.setColumnStretch(i, 0) for i in range(self.pos_layout.columnCount())]
        self.pos_layout.setHorizontalSpacing(0)
        pos_widget = QWidget()
        pos_widget.setLayout(self.pos_layout)
        self.pos_thread: Union[ThreadPos, None] = None
        return pos_widget

    def make_save_config_panel(self) -> QWidget:
        self.current_use_overwrite_savepath: bool = False
        self.current_overwrite_savepath: str = ""
        self.current_save_figure: bool = True
        self.savecfg_configpath_label = EvoGUI.make_label(text="Config Savepath", font=SMALL)
        self.savecfg_configpath_value = EvoGUI.make_label(text=str(self.cam.cfg_device.path_to_save), font=SMALL)
        self.savecfg_configpath_scroll = QScrollArea()
        self.savecfg_configpath_scroll.setWidgetResizable(True)
        self.savecfg_configpath_scroll.setFixedWidth(300)
        self.savecfg_configpath_scroll.setFixedHeight(40)
        self.savecfg_configpath_scroll.setWidget(self.savecfg_configpath_value)
        self.savecfg_overwrite_label = EvoGUI.make_label(text="Overwrite", font=SMALL)
        self.savecfg_overwrite_lineedit = QLineEdit()
        self.savecfg_overwrite_lineedit.returnPressed.connect(self.on_enter_pressed_savepath)
        savecfg_overwrite_browse = self.make_button(text="Browse", font=SMALL, func=self.browse_savepath)
        savecfg_overwrite_checkbox = self.make_checkbox(
            text="Use Overwrite Path",
            font=SMALL,
            set_true=self.current_use_overwrite_savepath,
            func=self.toggle_current_savepath,
        )
        savecfg_savefig_checkbox = self.make_checkbox(
            text="Save Figures",
            font=SMALL,
            set_true=self.current_save_figure,
            func=self.toggle_save_figure,
        )
        self.savecfg_layout = QGridLayout()
        self.savecfg_layout.addWidget(EvoGUI.make_label(text="Save Config", font=NORMAL), 0, 0, 1, 3, LEFT)
        self.savecfg_layout.addWidget(self.savecfg_configpath_label, 1, 0, CENTER)
        self.savecfg_layout.addWidget(self.savecfg_configpath_scroll, 1, 1, 1, 2)
        self.savecfg_layout.addWidget(self.savecfg_overwrite_label, 2, 0, CENTER)
        self.savecfg_layout.addWidget(self.savecfg_overwrite_lineedit, 2, 1, CENTER)
        self.savecfg_layout.addWidget(savecfg_overwrite_browse, 2, 2, CENTER)
        self.savecfg_layout.addWidget(savecfg_overwrite_checkbox, 3, 0, CENTER)
        self.savecfg_layout.addWidget(savecfg_savefig_checkbox, 3, 1, CENTER)
        savecfg_widget = QWidget()
        savecfg_widget.setLayout(self.savecfg_layout)
        return savecfg_widget

    def make_software_focus_panel(self) -> QWidget:
        self.cfg_focus = self.cam.cfg_focus
        self.cfg_focus_default = copy.copy(self.cam.cfg_focus)
        self.swfocus_labels_values = {
            'exposure_time': [EvoGUI.make_label(text="Exposure [ms]", font=SMALL),
                              EvoGUI.make_lineedit(text=str(int(self.cfg_focus.exposure_time)),
                                                   func=self.swfocus_update, param='exposure_time')],
            'focus_channel': [EvoGUI.make_label(text="Channel number [0,...,3]", font=SMALL),
                              EvoGUI.make_lineedit(text=str(int(self.cfg_focus.focus_channel)),
                                                   func=self.swfocus_update, param='focus_channel')],
            'rel_range': [EvoGUI.make_label(text="Relative range [um/10]", font=SMALL),
                          EvoGUI.make_lineedit(text=str(int(self.cfg_focus.rel_range)),
                                               func=self.swfocus_update, param='rel_range')],
            'steps_size': [EvoGUI.make_label(text="Step Size [um/10]", font=SMALL),
                           EvoGUI.make_lineedit(text=str(int(self.cfg_focus.steps_size)),
                                                func=self.swfocus_update, param='steps_size')],
        }
        self.swfocus_start_button = self.make_button(text="Start", font=SMALL, func=self.swfocus_start)
        self.swfocus_stop_button = self.make_button(text="Stop", font=SMALL, func=self.swfocus_stop)
        self.swfocus_reset_button = self.make_button(text="Reset", font=SMALL, func=self.swfocus_reset)
        self.swfocus_layout = QGridLayout()
        self.swfocus_layout.addWidget(EvoGUI.make_label(text="Software Focus", font=NORMAL), 0, 0, 1, 3, LEFT)
        for i, lab_val in enumerate(self.swfocus_labels_values.values(), start=1):
            self.swfocus_layout.addWidget(lab_val[0], i, 0, CENTER)
            self.swfocus_layout.addWidget(lab_val[1], i, 1, CENTER)
        self.swfocus_layout.addWidget(self.swfocus_start_button, len(self.swfocus_labels_values)+2, 0, CENTER)
        self.swfocus_layout.addWidget(self.swfocus_stop_button, len(self.swfocus_labels_values)+2, 1, CENTER)
        self.swfocus_layout.addWidget(self.swfocus_reset_button, len(self.swfocus_labels_values)+2, 2, CENTER)
        swfocus_widget = QWidget()
        swfocus_widget.setLayout(self.swfocus_layout)
        self.swfocus_start_thread: Union[None, ThreadSWFocus] = None
        self.swfocus_reset_thread: Union[None, ThreadConfigReset] = None
        return swfocus_widget

    def browse_savepath(self):
        savepath = QFileDialog.getExistingDirectory(self, "Select Save Path")
        if savepath:
            self.savecfg_overwrite_lineedit.setText(savepath)
            self.current_overwrite_savepath = savepath

    def toggle_current_savepath(self, state):
        if state == Qt.Checked:
            self.current_use_overwrite_savepath = True
            self.current_overwrite_savepath = self.savecfg_overwrite_lineedit.text()
        else:
            self.current_use_overwrite_savepath = False

    def on_enter_pressed_savepath(self):
        self.current_overwrite_savepath = self.savecfg_overwrite_lineedit.text()

    def toggle_save_figure(self, state):
        if state == Qt.Checked:
            self.current_save_figure = True
        else:
            self.current_save_figure = False

    def on_enter_pressed_exposure(self):
        entered_text = self.pic_exposure_edit.text()
        try:
            self.current_exposure = int(entered_text)
            self.cam.set_exposure(exposure_time=self.current_exposure)
            self.pic_exposure_value.setText(f"{self.current_exposure}")
        except ValueError as e:
            logger.warning(f"EvoGUI.on_enter_pressed_exposure: {e}")

    def toggle_normalise_frame(self, state):
        if state == Qt.Checked:
            self.current_normalise_frame = True
        else:
            self.current_normalise_frame = False

    def _get_savepath(self):
        if self.current_use_overwrite_savepath and self.current_overwrite_savepath:
            path_to_save = self.current_overwrite_savepath
        else:
            path_to_save = str(self.cam.cfg_device.path_to_save)
        return path_to_save if self.current_save_figure else None

    def take_frame(self):
        frame = self.cam.display_save_frame(
            i_chan=self.current_led_id,
            i_period=None,
            path_to_save=self._get_savepath(),
            filename=None,
            display_frame=None,
        )
        if self.current_normalise_frame and frame is not None:
            frame = self.cam.normalise_frame(frame)
        self.mpl_canvas.update_display_mode(
            display_mode=self.pic_crop_current_option,
            cropping_boxes=self.pic_get_cropping_boxes(),
        )
        self.mpl_canvas.plot_image(image_array=frame)

    def halt_position(self):
        try:
            self.cam.halt_stage()
        except (SerialException, ValueError) as e:
            logger.warning(f"EvoGUI.halt_position: {e}")
            time.sleep(0.5)

        self.update_position()

    def update_position(self):
        try:
            pos_dict = self.cam.get_coordinates(AXES)
            _ = [lab.setText(EvoGUI.make_pos_str(pos_dict[ax])) for lab, ax in zip(self.pos_values, AXES)]
        except (SerialException, KeyError) as e:
            logger.warning(f"EvoGUI.update_position: {e}")

    def move_thread(self, i_direction: int):
        self.pos_thread = ThreadPos(
            cam=self.cam,
            pos_values=self.pos_values,
            i_direction=i_direction,
            pos_move_lineedits=self.pos_move_lineedits,
        )
        self.pos_thread.start()

    def exp_record_param(self, which: Tuple[int, str]):
        i, from_to = which
        try:
            pos_dict = self.cam.get_coordinates(AXES)
            pos_str = f"({EvoGUI.make_pos_str(pos_dict['X']), EvoGUI.make_pos_str(pos_dict['Y'])})"
            self.exp_readin_display[i][from_to].setText(pos_str)
            self.exp_readin_positions[i][from_to] = pos_dict
        except (SerialException, KeyError) as e:
            logger.warning(f"EvoGUI.exp_record_param: {e}")
            self.exp_readin_display[i][from_to].setText("?")
            self.exp_readin_positions[i][from_to] = None
        self.exp_readin_label = {i: self.make_label(text=f"Path {i}", font=SMALL)
                                   for i in range(self.exp_num_read_ins)}

    def pic_record_param(self, which: Tuple[int, str]):
        def is_valid_format(txt: str):
            pattern = re.compile(r'\[(\d+),(\d+)\]')
            match = pattern.match(txt)
            if match:
                x, y = map(int, match.groups())
                if 0 <= x <= 3199 and 0 <= y <= 3199:
                    return True
            return False
        i, bl_or_tl = which
        text = self.pic_readins[i][bl_or_tl].text()
        if not is_valid_format(text):
            logger.warning(f"EvoGUI.pic_record_param: Expected format [int, int] "
                           f"with int in [0,3199] but received {text}")
            self.pic_readins[i][bl_or_tl].setText("[0,0]")
            self.pic_readin_positions[i][bl_or_tl] = [0, 0]
        else:
            self.pic_readin_positions[i][bl_or_tl] = [int(x) for x in text.lstrip("[").rstrip("]").split(",")]

    def pic_get_cropping_boxes(self):
        for i in range(self.pic_num_read_ins):
            for k in ["tl", "br"]:
                self.pic_record_param(which=(i, k))
        return {key: [tuple(v) for v in val.values() if v is not None]
                for key, val in self.pic_readin_positions.items()}

    def exp_clear_param(self):
        self.exp_readin_positions = {i: {"from": None, "to": None} for i in range(self.exp_num_read_ins)}
        self.exp_clear_thread = ThreadClearReadin(labels=self.exp_readin_display)
        self.exp_clear_thread.start()

    def pic_clear_param(self):
        self.pic_readin_positions = {i: {"tl": [0, 0], "br": [0, 0]} for i in range(self.pic_num_read_ins)}
        self.update_signal_pic_clear_readin.emit("[0,0]")

    def on_update_signal_pic_clear_readin(self, text):
        for _dict in self.pic_readins.values():
            for edit in _dict.values():
                edit.setText(text)
        self.mpl_canvas.remove_boxes()
        self.mpl_canvas.update_figure()

    def pic_show_boxes(self):
        self.update_signal_pic_show_boxes.emit()

    def on_update_signal_pic_show_boxes(self):
        self.mpl_canvas.update_display_mode(
            display_mode=self.pic_crop_current_option,
            cropping_boxes=self.pic_get_cropping_boxes(),
        )
        self.mpl_canvas.show_boxes()

    def pic_update_crop_option(self):
        self.pic_crop_current_option = DisplayMode.from_string(self.pic_crop_dropdown.currentText())

    def zero_position(self):
        self.cam.zero_coordinates()
        self.update_position()

    def set_dmd(self, display_mode: int):
        if display_mode == DMDModes.DISPLAY_NONE.value:
            self.dmd.display_none()
        elif display_mode == DMDModes.DISPLAY_FULL.value:
            self.dmd.display_full()
        self.dmd_thread = ThreadDMD(buttons=self.dmd_buttons, i_active=display_mode)
        self.dmd_thread.start()

    def set_led(self, i_channel: int):
        if self.is_testmode:
            logger.info("set_led: no LEDs in testmode.")
            return
        self.cam._set_channel(i_chan=i_channel)
        self.current_led_id = i_channel
        self.led_thread = ThreadLED(buttons=self.led_buttons, i_active=i_channel)
        self.led_thread.start()

    def start_live_mode(self):
        self.mpl_canvas.update_display_mode(
            display_mode=self.pic_crop_current_option,
            cropping_boxes=self.pic_get_cropping_boxes(),
        )
        self.live_mode_thread = ThreadLiveMode(
            cam=self.cam,
            mpl_canvas=self.mpl_canvas,
            normalise=self.current_normalise_frame,
            exposure_time=self.current_exposure,
            img_channel=self.current_led_id,
        )
        self.live_mode_thread.started.connect(self.show_live_mode_processing)
        self.live_mode_thread.finished.connect(self.show_live_mode_done)
        self.live_mode_thread.start()

    def show_live_mode_processing(self):
        self.pic_live_frame_start_button.setStyleSheet("background-color: orange;")
        self.pic_live_frame_start_button.setText("ON")

    def show_live_mode_done(self):
        self.pic_live_frame_start_button.setText("Start")
        self.pic_live_frame_start_button.setStyleSheet("background-color: white;")

    def stop_live_mode(self):
        if self.live_mode_thread is not None:
            self.live_mode_thread.stop()

    def start_crisp(self):
        self.crisp_thread = ThreadStartCRISP(cam=self.cam, cfg_crisp=self.cfg_crisp)
        self.crisp_thread.started.connect(self.show_crisp_processing)
        self.crisp_thread.finished.connect(self.show_crisp_done)
        self.crisp_thread.start()

    def show_crisp_processing(self):
        self.crisp_enable_crisp_button.setText("In progress")
        self.crisp_enable_crisp_button.setStyleSheet("background-color: orange;")

    def show_crisp_done(self):
        self.crisp_enable_crisp_button.setText("Enable")
        self.crisp_enable_crisp_button.setStyleSheet("background-color: white;")
        time.sleep(1)
        if self.cam.crisp_is_locked():
            self.crisp_locked_value.setText("Yes")
            self.crisp_locked_value.setStyleSheet("background-color: green;")
        else:
            self.crisp_locked_value.setStyleSheet("background-color: red;")
            self.crisp_locked_value.setText("No")

    def end_crisp(self):
        self.cam.crisp_unlock()
        time.sleep(1)
        if self.cam.crisp_is_locked():
            self.crisp_locked_value.setText("Yes")
            self.crisp_locked_value.setStyleSheet("background-color: green;")
        else:
            self.crisp_locked_value.setStyleSheet("background-color: red;")
            self.crisp_locked_value.setText("No")

    def on_enter_pressed_multi_param(self, key):
        value_str = self.multi_param_lineedits[key].text()
        if key in ['X', 'Y']:
            try:
                self.multi_param_lineedits[key] = int(value_str)
            except ValueError as e:
                logger.warning(f"EvoGUI.on_enter_pressed_multi_param: {e}")
                self.multi_param_lineedits[key].setText(str(self.multi_param_lineedits[key]))
        else:
            try:
                self.multi_param_lineedits[key] = [int(x) for x in value_str.rstrip("]").lstrip("[").split(",")]
            except ValueError as e:
                logger.warning(f"EvoGUI.on_enter_pressed_multi_param: {e}")
                self.multi_param_lineedits[key].setText(",".join([str(x) for x in self.multi_param_lineedits[key]]))

    def multi_dim_acquisition(self):
        self.multi_param_thread = ThreadMultiParam(
            cam=self.cam,
            multi_param_lineedits=self.multi_param_lineedits,
            savepath=self._get_savepath(),
        )
        self.multi_param_thread.started.connect(self.show_multi_param_processing)
        self.multi_param_thread.finished.connect(self.show_multi_param_done)
        self.multi_param_thread.start()

    def show_multi_param_processing(self):
        self.multi_param_button.setStyleSheet("background-color: orange;")

    def show_multi_param_done(self):
        self.multi_param_button.setStyleSheet("background-color: white;")

    def exp_start_acquisition(self):
        self.exp_thread = ThreadExperiment(
            cam=self.cam,
            mpl_canvas=self.mpl_canvas,
            positions=self.exp_readin_positions,
        )
        self.exp_thread.started.connect(self.show_exp_processing)
        self.exp_thread.finished.connect(self.show_exp_done)
        self.exp_thread.start()

    def exp_stop_acquisition(self):
        if self.exp_thread is not None:
            self.exp_thread.stop()

    def show_exp_processing(self):
        self.exp_start_button.setStyleSheet("background-color: orange;")

    def show_exp_done(self):
        self.exp_start_button.setStyleSheet("background-color: white;")

    def swfocus_get_param(self, param_name: str):
        val = ConfigFocus.get_attr_from_str(
            attr_name=param_name,
            attr_value_str=self.swfocus_labels_values[param_name][1].text()
        )
        if not self.cfg_focus.attr_is_valid(attr_name=param_name, attr_value=val):
            raise ValueError("Check parameter range and type in evomachine.config.")
        return val

    def crisp_get_param(self, param_name: str):
        val = ConfigCRISP.get_attr_from_str(
            attr_name=param_name,
            attr_value_str=self.crisp_labels_values[param_name][1].text()
        )
        if not self.cfg_crisp.attr_is_valid(attr_name=param_name, attr_value=val):
            raise ValueError("Check parameter range and type in evomachine.config.")
        return val

    def swfocus_update(self, param_name: str):
        try:
            val = self.swfocus_get_param(param_name=param_name)
            setattr(self.cfg_focus, param_name, val)
        except ValueError as e:
            logger.warning(f"swfocus_update: invalid parameter for key {param_name}: {e}")
            self.swfocus_labels_values[param_name][1].setText(str(getattr(self.cfg_focus_default, param_name)))

    def crisp_update(self, param_name: str):
        try:
            val = self.crisp_get_param(param_name=param_name)
            setattr(self.cfg_crisp, param_name, val)
        except ValueError as e:
            logger.warning(f"focus_update: invalid parameter for key {param_name}: {e}")
            self.crisp_labels_values[param_name][1].setText(str(getattr(self.cfg_crisp_default, param_name)))

    def swfocus_reset(self):
        self.swfocus_reset_thread = ThreadConfigReset(
            this_cfg=self.cfg_focus_default,
            labels_values=self.swfocus_labels_values
        )
        self.cfg_focus = copy.copy(self.cfg_focus_default)
        self.swfocus_reset_thread.start()

    def crisp_reset(self):
        self.crisp_reset_thread = ThreadConfigReset(
            this_cfg=self.cfg_crisp_default,
            labels_values=self.crisp_labels_values
        )
        self.cfg_crisp = copy.copy(self.cfg_crisp_default)
        self.crisp_reset_thread.start()

    def swfocus_start(self):
        if self.is_testmode:
            logger.info("swfocus_start: no swfocus in testmode.")
            return
        try:
            for key in self.swfocus_labels_values.keys():
                self.swfocus_get_param(param_name=key)
        except ValueError as e:
            logging.warning(f"swfocus_start: invalid parameter provided. Aborting. {e}")
            return
        if self.cam.crisp_is_locked():
            logger.info("swfocus_start: unlocking CRISP autofocus before software focus.")
            self.end_crisp()
            if self.cam.crisp_is_locked():
                logger.warning("swfocus_start: unable to unlock CRISP. Aborting.")
                return
        self.swfocus_start_thread = ThreadSWFocus(cam=self.cam, cfg_focus=self.cfg_focus)
        self.swfocus_start_thread.started.connect(self.show_swfocus_processing)
        self.swfocus_start_thread.finished.connect(self.show_swfocus_done)
        self.swfocus_start_thread.start()

    def swfocus_stop(self):
        if self.swfocus_start_thread is not None:
            self.swfocus_start_thread.stop()

    def show_swfocus_processing(self):
        self.swfocus_start_button.setStyleSheet("background-color: orange;")

    def show_swfocus_done(self):
        self.swfocus_start_button.setStyleSheet("background-color: white;")


class ThreadConfigReset(QThread):
    def __init__(
            self,
            this_cfg: Union[ConfigFocus, ConfigCRISP],
            labels_values: Dict[str, List[Union[QLabel, QLineEdit]]],
    ):
        super(QThread, self).__init__()
        self.this_cfg: Union[ConfigFocus, ConfigCRISP] = this_cfg
        self.labels_values: Dict[str, List[QLabel, QLineEdit]] = labels_values

    def run(self):
        for param_name in self.labels_values.keys():
            self.labels_values[param_name][1].setText(str(getattr(self.this_cfg, param_name)))
            time.sleep(0.1)  # need this otherwise it tries to join the thread while GUI updating -> crash


class ThreadSWFocus(QThread):
    def __init__(
            self,
            cam: EvoCamera,
            cfg_focus: ConfigFocus,
    ):
        super(QThread, self).__init__()
        self.cam = cam
        self.cfg_focus = cfg_focus
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()

    def run(self):
        cfg_focus = self.cfg_focus
        try:
            cfg_focus.check_config()
        except ConfigError as e:
            logger.warning(f"ThreadSWFocus.run: Invalid focus configuration:\n{e.message}\nAborting...")
            return
        curr_pos = int(self.cam.get_coordinates(['Z'])['Z'])
        coords = range(curr_pos - cfg_focus.rel_range, curr_pos + cfg_focus.rel_range, cfg_focus.steps_size)
        logger.warning(f"ThreadSWFocus.run: Starting software autofocus configured as\n"
                       f"{cfg_focus.__str__()}\nThis will move the stage up and down in the range "
                       f"[{(curr_pos - cfg_focus.rel_range) / 10},{(curr_pos + cfg_focus.rel_range) / 10}] μm"
                       f" (current position = {curr_pos / 10} μm). "
                       f"If there are objects blocking the stage movement, this will crash the "
                       f"objective and break it. You have 5 seconds to press stop. ")
        time.sleep(5)
        old_channel = self.cam.current_channel
        self.cam.set_exposure(exposure_time=int(self.cfg_focus.exposure_time))
        self.cam.studio.live().set_live_mode(False)
        best_focus_score = 0
        best_focus_position = 0
        focus_curve = []
        self.cam._set_channel(i_chan=ConfigLED.LED_NO_LED.value)
        for ipos, z_coord in enumerate(coords):
            if self.stopped():
                self.cam._set_channel(i_chan=old_channel)
                logger.warning("ThreadSWFocus.run: Aborting.")
                return
            self.cam.move_to(coordinates={'Z': z_coord}, block=True)
            image_raw = self.cam.display_save_frame(
                i_chan=cfg_focus.focus_channel,
                i_period=None,
                path_to_save=False,
                filename=None,
                display_frame=False,
            )
            if image_raw is None:
                logger.warning("ThreadSWFocus.run: self._take_frame returned None. Aborting...")
                return
            if self.cfg_focus.algorithm == ConfigFocusAlgorithm.LAPLACIAN_VAR.value:
                laplacian = cv2.Laplacian(image_raw, cv2.CV_64F)
                focus_score = laplacian.var()
            elif self.cfg_focus.algorithm == ConfigFocusAlgorithm.SQUARED_THRESHOLDED.value:
                laplacian = cv2.Laplacian(image_raw, cv2.CV_64F)
                focus_score = laplacian.var()
            focus_curve.append(focus_score)
            if focus_score > best_focus_score:
                best_focus_position = ipos
                best_focus_score = focus_score
        best_coordinate = coords[best_focus_position]
        logger.info(f"ThreadSWFocus.run: Finished scanning. Coordinate before focus={curr_pos / 10} μm,"
                    f"coordinate after focus={best_coordinate / 10} μm. Finalising software_focus.")
        self.cam.move_to(coordinates={'Z': best_coordinate}, block=True)
        self.cam._set_channel(i_chan=old_channel)


class ThreadMultiParam(QThread):
    def __init__(
            self,
            cam: EvoCamera,
            multi_param_lineedits: Dict[str, QLineEdit],
            savepath: str,
    ):
        super(QThread, self).__init__()
        self.cam = cam
        self.multi_param_lineedits = multi_param_lineedits
        self.start_run = True
        self.params = {key: 0 for key in self.multi_param_lineedits.keys()}
        self.savepath = savepath

    def get_params(self):
        for key in self.multi_param_lineedits.keys():
            value_str = self.multi_param_lineedits[key].text()
            if key in ['X', 'Y']:
                try:
                    self.params[key] = int(value_str)
                except ValueError as e:
                    logger.warning(f"EvoGUI.on_enter_pressed_multi_param: {e}")
                    self.params[key].setText(str(self.multi_param_lineedits[key]))
                    self.start_run = False
            else:
                try:
                    self.params[key] = [int(x) for x in value_str.rstrip("]").lstrip("[").split(",")]
                except ValueError as e:
                    logger.warning(f"EvoGUI.on_enter_pressed_multi_param: {e}")
                    self.multi_param_lineedits[key].setText(",".join([str(x) for x in self.multi_param_lineedits[key]]))
                    self.start_run = False

    def run(self):
        self.get_params()
        if not self.start_run:
            logger.warning(f"ThreadMultiParam.multi_dim_acquisition: Invalid parameters for multi-acquisition. Returning.")
            return
        if self.savepath is None:
            logger.warning(f"ThreadMultiParam.multi_dim_acquisition: Invalid path provided or save figure not pressed.")
            return
        current_pos = self.cam.get_coordinates(AXES)
        func_vert = self.cam.move_fov_up if self.params['Y'] >= 0 else self.cam.move_fov_down
        func_horiz_1 = self.cam.move_fov_right if self.params['X'] >= 0 else self.cam.move_fov_left
        func_horiz_2 = self.cam.move_fov_left if self.params['X'] >= 0 else self.cam.move_fov_right
        for i_vert in range(abs(self.params['Y'])):
            for i_horiz in range(abs(self.params['X'])):
                logger.info(f"At X={i_vert}/{abs(self.params['Y'])}, Y={i_horiz}/{abs(self.params['X'])}")
                for img_channel in self.params['LED']:
                    _ = self.cam.display_save_frame(
                        i_chan=img_channel,
                        path_to_save=self.savepath,
                        display_frame=False,
                    )
                if i_vert % 2 == 0:
                    func_horiz_1(block=True)
                else:
                    func_horiz_2(block=True)
            func_vert(block=True)
        try:
            self.cam.move_to(coordinates=current_pos, block=True)
        except (SerialException, ValueError) as e:
            logger.warning(f"ThreadMultiParam.run: {e}")


class ThreadPos(QThread):
    def __init__(
            self,
            cam: EvoCamera,
            pos_values: List[QLabel],
            i_direction: int,
            pos_move_lineedits: Dict[str, QLineEdit],
    ):
        super(QThread, self).__init__()
        self.cam = cam
        self.pos_values = pos_values
        self.i_direction = i_direction
        self.pos_move_lineedits = pos_move_lineedits

    def run(self):
        try:
            if self.i_direction == Direction.LEFT.value:
                self.cam.move_fov_left(block=True)
            elif self.i_direction == Direction.RIGHT.value:
                self.cam.move_fov_right(block=True)
            elif self.i_direction == Direction.UP.value:
                self.cam.move_fov_up(block=True)
            elif self.i_direction == Direction.DOWN.value:
                self.cam.move_fov_down(block=True)
            elif self.i_direction == Direction.HOME.value:
                self.cam.move_home(block=True)
            elif self.i_direction == Direction.MOVETO.value:
                self.cam.move_to(coordinates=self.get_lineedit_coords(), block=True)
        except (SerialException, ValueError, ASIErrors.ParameterOutOfRangeError, TigerError) as e:
            if isinstance(e, ASIErrors.ParameterOutOfRangeError):
                logger.warning(f"ThreadPos.run: Move is outside stage limits {SAFE_STAGE_LIMITS}. "
                               f"Use ZERO button to zero coordinate system if safe.")
            else:
                logger.warning(f"ThreadPos.run: {e}")

        self.update_position()

    def update_position(self):
        try:
            pos_dict = self.cam.get_coordinates(AXES)
            _ = [lab.setText(EvoGUI.make_pos_str(pos_dict[ax])) for lab, ax in zip(self.pos_values, AXES)]
        except (SerialException, KeyError, ValueError) as e:
            logger.warning(f"ThreadPos.update_position: {e}")

    def get_lineedit_coords(self):
        return {key: int(self.pos_move_lineedits[key].text()) for key in ['X', 'Y']}


class ThreadClearReadin(QThread):
    def __init__(self, labels: Dict[int, Dict[str, Union[QLabel, QLineEdit]]], text: Optional[str] = "?"):
        super(QThread, self).__init__()
        self.labels = labels
        self.text = text

    def run(self):
        for _dict in self.labels.values():
            for label_or_edit in _dict.values():
                label_or_edit.setText(self.text)


class ThreadLED(QThread):
    def __init__(self, buttons: Dict[int, QPushButton], i_active: int):
        super(QThread, self).__init__()
        self.buttons = buttons
        self.i_active = i_active

    def run(self):
        for i_chan, button in self.buttons.items():
            if i_chan == self.i_active:
                button.setStyleSheet("background-color: green;")
                button.setText("ON")
            else:
                button.setStyleSheet("background-color: red;")
                button.setText("OFF")


class ThreadDMD(QThread):
    def __init__(self, buttons: Dict[int, QPushButton], i_active: int):
        super(QThread, self).__init__()
        self.buttons = buttons
        self.i_active = i_active

    def run(self):
        for i, button in self.buttons.items():
            if i == self.i_active:
                button.setStyleSheet("background-color: green;")
            else:
                button.setStyleSheet("background-color: red;")


class ThreadStartCRISP(QThread):
    def __init__(self, cam: EvoCamera, cfg_crisp: ConfigCRISP):
        super(QThread, self).__init__()
        self.cam = cam
        self.cfg_crisp = cfg_crisp

    def run(self):
        self.cam.crisp_autofocus(this_cfg_crisp=self.cfg_crisp, user_input=False)


class FigureWidget(FigureCanvas):
    def __init__(self, parent=None, width=10, height=10, dpi=300):
        self.parent = parent
        self.fig_width: int = width
        self.fig_height: int = height
        self.fig_dpi: int = dpi
        self.cropping_boxes: Union[None, Dict[int, List[Tuple[int, int]]]] = None
        self._cropping_indices: Union[None, Dict[int, Tuple[Tuple[int, int], Tuple[int, int]]]] = None
        self.num_subplots: int = 1
        self.display_mode: DisplayMode = DisplayMode.NO_CROP
        self.fig = Figure(figsize=(self.fig_width, self.fig_height), dpi=self.fig_dpi)
        super(FigureWidget, self).__init__(self.fig)
        self.setParent(parent)
        self.ax = None
        self.make_figure()

    @staticmethod
    def cropping_boxes_are_valid(cropping_boxes: Union[None, Dict[int, List[Tuple[int, int]]]]):
        if not isinstance(cropping_boxes, dict):
            logger.warning(f"(1) Invalid cropping boxes {cropping_boxes}")
            return False
        for values in cropping_boxes.values():
            if (not isinstance(values, list)) or (len(values) < 1):
                logger.warning(f"(2) Invalid cropping boxes {cropping_boxes}")
                return False
            for val in values:
                if (not isinstance(val, tuple)) or (len(val) != 2):
                    logger.warning(f"(3) Invalid cropping boxes {cropping_boxes}")
                    return False
                for v in val:
                    if not isinstance(v, int):
                        logger.warning(f"(4) Invalid cropping boxes {cropping_boxes}")
                        return False
        if not len(FigureWidget.get_cropping_indices(cropping_boxes)) > 0:
            logger.warning(f"(5) Invalid cropping boxes {cropping_boxes}")
            return False
        return True

    @staticmethod
    def get_cropping_indices(
            cropping_boxes: Dict[int, List[Tuple[int, int]]]
    ) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        cropping_indices = []
        for key, point_list in cropping_boxes.items():
            x_coords = [p[0] for p in point_list]
            y_coords = [p[1] for p in point_list]
            if (min(x_coords) < max(x_coords)) and (min(y_coords) < max(y_coords)):
                cropping_indices.append(((min(x_coords), max(x_coords)), (min(y_coords), max(y_coords))))
        return cropping_indices

    def update_display_mode(
            self,
            display_mode: DisplayMode,
            cropping_boxes: Union[None, Dict[int, List[Tuple[int, int]]]]
    ):
        old_num_subplots = self.num_subplots
        if display_mode in [DisplayMode.CROP, DisplayMode.SHOW_FRAME]:
            if FigureWidget.cropping_boxes_are_valid(cropping_boxes):
                self._cropping_indices = FigureWidget.get_cropping_indices(cropping_boxes)
                self.display_mode = display_mode
                self.num_subplots = len(self._cropping_indices) if display_mode == DisplayMode.CROP else 1
            else:
                self._cropping_indices = None
                self.display_mode = DisplayMode.NO_CROP
                logger.warning(f"FigureWidget.update_display_mode: Invalid cropping boxes for mode {display_mode}")
                self.num_subplots = 1
        else:
            self.display_mode = DisplayMode.NO_CROP
            self._cropping_indices = None
            self.num_subplots = 1
        if self.num_subplots != old_num_subplots:
            self.make_figure()

    def _use_subplots(self):
        return (self.display_mode == DisplayMode.CROP) and (len(self._cropping_indices) > 1)

    def make_figure(self):
        if self._use_subplots():
            if self.ax is not None:
                if isinstance(self.ax, list):
                    [self.fig.delaxes(a) for a in self.ax]
                else:
                    self.fig.delaxes(self.ax)
            self.ax = []
            for i in range(len(self._cropping_indices)):
                self.ax.append(self.fig.add_subplot(len(self._cropping_indices), 1, i + 1))
                ((xmin, xmax), (ymin, ymax)) = self._cropping_indices[i]
        else:
            if self.ax is not None:
                if isinstance(self.ax, list):
                    [self.fig.delaxes(a) for a in self.ax]
                else:
                    self.fig.delaxes(self.ax)
            self.ax = self.fig.add_subplot(111)

    @staticmethod
    def create_image_with_text(
            text: Optional[str] = "Hello, World!",
            img_fraction: Optional[float] = 0.5,
            path_to_font: Optional[str] = "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
            size: Optional[Tuple[int, int]] = (3200, 3200),
    ):
        img_height, img_width = size
        image_pil = PIL.Image.fromarray(np.transpose(np.zeros(size, dtype=np.uint8)))
        font_size = 2
        font = PIL.ImageFont.truetype(path_to_font, font_size)
        while font.getlength(text) < img_fraction * image_pil.size[0]:
            font_size += 1
            font = PIL.ImageFont.truetype(path_to_font, font_size)
        draw = PIL.ImageDraw.Draw(image_pil)
        font = PIL.ImageFont.truetype(path_to_font, font_size)
        draw.text((int(img_width / 2), int(img_height / 2)), text, fill=255, font=font, anchor='mm')
        return np.array(image_pil)

    def plot_image(self, image_array=None):
        if self._use_subplots():
            for (a, ((xmin, xmax), (ymin, ymax))) in zip(self.ax, self._cropping_indices):
                a.clear()
                if image_array is None:
                    a.imshow(FigureWidget.create_image_with_text(text="no image", size=(xmax-xmin+1, ymax-ymin+1)))
                else:
                    a.imshow(image_array[xmin:xmax, ymin:ymax])
                a.set_xticks(a.get_xticks());
                a.set_yticks(a.get_yticks())
                a.set_xticklabels([int(tick + ymin) for tick in a.get_xticks()])
                a.set_yticklabels([int(tick + xmin) for tick in a.get_yticks()])
                # a.set_title(f"From ({xmin},{ymin}) to ({xmax},{ymax})")
        else:
            self.ax.clear()
            if self.display_mode == DisplayMode.SHOW_FRAME:
                if image_array is None:
                    self.ax.imshow(FigureWidget.create_image_with_text(text="no image", size=(3200, 3200)))
                else:
                    self.ax.imshow(image_array)
                for ((xmin, xmax), (ymin, ymax)) in self._cropping_indices:
                    rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                                             linewidth=1, edgecolor='r', facecolor='none')
                    self.ax.add_patch(rect)
            elif (self.display_mode == DisplayMode.CROP) and (len(self._cropping_indices) > 0):
                ((xmin, xmax), (ymin, ymax)) = self._cropping_indices[0]
                if image_array is None:
                    self.ax.imshow(FigureWidget.create_image_with_text(text="no image", size=(xmax-xmin+1, ymax-ymin+1)))
                else:
                    self.ax.imshow(image_array[xmin:xmax, ymin:ymax])
                self.ax.set_xticks(self.ax.get_xticks());
                self.ax.set_yticks(self.ax.get_yticks())
                self.ax.set_xticklabels([int(tick + ymin) for tick in self.ax.get_xticks()])
                self.ax.set_yticklabels([int(tick + xmin) for tick in self.ax.get_yticks()])
            else:
                if image_array is None:
                    self.ax.imshow(FigureWidget.create_image_with_text(text="no image", size=(3200, 3200)))
                else:
                    self.ax.imshow(image_array)
        self.update_figure()

    def update_figure(self):
        self.draw()
        self.flush_events()
        self.update()

    def show_boxes(self):
        if self.ax is None or self._cropping_indices is None or self.display_mode != DisplayMode.SHOW_FRAME:
            return
        self.remove_boxes()
        for ((xmin, xmax), (ymin, ymax)) in self._cropping_indices:
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                     linewidth=1, edgecolor='r', facecolor='none')
            self.ax.add_patch(rect)
        self.update_figure()

    def remove_boxes(self):
        if self.ax is None:
            return
        if isinstance(self.ax, list):
            for a in self.ax:
                for patch in a.patches:
                    patch.remove()
        else:
            for patch in self.ax.patches:
                patch.remove()


class ThreadLiveMode(QThread):
    def __init__(
            self,
            cam: EvoCamera,
            mpl_canvas: FigureWidget,
            normalise: bool,
            exposure_time: Union[int, float],
            img_channel: int,
    ):
        super(QThread, self).__init__()
        self.cam = cam
        self.mpl_canvas = mpl_canvas
        self.normalise = normalise
        self.img_channel = img_channel
        self.exposure_time = exposure_time
        self.pause_time = max(500, 2*exposure_time)
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()

    def run(self):
        self.cam.set_exposure(exposure_time=self.exposure_time)
        while not self.stopped():
            frame = self.cam.display_save_frame(
                i_chan=self.img_channel,
                path_to_save=False,
                display_frame=False,
            )
            if self.normalise:
                frame = self.cam.normalise_frame(frame)
            self.mpl_canvas.plot_image(image_array=frame)
            time.sleep(float(self.pause_time)/1000.0)


class ThreadExperiment(QThread):
    def __init__(
            self,
            cam: EvoCamera,
            mpl_canvas: FigureWidget,
            positions: Dict[int, Dict[str, Union[None, Dict[str, Union[float, int]]]]],
    ):
        super(QThread, self).__init__()
        self.cam = cam
        self.mpl_canvas = mpl_canvas
        self.valid_coordinates = False
        # self.coordinates = [{'X': 185501.2, 'Y': -62229.3}]
        self.coordinates: Union[None, List[Dict[str, float]]] = self.get_positions_from_dict(positions)
        self.img_channels = [1, 2]
        self.disp_channel = 1
        self.exposure_time = 1000
        self.pause_time = 1
        self.savepath = "/mnt/ImageData/Scott/2023-12-12"
        self._stop_event = threading.Event()

    def get_positions_from_dict(self, positions: Dict[int, Dict[str, Union[None, Dict[str, Union[float, int]]]]]):
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
        dfov = self.cam.get_delta_fov()
        coordinates = []
        for from_to in valid_dict:
            coord_from = from_to["from"]
            coord_to = from_to["to"]
            num_pos_x = int((coord_to["X"] - coord_from["X"]) / dfov)
            num_pos_y = int((coord_to["Y"] - coord_from["Y"]) / dfov)
            if (num_pos_x == 0) and (num_pos_y == 0):
                coordinates.append(coord_from)
            elif num_pos_x > num_pos_y:
                coords_x = [coord_from["X"] + i * dfov for i in range(num_pos_x)]
                coords_y = [coord_from["Y"] + float(i)/float(num_pos_x) * (coord_from["Y"] - coord_to["Y"])
                            for i in range(num_pos_x)]
                coordinates.extend([{"X": x, "Y": y} for x, y in zip(coords_x, coords_y)])
            else:
                coords_y = [coord_from["Y"] + i * dfov for i in range(num_pos_y)]
                coords_x = [coord_from["X"] + float(i)/float(num_pos_y) * (coord_from["X"] - coord_to["X"])
                            for i in range(num_pos_x)]
                coordinates.extend([{"X": x, "Y": y} for x, y in zip(coords_x, coords_y)])
        logger.info(f"Extracted {len(coordinates)} coordinates: {coordinates}")
        return coordinates

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()

    def run(self):
        if not self.valid_coordinates or self.coordinates is None:
            logger.warning("No valid coordinates provided. Aborting.")
            return
        logger.info(f"{self.coordinates}")
        self.cam.set_exposure(exposure_time=self.exposure_time)
        self.test_positions()
        iteration = 1
        while True:
            logger.info(f"At iteration {iteration}")
            self.cam._set_channel(i_chan=-1)
            for coord in self.coordinates:
                if self.stopped():
                    logger.info("Stopping acquisition.")
                    return
                self.cam.move_to(coordinates=coord, block=True)
                time.sleep(2)
                for img_channel in self.img_channels:
                    frame = self.cam.display_save_frame(
                        i_chan=img_channel,
                        path_to_save=self.savepath,
                        display_frame=False,
                    )
                    if img_channel == self.disp_channel:
                        frame = self.cam.normalise_frame(frame)
                        self.mpl_canvas.plot_image(image_array=frame)
            time.sleep(self.pause_time)
            iteration = iteration + 1

    def test_positions(self):
        logger.info("Testing positions before starting run.")
        for coord in self.coordinates:
            self.cam.move_to(coordinates=coord, block=True)
            frame = self.cam.display_save_frame(
                i_chan=1,
                path_to_save=False,
                display_frame=False,
            )
            frame = self.cam.normalise_frame(frame)
            self.mpl_canvas.plot_image(image_array=frame)
            time.sleep(2)
        logger.info("Positions tested.")


if __name__ == '__main__':
    is_testmode = True
    is_oil_objective = False

    if is_testmode:
        filenames = sorted(glob.glob("/mnt/ImageData/Scott/2023-12-08/*.tiff"))
        cam = TestCamera(
            cfg_device=DEVICE_CONFIG_EVO_TEST,
            cfg_objective=OBJECTIVE_CONFIG_OIL if is_oil_objective else OBJECTIVE_CONFIG_AIR,
            cfg_image=IMAGE_CONFIG_DEFAULT,
            cfg_crisp=CRISP_CONFIG_DEFAULT,
            cfg_focus=FOCUS_CONFIG_DEFAULT,
            filenames=filenames,
            pos_to_filename=None,
        )
    else:
        cam = EvoCamera(
            cfg_device=DEVICE_CONFIG_EVO_TEST,
            cfg_image=IMAGE_CONFIG_DEFAULT,
            cfg_objective=OBJECTIVE_CONFIG_OIL if is_oil_objective else OBJECTIVE_CONFIG_AIR,
            cfg_focus=FOCUS_CONFIG_DEFAULT,
            cfg_crisp=CRISP_CONFIG_DEFAULT,
        )
    dmd = DMDControl()

    app = QApplication(sys.argv)
    w = EvoGUI(cam, dmd)
    w.show()
    sys.exit(app.exec_())
