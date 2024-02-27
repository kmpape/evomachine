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
from evomachine.automaton import Automaton, AutomatonQueueDataType
from evomachine.commands import AutomatonCommand, AutomatonCommandType
from evomachine.config import ConfigCRISP, ConfigFocus, get_logger
from evomachine.coordinates import Coordinate, CoordinateFactory
from evomachine.dmd import DMDControl
from evomachine.exceptions import ConfigError, TigerError
from evomachine.evotypes import LEDType, FocusAlgorithmType
from evomachine.guidir.guitemplates import EvoPanelTemplate, EvoWorkerTemplate, EvoGUIThread
from evomachine.guidir.guitypes import DisplayMode, Direction, ARROW_LEFT, ARROW_RIGHT, ARROW_UP, ARROW_DOWN, AXES, \
    SMALL, CENTER, LEFT, RIGHT, NORMAL


logger = get_logger(name=__name__)


class PositionWorker(EvoWorkerTemplate):
    def __init__(
            self,
            cam: AbstractCamera,
            data_curr_pos: pyqtSignal,
            parent: Optional[QMainWindow] = None,
    ):
        super().__init__(parent)
        self.data_curr_pos = data_curr_pos
        self.cam = cam
        self.step_z = 5  # half a micrometer

    def move_to(self, direction: Direction, move_to_pos: Optional[Coordinate] = None):
        if self.is_disabled():
            logger.warning("MoveThread.move_to: Thread is disabled.")
            return

        if direction == Direction.MOVETO and move_to_pos is None:
            logger.warning("MoveThread.move_to: No move_to coordinate given.")
            return
        try:
            if direction == Direction.LEFT:
                self.cam.move_fov_left(block=True)
            elif direction == Direction.RIGHT:
                self.cam.move_fov_right(block=True)
            elif direction == Direction.UP:
                self.cam.move_fov_up(block=True)
            elif direction == Direction.DOWN:
                self.cam.move_fov_down(block=True)
            elif direction == Direction.HOME:
                self.cam.move_home(block=True)
            elif direction == Direction.MOVETO:
                self.cam.move_to(coordinate=move_to_pos, block=True)
        except (SerialException, ValueError, ASIErrors.ParameterOutOfRangeError, TigerError) as e:
            if isinstance(e, ASIErrors.ParameterOutOfRangeError):
                logger.warning(f"MoveThread.move_to: Move is outside stage limits")
            else:
                logger.warning(f"ThreadPos.move_to: {e}")

        self.update_position()

    def halt_stage(self):
        if self.is_disabled():
            logger.warning("MoveThread.halt_stage: Thread is disabled.")
            return
        self.cam.halt_stage()
        self.update_position()

    def zero_position(self):
        if self.is_disabled():
            logger.warning("MoveThread.update_position: Thread is disabled.")
            return
        self.cam.zero_coordinates()
        self.update_position()

    @pyqtSlot()
    def update_position(self):
        if self.is_disabled():
            logger.warning("MoveThread.update_position: Thread is disabled.")
            return
        try:
            pos_dict = self.cam.get_coordinates(AXES)
            self.data_curr_pos.emit(pos_dict)
        except (SerialException, KeyError) as e:
            logger.warning(f"EvoGUI.update_position: {e}")


class PositionPanel(EvoPanelTemplate):
    # Broadcasts current position of stage.
    data_curr_pos = pyqtSignal(dict)
    # Request current position of stage.
    request_curr_pos = pyqtSignal()
    # Request halt of stage.
    request_halt = pyqtSignal()

    MOVES = [Direction.LEFT.value, Direction.RIGHT.value, Direction.UP.value, Direction.DOWN.value,
             Direction.DOWN_Z.value, Direction.UP_Z.value]
    MOVES_STR = [ARROW_LEFT, ARROW_RIGHT, ARROW_LEFT, ARROW_RIGHT, ARROW_LEFT, ARROW_RIGHT]

    def __init__(
            self,
            cam: AbstractCamera,
    ):
        super().__init__(cam=cam)

        self.data_curr_pos.connect(self.update_position_str)

        self.worker = PositionWorker(cam=self.cam, data_curr_pos=self.data_curr_pos)
        self.workers.append(self.worker)
        thread = EvoGUIThread()
        self.worker.moveToThread(thread)
        # thread.start()
        self.threads.append(thread)

        self.pos_values = [self.make_label(text=self.make_pos_str(None), font=SMALL, align=RIGHT) for _ in AXES]
        "X/Y/Z values."

        tmp = self.cam.get_stage_limits()
        curr_limits = {'X': (tmp[0].x, tmp[1].x), 'Y': (tmp[0].y, tmp[1].y), 'Z': (tmp[0].z, tmp[1].z)}
        self.pos_labels = [self.make_label(text=f"{ax} [{self.make_pos_str(curr_limits[ax][0], unit='cm')}, "
                                                f"{self.make_pos_str(curr_limits[ax][1], unit='cm')}]", font=SMALL)
                           for ax in AXES]
        "X/Y/Z labels."

        self.request_curr_pos.connect(self.worker.update_position)
        self.pos_update_button = self.make_button(text="Update", func=self.request_curr_pos.emit, font=SMALL)
        "Trigger update of self.pos_values."

        self.pos_halt_button = self.make_button(text="Halt", func=self.worker.halt_stage, font=SMALL)
        "Halt stage."

        self.pos_home_button = self.make_button(text="Home", func=self.worker.move_to, font=SMALL,
                                                direction=Direction.HOME)
        "Move to home position."

        self.pos_zero_button = self.make_button(text="Zero", func=self.worker.zero_position, font=SMALL)
        "Zero coordinate system."

        self.pos_arrow_buttons = [
            self.make_button(text=self.MOVES_STR[i], func=self.worker.move_to, font=SMALL, direction=Direction(i))
            for i in self.MOVES
        ]
        "Arrow buttons for moving stage."

        self.current_moveto = {'X': 0, 'Y': 0, 'Z': 0}
        line_edit_validator = QDoubleValidator(
            bottom=-1e10,
            top=1e10,
            decimals=5,
        )
        self.pos_move_lineedits = {key: self.make_lineedit(text=str(self.current_moveto[key]),
                                                           func=self.update_current_move_to,
                                                           param=key,
                                                           validator=line_edit_validator)
                                   for key in AXES}
        "Lineedits for entering move_to coordinates."

        self.pos_move_button = self.make_button(text="Move to", func=self.move_to_pos, font=SMALL)
        "Move to entered coordinates."

        self.layout = QGridLayout()
        self.layout.addWidget(self.make_label(text="Stage Control", font=NORMAL), 0, 0, 1, 5, LEFT)
        _ = [self.layout.addWidget(pos_label, i, 0, CENTER) for i, pos_label in enumerate(self.pos_labels, start=1)]
        _ = [self.layout.addWidget(pos_value, i, 1, CENTER) for i, pos_value in enumerate(self.pos_values, start=1)]
        self.layout.addWidget(self.pos_update_button, 4, 0, 1, 1)
        self.layout.addWidget(self.pos_home_button, 4, 1, 1, 1)
        self.layout.addWidget(self.pos_halt_button, 4, 2, 1, 1)
        self.layout.addWidget(self.pos_zero_button, 4, 3, 1, 1)
        _ = [self.layout.addWidget(self.pos_arrow_buttons[i], int(i/2) + 1, (i % 2) + 2) for i in self.MOVES]
        _ = [self.layout.addWidget(self.pos_move_lineedits[key], i + 1, 4) for i, key in enumerate(AXES)]
        # _ = [self.layout.addWidget(pos_lim, i, 5, CENTER) for i, pos_lim in enumerate(self.pos_limits, start=1)]
        self.layout.addWidget(self.pos_move_button, 4, 4, 1, 1)
        _ = [self.layout.setColumnMinimumWidth(i, 0) for i in range(self.layout.columnCount())]
        _ = [self.layout.setColumnStretch(i, 0) for i in range(self.layout.columnCount())]
        self.layout.setHorizontalSpacing(0)

        self.widget = QWidget()
        self.widget.setLayout(self.layout)

    def move_to_pos(self):
        self.worker.move_to(
            direction=Direction.MOVETO,
            move_to_pos=Coordinate(
                x=self.current_moveto['X'],
                y=self.current_moveto['Y'],
                z=self.current_moveto['Z']
            )
        )

    def update_current_move_to(self, key: str):
        try:
            self.current_moveto[key] = float(self.pos_move_lineedits[key].text())
        except ValueError:
            self.current_moveto[key] = None

    @pyqtSlot(dict)
    def update_position_str(self, pos_dict: Dict[str, Union[int, float, None]]):
        try:
            _ = [lab.setText(self.make_pos_str(pos_dict[ax])) for lab, ax in zip(self.pos_values, AXES)]
        except (SerialException, KeyError) as e:
            logger.warning(f"EvoGUI.update_position: {e}")
