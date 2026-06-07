from multiprocessing import Event, Queue
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


from evomachine.commands import AutomatonCommand
from typing import Any

from evomachine.config import ImageProcessorConfig, get_logger
from evomachine.coordinates import Coordinate, CoordinateFactory
from evomachine.types import AutomatonCommandType, FovDirectionType
from evomachine.guidir_bkp.guitemplates import EvoPanelTemplate, EvoWorkerTemplate, EvoGUIThread
from evomachine.guidir_bkp.guitypes import DisplayMode, Direction, ARROW_LEFT, ARROW_RIGHT, ARROW_UP, ARROW_DOWN, AXES, \
    SMALL, CENTER, LEFT, RIGHT, NORMAL
from evomachine.guidir_bkp.queuemanager import QueueManager


logger = get_logger(name=__name__, is_gui=True)


class PositionWorker(EvoWorkerTemplate):
    def __init__(
            self,
            queue_manager: QueueManager,
            data_curr_pos: pyqtSignal,
            parent: QMainWindow | None = None,
    ):
        super().__init__(parent)
        self.data_curr_pos = data_curr_pos
        self.queue_manager = queue_manager
        self.step_z = 5  # half a micrometer

    @pyqtSlot(Direction)
    def move_fov(self, direction: Direction):
        logger.debug(f"move_fov: Move stage direction {direction}.")
        if self.is_disabled():
            logger.warning("MoveThread.move: Thread is disabled.")
            return
        fov_directions = {
            Direction.LEFT: FovDirectionType.LEFT,
            Direction.RIGHT: FovDirectionType.RIGHT,
            Direction.UP: FovDirectionType.UP,
            Direction.DOWN: FovDirectionType.DOWN,
        }
        if direction == Direction.LEFT:
            self.queue_manager.request(
                req_str='self.cam.move',
                kwargs_dict={'target': [(fov_directions[direction], 1.0)], 'block': True},
                callback=self.update_position,
            )
        elif direction == Direction.RIGHT:
            self.queue_manager.request(
                req_str='self.cam.move',
                kwargs_dict={'target': [(fov_directions[direction], 1.0)], 'block': True},
                callback=self.update_position,
            )
        elif direction == Direction.UP:
            self.queue_manager.request(
                req_str='self.cam.move',
                kwargs_dict={'target': [(fov_directions[direction], 1.0)], 'block': True},
                callback=self.update_position,
            )
        elif direction == Direction.DOWN:
            self.queue_manager.request(
                req_str='self.cam.move',
                kwargs_dict={'target': [(fov_directions[direction], 1.0)], 'block': True},
                callback=self.update_position,
            )
        elif direction == Direction.HOME:
            self.queue_manager.request(
                req_str='self.cam.move',
                kwargs_dict={'target': (FovDirectionType.HOME, 1.0), 'block': True},
                callback=self.update_position,
            )

    @pyqtSlot(Coordinate)
    def move_coord(self, coordinate: Coordinate):
        logger.debug(f"move_coord: Move stage direction {coordinate}.")
        if self.is_disabled():
            logger.warning("MoveThread.move: Thread is disabled.")
            return
        self.queue_manager.request(
            req_str='self.cam.move',
            kwargs_dict={'block': True, 'target': coordinate},
            callback=self.update_position,
        )

    @pyqtSlot()
    def halt_stage(self):
        logger.debug(f"MoveThread.halt_stage.")
        if self.is_disabled():
            logger.warning("MoveThread.halt_stage: Thread is disabled.")
            return
        self.queue_manager.request(
            req_str='self.cam.halt_stage',
            kwargs_dict={},
            callback=self.update_position,
        )

    @pyqtSlot()
    def zero_position(self):
        logger.debug("zero_position.")
        if self.is_disabled():
            logger.warning("MoveThread.update_position: Thread is disabled.")
            return
        self.queue_manager.request(
            req_str='self.cam.zero_coordinates',
            kwargs_dict={},
            callback=self.update_position,
        )

    @pyqtSlot(int)
    def update_position(self, data: int | None):
        logger.debug("update_position.")
        if self.is_disabled():
            logger.warning("MoveThread.update_position: Thread is disabled.")
            return
        self.queue_manager.request(
            req_str='self.cam.get_coordinates',
            kwargs_dict={'axes': AXES},
            callback=self.data_curr_pos.emit,
        )


class PositionPanel(EvoPanelTemplate):
    # Broadcasts current position of stage.
    data_curr_pos = pyqtSignal(dict)
    # Request current position of stage.
    request_curr_pos = pyqtSignal(int)
    # Request halt of stage.
    request_halt = pyqtSignal()
    # Move FoV LEFT/RIGHT/UP/DOWN.
    request_move_fov = pyqtSignal(Direction)
    # Move FoV HOME.
    request_move_home = pyqtSignal(Direction)
    # Move FoV LEFT/RIGHT/UP/DOWN/HOME.
    request_move_coord = pyqtSignal(Coordinate)
    # Zero position.
    request_zero_position = pyqtSignal()
    # Move FoV to FoV ID.
    request_move_selected_fov = pyqtSignal(int)

    MOVES = [Direction.LEFT.value, Direction.RIGHT.value, Direction.UP.value, Direction.DOWN.value,
             Direction.DOWN_Z.value, Direction.UP_Z.value]
    MOVES_STR = [ARROW_LEFT, ARROW_RIGHT, ARROW_LEFT, ARROW_RIGHT, ARROW_LEFT, ARROW_RIGHT]

    def __init__(
            self,
            queue_manager: QueueManager,
            camera_config: Any,
            processor_config: ImageProcessorConfig,
            start_strategy_event: Event,
            stop_strategy_event: Event,
            stop_event: Event,
            shutdown_event: Event,
            signal_init_crisp_values: pyqtSignal | None = None
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
        self.signal_init_crisp_values = signal_init_crisp_values

        self.data_curr_pos.connect(self.update_position_str)

        self.worker = PositionWorker(queue_manager=self.queue_manager, data_curr_pos=self.data_curr_pos)
        self.workers.append(self.worker)
        thread = EvoGUIThread()
        self.worker.moveToThread(thread)
        thread.start()
        self.threads.append(thread)

        self.pos_values = [self.make_label(text=self.make_pos_str(None), font=SMALL, align=RIGHT) for _ in AXES]
        "X/Y/Z values."

        # tmp = self.cam.get_stage_limits()
        # curr_limits = {'X': (tmp[0].x, tmp[1].x), 'Y': (tmp[0].y, tmp[1].y), 'Z': (tmp[0].z, tmp[1].z)}
        curr_limits = {'X': (0, 0), 'Y': (0, 0), 'Z': (0, 0)}
        self.pos_labels = [self.make_label(text=f"{ax} [{self.make_pos_str(curr_limits[ax][0], unit='cm')}, "
                                                f"{self.make_pos_str(curr_limits[ax][1], unit='cm')}]", font=SMALL)
                           for ax in AXES]
        "X/Y/Z labels."
        self.queue_manager.request(
            req_str='self.cam.get_stage_limits',
            kwargs_dict={},
            callback=self.update_limits,
        )

        self.request_curr_pos.connect(self.worker.update_position)
        self.pos_update_button = self.make_button_w_emit(text="Update", signal=self.request_curr_pos, font=SMALL, param=0)
        self.request_curr_pos.emit(0)
        "Trigger update of self.pos_values."

        self.request_halt.connect(self.worker.halt_stage)
        self.pos_halt_button = self.make_button_w_emit(text="Halt", signal=self.request_halt, font=SMALL)
        "Halt stage."

        self.request_move_home.connect(self.worker.move_fov)
        self.pos_home_button = self.make_button_w_emit(text="Home", signal=self.request_move_home, font=SMALL,
                                                       param=Direction.HOME)
        "Move to home position."

        self.request_zero_position.connect(self.worker.zero_position)
        self.pos_zero_button = self.make_button_w_emit(text="Zero", signal=self.request_zero_position, font=SMALL)
        "Zero coordinate system."

        self.request_move_fov.connect(self.worker.move_fov)
        self.pos_arrow_buttons = [
            self.make_button_w_emit(text=self.MOVES_STR[i], signal=self.request_move_fov, font=SMALL,
                                    param=Direction(i))
            for i in self.MOVES
        ]
        self.pos_arrow_buttons[Direction.DOWN_Z.value].setEnabled(False)
        self.pos_arrow_buttons[Direction.UP_Z.value].setEnabled(False)
        "Arrow buttons for moving stage."

        self.current_moveto = {'X': None, 'Y': None, 'Z': None}
        self.pos_move_lineedits = {key: self.make_lineedit(text=str(self.current_moveto[key]),
                                                           func=self.update_current_move,
                                                           param=key)
                                   for key in AXES}
        "Lineedits for entering move coordinates."

        self.request_move_coord.connect(self.worker.move_coord)
        self.pos_move_button = self.make_button(
            text="Move to",
            func=lambda: self.request_move_coord.emit(Coordinate.from_dict(self.current_moveto)),
            font=SMALL)
        "Move to entered coordinates."

        self.fovs: dict[int, Coordinate] = {}
        self.curr_fov: int | None = None
        self.fov_combo_box: QComboBox = QComboBox()
        self.fov_combo_box.addItems(["None"])
        self.fov_combo_box.setEnabled(False)
        self.fov_combo_box.currentIndexChanged.connect(self.update_current_fov)
        self.pos_move_fov_button = self.make_button(
            text="Move to FoV",
            func=self.move_fov,
            font=SMALL)
        "Move to selected FoV ID."
        self.pos_move_fov_button.setEnabled(False)

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

        self.layout.addWidget(self.fov_combo_box, 3, 5, 1, 1)
        self.layout.addWidget(self.pos_move_fov_button, 4, 5, 1, 1)
        _ = [self.layout.setColumnMinimumWidth(i, 0) for i in range(self.layout.columnCount())]
        _ = [self.layout.setColumnStretch(i, 0) for i in range(self.layout.columnCount())]
        self.layout.setHorizontalSpacing(0)

        self.widget = QWidget()
        self.widget.setLayout(self.layout)

        # This will catch FoV list after initialisation.
        queue_manager.register(self.update_fovs, AutomatonCommandType.FOV_DATA)

    def move_coordinate(self):
        coord = Coordinate.from_dict(self.current_moveto)
        self.request_move_coord.emit(coord)
        logger.debug(f"move_coordinate {coord}.")

    def move_fov(self):
        if (self.fovs is None) or (self.curr_fov not in self.fovs):
            logger.error(f"Cannot move to FoV {self.curr_fov}. Available = {self.fovs}")
            return
        logger.info(f"Moving stage to FoV {self.curr_fov} at {self.fovs[self.curr_fov]}.")
        self.request_move_coord.emit(self.fovs[self.curr_fov])

    def update_current_fov(self):
        self.curr_fov = None if self.fov_combo_box.currentText() in ["None", ""] \
            else int(self.fov_combo_box.currentText())

    def update_fovs(self, cmd: AutomatonCommand):
        logger.debug(f"Position: Updating FoVs: {cmd.command_args['fovs']}.")
        self.fovs = cmd.command_args['fovs']
        self.fov_combo_box.clear()
        self.fov_combo_box.addItems([str(fov) for fov in self.fovs.keys()])
        self.fov_combo_box.setEnabled(True)
        self.pos_move_fov_button.setEnabled(True)

    def update_current_move(self, key: str):
        try:
            self.current_moveto[key] = float(self.pos_move_lineedits[key].text())
        except ValueError:
            logger.error(f"Invalid move to input: {self.pos_move_lineedits[key].text()}")
            self.current_moveto[key] = None

    def update_limits(self, data: tuple[Coordinate, Coordinate]):
        curr_limits = {'X': (data[0].x, data[1].x), 'Y': (data[0].y, data[1].y), 'Z': (data[0].z, data[1].z)}
        for i, ax in enumerate(AXES):
            self.pos_labels[i].setText(f"{ax} [{self.make_pos_str(curr_limits[ax][0], unit='mm')}, "
                                       f"{self.make_pos_str(curr_limits[ax][1], unit='mm')}]")

    @pyqtSlot(dict)
    def update_position_str(self, pos_dict: dict[str, int | float | None]):
        logger.debug(f"update_position_str: {pos_dict}.")
        try:
            _ = [lab.setText(self.make_pos_str(pos_dict[ax])) for lab, ax in zip(self.pos_values, AXES)]
        except (SerialException, KeyError) as e:
            logger.warning(f"EvoGUI.update_position: {e}")
