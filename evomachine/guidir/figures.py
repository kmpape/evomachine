import time
from multiprocessing import Event
from pathlib import Path

from matplotlib.axes import Axes
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtGui import QValidator, QIntValidator
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import PIL
from PyQt5.QtCore import pyqtSignal, pyqtSlot, QEventLoop, QThread, QTimer, QObject, QRegExp, Qt
from PyQt5.QtWidgets import (
    QWidget,
    QMainWindow, QApplication,
    QLabel, QLineEdit, QPushButton, QComboBox, QMessageBox,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QSizePolicy, QScrollArea, QFileDialog, QCheckBox
)

from evomachine.commands import AutomatonCommand
from evomachine.coordinates import Coordinate
from evomachine.evotypes import AutomatonCommandType, LEDType
from evomachine.config import ConfigCamera, ConfigImageProcessor, get_logger
from evomachine.guidir.guitemplates import EvoPanelTemplate, EvoGUIThread, EvoWorkerTemplate, FolderExistsValidator, \
    FilenameValidator, EVO_STYLE
from evomachine.guidir.guitypes import SMALL, LEFT
from evomachine.guidir.queuemanager import QueueManager
from evomachine.utils import EvoCroppingBox


logger = get_logger(name=__name__)


# font = {'size': 6}
# matplotlib.rc('font', **font)


class FigureWindow(QWidget):
    def __init__(self, fig, title):
        super().__init__()
        self.setWindowTitle(title)
        layout = QVBoxLayout()
        self.canvas = FigureCanvas(fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.canvas)
        self.setLayout(layout)


class ImageCroppingBoxes(EvoWorkerTemplate):
    cropping_box_drawn = pyqtSignal(int, int, EvoCroppingBox)
    cropping_box_str = pyqtSignal(int, str)
    cropping_unselect = pyqtSignal()

    FONT_SIZE = 8

    def __init__(
            self,
            canvas: FigureCanvas,
            ax: Axes,
            fig: Figure,
            fov_id: int = -1,
            parent=None,
    ):
        super().__init__(parent)
        self.canvas = canvas
        self.ax = ax
        self.fig = fig
        self.rectangles = {0: None, 1: None}
        self.fov_id = fov_id

        self.rectangle0 = None
        self.rectangle1 = None
        self.start_point = None
        self.current_point: Union[None, Tuple[int, int]] = None
        self.box_id = None
        self.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        # button_layout = QVBoxLayout()
        # self.layout.addLayout(button_layout)

        # self.widget = QWidget()
        # self.widget.setLayout(self.layout)

    def update_rectangle(self):
        if self.box_id is None:
            return
        rectangle = self.rectangles[self.box_id]
        if rectangle in self.ax.patches:  # Check if the rectangle is in the list of patches
            rectangle.remove()
        try:
            if self.start_point:
                width = self.current_point[0] - self.start_point[0]
                height = self.current_point[1] - self.start_point[1]
                rectangle = Rectangle(self.start_point, width, height, fill=False, edgecolor='red')
                self.rectangles[self.box_id] = rectangle
                setattr(self, f"rectangle{self.box_id}", rectangle)
                self.ax.add_patch(rectangle)
                self.canvas.draw()
        except TypeError as e:
            logger.warning(e)

    @pyqtSlot()
    def update_all_boxes(self):
        if self.box_id is None:
            return
        for box_id, rectangle in self.rectangles.items():
            if rectangle is None:
                pass
            rectangle = self.rectangles[self.box_id]
            if rectangle not in self.ax.patches:  # Check if the rectangle is in the list of patches
                rectangle.remove()
                self.ax.add_patch(rectangle)
                self.canvas.draw()

    def on_mouse_press(self, event):
        if event.button == 1:
            if event.xdata is None or event.ydata is None:
                self.start_point = None
            else:
                x, y = event.xdata, event.ydata
                self.start_point = (x, y)
                self.current_point = (x, y)
                self.update_rectangle()

    def on_mouse_move(self, event):
        if event.button == 1 and self.start_point:
            x = self.current_point[0] if event.xdata is None else event.xdata
            y = self.current_point[1] if event.ydata is None else event.ydata
            self.current_point = (x, y)
            self.update_rectangle()

    def on_mouse_release(self, event):
        if event.button == 1 and self.start_point:
            x = self.current_point[0] if event.xdata is None else event.xdata
            y = self.current_point[1] if event.ydata is None else event.ydata
            self.current_point = (x, y)
            self.update_rectangle()
            self.emit_rectangle()

    @pyqtSlot(int)
    def on_select_button_clicked(self, box_id: int):
        # sender = self.sender()
        # sender.setChecked(True)
        # other_button = self.select_button0 if box_id == 1 else self.select_button1
        # other_button.setChecked(False)
        self.box_id = box_id if box_id >= 0 else None
        # self.setFocus()

    @pyqtSlot(int)
    def clear_selected_box(self, box_id):
        logger.debug(f"Clearing box {box_id}")
        if box_id >= len(self.rectangles) or self.rectangles[box_id] is None:
            return
        rectangle = self.rectangles[box_id]
        if rectangle:
            rectangle.remove()
            self.rectangles[box_id] = None  # Reset the rectangle for the box ID
            self.canvas.draw()
        self.cropping_box_drawn.emit(self.fov_id, box_id, EvoCroppingBox.none_box())
        self.cropping_box_str.emit(box_id, "xtl=None, xbr=None, ytl=None, ybr=None")
        # if box_id == 0:
        #     self.label_box0.setText("Box 0: None")
        # else:
        #     self.label_box1.setText("Box 1: None")
        self.box_id = None
        setattr(self, f"rectangle{self.box_id}", None)

    def emit_rectangle(self):
        if self.box_id is None or self.start_point is None:
            return
        cropping_coords = {
            'xtl': int(self.start_point[0]),
            'xbr': int(self.current_point[0]),
            'ytl': int(self.start_point[1]),
            'ybr': int(self.current_point[1]),
        }
        cropping_coords_str = ", ".join([f"{k}={v}" for k, v in cropping_coords.items()])
        self.cropping_box_str.emit(self.box_id, cropping_coords_str)
        self.cropping_box_drawn.emit(self.fov_id, self.box_id, EvoCroppingBox.from_dict(cropping_coords))

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            # self.select_button0.setChecked(False)
            # self.select_button1.setChecked(False)
            self.box_id = None
            # self.setFocus()
            self.cropping_unselect.emit()

    @pyqtSlot(np.ndarray, str)
    def update_plot(self, image_to_plot: np.ndarray, title: str):
        self.ax.clear()
        self.ax.imshow(image_to_plot, cmap='gray')
        self.ax.set_title(title, fontsize=self.FONT_SIZE)
        self.ax.tick_params(axis='both', labelsize=self.FONT_SIZE)
        self.canvas.draw()
        self.update_all_boxes()


class ChannelPlotter(QWidget):
    FONT_SIZE = 8

    def __init__(
            self,
            img: np.ndarray,
            channel_to_index: Dict[LEDType, int],
            width: int = 8,
            height: int = 8,
            title_prefix: str = "",
    ):
        super().__init__()
        self.img = img
        self.channel_to_index = channel_to_index
        self.title_prefix = title_prefix

        self.curr_channel = list(self.channel_to_index.keys())[0]

        self.fig = Figure(figsize=(width, height))
        self.fig.patch.set_facecolor('#262626')
        self.ax = self.fig.add_subplot(111)
        self.ax.imshow(self.img[self.channel_to_index[self.curr_channel], :, :], cmap='gray')
        self.ax.set_title(self.title_prefix + f" {self.curr_channel}", fontsize=self.FONT_SIZE)
        self.ax.tick_params(axis='both', labelsize=self.FONT_SIZE)
        self.fig.tight_layout(pad=5)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setStyleSheet(EVO_STYLE)

        self.channel_combo_box = QComboBox()
        self._channels = [ch for ch in self.channel_to_index.keys()]
        self.channel_combo_box.addItems([str(ch) for ch in self._channels])
        self.channel_combo_box.currentIndexChanged.connect(self.update_plot)

        self.layout = QGridLayout()
        self.layout.addWidget(self.canvas, 0, 0, 1, 1)
        self.layout.addWidget(self.channel_combo_box, 1, 0, 1, 1)

        self.widget = QWidget()
        self.widget.setLayout(self.layout)

    def update_plot(self, index):
        self.curr_channel = self._channels[self.channel_combo_box.currentIndex()]
        self.ax.clear()
        self.ax.imshow(self.img[self.channel_to_index[self.curr_channel], :, :], cmap='gray')
        self.ax.set_title(self.title_prefix + f" {self.curr_channel}", fontsize=self.FONT_SIZE)
        self.ax.tick_params(axis='both', labelsize=self.FONT_SIZE)
        self.canvas.draw()

    def update_image(self, img: np.ndarray):
        self.img = img
        self.update_plot(0)


class ImagePlotter(EvoPanelTemplate):
    signal_draw = pyqtSignal(int)
    signal_clear = pyqtSignal(int)
    signal_update_all_boxes = pyqtSignal()
    signal_update_plot = pyqtSignal()
    signal_new_image = pyqtSignal(np.ndarray, str)

    FONT_SIZE = 8
    NO_BOX = "xtl=None, xbr=None, ytl=None, ybr=None"

    def __init__(
            self,
            queue_manager: QueueManager,
            camera_config: ConfigCamera,
            processor_config: ConfigImageProcessor,
            start_strategy_event: Event,
            stop_strategy_event: Event,
            stop_event: Event,
            shutdown_event: Event,
            width: int = 10,
            height: int = 10,
            dpi: int = 300,
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

        self.fovs: Dict[int, Coordinate] = {}
        self.channel_to_index = self.processor_config.channel_to_index
        self.fig = Figure(figsize=(width, height))
        self.fig.patch.set_facecolor('#262626')
        self.ax = self.fig.add_subplot(111)
        self.ax.imshow(np.zeros(self.camera_config.image.shape), cmap='gray')
        self.ax.set_title("No Image", fontsize=self.FONT_SIZE)
        self.ax.tick_params(axis='both', labelsize=self.FONT_SIZE)
        self.fig.tight_layout(pad=5)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setStyleSheet(EVO_STYLE)

        self.worker = ImageCroppingBoxes(canvas=self.canvas, ax=self.ax, fig=self.fig)
        self.workers.append(self.worker)
        thread = EvoGUIThread()
        self.worker.moveToThread(thread)
        thread.start()
        self.threads.append(thread)

        self.current_led: LEDType = LEDType.NO_LED
        self.current_exposure: int = int(self.camera_config.default_exposure_time)
        self.current_normalise_frame: bool = True
        self.exposure_label = self.make_label(text="Exposure [ms]", font=SMALL, width_px=100)
        self.exposure_value = self.make_label(text=f"{self.camera_config.default_exposure_time}", font=SMALL,
                                              width_px=100)
        self.exposure_edit = QLineEdit()
        self.exposure_edit.returnPressed.connect(self.on_enter_pressed_exposure)
        self.exposure_edit.setValidator(QIntValidator(0, 2147483647))

        self.take_frame_button = self.make_button(text="Take Frame", font=SMALL, func=self.take_frame)

        self.current_live_mode_interval: int = 1000
        self.live_interval_value = self.make_label(text=f"{self.current_live_mode_interval} ms", font=SMALL, width_px=100)
        self.live_mode_timer = QTimer(self)
        self.live_interval_edit = QLineEdit()
        self.live_interval_edit.returnPressed.connect(self.on_enter_pressed_live_interval)
        self.live_interval_edit.setValidator(QIntValidator(0, 2147483647))
        self.live_mode_timer.timeout.connect(self._take_frame)
        self.live_frame_label = self.make_label(text="Live mode", font=SMALL)
        self.live_frame_start_button = self.make_button(text="Start Live", font=SMALL, func=self.start_live_mode)
        self.live_frame_stop_button = self.make_button(text="Stop Live", font=SMALL, func=self.stop_live_mode)
        self.live_frame_stop_button.setEnabled(False)

        self.normalise_checkbox = self.make_checkbox(
            text="Normalise",
            font=SMALL,
            set_true=self.current_normalise_frame,
            func=self.toggle_normalise_frame,
        )
        self.current_savepath: Path = self.camera_config.path_to_save
        self.savepath_label = self.make_label(text="Savepath", font=SMALL)
        self.savepath_value = self.make_label(text=str(self.current_savepath), font=SMALL, width_px=400)
        self.savepath_value.setWordWrap(True)
        self.savepath_edit = QLineEdit()
        self.savepath_edit.returnPressed.connect(self.on_enter_pressed_savepath)
        self.save_frame: bool = False
        self.savepath_checkbox = self.make_checkbox(text="Save", font=SMALL, set_true=self.save_frame,
                                                    func=self.toggle_save)
        self.current_filename: Union[str, None] = None
        self.filename_label = self.make_label(text="Filename", font=SMALL)
        self.filename_value = self.make_label(text=str(self.current_filename), font=SMALL, width_px=400)
        self.filename_value.setWordWrap(True)
        self.queue_manager.request(
            req_str="self.cam.get_default_filename",
            kwargs_dict={},
            callback=self.update_filename_value,
        )
        self.filename_edit = QLineEdit()
        self.filename_edit.returnPressed.connect(self.on_enter_pressed_filename)

        self.fov_combo_box: QComboBox = QComboBox()
        self.fov_combo_box.addItems(["None"])
        self.fov_combo_box.currentIndexChanged.connect(self.update_plot)

        self.channel_combo_box = QComboBox()
        self._channels = list(self.channel_to_index.keys())
        self.channel_combo_box.addItems([str(ch) for ch in self._channels])
        self.channel_combo_box.currentIndexChanged.connect(self.update_plot)

        self.labels_cropping = [self.make_label(f"Cropping Box {i}", align=LEFT) for i in range(2)]
        self.values_cropping = [self.make_label(self.NO_BOX, font=SMALL, align=LEFT) for _ in range(2)]
        self.select_cropping = [QPushButton("Draw") for _ in range(2)]
        self.clear_cropping = [QPushButton("Clear") for _ in range(2)]
        self.select_cropping[0].setCheckable(True)
        self.select_cropping[0].setChecked(False)
        self.select_cropping[0].setFont(SMALL)
        self.select_cropping[0].clicked.connect(lambda: self.select_button_clicked(0))
        self.clear_cropping[0].clicked.connect(lambda: self.signal_clear.emit(0))
        self.clear_cropping[0].setFont(SMALL)
        self.select_cropping[1].setCheckable(True)
        self.select_cropping[1].setChecked(False)
        self.select_cropping[1].setFont(SMALL)
        self.select_cropping[1].clicked.connect(lambda: self.select_button_clicked(1))
        self.clear_cropping[1].clicked.connect(lambda: self.signal_clear.emit(1))
        self.clear_cropping[1].setFont(SMALL)

        self.layout = QGridLayout()
        self.layout.addWidget(self.exposure_label, 0, 0, 1, 1)
        self.layout.addWidget(self.exposure_edit, 0, 1, 1, 1)
        self.layout.addWidget(self.exposure_value, 0, 2, 1, 1)

        self.layout.addWidget(self.savepath_label, 1, 0, 1, 1)
        self.layout.addWidget(self.savepath_edit, 1, 1, 1, 1)
        self.layout.addWidget(self.savepath_value, 1, 2, 1, 3)

        self.layout.addWidget(self.filename_label, 2, 0, 1, 1)
        self.layout.addWidget(self.filename_edit, 2, 1, 1, 1)
        self.layout.addWidget(self.filename_value, 2, 2, 1, 3)

        self.layout.addWidget(self.take_frame_button, 3, 0, 1, 1)
        self.layout.addWidget(self.savepath_checkbox, 3, 1, 1, 1)
        self.layout.addWidget(self.live_frame_start_button, 4, 0, 1, 1)
        self.layout.addWidget(self.live_frame_stop_button, 5, 0, 1, 1)
        self.layout.addWidget(self.live_interval_edit, 4, 1, 1, 1)
        self.layout.addWidget(self.live_interval_value, 4, 2, 1, 1)

        # self.layout.addWidget(self.make_label("Experiment", align=LEFT, font=SMALL), 4, 0, 1, 1)
        # self.layout.addWidget(self.make_label("FoV", align=LEFT, font=SMALL), 4, 1, 1, 1)
        # self.layout.addWidget(self.fov_combo_box, 4, 2, 1, 1)
        # self.layout.addWidget(self.make_label("Channel", align=LEFT, font=SMALL), 4, 3, 1, 1)
        # self.layout.addWidget(self.channel_combo_box, 4, 4, 1, 1)

        self.layout.addWidget(self.canvas, 6, 0, 4, 4)
        self.layout.addWidget(self.fov_combo_box, 6, 4, 1, 1)
        self.layout.addWidget(self.channel_combo_box, 7, 4, 1, 1)

        for i in range(2):
            self.layout.addWidget(self.labels_cropping[i], i + 11, 0, 1, 1)
            self.layout.addWidget(self.values_cropping[i], i + 11, 1, 1, 1)
            self.layout.addWidget(self.select_cropping[i], i + 11, 2, 1, 1)
            self.layout.addWidget(self.clear_cropping[i], i + 11, 3, 1, 1)

        self.worker.cropping_box_str.connect(self.update_cropping_label)
        self.signal_clear.connect(self.worker.clear_selected_box)
        self.worker.cropping_unselect.connect(self.unselect_buttons)
        self.signal_draw.connect(self.worker.on_select_button_clicked)
        self.signal_update_all_boxes.connect(self.worker.update_all_boxes)
        self.signal_update_plot.connect(self.update_plot)
        self.signal_new_image.connect(self.worker.update_plot)

        self.image_array = {-1: np.zeros((len(self.channel_to_index), *self.camera_config.image.shape))}
        self.image_time_str = "None"
        self.signal_update_plot.emit()

        self.widget = QWidget()
        self.widget.setLayout(self.layout)

        queue_manager.register(self.update_image, AutomatonCommandType.PROCESS_DATA)
        queue_manager.register(self.update_fovs, AutomatonCommandType.FOV_DATA)

    def on_enter_pressed_exposure(self):
        self.current_exposure = int(self.exposure_edit.text())
        self.queue_manager.request(
            req_str="self.cam.set_exposure",
            kwargs_dict={'exposure_time': self.current_exposure},
            callback=self.update_exposure_value,
        )

    def on_enter_pressed_filename(self):
        input_str = self.filename_edit.text().strip()
        if input_str == "":
            self.current_filename = input_str
            self.queue_manager.request(
                req_str="self.cam.get_default_filename",
                kwargs_dict={},
                callback=self.update_filename_value,
            )
        else:
            if not FilenameValidator.validate(input_str):
                logger.warning(f"Invalid filename: {input_str}")
                return
            self.current_filename = input_str
            self.filename_value.setText(self.current_filename)

    def on_enter_pressed_live_interval(self):
        try:
            current_value = int(self.live_interval_edit.text())
        except ValueError as e:
            logger.warning(f"Wrong format for interval {self.live_interval_edit.text()}.")
            return
        if current_value <= 0:
            logger.warning(f"Invalid interval {current_value}.")
            return
        self.current_live_mode_interval = current_value
        self.live_interval_value.setText(f"{str(self.current_live_mode_interval)} ms")

    def on_enter_pressed_savepath(self):
        input_str = self.savepath_edit.text().strip()
        if input_str == "":
            self.current_savepath = self.camera_config.path_to_save
            self.savepath_value.setText(str(self.current_savepath))
        else:
            if not FolderExistsValidator.validate(input_str):
                logger.warning(f"Invalid savepath: {input_str}")
                return
            self.current_savepath = Path(input_str)
            self.savepath_value.setText(str(self.current_savepath))

    @pyqtSlot(int)
    def select_button_clicked(self, box_id: int):
        logger.debug(f"Select button clicked: {box_id}")
        if not self.select_cropping[box_id].isChecked():
            logger.debug("Unselecting")
            self.signal_draw.emit(-1)
            self.select_cropping[box_id].setChecked(False)
        else:
            logger.debug("Selecting")
            self.signal_draw.emit(box_id)
            self.select_cropping[box_id].setChecked(True)
            other_id = 0 if box_id == 1 else 1
            self.select_cropping[other_id].setChecked(False)
            self.setFocus()

    def start_live_mode(self):
        self.live_mode_timer.start(self.current_live_mode_interval)
        self.live_frame_start_button.setEnabled(False)
        self.live_frame_stop_button.setEnabled(True)

    def stop_live_mode(self):
        self.live_mode_timer.stop()
        self.live_frame_start_button.setEnabled(True)
        self.live_frame_stop_button.setEnabled(False)

    def _take_frame(
            self,
            save_frame: bool = False,

    ):
        self.queue_manager.request(
            req_str="self.cam.display_save_frame",
            kwargs_dict={
                'i_chan': self.current_led,
                'path_to_save': self.current_savepath if save_frame else None,
                'filename': self.current_filename,
                'normalise': self.current_normalise_frame,
                'reset_led': False,
                'display_frame': False,
            },
            callback=self.update_image_take_frame,
            callback_args=(self.current_led,),
        )

    def take_frame(self):
        self._take_frame(save_frame=self.save_frame)
        self.take_frame_button.setEnabled(False)

    def toggle_normalise_frame(self, state):
        if state == Qt.Checked:
            self.current_normalise_frame = True
        else:
            self.current_normalise_frame = False

    def toggle_save(self, state):
        if state == Qt.Checked:
            self.save_frame = True
        else:
            self.save_frame = False

    @pyqtSlot()
    def unselect_buttons(self):
        self.select_cropping[0].setChecked(False)
        self.select_cropping[1].setChecked(False)

    def update_exposure_value(self, data: Any):
        self.exposure_value.setText(str(self.current_exposure))

    def update_filename_value(self, filename: str):
        self.filename_value.setText(filename)

    def update_fovs(self, cmd: AutomatonCommand):
        logger.debug(f"Updating FoVs: {cmd.command_args}")
        self.fovs = cmd.command_args['fovs']
        for fov_id in self.fovs.keys():
            if fov_id not in self.image_array.keys():
                self.image_array[fov_id] = np.zeros((len(self.channel_to_index), *self.camera_config.image.shape))
        self.fov_combo_box.clear()
        self.fov_combo_box.addItems([str(fov) for fov in self.fovs.keys()])

    @pyqtSlot(LEDType)
    def update_led(self, led: LEDType):
        self.current_led = led

    @pyqtSlot(int, str)
    def update_cropping_label(self, box_id: int, text: str):
        logger.debug(f"Updating cropping label {box_id}: {text} with current {self.values_cropping}")
        self.values_cropping[box_id].setText(text)
        if "None" in text:
            self.select_cropping[box_id].setChecked(False)
            self.signal_draw.emit(-1)

    @pyqtSlot(AutomatonCommand)
    def update_image(self, cmd: AutomatonCommand):
        if cmd.command_type == AutomatonCommandType.IMAGE:
            fov_id = cmd.fov_id
            channels_int = [self.channel_to_index[c] for c in cmd.command_args['channels']]
            if not fov_id in self.image_array.keys():
                self.image_array[fov_id] = np.zeros((len(self.channel_to_index), *self.camera_config.image.shape))
            self.image_array[fov_id][channels_int, :, :] = cmd.command_data
            self.image_time_str = cmd.get_exec_time()
            self.signal_update_plot.emit()

    def update_image_take_frame(self, data: np.ndarray, i_chan: LEDType):
        time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.signal_new_image.emit(data, f"{time_str}: Channel {i_chan}")
        self.signal_update_all_boxes.emit()
        self.take_frame_button.setEnabled(True)

    @pyqtSlot()
    def update_plot(self):
        fov_index = -1 if self.fov_combo_box.currentText() == "None" else int(self.fov_combo_box.currentText())
        channel_index = self.channel_to_index[self._channels[self.channel_combo_box.currentIndex()]]
        image_to_plot = self.image_array[fov_index][channel_index, :, :]
        # self.ax.clear()
        # self.ax.imshow(image_to_plot, cmap='gray')
        title = f"{self.image_time_str}: FoV {fov_index} - Channel {list(self.channel_to_index.keys())[channel_index]}"
        # self.ax.set_title(title, fontsize=self.FONT_SIZE)
        # self.ax.tick_params(axis='both', labelsize=self.FONT_SIZE)
        # self.canvas.draw()
        self.signal_new_image.emit(image_to_plot, title)
        self.signal_update_all_boxes.emit()

class FigureMultiWindow(QWidget):
    def __init__(self, fig_dict):
        super().__init__()
        self.fig_dict = fig_dict
        self.current_index = 0

        self.setWindowTitle("Focus Curves")
        layout = QVBoxLayout()
        self.canvas = FigureCanvas(self.fig_dict[self.current_index])
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Left:
            self.current_index = (self.current_index - 1) % len(self.fig_dict)
        elif event.key() == Qt.Key_Right:
            self.current_index = (self.current_index + 1) % len(self.fig_dict)
        self.update_figure()

    def update_figure(self):
        fig = self.fig_dict[self.current_index]
        self.canvas.figure = fig
        self.canvas.draw()

