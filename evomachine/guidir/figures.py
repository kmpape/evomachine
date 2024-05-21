import time
from multiprocessing import Event
from pathlib import Path

from matplotlib.axes import Axes
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

import numpy as np
from PyQt5.QtGui import QIntValidator
from typing import Any, Dict, Tuple, Union
from PyQt5.QtCore import pyqtSignal, pyqtSlot, QTimer, Qt, QThread
from PyQt5.QtWidgets import (
    QWidget, QLineEdit,  QComboBox,
    QVBoxLayout, QGridLayout,
    QSizePolicy,
)

import delta.utils
from delta.utils import CroppingBox as DeltaCroppingBox

from evomachine.commands import AutomatonCommand
from evomachine.coordinates import Coordinate
from evomachine.evotypes import AutomatonCommandType, LEDType
from evomachine.config import ConfigCamera, ConfigImageProcessor, get_logger
from evomachine.guidir.guitemplates import EvoPanelTemplate, EvoGUIThread, EvoWorkerTemplate, FolderExistsValidator, \
    FilenameValidator, EVO_STYLE
from evomachine.guidir.guitypes import SMALL, LEFT, RIGHT  # noqa
from evomachine.guidir.queuemanager import QueueManager
from evomachine.utils import EvoCroppingBox


logger = get_logger(name=__name__)


class FigureWindow(QWidget):
    def __init__(self, fig, title):
        super().__init__()  # noqa
        self.setWindowTitle(title)
        layout = QVBoxLayout()
        self.canvas = FigureCanvas(fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.canvas)
        self.setLayout(layout)


class ImageROIBoxes(EvoWorkerTemplate):
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
        self.fov_id = fov_id

    def draw_roi_boxes(self, roi_boxes: list[EvoCroppingBox | None]):
        for i, box in enumerate(roi_boxes):
            if box is not None and ((isinstance(box, EvoCroppingBox) and not box.is_none) or
                                    isinstance(box, delta.utils.CroppingBox)):
                color_str = 'yellow'
                width = box.xbr - box.xtl
                height = box.ybr - box.ytl
                rect = Rectangle((box.xtl, box.ytl), width, height, edgecolor='black', facecolor=color_str, alpha=0.2,
                                 linewidth=2)
                self.ax.add_patch(rect)

    @pyqtSlot(np.ndarray, str, list)  # noqa
    def update_plot(self, image_to_plot: np.ndarray, title: str, roi_boxes: list[EvoCroppingBox | None]):
        self.ax.clear()
        self.ax.imshow(image_to_plot, cmap='gray')
        self.ax.set_title(title, fontsize=self.FONT_SIZE)
        self.ax.tick_params(axis='both', labelsize=self.FONT_SIZE)
        self.draw_roi_boxes(roi_boxes)
        self.canvas.draw()


class ImageCroppingBoxes(EvoWorkerTemplate):
    cropping_box_drawn = pyqtSignal(int, int, EvoCroppingBox)  # noqa
    cropping_box_str = pyqtSignal(int, str)  # noqa
    cropping_unselect = pyqtSignal()  # noqa

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
        self.canvas.mpl_connect('button_press_event', self.on_mouse_press)  # noqa
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)  # noqa
        self.canvas.mpl_connect('button_release_event', self.on_mouse_release)  # noqa
        # button_layout = QVBoxLayout()
        # self.layout.addLayout(button_layout)

        # self.widget = QWidget()
        # self.widget.setLayout(self.layout)

    def update_rectangle(self):
        if self.box_id is None:
            return
        rectangle = self.rectangles[self.box_id]
        if rectangle in self.ax.patches:  # Check if the rectangle is in the list of patches
            rectangle.remove()  # noqa
        try:
            if self.start_point:
                width = self.current_point[0] - self.start_point[0]
                height = self.current_point[1] - self.start_point[1]
                rectangle = Rectangle(self.start_point, width, height, fill=False, edgecolor='red')
                self.rectangles[self.box_id] = rectangle  # noqa
                setattr(self, f"rectangle{self.box_id}", rectangle)
                self.ax.add_patch(rectangle)
                self.canvas.draw()
        except TypeError as e:
            logger.warning(e)

    @pyqtSlot()  # noqa
    def update_all_boxes(self):
        if self.box_id is None:
            return
        for box_id, rectangle in self.rectangles.items():
            if rectangle is None:
                pass
            rectangle = self.rectangles[self.box_id]
            if rectangle not in self.ax.patches:  # Check if the rectangle is in the list of patches
                rectangle.remove()  # noqa
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

    @pyqtSlot(int)  # noqa
    def on_select_button_clicked(self, box_id: int):
        # sender = self.sender()
        # sender.setChecked(True)
        # other_button = self.select_button0 if box_id == 1 else self.select_button1
        # other_button.setChecked(False)
        self.box_id = box_id if box_id >= 0 else None
        # self.setFocus()

    @pyqtSlot(int)  # noqa
    def clear_selected_box(self, box_id):
        logger.debug(f"Clearing box {box_id}")
        if box_id >= len(self.rectangles) or self.rectangles[box_id] is None:
            return
        rectangle = self.rectangles[box_id]
        if rectangle:
            rectangle.remove()  # noqa
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

    def keyPressEvent(self, event):  # noqa
        if event.key() == Qt.Key_Escape:
            # self.select_button0.setChecked(False)
            # self.select_button1.setChecked(False)
            self.box_id = None
            # self.setFocus()
            self.cropping_unselect.emit()

    @pyqtSlot(np.ndarray, str)  # noqa
    def update_plot(self, image_to_plot: np.ndarray, title: str):
        self.ax.clear()
        self.ax.imshow(image_to_plot, cmap='gray')
        self.ax.set_title(title, fontsize=self.FONT_SIZE)
        self.ax.tick_params(axis='both', labelsize=self.FONT_SIZE)
        self.canvas.draw()
        self.update_all_boxes()


class ChannelWorker(EvoWorkerTemplate):
    def __init__(
            self,
            canvas: FigureCanvas,
            ax: Axes,
            img: np.ndarray,
            title: str,
            font_size: int = 8,
            roi_boxes: list[DeltaCroppingBox] | None = None,
    ):

        super().__init__()
        self.canvas = canvas
        self.ax = ax
        self.font_size = font_size
        self.update_plot(img, title, roi_boxes)

    @pyqtSlot(np.ndarray, str, list)  # noqa
    def update_plot(self, img: np.ndarray, title: str, roi_boxes: list[DeltaCroppingBox]):
        self.ax.clear()
        self.ax.imshow(img, cmap='gray')
        self.ax.set_title(title, fontsize=self.font_size)
        self.ax.tick_params(axis='both', labelsize=self.font_size)
        if self.roi_boxes is not None:
            for i, box in enumerate(self.roi_boxes):
                width = box.xbr - box.xtl
                height = box.ybr - box.ytl
                rect = Rectangle((box.xtl, box.ytl), width, height, edgecolor='red', facecolor='red', alpha=0.1)
                _ = self.ax.add_patch(rect)
                _ = self.ax.text((box.xbr + box.xtl) * 0.5, (box.ybr + box.ytl) * 0.5, str(i), color='blue', fontsize=6)
        self.canvas.draw()


class ChannelPlotter(QWidget):
    FONT_SIZE = 8
    signal_worker_update = pyqtSignal(np.ndarray, str, list)  # noqa

    def __init__(
            self,
            img: np.ndarray,
            channel_to_index: Dict[LEDType, int],
            width: int = 12,
            height: int = 12,
            title_prefix: str = "",
            roi_boxes: list[DeltaCroppingBox] | None = None,
    ):
        super().__init__()  # noqa
        self.img = img
        self.channel_to_index = channel_to_index
        self.title_prefix = title_prefix
        self.roi_boxes = roi_boxes

        self.curr_channel = list(self.channel_to_index.keys())[0]

        self.fig = Figure(figsize=(width, height))
        self.fig.patch.set_facecolor('#262626')
        self.ax = self.fig.add_subplot(111)
        self.ax.imshow(self.img[self.channel_to_index[self.curr_channel], :, :], cmap='gray')
        self.title = self.title_prefix + f" {self.curr_channel}"
        self.ax.set_title(self.title, fontsize=self.FONT_SIZE)
        self.ax.tick_params(axis='both', labelsize=self.FONT_SIZE)
        if self.roi_boxes is not None:
            for i, box in enumerate(self.roi_boxes):
                width = box.xbr - box.xtl
                height = box.ybr - box.ytl
                rect = Rectangle((box.xtl, box.ytl), width, height, edgecolor='red', facecolor='red', alpha=0.1)
                _ = self.ax.add_patch(rect)
                _ = self.ax.text((box.xbr + box.xtl) * 0.5, (box.ybr + box.ytl) * 0.5, str(i), color='blue', fontsize=6)
        self.fig.tight_layout(pad=5)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setStyleSheet(EVO_STYLE)

        self.channel_combo_box = QComboBox()  # noqa
        self._channels = [ch for ch in self.channel_to_index.keys()]
        self.channel_combo_box.addItems([str(ch) for ch in self._channels])
        self.channel_combo_box.currentIndexChanged.connect(self.update_plot)  # noqa

        self.layout = QGridLayout()
        self.layout.addWidget(self.canvas, 0, 0, 1, 1)
        self.layout.addWidget(self.channel_combo_box, 1, 0, 1, 1)

        self.widget = QWidget()  # noqa
        self.widget.setLayout(self.layout)
        self.worker = ChannelWorker(
            canvas=self.canvas,
            ax=self.ax,
            img=self.img[self.channel_to_index[self.curr_channel], :, :],
            title=self.title,
            font_size=self.FONT_SIZE,
            roi_boxes=self.roi_boxes,
        )
        self.signal_worker_update.connect(self.worker.update_plot)
        self.thread = QThread()  # noqa
        self.worker.moveToThread(self.thread)
        self.thread.start()

    def update_plot(self, index):
        self.curr_channel = self._channels[self.channel_combo_box.currentIndex()]
        self.title = self.title_prefix + f" {self.curr_channel}"
        self.signal_worker_update.emit(
            self.img[self.channel_to_index[self.curr_channel], :, :],
            self.title,
            self.roi_boxes,
        )
        # self.ax.clear()
        # self.ax.imshow(self.img[self.channel_to_index[self.curr_channel], :, :], cmap='gray')
        # self.ax.set_title(self.title_prefix + f" {self.curr_channel}", fontsize=self.FONT_SIZE)
        # self.ax.tick_params(axis='both', labelsize=self.FONT_SIZE)
        # self.canvas.draw()

    def update_image(self, img: np.ndarray, roi_boxes: list[DeltaCroppingBox] | None):
        self.img = img
        self.curr_channel = self._channels[self.channel_combo_box.currentIndex()]
        self.roi_boxes = roi_boxes
        self.signal_worker_update.emit(
            self.img[self.channel_to_index[self.curr_channel], :, :],
            self.title,
            self.roi_boxes,
        )
        # self.img = img
        # self.update_plot(0)


class ImagePlotter(EvoPanelTemplate):
    signal_draw = pyqtSignal(int)  # noqa
    signal_clear = pyqtSignal(int)  # noqa
    signal_update_all_boxes = pyqtSignal()  # noqa
    signal_update_plot = pyqtSignal()  # noqa
    signal_new_image = pyqtSignal(np.ndarray, str, list)  # noqa

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
            width: int = 20,
            height: int = 20,
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

        self.worker = ImageROIBoxes(canvas=self.canvas, ax=self.ax, fig=self.fig)
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
        self.exposure_edit.returnPressed.connect(self.on_enter_pressed_exposure)  # noqa
        self.exposure_edit.setValidator(QIntValidator(0, 2147483647))

        self.take_frame_button = self.make_button(text="Take Frame", font=SMALL, func=self.take_frame)

        self.current_live_mode_interval: int = 1000
        self.live_interval_value = self.make_label(text=f"{self.current_live_mode_interval} ms", font=SMALL, width_px=100)
        self.live_mode_timer = QTimer(self)
        self.live_interval_edit = QLineEdit()
        self.live_interval_edit.returnPressed.connect(self.on_enter_pressed_live_interval)  # noqa
        self.live_interval_edit.setValidator(QIntValidator(0, 2147483647))
        self.live_mode_timer.timeout.connect(self._take_frame)  # noqa
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
        self.savepath_edit.returnPressed.connect(self.on_enter_pressed_savepath)  # noqa
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
        self.filename_edit.returnPressed.connect(self.on_enter_pressed_filename)  # noqa

        self.fov_combo_box: QComboBox = QComboBox()  # noqa
        self.fov_combo_box.addItems(["None"])
        self.fov_combo_box.currentIndexChanged.connect(self.update_plot)  # noqa

        self.channel_combo_box = QComboBox()  # noqa
        self._channels = list(self.channel_to_index.keys())
        self.channel_combo_box.addItems([str(ch) for ch in self._channels])
        self.channel_combo_box.currentIndexChanged.connect(self.update_plot)  # noqa

        self.layout = QGridLayout()
        self.layout.addWidget(self.exposure_label, 0, 0, 1, 1)
        self.layout.addWidget(self.exposure_edit, 0, 1, 1, 1)
        self.layout.addWidget(self.exposure_value, 0, 2, 1, 1)

        self.layout.addWidget(self.savepath_label, 1, 0, 1, 1)
        self.layout.addWidget(self.savepath_edit, 1, 1, 1, 1)
        self.layout.addWidget(self.savepath_value, 1, 2, 1, 5)

        self.layout.addWidget(self.filename_label, 2, 0, 1, 1)
        self.layout.addWidget(self.filename_edit, 2, 1, 1, 1)
        self.layout.addWidget(self.filename_value, 2, 2, 1, 3)

        self.layout.addWidget(self.take_frame_button, 3, 0, 1, 1)
        self.layout.addWidget(self.savepath_checkbox, 3, 1, 1, 1)

        self.layout.addWidget(self.canvas, 4, 0, 7, 7)
        rr = 4 + 9
        self.layout.addWidget(self.make_label(f"FoV:", align=RIGHT), rr, 0, 1, 1)
        self.layout.addWidget(self.fov_combo_box, rr, 1, 1, 1)
        self.layout.addWidget(self.make_label(f"Channel:", align=RIGHT), rr, 2, 1, 1)
        self.layout.addWidget(self.channel_combo_box, rr, 3, 1, 1)

        # self.worker.cropping_box_str.connect(self.update_cropping_label)
        # self.signal_clear.connect(self.worker.clear_selected_box)
        # self.worker.cropping_unselect.connect(self.unselect_buttons)
        # self.signal_draw.connect(self.worker.on_select_button_clicked)
        # self.signal_update_all_boxes.connect(self.worker.update_all_boxes)
        self.signal_update_plot.connect(self.update_plot)
        self.signal_new_image.connect(self.worker.update_plot)

        self.roi_data: dict[int, dict] = {}

        self.reference_array = {-1: np.zeros((len(self.channel_to_index), *self.camera_config.image.shape))}
        self.image_array = {-1: np.zeros((len(self.channel_to_index), *self.camera_config.image.shape))}
        self.image_time_str = "None"
        self.signal_update_plot.emit()

        self.widget = QWidget()  # noqa
        self.widget.setLayout(self.layout)

        queue_manager.register(self.update_image, AutomatonCommandType.PROCESS_DATA)
        queue_manager.register(self.update_image, AutomatonCommandType.REF_DATA)
        queue_manager.register(self.update_fovs, AutomatonCommandType.FOV_DATA)
        queue_manager.register(self.read_roi_data, AutomatonCommandType.ROI_DATA)

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

    def read_roi_data(self, data: AutomatonCommand):
        logger.info(f"ImagePlotter.read_roi_data: Received ROI data: {data.command_args['roi_boxes']}.")
        if not data.command_args['fov_id'] in self.roi_data:
            logger.warning(f"ImagePlotter.read_roi_data: fov_id {data.command_args['fov_id']} not in {self.roi_data.keys()}")
            return
        self.roi_data[data.command_args['fov_id']] = {
            'rotation': data.command_args['rotation'],
            'roi_boxes': data.command_args['roi_boxes'],
        }
        self.signal_update_plot.emit()

    @pyqtSlot(int)  # noqa
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

    @pyqtSlot()  # noqa
    def unselect_buttons(self):
        self.select_cropping[0].setChecked(False)
        self.select_cropping[1].setChecked(False)

    def update_exposure_value(self, data: Any):
        self.exposure_value.setText(str(self.current_exposure))

    def update_filename_value(self, filename: str):
        self.filename_value.setText(filename)

    def update_fovs(self, cmd: AutomatonCommand):
        logger.info(f"ImagePlotter.Updating FoVs: {cmd.command_args}")
        self.fovs = cmd.command_args['fovs']
        self.roi_data = {fov_id: {} for fov_id in self.fovs.keys()}
        for fov_id in self.fovs.keys():
            if fov_id not in self.image_array.keys():
                self.image_array[fov_id] = np.zeros((len(self.channel_to_index), *self.camera_config.image.shape))
        self.fov_combo_box.clear()
        self.fov_combo_box.addItems([str(fov) for fov in self.fovs.keys()])

    @pyqtSlot(LEDType)  # noqa
    def update_led(self, led: LEDType):
        self.current_led = led

    @pyqtSlot(int, str)  # noqa
    def update_cropping_label(self, box_id: int, text: str):
        logger.debug(f"Updating cropping label {box_id}: {text} with current {self.values_cropping}")
        self.values_cropping[box_id].setText(text)
        if "None" in text:
            self.select_cropping[box_id].setChecked(False)
            self.signal_draw.emit(-1)

    @pyqtSlot(AutomatonCommand)  # noqa
    def update_image(self, cmd: AutomatonCommand):
        if cmd.command_type == AutomatonCommandType.IMAGE:
            fov_id = cmd.fov_id
            channels_int = [self.channel_to_index[c] for c in cmd.command_args['channels']]
            if not fov_id in self.image_array.keys():
                self.image_array[fov_id] = np.zeros((len(self.channel_to_index), *self.camera_config.image.shape))
            self.image_array[fov_id][channels_int, :, :] = cmd.command_data
            self.image_time_str = cmd.get_exec_time()
        elif cmd.command_type == AutomatonCommandType.REF_DATA:
            self.image_array = cmd.command_args
            self.image_time_str = cmd.get_exec_time()
        self.signal_update_plot.emit()

    def update_image_take_frame(self, data: np.ndarray, i_chan: LEDType):
        time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.signal_new_image.emit(data, f"{time_str}: Channel {i_chan}", [EvoCroppingBox.none_box()])
        self.signal_update_all_boxes.emit()
        self.take_frame_button.setEnabled(True)

    @pyqtSlot()  # noqa
    def update_plot(self):
        if self.fov_combo_box.currentText() == "None":
            fov_index = -1
        elif self.fov_combo_box.currentText() == "":
            logger.error("self.fov_combo_box.currentText() empty in upldate_plot. TODO")
            return
        else:
            fov_index = int(self.fov_combo_box.currentText())
        if fov_index in self.roi_data and self.roi_data[fov_index]:
            logger.info("Displaying roi boxes.")
            boxes = self.roi_data[fov_index]['roi_boxes']
        else:
            boxes = [EvoCroppingBox.none_box()]
        channel_index = self.channel_to_index[self._channels[self.channel_combo_box.currentIndex()]]
        image_to_plot = self.image_array[fov_index][channel_index, :, :]
        # self.ax.clear()
        # self.ax.imshow(image_to_plot, cmap='gray')
        title = f"{self.image_time_str}: FoV {fov_index} - Channel {list(self.channel_to_index.keys())[channel_index]}"
        # self.ax.set_title(title, fontsize=self.FONT_SIZE)
        # self.ax.tick_params(axis='both', labelsize=self.FONT_SIZE)
        # self.canvas.draw()
        self.signal_new_image.emit(image_to_plot, title, boxes)


class FigureMultiWindow(QWidget):
    def __init__(self, fig_dict):
        super().__init__()  # noqa
        self.fig_dict = fig_dict
        self.current_index = 0

        self.setWindowTitle("Focus Curves")
        layout = QVBoxLayout()
        self.canvas = FigureCanvas(self.fig_dict[self.current_index])
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def keyPressEvent(self, event):  # noqa
        if event.key() == Qt.Key_Left:
            self.current_index = (self.current_index - 1) % len(self.fig_dict)
        elif event.key() == Qt.Key_Right:
            self.current_index = (self.current_index + 1) % len(self.fig_dict)
        self.update_figure()

    def update_figure(self):
        fig = self.fig_dict[self.current_index]
        self.canvas.figure = fig
        self.canvas.draw()

