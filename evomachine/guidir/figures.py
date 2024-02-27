from matplotlib.axes import Axes
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
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
import queue
import threading

from evomachine.acquisition import AbstractCamera
from evomachine.automaton import Automaton
from evomachine.commands import AutomatonCommand
from evomachine.dmd import DMDControl
from evomachine.config import get_logger
from evomachine.guidir.guitemplates import EvoPanelTemplate, EvoGUIThread, EvoWorkerTemplate, QueueManager
from evomachine.guidir.guitypes import DisplayMode
from evomachine.evotypes import AutomatonQueueDataType, AutomatonCommandType


logger = get_logger(name=__name__)


class FigureWidget(FigureCanvas):
    def __init__(self, parent=None, width=10, height=10, dpi=300):
        self.parent = parent
        self.fig_width: int = width
        self.fig_height: int = height
        self.fig_dpi: int = dpi
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
        self.fig.delaxes(self.ax)
        self.ax = self.fig.add_subplot(111)

    @staticmethod
    def create_image_with_text(
            text: Optional[str] = "None",
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

    def plot_image(self, image_array=None, title=None):
        if self._use_subplots():
            for (a, ((xmin, xmax), (ymin, ymax))) in zip(self.ax, self._cropping_indices):
                a.clear()
                if image_array is None:
                    a.imshow(FigureWidget.create_image_with_text(text="no image", size=(xmax-xmin+1, ymax-ymin+1)))
                else:
                    a.imshow(image_array[ymin:ymax, xmin:xmax])
                a.set_xticks(a.get_xticks())
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
                    # self.ax.imshow(image_array[xmin:xmax, ymin:ymax])
                    self.ax.imshow(image_array[ymin:ymax, xmin:xmax])
                self.ax.set_xlim(left=0, right=xmax-xmin)
                self.ax.set_ylim(bottom=ymax-ymin, top=0)
                self.ax.set_xticks(self.ax.get_xticks())
                self.ax.set_yticks(self.ax.get_yticks())
                # self.ax.set_xticklabels([int(tick + ymin) for tick in self.ax.get_xticks()])
                # self.ax.set_yticklabels([int(tick + xmin) for tick in self.ax.get_yticks()])
                self.ax.set_xticklabels([int(tick + xmin) for tick in self.ax.get_xticks()])
                self.ax.set_yticklabels([int(tick + ymin) for tick in self.ax.get_yticks()])
            else:
                if image_array is None:
                    self.ax.imshow(FigureWidget.create_image_with_text(text="no image", size=(3200, 3200)))
                else:
                    self.ax.imshow(image_array)
            if title is not None:
                self.ax.set_title(title)
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


class ImageCroppingBoxes(EvoPanelTemplate):
    rectangleDrawn = pyqtSignal(str, dict)
    rectangleCleared = pyqtSignal(str, type(None))

    def __init__(
            self,
            canvas: FigureCanvas,
            ax: Axes,
            parent=None,
    ):
        super().__init__(parent)
        self.canvas = canvas
        self.ax = ax
        self.rectangles = {0: None, 1: None}

        layout = QVBoxLayout(self)
        self.rectangle0 = None
        self.rectangle1 = None
        self.start_point = None
        self.current_point: Union[None, Tuple[int, int]] = None
        self.box_id = None
        self.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        button_layout = QVBoxLayout()
        layout.addLayout(button_layout)

        # Box 0
        self.layout.addWidget(self.make_label("Box 0: None"), 0, 0, 1, 1)
        self.select_button0 = QPushButton("Draw")
        self.select_button0.setCheckable(True)
        self.select_button0.clicked.connect(lambda: self.on_select_button_clicked(0))
        self.layout.addWidget(self.select_button0, 0, 1, 1, 1)
        clear_button0 = QPushButton("Clear")
        clear_button0.clicked.connect(lambda: self.clear_selected_box(0))
        self.layout.addWidget(clear_button0, 0, 2, 1, 1)

        # Box 1
        self.layout.addWidget(self.make_label("Box 1: None"), 1, 0, 1, 1)
        self.select_button1 = QPushButton("Select")
        self.select_button1.setCheckable(True)
        self.select_button1.clicked.connect(lambda: self.on_select_button_clicked(1))
        self.layout.addWidget(self.select_button1, 1, 1, 1, 1)
        clear_button1 = QPushButton("Clear")
        clear_button1.clicked.connect(lambda: self.clear_selected_box(1))
        self.layout.addWidget(clear_button1, 1, 2, 1, 1)

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
            print(e)

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

    def on_select_button_clicked(self, box_id):
        sender = self.sender()
        sender.setChecked(True)
        other_button = self.select_button0 if box_id == 1 else self.select_button1
        other_button.setChecked(False)
        self.box_id = box_id
        self.setFocus()

    def clear_selected_box(self, box_id):
        rectangle = self.rectangles[box_id]
        if rectangle:
            rectangle.remove()
            self.rectangles[box_id] = None  # Reset the rectangle for the box ID
            self.canvas.draw()
        self.rectangleCleared.emit(str(box_id), None)
        if box_id == 0:
            self.label_box0.setText("Box 0: None")
        else:
            self.label_box1.setText("Box 1: None")
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
        if self.box_id == 0:
            self.label_box0.setText(f"Box 0: {cropping_coords:04d}")
        else:
            self.label_box1.setText(f"Box 1: {cropping_coords:04d}")
        self.rectangleDrawn.emit(str(self.box_id), cropping_coords)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.select_button0.setChecked(False)
            self.select_button1.setChecked(False)
            self.box_id = None
            self.setFocus()


class ImagePlotter(EvoPanelTemplate):
    def __init__(
            self,
            cam: AbstractCamera,
            automaton: Automaton,
            queue_manager: QueueManager,
            parent: Optional[QWidget] = None,
            width: int = 10,
            height: int = 10,
            dpi: int = 300,
    ):
        super().__init__(automaton=automaton, cam=cam, parent=parent)

        self.fov_ids = []
        self.channel_to_index = self.automaton.get_channel_to_index()
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        self.layout.addWidget(self.canvas, 0, 0, 1, 1)

        self.channel_combo_box = QComboBox()
        self.channel_combo_box.addItems(["None"])
        self.channel_combo_box.currentIndexChanged.connect(self.update_plot)
        self.layout.addWidget(self.make_label("FoV"), 0, 1, 1, 1)
        self.layout.addWidget(self.channel_combo_box, 0, 2, 1, 1)

        self.channel_combo_box = QComboBox()
        self.channel_combo_box.addItems([str(ch) for ch in self.channel_to_index.keys()])
        self.channel_combo_box.currentIndexChanged.connect(self.update_plot)
        self.layout.addWidget(self.make_label("Channel"), 1, 1, 1, 1)
        self.layout.addWidget(self.channel_combo_box, 1, 2, 1, 1)

        self.image_cropping = ImageCroppingBoxes(canvas=self.canvas, ax=self.ax)
        self.layout.addWidget(self.image_cropping, 2, 2, 2, 2)

        self.image_array = np.zeros((1, len(self.channel_indices), *self.cam.cfg.image.shape))
        self.image_time_str = "None"
        self.update_plot()

        queue_manager.register(self.update_image, AutomatonQueueDataType.IMAGE)

    def update_fovs(self, cmd: AutomatonCommand):
        channels_int = [self.channel_to_index[c] for c in cmd.command_args['channels']]
        self.image_array[channels_int, :, :] = cmd.command_data
        self.image_time_str = cmd.get_exec_time()

    def update_image(self, cmd: AutomatonCommand):
        channels_int = [self.channel_to_index[c] for c in cmd.command_args['channels']]
        self.image_array[channels_int, :, :] = cmd.command_data
        self.image_time_str = cmd.get_exec_time()

    def update_plot(self):
        channel_index = self.channel_combo_box.currentIndex()
        image_to_plot = self.image_array[channel_index, :, :]
        self.ax.clear()
        self.ax.imshow(image_to_plot, cmap='gray')
        self.ax.set_title(self.image_time_str)
        self.canvas.draw()


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

