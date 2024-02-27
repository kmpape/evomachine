import queue
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

from evomachine.acquisition import AbstractCamera
from evomachine.automaton import Automaton, AutomatonQueueDataType
from evomachine.config import get_logger
from evomachine.dmd import DMDControl
from evomachine.guidir.guitypes import NORMAL


logger = get_logger(name=__name__)


class EvoGUIThread(QThread):
    def __init__(
            self,
    ):
        super(QThread, self).__init__()
        self._stop_event = threading.Event()

    # def run(self) -> None:
    #     while True:
    #         while not self.stopped():
    #             self._run()
    #
    # def _run(self):
    #     pass

    def sleep(self, duration: float):
        now = time.perf_counter()
        end = now + duration
        while (now < end) and not self.stopped():
            now = time.perf_counter()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()


class QueueManager(EvoGUIThread):  # TODO make this a normal thread?
    def __init__(self, data_queue: queue.Queue, queue_timeout: float = 0):
        super().__init__()
        self.main_queue: queue.Queue = data_queue
        self._listeners: Dict[AutomatonQueueDataType, List[Callable[[Any], None]]] = {
            key: [] for key in AutomatonQueueDataType.get_all()
        }
        self.queue_timeout = queue_timeout

    def register(self, func: Callable[[Any], None], msg_type: AutomatonQueueDataType):
        # TODO merge AutomatonQueueDataType and AutomatonCommandType
        # TODO make these signals?
        self._listeners[msg_type].append(func)

    def run(self) -> None:
        while True:
            while not self.stopped():
                try:
                    tmp_data = self.main_queue.get(block=True, timeout=self.queue_timeout)
                    for func in self._listeners[tmp_data[0]]:
                        func(tmp_data[1])
                except queue.Empty:
                    pass
                if self.queue_timeout:
                    self.sleep(self.queue_timeout)


class EvoPanelTemplate(QWidget):
    def __init__(
            self,
            cam: Optional[AbstractCamera] = None,
            dmd: Optional[DMDControl] = None,
            automaton: Optional[Automaton] = None,
            parent=None,
    ):
        super().__init__(parent)

        self.threads: List[Union[EvoGUIThread, None]] = []
        "List of threads that are running in the panel."
        self.layout: QGridLayout = QGridLayout(self)
        "Layout of the panel."
        self.widget: Union[QWidget, None] = None
        "Widget of the panel."
        self.workers: List[EvoWorkerTemplate] = []
        "List of workers that are running in the panel."

        self.cam: Union[AbstractCamera, None] = cam
        "Camera object."
        self.dmd: Union[DMDControl, None] = dmd
        "DMD object."
        self.automaton: Union[Automaton, None] = automaton
        "Automaton object."

    def clean_threads(self):
        self.threads = [thread for thread in self.threads if thread is not None and not thread.isRunning()]

    def close_threads(self):
        for thread in self.threads:
            if thread is not None:
                thread.stop()
                thread.quit()

    def disable_workers(self):
        for worker in self.workers:
            worker.disable()

    def enable_workers(self):
        for worker in self.workers:
            worker.enable()

    def stop_threads(self):
        for thread in self.threads:
            if thread is not None:
                thread.stop()

    @staticmethod
    def make_button(
            text: str,
            func: Callable,
            font: QFont = NORMAL,
            stylesheet: str = None,
            **kwargs,
    ) -> QPushButton:
        button = QPushButton(text)
        # if kwargs:
        #     button.clicked.connect(lambda: func(**kwargs))
        # else:
        #     button.clicked.connect(func)
        button.clicked.connect(lambda: func(**kwargs))
        button.setFont(font)
        button.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        button.setMinimumSize(button.sizeHint())
        if stylesheet is not None:
            button.setStyleSheet(stylesheet)
        return button

    @staticmethod
    def make_dropdown(
            items: List[str],
            func: Optional[Union[Callable, None]]=None,
    ):
        dropdown = QComboBox()
        dropdown.addItems(items)
        if func:
            dropdown.currentIndexChanged.connect(func)
        return dropdown

    @staticmethod
    def make_lineedit(
            text: Union[str, None],
            func: Optional[Callable] = None,
            validator: Optional[QValidator] = None,
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
        if validator is not None:
            lineedit.setValidator(validator)

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

    @staticmethod
    def make_checkbox(
            text: str,
            func: Callable,
            font: QFont = NORMAL,
            param: Any = None,
            set_true: bool = False,
            stylesheet: str = None,
    ):
        checkbox = QCheckBox(text)
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
    def make_pos_str(value: Union[int, float, None], unit: Optional[str] = None) -> str:
        try:
            if unit == "um":
                f, ustr = 0.1, "\u03BCm"
            elif unit == "mm":
                f, ustr = 0.0001, "mm"
            elif unit == "cm":
                f, ustr = 0.0000001, "cm"
            else:
                f, ustr = 1.0, ""
            return f"+{float(abs(value))*f:.1f}{ustr}" if value > 0 else f"-{float(abs(value))*f:.1f}{ustr}"
        except TypeError as e:
            return "?"*7


class EvoWorkerTemplate(QObject):
    def __init__(self, parent=None):
        super(QObject, self).__init__(parent)
        self._disable: bool = False

    def disable(self):
        self._disable = True

    def enable(self):
        self._disable = False

    def is_disabled(self) -> bool:
        return self._disable

    def set_labels(self, labels: Dict[int, Dict[str, Union[QLabel, QLineEdit]]], text: Optional[str] = "?"):
        for _dict in labels.values():
            for label_or_edit in _dict.values():
                label_or_edit.setText(text)