from multiprocessing import Event, Queue
from typing import Any
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

from evomachine.config import CameraSystemConfig, ImageProcessorConfig, get_logger
from evomachine.types import FilterWheelType
from evomachine.guidir.guitemplates import EvoPanelTemplate, EvoWorkerTemplate, EvoGUIThread
from evomachine.guidir.guitypes import SMALL, CENTER, LEFT, RIGHT, NORMAL
from evomachine.guidir.queuemanager import QueueManager


logger = get_logger(name=__name__, is_gui=True)


class FilterWorker(EvoWorkerTemplate):
    def __init__(
            self,
            buttons: dict[FilterWheelType, QPushButton],
            parent: QObject | None = None,
    ):
        super().__init__(parent)
        self.buttons = buttons

    @pyqtSlot(FilterWheelType)
    def set_fw_states(self, i_active: FilterWheelType):
        for i_chan, button in self.buttons.items():
            if i_chan == i_active:
                button.setStyleSheet("background-color: green;")
                button.setText("ON")
            else:
                button.setStyleSheet("background-color: red;")
                button.setText("OFF")


class FilterWheelPanel(EvoPanelTemplate):
    signal_set_fw = pyqtSignal(FilterWheelType)

    def __init__(
            self,
            queue_manager: QueueManager,
            camera_config: CameraSystemConfig,
            processor_config: ImageProcessorConfig,
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

        self.filters = self.camera_config.filters

        self.current_fw: FilterWheelType = FilterWheelType.FILTER  # NOTE: This is not checked
        self.fw_labels = [self.make_label(
            text=str(f).replace('_', ' '),
            font=SMALL,
            width_px=100,
        ) for f in self.filters]
        self.fw_buttons = {f: self.make_button(
            text="OFF",
            func=self.set_fw,
            font=SMALL,
            i_channel=f,
            stylesheet="QPushButton {background-color: red;}",
        ) for f in self.filters}
        self.fw_buttons[self.current_fw].setStyleSheet("background-color: green;")
        self.fw_buttons[self.current_fw].setText("ON")
        self.layout = QGridLayout()
        self.layout.addWidget(self.make_label(text="Filter Wheels", font=NORMAL), 0, 0, 1, 2, LEFT)
        _ = [self.layout.addWidget(label, 1, i, CENTER) for i, label in enumerate(self.fw_labels, start=1)]
        _ = [self.layout.addWidget(button, 3, i, CENTER) for i, button in enumerate(self.fw_buttons.values(), start=1)]

        self.widget = QWidget()
        self.widget.setLayout(self.layout)

        self.worker = FilterWorker(buttons=self.fw_buttons)
        self.signal_set_fw.connect(self.worker.set_fw_states)
        self.workers.append(self.worker)
        thread = EvoGUIThread()
        self.worker.moveToThread(thread)
        thread.start()
        self.threads.append(thread)

    def set_fw(self, i_channel: FilterWheelType):
        self.queue_manager.request(
            req_str='self.cam.set_filter_wheel',
            kwargs_dict={'filter_type': i_channel},
            callback=self.update_fw,
            callback_args=(i_channel,),
        )
        self.current_fw = i_channel

    def update_fw(self, data: Any, i_channel: FilterWheelType):
        self.signal_set_fw.emit(i_channel)
