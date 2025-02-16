from multiprocessing import Event, Queue
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

from evomachine.config import ConfigCamera, get_logger
from evomachine.config_delta import ConfigImageProcessorFactory
from evomachine.guidir.guitemplates import EvoPanelTemplate, EvoWorkerTemplate, EvoGUIThread
from evomachine.guidir.guitypes import SMALL, CENTER, LEFT, RIGHT, NORMAL
from evomachine.guidir.queuemanager import QueueManager


logger = get_logger(name=__name__, is_gui=True)


class BrightfieldWorker(EvoWorkerTemplate):
    def __init__(
            self,
            buttons: Dict[int, QPushButton],
            queue_manager: QueueManager,
            parent: Optional[QObject] = None,
    ):
        super().__init__(parent)
        self.queue_manager: QueueManager = queue_manager
        self.buttons = buttons

    @pyqtSlot(int)
    def set_brightfield_states(self, i_active: int):
        for i_chan, button in self.buttons.items():
            if i_chan == i_active:
                button.setStyleSheet("background-color: green;")
                button.setText("ON")
            else:
                button.setStyleSheet("background-color: red;")
                button.setText("OFF")


class BrightfieldPanel(EvoPanelTemplate):
    signal_set_brightfield = pyqtSignal(int)

    def __init__(
            self,
            queue_manager: QueueManager,
            camera_config: ConfigCamera,
            processor_config: ConfigImageProcessor,
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

        self.brightfield_channels = [0, 1]

        self.current_brightfield_id: int = 0
        self.brightfield_labels = [self.make_label(
            text=("OFF", "White")[i],
            font=SMALL,
            width_px=100,
        ) for i in self.brightfield_channels]
        self.brightfield_buttons = {i: self.make_button(
            text="OFF",
            func=self.set_brightfield,
            font=SMALL,
            i_channel=i,
            stylesheet="QPushButton {background-color: red;}",
        ) for i in self.brightfield_channels}
        self.brightfield_textinputs = {i: self.make_lineedit(
            text="100",
            func=self.set_brightfield,
            param=i,
        ) for i in self.brightfield_channels}
        self.brightfield_textinputs[0].setText("0")
        self.brightfield_textinputs[0].setReadOnly(True)
        self.brightfield_textinputs[0].setStyleSheet("background-color: LightGray;")
        self.brightfield_buttons[0].setStyleSheet("background-color: green;")
        self.brightfield_buttons[0].setText("ON")
        self.layout = QGridLayout()
        self.layout.addWidget(self.make_label(text="Brightfield Control", font=NORMAL), 0, 0, 1, 2, LEFT)
        _ = [self.layout.addWidget(label, 1, i, CENTER) for i, label in enumerate(self.brightfield_labels, start=1)]
        _ = [self.layout.addWidget(button, 3, i, CENTER) for i, button in enumerate(self.brightfield_buttons.values(), start=1)]
        _ = [self.layout.addWidget(textinput, 2, i, CENTER) for i, textinput in
             enumerate(self.brightfield_textinputs.values(), start=1)]

        self.widget = QWidget()
        self.widget.setLayout(self.layout)

        self.worker = BrightfieldWorker(
            buttons=self.brightfield_buttons,
            queue_manager=self.queue_manager,
        )
        self.signal_set_brightfield.connect(self.worker.set_brightfield_states)
        self.workers.append(self.worker)
        thread = EvoGUIThread()
        self.worker.moveToThread(thread)
        thread.start()
        self.threads.append(thread)

        self.update_brightfield(None, 0)

    def set_brightfield(self, i_channel: int):
        try:
            brightness = float(self.brightfield_textinputs[i_channel].text())
        except ValueError as e:
            print(e)
            logger.warning(f"Could not parse brightness {self.brightfield_textinputs[i_channel].text()}. Defaulting to 50.")
            brightness = 50

        self.queue_manager.request(
            req_str='self.cam.set_brightfield',
            kwargs_dict={'brightness': brightness},  #  kwargs_dict={'i_chan': i_channel, 'brightness': brightness},
            callback=self.update_brightfield,
            callback_args=(i_channel,),
        )
        self.current_brightfield_id = i_channel

    def update_brightfield(self, data: Any, i_channel: int):
        self.signal_set_brightfield.emit(i_channel)
