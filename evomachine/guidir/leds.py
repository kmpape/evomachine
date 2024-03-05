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
from serial import SerialException

from asitiger.errors import Errors as ASIErrors

from evomachine.acquisition import AbstractCamera, EvoCamera
from evomachine.automaton import Automaton
from evomachine.commands import AutomatonCommand, AutomatonCommandType
from evomachine.config import ConfigCamera, ConfigCRISP, ConfigFocus, ConfigImageProcessor, get_logger
from evomachine.coordinates import Coordinate, CoordinateFactory
from evomachine.dmd import DMDControl
from evomachine.exceptions import ConfigError, TigerError
from evomachine.evotypes import LEDType, FocusAlgorithmType
from evomachine.guidir.guitemplates import EvoPanelTemplate, EvoWorkerTemplate, EvoGUIThread
from evomachine.guidir.guitypes import DisplayMode, Direction, ARROW_LEFT, ARROW_RIGHT, ARROW_UP, ARROW_DOWN, AXES, \
    SMALL, CENTER, LEFT, RIGHT, NORMAL
from evomachine.guidir.queuemanager import QueueManager


logger = get_logger(name=__name__, is_gui=True)


class LEDWorker(EvoWorkerTemplate):
    def __init__(
            self,
            buttons: Dict[LEDType, QPushButton],
            parent: Optional[QObject] = None,
    ):
        super().__init__(parent)
        self.buttons = buttons

    @pyqtSlot(LEDType)
    def set_led_states(self, i_active: LEDType):
        for i_chan, button in self.buttons.items():
            if i_chan == i_active:
                button.setStyleSheet("background-color: green;")
                button.setText("ON")
            else:
                button.setStyleSheet("background-color: red;")
                button.setText("OFF")


class LEDPanel(EvoPanelTemplate):
    signal_set_led = pyqtSignal(LEDType)

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

        self.led_channels = self.camera_config.leds

        self.current_led_id: LEDType = LEDType.NO_LED
        self.led_labels = [self.make_label(
            text=str(led).replace('_NM', ' nm').replace('LED_', '').replace('NO_LED', 'OFF'),
            font=SMALL,
            width_px=100,
        ) for led in self.led_channels]
        self.led_buttons = {led: self.make_button(
            text="OFF",
            func=self.set_led,
            font=SMALL,
            i_channel=led,
            stylesheet="QPushButton {background-color: red;}",
        ) for led in self.led_channels}
        self.led_textinputs = {led: self.make_lineedit(
            text="100",
            func=self.set_led,
            param=led,
        ) for led in self.led_channels}
        self.led_textinputs[LEDType.NO_LED].setText("0")
        self.led_textinputs[LEDType.NO_LED].setReadOnly(True)
        self.led_textinputs[LEDType.NO_LED].setStyleSheet("background-color: LightGray;")
        self.led_buttons[LEDType.NO_LED].setStyleSheet("background-color: green;")
        self.led_buttons[LEDType.NO_LED].setText("ON")
        self.layout = QGridLayout()
        self.layout.addWidget(self.make_label(text="LED Control", font=NORMAL), 0, 0, 1, 2, LEFT)
        _ = [self.layout.addWidget(label, 1, i, CENTER) for i, label in enumerate(self.led_labels, start=1)]
        _ = [self.layout.addWidget(button, 3, i, CENTER) for i, button in enumerate(self.led_buttons.values(), start=1)]
        _ = [self.layout.addWidget(textinput, 2, i, CENTER) for i, textinput in
             enumerate(self.led_textinputs.values(), start=1)]

        self.widget = QWidget()
        self.widget.setLayout(self.layout)

        self.worker = LEDWorker(buttons=self.led_buttons)
        self.signal_set_led.connect(self.worker.set_led_states)
        self.workers.append(self.worker)
        thread = EvoGUIThread()
        self.worker.moveToThread(thread)
        thread.start()
        self.threads.append(thread)

        self.update_led(None, LEDType.NO_LED)

    def set_led(self, i_channel: LEDType):
        try:
            brightness = int(self.led_textinputs[i_channel].text())
        except ValueError:
            logger.warning(f"Could not parse brightness {self.led_textinputs[i_channel]}. Defaulting to 50.")
            brightness = 50

        self.queue_manager.request(
            req_str='self.cam.set_led',
            kwargs_dict={'i_chan': i_channel, 'brightness': brightness},
            callback=self.update_led,
            callback_args=(i_channel,),
        )
        self.current_led_id = i_channel

    def update_led(self, data: Any, i_channel: LEDType):
        self.signal_set_led.emit(i_channel)
