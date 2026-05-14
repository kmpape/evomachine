from multiprocessing import Event, Queue
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Any
from PyQt5.QtCore import pyqtSignal, pyqtSlot, QEventLoop, QThread, QTimer, QObject, QRegExp, Qt
from PyQt5 import QtGui
from PyQt5.QtGui import QRegExpValidator, QDoubleValidator, QFont, QPalette, QColor, QValidator
from PyQt5.QtWidgets import (
    QWidget, QDialog, QTableWidget, QTableWidgetItem,
    QMainWindow, QApplication,
    QLabel, QLineEdit, QPushButton, QComboBox, QMessageBox,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QSizePolicy, QScrollArea, QFileDialog, QCheckBox
)

from evomachine.config import ConfigCamera, ConfigImageProcessor, get_logger
from evomachine.config_types import DMDCalibConfigType, DMDCalibConfigTypeFactory
from evomachine.guidir.figures import FigureWindow, FigureMultiWindow
from evomachine.guidir.guitemplates import EvoPanelTemplate, EvoWorkerTemplate, EvoGUIThread
from evomachine.guidir.guitypes import ButtonState, DMDModes, SMALL, CENTER, LEFT, RIGHT, NORMAL
from evomachine.guidir.queuemanager import QueueManager


logger = get_logger(name=__name__, is_gui=True)


class DMDWorker(EvoWorkerTemplate):
    def __init__(
            self,
            buttons: dict[int, QPushButton],
            disp_modes_keys: list[int],
            parent: QObject | None = None,
    ):
        super().__init__(parent)
        self.buttons = buttons
        self.disp_modes_keys = disp_modes_keys

    @pyqtSlot(int)
    def set_dmd_states(self, i_active: int):
        for i, button in self.buttons.items():
            if i == i_active:
                button.setStyleSheet("background-color: green;")
            elif i in self.disp_modes_keys:
                button.setStyleSheet("background-color: red;")

    @pyqtSlot()
    def dmd_click_start(self):
        for button in self.buttons.values():
            button.setEnabled(False)

    @pyqtSlot()
    def dmd_click_stop(self):
        for button in self.buttons.values():
            button.setEnabled(True)


class DMDCalibDialog(QDialog):
    def __init__(self, cfg: DMDCalibConfigType):
        super().__init__()

        self.cfg = cfg
        self.override_values = {field: None for field in cfg.__annotations__}

        self.setWindowTitle("Configuration Dialog")

        layout = QVBoxLayout()

        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setRowCount(len(cfg.__annotations__))
        self.table.setHorizontalHeaderLabels(["Config", "Override", "Value"])

        row = 0
        for field, value in cfg.__dict__.items():
            self.table.setItem(row, 0, QTableWidgetItem(field))
            self.table.setItem(row, 1, QTableWidgetItem("None"))
            self.table.setItem(row, 2, QTableWidgetItem(str(value)))
            row += 1

        self.table.itemChanged.connect(self.update_override)

        layout.addWidget(self.table)

        buttons_layout = QVBoxLayout()
        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Cancel")
        ok_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)
        buttons_layout.addWidget(ok_button)
        buttons_layout.addWidget(cancel_button)

        layout.addLayout(buttons_layout)
        self.setLayout(layout)

    def update_override(self, item):
        row = item.row()
        col = item.column()
        if col == 2:  # Value column
            value = item.text()
            self.override_values[self.table.item(row, 0).text()] = value
            self.table.item(row, 1).setText("Overridden")

    def accept(self):
        for field, value in self.override_values.items():
            if value is not None:
                setattr(self.cfg, field, value)
        super().accept()

    def reject(self):
        super().reject()


class DMDPanel(EvoPanelTemplate):
    signal_set_dmd = pyqtSignal(int)
    signal_dmd = pyqtSignal()
    signal_dmd_done = pyqtSignal()
    signal_filename = pyqtSignal(str)

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
        self.calib_config: DMDCalibConfigType = DMDCalibConfigTypeFactory.default()
        # calib_data = (calib_data, homography_mat, homography_mat_inv, calib_file)
        self.calib_data: tuple[list, np.ndarray, np.ndarray, Path] | tuple[None, None, None, None] = None, None, None, None
        self.filename: str | None = None
        self.img_label = self.make_label(text=self.make_save_path_label(str(None)), font=SMALL)
        self.img_load_button = self.make_button(
            text="Load",
            func=self.show_file_dialog,
            font=SMALL,
        )
        tmp_modes = [
            (DMDModes.DISPLAY_NONE.value, "NONE"),
            (DMDModes.DISPLAY_FULL.value, "FULL"),
            (DMDModes.DISPLAY_IMG.value, "IMG"),
            (DMDModes.DISPLAY_CHECKB.value, "CHECKB"),
            (DMDModes.DISPLAY_CALIB.value, "CALIMG"),
            (DMDModes.DISPLAY_ROI.value, "ROI"),
        ]
        self.dmd_buttons = {i: self.make_button(
            text=txt,
            func=self.set_dmd,
            font=SMALL,
            mode=i,
            stylesheet="QPushButton {background-color: red;}",
        ) for i, txt in tmp_modes}
        self.dmd_buttons[DMDModes.DISPLAY_FULL.value].setStyleSheet("background-color: green;")
        self.dmd_init_button = self.make_button(
            text="Initialise",
            func=self.initialise_dmd,
            font=SMALL,
        )
        self.dmd_finalise_button = self.make_button(
            text="Finalise",
            func=self.finalise_dmd,
            font=SMALL,
        )
        self.dmd_calibrate_button = self.make_button(
            text="Calibrate",
            func=self.calibrate_dmd,
            font=SMALL,
        )
        self.dmd_calib_curves_button = self.make_button(
            text="Show Loaded Calibration",
            func=self.show_calibration,
            font=SMALL,
        )
        self.dmd_calib_curves_button.setEnabled(True)

        # DIRTY HACK
        self.fill_y = 1
        self.fill_y_textinputs = self.make_lineedit(
            text="0.1",
            func=self.set_fill_y,
            param=None,
        )

        self.layout = QGridLayout()
        self.layout.addWidget(self.make_label(text="DMD Control", font=NORMAL), 0, 0, 1, 1, LEFT)
        _ = [self.layout.addWidget(button, i+1, 0, CENTER) for i, button in enumerate(self.dmd_buttons.values())]
        self.layout.addWidget(self.dmd_init_button, 1, 2, 1, 1, CENTER)
        self.layout.addWidget(self.dmd_calibrate_button, 1, 3, 1, 1, CENTER)
        self.layout.addWidget(self.dmd_finalise_button, 2, 2, 1, 1, CENTER)
        self.layout.addWidget(self.dmd_calib_curves_button, 2, 3, 1, 1, CENTER)
        self.layout.addWidget(self.img_load_button, 3, 2, 1, 1, CENTER)
        self.layout.addWidget(self.img_label, 3, 3, 1, 1, CENTER)
        # # DIRTY HACK
        # self.layout.addWidget(self.fill_y_textinputs, 4, 0, CENTER)
        self.widget = QWidget()
        self.widget.setLayout(self.layout)

        self.dmd_buttons[max(list(self.dmd_buttons.keys()))+1] = self.dmd_init_button
        self.dmd_buttons[max(list(self.dmd_buttons.keys()))+1] = self.dmd_finalise_button
        self.dmd_buttons[max(list(self.dmd_buttons.keys()))+1] = self.dmd_calibrate_button
        self.dmd_buttons[max(list(self.dmd_buttons.keys()))+1] = self.dmd_calib_curves_button
        self.worker = DMDWorker(
            buttons=self.dmd_buttons,
            disp_modes_keys=[tmp[0] for tmp in tmp_modes],
        )
        self.signal_set_dmd.connect(self.worker.set_dmd_states)
        # TODO this seems to bug on repeated clicks
        # self.signal_dmd.connect(self.worker.dmd_click_start)
        # self.signal_dmd_done.connect(self.worker.dmd_click_stop)
        self.workers.append(self.worker)
        thread = EvoGUIThread()
        self.worker.moveToThread(thread)
        thread.start()
        self.threads.append(thread)

        # self.get_preloaded_calibration()

    # IK: DIRTY HACK
    def set_fill_y(self):
        try:
            self.fill_y = float(self.fill_y_textinputs.text())
            print(f"Setting fill_y to {self.fill_y}")
        except ValueError:
            self.fill_y = 1
            print(f"Cannot parse value")

    def set_dmd(self, mode: int):
        func_dict = {
            DMDModes.DISPLAY_NONE.value: 'self._dmd.display_none',
            DMDModes.DISPLAY_FULL.value: 'self._dmd.display_full',
            DMDModes.DISPLAY_IMG.value: 'self._dmd.display_loaded_image',
            DMDModes.DISPLAY_CHECKB.value: 'self._dmd.display_checkerboard',
            # DMDModes.DISPLAY_CALIB.value: 'self._dmd.display_calibration_image',
            DMDModes.DISPLAY_CALIB.value: 'self._dmd.display_circles',
            DMDModes.DISPLAY_ROI.value: 'self.project_roi',
            # Comment the above and insert your own test functions below
            # DMDModes.DISPLAY_CHECKB.value: 'self.project_custom_1',
            # DMDModes.DISPLAY_CALIB.value: 'self.project_custom_2',
            # DMDModes.DISPLAY_ROI.value: 'self.project_custom_3',
        }
        kwargs_dict = {}
        if func_dict[mode] == "self._dmd.display_circles":
            kwargs_dict = {
                'start_col': self.calib_config.start_col,
                'end_col': self.calib_config.end_col,
                # 'start_row': self.calib_config.start_row,
                # 'end_row': self.calib_config.end_row,
                'start_row': 1250,
                'end_row': 2000,
                'step_row': self.calib_config.step,
                'step_col': self.calib_config.step,
                'radius': self.calib_config.line_width,
            }
        self.queue_manager.request(
            req_str=func_dict[mode],
            kwargs_dict=kwargs_dict,
            callback=self.show_dmd_done,
        )
        self.signal_dmd.emit()
        self.signal_set_dmd.emit(mode)

    def show_file_dialog(self):
        file_dialog = QFileDialog(self)
        self.filename, _ = file_dialog.getOpenFileName(self, 'Select File')
        self.queue_manager.request(
            req_str='self._dmd.load_image',
            kwargs_dict={'filename': self.filename},
            callback=self.show_img_loaded,
        )

    def show_img_loaded(self, data: Any):
        if isinstance(data, Exception):
            self.img_label.setText(self.make_save_path_label("None"))
            self.filename = None
        else:
            self.img_label.setText(self.make_save_path_label(self.filename))
            self.signal_set_dmd.emit(DMDModes.DISPLAY_IMG.value)

    def initialise_dmd(self):
        self.queue_manager.request(
            req_str='self._dmd.initialise',
            kwargs_dict={},
            callback=self.show_dmd_done,
        )
        self.signal_dmd.emit()

    def calibrate_dmd(self):
        self.queue_manager.request(
            req_str='self.dmd_calibrate',
            kwargs_dict={'cfg': self.calib_config},
            callback=self.update_calibration,
        )
        self.signal_dmd.emit()

    def get_preloaded_calibration(self):
        self.queue_manager.request(
            req_str='self._dmd.get_calibration_data',
            kwargs_dict={},
            callback=self.update_calibration,
        )

    def update_calibration(self, data: tuple[list, np.ndarray, np.ndarray, Path] | tuple[None, None, None, None]):
        self.calib_data = data
        self.show_calibration_plot()
        # self.dmd_calib_curves_button.setEnabled(True)
        self.signal_dmd_done.emit()

    def finalise_dmd(self):
        self.queue_manager.request(
            req_str='self._dmd.finalise',
            kwargs_dict={},
            callback=self.show_dmd_done,
        )
        self.signal_dmd.emit()

    def show_dmd_done(self, data: Any):
        self.signal_dmd_done.emit()

    def show_calibration(self):
        logger.debug("Showing calibration data.")
        if self.calib_data[0] is None or not self.calib_data[0]:
            self.get_preloaded_calibration()
            logger.info("show_calibration: Requesting calibration data.")
            return
        self.show_calibration_plot()

    def show_calibration_plot(self):
        if self.calib_data[0] is None:
            logger.error(f"show_calibration_plot: No data received.")
            return
        logger.info("show_calibration_plot: plotting data. This might take a while.")
        data = self.calib_data[0]
        DMD_WIDTH_HEIGHT = (2716, 1600)
        CAM_WIDTH_HEIGHT = (3200, 3200)
        dmd_img = np.ones(DMD_WIDTH_HEIGHT, dtype=np.uint8) * 100
        cam_img = np.ones(CAM_WIDTH_HEIGHT, dtype=np.uint8) * 100
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(dmd_img, cmap='gray')
        axs[1].imshow(cam_img, cmap='gray')
        for i, ((r_dmd, c_dmd), (r_cam, c_cam), _) in enumerate(data):
            marker = str(i)
            _ = axs[0].scatter(c_dmd, r_dmd, marker='$' + marker + '$')
            _ = axs[0].set_title('DMD Points')
            _ = axs[0].set_xlabel('Column')
            _ = axs[0].set_ylabel('Row')
            _ = axs[1].scatter(c_cam, r_cam, marker='$' + marker + '$')
            _ = axs[1].set_title('Camera Points')
            _ = axs[1].set_xlabel('Column')
            _ = axs[1].set_ylabel('Row')
        self.calib_window = FigureWindow(fig=fig, title=f"DMD Calibration Curves: {str(self.calib_data[-1])}")
        self.calib_window.show()
        logger.debug("show_calibration_plot: Showing window.")

    def show_config(self):
        dialog = DMDCalibDialog(cfg=self.calib_config)
        if dialog.exec_():
            logger.info(f"Updated DMD calibration configuration: {dialog.cfg}")
            self.calib_config = dialog.cfg
        else:
            logger.info("Configuration not modified")

    @staticmethod
    def make_save_path_label(s: str) -> str:
        max_len = 30
        return "..." + s[len(s)-max_len+3:] if len(s) > max_len else s
