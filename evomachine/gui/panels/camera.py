from __future__ import annotations

from PyQt5.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class CameraPanel(QGroupBox):
    """Low-level camera controls."""

    def __init__(self, controller, parent: QWidget | None = None):
        super().__init__("Camera", parent)
        self.controller = controller
        self.devices_initialised = False
        self.status_label = QLabel("Run Initialise Devices before using camera controls.")
        self.exposure_label = QLabel("exposure: -")
        self.image_label = QLabel("image: -")
        self.readout_label = QLabel("readout: -")
        self.exposure_input = self._exposure_input()

        self.refresh_button = QPushButton("Refresh")
        self.set_exposure_button = QPushButton("Set Exposure")
        self.acquire_frame_button = QPushButton("Acquire Frame")

        form = QFormLayout()
        form.addRow("Exposure ms", self.exposure_input)

        buttons = QGridLayout()
        buttons.addWidget(self.refresh_button, 0, 0)
        buttons.addWidget(self.set_exposure_button, 0, 1)
        buttons.addWidget(self.acquire_frame_button, 1, 0, 1, 2)

        layout = QVBoxLayout()
        layout.addWidget(self.status_label)
        layout.addWidget(self.exposure_label)
        layout.addWidget(self.image_label)
        layout.addWidget(self.readout_label)
        layout.addLayout(form)
        layout.addLayout(buttons)
        self.setLayout(layout)

        self.refresh_button.clicked.connect(self.controller.refresh_camera)
        self.set_exposure_button.clicked.connect(self._set_exposure)
        self.acquire_frame_button.clicked.connect(self._acquire_frame)
        self.controller.camera_status_received.connect(self.update_status)
        self.controller.frame_received.connect(self.update_frame_status)
        self.controller.lifecycle_status_received.connect(self.update_lifecycle_status)
        self.controller.response_error.connect(self._show_error)
        self._sync_controls_enabled()

    @staticmethod
    def _exposure_input() -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setRange(0.1, 600000.0)
        box.setDecimals(1)
        box.setSingleStep(10.0)
        box.setValue(200.0)
        return box

    def _set_exposure(self) -> None:
        if not self.devices_initialised:
            self.status_label.setText("Run Initialise Devices before using camera controls.")
            return
        self.controller.set_camera_exposure(exposure=self.exposure_input.value())

    def _acquire_frame(self) -> None:
        if not self.devices_initialised:
            self.status_label.setText("Run Initialise Devices before using camera controls.")
            return
        self.status_label.setText("Acquiring frame.")
        self.controller.acquire_frame()

    def update_status(self, payload: dict) -> None:
        self.status_label.setText(
            f"initialised: {payload.get('is_initialised')}, alive: {payload.get('is_alive')}"
        )
        exposure = payload.get("exposure")
        default_exposure = payload.get("default_exposure")
        self.exposure_label.setText(f"exposure: {exposure} ms, default: {default_exposure} ms")
        image_shape = payload.get("image_shape")
        self.image_label.setText(f"image: {image_shape}, dtype: {payload.get('dtype')}")
        self.readout_label.setText(f"readout: {payload.get('readout_mode') or '-'}")
        if isinstance(exposure, int | float):
            self._set_exposure_value(float(exposure))

    def update_frame_status(self, payload: dict) -> None:
        self.status_label.setText("Frame acquired.")
        self.image_label.setText(f"image: {payload.get('image_shape')}, dtype: {payload.get('dtype')}")

    def update_lifecycle_status(self, payload: dict) -> None:
        self.devices_initialised = bool(payload.get("devices_initialised"))
        if payload.get("shutdown"):
            self.devices_initialised = False
        self._sync_controls_enabled()
        if not self.devices_initialised:
            self.status_label.setText("Run Initialise Devices before using camera controls.")
        elif self.status_label.text().startswith("Run Initialise Devices"):
            self.status_label.setText("Refresh camera to read status.")

    def _sync_controls_enabled(self) -> None:
        for widget in (
            self.exposure_input,
            self.refresh_button,
            self.set_exposure_button,
            self.acquire_frame_button,
        ):
            widget.setEnabled(self.devices_initialised)

    def _set_exposure_value(self, exposure: float) -> None:
        was_blocked = self.exposure_input.blockSignals(True)
        self.exposure_input.setValue(exposure)
        self.exposure_input.blockSignals(was_blocked)

    def _show_error(self, error: str) -> None:
        if self.status_label.text().startswith("Acquiring frame"):
            self.status_label.setText(error)
