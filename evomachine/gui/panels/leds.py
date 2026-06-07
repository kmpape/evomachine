from __future__ import annotations

from PyQt5.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class LedManagerPanel(QGroupBox):
    """Simple LEDManager control panel."""

    def __init__(self, controller, parent: QWidget | None = None):
        super().__init__("LEDManager", parent)
        self.controller = controller
        self.led_combo = QComboBox()
        self.brightness_input = QDoubleSpinBox()
        self.brightness_input.setRange(0.0, 100.0)
        self.brightness_input.setDecimals(1)
        self.brightness_input.setValue(10.0)
        self.duration_input = QDoubleSpinBox()
        self.duration_input.setRange(0.0, 600000.0)
        self.duration_input.setDecimals(1)
        self.state_label = QLabel("No LED selected")

        refresh_button = QPushButton("Refresh")
        set_button = QPushButton("Set")
        disable_button = QPushButton("Disable")
        disable_all_button = QPushButton("Disable All")

        form = QFormLayout()
        form.addRow("LED", self.led_combo)
        form.addRow("Brightness", self.brightness_input)
        form.addRow("Duration ms", self.duration_input)

        buttons = QGridLayout()
        buttons.addWidget(refresh_button, 0, 0)
        buttons.addWidget(set_button, 0, 1)
        buttons.addWidget(disable_button, 1, 0)
        buttons.addWidget(disable_all_button, 1, 1)

        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addLayout(buttons)
        layout.addWidget(self.state_label)
        self.setLayout(layout)

        refresh_button.clicked.connect(self.controller.refresh_leds)
        set_button.clicked.connect(self._set_led)
        disable_button.clicked.connect(self._disable_led)
        disable_all_button.clicked.connect(self.controller.disable_all_leds)
        self.controller.led_list_received.connect(self.update_leds)
        self.controller.led_state_received.connect(self.update_state)

    def _selected_led(self) -> str:
        return self.led_combo.currentText()

    def _set_led(self) -> None:
        led = self._selected_led()
        if led:
            duration = self.duration_input.value()
            self.controller.set_led(
                led=led,
                brightness=self.brightness_input.value(),
                duration=None if duration <= 0 else duration,
            )

    def _disable_led(self) -> None:
        led = self._selected_led()
        if led:
            self.controller.disable_led(led=led)

    def update_leds(self, leds: list[str]) -> None:
        current = self.led_combo.currentText()
        self.led_combo.clear()
        self.led_combo.addItems(leds)
        if current in leds:
            self.led_combo.setCurrentText(current)

    def update_state(self, state: dict) -> None:
        self.state_label.setText(
            f"{state.get('led')}: brightness {state.get('brightness')}, on {state.get('is_on')}"
        )

