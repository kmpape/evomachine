from __future__ import annotations

from PyQt5.QtWidgets import (
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from evomachine.types import LEDType


LED_GROUPS = (
    (
        "SyncBoard",
        (
            LEDType.LED_385_NM,
            LEDType.LED_450_NM,
            LEDType.LED_515_NM,
            LEDType.LED_565_NM,
            LEDType.LED_645_NM,
        ),
    ),
    ("ASI Tiger", (LEDType.LED_OVERHEAD_TIGER,)),
    ("KWR103", (LEDType.LED_OVERHEAD,)),
)

LED_LABELS = {
    LEDType.LED_385_NM: "385",
    LEDType.LED_450_NM: "450",
    LEDType.LED_515_NM: "515",
    LEDType.LED_565_NM: "565",
    LEDType.LED_645_NM: "645",
    LEDType.LED_OVERHEAD_TIGER: "Tiger overhead",
    LEDType.LED_OVERHEAD: "KWR103 overhead",
}

MANUAL_BRIGHTNESS_MAX = 29.0
MANUAL_BRIGHTNESS_DEFAULT = 29.0


class LedManagerPanel(QGroupBox):
    """Simple LEDManager control panel."""

    def __init__(self, controller, parent: QWidget | None = None):
        super().__init__("LEDManager", parent)
        self.controller = controller
        self.led_buttons: dict[LEDType, QPushButton] = {}
        self.brightness_inputs: dict[LEDType, QDoubleSpinBox] = {}
        self.available_leds: set[LEDType] = set()
        self.devices_initialised = False
        self.state_label = QLabel("Run Initialise Devices before using LED controls.")

        refresh_button = QPushButton("Refresh")
        refresh_button.setEnabled(False)
        self.refresh_button = refresh_button

        led_layout = QVBoxLayout()
        for group_name, led_types in LED_GROUPS:
            group_label = QLabel(group_name)
            group_label.setStyleSheet("font-weight: 600;")
            led_layout.addWidget(group_label)
            led_layout.addLayout(self._build_led_button_grid(led_types))

        buttons = QGridLayout()
        buttons.addWidget(refresh_button, 0, 0)

        layout = QVBoxLayout()
        layout.addLayout(led_layout)
        layout.addLayout(buttons)
        layout.addWidget(self.state_label)
        self.setLayout(layout)

        refresh_button.clicked.connect(self.controller.refresh_leds)
        self.controller.led_list_received.connect(self.update_leds)
        self.controller.led_state_received.connect(self.update_state)
        self.controller.lifecycle_status_received.connect(self.update_lifecycle_status)

    def _build_led_button_grid(self, led_types: tuple[LEDType, ...]) -> QGridLayout:
        grid = QGridLayout()
        grid.addWidget(QLabel("LED"), 0, 0)
        grid.addWidget(QLabel("Brightness"), 0, 1)
        for index, led_type in enumerate(led_types):
            button = QPushButton(LED_LABELS.get(led_type, led_type.name))
            button.setCheckable(True)
            button.setEnabled(False)
            button.setToolTip(led_type.name)
            button.toggled.connect(lambda checked, selected=led_type: self._toggle_led(selected, checked))

            brightness_input = self._brightness_input()
            brightness_input.setEnabled(False)
            brightness_input.valueChanged.connect(lambda _value, selected=led_type: self._update_active_led(selected))

            self.led_buttons[led_type] = button
            self.brightness_inputs[led_type] = brightness_input

            row = index + 1
            grid.addWidget(button, row, 0)
            grid.addWidget(brightness_input, row, 1)
        return grid

    @staticmethod
    def _brightness_input() -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setRange(0.0, MANUAL_BRIGHTNESS_MAX)
        box.setDecimals(0)
        box.setSingleStep(1.0)
        box.setValue(MANUAL_BRIGHTNESS_DEFAULT)
        return box

    def _toggle_led(self, led_type: LEDType, checked: bool) -> None:
        if not self.devices_initialised:
            self._set_led_checked(led_type, False)
            self.state_label.setText("Run Initialise Devices before using LED controls.")
            return
        if led_type not in self.available_leds:
            return
        if checked:
            self._set_led(led_type)
            return
        self.controller.disable_led(led=led_type.name)
        self.state_label.setText(f"Disabling {self._format_led(led_type)}")

    def _set_led(self, led_type: LEDType) -> None:
        self.controller.set_led(
            led=led_type.name,
            brightness=self.brightness_inputs[led_type].value(),
            duration=None,
        )
        self.state_label.setText(f"Setting {self._format_led(led_type)}")

    def _update_active_led(self, led_type: LEDType) -> None:
        button = self.led_buttons[led_type]
        if self.devices_initialised and button.isEnabled() and button.isChecked():
            self._set_led(led_type)

    def update_leds(self, leds: list[str]) -> None:
        self.available_leds = {self._led_type_from_name(led) for led in leds}
        self._sync_controls_enabled()
        self._update_status_label()

    def update_lifecycle_status(self, payload: dict) -> None:
        self.devices_initialised = bool(payload.get("devices_initialised"))
        if payload.get("shutdown"):
            self.devices_initialised = False
        self._sync_controls_enabled()
        self._update_status_label()

    def _sync_controls_enabled(self) -> None:
        self.refresh_button.setEnabled(self.devices_initialised)
        for led_type, button in self.led_buttons.items():
            is_enabled = self.devices_initialised and led_type in self.available_leds
            button.setEnabled(is_enabled)
            self.brightness_inputs[led_type].setEnabled(is_enabled)
            if not is_enabled:
                self._set_led_checked(led_type, False)

    def update_state(self, state: dict) -> None:
        led_type = self._led_type_from_name(state.get("led"))
        if led_type in self.led_buttons:
            brightness = state.get("brightness")
            if state.get("is_on") is True and isinstance(brightness, int | float):
                self._set_brightness_value(led_type, float(brightness))
            if state.get("is_on") is False:
                self._set_led_checked(led_type, False)
            elif state.get("is_on") is True:
                self._set_led_checked(led_type, True)
        self.state_label.setText(
            f"{self._format_led(led_type)}: brightness {state.get('brightness')}, on {state.get('is_on')}"
        )

    def _set_led_checked(self, led_type: LEDType, checked: bool) -> None:
        button = self.led_buttons[led_type]
        was_blocked = button.blockSignals(True)
        button.setChecked(checked)
        button.blockSignals(was_blocked)

    def _set_brightness_value(self, led_type: LEDType, brightness: float) -> None:
        brightness_input = self.brightness_inputs[led_type]
        was_blocked = brightness_input.blockSignals(True)
        brightness_input.setValue(min(brightness, MANUAL_BRIGHTNESS_MAX))
        brightness_input.blockSignals(was_blocked)

    def _update_status_label(self) -> None:
        if not self.devices_initialised:
            self.state_label.setText("Run Initialise Devices before using LED controls.")
            return
        if self.available_leds:
            self.state_label.setText("Press an LED button to enable or disable it")
        else:
            self.state_label.setText("Refresh LEDs to enable available channels")

    @staticmethod
    def _led_type_from_name(led: str | LEDType | None) -> LEDType:
        if isinstance(led, LEDType):
            return led
        if isinstance(led, str):
            return LEDType[led]
        raise ValueError(f"Unknown LED value: {led!r}")

    @staticmethod
    def _format_led(led_type: LEDType) -> str:
        return LED_LABELS.get(led_type, led_type.name)
