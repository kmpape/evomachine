from __future__ import annotations

import time

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from evomachine.gui.panels.config_dialog import ConfigDialog, ConfigFieldSpec
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

MANUAL_BRIGHTNESS_MAX = 100.0
MANUAL_BRIGHTNESS_DEFAULT = 29.0
HIGH_BRIGHTNESS_THRESHOLD = 29.0
DEFAULT_HIGH_BRIGHTNESS_DURATION_S = 3.0
TIMED_LED_REFRESH_GRACE_MS = 100

WAVELENGTH_INDICATOR_COLOURS = {
    LEDType.LED_385_NM: "#74608f",
    LEDType.LED_450_NM: "#4f7197",
    LEDType.LED_515_NM: "#5f8468",
    LEDType.LED_565_NM: "#a1844f",
    LEDType.LED_645_NM: "#9a5d62",
}


class LedManagerPanel(QGroupBox):
    """Simple LEDManager control panel."""

    def __init__(self, controller, parent: QWidget | None = None):
        super().__init__("LEDManager", parent)
        self.controller = controller
        self.led_buttons: dict[LEDType, QPushButton] = {}
        self.wavelength_indicators: dict[LEDType, QLabel] = {}
        self.brightness_inputs: dict[LEDType, QDoubleSpinBox] = {}
        self.available_leds: set[LEDType] = set()
        self._timed_led_stop_times: dict[LEDType, float] = {}
        self.devices_initialised = False
        self.strategy_running = False
        self.state_label = QLabel("Run Initialise Devices before using LED controls.")

        refresh_button = QPushButton("Refresh")
        refresh_button.setEnabled(False)
        self.refresh_button = refresh_button
        self.configure_button = QPushButton("Configure")
        self.configure_button.setEnabled(False)
        self.custom_duration_checkbox = QCheckBox("Custom high-brightness duration")
        self.custom_duration_checkbox.setEnabled(False)
        self.high_brightness_duration_input = QDoubleSpinBox()
        self.high_brightness_duration_input.setRange(0.1, 3600.0)
        self.high_brightness_duration_input.setDecimals(1)
        self.high_brightness_duration_input.setSingleStep(0.5)
        self.high_brightness_duration_input.setSuffix(" s")
        self.high_brightness_duration_input.setValue(DEFAULT_HIGH_BRIGHTNESS_DURATION_S)
        self.high_brightness_duration_input.setEnabled(False)

        led_layout = QVBoxLayout()
        for group_name, led_types in LED_GROUPS:
            group_label = QLabel(group_name)
            group_label.setStyleSheet("font-weight: 600;")
            led_layout.addWidget(group_label)
            led_layout.addLayout(self._build_led_button_grid(led_types))

        buttons = QGridLayout()
        buttons.addWidget(refresh_button, 0, 0)
        buttons.addWidget(self.configure_button, 0, 1)

        duration_controls = QGridLayout()
        duration_controls.addWidget(self.custom_duration_checkbox, 0, 0, 1, 2)
        duration_controls.addWidget(QLabel("Duration"), 1, 0)
        duration_controls.addWidget(self.high_brightness_duration_input, 1, 1)

        layout = QVBoxLayout()
        layout.addLayout(led_layout)
        layout.addLayout(duration_controls)
        layout.addLayout(buttons)
        layout.addWidget(self.state_label)
        self.setLayout(layout)

        refresh_button.clicked.connect(self.controller.refresh_leds)
        self.configure_button.clicked.connect(self._open_config_dialog)
        self.custom_duration_checkbox.toggled.connect(self._sync_controls_enabled)
        self.controller.led_list_received.connect(self.update_leds)
        self.controller.led_state_received.connect(self.update_state)
        self.controller.lifecycle_status_received.connect(self.update_lifecycle_status)
        self.controller.strategy_status_received.connect(self.update_strategy_status)

    def _build_led_button_grid(self, led_types: tuple[LEDType, ...]) -> QGridLayout:
        grid = QGridLayout()
        grid.addWidget(QLabel("LED"), 0, 0, 1, 2)
        grid.addWidget(QLabel("Brightness"), 0, 2)
        for index, led_type in enumerate(led_types):
            button = QPushButton(LED_LABELS.get(led_type, led_type.name))
            button.setCheckable(True)
            button.setEnabled(False)
            button.setToolTip(led_type.name)
            button.toggled.connect(
                lambda checked, selected=led_type: self._toggle_led(selected, checked)
            )

            indicator = self._wavelength_indicator(led_type)
            self.wavelength_indicators[led_type] = indicator

            brightness_input = self._brightness_input()
            brightness_input.setEnabled(False)
            brightness_input.valueChanged.connect(
                lambda _value, selected=led_type: self._update_active_led(selected)
            )

            self.led_buttons[led_type] = button
            self.brightness_inputs[led_type] = brightness_input

            row = index + 1
            grid.addWidget(indicator, row, 0)
            grid.addWidget(button, row, 1)
            grid.addWidget(brightness_input, row, 2)
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
        brightness = self.brightness_inputs[led_type].value()
        self.controller.set_led(
            led=led_type.name,
            brightness=brightness,
            duration=self._high_brightness_duration_ms(led_type, brightness),
        )
        self.state_label.setText(f"Setting {self._format_led(led_type)}")

    def _high_brightness_duration_ms(
            self,
            led_type: LEDType,
            brightness: float,
    ) -> float | None:
        if (
            led_type not in WAVELENGTH_INDICATOR_COLOURS
            or brightness <= HIGH_BRIGHTNESS_THRESHOLD
            or not self.custom_duration_checkbox.isChecked()
        ):
            return None
        return self.high_brightness_duration_input.value() * 1000.0

    def _update_active_led(self, led_type: LEDType) -> None:
        button = self.led_buttons[led_type]
        if self.devices_initialised and button.isEnabled() and button.isChecked():
            self._set_led(led_type)

    def _open_config_dialog(self) -> None:
        if not self.devices_initialised:
            self.state_label.setText("Run Initialise Devices before using LED controls.")
            return
        dialog = ConfigDialog(
            title="LED Manager Configuration",
            fields=self._config_fields(),
            parent=self,
        )
        dialog.exec_()

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

    def update_strategy_status(self, payload: dict) -> None:
        self.strategy_running = bool(payload.get("running"))
        self._sync_controls_enabled()

    def _sync_controls_enabled(self) -> None:
        self.refresh_button.setEnabled(self.devices_initialised)
        manual_controls_enabled = self.devices_initialised and not self.strategy_running
        self.configure_button.setEnabled(manual_controls_enabled)
        self.custom_duration_checkbox.setEnabled(manual_controls_enabled)
        self.high_brightness_duration_input.setEnabled(
            manual_controls_enabled and self.custom_duration_checkbox.isChecked()
        )
        for led_type, button in self.led_buttons.items():
            is_available = self.devices_initialised and led_type in self.available_leds
            is_enabled = manual_controls_enabled and led_type in self.available_leds
            button.setEnabled(is_enabled)
            self.brightness_inputs[led_type].setEnabled(is_enabled)
            if not is_available:
                self._set_led_checked(led_type, False)

    def update_state(self, state: dict) -> None:
        led_type = self._led_type_from_name(state.get("led"))
        if led_type in self.led_buttons:
            brightness = state.get("brightness")
            if state.get("is_on") is True and isinstance(brightness, int | float):
                self._set_brightness_value(led_type, float(brightness))
            if state.get("is_on") is False:
                self._set_led_checked(led_type, False)
                self._timed_led_stop_times.pop(led_type, None)
            elif state.get("is_on") is True:
                self._set_led_checked(led_type, True)
                self._schedule_timed_state_refresh(
                    led_type=led_type, stop_time=state.get("stop_time")
                )
        if state.get("is_on") is True and isinstance(state.get("stop_time"), int | float):
            self.state_label.setText(
                f"{self._format_led(led_type)}: brightness {state.get('brightness')}, "
                "timed illumination active"
            )
        else:
            self.state_label.setText(
                f"{self._format_led(led_type)}: brightness {state.get('brightness')}, "
                f"on {state.get('is_on')}"
            )

    def _schedule_timed_state_refresh(self, led_type: LEDType, stop_time: object) -> None:
        if not isinstance(stop_time, int | float):
            return
        self._timed_led_stop_times[led_type] = float(stop_time)
        delay_ms = (
            max(0, int(round((float(stop_time) - time.time()) * 1000))) + TIMED_LED_REFRESH_GRACE_MS
        )
        QTimer.singleShot(
            delay_ms,
            lambda selected=led_type, deadline=float(stop_time): self._refresh_timed_led_state(
                selected, deadline
            ),
        )

    def _refresh_timed_led_state(self, led_type: LEDType, stop_time: float) -> None:
        if self._timed_led_stop_times.get(led_type) != stop_time:
            return
        if self.devices_initialised and led_type in self.available_leds:
            self.controller.refresh_led_state(led=led_type.name)

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

    def _config_fields(self) -> list[ConfigFieldSpec]:
        return [
            ConfigFieldSpec(
                "Available LEDs", "available_leds", self._available_led_labels(), editable=False
            ),
            ConfigFieldSpec(
                "Brightness max", "brightness_max", MANUAL_BRIGHTNESS_MAX, editable=False
            ),
            ConfigFieldSpec(
                "Brightness default",
                "brightness_default",
                MANUAL_BRIGHTNESS_DEFAULT,
                editable=False,
            ),
            ConfigFieldSpec("Timed threshold", "timed_threshold", "> 29", editable=False),
        ]

    def _available_led_labels(self) -> list[str]:
        return [
            self._format_led(led_type)
            for led_type in sorted(self.available_leds, key=lambda led_type: led_type.name)
        ]

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

    @staticmethod
    def _wavelength_indicator(led_type: LEDType) -> QLabel:
        indicator = QLabel()
        indicator.setFixedSize(12, 12)
        colour = WAVELENGTH_INDICATOR_COLOURS.get(led_type)
        if colour is not None:
            indicator.setStyleSheet(
                f"background-color: {colour}; border-radius: 6px;"
            )
            indicator.setToolTip(f"{LED_LABELS[led_type]} nm wavelength")
        return indicator
