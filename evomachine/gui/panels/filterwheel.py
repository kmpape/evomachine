from __future__ import annotations

from PyQt5.QtWidgets import (
    QButtonGroup,
    QGridLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from evomachine.gui.panels.config_dialog import ConfigDialog, ConfigFieldSpec


MULTIBAND_FILTER_NAME = "BANDPASS_405_445_514_561_640nm"
FILTER_BUTTON_STYLES = {
    MULTIBAND_FILTER_NAME: (
        "QPushButton {"
        " border: 2px solid #8f00ff;"
        " border-radius: 4px;"
        " color: #111111;"
        " background: qlineargradient(x1:0, y1:0, x2:1, y2:0,"
        " stop:0 #8f00ff, stop:0.24 #004cff, stop:0.49 #00e65c,"
        " stop:0.74 #d6e600, stop:1 #ff3030);"
        "}"
        "QPushButton:checked { border: 3px solid #ffffff; }"
    ),
    "FILTER": (
        "QPushButton { border: 2px solid #8f00ff; border-radius: 4px; }"
    ),
    "FILTER_465nm": (
        "QPushButton { border: 2px solid #004cff; border-radius: 4px; }"
    ),
    "FILTER_527nm": (
        "QPushButton { border: 2px solid #00e65c; border-radius: 4px; }"
    ),
    "FILTER_592nm": (
        "QPushButton { border: 2px solid #ff9800; border-radius: 4px; }"
    ),
    "NO_FILTER": (
        "QPushButton { border: 2px solid #aab2bd; border-radius: 4px; }"
    ),
    "BLOCKING": (
        "QPushButton { border: 2px solid #30343b; border-radius: 4px; }"
    ),
}


class FilterWheelPanel(QWidget):
    """Low-level filter wheel controls."""

    def __init__(self, controller, parent: QWidget | None = None):
        """Create filter-wheel status and position controls.

        Args:
            controller: Shared GUI controller used for commands and status signals.
            parent: Optional owning Qt widget.
        """
        super().__init__(parent)
        self.controller = controller
        self.devices_initialised = False
        self.strategy_running = False
        self._available_filters: list[dict] = []
        self.current_filter_name: str | None = None
        self.filter_buttons: dict[str, QPushButton] = {}
        self.filter_button_group = QButtonGroup(self)
        self.filter_button_group.setExclusive(True)
        self.status_label = QLabel("Run Initialise Devices before using filter wheel controls.")
        self.status_label.setWordWrap(True)
        self.current_label = QLabel("current: -")
        self.refresh_button = QPushButton("Refresh")
        self.configure_button = QPushButton("Configure")
        self.positions_widget = QWidget()
        self.positions_layout = QGridLayout(self.positions_widget)
        self.positions_layout.setContentsMargins(0, 0, 0, 0)
        self.positions_layout.setColumnStretch(0, 1)

        buttons = QGridLayout()
        buttons.addWidget(self.refresh_button, 0, 0)
        buttons.addWidget(self.configure_button, 0, 1)

        positions_label = QLabel("Positions")
        positions_label.setStyleSheet("font-weight: 600;")
        actions_label = QLabel("Actions")
        actions_label.setStyleSheet("font-weight: 600;")
        layout = QVBoxLayout()
        layout.addWidget(self.status_label)
        layout.addWidget(self.current_label)
        layout.addWidget(positions_label)
        layout.addWidget(self.positions_widget)
        layout.addWidget(actions_label)
        layout.addLayout(buttons)
        layout.addStretch(1)
        self.setLayout(layout)

        self.refresh_button.clicked.connect(self.controller.refresh_filter_wheel)
        self.configure_button.clicked.connect(self._open_config_dialog)
        self.controller.filter_wheel_status_received.connect(self.update_status)
        self.controller.lifecycle_status_received.connect(self.update_lifecycle_status)
        self.controller.strategy_status_received.connect(self.update_strategy_status)
        self.controller.response_error.connect(self._show_error)
        self._sync_controls_enabled()

    def _set_filter_wheel(self, filter_name: str) -> None:
        """Request one filter position selected by its button.

        Args:
            filter_name: Authoritative filter enum name sent to the controller.
        """
        if not self.devices_initialised:
            self.status_label.setText("Run Initialise Devices before using filter wheel controls.")
            return
        self.status_label.setText(
            f"Setting filter wheel to {self._format_filter_name(filter_name)}."
        )
        self.controller.set_filter_wheel(filter_wheel=filter_name)
        if self.current_filter_name is not None:
            self._select_filter(self.current_filter_name)

    def _open_config_dialog(self) -> None:
        if not self.devices_initialised:
            self.status_label.setText("Run Initialise Devices before using filter wheel controls.")
            return
        dialog = ConfigDialog(
            title="Filter Wheel Configuration",
            fields=self._config_fields(),
            parent=self,
        )
        dialog.exec_()

    def update_status(self, payload: dict) -> None:
        self._available_filters = list(payload.get("available_filters", []))
        self._set_available_filters(self._available_filters)
        current_filter = payload.get("current_filter", {})
        current_name = current_filter.get("name")
        if current_name:
            self.current_filter_name = current_name
            self._select_filter(current_name)
        self.status_label.setText(
            f"initialised: {payload.get('is_initialised')}, alive: {payload.get('is_alive')}"
        )
        self.current_label.setText(f"current: {self._format_filter_name(current_name)}")
        self._sync_controls_enabled()

    def update_lifecycle_status(self, payload: dict) -> None:
        self.devices_initialised = bool(payload.get("devices_initialised"))
        if payload.get("shutdown"):
            self.devices_initialised = False
        self._sync_controls_enabled()
        if not self.devices_initialised:
            self.status_label.setText("Run Initialise Devices before using filter wheel controls.")
        elif self.status_label.text().startswith("Run Initialise Devices"):
            self.status_label.setText("Refresh filter wheel to read status.")

    def update_strategy_status(self, payload: dict) -> None:
        self.strategy_running = bool(payload.get("running"))
        self._sync_controls_enabled()

    def _set_available_filters(self, available_filters: list[dict]) -> None:
        """Rebuild checkable buttons from authoritative filter entries."""
        while self.positions_layout.count():
            item = self.positions_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                self.filter_button_group.removeButton(widget)
                widget.deleteLater()
        self.filter_buttons.clear()
        for index, item in enumerate(available_filters):
            name = item.get("name")
            if name:
                button = QPushButton(self._format_filter_name(name))
                button.setCheckable(True)
                button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
                button.setToolTip(self._filter_tooltip(name))
                button.setStyleSheet(FILTER_BUTTON_STYLES.get(name, ""))
                button.clicked.connect(
                    lambda _checked=False, selected=name: self._set_filter_wheel(selected)
                )
                self.filter_button_group.addButton(button)
                self.filter_buttons[name] = button
                self.positions_layout.addWidget(button, index, 0)
        if self.current_filter_name:
            self._select_filter(self.current_filter_name)
        self._sync_controls_enabled()

    def _select_filter(self, filter_name: str) -> None:
        """Check the authoritative position without submitting a command."""
        button = self.filter_buttons.get(filter_name)
        if button is not None:
            blocked = self.filter_button_group.blockSignals(True)
            button.setChecked(True)
            self.filter_button_group.blockSignals(blocked)

    def _sync_controls_enabled(self) -> None:
        self.refresh_button.setEnabled(self.devices_initialised)
        manual_controls_enabled = self.devices_initialised and not self.strategy_running
        for button in self.filter_buttons.values():
            button.setEnabled(manual_controls_enabled)
        self.configure_button.setEnabled(manual_controls_enabled)

    def _show_error(self, error: str) -> None:
        if "filter wheel" in error.lower() or self.status_label.text().startswith(
            "Setting filter wheel"
        ):
            self.status_label.setText(error)

    @staticmethod
    def _format_filter_name(filter_name: str | None) -> str:
        if not filter_name:
            return "-"
        labels = {
            MULTIBAND_FILTER_NAME: "405/445/514/561/640 nm bandpass",
            "FILTER": "405/445/514/561/640 nm bandpass",
            "FILTER_465nm": "465 nm",
            "FILTER_527nm": "527 nm",
            "FILTER_592nm": "592 nm",
            "NO_FILTER": "No filter",
            "BLOCKING": "Blocking",
            "UNKNOWN": "Unknown",
        }
        return labels.get(filter_name, filter_name)

    @staticmethod
    def _filter_tooltip(filter_name: str) -> str:
        """Return a device-specific explanation for one filter position.

        Args:
            filter_name: FilterWheelType member name received from the backend.

        Returns:
            Description displayed when the user hovers over the position button.
        """
        if filter_name in {MULTIBAND_FILTER_NAME, "FILTER"}:
            return (
                "Set the Chroma ZET405/445/514/561/640m multi-bandpass filter "
                "(405, 445, 514, 561, and 640 nm bands)."
            )
        return f"Set the filter wheel to {FilterWheelPanel._format_filter_name(filter_name)}."

    def _config_fields(self) -> list[ConfigFieldSpec]:
        return [
            ConfigFieldSpec(
                "Available filters",
                "available_filters",
                [
                    self._format_filter_name(item.get("name"))
                    for item in self._available_filters
                    if isinstance(item, dict) and item.get("name")
                ],
                editable=False,
                help_text="Read-only list of filter positions reported by the device.",
            ),
            ConfigFieldSpec(
                "Current filter",
                "current_filter",
                self.current_filter_name,
                editable=False,
                help_text="Read-only filter position reported as currently selected.",
            ),
        ]
