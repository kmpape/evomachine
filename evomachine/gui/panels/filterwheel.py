from __future__ import annotations

from PyQt5.QtWidgets import (
    QComboBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from evomachine.gui.panels.config_dialog import ConfigDialog, ConfigFieldSpec


class FilterWheelPanel(QGroupBox):
    """Low-level filter wheel controls."""

    def __init__(self, controller, parent: QWidget | None = None):
        super().__init__("Filter Wheel", parent)
        self.controller = controller
        self.devices_initialised = False
        self._available_filters: list[dict] = []
        self.status_label = QLabel("Run Initialise Devices before using filter wheel controls.")
        self.status_label.setWordWrap(True)
        self.current_label = QLabel("current: -")
        self.filter_combo = QComboBox()
        self.refresh_button = QPushButton("Refresh")
        self.set_button = QPushButton("Set Position")
        self.configure_button = QPushButton("Configure")

        buttons = QGridLayout()
        buttons.addWidget(self.refresh_button, 0, 0)
        buttons.addWidget(self.set_button, 0, 1)
        buttons.addWidget(self.configure_button, 1, 0, 1, 2)

        layout = QVBoxLayout()
        layout.addWidget(self.status_label)
        layout.addWidget(self.current_label)
        layout.addWidget(self.filter_combo)
        layout.addLayout(buttons)
        self.setLayout(layout)

        self.refresh_button.clicked.connect(self.controller.refresh_filter_wheel)
        self.set_button.clicked.connect(self._set_filter_wheel)
        self.configure_button.clicked.connect(self._open_config_dialog)
        self.controller.filter_wheel_status_received.connect(self.update_status)
        self.controller.lifecycle_status_received.connect(self.update_lifecycle_status)
        self.controller.response_error.connect(self._show_error)
        self._sync_controls_enabled()

    def _set_filter_wheel(self) -> None:
        if not self.devices_initialised:
            self.status_label.setText("Run Initialise Devices before using filter wheel controls.")
            return
        filter_name = self.filter_combo.currentData()
        if not filter_name:
            self.status_label.setText("Refresh filter wheel to load available positions.")
            return
        self.status_label.setText(
            f"Setting filter wheel to {self._format_filter_name(filter_name)}."
        )
        self.controller.set_filter_wheel(filter_wheel=filter_name)

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

    def _set_available_filters(self, available_filters: list[dict]) -> None:
        current_data = self.filter_combo.currentData()
        self.filter_combo.clear()
        for item in available_filters:
            name = item.get("name")
            if name:
                self.filter_combo.addItem(self._format_filter_name(name), name)
        if current_data:
            self._select_filter(current_data)

    def _select_filter(self, filter_name: str) -> None:
        index = self.filter_combo.findData(filter_name)
        if index >= 0:
            self.filter_combo.setCurrentIndex(index)

    def _sync_controls_enabled(self) -> None:
        has_filters = self.filter_combo.count() > 0
        self.refresh_button.setEnabled(self.devices_initialised)
        self.filter_combo.setEnabled(self.devices_initialised and has_filters)
        self.set_button.setEnabled(self.devices_initialised and has_filters)
        self.configure_button.setEnabled(self.devices_initialised)

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
            "FILTER": "Filter",
            "FILTER_465nm": "465 nm",
            "FILTER_527nm": "527 nm",
            "FILTER_592nm": "592 nm",
            "NO_FILTER": "No filter",
            "BLOCKING": "Blocking",
            "UNKNOWN": "Unknown",
        }
        return labels.get(filter_name, filter_name)

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
            ),
            ConfigFieldSpec(
                "Current filter", "current_filter", self.filter_combo.currentData(), editable=False
            ),
        ]
