from __future__ import annotations

from pathlib import Path

from PyQt5.QtWidgets import QGridLayout, QGroupBox, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget


PATTERN_ACTIONS = (
    ("Clear", "clear"),
    ("Full", "full"),
    ("Checkerboard", "checkerboard"),
    ("Calibration Image", "calibration_image"),
    ("Half", "half"),
    ("Crosshair", "crosshair"),
)

UTILITY_ACTIONS = (
    ("Refresh", "refresh"),
    ("Calibrate", "calibrate"),
)


class DmdPanel(QGroupBox):
    """DMD controls and pattern buttons."""

    def __init__(self, controller, parent: QWidget | None = None):
        super().__init__("DMD", parent)
        self.controller = controller
        self.devices_initialised = False
        self.pattern_buttons: dict[str, QPushButton] = {}
        self.utility_buttons: dict[str, QPushButton] = {}
        self.status_label = QLabel("Run Initialise Devices before using DMD controls.")
        self.status_label.setWordWrap(True)
        self.status_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)

        layout = QVBoxLayout()
        self._add_button_section(layout, "Patterns", PATTERN_ACTIONS, self.pattern_buttons)
        self._add_button_section(layout, "Utilities", UTILITY_ACTIONS, self.utility_buttons)
        layout.addWidget(self.status_label)
        self.setLayout(layout)

        for pattern, button in self.pattern_buttons.items():
            button.clicked.connect(lambda _checked=False, selected=pattern: self._display_pattern(selected))
        self.utility_buttons["refresh"].clicked.connect(self.controller.refresh_dmd)
        self.utility_buttons["calibrate"].clicked.connect(self._calibrate)
        self.utility_buttons["calibrate"].setToolTip("Runs the default DMD calibration workflow.")
        self.controller.dmd_status_received.connect(self.update_status)
        self.controller.lifecycle_status_received.connect(self.update_lifecycle_status)
        self.controller.response_error.connect(self._show_error)
        self._sync_controls_enabled()

    def _add_button_section(
            self,
            layout: QVBoxLayout,
            title: str,
            actions: tuple[tuple[str, str], ...],
            buttons: dict[str, QPushButton],
    ) -> None:
        section_label = QLabel(title)
        section_label.setStyleSheet("font-weight: 600;")
        grid = QGridLayout()
        for index, (button_label, action_name) in enumerate(actions):
            button = QPushButton(button_label)
            button.setProperty("dmd_action", action_name)
            button.setEnabled(False)
            buttons[action_name] = button
            row, column = divmod(index, 2)
            grid.addWidget(button, row, column)
        layout.addWidget(section_label)
        layout.addLayout(grid)

    def _display_pattern(self, pattern: str) -> None:
        if not self.devices_initialised:
            self.status_label.setText("Run Initialise Devices before using DMD controls.")
            return
        self.status_label.setText(f"Displaying {self._format_pattern(pattern)}")
        self.controller.display_dmd_pattern(pattern=pattern)

    def _calibrate(self) -> None:
        if not self.devices_initialised:
            self.status_label.setText("Run Initialise Devices before using DMD controls.")
            return
        self.status_label.setText("Starting DMD calibration.")
        self.controller.calibrate_dmd()

    def update_status(self, payload: dict) -> None:
        last_pattern = payload.get("last_pattern")
        pattern_text = f", pattern: {self._format_pattern(last_pattern)}" if last_pattern else ""
        calibration_file = self._format_calibration_file(payload.get("calibration_file"))
        self.status_label.setText(
            f"initialised: {payload.get('is_initialised')}, alive: {payload.get('is_alive')}, "
            f"calibrated: {payload.get('is_calibrated')}{pattern_text}, calibration: {calibration_file}"
        )

    def update_lifecycle_status(self, payload: dict) -> None:
        self.devices_initialised = bool(payload.get("devices_initialised"))
        if payload.get("shutdown"):
            self.devices_initialised = False
        self._sync_controls_enabled()
        if not self.devices_initialised:
            self.status_label.setText("Run Initialise Devices before using DMD controls.")
        elif self.status_label.text().startswith("Run Initialise Devices"):
            self.status_label.setText("Refresh DMD to read status.")

    def _sync_controls_enabled(self) -> None:
        for button in (*self.pattern_buttons.values(), *self.utility_buttons.values()):
            button.setEnabled(self.devices_initialised)

    def _show_error(self, error: str) -> None:
        text = self.status_label.text()
        if "dmd" in error.lower() or "calibrat" in error.lower() or text.startswith("Starting DMD"):
            self.status_label.setText(error)

    @staticmethod
    def _format_pattern(pattern: str | None) -> str:
        if pattern is None:
            return "-"
        labels = {action_name: label for label, action_name in PATTERN_ACTIONS}
        return labels.get(pattern, pattern)

    @staticmethod
    def _format_calibration_file(calibration_file: str | None) -> str:
        if not calibration_file:
            return "-"
        return Path(calibration_file).name
