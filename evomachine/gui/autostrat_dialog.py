from __future__ import annotations

from PyQt5.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from evomachine.gui.autostrat_config import (
    ANTHROPIC_PROVIDER,
    AUTOSTRAT_CONNECTION_PRESETS,
    CUSTOM_PRESET_ID,
    OPENAI_COMPATIBLE_PROVIDER,
    ROBIN_OFFICE_2_PRESET,
    AutoStratModelConfig,
    autostrat_preset,
)
from evomachine.gui.panels.common import NoWheelComboBox


class AutoStratConfigurationDialog(QDialog):
    """Collect one session's model connection without persisting credentials."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("AutoStrat model configuration")
        self._applying_preset = False
        self.preset_combo = NoWheelComboBox()
        for preset in AUTOSTRAT_CONNECTION_PRESETS:
            self.preset_combo.addItem(preset.label, preset.preset_id)
        self.preset_combo.addItem("Custom", CUSTOM_PRESET_ID)

        self.provider_combo = NoWheelComboBox()
        self.provider_combo.addItem("OpenAI-compatible", OPENAI_COMPATIBLE_PROVIDER)
        self.provider_combo.addItem("Anthropic", ANTHROPIC_PROVIDER)
        self.model_input = QLineEdit()
        self.model_input.setPlaceholderText("Model identifier")
        self.base_url_input = QLineEdit()
        self.base_url_input.setPlaceholderText("Optional API base URL")
        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.Password)
        self.api_key_input.setPlaceholderText("Held in memory for this session only")

        form = QFormLayout()
        form.addRow("Setup", self.preset_combo)
        form.addRow("Provider", self.provider_combo)
        form.addRow("Model", self.model_input)
        form.addRow("Base URL", self.base_url_input)
        form.addRow("API key", self.api_key_input)

        note = QLabel(
            "The API key is sent to the local EvoMachine backend for this session only. "
            "It is not saved or returned in status messages. Leave it blank to keep AI "
            "generation disabled; saved and hand-written strategies still work."
        )
        note.setWordWrap(True)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Apply")
        buttons.button(QDialogButtonBox.Cancel).setText("Continue without AI")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(note)
        layout.addWidget(buttons)
        self.setMinimumWidth(480)

        self.preset_combo.currentIndexChanged.connect(self._apply_selected_preset)
        self.provider_combo.currentIndexChanged.connect(self._provider_changed)
        self.model_input.textEdited.connect(self._mark_custom)
        self.base_url_input.textEdited.connect(self._mark_custom)
        self._set_preset(ROBIN_OFFICE_2_PRESET.preset_id)

    def configuration(self, *, enabled: bool = True) -> AutoStratModelConfig:
        return AutoStratModelConfig(
            preset_id=str(self.preset_combo.currentData()),
            provider=str(self.provider_combo.currentData()),
            model=self.model_input.text(),
            base_url=self.base_url_input.text(),
            api_key=self.api_key_input.text() if enabled else "",
        )

    def accept(self) -> None:
        try:
            self.configuration()
        except (TypeError, ValueError) as error:
            QMessageBox.warning(self, "Invalid AutoStrat configuration", str(error))
            return
        super().accept()

    def _set_preset(self, preset_id: str) -> None:
        index = self.preset_combo.findData(preset_id)
        if index >= 0:
            self.preset_combo.setCurrentIndex(index)
        self._apply_selected_preset()

    def _apply_selected_preset(self) -> None:
        preset = autostrat_preset(str(self.preset_combo.currentData()))
        if preset is None:
            return
        self._applying_preset = True
        try:
            provider_index = self.provider_combo.findData(preset.provider)
            self.provider_combo.setCurrentIndex(provider_index)
            self.model_input.setText(preset.model)
            self.base_url_input.setText(preset.base_url or "")
        finally:
            self._applying_preset = False

    def _mark_custom(self) -> None:
        if self._applying_preset:
            return
        index = self.preset_combo.findData(CUSTOM_PRESET_ID)
        if index >= 0 and self.preset_combo.currentIndex() != index:
            self.preset_combo.blockSignals(True)
            self.preset_combo.setCurrentIndex(index)
            self.preset_combo.blockSignals(False)

    def _provider_changed(self) -> None:
        """Clear incompatible preset values when the provider is changed manually."""
        if self._applying_preset:
            return
        previous_model = self.model_input.text().strip()
        previous_base_url = self.base_url_input.text().strip()
        self._mark_custom()
        if (
                self.provider_combo.currentData() == ANTHROPIC_PROVIDER
                and previous_model == ROBIN_OFFICE_2_PRESET.model
                and previous_base_url == ROBIN_OFFICE_2_PRESET.base_url
        ):
            self.model_input.clear()
            self.base_url_input.clear()
        self.base_url_input.setPlaceholderText(
            "Optional Anthropic API base URL"
            if self.provider_combo.currentData() == ANTHROPIC_PROVIDER
            else "Optional OpenAI-compatible API base URL"
        )


def prompt_autostrat_configuration(controller, parent: QWidget | None = None) -> None:
    dialog = AutoStratConfigurationDialog(parent)
    if dialog.exec_() == dialog.Accepted:
        config = dialog.configuration()
    else:
        config = AutoStratModelConfig(
            preset_id=ROBIN_OFFICE_2_PRESET.preset_id,
            provider=ROBIN_OFFICE_2_PRESET.provider,
            model=ROBIN_OFFICE_2_PRESET.model,
            base_url=ROBIN_OFFICE_2_PRESET.base_url,
        )
    controller.configure_autostrat(
        api_key=config.api_key,
        provider=config.provider,
        model=config.model,
        base_url=config.base_url,
        preset=config.preset_id,
    )
