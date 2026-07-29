from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


@dataclass(frozen=True)
class ConfigFieldSpec:
    """One editable or read-only field in a GUI configuration dialog."""

    label: str
    key: str
    value: Any
    kind: str = "text"
    minimum: float | int | None = None
    maximum: float | int | None = None
    decimals: int = 3
    single_step: float | int = 1
    choices: tuple[str, ...] = ()
    editable: bool = True
    enabled_when_key: str | None = None
    enabled_when_value: Any = True


class ConfigDialog(QDialog):
    """Small modal dialog for editing simple config payload fields."""

    def __init__(
        self,
        title: str,
        fields: list[ConfigFieldSpec],
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle(title)
        self._fields = fields
        self._widgets: dict[str, QWidget] = {}

        form = QFormLayout()
        for field in fields:
            widget = self._make_widget(field)
            self._widgets[field.key] = widget
            form.addRow(field.label, widget)
        self._connect_dependency_widgets()
        self._sync_dependency_states()

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addWidget(buttons)
        self.setLayout(layout)
        self.setMinimumWidth(360)

    def values(self) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for field in self._fields:
            widget = self._widgets[field.key]
            if isinstance(widget, QCheckBox):
                values[field.key] = widget.isChecked()
            elif isinstance(widget, QSpinBox | QDoubleSpinBox):
                values[field.key] = widget.value()
            elif isinstance(widget, QComboBox):
                values[field.key] = widget.currentData()
            elif isinstance(widget, QLineEdit):
                values[field.key] = widget.text()
        return values

    def _make_widget(self, field: ConfigFieldSpec) -> QWidget:
        if field.kind == "bool":
            widget = QCheckBox()
            widget.setChecked(bool(field.value))
            widget.setEnabled(field.editable)
            return widget
        if field.kind == "int":
            widget = QSpinBox()
            widget.setRange(
                int(field.minimum if field.minimum is not None else -2147483647),
                int(field.maximum if field.maximum is not None else 2147483647),
            )
            widget.setSingleStep(int(field.single_step))
            widget.setValue(int(field.value))
            widget.setEnabled(field.editable)
            return widget
        if field.kind == "float":
            widget = QDoubleSpinBox()
            widget.setRange(
                float(field.minimum if field.minimum is not None else -1e12),
                float(field.maximum if field.maximum is not None else 1e12),
            )
            widget.setDecimals(field.decimals)
            widget.setSingleStep(float(field.single_step))
            widget.setValue(float(field.value))
            widget.setEnabled(field.editable)
            return widget
        if field.kind == "choice":
            widget = QComboBox()
            for choice in field.choices:
                widget.addItem(choice, choice)
            index = widget.findData(field.value)
            if index >= 0:
                widget.setCurrentIndex(index)
            widget.setEnabled(field.editable)
            return widget
        widget = QLineEdit(self._format_value(field.value))
        widget.setEnabled(field.editable)
        return widget

    def _connect_dependency_widgets(self) -> None:
        dependency_keys = {
            field.enabled_when_key
            for field in self._fields
            if field.enabled_when_key is not None
        }
        for key in dependency_keys:
            widget = self._widgets.get(key)
            if isinstance(widget, QCheckBox):
                widget.stateChanged.connect(self._sync_dependency_states)
            elif isinstance(widget, QComboBox):
                widget.currentIndexChanged.connect(self._sync_dependency_states)
            elif isinstance(widget, QSpinBox | QDoubleSpinBox):
                widget.valueChanged.connect(self._sync_dependency_states)
            elif isinstance(widget, QLineEdit):
                widget.textChanged.connect(self._sync_dependency_states)

    def _sync_dependency_states(self) -> None:
        for field in self._fields:
            widget = self._widgets[field.key]
            widget.setEnabled(field.editable and self._dependency_is_met(field))

    def _dependency_is_met(self, field: ConfigFieldSpec) -> bool:
        if field.enabled_when_key is None:
            return True
        dependency_widget = self._widgets.get(field.enabled_when_key)
        if dependency_widget is None:
            return True
        return self._widget_value(dependency_widget) == field.enabled_when_value

    @staticmethod
    def _widget_value(widget: QWidget) -> Any:
        if isinstance(widget, QCheckBox):
            return widget.isChecked()
        if isinstance(widget, QSpinBox | QDoubleSpinBox):
            return widget.value()
        if isinstance(widget, QComboBox):
            return widget.currentData()
        if isinstance(widget, QLineEdit):
            return widget.text()
        return None

    @staticmethod
    def _format_value(value: Any) -> str:
        if value is None:
            return "-"
        if isinstance(value, list | tuple):
            return ", ".join(ConfigDialog._format_value(item) for item in value)
        return str(value)
