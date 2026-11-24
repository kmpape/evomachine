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
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from evomachine.gui.panels.common import (
    NoWheelComboBox,
    NoWheelDoubleSpinBox,
    NoWheelSpinBox,
)


@dataclass(frozen=True)
class ConfigFieldSpec:
    """Describe one editable or read-only GUI configuration field.

    Args:
        label: User-facing field name.
        key: Stable payload key returned by the form.
        value: Current field value.
        kind: Widget kind: text, bool, int, float, or choice.
        minimum: Optional inclusive numeric minimum.
        maximum: Optional inclusive numeric maximum.
        decimals: Decimal precision for floating-point fields.
        single_step: Step used by numeric controls.
        choices: Allowed values for choice fields.
        editable: Whether the user may change the value.
        enabled_when_key: Optional controlling field key.
        enabled_when_value: Controlling value which enables this field.
        emphasized: Whether label and value are rendered in bold.
        help_text: Hover text explaining the field's effect and constraints.
    """

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
    emphasized: bool = False
    help_text: str = ""

    def __post_init__(self) -> None:
        """Supply useful range and editability help when none is provided."""
        if self.help_text.strip():
            return
        details = [f"{self.label} configuration value."]
        if not self.editable:
            details.append("Read-only value reported by the current configuration.")
        if self.minimum is not None or self.maximum is not None:
            low = "unbounded" if self.minimum is None else str(self.minimum)
            high = "unbounded" if self.maximum is None else str(self.maximum)
            details.append(f"Accepted range: {low} to {high}.")
        if self.choices:
            details.append(f"Available choices: {', '.join(self.choices)}.")
        object.__setattr__(self, "help_text", " ".join(details))


class ConfigForm(QWidget):
    """Reusable form for editing simple config payload fields."""

    def __init__(
        self,
        fields: list[ConfigFieldSpec],
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self._fields = fields
        self._widgets: dict[str, QWidget] = {}

        form = QFormLayout()
        for field in fields:
            widget = self._make_widget(field)
            self._widgets[field.key] = widget
            label = QLabel(field.label)
            label.setToolTip(field.help_text)
            widget.setToolTip(field.help_text)
            if field.emphasized:
                label_font = label.font()
                label_font.setBold(True)
                label.setFont(label_font)
                widget_font = widget.font()
                widget_font.setBold(True)
                widget.setFont(widget_font)
            form.addRow(label, widget)
        self._connect_dependency_widgets()
        self._sync_dependency_states()

        layout = QVBoxLayout(self)
        layout.addLayout(form)

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
            widget = NoWheelSpinBox()
            widget.setRange(
                int(field.minimum if field.minimum is not None else -2147483647),
                int(field.maximum if field.maximum is not None else 2147483647),
            )
            widget.setSingleStep(int(field.single_step))
            widget.setValue(int(field.value))
            widget.setEnabled(field.editable)
            return widget
        if field.kind == "float":
            widget = NoWheelDoubleSpinBox()
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
            widget = NoWheelComboBox()
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


class ConfigDialog(QDialog):
    """Modal wrapper around the reusable typed field form."""

    def __init__(self, title: str, fields: list[ConfigFieldSpec], parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self._form = ConfigForm(fields)
        self._fields = fields
        self._widgets = self._form._widgets
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(self._form)
        layout.addWidget(buttons)
        self.setMinimumWidth(360)

    def values(self):
        return self._form.values()

    _widget_value = staticmethod(ConfigForm._widget_value)
    _format_value = staticmethod(ConfigForm._format_value)
