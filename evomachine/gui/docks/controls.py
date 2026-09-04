from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QGroupBox,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from evomachine.gui.central_workspace import CentralVisualWorkspace
from evomachine.gui.controller import EvoMachineGuiController
from evomachine.gui.panels.acquisition import (
    AcquisitionStatusPanel,
    FrameAcquisitionSettingsPanel,
    ManualAcquisitionPanel,
    OutputDirectoryPanel,
    SavedImageLoaderPanel,
    ZStackSettingsPanel,
)
from evomachine.gui.panels.autofocus import AutofocusPanel
from evomachine.gui.panels.automaton import AutomatonPanel
from evomachine.gui.panels.camera import CameraPanel
from evomachine.gui.panels.dmd import DmdPanel
from evomachine.gui.panels.filterwheel import FilterWheelPanel
from evomachine.gui.panels.leds import LedManagerPanel
from evomachine.gui.panels.software_focus import SoftwareFocusPanel
from evomachine.gui.panels.stage import StagePanel
from evomachine.gui.panels.strategy import AiAssistancePanel, FovSetupPanel, StrategySetupPanel


class EvoMachineControlsDock(QWidget):
    """Napari dock widget containing modular evomachine control panels."""

    def __init__(self, napari_viewer=None):
        super().__init__()
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.viewer = napari_viewer
        self.controller = EvoMachineGuiController()
        application = QApplication.instance()
        if application is not None:
            application.aboutToQuit.connect(self.controller.close)
        self.central_workspace = (
            CentralVisualWorkspace(viewer=self.viewer, controller=self.controller)
            if self.viewer is not None
            else None
        )
        self.controller.probe_image_transport()

        main_controls = QGroupBox("Main Controls")
        main_controls.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_controls_layout = QVBoxLayout()
        main_controls_layout.addWidget(
            self._scrollable_tab(self._build_main_controls_column()),
            stretch=1,
        )
        main_controls.setLayout(main_controls_layout)

        workflow_tabs = QTabWidget()
        workflow_tabs.addTab(self._scrollable_tab(self._build_acquisition_tab()), "Acquisition")
        workflow_tabs.addTab(self._scrollable_tab(self._build_strategy_tab()), "Strategy")

        controls_splitter = QSplitter(Qt.Horizontal)
        controls_splitter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        controls_splitter.setChildrenCollapsible(False)
        controls_splitter.addWidget(main_controls)
        controls_splitter.addWidget(workflow_tabs)
        controls_splitter.setStretchFactor(0, 1)
        controls_splitter.setStretchFactor(1, 1)
        controls_splitter.setSizes([1, 1])

        self.main_controls = main_controls
        self.workflow_tabs = workflow_tabs
        self.controls_splitter = controls_splitter

        layout = QVBoxLayout()
        layout.addWidget(controls_splitter, stretch=1)
        self.setLayout(layout)

    def closeEvent(self, event) -> None:  # noqa: N802
        super().closeEvent(event)

    @staticmethod
    def _scrollable_tab(widget: QWidget) -> QScrollArea:
        EvoMachineControlsDock._make_panel_contents_responsive(widget)
        widget_policy = widget.sizePolicy()
        widget_policy.setHorizontalPolicy(QSizePolicy.Ignored)
        widget.setSizePolicy(widget_policy)
        scroll_area = QScrollArea()
        scroll_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(widget)
        return scroll_area

    @staticmethod
    def _make_panel_contents_responsive(widget: QWidget) -> None:
        for label in widget.findChildren(QLabel):
            label.setWordWrap(True)
            label.setMinimumWidth(0)
            label_policy = label.sizePolicy()
            label_policy.setHorizontalPolicy(QSizePolicy.Ignored)
            label.setSizePolicy(label_policy)
        for button in widget.findChildren(QPushButton):
            button.setMinimumWidth(0)
            button.setMinimumHeight(button.fontMetrics().height() + 10)
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

    def _build_main_controls_column(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(AutomatonPanel(controller=self.controller))
        layout.addWidget(StagePanel(controller=self.controller))
        layout.addWidget(LedManagerPanel(controller=self.controller))
        layout.addWidget(CameraPanel(controller=self.controller))
        layout.addWidget(DmdPanel(controller=self.controller))
        layout.addWidget(FilterWheelPanel(controller=self.controller))
        layout.addWidget(AutofocusPanel(controller=self.controller))
        layout.addWidget(SoftwareFocusPanel(controller=self.controller))
        layout.addStretch(1)
        widget.setLayout(layout)
        return widget

    def _build_acquisition_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout()
        settings_panel = FrameAcquisitionSettingsPanel()
        layout.addWidget(settings_panel)
        layout.addWidget(OutputDirectoryPanel(controller=self.controller))
        layout.addWidget(ManualAcquisitionPanel(
            controller=self.controller,
            settings_provider=settings_panel.payload,
        ))
        layout.addWidget(ZStackSettingsPanel(
            controller=self.controller,
            settings_provider=settings_panel.z_stack_payload,
        ))
        layout.addWidget(SavedImageLoaderPanel(controller=self.controller))
        layout.addWidget(AcquisitionStatusPanel(controller=self.controller))
        layout.addStretch(1)
        widget.setLayout(layout)
        return widget

    def _build_strategy_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(FovSetupPanel(controller=self.controller))
        layout.addWidget(OutputDirectoryPanel(controller=self.controller))
        layout.addWidget(StrategySetupPanel(controller=self.controller))
        layout.addWidget(AiAssistancePanel())
        layout.addStretch(1)
        widget.setLayout(layout)
        return widget
