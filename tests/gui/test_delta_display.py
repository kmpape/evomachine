"""Tests for polling existing DeLTA state and displaying FOV-specific results."""

from dataclasses import dataclass, field
import json
import os
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pytest

from delta.imgops import CroppingBox, affine_transform, correct_drift
from evomachine.gui.central_workspace import CentralVisualWorkspace, LAST_IMAGE_LAYER, ROI_BOXES_LAYER
from evomachine.gui.controller import EvoMachineGuiController
from evomachine.gui.facade import AutomatonGuiFacade
from evomachine.gui.image_payloads import array_to_preview_payload, stack_to_preview_payload
from evomachine.gui.protocol import GuiCommandType, GuiRequest, GuiResponse
from tests.gui.test_controller import RecordingClient
from tests.test_automaton import make_automaton


@dataclass
class DisplayViewer:
    """Minimal viewer storing camera and shape layers for presentation assertions."""

    layers: dict[str, Any] = field(default_factory=dict)
    grid: Any = field(default_factory=lambda: SimpleNamespace(enabled=False))

    def add_image(self, data, *, name, **kwargs):
        """Store image data and keyword display settings, returning the new layer."""
        layer = SimpleNamespace(data=data, **kwargs)
        self.layers[name] = layer
        return layer

    def add_shapes(self, data, *, name, **kwargs):
        """Store ROI geometry and keyword style settings, returning the new layer."""
        layer = SimpleNamespace(data=data, **kwargs)
        self.layers[name] = layer
        return layer


def delta_entry(fov_id=0, *, status="ready", angle=0.0, drift=(0.0, 0.0)):
    """Return a wire entry for FOV ID, status, angle in degrees and pixel drift."""
    return {
        "fov_id": fov_id, "status": status, "image_shape": [12, 16],
        "rotation_degrees": angle, "drift_xy": drift,
        "boxes": [{"roi_id": 7, "xtl": 2, "ytl": 3, "xbr": 8, "ybr": 6}]
        if status == "ready" else [],
    }


@pytest.fixture
def display_workspace():
    """Return a workspace using real Qt controller signals and a fake viewer."""
    controller = EvoMachineGuiController(client=RecordingClient(), start_worker=False)
    workspace = CentralVisualWorkspace(DisplayViewer(), controller)
    # Avoid composing the unrelated dashboard in every test redraw.
    workspace._refresh = lambda: None
    return workspace


def test_delta_rpc_reads_existing_state_and_clears_on_failure_and_reset():
    """Read live processor attributes, preserving valid zeros and empty results."""
    automaton, *_ = make_automaton()
    position = SimpleNamespace(
        roi_boxes=[CroppingBox(xtl=2, ytl=3, xbr=8, ybr=6)],
        config=SimpleNamespace(rotation_correction=True, drift_correction=True),
        rotate=0.0, drift_values=[(0, 0)], reference=np.zeros((1, 12, 16)),
    )
    automaton._fov_processors[0] = position
    automaton._fov_to_roi[0] = [0]
    facade = AutomatonGuiFacade(automaton)
    # Read-only polling must remain allowed during strategy execution.
    automaton._start_strategy_event.set()
    request = GuiRequest(command=GuiCommandType.DELTA_STATUS)
    response = facade.handle(request)
    assert response.ok
    entry, absent = response.payload["delta"]["fovs"]
    assert entry["rotation_degrees"] == 0.0 and entry["drift_xy"] == [0.0, 0.0]
    assert entry["boxes"][0]["roi_id"] == 0
    assert absent["status"] == "not_run" and absent["boxes"] == []
    json.dumps(response.to_dict(), allow_nan=False)
    position.rotate = 3.5
    position.drift_values.append((2, -1))
    assert facade.handle(request).payload["delta"]["fovs"][0]["drift_xy"] == [2.0, -1.0]
    position.rotate = float("nan")
    position.drift_values.append((float("inf"), 0))
    entry = facade.handle(request).payload["delta"]["fovs"][0]
    assert entry["rotation_degrees"] is None and entry["drift_xy"] is None
    # Empty detection can return before a drift history is initialised.
    del position.drift_values
    del position.rotate
    entry = facade.handle(request).payload["delta"]["fovs"][0]
    assert entry["rotation_degrees"] is None and entry["drift_xy"] is None
    position.config.rotation_correction = False
    position.config.drift_correction = False
    entry = facade.handle(request).payload["delta"]["fovs"][0]
    assert entry["rotation_degrees"] is None and entry["drift_xy"] is None
    position.roi_boxes = []
    automaton._fov_to_roi[0] = []
    assert facade.handle(request).payload["delta"]["fovs"][0]["status"] == "empty"
    automaton._delta_processor._failed.add(0)
    entry = facade.handle(request).payload["delta"]["fovs"][0]
    assert entry["status"] == "failed" and not entry["boxes"]
    automaton._delta_processor.reset()
    automaton._fov_processors.clear()
    assert facade.handle(request).payload["delta"]["fovs"][0]["status"] == "not_run"
    assert automaton._delta_processor._roi_model is None


def test_delta_poll_routing_pending_suppression_and_error_recovery():
    """Poll independently, suppress concurrent polls, and clear unavailable results."""
    controller = EvoMachineGuiController(client=RecordingClient(), start_worker=False)
    requests, results = [], []
    controller._worker = object()  # Queue requests through signals without a socket.
    controller.request_ready.connect(requests.append)
    controller.delta_status_received.connect(results.append)
    assert controller._delta_poll_timer.interval() == 250
    controller.refresh_delta_status()
    controller.refresh_delta_status()
    assert len(requests) == 1
    assert requests[0].command is GuiCommandType.DELTA_STATUS
    controller._handle_response(GuiResponse(
        request_id=requests[0].request_id, ok=True, payload={"delta": {"fovs": [delta_entry()]}},
    ))
    assert results[-1]["fovs"][0]["fov_id"] == 0
    controller.refresh_delta_status()
    assert len(requests) == 2
    controller._handle_request_failure(requests[-1], "unavailable")
    assert results[-1] == {"fovs": []}
    controller.refresh_delta_status()
    assert len(requests) == 3
    controller._closed = True
    controller._request_commands.clear()
    controller.refresh_delta_status()
    assert len(requests) == 3


@pytest.mark.parametrize("results_first", [True, False])
def test_roi_overlay_matches_image_and_clears(display_workspace, results_first):
    """Match FOV results in either arrival order and clear stale boxes.

    Args:
        display_workspace: Image workspace fixture.
        results_first: Whether processing results precede the displayed image.
    """
    workspace = display_workspace
    payload = {"fov_id": 0, "preview": array_to_preview_payload(
        np.zeros((12, 16), dtype=np.uint16), transport="raw",
    )}
    workspace.set_delta_display_options(show_boxes=True, rotation=False, drift=False)
    if results_first:
        workspace.update_delta_status({"fovs": [delta_entry()]})
        workspace.update_frame(payload)
    else:
        workspace.update_frame(payload)
        workspace.update_delta_status({"fovs": [delta_entry()]})
    boxes = workspace.viewer.layers[ROI_BOXES_LAYER]
    assert boxes.visible and not boxes.editable
    assert boxes.properties["roi_id"].tolist() == [7]
    assert boxes.data == [[[3, 2], [3, 8], [6, 8], [6, 2]]]
    assert boxes.scale == workspace.viewer.layers[LAST_IMAGE_LAYER].scale[-2:]
    assert boxes.translate == workspace.viewer.layers[LAST_IMAGE_LAYER].translate[-2:]
    workspace.update_delta_status({"fovs": [delta_entry(1)]})
    assert not boxes.visible
    workspace.update_frame({**payload, "fov_id": 1})
    assert boxes.visible
    workspace.update_delta_status({"fovs": [delta_entry(1, status="empty")]})
    assert not boxes.visible and boxes.data == []
    workspace.update_delta_status({"fovs": [delta_entry(1)]})
    assert boxes.visible
    workspace.update_frame({"preview": payload["preview"]})
    assert not boxes.visible
    workspace.update_delta_status({"fovs": []})
    assert not workspace.delta_results


def test_mismatched_image_dimensions_suppress_results(display_workspace):
    """Do not draw full-frame boxes or apply corrections to differently sized pixels.

    Args:
        display_workspace: Image workspace fixture.
    """
    workspace = display_workspace
    workspace.update_delta_status({"fovs": [delta_entry(angle=8, drift=[2, 1])]})
    image = np.arange(5 * 6, dtype=np.uint16).reshape(5, 6)
    workspace.update_frame({"fov_id": 0, "preview": array_to_preview_payload(image)})
    workspace.set_delta_display_options(show_boxes=True, rotation=True, drift=True)
    assert workspace.current_delta_result() is None
    assert np.array_equal(workspace.viewer.layers[LAST_IMAGE_LAYER].data, image)
    assert ROI_BOXES_LAYER not in workspace.viewer.layers


@pytest.mark.parametrize("rotation,drift", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("stack", [False, True])
def test_corrections_use_raw_pixels(display_workspace, rotation, drift, stack):
    """Match DeLTA correction order without modifying raw image or stack data.

    Args:
        display_workspace: Image workspace fixture.
        rotation: Whether to apply rotation.
        drift: Whether to apply translation.
        stack: Whether to display multiple planes.
    """
    workspace = display_workspace
    image = np.arange(12 * 16, dtype=np.uint16).reshape(12, 16)
    raw = np.stack([image, image + 50]) if stack else image
    original = raw.copy()
    payload = {"fov_id": 0}
    if stack:
        payload["stack_preview"] = stack_to_preview_payload(raw)
    else:
        payload["preview"] = array_to_preview_payload(raw)
    workspace.update_frame(payload)
    assert not workspace.show_roi_boxes and not workspace.apply_rotation and not workspace.apply_drift
    workspace.update_delta_status({"fovs": [delta_entry(angle=8.5, drift=[2, -1])]})
    workspace.set_delta_display_options(show_boxes=False, rotation=rotation, drift=drift)
    expected = []
    for plane in raw if stack else [raw]:
        result = affine_transform(plane, angle=8.5, order=1, border_mode=cv2.BORDER_CONSTANT) if rotation else plane
        expected.append(correct_drift(result, (2, -1)) if drift else result)
    expected = np.stack(expected) if stack else expected[0]
    assert np.array_equal(workspace.viewer.layers[LAST_IMAGE_LAYER].data, expected)
    workspace.set_delta_display_options(show_boxes=False, rotation=rotation, drift=drift)
    assert np.array_equal(workspace.viewer.layers[LAST_IMAGE_LAYER].data, expected)
    assert np.array_equal(raw, original)
    source = workspace.last_stack if stack else workspace.last_image
    assert np.array_equal(source, original)
    workspace.update_delta_status({"fovs": []})
    assert np.array_equal(workspace.viewer.layers[LAST_IMAGE_LAYER].data, original)


@pytest.mark.skipif(os.environ.get("EVOMACHINE_GUI_RUN_QT_TESTS") != "1", reason="Qt tests are opt-in")
def test_napari_layers_and_display_controls():
    """Exercise real Napari rectangles and Qt toggles, including stack updates."""
    from napari.components import ViewerModel
    from PyQt5.QtWidgets import QApplication
    from evomachine.gui.panels.acquisition import DeltaDisplayPanel

    app = QApplication.instance() or QApplication([])
    viewer = ViewerModel()
    try:
        controller = EvoMachineGuiController(client=RecordingClient(), start_worker=False)
        workspace = CentralVisualWorkspace(viewer, controller)
        panel = DeltaDisplayPanel(controller, workspace)
        assert not panel.boxes_checkbox.isChecked()
        assert not panel.rotation_checkbox.isEnabled() and not panel.drift_checkbox.isEnabled()
        stack = np.arange(2 * 12 * 16, dtype=np.uint16).reshape(2, 12, 16)
        controller.frame_received.emit({"fov_id": 0, "stack_preview": stack_to_preview_payload(stack)})
        controller.delta_status_received.emit({"fovs": [delta_entry()]})
        assert panel.rotation_checkbox.isEnabled() and panel.drift_checkbox.isEnabled()
        panel.boxes_checkbox.setChecked(True)
        layer = viewer.layers[ROI_BOXES_LAYER]
        assert len(layer.data) == 1 and layer.text.values.tolist() == ["7"]
        panel.rotation_checkbox.setChecked(True)
        panel.drift_checkbox.setChecked(True)
        assert viewer.layers[LAST_IMAGE_LAYER].data.shape == stack.shape
        controller.delta_status_received.emit({"fovs": [delta_entry(status="empty", angle=None, drift=None)]})
        assert not layer.visible and len(layer.data) == 0
        assert not panel.rotation_checkbox.isEnabled() and panel.rotation_checkbox.isChecked()
        assert "No ROIs detected" in panel.status_label.text()
        controller.frame_received.emit({"preview": array_to_preview_payload(stack[0])})
        assert "No FOV-associated image" in panel.status_label.text()
        panel.close()
        app.processEvents()
    finally:
        viewer.layers.clear()
