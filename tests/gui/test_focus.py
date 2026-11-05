"""Focused checks for manual isolation, strategy review and mode-specific forms."""

from concurrent.futures import ThreadPoolExecutor
import os
from time import monotonic, sleep
from types import SimpleNamespace
import threading

import pytest

from evomachine.gui.facade import AutomatonGuiFacade
from evomachine.gui.focus import reference_payload
from evomachine.gui.protocol import GuiCommandType, GuiRequest
from evomachine.gui.request_map import gui_focus_navigator_config_from_payload
from evomachine.softwarefocus import SoftwareFocus, SoftwareFocusConfig
from evomachine.types import FocusAlgorithmType, FocusMode
from tests import test_referencefocus as focus_fakes


@pytest.fixture
def setup():
    return focus_fakes.setup.__wrapped__()


@pytest.fixture
def facade(setup):
    navigator, reference, _, _, _, _ = setup
    navigator.software_focus = SoftwareFocus(
        reference.acquisition_manager,
        SoftwareFocusConfig(
            focus_frames=[reference.config.focus_frame],
            acquisition_settings=None,
            rel_range=100,
            step_size=5,
            algorithm=FocusAlgorithmType.SQUARED_GRAD_AVG,
        ),
    )
    automaton = SimpleNamespace(
        focus_nav=navigator,
        acq_mngr=reference.acquisition_manager,
        strategy_has_started=lambda: False,
        strategy_has_stopped=lambda: False,
        stopped=lambda: False,
        devices_is_initialised=lambda: True,
    )
    return AutomatonGuiFacade(automaton)


def _request(facade, command, **payload):
    response = facade.handle(GuiRequest(command=command, payload=payload))
    assert response.ok, response.error
    return response.payload


def test_manual_settings_do_not_change_strategy(facade):
    nav = facade.automaton.focus_nav
    manual = facade.gui_software_focus()
    assert manual is not nav.software_focus
    _request(
        facade,
        GuiCommandType.SOFTWARE_FOCUS_CONFIGURE,
        config={"rel_range": 20, "algorithm": "LAPLACIAN_VAR"},
    )
    assert nav.software_focus.default_config.rel_range == 100
    assert nav.software_focus.default_config.algorithm == FocusAlgorithmType.SQUARED_GRAD_AVG
    settings = reference_payload(facade.focus_state.reference().config)
    settings["calibration_range"] = 80
    _request(facade, GuiCommandType.REFERENCE_FOCUS_CONFIGURE, config=settings)
    assert nav.reference_focus.config.calibration_range == 100
    defaults = _request(facade, GuiCommandType.FOCUS_DEFAULTS)["focus_defaults"]
    assert defaults["reference_profile"]["calibration_range"] == 100


def test_manual_calibration_refit_and_test_leave_strategy_reference_untouched(facade, setup):
    nav, _, camera, stage, _, _ = setup
    _request(facade, GuiCommandType.REFERENCE_FOCUS_RUN, action="calibrate")
    assert facade.gui_operations.wait_for_completion("reference_focus", 3)
    manual = facade.focus_state.reference()
    assert manual.pending_reference is not None
    assert nav.reference_focus.reference is None
    _request(facade, GuiCommandType.REFERENCE_FOCUS_EDIT, action="refit", smoothing=1)
    _request(facade, GuiCommandType.REFERENCE_FOCUS_EDIT, action="accept")
    camera.focus_z += 7.5
    _request(facade, GuiCommandType.REFERENCE_FOCUS_RUN, action="test")
    assert facade.gui_operations.wait_for_completion("reference_focus", 3)
    assert manual.last_result.focus_status.name == "IN_FOCUS"
    assert not manual.last_result.correction_applied
    assert stage.get_coordinates().z == 120.25
    assert nav.reference_focus.reference is None


def test_strategy_waits_for_current_review_acceptance(facade, setup):
    nav, reference, _, _, _, _ = setup
    nav._fov_states[0].coordinate.z = 120.25
    stop = threading.Event()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(nav.move, 0, stop_event=stop)
        try:
            deadline = monotonic() + 3
            while (
                facade.focus_state.review is None and not future.done() and monotonic() < deadline
            ):
                sleep(0.01)
            review = facade.focus_state.strategy_status()["review"]
            assert review is not None
            assert not future.done()
            response = facade.handle(
                GuiRequest(
                    command=GuiCommandType.STRATEGY_REFERENCE_DECIDE,
                    payload={"id": "stale", "accepted": True},
                )
            )
            assert not response.ok
            _request(
                facade, GuiCommandType.STRATEGY_REFERENCE_DECIDE, id=review["id"], accepted=True
            )
            assert not future.result(timeout=3).skipped
            assert reference.reference.focus_z == 120.25
        finally:
            stop.set()


def test_strategy_scan_services_stop_between_planes(facade, setup):
    nav, reference, _, stage, _, _ = setup
    reference.calibrate()
    reference.accept_reference()
    stop = threading.Event()
    automaton = facade.automaton
    automaton._strategy_processing = True
    automaton._stop_event = stop
    automaton.gui_process_requests = stop.set
    stage.moves.clear()
    result = nav.move(1, stop_event=stop)
    assert result.skipped
    assert len(stage.moves) == 2  # XY arrival, one scan plane, then Stop is serviced.
    assert reference.last_result is None


def test_mode_payload_supports_inheritance_and_full_per_fov_overrides(facade, setup):
    from evomachine.gui.panels.strategy import FovSetupPanel

    config = FovSetupPanel._default_global_focus_config()
    config["focus_mode"] = "REFERENCE_PROFILE"
    config["reference_profile"] = reference_payload(setup[1].config)
    standard, fovs = gui_focus_navigator_config_from_payload(
        facade,
        {
            "focus_config": config,
            "fovs": [
                {
                    "fov_id": 1,
                    "focus_config": {"focus_mode": "STANDARD", "override_settings": False},
                }
            ],
        },
    )
    nav = setup[0]
    nav.update_config(standard)
    nav.update_fov_config(1, fovs[1])
    assert nav.effective_config(0).focus_mode == FocusMode.REFERENCE_PROFILE
    assert nav.effective_config(1).focus_mode == FocusMode.STANDARD
    assert nav.effective_config(1).use_autofocus


@pytest.mark.skipif(os.environ.get("EVOMACHINE_GUI_RUN_QT_TESTS") != "1", reason="Qt opt-in")
def test_mode_editor_retains_values_and_manual_status_does_not_rewrite_strategy():
    from tests.gui.test_panels import _app, FakeController
    from evomachine.gui.panels.focus_settings import FocusSettingsDialog
    from evomachine.gui.panels.strategy import FovSetupPanel

    _app()
    controller = FakeController()
    panel = FovSetupPanel(controller)
    original = dict(panel.global_focus_config["software_focus"])
    controller.software_focus_status_received.emit(
        {"config": {"rel_range": 4, "algorithm": "STEEL"}}
    )
    assert panel.global_focus_config["software_focus"] == original
    dialog = FocusSettingsDialog("Focus", panel.global_focus_config)
    dialog.standard.navigation._widgets["max_refocus_trials"].setValue(3)
    dialog.mode.setCurrentIndex(dialog.mode.findData("REFERENCE_PROFILE"))
    assert dialog.pages.currentWidget() is dialog.reference
    values = dialog.values()
    assert values["focus_mode"] == "REFERENCE_PROFILE"
    assert values["max_refocus_trials"] == 3
    assert values["reference_profile"]["algorithm"] == "SQUARED_GRAD_AVG"
    dialog.mode.setCurrentIndex(dialog.mode.findData("STANDARD"))
    assert dialog.standard.navigation.values()["max_refocus_trials"] == 3
    panel._append_coordinate({"x": 1.25, "y": 2.5, "z": 120.25})
    assert panel._fov_payload()[0]["z"] == 120.25
    dialog.close()
    panel.close()
