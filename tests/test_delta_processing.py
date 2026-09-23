"""No models, network or microscope required: exercise the DeLTA/application boundary."""

from types import SimpleNamespace

import numpy as np
import pytest

from autostrat.language.evaluator import CollectionSelection
from evomachine.delta_processing import (
    DeltaProcessor,
    DeltaProcessingError,
    FovMeasurements,
    MicroscopyState,
    MotherMeasurement,
    TargetedProjectionError,
    TrenchState,
)
from evomachine.strategy_generation import (
    AutoStratStrategy,
    MicroscopyCollectionProvider,
    MicroscopyCommandAdapter,
    MicroscopyObservationProvider,
    MicroscopyRuntimeErrorProvider,
)
from evomachine.types import LEDType
from evomachine.runtime_errors import CommandExecutionError
from evomachine.navigation import FocusNavigatorFovRecord, FovConfig
from evomachine.coordinates import Coordinate
from tests.test_automaton import make_automaton, make_cfg, FakeFovProcessor
from tests.test_strategy_generation_integration import _domain, _verified


class FakePosition:
    """Same public processing/lineage interface as PositionRT, deterministic lengths."""

    def __init__(self, **kwargs):
        self.config = kwargs["config"]
        self.frame_id = 0
        self.initialisations = 0
        self.roi_boxes = [object(), object()]
        self.rois = [
            SimpleNamespace(
                lineage=SimpleNamespace(cells={}),
                seg_stack=[np.ones((4, 4), dtype=np.uint8) for _ in range(2)],
                label_stack=[np.ones((4, 4), dtype=np.uint8) for _ in range(2)],
            )
            for _ in range(2)
        ]

    def _measure(self):
        for roi in self.rois:
            length = float(10 + self.frame_id)
            feature = SimpleNamespace(
                length=length, growthrate_length=np.nan if self.frame_id == 0 else 0.1
            )
            roi.lineage.cells[1] = SimpleNamespace(
                frames=range(self.frame_id + 1), features=lambda frame, f=feature: f
            )

    def initialise(self, **kwargs):
        self.initialisations += 1
        if kwargs["seg_model"] is not None:
            self._measure()

    def process_new_frame(self, **kwargs):
        assert kwargs["lineage_enabled"] is True
        self.frame_id += 1
        self._measure()

    def get_frame_id(self):
        return self.frame_id

    def get_seg(self):
        return {i: np.ones((4, 4), dtype=np.uint8) for i in range(2)}


def backend(**kwargs):
    return DeltaProcessor(
        make_cfg(), position_factory=FakePosition, model_loader=lambda name: object(), **kwargs
    )


def process(processor, state, positions, ids, when, fov_id=0):
    return processor.process(
        fov_id=fov_id,
        image=np.ones((1, 8, 8)),
        acquired_at=when,
        processors=positions,
        roi_ids=ids,
        state=state,
        detect_rois=fov_id not in positions,
    )


@pytest.mark.parametrize("duration, valid", [(100, True), (100.1, False), (0, False)])
def test_roi_exposure_limit(duration, valid):
    from autostrat.exceptions import StrategyValidationError

    source = (
        "initialise\nstep\n    loop fovs:\n"
        f"        project_selected(illumination_led=385nm, illumination_brightness=10, duration={duration})\n"
        "    terminate\nfinalise\n"
    )
    if valid:
        _verified(source)
    else:
        with pytest.raises(StrategyValidationError):
            _verified(source)


def test_stable_trenches_multiple_fovs_and_finite_measurements():
    processor = backend(selected_targets={0: {1}})
    state, positions, ids = MicroscopyState(), {}, {}
    process(processor, state, positions, ids, 1)
    process(processor, state, positions, ids, 2)
    process(processor, state, positions, ids, 3, fov_id=4)
    assert ids == {0: [0, 1], 4: [0, 1]}
    assert positions[0].initialisations == 1
    assert positions[0].config.rotation_correction is False
    assert positions[0].config.drift_correction is False
    trench = state.fovs[0].rois[1]
    assert trench.selected and not state.fovs[0].rois[0].selected
    assert not state.fovs[4].rois[1].selected
    assert trench.measurement.length == 11
    assert trench.previous.length == 10
    assert trench.measurement.growth_rate == 0.1
    assert state.fovs[4].rois[0].measurement.growth_rate is None


def test_missing_mother_is_invalid_not_zero_or_last_good_value():
    processor, state, positions, ids = backend(), MicroscopyState(), {}, {}
    process(processor, state, positions, ids, 1)
    positions[0]._measure = lambda: positions[0].rois[0].lineage.cells.clear()
    process(processor, state, positions, ids, 2)
    assert not state.fovs[0].rois[0].measurement.valid
    assert state.fovs[0].rois[0].measurement.length is None
    provider = MicroscopyCollectionProvider()
    provider.bind(fovs={0: None}, region_of_interests=ids, fov_processors=positions)
    provider.bind_processing_state(state)
    values = provider.observe((CollectionSelection("fovs", 0), CollectionSelection("rois", 0)))
    assert values["measurement_valid"] is False
    assert "length" not in values and "growth_rate" not in values


def test_processing_failure_invalidates_state_and_blocks_retry():
    processor, state, positions, ids = backend(), MicroscopyState(), {}, {}
    process(processor, state, positions, ids, 1)
    positions[0].process_new_frame = lambda **kwargs: (_ for _ in ()).throw(ValueError("bad model"))
    with pytest.raises(DeltaProcessingError, match="bad model"):
        process(processor, state, positions, ids, 2)
    assert not state.fovs[0].valid
    with pytest.raises(DeltaProcessingError, match="new experiment"):
        process(processor, state, positions, ids, 3)


def test_empty_detection_remains_an_empty_collection_on_later_frames():
    class EmptyPosition(FakePosition):
        def initialise(self, **kwargs):
            self.rois = []
            self.roi_boxes = []

        def process_new_frame(self, **kwargs):
            pytest.fail("Do not call DeLTA's concatenate-on-empty segmentation path")

    processor = DeltaProcessor(
        make_cfg(), position_factory=EmptyPosition, model_loader=lambda name: object()
    )
    state, positions, ids = MicroscopyState(), {}, {}
    assert process(processor, state, positions, ids, 1) == {}
    assert process(processor, state, positions, ids, 2) == {}
    assert ids == {0: []} and state.fovs[0].valid


def test_undeclared_target_id_fails_closed():
    with pytest.raises(DeltaProcessingError, match="not detected"):
        process(backend(selected_targets={0: {999}}), MicroscopyState(), {}, {}, 1)


def test_treatment_comparisons_exclude_baseline_and_invalid_gaps():
    trench = TrenchState()
    trench.update(MotherMeasurement(0, 0, 10))
    trench.treated(1)
    trench.update(MotherMeasurement(1, 2, 11))
    assert trench.comparison_count == 0
    trench.update(MotherMeasurement(2, 3, 12))
    trench.update(MotherMeasurement(3, 4, 11))
    trench.update(MotherMeasurement(4, 5))
    trench.update(MotherMeasurement(5, 6, 14))
    assert (trench.comparison_count, trench.length_increase_count) == (2, 1)
    trench.treated(7)
    assert (trench.treatment_count, trench.last_treatment_time) == (2, 7)
    assert trench.comparison_count == trench.length_increase_count == 0


@pytest.mark.parametrize("segment_first", [True, False])
def test_real_positionrt_with_synthetic_segmentation_outputs(segment_first, monkeypatch):
    """Exercise real ROI/lineage API without model files or learned predictions."""
    from delta.imgops import CroppingBox
    from delta.rt import ROIRT

    original_init = ROIRT.__init__

    def poisoned_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # Reproduce dirty np.empty allocations deterministically, on all platforms.
        for mask in (*self.seg_stack, *self.label_stack):
            mask.fill(1)

    monkeypatch.setattr(ROIRT, "__init__", poisoned_init)

    cfg = make_cfg()
    cfg.cfg_delta.target_size_seg = (16, 64)
    cfg.cfg_delta.target_size_track = (16, 64)
    cfg.cfg_delta.crop_windows = False
    cfg.cfg_delta.min_cell_area = 3

    class Model:
        def predict(self, inputs, **kwargs):
            output = np.full(inputs.shape, -1.0, dtype=np.float32)
            output[:, 4:12, 6:26, :] = 1
            return output

    processor = DeltaProcessor(cfg, model_loader=lambda name: Model())
    state, positions, ids = MicroscopyState(), {}, {}
    image = np.arange(96 * 512, dtype=np.uint16).reshape(1, 96, 512)
    for acquired_at in (1, 5, 17):
        processor.process(
            fov_id=0,
            image=image,
            acquired_at=acquired_at,
            processors=positions,
            roi_ids=ids,
            state=state,
            detect_rois=acquired_at == 1,
            segment=segment_first or acquired_at != 1,
            roi_boxes=[CroppingBox(xtl=200, xbr=400, ytl=20, ybr=36)],
        )
    assert ids[0] == [0]
    assert state.fovs[0].rois[0].measurement.valid
    assert state.fovs[0].rois[0].measurement.growth_rate is not None
    assert state.fovs[0].rois[0].measurement.time == 17


def test_detection_only_then_segmentation_and_redetection_reset():
    loaded = []
    processor = backend(selected_targets={0: {1}})
    processor._model_loader = lambda name: loaded.append(name) or object()
    state, positions, ids = MicroscopyState(), {}, {}

    def acquire(detect_rois, segment):
        return processor.process(
            fov_id=0, image=np.ones((1, 8, 8)), acquired_at=1,
            processors=positions, roi_ids=ids, state=state,
            detect_rois=detect_rois, segment=segment,
        )

    assert acquire(True, False) == {}
    assert loaded == ["rois"]
    assert ids[0] == [0, 1]
    assert state.fovs[0].valid
    assert state.fovs[0].rois[1].measurement is None
    first_position = positions[0]
    for roi in first_position.rois:
        assert all(not np.any(mask) for mask in (*roi.seg_stack, *roi.label_stack))
    acquire(False, True)
    assert positions[0] is first_position
    assert state.fovs[0].rois[1].measurement.valid
    state.fovs[0].rois[1].treated(1)
    state.fovs[0].projection_targets.add(1)
    acquire(True, False)
    assert positions[0] is not first_position
    assert not state.fovs[0].projection_targets
    assert state.fovs[0].measurement_time is None
    for trench in state.fovs[0].rois.values():
        assert trench.treatment_count == 0
        assert trench.last_treatment_time is None
        assert trench.measurement is None and trench.previous is None
        assert trench.comparison_count == trench.length_increase_count == 0
        assert not trench.selected


def test_segmentation_does_not_silently_detect_rois():
    processor, state, positions, ids = backend(), MicroscopyState(), {}, {}
    with pytest.raises(DeltaProcessingError, match="requires existing ROIs"):
        processor.process(
            fov_id=0, image=np.ones((1, 8, 8)), acquired_at=1,
            processors=positions, roi_ids=ids, state=state, segment=True,
        )
    assert not positions and not ids


def test_acquisition_only_loads_no_models():
    processor, state = backend(), MicroscopyState()
    processor._model_loader = lambda name: pytest.fail("Unexpected model load")
    assert processor.process(
        fov_id=0, image=np.ones((1, 8, 8)), acquired_at=1,
        processors={}, roi_ids={}, state=state, detect_rois=False, segment=False,
    ) == {}


def targeted_command(automaton):
    automaton.focus_nav.move(fov_id=0)
    automaton._fov_processors[0] = FakeFovProcessor()
    automaton._fov_to_roi[0] = [0]
    automaton.processing_state.fovs[0] = FovMeasurements(rois={0: TrenchState()}, valid=True)
    factory = automaton._strategy.command_factory
    command = factory.command_project_roi(
        channel=LEDType.LED_385_NM, fov_id=0, roi_ids=[0], duration=0.001, brightness=10
    )
    command.command_args["record_treatment"] = True
    return command


def test_targeted_projection_records_only_confirmed_completion():
    automaton, _, _, _, leds, _ = make_automaton()
    command = targeted_command(automaton)
    automaton._execute_project_roi(command)
    trench = automaton.processing_state.fovs[0].rois[0]
    assert trench.treatment_count == 1 and trench.last_treatment_time is not None
    assert leds.disable_count >= 1
    automaton.sleep = lambda **kwargs: automaton._stop_event.set()
    with pytest.raises(TargetedProjectionError, match="interrupted"):
        automaton._execute_project_roi(command)
    assert trench.treatment_count == 1


def test_failed_or_wrong_fov_projection_does_not_record_or_retry():
    automaton, _, _, _, leds, dmd = make_automaton()
    command = targeted_command(automaton)
    automaton.focus_nav.move(fov_id=1)
    with pytest.raises(TargetedProjectionError):
        automaton._execute_project_roi(command)
    assert not dmd.images and not leds.set_calls
    automaton.focus_nav.move(fov_id=0)
    automaton.sleep = lambda **kwargs: (_ for _ in ()).throw(ConnectionError("lost connection"))
    with pytest.raises(TargetedProjectionError) as captured:
        automaton._execute_project_roi(command)
    assert automaton.processing_state.fovs[0].rois[0].treatment_count == 0
    assert leds.disable_count >= 1
    failure = CommandExecutionError(
        command_id=command.command_id,
        command_type=command.command_type,
        command_args=command.command_args,
        lifecycle_section="step",
        original_error=captured.value,
    )
    classified = MicroscopyRuntimeErrorProvider().classify(
        errors=[failure], command_origins={command.command_id: object()}
    )
    assert "targeted_projection_failed" in classified
    assert _domain().runtime_errors["targeted_projection_failed"].action == "abort"


@pytest.mark.parametrize("separate_detection", [False, True])
def test_processed_image_populates_collections_and_projects_one_pattern_per_fov(separate_detection):
    automaton, *_ = make_automaton(delta_processor=backend())

    def move(fov_id, manage_focus=True):
        automaton.focus_nav.current_fov_id = fov_id
        return FocusNavigatorFovRecord(
            fov_id=fov_id,
            coordinate=Coordinate(fov_id, 0, 0),
            fov_config=FovConfig(),
            is_locked=True,
        )

    automaton.focus_nav.move = move
    image_commands = (
        "        image(detect_rois=true, segment=false, exposure=25, led=450nm, led_brightness=10, filter=465nm)\n"
        "        image(detect_rois=false, segment=true, exposure=25, led=450nm, led_brightness=10, filter=465nm)\n"
        if separate_detection else
        "        image(detect_rois=true, segment=true, exposure=25, led=450nm, led_brightness=10, filter=465nm)\n"
    )
    strategy = AutoStratStrategy(
        cfg=make_cfg().updated(
            preproc_enabled=True, seg_enabled=True, track_enabled=True, lineage_enabled=True
        ),
        domain=_domain(),
        verified=_verified(
            "initialise\nstep\n    loop fovs:\n        move_fov(target=current_fov)\n"
            f"{image_commands}"
            "        clear_projection_targets()\n        loop rois:\n"
            "            if observations.measurement_valid:\n                select_roi()\n                select_roi()\n"
            "        project_selected(illumination_led=385nm, illumination_brightness=10, duration=0.001)\n"
            "        project_selected(illumination_led=385nm, illumination_brightness=10, duration=0.001)\n    terminate\nfinalise\n"
        ),
        command_adapter=MicroscopyCommandAdapter(segment_images=False, save_images=False),
        collection_provider=MicroscopyCollectionProvider(),
        observation_provider=MicroscopyObservationProvider(),
        runtime_error_provider=MicroscopyRuntimeErrorProvider(),
    )
    automaton.set_strategy(strategy)
    for _ in range(30):
        automaton._process()
        if automaton.stopped():
            break
    assert automaton.strategy_has_stopped()
    assert strategy.failure_history == ()
    assert automaton._fov_to_roi == {0: [0, 1], 1: [0, 1]}
    assert len(automaton._dmd.images) == 2
    assert len(automaton._led_mngr.set_calls) == 2  # One exposure, not one per ROI.
    for fov in automaton.processing_state.fovs.values():
        assert fov.valid
        assert not fov.projection_targets
        assert all(roi.treatment_count == 1 for roi in fov.rois.values())
        assert len({roi.last_treatment_time for roi in fov.rois.values()}) == 1
    assert strategy._execution.variables == {}  # section locals do not leak into finalise


def test_selection_buffers_are_independent_and_cleared_by_acquisition():
    state = MicroscopyState()
    state.fovs[0] = FovMeasurements(projection_targets={0, 1}, valid=True)
    state.fovs[1] = FovMeasurements(projection_targets={2}, valid=True)
    state.invalidate(0)
    assert state.fovs[0].projection_targets == set()
    assert state.fovs[1].projection_targets == {2}


def test_explicit_clear_and_unauthorised_selection_never_expose():
    automaton, *_ = make_automaton()
    command = targeted_command(automaton)
    del command
    fov = automaton.processing_state.fovs[0]
    fov.projection_targets.add(0)
    factory = automaton._strategy.command_factory
    automaton.next_commands = [factory.command_projection_selection(0)]
    automaton._execute_strategy_batch()
    assert not fov.projection_targets
    fov.rois[0].selected = False
    automaton.next_commands = [factory.command_projection_selection(0, 0)]
    automaton._execute_strategy_batch()
    assert not fov.projection_targets
    assert not automaton._dmd.images and not automaton._led_mngr.set_calls
    assert automaton.runtime_failure_history


def test_pack_has_no_scheduler_or_experiment_specific_guidance():
    domain = _domain()
    assert "schedule_fov" not in domain.commands
    assert "next_imaging_time" not in domain.all_observations
    guidance = str(domain.semantic_guidance)
    assert "45-minute" not in guidance and "2700" not in guidance
    assert "paper criterion" not in guidance
