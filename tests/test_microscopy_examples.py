"""Execute the domain's teaching examples against scripted, hardware-free observations."""

from dataclasses import dataclass

import pytest

from autostrat.language.evaluator import StrategyExecution
from autostrat.language.parser import parse_strategy
from autostrat.language.validator import validate_strategy
from tests.test_strategy_generation_integration import _domain


@dataclass
class ExampleHost:
    elapsed: float = 0.0
    steps: int = 0
    saturation: float = 0.0
    valid_measurements: bool = True

    def items(self, collection, context):
        return (0, 1)

    def observe(self, context):
        values = dict(
            elapsed_time=self.elapsed, step_count=self.steps,
            focus_recovery_exhausted=False, saturation_fraction=self.saturation,
        )
        if context:
            values["roi_count"] = 12 if context[0].item_id == 0 else 2
        if len(context) == 2:
            valid = self.valid_measurements and context[-1].item_id == 0
            values.update(
                target_selected=True, measurement_valid=valid,
                previous_measurement_valid=valid,
                treatment_count=1, last_treatment_time=0.0,
                comparison_count=3, length_increase_count=2,
            )
            if valid:
                values.update(
                    length=14.0, previous_length=10.0,
                    measurement_time=2.0, previous_measurement_time=0.0,
                )
        return values


@pytest.mark.parametrize(
    "index,saturation,valid_measurements",
    [(0, 0, True), (1, 0, True), (2, 0, True), (3, 0, True),
     (3, 0, False), (4, 0, True), (5, 0, True), (5, 0.2, True)],
)
def test_example_command_traces(index, saturation, valid_measurements):
    domain = _domain()
    program = validate_strategy(parse_strategy(domain.few_shot_examples[index].strategy), domain)
    execution = StrategyExecution(domain, program)
    host = ExampleHost(saturation=saturation, valid_measurements=valid_measurements)
    trace = []

    def run(section):
        execution.start(section)
        for _ in range(200):
            event = execution.resume(host)
            if event.call is None:
                return event.action
            trace.append((section, host.steps, host.elapsed, event.call, event.context))
            if event.call.name == "wait":
                host.elapsed += event.call.arguments["duration"]
            execution.acknowledge()
        pytest.fail("Example section did not finish")

    run("initialise")
    for _ in range(100):
        action = run("step")
        if action:
            break
        host.steps += 1
    else:
        pytest.fail("Example did not reach its stopping condition")
    assert action == ("abort" if saturation else "terminate")
    if action == "terminate":
        run("finalise")
        assert trace[-1][3].arguments["target"] == "first_fov"
    steps = [entry for entry in trace if entry[0] == "step"]
    images = [entry for entry in steps if entry[3].name == "image"]
    pauses = [entry[3].arguments["duration"] for entry in steps if entry[3].name == "wait"]
    if index == 0:
        assert len(images) == 8
    elif index == 1:
        assert host.elapsed == 60
        assert len(images) == 30
        assert execution.persistent["phase"] == 2
        assert execution.persistent["started_at"] == 30
    elif index == 2:
        assert len(images) == 36
        assert [entry[1] for entry in images if entry[4][0].item_id == 1] == [4, 9, 14, 19, 24, 29]
        assert all(not entry[3].arguments["detect_rois"] for entry in images)
    elif index == 3:
        assert len(images) == 24
        assert pauses == pytest.approx([5 if valid_measurements else 10] * 12)
    elif index == 4:
        selections = [entry for entry in steps if entry[3].name == "select_roi"]
        assert selections and min(entry[2] for entry in selections) == 60
        assert all(entry[4][-1].item_id == 0 for entry in selections)
        assert host.elapsed == 300
    elif index == 5:
        assert len(images) == (1 if saturation else 20)
        if saturation:
            assert pauses == []
            assert not any(entry[0] == "finalise" for entry in trace)
