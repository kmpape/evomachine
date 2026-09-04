import pytest

from evomachine.bindings.asitiger.stage import FakeTigerStageController, TigerStage
from evomachine.bindings.asitiger.peripheralcontroller import TigerPeripheralController
from evomachine.bindings.virtual.peripheralcontroller import VirtualPeripheralController
from evomachine.bindings.virtual.stage import VirtualStage
from evomachine.coordinates import Coordinate, CoordinateBounds
from evomachine.peripherals.stage import StageConfig, StageFactory
from evomachine.bindings.binding_types import BindingType
from evomachine.types import AxisType, FovDirectionType


# TODO(CODEX): Make these Fake classes import dependent. If some global variable is true, the real classes are imported and the real bindings tested. For security reasons, we need test settings defined somewhere.

def make_test_stage() -> VirtualStage:
    peripheral_ctrl = VirtualPeripheralController()
    peripheral_ctrl.initialise()
    stage = VirtualStage(peripheral_ctrl=peripheral_ctrl, fov_step_size=100.0)
    stage.initialise()
    return stage

def make_tiger_stage() -> TigerStage:
    tiger = FakeTigerStageController()
    peripheral_ctrl = TigerPeripheralController(tiger=tiger)
    peripheral_ctrl.initialise()
    stage = TigerStage(peripheral_ctrl=peripheral_ctrl, fov_step_size=100.0)
    stage.initialise()
    return stage


STAGE_FACTORIES = [make_test_stage, make_tiger_stage]


@pytest.mark.parametrize("make_stage", STAGE_FACTORIES)
def test_stage_initialises_and_reports_coordinates(make_stage):
    stage = make_stage()

    assert stage.is_initialised()
    assert stage.is_alive()
    assert stage.get_coordinates(query_hardware=False) == Coordinate(0, 0, 0)


def test_tiger_stage_converts_between_micrometres_and_tiger_units():
    stage = make_tiger_stage()

    stage.move(Coordinate(346.7, -12.5, 3.2))

    assert stage.peripheral_ctrl.tiger.move_calls[-1] == {
        "X": pytest.approx(3467),
        "Y": pytest.approx(-125),
        "Z": pytest.approx(32),
    }
    assert stage.get_coordinates(query_hardware=True) == Coordinate(
        pytest.approx(346.7),
        pytest.approx(-12.5),
        pytest.approx(3.2),
    )


@pytest.mark.parametrize("make_stage", STAGE_FACTORIES)
def test_stage_old_registered_position_api_is_removed(make_stage):
    """
    Check old registered-position stage API names are absent.

    Parameters
    ----------
    make_stage
        Stage factory fixture parameter.

    Returns
    -------
    None
    """
    stage = make_stage()

    assert not hasattr(stage, "get_pos")
    assert not hasattr(stage, "set_pos_id_to_coordinate")
    assert not hasattr(stage, "UNKNOWN_POSITION_ID")


@pytest.mark.parametrize("make_stage", STAGE_FACTORIES)
def test_stage_move_coordinate_preserves_unset_axes(make_stage):
    stage = make_stage()

    stage.move(Coordinate(10, None, 5))

    assert stage.get_coordinates(query_hardware=False) == Coordinate(10, 0, 5)
    assert stage.get_fov_id() == stage.UNKNOWN_FOV_ID


@pytest.mark.parametrize("make_stage", STAGE_FACTORIES)
def test_stage_move_fov_id(make_stage):
    stage = make_stage()
    assert stage.set_fov_id_to_coordinate({7: Coordinate(20, 30, None)}, use_autofocus=True)

    stage.move(7)

    assert stage.get_fov_id() == 7
    assert stage.get_coordinates(query_hardware=False) == Coordinate(20, 30, 0)


@pytest.mark.parametrize("make_stage", STAGE_FACTORIES)
def test_stage_move_unknown_fov_id_raises(make_stage):
    stage = make_stage()

    with pytest.raises(IndexError):
        stage.move(99)


@pytest.mark.parametrize("make_stage", STAGE_FACTORIES)
def test_stage_fov_list_checks_autofocus_z_rules(make_stage):
    stage = make_stage()

    assert stage.set_fov_id_to_coordinate({1: Coordinate(1, 2, 3)}, use_autofocus=False)
    assert not stage.set_fov_id_to_coordinate({1: Coordinate(1, 2, None)}, use_autofocus=False)
    assert not stage.set_fov_id_to_coordinate({1: Coordinate(1, 2, 3)}, use_autofocus=True)
    assert stage.set_fov_id_to_coordinate({1: Coordinate(1, 2, None)}, use_autofocus=True)


@pytest.mark.parametrize("make_stage", STAGE_FACTORIES)
def test_stage_rejects_out_of_bounds_coordinate(make_stage):
    stage = make_stage()

    with pytest.raises(ValueError):
        stage.move(Coordinate(2e7, None, None))


@pytest.mark.parametrize("make_stage", STAGE_FACTORIES)
def test_stage_move_fov_tuples_use_fov_step_size(make_stage):
    stage = make_stage()

    stage.move([
        (FovDirectionType.RIGHT, 1.0),
        (FovDirectionType.DOWN, 0.5),
        (FovDirectionType.LEFT, 2),
        (FovDirectionType.UP, 1.0),
    ])

    assert stage.get_coordinates(query_hardware=False) == Coordinate(-100, -50, 0)


@pytest.mark.parametrize("make_stage", STAGE_FACTORIES)
def test_stage_move_fov_tuples_combine_repeated_directions(make_stage):
    stage = make_stage()

    stage.move([
        (FovDirectionType.RIGHT, 1.0),
        (FovDirectionType.RIGHT, 0.5),
        (FovDirectionType.DOWN, 2),
    ])

    assert stage.get_coordinates(query_hardware=False) == Coordinate(150, 200, 0)


@pytest.mark.parametrize("make_stage", STAGE_FACTORIES)
def test_stage_move_single_fov_tuple_and_home(make_stage):
    stage = make_stage()
    stage.move(Coordinate(10, 20, 30))

    stage.move((FovDirectionType.LEFT, 1.0))
    assert stage.get_coordinates(query_hardware=False) == Coordinate(-90, 20, 30)

    stage.move((FovDirectionType.HOME, 1.0))
    assert stage.get_coordinates(query_hardware=False) == Coordinate(0, 0, 0)
    assert stage.get_fov_id() == 0


@pytest.mark.parametrize("make_stage", STAGE_FACTORIES)
def test_stage_move_partial_coordinate_preserves_unset_axes(make_stage):
    stage = make_stage()
    stage.move(Coordinate(10, 20, 30))

    stage.move(Coordinate(None, None, 5))

    assert stage.get_coordinates(query_hardware=False) == Coordinate(10, 20, 5)


@pytest.mark.parametrize("make_stage", STAGE_FACTORIES)
def test_stage_move_empty_coordinate_is_noop(make_stage):
    stage = make_stage()
    stage.move(Coordinate(10, 20, 30))

    stage.move(Coordinate.none_coordinate())

    assert stage.get_coordinates(query_hardware=False) == Coordinate(10, 20, 30)


@pytest.mark.parametrize("make_stage", STAGE_FACTORIES)
def test_stage_move_rejects_bad_fov_inputs(make_stage):
    stage = make_stage()

    with pytest.raises(TypeError, match="tuple"):
        stage.move([FovDirectionType.RIGHT])
    with pytest.raises(TypeError, match="FovDirectionType"):
        stage.move([("RIGHT", 1.0)])
    with pytest.raises(TypeError, match="multiplier"):
        stage.move([(FovDirectionType.RIGHT, "bad")])
    with pytest.raises(TypeError, match="multiplier"):
        stage.move([(FovDirectionType.RIGHT, None)])
    with pytest.raises(ValueError, match="positive"):
        stage.move([(FovDirectionType.RIGHT, 0)])
    with pytest.raises(ValueError, match="HOME"):
        stage.move([(FovDirectionType.HOME, 1.0)])
    with pytest.raises(TypeError, match="multiplier"):
        stage.move((FovDirectionType.HOME, None))
    with pytest.raises(TypeError, match="expected int"):
        stage.move(object())


@pytest.mark.parametrize("make_stage", STAGE_FACTORIES)
def test_stage_can_filter_returned_axes(make_stage):
    stage = make_stage()
    stage.move(Coordinate(10, 20, 30))

    coordinate = stage.get_coordinates(axes=[AxisType.X, AxisType.Z], query_hardware=False)

    assert coordinate == Coordinate(10, None, 30)


@pytest.mark.parametrize("make_stage", STAGE_FACTORIES)
def test_stage_home_zero_and_halt(make_stage):
    stage = make_stage()
    stage.move(Coordinate(10, 20, 30))

    stage.move((FovDirectionType.HOME, 1.0))
    assert stage.get_coordinates(query_hardware=False) == Coordinate(0, 0, 0)
    assert stage.get_fov_id() == 0

    stage.move(Coordinate(10, 20, 30))
    stage.zero_coordinates()
    assert stage.get_coordinates(query_hardware=False) == Coordinate(0, 0, 0)
    assert stage.get_fov_id() == stage.UNKNOWN_FOV_ID

    stage.move(Coordinate(5, 6, 7))
    assert stage.get_coordinates(query_hardware=False) == Coordinate(5, 6, 7)

    stage.zero_coordinates()
    assert stage.get_coordinates(query_hardware=False) == Coordinate(0, 0, 0)

    stage.halt()
    if isinstance(stage, VirtualStage):
        assert stage.halt_was_called()
    else:
        assert stage.peripheral_ctrl.tiger.halt_was_called


def test_stage_config_validates_inputs():
    with pytest.raises(TypeError):
        StageConfig(binding="virtual", fov_step_size=1)
    with pytest.raises(ValueError):
        StageConfig(binding=BindingType.VIRTUAL, fov_step_size=0)
    with pytest.raises(TypeError):
        StageConfig(binding=BindingType.VIRTUAL, fov_step_size=1, check_alive="yes")
    with pytest.raises(TypeError):
        StageConfig(binding=BindingType.VIRTUAL, fov_step_size=1, zero_on_initialise="yes")
    with pytest.raises(TypeError):
        StageConfig(
            binding=BindingType.VIRTUAL,
            fov_step_size=1,
            coordinate_bounds="bad",
        )


def test_stage_readiness_checks_can_be_disabled():
    peripheral_ctrl = VirtualPeripheralController()
    stage = VirtualStage(
        peripheral_ctrl=peripheral_ctrl,
        fov_step_size=100.0,
        check_initialised=False,
        check_alive=False,
    )

    stage.move(Coordinate(1, 2, 3))

    assert stage.get_coordinates(query_hardware=False) == Coordinate(1, 2, 3)


def test_stage_query_before_initialise_raises_by_default():
    peripheral_ctrl = VirtualPeripheralController()
    peripheral_ctrl.initialise()
    stage = VirtualStage(peripheral_ctrl=peripheral_ctrl, fov_step_size=100.0)

    with pytest.raises(RuntimeError):
        stage.get_coordinates()


def test_stage_factory_creates_virtual_stage():
    peripheral_ctrl = VirtualPeripheralController()
    peripheral_ctrl.initialise()
    stage = StageFactory.create(
        StageConfig(
            binding=BindingType.VIRTUAL,
            fov_step_size=50.0,
            initial_coordinate=Coordinate(1, 2, 3),
        ),
        peripheral_controllers=peripheral_ctrl,
    )

    assert isinstance(stage, VirtualStage)
    assert stage.get_fov_step_size() == 50.0
    stage.initialise()
    assert stage.get_coordinates(query_hardware=False) == Coordinate(1, 2, 3)


def test_stage_factory_passes_virtual_config_values():
    bounds = CoordinateBounds(low=Coordinate(-5, None, -7), high=Coordinate(5, 6, 7))
    peripheral_ctrl = VirtualPeripheralController()
    peripheral_ctrl.initialise()
    stage = StageFactory.create(
        StageConfig(
            binding=BindingType.VIRTUAL,
            fov_step_size=25.0,
            name="Debug Stage",
            check_initialised=False,
            check_alive=False,
            initial_coordinate=Coordinate(3, 2, 1),
            coordinate_bounds=bounds,
        ),
        peripheral_controllers=peripheral_ctrl,
    )

    assert stage.name == "Debug Stage"
    assert stage.get_fov_step_size() == 25.0
    stage.move(Coordinate(4, 5, 6))
    assert stage.get_coordinates(query_hardware=False) == Coordinate(4, 5, 6)
    assert stage.get_coordinate_bounds() == bounds
    stage.move(Coordinate(4, -1e6, 6))
    assert stage.get_coordinates(query_hardware=False) == Coordinate(4, -1e6, 6)


def test_stage_factory_requires_tiger_peripheral_controller_for_asitiger():
    with pytest.raises(ValueError):
        StageFactory.create(StageConfig(binding=BindingType.ASI_TIGER, fov_step_size=100.0))


def test_stage_factory_rejects_unsupported_shared_binding():
    """Check that shared BindingType values are still scoped per factory."""
    with pytest.raises(ValueError, match="unsupported stage binding"):
        StageFactory.create(StageConfig(binding=BindingType.KWR103, fov_step_size=100.0))


def test_stage_factory_creates_asitiger_stage_with_fake_controller():
    tiger = FakeTigerStageController()
    peripheral_ctrl = TigerPeripheralController(tiger=tiger)
    peripheral_ctrl.initialise()

    stage = StageFactory.create(
        StageConfig(
            binding=BindingType.ASI_TIGER,
            fov_step_size=10.0,
            name="Fake Tiger Stage",
        ),
        peripheral_controllers=peripheral_ctrl,
    )

    assert isinstance(stage, TigerStage)
    assert stage.name == "Fake Tiger Stage"


def test_stage_zeroes_hardware_at_initialisation_and_enforces_configured_bounds() -> None:
    tiger = FakeTigerStageController()
    tiger.coordinates = {"X": 100, "Y": 200, "Z": 300}
    peripheral_ctrl = TigerPeripheralController(tiger=tiger)
    peripheral_ctrl.initialise()
    bounds = CoordinateBounds(
        low=Coordinate(-5, -6, -7),
        high=Coordinate(5, 6, 7),
    )
    stage = StageFactory.create(
        StageConfig(
            binding=BindingType.ASI_TIGER,
            fov_step_size=10,
            coordinate_bounds=bounds,
            zero_on_initialise=True,
        ),
        peripheral_controllers=peripheral_ctrl,
    )

    stage.initialise()

    assert tiger.zero_was_called
    assert stage.get_coordinates(query_hardware=False) == Coordinate(0, 0, 0)
    assert stage.get_coordinate_bounds() == bounds
    with pytest.raises(ValueError, match="out of bounds"):
        stage.move(Coordinate(6, 0, 0))


def test_stage_rejects_relative_fov_move_beyond_configured_bounds() -> None:
    peripheral_ctrl = VirtualPeripheralController()
    peripheral_ctrl.initialise()
    stage = VirtualStage(
        peripheral_ctrl=peripheral_ctrl,
        fov_step_size=2,
        coordinate_bounds=CoordinateBounds(
            low=Coordinate(-1, -1, -1),
            high=Coordinate(1, 1, 1),
        ),
    )
    stage.initialise()

    with pytest.raises(ValueError, match="out of bounds"):
        stage.move((FovDirectionType.RIGHT, 1))
