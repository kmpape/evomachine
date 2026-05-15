import pytest

from evomachine.bindings.asitiger.stage import TigerStage
from evomachine.bindings.asitiger.peripheralcontroller import TigerPeripheralController
from evomachine.bindings.virtual.peripheralcontroller import VirtualPeripheralController
from evomachine.bindings.virtual.stage import VirtualStage
from evomachine.coordinates import Coordinate
from evomachine.stage import StageConfig, StageFactory
from evomachine.types import AxisType, StageBindingType


def make_test_stage() -> VirtualStage:
    peripheral_ctrl = VirtualPeripheralController()
    peripheral_ctrl.initialise()
    stage = VirtualStage(peripheral_ctrl=peripheral_ctrl, delta_fov=100.0)
    stage.initialise()
    return stage


def make_tiger_stage() -> TigerStage:
    pytest.skip("TigerStage tests will be added with a fake TigerController.")


class FakeTigerController:
    def __init__(self):
        self.coordinates = {"X": 0, "Y": 0, "Z": 0}
        self.connection = None

    def status(self):
        return True

    def where(self):
        return self.coordinates.copy()

    def get_stage_limits(self):
        return {"X": (-1000, 1000), "Y": (-1000, 1000), "Z": (-1000, 1000)}

    def move(self, coordinates):
        self.coordinates.update(coordinates)

    def home(self):
        self.coordinates = {"X": 0, "Y": 0, "Z": 0}

    def wait_until_idle(self, card_address_crisp=None):
        return

    def halt(self):
        return

    def zero(self):
        self.coordinates = {"X": 0, "Y": 0, "Z": 0}


STAGE_FACTORIES = [make_test_stage]


@pytest.mark.parametrize("make_stage", STAGE_FACTORIES)
def test_stage_initialises_and_reports_coordinates(make_stage):
    stage = make_stage()

    assert stage.is_initialised()
    assert stage.is_alive()
    assert stage.get_coordinates(query_hardware=False) == Coordinate(0, 0, 0)


@pytest.mark.parametrize("make_stage", STAGE_FACTORIES)
def test_stage_move_to_coordinate_preserves_unset_axes(make_stage):
    stage = make_stage()

    stage.move_to(Coordinate(10, None, 5))

    assert stage.get_coordinates(query_hardware=False) == Coordinate(10, 0, 5)
    assert stage.get_pos() == stage.UNKNOWN_POSITION_ID


@pytest.mark.parametrize("make_stage", STAGE_FACTORIES)
def test_stage_move_to_position_id(make_stage):
    stage = make_stage()
    assert stage.set_pos_id_to_coordinate({7: Coordinate(20, 30, None)}, use_autofocus=True)

    stage.move_to(7)

    assert stage.get_pos() == 7
    assert stage.get_coordinates(query_hardware=False) == Coordinate(20, 30, 0)


@pytest.mark.parametrize("make_stage", STAGE_FACTORIES)
def test_stage_move_to_unknown_position_id_raises(make_stage):
    stage = make_stage()

    with pytest.raises(IndexError):
        stage.move_to(99)


@pytest.mark.parametrize("make_stage", STAGE_FACTORIES)
def test_stage_position_list_checks_autofocus_z_rules(make_stage):
    stage = make_stage()

    assert stage.set_pos_id_to_coordinate({1: Coordinate(1, 2, 3)}, use_autofocus=False)
    assert not stage.set_pos_id_to_coordinate({1: Coordinate(1, 2, None)}, use_autofocus=False)
    assert not stage.set_pos_id_to_coordinate({1: Coordinate(1, 2, 3)}, use_autofocus=True)
    assert stage.set_pos_id_to_coordinate({1: Coordinate(1, 2, None)}, use_autofocus=True)


@pytest.mark.parametrize("make_stage", STAGE_FACTORIES)
def test_stage_rejects_out_of_bounds_coordinate(make_stage):
    stage = make_stage()

    with pytest.raises(ValueError):
        stage.move_to(Coordinate(2e7, None, None))


@pytest.mark.parametrize("make_stage", STAGE_FACTORIES)
def test_stage_fov_moves_use_delta_fov(make_stage):
    stage = make_stage()

    stage.move_fov_right()
    stage.move_fov_down(multiplier=0.5)
    stage.move_fov_left(multiplier=2)
    stage.move_fov_up()

    assert stage.get_coordinates(query_hardware=False) == Coordinate(-100, -50, 0)


@pytest.mark.parametrize("make_stage", STAGE_FACTORIES)
def test_stage_can_filter_returned_axes(make_stage):
    stage = make_stage()
    stage.move_to(Coordinate(10, 20, 30))

    coordinate = stage.get_coordinates(axes=[AxisType.X, AxisType.Z], query_hardware=False)

    assert coordinate == Coordinate(10, None, 30)


@pytest.mark.parametrize("make_stage", STAGE_FACTORIES)
def test_stage_home_zero_and_halt(make_stage):
    stage = make_stage()
    stage.move_to(Coordinate(10, 20, 30))

    stage.move_home()
    assert stage.get_coordinates(query_hardware=False) == Coordinate(0, 0, 0)
    assert stage.get_pos() == 0

    stage.move_to(Coordinate(10, 20, 30))
    stage.zero_coordinates()
    assert stage.get_coordinates(query_hardware=False) == Coordinate(0, 0, 0)
    assert stage.get_pos() == stage.UNKNOWN_POSITION_ID

    stage.halt()
    assert stage.halt_was_called()


def test_stage_readiness_checks_can_be_disabled():
    peripheral_ctrl = VirtualPeripheralController()
    stage = VirtualStage(
        peripheral_ctrl=peripheral_ctrl,
        delta_fov=100.0,
        check_initialised=False,
        check_alive=False,
    )

    stage.move_to(Coordinate(1, 2, 3))

    assert stage.get_coordinates(query_hardware=False) == Coordinate(1, 2, 3)


def test_stage_query_before_initialise_raises_by_default():
    peripheral_ctrl = VirtualPeripheralController()
    peripheral_ctrl.initialise()
    stage = VirtualStage(peripheral_ctrl=peripheral_ctrl, delta_fov=100.0)

    with pytest.raises(RuntimeError):
        stage.get_coordinates()


def test_stage_factory_creates_virtual_stage():
    peripheral_ctrl = VirtualPeripheralController()
    peripheral_ctrl.initialise()
    stage = StageFactory.create(
        StageConfig(
            binding=StageBindingType.VIRTUAL,
            delta_fov=50.0,
            initial_coordinate=Coordinate(1, 2, 3),
        ),
        peripheral_controllers=peripheral_ctrl,
    )

    assert isinstance(stage, VirtualStage)
    assert stage.get_delta_fov() == 50.0
    stage.initialise()
    assert stage.get_coordinates(query_hardware=False) == Coordinate(1, 2, 3)


def test_stage_factory_passes_virtual_config_values():
    limits = (Coordinate(-5, -6, -7), Coordinate(5, 6, 7))
    peripheral_ctrl = VirtualPeripheralController()
    peripheral_ctrl.initialise()
    stage = StageFactory.create(
        StageConfig(
            binding=StageBindingType.VIRTUAL,
            delta_fov=25.0,
            name="Debug Stage",
            check_initialised=False,
            check_alive=False,
            initial_coordinate=Coordinate(3, 2, 1),
            stage_limits=limits,
        ),
        peripheral_controllers=peripheral_ctrl,
    )

    assert stage.name == "Debug Stage"
    assert stage.get_delta_fov() == 25.0
    stage.move_to(Coordinate(4, 5, 6))
    assert stage.get_coordinates(query_hardware=False) == Coordinate(4, 5, 6)
    assert stage.get_stage_limits() == limits


def test_stage_factory_requires_tiger_peripheral_controller_for_asitiger():
    with pytest.raises(ValueError):
        StageFactory.create(StageConfig(binding=StageBindingType.ASI_TIGER, delta_fov=100.0))


def test_stage_factory_creates_asitiger_stage_with_fake_controller():
    tiger = FakeTigerController()
    peripheral_ctrl = TigerPeripheralController(tiger=tiger)
    peripheral_ctrl.initialise()

    stage = StageFactory.create(
        StageConfig(
            binding=StageBindingType.ASI_TIGER,
            delta_fov=10.0,
            name="Fake Tiger Stage",
            card_address_crisp=3,
        ),
        peripheral_controllers=peripheral_ctrl,
    )

    assert isinstance(stage, TigerStage)
    assert stage.name == "Fake Tiger Stage"
    assert stage.card_address_crisp == 3
