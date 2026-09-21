import threading

import numpy as np
import pytest
from delta.utils import CroppingBox

from evomachine import software_focus_bkp
from evomachine.acquisition import FrameAcquisitionManager, FrameAcquisitionSettings
from evomachine.bindings.software_focus.software_focus_algorithms import (
    BandpassFFTFocusAlgorithm,
    LaplacianVarianceFocusAlgorithm,
    SoftwareFocusAlgorithm,
    SquaredGradientAverageFocusAlgorithm,
    SteelFocusAlgorithm,
    create_software_focus_algorithm,
)
from evomachine.coordinates import Coordinate
from evomachine.frame import FrameMetaData
from evomachine.peripherals.leds import LedState
from evomachine.softwarefocus import SoftwareFocus, SoftwareFocusConfig, SoftwareFocusConfigFactory
from evomachine.types import FilterWheelType, FocusAlgorithmType, FocusCurveType, FocusStatusType, LEDType


class FakeStage:
    """Small stage fake whose current Z controls fake camera sharpness."""

    def __init__(self, coordinate: Coordinate | None = None, fov_id: int = 0):
        """
        Initialise a fake stage.

        Parameters
        ----------
        coordinate
            Initial stage coordinate.
        fov_id
            FoV ID returned by get_fov_id().

        Returns
        -------
        None
        """
        self.coordinate = coordinate or Coordinate(0, 0, 0)
        self.fov_id = fov_id
        self.moves: list[Coordinate] = []
        self.stop_count = 0

    def get_coordinates(self, query_hardware: bool = True, axes=None) -> Coordinate:
        """
        Return the current fake coordinate.

        Parameters
        ----------
        query_hardware
            Accepted for API compatibility.
        axes
            Accepted for API compatibility.

        Returns
        -------
        Coordinate
            Copy of the current coordinate.
        """
        return self.coordinate.copy()

    def get_fov_id(self) -> int:
        """
        Return the fake fov ID.

        Parameters
        ----------
        None

        Returns
        -------
        int
            Current fake fov ID.
        """
        return self.fov_id

    def move(self, target: Coordinate, block: bool = True) -> None:
        """
        Move the fake stage to a coordinate.

        Parameters
        ----------
        target
            Coordinate containing axes to update.
        block
            Accepted for API compatibility.

        Returns
        -------
        None
        """
        self.moves.append(target.copy())
        if target.x is not None:
            self.coordinate.x = target.x
        if target.y is not None:
            self.coordinate.y = target.y
        if target.z is not None:
            self.coordinate.z = target.z

    def stop(self) -> None:
        """
        Record that stage stop was requested.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.stop_count += 1


class FakeCamera:
    """Small camera fake returning images whose sharpness peaks at Z=0."""

    def __init__(self, stage: FakeStage, shape: tuple[int, int] = (6, 6)):
        """
        Initialise a fake camera.

        Parameters
        ----------
        stage
            FakeStage used to read the current Z coordinate.
        shape
            Shape of generated images.

        Returns
        -------
        None
        """
        self.stage = stage
        self.shape = shape
        self.exposures: list[float | int] = []
        self.frames_captured = 0
        self.stop_count = 0

    def set_exposure(self, exposure_time: float | int) -> None:
        """
        Record a fake exposure setting.

        Parameters
        ----------
        exposure_time
            Exposure time in milliseconds.

        Returns
        -------
        None
        """
        self.exposures.append(exposure_time)

    def get_frame(self, normalise: bool = False) -> np.ndarray:
        """
        Return a deterministic image based on current fake stage Z.

        Parameters
        ----------
        normalise
            Accepted for API compatibility.

        Returns
        -------
        np.ndarray
            Generated image.
        """
        self.frames_captured += 1
        z = 0 if self.stage.coordinate.z is None else int(self.stage.coordinate.z)
        amplitude = max(0, 10 - abs(z))
        frame = np.zeros(self.shape, dtype=np.float64)
        frame[2:4, 2:4] = amplitude
        return frame

    def stop(self) -> None:
        """
        Record that camera stop was requested.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.stop_count += 1


class FakeLedManager:
    """Small LED manager fake that records set and disable calls."""

    def __init__(self):
        """Initialise fake LED command recording."""
        self.commands: list[tuple[LEDType | None, float | int]] = []
        self.disable_count = 0
        self.stop_count = 0
        self.states = {
            LEDType.LED_450_NM: LedState(led_type=LEDType.LED_450_NM),
            LEDType.LED_565_NM: LedState(led_type=LEDType.LED_565_NM),
        }

    def get_available_leds(self) -> list[LEDType]:
        """
        Return LEDs available through the fake manager.

        Parameters
        ----------
        None

        Returns
        -------
        list[LEDType]
            Available fake LEDs.
        """
        return list(self.states)

    def get_led_state(self, led_type: LEDType) -> LedState:
        """
        Return the cached fake LED state.

        Parameters
        ----------
        led_type
            LED to inspect.

        Returns
        -------
        LedState
            Copied fake state.
        """
        state = self.states[led_type]
        return LedState(led_type=state.led_type, brightness=state.brightness, is_on=state.is_on)

    def set_led(self, led_type: LEDType, brightness: float | int, duration: float | None = None) -> None:
        """
        Record a fake LED set command.

        Parameters
        ----------
        led_type
            LEDType to record.
        brightness
            Brightness to record.
        duration
            Accepted for API compatibility.

        Returns
        -------
        None
        """
        self.commands.append((led_type, brightness))
        self.states[led_type] = LedState(led_type=led_type, brightness=brightness, is_on=brightness > 0)

    def disable_led(self, led_type: LEDType | None = None) -> None:
        """
        Record a fake LED disable command.

        Parameters
        ----------
        led_type
            Optional LEDType to disable.

        Returns
        -------
        None
        """
        self.disable_count += 1
        self.commands.append((led_type, 0))
        led_types = list(self.states) if led_type is None else [led_type]
        for selected_led in led_types:
            self.states[selected_led] = LedState(led_type=selected_led)

    def stop(self) -> None:
        """
        Record that LED manager stop was requested.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.stop_count += 1


class FakeDmd:
    """Small DMD fake that records display calls."""

    def __init__(self):
        """Initialise fake DMD state."""
        self.full_count = 0
        self.none_count = 0
        self.stop_count = 0

    def display_full(self) -> None:
        """
        Record a fake full-display command.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.full_count += 1

    def display_none(self) -> None:
        """
        Record a fake blank-display command.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.none_count += 1

    def stop(self) -> None:
        """
        Record that DMD stop was requested.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.stop_count += 1


class FakeFilterWheel:
    """Small filter wheel fake that records filter fovs."""

    def __init__(self):
        """Initialise fake filter wheel state."""
        self.filters: list[FilterWheelType] = []
        self.stop_count = 0

    def set_filter_wheel(self, filter_type: FilterWheelType) -> None:
        """
        Record a fake filter wheel setting.

        Parameters
        ----------
        filter_type
            FilterWheelType to record.

        Returns
        -------
        None
        """
        self.filters.append(filter_type)

    def stop(self) -> None:
        """
        Record that filter wheel stop was requested.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.stop_count += 1


def _image() -> np.ndarray:
    """
    Return a deterministic image for software focus score tests.

    Parameters
    ----------
    None

    Returns
    -------
    np.ndarray
        Small deterministic image with non-zero gradients.
    """
    return np.arange(36, dtype=np.float64).reshape(6, 6)


def _frame(
        frame_id: int = -1,
        led_type: LEDType = LEDType.LED_450_NM,
        brightness: float | int = 29,
        exposure: float | int = 200,
        filter_wheel: FilterWheelType | None = None,
) -> FrameMetaData:
    """
    Return focus frame metadata for tests.

    Parameters
    ----------
    frame_id
        Frame ID for the metadata.
    led_type
        LED channel to use.
    brightness
        LED brightness to use.
    exposure
        Camera exposure to use.
    filter_wheel
        Optional filter wheel setting.

    Returns
    -------
    FrameMetaData
        Valid frame metadata.
    """
    return FrameMetaData(
        frame_id=frame_id,
        leds={led_type: brightness},
        filter_wheel=filter_wheel,
        exposure=exposure,
    )


def _config(**kwargs) -> SoftwareFocusConfig:
    """
    Return a valid SoftwareFocusConfig with optional field overrides.

    Parameters
    ----------
    **kwargs
        SoftwareFocusConfig field values to override.

    Returns
    -------
    SoftwareFocusConfig
        Valid software focus configuration.
    """
    values = {
        "focus_frames": [_frame()],
        "acquisition_settings": None,
        "rel_range": 3,
        "step_size": 1,
        "algorithm": FocusAlgorithmType.LAPLACIAN_VAR,
        "algorithm_kwargs": {},
        "cropping_box": None,
    }
    values.update(kwargs)
    return SoftwareFocusConfig(**values)


def _software_focus(
        config: SoftwareFocusConfig | None = None,
        filter_wheel: FakeFilterWheel | None = None,
) -> tuple[SoftwareFocus, FakeStage, FakeCamera, FakeLedManager, FakeDmd]:
    """
    Return a SoftwareFocus instance with fake peripherals.

    Parameters
    ----------
    config
        Optional software focus config.
    filter_wheel
        Optional fake filter wheel.

    Returns
    -------
    tuple[SoftwareFocus, FakeStage, FakeCamera, FakeLedManager, FakeDmd]
        SoftwareFocus and the fakes it uses.
    """
    stage = FakeStage()
    camera = FakeCamera(stage=stage)
    leds = FakeLedManager()
    dmd = FakeDmd()
    acquisition_manager = FrameAcquisitionManager(
        camera=camera,
        led_manager=leds,
        filter_wheel=filter_wheel,
        dmd=dmd,
        stage=stage,
    )
    focus = SoftwareFocus(
        acquisition_manager=acquisition_manager,
        config=_config() if config is None else config,
    )
    return focus, stage, camera, leds, dmd


def test_backup_module_exposes_legacy_focus_helpers() -> None:
    """
    Check that the backup module preserves the legacy helper functions.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    img = _image()

    assert software_focus_bkp.get_focus_score_laplacian_var(img) >= 0
    assert software_focus_bkp.get_focus_score_squared_gradient(img) >= 0
    assert software_focus_bkp.get_focus_score_steel(img, rowshift=1, colshift=2) >= 0


def test_algorithm_classes_match_backup_scores() -> None:
    """
    Check that algorithm classes match the backed-up implementations.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    img = _image()

    assert LaplacianVarianceFocusAlgorithm().score_image(img) == software_focus_bkp.get_focus_score_laplacian_var(img)
    assert SquaredGradientAverageFocusAlgorithm(threshold=0).score_image(img) == software_focus_bkp.get_focus_score_squared_gradient(img, threshold=0)
    assert SteelFocusAlgorithm(rowshift=1, colshift=2).score_image(img) == software_focus_bkp.get_focus_score_steel(img, rowshift=1, colshift=2)


def test_algorithm_factory_selects_algorithms() -> None:
    """
    Check algorithm factory type selection and parameter use.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    assert isinstance(create_software_focus_algorithm(FocusAlgorithmType.LAPLACIAN_VAR), SoftwareFocusAlgorithm)
    assert isinstance(create_software_focus_algorithm(FocusAlgorithmType.SQUARED_GRAD_AVG), SquaredGradientAverageFocusAlgorithm)
    steel = create_software_focus_algorithm(FocusAlgorithmType.STEEL, rowshift=1, colshift=2)

    assert isinstance(steel, SteelFocusAlgorithm)
    assert steel.rowshift == 1
    assert steel.colshift == 2

    fft_algo = create_software_focus_algorithm(
        FocusAlgorithmType.BANDPASS_FFT, cell_radius=15, saturation=4095
    )
    assert isinstance(fft_algo, BandpassFFTFocusAlgorithm)
    assert fft_algo.cell_radius == 15
    assert fft_algo.saturation == 4095

    with pytest.raises(TypeError):
        create_software_focus_algorithm("STEEL")


def test_bandpass_fft_algorithm_scores_sharp_above_blurred() -> None:
    """
    Check that the bandpass-FFT algorithm ranks a sharp image above a blurred copy.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(0)
    sharp = np.zeros((128, 128), dtype=np.float64)
    ys = rng.integers(5, 123, size=40)
    xs = rng.integers(5, 123, size=40)
    sharp[ys, xs] = 255.0
    sharp = gaussian_filter(sharp, sigma=1.5)  # give the "cells" a small finite size
    blurred = gaussian_filter(sharp, sigma=4.0)  # simulate defocus

    algo = BandpassFFTFocusAlgorithm(cell_radius=4, downsample=1)
    assert isinstance(algo.score_image(sharp), float)
    assert algo.score_image(sharp) > algo.score_image(blurred)

    with pytest.raises(TypeError):
        BandpassFFTFocusAlgorithm(cell_radius=-1)


def test_software_focus_config_validation_and_factory() -> None:
    """
    Check software focus config validation and factory defaults.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    config = SoftwareFocusConfigFactory.default_config()

    assert isinstance(config, SoftwareFocusConfig)
    assert config.focus_frames[0].leds == {LEDType.LED_450_NM: 29}
    assert config.copy().focus_frames == config.focus_frames
    with pytest.raises(TypeError):
        _config(focus_frames=[])
    with pytest.raises(TypeError):
        _config(acquisition_settings="bad")
    with pytest.raises(TypeError):
        _config(algorithm_kwargs={1: "bad"})
    with pytest.raises(ValueError):
        _config(cropping_box=[])


def test_software_focus_update_config_and_fovs() -> None:
    """
    Check default and fov-specific config management.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    focus, _, _, _, _ = _software_focus()
    default_replacement = _config(focus_frames=[_frame(exposure=80)])
    fov_replacement = _config(focus_frames=[_frame(exposure=90)])

    focus.initialise_fovs([1, 2], fov_configs={2: fov_replacement})
    assert not hasattr(focus.get_fov_result(1), "fov_id")
    assert not hasattr(focus, "get_fov_state")
    assert focus._fov_config.get(1, focus.default_config) is focus.default_config
    assert focus._fov_config.get(2, focus.default_config) is fov_replacement

    focus.update_config(default_replacement)
    focus.update_config(fov_replacement, fov_id=1)

    assert focus.default_config is default_replacement
    assert focus._fov_config.get(1, focus.default_config) is fov_replacement
    with pytest.raises(KeyError):
        focus.get_fov_result(3)
    with pytest.raises(TypeError):
        focus.update_config("bad")


def test_software_focus_score_image_crops_and_averages() -> None:
    """
    Check image scoring with no crop, one crop, and multiple crops.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    focus, _, _, _, _ = _software_focus()
    img = _image()
    config = _config(algorithm=FocusAlgorithmType.STEEL, algorithm_kwargs={"rowshift": 1, "colshift": 2})
    scorer = create_software_focus_algorithm(FocusAlgorithmType.STEEL, rowshift=1, colshift=2)
    first_box = CroppingBox(xtl=0, xbr=3, ytl=0, ybr=3)
    second_box = CroppingBox(xtl=2, xbr=6, ytl=2, ybr=6)

    assert focus.score_image(img=img, config=config) == scorer.score_image(img)
    assert focus.score_image(img=img, config=config, cropping_box=first_box) == scorer.score_image(first_box.crop(img))
    assert focus.score_image(img=img, config=config, cropping_box=[first_box, second_box]) == pytest.approx(
        np.mean([scorer.score_image(first_box.crop(img)), scorer.score_image(second_box.crop(img))])
    )


def test_software_focus_run_scans_and_moves_to_best_z() -> None:
    """
    Check SoftwareFocus scans Z fovs and moves to the best non-boundary Z.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    focus, stage, camera, leds, dmd = _software_focus()

    result = focus.run(fov_id=5)
    stored_result = focus.get_fov_result(5)

    assert result.focus_status == FocusStatusType.IN_FOCUS
    assert result.curve_status == FocusCurveType.HAS_GLOBAL_MAXIMUM
    assert result.best_coordinate.z == 0
    assert stage.coordinate.z == 0
    assert np.array_equal(result.z_coordinates, np.array([-3, -2, -1, 0, 1, 2]))
    assert stored_result is result
    assert result.focus_stack.shape == (6, 6, 6)
    assert result.previous_image.shape == (6, 6)
    assert camera.exposures[0] == 200
    assert dmd.full_count == 7
    assert leds.disable_count >= 1
    assert (LEDType.LED_450_NM, 29) in leds.commands


def test_software_focus_run_uses_fov_specific_config() -> None:
    """
    Check run uses fov-specific config when present.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    focus, _, camera, leds, _ = _software_focus()
    fov_config = _config(
        focus_frames=[_frame(led_type=LEDType.LED_565_NM, brightness=20, exposure=60)],
        acquisition_settings=FrameAcquisitionSettings(illuminate_dmd=False, restore_leds_after=False),
        rel_range=2,
        step_size=1,
    )
    focus.initialise_fovs([4], fov_configs={4: fov_config})

    result = focus.run(fov_id=4)

    assert np.array_equal(result.z_coordinates, np.array([-2, -1, 0, 1]))
    assert 60 in camera.exposures
    assert (LEDType.LED_565_NM, 20) in leds.commands
    assert leds.disable_count == 0


def test_software_focus_run_averages_multiple_frame_metadata_scores() -> None:
    """
    Check SoftwareFocus averages scores from multiple FrameMetaData captures.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    frame_a = _frame(frame_id=1, led_type=LEDType.LED_450_NM, brightness=10, exposure=50, filter_wheel=FilterWheelType.FILTER_465nm)
    frame_b = _frame(frame_id=2, led_type=LEDType.LED_565_NM, brightness=20, exposure=60, filter_wheel=FilterWheelType.FILTER_592nm)
    filter_wheel = FakeFilterWheel()
    focus, _, camera, leds, _ = _software_focus(
        config=_config(focus_frames=[frame_a, frame_b]),
        filter_wheel=filter_wheel,
    )

    result = focus.run(fov_id=0)

    assert result.focus_status == FocusStatusType.IN_FOCUS
    assert camera.frames_captured == 14
    assert filter_wheel.filters[:2] == [FilterWheelType.FILTER_465nm, FilterWheelType.FILTER_592nm]
    assert (LEDType.LED_450_NM, 10) in leds.commands
    assert (LEDType.LED_565_NM, 20) in leds.commands
    assert 50 in camera.exposures
    assert 60 in camera.exposures


def test_software_focus_run_rejects_missing_filter_wheel_for_frame_metadata() -> None:
    """
    Check SoftwareFocus raises when frame metadata needs a missing filter wheel.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    frame = _frame(frame_id=1, led_type=LEDType.LED_450_NM, brightness=10, exposure=50, filter_wheel=FilterWheelType.FILTER_465nm)
    focus, _, _, leds, _ = _software_focus(config=_config(focus_frames=[frame]))

    with pytest.raises(RuntimeError, match="filter wheel"):
        focus.run(fov_id=0)
    assert leds.disable_count >= 1


def test_software_focus_run_respects_stop_event() -> None:
    """
    Check SoftwareFocus stops before scanning when the stop event is set.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    focus, stage, _, leds, dmd = _software_focus()
    stop_event = threading.Event()
    stop_event.set()

    result = focus.run(fov_id=0, stop_event=stop_event)

    assert result.focus_status != FocusStatusType.IN_FOCUS
    assert result.focus_scores.size == 0
    assert len(stage.moves) == 0
    assert dmd.full_count == 0
    assert leds.disable_count == 0
