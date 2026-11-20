"""Acquire a z-stack at every FoV and save each stack as ONE multi-page TIFF (issue #96).

Each stack is written by ``evomachine.zstack_files.ZStackFileWriter``:

* one file per stack, planes sorted by Z, all per-plane FrameMetaData in the file;
* a persistent, increasing stack ID at the start of the filename (unique per folder);
* no stage coordinates in the filename (they are in the metadata), e.g.
  ``000123_450_FOV03_Z-10,1,10_261108_155754.tiff``.

How it works: the strategy requests the existing ``Z_STACK`` command with
``save=False`` (so the per-plane FileManager files are NOT written) and writes the
returned stack itself in ``_callback``. Each batch contains at most one Z_STACK
(``[MOVE, Z_STACK]`` for a FoV's first channel, ``[Z_STACK]`` for further channels),
so a stack is always saved before a Stop can end the run.

Notes for use on the microscope:

* Output goes to ``output_directory``. Strategies cannot see the GUI's Output Folder,
  so the folder is logged when the strategy is set. Override it per machine with the
  ``EVOMACHINE_ZSTACK_DIR`` environment variable.
* The Automaton keeps the last ``frame_history_limit`` (default 2) frames per FoV in
  memory, and a z-stack counts as one frame: a 21-plane full-sensor stack is ~430 MB.
* Z_STACK moves Z directly. Run with hardware autofocus (CRISP) focus management off,
  or make sure it is not locked while stacks are taken.
* Load the files with ``evomachine.zstack_files.read_zstack`` or ``tifffile.imread``,
  not skimage (which returns 3- or 4-plane stacks as (H, W, N)).

Edit the class attributes below to change the experiment.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import time
from typing import TYPE_CHECKING

from evomachine.commands import AutomatonCommand
from evomachine.config import DATA_DIR
from evomachine.frame import FrameMetaDataFactory
from evomachine.strategy import AbstractStrategy
from evomachine.types import AutomatonCommandType, LEDType
from evomachine.zstack_files import (
    ZStackChannel,
    ZStackFileConfig,
    ZStackFileRecord,
    ZStackFileWriter,
    expected_plane_count,
)

if TYPE_CHECKING:
    from evomachine.image_processing_config import ImageProcessorConfig

logger = logging.getLogger(__name__)


def _default_output_directory() -> Path:
    override = os.getenv("EVOMACHINE_ZSTACK_DIR")
    return Path(override).expanduser() if override else DATA_DIR / "zstacks"


class SingleFileZStackStrategy(AbstractStrategy):
    """
    Z-stack every FoV, one multi-page TIFF per stack and channel, repeated for ``n_rounds``.

    Parameters
    ----------
    cfg
        Image processor configuration used by command validation.
    """

    # --- experiment settings (edit these) -------------------------------------------
    channels: tuple[ZStackChannel, ...] = (
        ZStackChannel(led=LEDType.LED_450_NM, brightness=10, exposure_ms=100),
    )
    "One z-stack (and one file) per channel at each FoV."
    relative_range_um: float = 10.0
    "Half-range around the current focus, in um: planes cover -range..+range."
    step_size_um: float = 1.0
    "Distance between planes, in um."
    n_rounds: int = 1
    "How many times to visit every FoV."
    round_period_s: float = 0.0
    "Minimum time from the start of one round to the start of the next."
    z_in_filename: str = "relative"
    "'relative' (to the stack centre) or 'absolute' stage Z in the filename."
    compression: str | None = None
    "Optional TIFF compression, e.g. 'zlib' (smaller files, slower writes)."
    output_directory: Path | None = None
    "Where stack files go. None: $EVOMACHINE_ZSTACK_DIR, else <configured images dir>/zstacks."
    # -----------------------------------------------------------------------------------

    def __init__(self, cfg: ImageProcessorConfig):
        super().__init__(cfg=cfg)
        self._fov_order: list[int] = []
        self._fov_index: int = 0
        self._channel_index: int = 0
        self._round: int = 0
        self._round_started: float | None = None
        self._writer: ZStackFileWriter | None = None
        self.saved_records: list[ZStackFileRecord] = []
        "Records of every stack file written in this run."

    def register_automaton_commands(self) -> set[AutomatonCommandType]:
        """Return every command type this strategy may emit."""
        return {
            AutomatonCommandType.MOVE,
            AutomatonCommandType.Z_STACK,
            AutomatonCommandType.WAIT,
            AutomatonCommandType.TERMINATE_STRATEGY,
            AutomatonCommandType.ABORT_STRATEGY,
        }

    # ----------------------------------------------------------------- lifecycle
    def _initialise(self) -> list[AutomatonCommand]:
        """Validate settings, check the output folder and return the first commands."""
        if not self.channels:
            raise ValueError(f"{self.name()}: at least one channel is required.")
        if self.n_rounds < 1:
            raise ValueError(f"{self.name()}: n_rounds must be >= 1.")
        if not self.fovs:
            raise ValueError(f"{self.name()}: no FoVs are registered.")
        self._fov_order = sorted(self.fovs)
        self._fov_index = 0
        self._channel_index = 0
        self._round = 0
        self._round_started = None  # set from the first acquired plane
        self.saved_records = []
        directory = Path(self.output_directory or _default_output_directory()).expanduser()
        self._check_output_directory(directory)
        self._writer = ZStackFileWriter(ZStackFileConfig(
            directory=directory,
            z_in_filename=self.z_in_filename,
            compression=self.compression,
        ))
        planes = expected_plane_count(self.relative_range_um, self.step_size_um)
        logger.info(
            "%s: %d FoV(s) x %d channel(s) x %d round(s), %d planes per stack; saving to %s",
            self.name(), len(self._fov_order), len(self.channels), self.n_rounds, planes, directory,
        )
        return self._next_stack_commands()

    def _callback(
            self,
            fov_id: int,
            data: list[AutomatonCommand],
            errors: list[Exception],
    ) -> list[AutomatonCommand]:
        """Save the stack just acquired, then request the next stack, round, or termination."""
        # Save whatever completed before raising, so no acquired stack is lost.
        save_failed = self._save_stacks(data)
        if errors:
            # Same policy as the other bundled strategies: surface hardware failures.
            raise errors[0]
        if save_failed:
            logger.error("%s: aborting because a z-stack could not be saved.", self.name())
            return [self.command_factory.command_abort_strategy()]

        self._channel_index += 1
        if self._channel_index < len(self.channels):
            return self._next_stack_commands()
        self._channel_index = 0
        self._fov_index += 1
        if self._fov_index < len(self._fov_order):
            return self._next_stack_commands()

        self._fov_index = 0
        self._round += 1
        if self._round >= self.n_rounds:
            logger.info("%s: finished %d round(s), %d stack file(s) written.",
                        self.name(), self._round, len(self.saved_records))
            return [self.command_factory.command_terminate_strategy()]
        commands: list[AutomatonCommand] = []
        wait_s = 0.0
        if self._round_started is not None and self.round_period_s > 0:
            wait_s = max(0.0, self.round_period_s - (time.monotonic() - self._round_started))
            if wait_s > 0:
                commands.append(self.command_factory.command_wait(duration=wait_s))
        # The next round starts once the wait (executed by the Automaton) has elapsed.
        self._round_started = time.monotonic() + wait_s
        return commands + self._next_stack_commands()

    def finalise(self) -> list[AutomatonCommand]:
        """No final commands."""
        return []

    # ------------------------------------------------------------------- helpers
    @staticmethod
    def _check_output_directory(directory: Path) -> None:
        """Create the folder and probe that it is writable before any light is switched on."""
        directory.mkdir(parents=True, exist_ok=True)
        probe = directory / ".zstack_write_test"
        probe.write_text("ok")
        probe.unlink()

    def _next_stack_commands(self) -> list[AutomatonCommand]:
        """[MOVE, Z_STACK] for a FoV's first channel, [Z_STACK] for further channels."""
        fov_id = self._fov_order[self._fov_index]
        channel = self.channels[self._channel_index]
        commands = []
        if self._channel_index == 0:
            commands.append(self.command_factory.command_move(fov_id=fov_id))
        metadata = FrameMetaDataFactory.default(
            leds={channel.led: channel.brightness},
            filter_wheel=channel.filter_wheel,
            exposure=channel.exposure_ms,
            fov_id=fov_id,
        )
        commands.append(self.command_factory.command_z_stack(
            frame_metadata=metadata,
            relative_range=self.relative_range_um,
            step_size=self.step_size_um,
            save=False,  # per-plane files are not written; _save_stacks writes one file
        ))
        return commands

    def _save_stacks(self, data: list[AutomatonCommand]) -> bool:
        """Write every completed Z_STACK in ``data``. Returns True if any write failed."""
        stacks = [c for c in data if c.command_type is AutomatonCommandType.Z_STACK]
        if not stacks:
            return False
        if self._writer is None:
            raise RuntimeError(f"{self.name()}: strategy was not initialised.")
        failed = False
        expected = expected_plane_count(self.relative_range_um, self.step_size_um)
        for command in stacks:
            result = command.command_data
            if not isinstance(result, dict) or result.get("skipped") or "img" not in result:
                reason = result.get("skip_reason") if isinstance(result, dict) else None
                logger.warning("%s: no z-stack data for FoV %s (%s); nothing saved.",
                               self.name(), command.fov_id, reason or "skipped")
                continue
            stack = result["img"][0]
            planes = result["frame_metadata"]
            if self._round_started is None:
                first = planes[0].acquisition_monotonic if planes else None
                self._round_started = first if first is not None else time.monotonic()
            # Automaton acquires the centre plane first (offsets 0, +s.., -s..).
            centre_z = planes[0].coordinate.z if planes and planes[0].coordinate is not None else None
            try:
                record = self._writer.write(
                    stack,
                    planes,
                    centre_z=centre_z,
                    z_step=float(self.step_size_um),
                    expected_planes=expected,
                    extra={
                        "strategy": self.name(),
                        "round": self._round,
                        "relative_range": float(self.relative_range_um),
                        "command_id": command.command_id,
                    },
                )
            except Exception:  # noqa: BLE001 - never let a write error kill the backend
                logger.exception("%s: failed to save z-stack for FoV %s.", self.name(), command.fov_id)
                failed = True
                continue
            self.saved_records.append(record)
            if not record.complete:
                logger.warning("%s: stack %d is incomplete (%d planes).",
                               self.name(), record.stack_id, record.n_planes)
            logger.info("%s: saved %s", self.name(), record.path.name)
        return failed
