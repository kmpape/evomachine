from pathlib import Path
import logging

import evomachine.config as evomachine_config
from evomachine.bindings.em_dmd_window.peripheralcontroller import EM_DMD_PROGRAM_PATH
from tests.binding_test_config import BindingTestConfig


EVOMACHINE_REPO = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = EVOMACHINE_REPO.parent
SYNC_BOARD_REPO = WORKSPACE_ROOT / "sync_board"


def test_evomachine_config_uses_repository_log_folder():
    log_path = Path(evomachine_config.file_handler.baseFilename)

    assert log_path.parent == EVOMACHINE_REPO / "logs"
    assert log_path.parent.exists()
    assert "/home/hslab/" not in str(log_path)


def test_syncboard_serialconnection_uses_repository_log_folder():
    """
    Check SyncBoard leaves file logging to evomachine.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    from syncboard import serialconnection

    assert not hasattr(serialconnection, "file_handler")
    assert not hasattr(serialconnection, "LOG_DIR")
    assert logging.getLogger("syncboard.serialconnection").handlers == []


def test_em_dmd_window_path_uses_sibling_repository_layout():
    assert EM_DMD_PROGRAM_PATH == WORKSPACE_ROOT / "em_dmd_window/Release/evomachine_dmd_window"


def test_default_binding_test_config_uses_fake_bindings():
    config = BindingTestConfig.load()

    assert not config.use_real_bindings
    assert "asitiger_fake" in config.stage_bindings
