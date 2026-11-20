from __future__ import annotations

from io import StringIO
import logging

import pytest

from evomachine.gui.log_buffer import GuiLogBufferHandler


def test_gui_log_buffer_filters_and_bounds_structured_records() -> None:
    handler = GuiLogBufferHandler(capacity=2, level=logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    logger = logging.Logger("test.gui.buffer", level=logging.DEBUG)
    logger.addHandler(handler)

    logger.debug("hidden")
    logger.info("first")
    logger.warning("second")
    logger.error("third")

    records = handler.records_after()
    assert [record["message"] for record in records] == ["second", "third"]
    assert records[0]["level"] == "WARNING"
    assert records[0]["logger"] == "test.gui.buffer"
    assert isinstance(records[0]["timestamp"], str)
    assert handler.records_after(records[0]["sequence"]) == (records[1],)


def test_gui_log_buffer_preserves_exact_terminal_format() -> None:
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    handler = GuiLogBufferHandler()
    handler.setFormatter(formatter)
    terminal_output = StringIO()
    terminal_handler = logging.StreamHandler(terminal_output)
    terminal_handler.setFormatter(formatter)
    logger = logging.Logger("test.gui.exact", level=logging.DEBUG)
    logger.addHandler(handler)
    logger.addHandler(terminal_handler)

    logger.debug("same debug record")

    record = handler.records_after()[0]
    assert record["formatted"] == terminal_output.getvalue().rstrip("\n")
    assert record["level"] == "DEBUG"


def test_reconfiguring_application_logger_does_not_duplicate_handlers() -> None:
    """Verify repeated logger configuration retains one handler of each kind."""
    from evomachine.config import file_handler, get_logger, gui_log_handler

    logger = logging.getLogger("test.gui.reconfiguration")
    original_handlers = list(logger.handlers)
    original_level = logger.level
    original_propagate = logger.propagate
    first_stream_handler = None
    try:
        get_logger(logger.name, is_peripheral=True)
        first_stream_handler = next(
            handler for handler in logger.handlers if type(handler) is logging.StreamHandler
        )
        get_logger(logger.name, is_peripheral=True)

        assert len(logger.handlers) == 3
        assert sum(handler is file_handler for handler in logger.handlers) == 1
        assert sum(handler is gui_log_handler for handler in logger.handlers) == 1
        assert sum(type(handler) is logging.StreamHandler for handler in logger.handlers) == 1
    finally:
        if first_stream_handler is not None:
            first_stream_handler.close()
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            if type(handler) is logging.StreamHandler:
                handler.close()
        for handler in original_handlers:
            logger.addHandler(handler)
        logger.setLevel(original_level)
        logger.propagate = original_propagate


def test_gui_log_buffer_rejects_invalid_bounds_and_cursors() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        GuiLogBufferHandler(capacity=0)

    handler = GuiLogBufferHandler()
    with pytest.raises(ValueError, match="non-negative integer"):
        handler.records_after(-1)
