from __future__ import annotations

import logging
from io import StringIO

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


def test_gui_log_buffer_rejects_invalid_bounds_and_cursors() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        GuiLogBufferHandler(capacity=0)

    handler = GuiLogBufferHandler()
    with pytest.raises(ValueError, match="non-negative integer"):
        handler.records_after(-1)


def test_application_gui_and_terminal_receive_identical_debug_and_exception_records():
    from evomachine.config import get_logger, gui_log_handler

    logger = get_logger("test.shared.peripheral", is_peripheral=True)
    # Reconfiguration must not duplicate any of the three handlers.
    logger = get_logger("test.shared.peripheral", is_peripheral=True)
    assert len(logger.handlers) == 3
    stream = next(h for h in logger.handlers if type(h) is logging.StreamHandler)
    output = StringIO()
    stream.setStream(output)
    cursor = gui_log_handler.latest_sequence
    try:
        logger.debug("device moved")
        try:
            raise ValueError("missing model")
        except ValueError:
            logger.exception("image processing failed")
        records = gui_log_handler.records_after(cursor)
        assert [r["level"] for r in records] == ["DEBUG", "ERROR"]
        assert "Traceback" in records[-1]["formatted"]
        assert "ValueError: missing model" in records[-1]["formatted"]
        assert output.getvalue() == "".join(r["formatted"] + "\n" for r in records)
    finally:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
