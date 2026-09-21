from __future__ import annotations

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


def test_gui_log_buffer_rejects_invalid_bounds_and_cursors() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        GuiLogBufferHandler(capacity=0)

    handler = GuiLogBufferHandler()
    with pytest.raises(ValueError, match="non-negative integer"):
        handler.records_after(-1)
