from __future__ import annotations

import inspect

from evomachine.gui import launchers


def test_launchers_do_not_import_legacy_guidir() -> None:
    assert "guidir" not in inspect.getsource(launchers)


def test_runtime_factory_spec_requires_module_and_function() -> None:
    try:
        launchers._load_runtime_factory("missing_separator")
    except ValueError as error:
        assert "module:function" in str(error)
    else:
        raise AssertionError("expected ValueError")

