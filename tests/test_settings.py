from pathlib import Path

import pytest

from evomachine.settings import EvoSettings, load_settings


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_defaults_apply_when_no_file_exists(tmp_path):
    settings = load_settings(
        machine_file=tmp_path / "missing-machine.toml",
        user_file=tmp_path / "missing-user.toml",
        default_data_dir=tmp_path / "default",
    )

    assert settings == EvoSettings(data_dir=tmp_path / "default", source_files=())


def test_derived_directories_hang_off_the_data_dir(tmp_path):
    settings = EvoSettings(data_dir=tmp_path)

    assert settings.images_dir == tmp_path / "images"
    assert settings.log_dir == tmp_path / "logs"
    assert settings.autostrat_strategies_dir == tmp_path / "strategies" / "autostrat"


def test_machine_file_sets_the_data_dir(tmp_path):
    machine = _write(tmp_path / "machine.toml", '[paths]\ndata_dir = "/scratch/data"\n')

    settings = load_settings(machine_file=machine, user_file=tmp_path / "missing.toml")

    assert settings.data_dir == Path("/scratch/data")
    assert settings.source_files == (machine,)


def test_user_file_overrides_machine_file(tmp_path):
    machine = _write(tmp_path / "machine.toml", '[paths]\ndata_dir = "/scratch/data"\n')
    user = _write(tmp_path / "user.toml", '[paths]\ndata_dir = "/mnt/mine"\n')

    settings = load_settings(machine_file=machine, user_file=user)

    assert settings.data_dir == Path("/mnt/mine")
    assert settings.source_files == (machine, user)


def test_empty_user_file_keeps_machine_value(tmp_path):
    machine = _write(tmp_path / "machine.toml", '[paths]\ndata_dir = "/scratch/data"\n')
    user = _write(tmp_path / "user.toml", "# nothing overridden yet\n")

    settings = load_settings(machine_file=machine, user_file=user)

    assert settings.data_dir == Path("/scratch/data")
    assert settings.source_files == (machine, user)


def test_paths_expand_user_and_environment_variables(tmp_path, monkeypatch):
    monkeypatch.setenv("EVO_TEST_USER", "alice")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    machine = _write(tmp_path / "machine.toml", '[paths]\ndata_dir = "~/data/$EVO_TEST_USER"\n')

    settings = load_settings(machine_file=machine, user_file=tmp_path / "missing.toml")

    assert settings.data_dir == tmp_path / "home" / "data" / "alice"


@pytest.mark.parametrize(
    "text, message",
    [
        ("[paths]\ndata_dir = 42\n", "must be a non-empty string"),
        ('[paths]\ndata_dir = ""\n', "must be a non-empty string"),
        ('[paths]\ndatadir = "/x"\n', "Unknown key"),
        ('[storage]\ndata_dir = "/x"\n', "Unknown section"),
        ('paths = "/x"\n', "must be a table"),
        ("[paths\n", "Invalid settings file"),
    ],
)
def test_malformed_files_are_rejected_with_the_file_named(tmp_path, text, message):
    user = _write(tmp_path / "user.toml", text)

    with pytest.raises(ValueError, match=message) as error:
        load_settings(machine_file=tmp_path / "missing.toml", user_file=user)

    assert str(user) in str(error.value)
