import pytest
from pathlib import Path
from tools.terminal_launcher import _resolve_directory

def test_resolve_directory_invalid_types():
    with pytest.raises(ValueError, match="A directory path is required."):
        _resolve_directory(None)
    with pytest.raises(ValueError, match="A directory path is required."):
        _resolve_directory(123)
    with pytest.raises(ValueError, match="A directory path is required."):
        _resolve_directory("")
    with pytest.raises(ValueError, match="A directory path is required."):
        _resolve_directory("   ")

def test_resolve_directory_invalid_paths():
    long_path = "a" * 4097
    with pytest.raises(ValueError, match="The directory path is invalid."):
        _resolve_directory(long_path)

    with pytest.raises(ValueError, match="The directory path is invalid."):
        _resolve_directory("path/with/\0/null/byte")

def test_resolve_directory_not_exists(tmp_path):
    not_exist = tmp_path / "does_not_exist"
    with pytest.raises(ValueError, match="Directory does not exist:"):
        _resolve_directory(str(not_exist))

def test_resolve_directory_not_a_directory(tmp_path):
    test_file = tmp_path / "test_file.txt"
    test_file.touch()
    with pytest.raises(ValueError, match="Path is not a directory:"):
        _resolve_directory(str(test_file))

def test_resolve_directory_absolute_path(tmp_path):
    result = _resolve_directory(str(tmp_path))
    assert result == tmp_path.resolve(strict=True)

def test_resolve_directory_relative_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sub_dir = tmp_path / "sub_dir"
    sub_dir.mkdir()

    result = _resolve_directory("sub_dir")
    assert result == sub_dir.resolve(strict=True)

def test_resolve_directory_expanduser(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))

    home_dir = tmp_path

    result = _resolve_directory("~")
    assert result == home_dir.resolve(strict=True)
