"""Fast, pure tests for the per-model calibration store - no model load
needed."""

import json

import pytest

from supervisor import calibration_store


def test_load_temperature_returns_1_0_when_file_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(calibration_store, "CALIBRATION_PATH", tmp_path / "calibration.json")
    assert calibration_store.load_temperature("some-model") == 1.0


def test_load_temperature_returns_1_0_for_unknown_model_id(tmp_path, monkeypatch):
    path = tmp_path / "calibration.json"
    monkeypatch.setattr(calibration_store, "CALIBRATION_PATH", path)
    calibration_store.save_entry("model-a", {"temperature": 3.5})

    assert calibration_store.load_temperature("model-b") == 1.0


def test_save_then_load_round_trips_the_right_model(tmp_path, monkeypatch):
    monkeypatch.setattr(calibration_store, "CALIBRATION_PATH", tmp_path / "calibration.json")
    calibration_store.save_entry("model-a", {"temperature": 3.5})

    assert calibration_store.load_temperature("model-a") == 3.5


def test_saving_one_model_does_not_clobber_another(tmp_path, monkeypatch):
    monkeypatch.setattr(calibration_store, "CALIBRATION_PATH", tmp_path / "calibration.json")
    calibration_store.save_entry("model-a", {"temperature": 3.5})
    calibration_store.save_entry("model-b", {"temperature": 8.0})

    assert calibration_store.load_temperature("model-a") == 3.5
    assert calibration_store.load_temperature("model-b") == 8.0


def test_re_saving_the_same_model_updates_it_in_place(tmp_path, monkeypatch):
    monkeypatch.setattr(calibration_store, "CALIBRATION_PATH", tmp_path / "calibration.json")
    calibration_store.save_entry("model-a", {"temperature": 3.5})
    calibration_store.save_entry("model-a", {"temperature": 4.0})

    assert calibration_store.load_temperature("model-a") == 4.0


def test_save_entry_creates_parent_dir_if_missing(tmp_path, monkeypatch):
    path = tmp_path / "nested" / "calibration.json"
    monkeypatch.setattr(calibration_store, "CALIBRATION_PATH", path)

    calibration_store.save_entry("model-a", {"temperature": 3.5})

    assert path.exists()


@pytest.mark.parametrize(
    "contents",
    [
        "{not json",
        "[]",
        json.dumps({"model-a": None}),
        json.dumps({"model-a": {"temperature": "hot"}}),
        json.dumps({"model-a": {"temperature": 0}}),
        json.dumps({"model-a": {"temperature": -2.0}}),
        '{"model-a": {"temperature": NaN}}',
        '{"model-a": {"temperature": Infinity}}',
    ],
    ids=["corrupt", "not-a-dict", "null-entry", "non-numeric", "zero", "negative", "nan", "inf"],
)
def test_load_temperature_falls_back_to_1_0_on_unusable_store(tmp_path, monkeypatch, contents):
    path = tmp_path / "calibration.json"
    path.write_text(contents)
    monkeypatch.setattr(calibration_store, "CALIBRATION_PATH", path)

    assert calibration_store.load_temperature("model-a") == 1.0


def test_save_entry_refuses_to_overwrite_a_corrupt_store(tmp_path, monkeypatch):
    # treating it as empty would silently wipe every other model's calibration
    path = tmp_path / "calibration.json"
    path.write_text("{not json")
    monkeypatch.setattr(calibration_store, "CALIBRATION_PATH", path)

    with pytest.raises(ValueError):
        calibration_store.save_entry("model-a", {"temperature": 3.5})
    assert path.read_text() == "{not json"


def test_save_entry_leaves_no_temp_file_behind(tmp_path, monkeypatch):
    monkeypatch.setattr(calibration_store, "CALIBRATION_PATH", tmp_path / "calibration.json")
    calibration_store.save_entry("model-a", {"temperature": 3.5})

    assert [p.name for p in tmp_path.iterdir()] == ["calibration.json"]
