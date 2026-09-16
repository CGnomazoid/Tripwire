"""Fast, pure tests for the per-model calibration store - no model load
needed."""

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
