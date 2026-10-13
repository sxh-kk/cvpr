"""Archive forecast world decoding needs a real last-observed reference."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from MotionExpert import infer
from MotionExpert.dataset import ee4d_motion_dataset
from MotionExpert.tests.fixtures import tiny_expert
from MotionExpert.tests.test_dense_dataset import synthetic_arrays, write_archive
from MotionExpert.tests.test_dense_forecast_data import forecast_dataset


def archive_sample(tmp_path, *, valid_current):
    root = tmp_path / "archive"
    root.mkdir()
    arrays = synthetic_arrays(61)
    arrays["observation.valid.head_reference_position"][30] = valid_current
    manifest, _ = write_archive(root, arrays)
    dataset = forecast_dataset(root, manifest)
    # A missing current pose still permits image-only prediction and masked
    # future supervision; it cannot supply a world-space decoding anchor.
    assert len(dataset) == 1
    sample = dataset[0]
    assert bool(sample["misc"]["trajectory_valid"][5, :9].all()) == valid_current
    return sample


def run_cli(tmp_path, monkeypatch, sample, *, decode):
    torch.set_num_threads(1)
    expert = tiny_expert()
    constructors = []
    dataset = SimpleNamespace(normalizer=expert.normalizer,
        get_sample=lambda index, task: deepcopy(sample))
    datamodule = SimpleNamespace(val_dataset=dataset, setup=lambda stage: None)
    monkeypatch.setattr("MotionExpert.config.load_config", lambda *args, **kwargs: expert.cfg)
    monkeypatch.setattr(ee4d_motion_dataset, "EE4D_Motion_DataModule", lambda cfg: datamodule)

    def construct_expert(*args, **kwargs):
        constructors.append(kwargs)
        return expert

    monkeypatch.setattr(infer, "MotionExpert", construct_expert)
    destination = tmp_path / "forecast.pt"
    arguments = ["motionexpert-infer", "--config", "unused.yaml", "--weights", "unused.pt",
                 "--output", str(destination)]
    if decode:
        arguments.append("--decode")
    monkeypatch.setattr("sys.argv", arguments)
    return infer.main, expert, constructors, destination


def test_archive_decode_rejects_invalid_current_reference_before_model_load(tmp_path, monkeypatch, capsys):
    sample = archive_sample(tmp_path, valid_current=False)
    torch.testing.assert_close(sample["misc"]["floor_transform"][5], torch.eye(4))
    main, expert, constructors, destination = run_cli(tmp_path, monkeypatch, sample, decode=True)
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "valid last-observed camera-floor reference" in capsys.readouterr().err
    assert not constructors and expert.model.calls == 0
    assert not destination.exists()


def test_invalid_current_reference_keeps_physical_forecast_available(tmp_path, monkeypatch):
    sample = archive_sample(tmp_path, valid_current=False)
    main, _, constructors, destination = run_cli(tmp_path, monkeypatch, sample, decode=False)
    main()
    output = torch.load(destination, weights_only=True)
    assert len(constructors) == 1
    assert output["future_motion_denormalized"].shape == (1, 5, 245)
    assert "decoded" not in output


def test_valid_current_reference_decodes_only_five_future_frames(tmp_path, monkeypatch):
    sample = archive_sample(tmp_path, valid_current=True)
    main, _, constructors, destination = run_cli(tmp_path, monkeypatch, sample, decode=True)
    main()
    output = torch.load(destination, weights_only=True)
    assert len(constructors) == 1
    assert output["future_motion_denormalized"].shape == (1, 5, 245)
    assert output["decoded"]["body_positions"].shape == (1, 5, 12, 3)
