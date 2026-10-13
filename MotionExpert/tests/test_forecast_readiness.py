"""Audit observed conditions, joint reconstruction and future prediction labels."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch

from MotionExpert import build_motion_qc_manifest as qc_cli
from MotionExpert import check_data as readiness
from MotionExpert.dataset.dense_motion_dataset import DenseMotionDataset
from MotionExpert.tests.test_dense_dataset import synthetic_arrays, write_archive


@pytest.fixture
def forecast_dataset(tmp_path):
    root=tmp_path/"archive"; root.mkdir()
    manifest,_=write_archive(root,synthetic_arrays(61))
    return DenseMotionDataset(archive_root=root,split_manifest=manifest,
        fps=5,window=11,forecast_prefix=6,segment_stride=5,train_task="fore",
        motion_supervision="future_only",
        require_full_window=True,cond_img_feat=True,cond_traj=True,
        min_valid_image_fraction=.5,do_normalization=False,
        rotation_basis_mode="source_native",target_joint_basis="egosuite_native_v1")


class SampleDataset:
    def __init__(self,dataset,samples):
        self.__dict__.update(dataset.__dict__)
        self.samples=samples
        self.idx_to_sidx_fidx=[(0,i*dataset.sampling_step) for i in range(len(samples))]

    def __getitem__(self,index):
        return self.samples[index]


def qc_config(dataset, *, fps=5,window=11,prefix=6,task="fore"):
    return SimpleNamespace(DATA=SimpleNamespace(ARCHIVE_ROOT=dataset.root,DATA_DIR=dataset.root,
        SPLIT_MANIFEST=dataset.split_manifest_path,COND_IMG_FEAT=True,COND_TRAJ=True,TRAIN_TASK=task,
        TRAJECTORY_CONDITION_PROB=0.,REQUIRE_FULL_WINDOW=task=="fore",FPS=fps,WINDOW=window,
        MOTION_SUPERVISION=getattr(dataset,"motion_supervision","future_only"),
        FORECAST_PREFIX=prefix,SEGMENT_STRIDE=5,ROTATION_BASIS_MODE="source_native",
        TARGET_JOINT_BASIS="egosuite_native_v1",MIN_VALID_IMAGE_FRACTION=.5))


def test_real_archive_readiness_reports_prefix_images_and_future_loss(forecast_dataset):
    report=readiness.inspect_dataset(forecast_dataset,trajectory_condition_prob=0.)
    assert report["profile"]=={"training_task":"fore","fps":5,"window":11,
        "forecast_prefix":6,"require_full_window":True,
        "motion_supervision":"future_only",
        "observed_image_frames":6,"future_motion_frames":5}
    checked=report["sample_checks"][0]
    assert checked["real_frames"]==11
    assert checked["supervised_frames"]==5
    assert checked["visible_image_frames"]==6
    assert checked["visible_trajectory_channels"]==0
    assert checked["future_only_loss"] and checked["history_motion_noise_only"]
    assert report["valid_visual_frames"]=={"egosuite":6}


def joint_sample_dataset(dataset):
    sample=deepcopy(dataset[0])
    sample["y"]["loss_mask"]=sample["y"]["padding_mask"].clone()
    result=SampleDataset(dataset,[sample])
    result.motion_supervision="joint"
    return result,sample


def test_joint_readiness_reports_eleven_supervised_and_five_evaluated_frames(forecast_dataset):
    dataset,sample=joint_sample_dataset(forecast_dataset)
    checked=readiness.inspect_sample(dataset,sample,require_visual=True)
    assert checked["motion_supervision"]=="joint"
    assert checked["supervised_frames"]==11
    assert checked["history_supervised_frames"]==6
    assert checked["future_evaluation_frames"]==5
    assert checked["visible_image_frames"]==6
    assert checked["visible_trajectory_channels"]==0
    assert not checked["future_only_loss"] and not checked["history_motion_noise_only"]


def test_real_archive_joint_dataset_reports_full_reconstruction_supervision(forecast_dataset):
    dataset=DenseMotionDataset(archive_root=forecast_dataset.root,
        split_manifest=forecast_dataset.split_manifest_path,fps=5,window=11,
        forecast_prefix=6,segment_stride=5,train_task="fore",require_full_window=True,
        motion_supervision="joint",cond_img_feat=True,do_normalization=False,
        rotation_basis_mode="source_native",target_joint_basis="egosuite_native_v1")
    report=readiness.inspect_dataset(dataset)
    assert report["profile"]["motion_supervision"]=="joint"
    assert report["sample_checks"][0]["supervised_frames"]==11
    assert report["sample_checks"][0]["future_evaluation_frames"]==5


def test_joint_readiness_rejects_missing_history_supervision(forecast_dataset):
    dataset,sample=joint_sample_dataset(forecast_dataset)
    sample["y"]["loss_mask"][:6]=False
    with pytest.raises(ValueError,match="all real frames"):
        readiness.inspect_sample(dataset,sample)


def test_joint_readiness_rejects_frozen_history_generation(forecast_dataset):
    dataset,sample=joint_sample_dataset(forecast_dataset)
    sample["y"]["motion_generation_mask"]=torch.tensor([False]*6+[True]*5)
    with pytest.raises(ValueError,match="include history and future"):
        readiness.inspect_sample(dataset,sample)


def test_joint_supervision_does_not_accept_history_only_labels(forecast_dataset):
    dataset,sample=joint_sample_dataset(forecast_dataset)
    sample["y"]["feature_valid_mask"][6:,:108]=False
    sample["misc"]["motion"][6:,:108]=0
    with pytest.raises(ValueError,match="future supervised dense12"):
        readiness.inspect_sample(dataset,sample)


def test_joint_qc_separates_full_supervision_and_future_quality(forecast_dataset):
    dataset,sample=joint_sample_dataset(forecast_dataset)
    sample["y"]["feature_valid_mask"][:6,:108]=False
    sample["misc"]["feature_valid_mask"][:6,:108]=False
    sample["misc"]["motion"][:6,:108]=0
    row=qc_cli.build_qc_report(dataset,require_visual=True)["sequence_status"][dataset.seq_names[0]][0]
    assert row["accepted"]
    assert row["supervised_frames"]==11
    assert row["body_position_valid_fraction"]==pytest.approx(5/11)
    assert row["future_body_position_valid_fraction"]==1.
    assert row["future_body_rotation_valid_fraction"]==1.


def test_readiness_never_repairs_bad_history_loss_before_checking(forecast_dataset):
    sample=forecast_dataset[0]; sample["y"]["loss_mask"][0]=True
    with pytest.raises(ValueError,match="future frames only"):
        readiness.inspect_sample(forecast_dataset,sample)


@pytest.mark.parametrize("key",["img_mask","traj_mask"])
def test_readiness_rejects_exposed_future_condition(forecast_dataset,key):
    sample=forecast_dataset[0]; sample["y"][key][6]=False
    with pytest.raises(ValueError,match="hide all future"):
        readiness.inspect_sample(forecast_dataset,sample)


def test_readiness_rejects_label_derived_generation_mask(forecast_dataset):
    sample=forecast_dataset[0]
    sample["y"]["motion_generation_mask"]=torch.zeros(11,dtype=torch.bool)
    with pytest.raises(ValueError,match="noise-only"):
        readiness.inspect_sample(forecast_dataset,sample)


def test_future_image_availability_does_not_change_observation_coverage(forecast_dataset):
    sample=forecast_dataset[0]; sample["y"]["valid_img_embs"][0]=False
    first=readiness.inspect_sample(forecast_dataset,sample,require_visual=True)
    sample["y"]["valid_img_embs"][6:]=False
    second=readiness.inspect_sample(forecast_dataset,sample,require_visual=True)
    assert first["visible_image_frames"]==second["visible_image_frames"]==5
    assert first["observed_image_valid_fraction"]==pytest.approx(5/6)


def test_future_images_cannot_satisfy_prefix_visual_readiness(forecast_dataset):
    sample=forecast_dataset[0]
    sample["y"]["valid_img_embs"][:6]=False
    sample["y"]["valid_img_embs"][6:]=True
    with pytest.raises(ValueError,match="observed interval"):
        readiness.inspect_sample(forecast_dataset,sample,require_visual=True)


def test_forecast_rejects_padding_and_full_window_ground_fallback(forecast_dataset):
    sample=forecast_dataset[0]; short=deepcopy(sample)
    short["misc"]["num_real_frames"]=10
    with pytest.raises(ValueError,match="complete physical window"):
        readiness.inspect_sample(forecast_dataset,short)
    metadata=json.loads(sample["misc"]["metadata_json"])
    metadata["ground"].update(window_fallback_used=True,estimation_scope="full_window")
    sample["misc"]["metadata_json"]=json.dumps(metadata)
    with pytest.raises(ValueError,match="observed prefix only"):
        readiness.inspect_sample(forecast_dataset,sample)


def test_readiness_rejects_wrong_source_sampling(forecast_dataset):
    sample=forecast_dataset[0]
    metadata=json.loads(sample["misc"]["metadata_json"])
    metadata["training_window"]["source_frame_indices"]=[i*2 for i in range(11)]
    sample["misc"]["metadata_json"]=json.dumps(metadata)
    with pytest.raises(ValueError,match="FPS/phase"):
        readiness.inspect_sample(forecast_dataset,sample)


@pytest.mark.parametrize("probability",[.35,1.])
def test_camera_training_experiment_keeps_deployment_image_only(forecast_dataset,probability):
    checked=readiness.inspect_sample(forecast_dataset,forecast_dataset[0],
                                     trajectory_condition_prob=probability)
    assert checked["training_auxiliary_trajectory_probability"]==probability
    assert checked["visible_trajectory_channels"]==0
    assert checked["supervised_frames"]==5
    assert checked["visible_image_frames"]==6


@pytest.mark.parametrize("probability",[True,-.1,1.1])
def test_readiness_rejects_invalid_auxiliary_camera_probability(forecast_dataset,probability):
    with pytest.raises(ValueError,match="trajectory_condition_prob"):
        readiness.inspect_sample(forecast_dataset,forecast_dataset[0],
                                 trajectory_condition_prob=probability)


def test_qc_fraction_uses_future_supervision_and_reports_unindexed_sequence(forecast_dataset):
    sample=forecast_dataset[0]
    sample["y"]["feature_valid_mask"][:6]=False
    sample["misc"]["motion"][:6]=0
    dataset=SampleDataset(forecast_dataset,[sample]); dataset.seq_names.append("unindexed")
    report=qc_cli.build_qc_report(dataset,require_visual=True,trajectory_condition_prob=0.)
    row=report["sequence_status"][dataset.seq_names[0]][0]
    assert row["accepted"]
    assert row["body_position_valid_fraction"]==1.
    assert row["body_rotation_valid_fraction"]==1.
    assert row["supervised_frames"]==5
    assert report["unindexed_sequences"]==["unindexed"]
    assert report["visual_features_checked"]


def test_qc_rejects_history_only_labels(forecast_dataset):
    sample=forecast_dataset[0]
    sample["y"]["feature_valid_mask"][6:,:108]=False
    sample["misc"]["motion"][6:,:108]=0
    dataset=SampleDataset(forecast_dataset,[sample])
    row=qc_cli.build_qc_report(dataset)["sequence_status"][dataset.seq_names[0]][0]
    assert not row["accepted"]
    assert "supervised dense12" in row["reason"]


def test_legacy_global_visual_check_keeps_later_valid_sample(forecast_dataset):
    samples=[deepcopy(forecast_dataset[0]) for _ in range(2)]
    for sample in samples:
        for key in ["loss_mask","img_mask","traj_mask","task_id"]:
            sample["y"].pop(key,None)
    samples[0]["y"]["valid_img_embs"]=torch.zeros(11,dtype=torch.bool)
    samples[1]["y"]["valid_img_embs"]=torch.ones(11,dtype=torch.bool)
    dataset=SampleDataset(forecast_dataset,samples)
    dataset.fps=15; dataset.train_task="recon"; dataset.require_full_window=False
    for sample in samples:
        sample["misc"].pop("metadata_json")
    report=readiness.inspect_dataset(dataset)
    assert report["checked_samples"]=={"egosuite":2}
    assert report["valid_visual_frames"]=={"egosuite":11}
    assert not report["sample_checks"][0]["future_only_loss"]


def test_qc_cli_forwards_forecast_profile_and_visual_policy(forecast_dataset,tmp_path,monkeypatch):
    capture={}
    def dataset_constructor(**kwargs):
        capture.update(kwargs)
        return forecast_dataset
    cfg=qc_config(forecast_dataset)
    monkeypatch.setattr(qc_cli,"load_config",lambda _:cfg)
    monkeypatch.setattr(qc_cli,"DenseMotionDataset",dataset_constructor)
    output=tmp_path/"qc.json"
    monkeypatch.setattr("sys.argv",["qc","--output",str(output)])
    qc_cli.main()
    assert capture["fps"]==5 and capture["train_task"]=="fore"
    assert capture["require_full_window"] and capture["cond_img_feat"]
    assert capture["min_valid_image_fraction"]==.5
    assert capture["motion_supervision"]=="future_only"
    report=json.loads(output.read_text())
    assert report["profile"]["future_motion_frames"]==5
    assert report["visual_features_checked"]


def test_legacy_qc_cli_keeps_null_task_and_motion_only_default(forecast_dataset,tmp_path,monkeypatch):
    cfg=qc_config(forecast_dataset,fps=15,window=4,prefix=2,task=None)
    monkeypatch.setattr(qc_cli,"load_config",lambda _:cfg)
    output=tmp_path/"legacy-qc.json"
    monkeypatch.setattr("sys.argv",["qc","--output",str(output)])
    qc_cli.main()
    report=json.loads(output.read_text())
    assert report["profile"]["fps"]==15
    assert report["profile"]["training_task"]=="recon"
    assert not report["visual_features_checked"]
    assert report["accepted_sequences"]==["egosuite:0"]


def test_forecast_qc_motion_only_reads_no_dino_feature_column(forecast_dataset,tmp_path,monkeypatch):
    constructed=[]
    def dataset_constructor(**kwargs):
        dataset=DenseMotionDataset(**kwargs); constructed.append(dataset); return dataset
    monkeypatch.setattr(qc_cli,"load_config",lambda _:qc_config(forecast_dataset))
    monkeypatch.setattr(qc_cli,"DenseMotionDataset",dataset_constructor)
    output=tmp_path/"motion-qc.json"
    monkeypatch.setattr("sys.argv",["qc","--output",str(output),"--motion-only"])
    qc_cli.main()
    assert not any("observation.features.dino" in columns for _,_,columns in constructed[0].reader.read_log)
    report=json.loads(output.read_text())
    assert not report["visual_features_checked"]
    assert report["sequence_status"]["egosuite:0"][0]["visible_image_frames"]==0
    assert report["accepted_sequences"]==["egosuite:0"]
