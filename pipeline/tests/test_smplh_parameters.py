"""Independent parameter FK, owned-file reads and temporal label boundaries."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from egomotion_pipeline.geometry import quat_to_matrix
from egomotion_pipeline.schema import PARENTS
from egomotion_pipeline.smplh_parameters import (
    DINO_COLUMN, MODEL_SHA256, PARAMETER_SCHEMA_VERSION, SMPLHParameterArchive,
    load_parameter_model, parameter_specs, parameters_from_fitted,
    prepare_parameter_window, validate_parameter_arrays, write_parameter_table,
)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def asset(tmp_path_factory):
    path = tmp_path_factory.mktemp("parameter-model") / "fixture.npz"
    parents = list(PARENTS)
    for wrist, first in ((20, 22), (21, 37)):
        for finger in range(5):
            j = first + 3 * finger
            parents.extend([wrist, j, j + 1])
    rest = np.zeros((52, 3)); rest[0] = [.3, -.2, .4]
    for j in range(1, 52):
        rest[j] = rest[parents[j]] + [.01 * (j % 3 + 1), .025, .02]
    template = np.zeros((6890, 3)); template[:52] = rest
    reg = np.zeros((52, 6890)); reg[np.arange(52), np.arange(52)] = 1
    shapes = np.zeros((6890, 3, 16))
    shapes[:52, 0, 0] = np.arange(52) * .001 + .01
    shapes[:52, 2, 15] = np.arange(52) * .0005 + .007
    np.savez_compressed(path, v_template=template, shapedirs=shapes, J_regressor=reg,
                        kintree_table=np.array([parents, np.arange(52)]))
    return path


@pytest.fixture
def model(asset):
    return load_parameter_model(asset, expected_sha256=_sha(asset), allow_test_fixture=True)


def _arrays(n=11):
    out = {}
    for key, spec in parameter_specs().items():
        width = spec["shape"][0]
        out[key] = np.zeros((n,) if width == 1 else (n, width), dtype=spec["dtype"])
    out["frame_index"] = np.arange(n, dtype=np.int64)
    out["index"] = np.arange(n, dtype=np.int64)
    out["timestamp"] = (np.arange(n) / 30).astype(np.float32)
    out["observation.source_timestamp"] = np.arange(n) / 30
    out["observation.source_frame_left"] = np.arange(n, dtype=np.int64)
    out["observation.source_frame_right"] = np.arange(n, dtype=np.int64)
    out["smplh.root_orient"][:, 2] = np.arange(n) * .03
    out["smplh.root_orient"][:, 0] = .07
    out["smplh.pose_body"] = np.linspace(-.12, .13, n * 63).reshape(n, 63).astype(np.float32)
    out["smplh.trans"] = np.array([2., 3., 1.]) + np.arange(n)[:, None] * [.003, -.002, .001]
    out["smplh.betas"][:, 0] = .2
    out["smplh.betas"][:, 15] = np.linspace(-.1, .3, n)
    for component in ("position", "rotation"):
        out[f"estimated.valid.body_{component}"][:] = True
        out[f"source.observed.body_{component}"][:] = True
        out[f"quality.{component}_confidence"][:] = .6
    out["source.feet_world"][:, [2, 5]] = .52
    out["source.feet_valid"][:] = True
    for key in list(out):
        if key.startswith("observation.state.") and key.endswith("_world_rotation"):
            out[key][:, 0] = 1
        if key.startswith("observation.valid."):
            out[key][:] = True
    for modality in ("camera", "head", "head_reference"):
        out[f"observation.state.{modality}_world"] = np.array([5., 4., 1.8]) + np.arange(n)[:, None] * [.003, -.002, .001]
    out["observation.state.hand_left_world"] = np.arange(n * 63).reshape(n, 63).astype(np.float32) * .001
    out["observation.state.hand_right_world"] = out["observation.state.hand_left_world"] + .2
    out["observation.environment.floor_height"][:] = .5
    out["observation.environment.floor_normal"][:, 2] = 1
    out["observation.video_timestamp"] = np.arange(n) / 30 + .001
    out["observation.video_time_error"][:] = .001
    return validate_parameter_arrays(out)


def _metadata(*, valid_ground=True):
    return {"coordinate_verified": True, "head_reference": {"basis_verified": True},
            "ground": {"height_m": .5, "valid": valid_ground, "verified": False,
                       "mode": "estimate", "window_fallback": {"mode": "mean_valid_feet", "offset_m": .02}},
            "source_data": "/removed-old-v5/source.parquet"}


def _rotation(aa):
    value = np.asarray(aa, float)
    angle = np.linalg.norm(value)
    if angle < 1e-12:
        return np.eye(3)
    x, y, z = value / angle
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * cross @ cross


def _independent_fk(asset, arrays):
    with np.load(asset, allow_pickle=False) as source:
        reg = source["J_regressor"][:22]
        j0 = reg @ source["v_template"]
        basis = np.einsum("jv,vcb->jcb", reg, source["shapedirs"])
    beta = arrays["smplh.betas"]
    rest = j0[None] + np.einsum("tb,jcb->tjc", beta, basis)
    aa = np.concatenate([arrays["smplh.root_orient"][:, None], arrays["smplh.pose_body"].reshape(-1, 21, 3)], 1)
    n = len(beta); p = np.zeros((n, 22, 3)); g = np.zeros((n, 22, 3, 3))
    for f in range(n):
        p[f, 0] = rest[f, 0] + arrays["smplh.trans"][f]
        g[f, 0] = _rotation(aa[f, 0])
        for j in range(1, 22):
            parent = PARENTS[j]
            p[f, j] = p[f, parent] + g[f, parent] @ (rest[f, j] - rest[f, parent])
            g[f, j] = g[f, parent] @ _rotation(aa[f, j])
    return p, g


def test_schema_has_body_parameters_and_no_dense_body_storage():
    specs = parameter_specs()
    assert specs["smplh.pose_body"]["shape"] == [63]
    assert specs["smplh.betas"]["shape"] == [16]
    assert "observation.state.body_world" not in specs
    assert "observation.state.body_world_rot6d_xyz" not in specs
    assert "observation.state.body_head_floor" not in specs
    assert "observation.state.hand_left_world" in specs
    assert DINO_COLUMN not in specs


def test_fk_even_phase_global_row6d_and_canonical_translation(model, asset):
    arrays = _arrays()
    selected = np.array([2, 4, 6, 8])
    p, g = _independent_fk(asset, {key: value[selected] for key, value in arrays.items()})
    window = prepare_parameter_window(arrays, _metadata(), model, start_frame=2, num_frames=4, canonicalize=False)
    np.testing.assert_allclose(window.arrays["observation.state.body_world"].reshape(4, 22, 3), p, atol=3e-7)
    features = window.arrays["observation.state.body_world_rot6d_xyz"].reshape(4, 22, 9)
    np.testing.assert_allclose(features[..., :6], g[..., :2, :].reshape(4, 22, 6), atol=1e-7)
    np.testing.assert_array_equal(window.arrays["frame_index"], selected)
    assert window.trajectory.shape == (4, 18)
    np.testing.assert_allclose(window.trajectory[1:, 15:18], np.tile([.006, -.004, 0], (3, 1)), atol=6e-7)
    heading_arrays = deepcopy(arrays)
    heading_arrays["observation.state.head_reference_world_rotation"][:] = [np.cos(.3), 0, 0, np.sin(.3)]
    canonical = prepare_parameter_window(heading_arrays, _metadata(), model, start_frame=2, num_frames=4)
    p2, g2 = _independent_fk(asset, canonical.arrays)
    np.testing.assert_allclose(canonical.arrays["observation.state.body_world"].reshape(4, 22, 3), p2, atol=5e-7)
    np.testing.assert_allclose(quat_to_matrix(canonical.arrays["observation.state.body_world_rotation"].reshape(4, 22, 4)), g2, atol=2e-7)
    assert canonical.metadata["absolute_rotation_accuracy_established"] is False
    assert canonical.metadata["target_model_axes_defined"] is True
    with pytest.raises(ValueError, match="even"):
        prepare_parameter_window(arrays, _metadata(), model, start_frame=1, num_frames=3)


def test_finite_parameters_never_promote_original_observations_or_estimate_masks(model):
    arrays = _arrays()
    arrays["source.observed.body_position"][:, 12] = False
    arrays["estimated.completed.body_position"][:, 12] = True
    arrays["estimated.valid.body_position"][:, 20] = False
    arrays["quality.position_confidence"][:, 20] = 0
    window = prepare_parameter_window(arrays, _metadata(), model, start_frame=0, num_frames=4)
    assert not window.arrays["observation.source_observed.body_position"][:, 12].any()
    assert window.arrays["observation.estimated.completed.body_position"][:, 12].all()
    assert not window.arrays["observation.valid.body_position"][:, 20].any()
    assert not window.arrays["observation.state.body_world"].reshape(4, 22, 3)[:, 20].any()
    bad = deepcopy(arrays); bad["estimated.completed.body_position"][:, 12] = False
    with pytest.raises(ValueError, match="completion"):
        validate_parameter_arrays(bad)


def test_fore_floor_fallback_uses_native_selected_prefix_not_fitted_or_future_feet(model):
    arrays = _arrays()
    feet = arrays["source.feet_world"].reshape(11, 2, 3)
    feet[:, :, 2] = 100
    feet[[0, 2], :, 2] = 1
    window = prepare_parameter_window(arrays, _metadata(valid_ground=False), model, start_frame=0,
                                      num_frames=5, task="fore", forecast_prefix=2, canonicalize=False)
    assert window.metadata["ground"]["height_m"] == pytest.approx(.98)
    assert window.metadata["ground"]["fallback_selected_frames"] == 2
    assert window.metadata["ground"]["window_fallback_used"] is True
    arrays["source.feet_valid"][:] = False
    arrays["source.observed.body_position"][:, [10, 11]] = False
    arrays["estimated.completed.body_position"][:, [10, 11]] = True
    window = prepare_parameter_window(arrays, _metadata(valid_ground=False), model, start_frame=0,
                                      num_frames=5, task="fore", forecast_prefix=2, canonicalize=False)
    assert not window.metadata["ground"]["valid"]
    assert not window.trajectory_valid.any()
    assert window.arrays["observation.valid.body_position"].all()


def _archive(tmp_path, asset, arrays, *, vision=True):
    root = tmp_path / "owned"
    (root / "weights").mkdir(parents=True)
    copied = root / "weights/model.npz"; copied.write_bytes(asset.read_bytes())
    write_parameter_table(root / "data/episode.parquet", arrays, row_group_size=3)
    (root / "meta").mkdir(); (root / ".pipeline/episodes").mkdir(parents=True)
    (root / "meta/info.json").write_text(json.dumps({"schema_version": PARAMETER_SCHEMA_VERSION, "fps": 30,
        "model": {"path": "weights/model.npz", "model_sha256": _sha(copied)}}))
    record = {"episode_index": 0, "length": len(arrays["timestamp"]), "source": "fixture",
              "data_path": "data/episode.parquet", "metadata": _metadata()}
    tokens = np.arange(len(arrays["timestamp"]) * 5120, dtype=np.float32).reshape(-1, 5120)
    if vision:
        (root / "vision").mkdir()
        pq.write_table(pa.table({"frame_index": arrays["frame_index"], "episode_index": arrays["episode_index"],
                                 "timestamp": arrays["timestamp"].astype(np.float64),
                                 "observation.video_timestamp": arrays["observation.video_timestamp"],
                                 "observation.video_time_error": arrays["observation.video_time_error"],
                                 DINO_COLUMN: pa.FixedSizeListArray.from_arrays(pa.array(tokens.reshape(-1)), 5120)}),
                       root / "vision/episode.parquet", row_group_size=4)
        record["vision_path"] = "vision/episode.parquet"
    (root / ".pipeline/episodes/00000000.json").write_text(json.dumps(record))
    return root, tokens


def test_owned_parameter_and_vision_reads_work_with_old_source_removed(tmp_path, asset, monkeypatch):
    arrays = _arrays()
    root, tokens = _archive(tmp_path, asset, arrays)
    import builtins
    original_open, original_parquet = builtins.open, pq.ParquetFile
    def guarded_open(path, *args, **kwargs):
        if str(path).startswith("/removed-old-v5"):
            raise AssertionError("Standalone reader touched an old dense dependency")
        return original_open(path, *args, **kwargs)
    def guarded_parquet(path, *args, **kwargs):
        if str(path).startswith("/removed-old-v5"):
            raise AssertionError("Standalone reader touched an old dense Parquet")
        return original_parquet(path, *args, **kwargs)
    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(pq, "ParquetFile", guarded_parquet)
    archive = SMPLHParameterArchive(root, allow_test_fixture=True)
    window = archive.read_window(0, start_frame=2, num_frames=4, canonicalize=False)
    np.testing.assert_array_equal(window.arrays[DINO_COLUMN], tokens[[2, 4, 6, 8]])
    np.testing.assert_array_equal(window.arrays["observation.video_timestamp"], arrays["observation.video_timestamp"][[2, 4, 6, 8]])
    assert window.metadata["training_window"]["requires_old_dense_archive"] is False
    assert not Path(window.metadata["source_data"]).exists()
    no_vision = archive.read_window(0, start_frame=0, num_frames=3, load_dino=False)
    assert DINO_COLUMN not in no_vision.arrays
    with pytest.raises(ValueError, match="shared official"):
        SMPLHParameterArchive(root)


def test_reader_rejects_wrong_vision_identity_and_external_dependency(tmp_path, asset):
    arrays = _arrays(); root, _ = _archive(tmp_path, asset, arrays)
    vision_path = root / "vision/episode.parquet"
    table = pq.read_table(vision_path)
    idx = table.schema.get_field_index("episode_index")
    pq.write_table(table.set_column(idx, "episode_index", pa.array(np.ones(11, np.int64))), vision_path)
    with pytest.raises(ValueError, match="DINO frame join"):
        SMPLHParameterArchive(root, allow_test_fixture=True).read_window(0, start_frame=0, num_frames=4)
    record_path = root / ".pipeline/episodes/00000000.json"
    record = json.loads(record_path.read_text()); record["data_path"] = "../old-v5/data.parquet"
    record_path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="owned relative"):
        SMPLHParameterArchive(root, allow_test_fixture=True).read_window(0, start_frame=0, num_frames=4)


def test_fitted_projection_uses_parameters_not_dense_cached_positions(model, asset):
    arrays = _arrays()
    p, g = _independent_fk(asset, arrays)
    source = {key: value for key, value in arrays.items() if key in parameter_specs()
              and not key.startswith(("smplh.", "source.", "estimated.", "quality."))}
    source["observation.state.body_world"] = (p + .02).reshape(11, 66)
    for component in ("position", "rotation"):
        source[f"observation.valid.body_{component}"] = arrays[f"source.observed.body_{component}"]
        source[f"observation.interpolated.body_{component}"] = arrays[f"source.interpolated.body_{component}"]
    aa = np.concatenate([arrays["smplh.root_orient"][:, None], arrays["smplh.pose_body"].reshape(11, 21, 3)], 1)
    fit = {"fitted.local_axis_angle": aa.reshape(11, 66), "fitted.smpl_trans": arrays["smplh.trans"],
           "fitted.betas": arrays["smplh.betas"], "estimated.valid.body_world_position": arrays["estimated.valid.body_position"],
           "estimated.valid.body_world_rotation": arrays["estimated.valid.body_rotation"],
           "quality.position_confidence": arrays["quality.position_confidence"],
           "quality.rotation_confidence": arrays["quality.rotation_confidence"]}
    projected = parameters_from_fitted(source, fit)
    window = prepare_parameter_window(projected, _metadata(), model, start_frame=0, num_frames=5, canonicalize=False)
    np.testing.assert_allclose(window.arrays["observation.state.body_world"].reshape(5, 22, 3), p[::2][:5], atol=3e-7)
    np.testing.assert_array_equal(projected["source.feet_world"].reshape(11, 2, 3), (p + .02)[:, [10, 11]].astype(np.float32))
    assert not projected["quality.observation_residual_valid"].any()


def test_optional_hand_pca_requires_mask_and_never_synthesizes_finger_truth():
    arrays = _arrays()
    arrays["smplh.pose_hand_pca"] = np.zeros((11, 90))
    with pytest.raises(ValueError, match="availability"):
        validate_parameter_arrays(arrays)
    arrays["smplh.valid.hand_pose"] = np.zeros((11, 2), bool)
    validated = validate_parameter_arrays(arrays)
    assert not validated["smplh.valid.hand_pose"].any()


def test_float64_video_error_is_lossless_in_owned_motion_and_vision(tmp_path, asset):
    arrays = _arrays()
    values = np.nextafter(np.linspace(.001, .002, 11), np.inf)
    arrays["observation.video_time_error"] = values
    root, _ = _archive(tmp_path, asset, arrays)
    physical = pq.read_table(root / "data/episode.parquet", columns=["observation.video_time_error"])
    assert physical["observation.video_time_error"].type == pa.float64()
    np.testing.assert_array_equal(physical["observation.video_time_error"].to_numpy(), values)
    window = SMPLHParameterArchive(root, allow_test_fixture=True).read_window(0, start_frame=0, num_frames=4)
    np.testing.assert_array_equal(window.arrays["observation.video_time_error"], values[[0, 2, 4, 6]])
    old_float32 = np.asarray(values, np.float32)
    arrays["observation.video_time_error"] = old_float32
    np.testing.assert_array_equal(validate_parameter_arrays(arrays)["observation.video_time_error"], old_float32.astype(np.float64))


def test_actual_hm_dino_error_precision_is_retained():
    root = Path("/gaozt-test1/datasets/EgoExo-4D-HM")
    # Read only a few scalar values, never full DINO features or a source audit.
    matches = list((root / "DINO/takes/cmu_bike02_4").glob("*.parquet"))
    if not matches:
        pytest.skip("Real HM DINO scalar sidecar unavailable")
    errors = None
    for path in matches:
        parquet = pq.ParquetFile(path)
        if "observation.video_time_error" in parquet.schema_arrow.names:
            errors = parquet.read_row_group(0, columns=["observation.video_time_error"])["observation.video_time_error"].to_numpy()[:11]
            break
    if errors is None or len(errors) < 11:
        pytest.skip("Real HM DINO error column unavailable")
    arrays = _arrays(); arrays["observation.video_time_error"] = errors
    np.testing.assert_array_equal(validate_parameter_arrays(arrays)["observation.video_time_error"], errors.astype(np.float64))


def test_reader_rejects_pts_mismatch_and_dense_columns(tmp_path, asset):
    arrays = _arrays(); root, _ = _archive(tmp_path, asset, arrays)
    vision_path = root / "vision/episode.parquet"
    original = pq.read_table(vision_path)
    column = original.schema.get_field_index("observation.video_timestamp")
    pq.write_table(original.set_column(column, "observation.video_timestamp", pa.array(arrays["observation.video_timestamp"] + .1)), vision_path)
    with pytest.raises(ValueError, match="time/validity/source-frame"):
        SMPLHParameterArchive(root, allow_test_fixture=True).read_window(0, start_frame=0, num_frames=3)
    motion_path = root / "data/episode.parquet"
    table = pq.read_table(motion_path)
    dense = pa.FixedSizeListArray.from_arrays(pa.array(np.zeros(11 * 66, np.float32)), 66)
    pq.write_table(table.append_column("observation.state.body_world", dense), motion_path)
    with pytest.raises(ValueError, match="stored dense body"):
        SMPLHParameterArchive(root, allow_test_fixture=True).read_window(0, start_frame=0, num_frames=3, load_dino=False)


def test_reader_keeps_hand_pca_separate_and_requires_owned_basis(tmp_path, asset):
    arrays = _arrays(); arrays["smplh.pose_hand_pca"] = np.full((11, 90), .1)
    arrays["smplh.valid.hand_pose"] = np.ones((11, 2), bool)
    root, _ = _archive(tmp_path, asset, arrays)
    with pytest.raises(ValueError, match="owned hand_basis"):
        SMPLHParameterArchive(root, allow_test_fixture=True).read_window(0, start_frame=0, num_frames=3, load_dino=False)
    basis_path = root / "weights/hand_basis.npz"
    np.savez(basis_path, left_hand_components=np.eye(45), right_hand_components=np.eye(45),
             left_hand_mean=np.zeros(45), right_hand_mean=np.zeros(45))
    info_path = root / "meta/info.json"; info = json.loads(info_path.read_text())
    info["hand_basis"] = {"path": "weights/hand_basis.npz", "sha256": _sha(basis_path)}
    info_path.write_text(json.dumps(info))
    window = SMPLHParameterArchive(root, allow_test_fixture=True).read_window(0, start_frame=0, num_frames=3,
                                                                           load_dino=False, canonicalize=False)
    np.testing.assert_array_equal(window.arrays["smplh.pose_hand_pca"], arrays["smplh.pose_hand_pca"][[0, 2, 4]].astype(np.float32))
    np.testing.assert_array_equal(window.arrays["observation.state.hand_left_world"], arrays["observation.state.hand_left_world"][[0, 2, 4]])
    # No decoder overwrites the preserved hand21 points from hand PCA.
    with basis_path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="basis SHA"):
        SMPLHParameterArchive(root, allow_test_fixture=True)


def test_full_hand_tree_and_real16_shape_contract_are_checked(tmp_path, asset):
    with np.load(asset, allow_pickle=False) as source:
        values = {key: source[key].copy() for key in source.files}
    values["kintree_table"][0, 23] = 20
    path = tmp_path / "bad-tree.npz"; np.savez_compressed(path, **values)
    with pytest.raises(ValueError, match="hand hierarchy"):
        load_parameter_model(path, expected_sha256=_sha(path), allow_test_fixture=True)
    values["shapedirs"] = values["shapedirs"][:, :, :10]
    path = tmp_path / "bad-shape.npz"; np.savez_compressed(path, **values)
    with pytest.raises(ValueError, match="sixteen actual"):
        load_parameter_model(path, expected_sha256=_sha(path), allow_test_fixture=True)


@pytest.mark.parametrize("flag", [{"training_eligible": False}, {"canonical_training_eligible": False},
                                  {"quarantine_reasons": ["Source frame mapping is unverified"]}])
def test_quarantine_cannot_bypass_canonical_admission_with_numeric_camera(tmp_path, asset, flag):
    arrays = _arrays(); root, _ = _archive(tmp_path, asset, arrays)
    record_path = root / ".pipeline/episodes/00000000.json"
    record = json.loads(record_path.read_text()); record["metadata"].update(flag)
    record_path.write_text(json.dumps(record))
    archive = SMPLHParameterArchive(root, allow_test_fixture=True)
    with pytest.raises(ValueError, match="Quarantined or ineligible"):
        archive.read_window(0, start_frame=0, num_frames=3)
    inspection = archive.read_window(0, start_frame=0, num_frames=3, canonicalize=False)
    np.testing.assert_array_equal(inspection.arrays["observation.state.hand_left_world"],
                                  arrays["observation.state.hand_left_world"][[0, 2, 4]])
    np.testing.assert_array_equal(inspection.arrays["source.observed.body_position"],
                                  arrays["source.observed.body_position"][[0, 2, 4]])
    assert not inspection.metadata["training_window"]["canonicalized"]
    for key, value in flag.items():
        assert inspection.metadata[key] == value


def test_actual_hm_parameters_and_same_real_smplh_asset_roundtrip():
    asset = Path("/gaozt-test1/datasets/MixEgoMotion/weights/smplh/male/model.npz")
    hm = Path("/gaozt-test1/datasets/EgoExo-4D-HM/EgoExo4D-HM/cmu_bike02_4/points_triangulated_world_results_merged.npz")
    if not asset.exists() or not hm.exists():
        pytest.skip("Licensed model and real HM sample are unavailable")
    model = load_parameter_model(asset)
    assert model.metadata["model_sha256"] == MODEL_SHA256
    with np.load(hm, allow_pickle=False) as raw:
        selected = np.array([0, 15, 54, 108])
        parameters = {"smplh.trans": raw["trans"][0, selected], "smplh.root_orient": raw["root_orient"][0, selected],
                      "smplh.pose_body": raw["pose_body"][0, selected], "smplh.betas": raw["betas_per_frame"][0, selected]}
    p, g = _independent_fk(asset, parameters)
    actual_p, actual_q, valid = model.decode(parameters["smplh.trans"], parameters["smplh.root_orient"],
                                             parameters["smplh.pose_body"], parameters["smplh.betas"], np.ones(4, bool))
    assert valid.all()
    np.testing.assert_allclose(actual_p, p, atol=2e-12)
    np.testing.assert_allclose(quat_to_matrix(actual_q), g, atol=2e-12)
