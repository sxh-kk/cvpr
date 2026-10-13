"""Read-only HM SMPL-H decoding oracle, with isolated numerical pilot outputs.

Uses the actual HM male asset and all sixteen shape bases. Agreement with
official SMPL-H forward verifies decoding, not the accuracy of fitted motion.
No production archive, camera pose or cross-source certification is created.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "2")

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation
import torch
import smplx
from smplx.lbs import batch_rigid_transform, batch_rodrigues
from smplx.utils import Struct

from egomotion_pipeline.adapters import read_hm
from egomotion_pipeline.body_model import BodyModel
from egomotion_pipeline.geometry import quat_to_matrix
from egomotion_pipeline.schema import BODY_NAMES, HAND_NAMES, PARENTS
from egomotion_pipeline.storage import atomic_json


DEFAULT_TAKES = ["cmu_bike02_4", "sfu_cooking026_8", "uniandes_dance_020_37"]
DEFAULT_MODEL = Path("/gaozt-test1/datasets/MixEgoMotion/weights/smplh/male/model.npz")
DEFAULT_X_MODEL = Path("/gaozt-test1/guanzerong/UniEgoMotion/body_models/smplx/SMPLX_NEUTRAL.npz")
DEFAULT_HM_ROOT = Path("/gaozt-test1/datasets/EgoExo-4D-HM/EgoExo4D-HM")
DEFAULT_HM_REVISION = "6e1cd8627a0c5751b02493a87172dd3709c0c652"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for data in iter(lambda: handle.read(1 << 20), b""):
            digest.update(data)
    return digest.hexdigest()


def identity(path):
    st = Path(path).stat()
    return dict(bytes=st.st_size, mtime_ns=st.st_mtime_ns, ctime_ns=st.st_ctime_ns,
                device=st.st_dev, inode=st.st_ino)


def numeric_model(path):
    before = identity(path)
    with np.load(path, allow_pickle=False) as archive:
        result = {key: np.array(archive[key], copy=True) for key in archive.files}
    if before != identity(path):
        raise RuntimeError("Model changed during safe numerical read")
    if any(value.dtype.hasobject for value in result.values()):
        raise ValueError("Object arrays are not accepted")
    return result


def parent_tree(model):
    tree = model["kintree_table"]
    if tree.shape != (2, 52):
        raise ValueError("Expected complete 52-joint SMPL-H tree")
    lookup = {int(child): index for index, child in enumerate(tree[1])}
    parents = np.array([-1] + [lookup[int(value)] for value in tree[0, 1:]], dtype=np.int64)
    if not np.array_equal(parents[:22], PARENTS) or any(parents[j] >= j for j in range(1, 52)):
        raise ValueError("Model joint tree/order is incompatible")
    return parents


def generic_fk(model, betas, root, pose, trans):
    """Independent full52 homogeneous FK using SciPy rotations and raw assets."""
    parents = parent_tree(model)
    shaped = model["v_template"][None] + np.einsum("tb,vcb->tvc", betas, model["shapedirs"])
    rest = np.einsum("jv,tvc->tjc", model["J_regressor"], shaped)
    count = len(root)
    aa = np.zeros((count, 52, 3), np.float64)
    aa[:, 0] = root
    aa[:, 1:22] = pose.reshape(count, 21, 3)
    local = Rotation.from_rotvec(aa.reshape(-1, 3)).as_matrix().reshape(count, 52, 3, 3)
    transforms = np.broadcast_to(np.eye(4), (count, 52, 4, 4)).copy()
    transforms[:, :, :3, :3] = local
    transforms[:, 0, :3, 3] = rest[:, 0] + trans
    for joint in range(1, 52):
        parent = parents[joint]
        transforms[:, joint, :3, 3] = rest[:, joint] - rest[:, parent]
        transforms[:, joint] = transforms[:, parent] @ transforms[:, joint]
    return transforms[:, :22, :3, 3], transforms[:, :22, :3, :3], rest, local


def official_model(model, x_model, batch_size, *, num_betas=16):
    """Follow the HM wrapper: raw SMPL-H plus licensed SMPL-X hand PCA only."""
    data = dict(model)
    for key in ("hands_componentsl", "hands_componentsr", "hands_meanl", "hands_meanr"):
        data[key] = x_model[key]
    # Official HM wrapper zero-pads to the library's shape-space dimension.
    # The actual bundle has SIXTEEN genuine basis vectors, all retained.
    basis = data["shapedirs"]
    count = smplx.SMPL.SHAPE_SPACE_DIM
    data["shapedirs"] = np.pad(basis, ((0, 0), (0, 0), (0, count-basis.shape[-1])))
    body = smplx.SMPLH(str(DEFAULT_MODEL.parent), data_struct=Struct(**data),
                      gender="male", num_betas=num_betas, num_pca_comps=45,
                      use_pca=True, flat_hand_mean=False, batch_size=batch_size,
                      dtype=torch.float64)
    return body.double().eval()


def sample_indices(raw, count):
    valid = (raw["valid"][0].astype(bool) & np.isfinite(raw["trans"][0]).all(-1)
             & np.isfinite(raw["root_orient"][0]).all(-1)
             & np.isfinite(raw["pose_body"][0]).all(-1)
             & np.isfinite(raw["betas_per_frame"][0]).all(-1))
    candidates = np.flatnonzero(valid)
    if len(candidates) < 8:
        raise ValueError("Need at least eight real valid HM frames")
    selected = set()
    # Include actual beta-changing chunk boundaries before evenly spaced frames.
    for boundary in raw.get("chunk_ranges", np.empty((0, 2), int))[1:4, 0]:
        for index in (int(boundary)-1, int(boundary)):
            if 0 <= index < len(valid) and valid[index]:
                selected.add(index)
    for index in candidates[np.linspace(0, len(candidates)-1, count, dtype=int)]:
        if len(selected) >= count:
            break
        selected.add(int(index))
    return np.asarray(sorted(selected), np.int64)


def summaries(value):
    value = np.asarray(value, float).reshape(-1)
    return dict(count=len(value), mean=float(value.mean()), p95=float(np.quantile(value, .95)),
                max=float(value.max()))


def controlled_checks(body, model, x_model, seed):
    """Invalid-edge and pivot checks use copies; source archives stay immutable."""
    trans, root, pose, beta = [np.repeat(value[:1], 6, axis=0).copy()
                               for value in seed]
    source_valid = np.ones(6, bool)
    source_valid[0] = False
    trans[1, 0] = np.nan
    beta[2, 3] = np.nan
    root[3, 1] = np.nan
    pose[4, 0] = np.nan  # left_hip and its descendants only
    position, quaternion, actual = body.decode(trans, root, pose, beta, source_valid)
    expected = np.ones((6, 22), bool)
    expected[:4] = False
    descendants = {1}
    for joint in range(2, 22):
        if int(PARENTS[joint]) in descendants:
            descendants.add(joint)
    expected[4, list(descendants)] = False
    if not np.array_equal(actual, expected) or not np.isfinite(position).all() or not np.isfinite(quaternion).all():
        raise AssertionError("Independent invalid-mask propagation check failed")
    clean_trans, clean_root, clean_pose, clean_beta = [np.array(v[:1], copy=True) for v in seed]
    clean_xyz, _, _ = body.decode(clean_trans, clean_root, clean_pose, clean_beta, [True])
    rest0 = body.rest[0] + np.einsum("b,cb->c", clean_beta[0], body.shape_basis[0])
    pivot_error = float(np.max(np.abs(clean_xyz[0, 0] - (rest0+clean_trans[0]))))
    if pivot_error > 1e-10:
        raise AssertionError("SMPL translation/root-joint pivot convention failed")
    wrong_pivot = Rotation.from_rotvec(clean_root[0]).apply(rest0)+clean_trans[0]
    # Extra coefficients are only discarded when explicit upstream zero padding
    # is requested. All original sixteen are kept and used.
    extra_beta = np.pad(clean_beta, ((0, 0), (0, 4)), constant_values=2.)
    padded = BodyModel(body.metadata["model_path"], pad_shape_coefficients=True)
    padded_xyz, _, _ = padded.decode(clean_trans, clean_root, clean_pose, extra_beta, [True])
    with torch.no_grad():
        official20 = official_model(model, x_model, 1, num_betas=20)
        out20 = official20(betas=torch.from_numpy(extra_beta),
                           global_orient=torch.from_numpy(clean_root),
                           body_pose=torch.from_numpy(clean_pose),
                           left_hand_pose=torch.zeros((1, 45), dtype=torch.float64),
                           right_hand_pose=torch.zeros((1, 45), dtype=torch.float64),
                           transl=torch.from_numpy(clean_trans)).joints[:, :22].numpy()
    padding_error = float(np.max(np.abs(padded_xyz-out20)))
    strict_rejected = False
    try:
        body.decode(clean_trans, clean_root, clean_pose, extra_beta, [True])
    except ValueError:
        strict_rejected = True
    if not strict_rejected or not np.array_equal(padded_xyz, clean_xyz) or padding_error > 2e-6:
        raise AssertionError("Explicit shape-basis zero-padding contract failed")
    beta10 = clean_beta.copy(); beta10[:, 10:] = 0
    truncated, _, _ = body.decode(clean_trans, clean_root, clean_pose, beta10, [True])
    return dict(invalid_mask_propagation_passed=True, invalid_mask=actual.tolist(),
                root_pivot_max_element_error_m=pivot_error,
                wrong_rotate_rest_root_error_m=float(np.linalg.norm(wrong_pivot-clean_xyz[0, 0])),
                explicit20beta_pad_official_max_error_m=padding_error,
                excess_beta_rejected_without_explicit_padding=strict_rejected,
                native16_vs_incorrect10_beta_max_joint_error_m=float(np.linalg.norm(truncated-clean_xyz, axis=-1).max()))


def validate_take(args, take, raw_model, x_model, body, output):
    path = args.hm_root/take/"points_triangulated_world_results_merged.npz"
    before = identity(path)
    with np.load(path, allow_pickle=False) as source:
        raw = {key: np.array(source[key], copy=True) for key in source.files}
    if before != identity(path):
        raise RuntimeError("HM source changed during numerical snapshot")
    trans, root, pose, beta = [raw[key][0].astype(np.float64)
                              for key in ("trans", "root_orient", "pose_body", "betas_per_frame")]
    source_valid = raw["valid"][0].astype(bool)
    if beta.shape[1] != 16:
        raise ValueError("Pilot requires released sixteen-beta HM parameters")
    full_xyz, full_q, full_mask = body.decode(trans, root, pose, beta, source_valid)
    selected = sample_indices(raw, args.samples)
    trans_s, root_s, pose_s, beta_s = [value[selected].copy() for value in (trans, root, pose, beta)]
    xyz, q, mask = body.decode(trans_s, root_s, pose_s, beta_s, source_valid[selected])
    generic_xyz, generic_rotation, rest52, local52 = generic_fk(raw_model, beta_s, root_s, pose_s, trans_s)
    # Official batch_rigid_transform follows the full52 tree, independent of
    # the pipeline's body22 quaternion chain. Its upper3x3 gives global rotations.
    aa52 = np.zeros((len(selected), 52, 3), np.float64)
    aa52[:, 0] = root_s; aa52[:, 1:22] = pose_s.reshape(-1, 21, 3)
    official_local = batch_rodrigues(torch.from_numpy(aa52.reshape(-1, 3))).reshape(-1, 52, 3, 3)
    rigid_xyz, official_rigid = batch_rigid_transform(official_local, torch.from_numpy(rest52),
                                                    torch.from_numpy(parent_tree(raw_model)), dtype=torch.float64)
    rigid_xyz = rigid_xyz[:, :22].numpy()+trans_s[:, None]
    rigid_rotation = official_rigid[:, :22, :3, :3].numpy()
    official = official_model(raw_model, x_model, len(selected))
    hand_pca = raw["hand_pose"][0, selected].astype(np.float64)
    with torch.no_grad():
        result = official(betas=torch.from_numpy(beta_s),
                          global_orient=torch.from_numpy(root_s),
                          body_pose=torch.from_numpy(pose_s),
                          left_hand_pose=torch.from_numpy(hand_pca[:, :45]),
                          right_hand_pose=torch.from_numpy(hand_pca[:, 45:]),
                          transl=torch.from_numpy(trans_s), return_full_pose=True)
    official_xyz = result.joints[:, :22].numpy()
    pipeline_rotation = quat_to_matrix(q)
    errors = {
        "generic_fk_joint_position_error_m": summaries(np.linalg.norm(xyz-generic_xyz, axis=-1)),
        "generic_fk_rotation_element_error": summaries(np.abs(pipeline_rotation-generic_rotation)),
        "official_full_smplh_joint_position_error_m": summaries(np.linalg.norm(xyz-official_xyz, axis=-1)),
        "official_full52_rigid_joint_position_error_m": summaries(np.linalg.norm(xyz-rigid_xyz, axis=-1)),
        "official_full52_rigid_rotation_element_error": summaries(np.abs(pipeline_rotation-rigid_rotation)),
    }
    for name, result_error in errors.items():
        if result_error["max"] > args.tolerance:
            raise AssertionError(f"{name} exceeds tolerance: {result_error['max']}")
    if not mask.all() or np.abs(root_s).max() < .01 or np.abs(pose_s).max() < .01 or np.abs(beta_s).max() < .01:
        raise AssertionError("Selected official oracle frames must be valid actual nonzero poses/shapes")
    episode = read_hm(path, take, {"require_camera": False, "name": "egoexo4d_hm",
                                  "revision": args.hm_revision}, body)
    if episode.position_valid["camera"].any() or episode.position_valid["head_reference"].any():
        raise AssertionError("Body-only conversion must not fabricate an ego camera")
    if not np.allclose(episode.positions["body"], full_xyz, rtol=0, atol=0):
        raise AssertionError("Body adapter differs from direct FK decode")
    for name, source_slice in (("hand_left", slice(25, 46)), ("hand_right", slice(46, 67))):
        source_hand = raw["joints3d"][0, :, source_slice].astype(np.float64)
        expected_valid = source_valid[:, None] & np.isfinite(source_hand).all(-1)
        if (not np.array_equal(episode.position_valid[name], expected_valid)
                or not np.array_equal(episode.positions[name], np.where(expected_valid[..., None], source_hand, 0.))):
            raise AssertionError("Explicit hand21 values/masks differ from source joint blocks")
    inputs = output/"inputs"; inputs.mkdir(exist_ok=True)
    dest = inputs/f"hm-body-{take}.npz"
    np.savez_compressed(dest, timestamp=episode.timestamps, frame_index=np.arange(len(trans)),
                        body_world=episode.positions["body"], body_world_rotation=episode.rotations["body"],
                        body_position_valid=episode.position_valid["body"],
                        body_rotation_valid=episode.rotation_valid["body"],
                        hand_left_world=episode.positions["hand_left"],
                        hand_right_world=episode.positions["hand_right"],
                        hand_left_position_valid=episode.position_valid["hand_left"],
                        hand_right_position_valid=episode.position_valid["hand_right"],
                        trans=trans, root_orient=root, pose_body=pose, betas_per_frame=beta,
                        hand_pose=raw["hand_pose"][0], source_valid=source_valid,
                        chunk_ranges=raw.get("chunk_ranges", np.array([[0, len(trans)]])),
                        oracle_frame_indices=selected)
    checks = controlled_checks(body, raw_model, x_model, (trans_s, root_s, pose_s, beta_s))
    ranges = raw.get("chunk_ranges", np.array([[0, len(trans)]]))
    boundaries = []
    for boundary in ranges[1:, 0]:
        index = int(boundary)
        diff = float(np.linalg.norm(beta[index]-beta[index-1]))
        boundaries.append(dict(frame=index, beta_change_l2=diff,
                               pelvis_offset_change_m=float(np.linalg.norm((full_xyz[index, 0]-trans[index])-(full_xyz[index-1, 0]-trans[index-1]))),
                               both_frames_in_official_sample=bool(index in selected and index-1 in selected)))
    if before != identity(path):
        raise RuntimeError("Read-only HM input changed during validation")
    record = dict(passed=True, take=take, frames=len(trans), source_path=str(path), source_sha256=sha(path),
                  source_identity_before=before, source_identity_after=identity(path),
                  valid_body_joint_frames=int(full_mask.sum()),
                  full_body_decode_finite=bool(np.isfinite(full_xyz).all() and np.isfinite(full_q).all()),
                  selected_actual_frames=selected.tolist(), official_forward_count=len(selected),
                  max_abs_actual_root_angle=float(np.abs(root_s).max()),
                  max_abs_actual_body_angle=float(np.abs(pose_s).max()),
                  max_abs_actual_beta=float(np.abs(beta_s).max()),
                  errors=errors, controlled_checks=checks, chunks=ranges.tolist(), chunk_boundary_checks=boundaries,
                  all_source_hand_points_preserved=True,
                  source_hand_semantics="HM released joints3d OpenPose67 blocks25:46/46:67; fitted label provenance; valid is chunk finiteness not per-joint confidence",
                  camera_fabricated=False, input_npz=str(dest), input_npz_sha256=sha(dest),
                  metadata=episode.metadata)
    atomic_json(output/"reports/hm-body"/f"{take}.json", record)
    atomic_json(dest.with_suffix(".json"), dict(take=take, body_joint_names=BODY_NAMES, hand_joint_names=HAND_NAMES,
                position_unit="meter", rotation_order="wxyz", rotation_semantics="global_smplh_fk_wxyz",
                model_asset_sha256=body.metadata["model_sha256"], fps=30,
                camera_pose=None, world_registration_verified=False, source_sha256=record["source_sha256"],
                provenance=episode.metadata))
    print(json.dumps(dict(take=take, passed=True, frames=len(trans), samples=len(selected), errors=errors)), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--smplx-hand-basis", type=Path, default=DEFAULT_X_MODEL)
    parser.add_argument("--hm-root", type=Path, default=DEFAULT_HM_ROOT)
    parser.add_argument("--hm-revision", default=DEFAULT_HM_REVISION)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--takes", nargs="+", default=DEFAULT_TAKES)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--tolerance", type=float, default=2e-6)
    args = parser.parse_args()
    if len(args.hm_revision) != 40 or any(ch not in "0123456789abcdef" for ch in args.hm_revision):
        parser.error("HM release revision must be a fixed forty-character commit")
    if not 8 <= args.samples <= 12 or not 0 < args.tolerance <= 1e-5:
        parser.error("Use 8–12 oracle frames and a tolerance no greater than1e-5")
    torch.set_num_threads(2)
    output = args.output.resolve(); (output/"reports/hm-body").mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    # No pickle is loaded, including the separately licensed existing hand asset.
    raw_model = numeric_model(args.model)
    with np.load(args.smplx_hand_basis, allow_pickle=False) as source:
        x_model = {key: source[key].copy() for key in
                   ("hands_componentsl", "hands_componentsr", "hands_meanl", "hands_meanr")}
    if raw_model["v_template"].shape != (6890, 3) or raw_model["J_regressor"].shape != (52, 6890) or raw_model["shapedirs"].shape != (6890, 3, 16):
        raise ValueError("Pilot requires the fixed official HM male SMPL-H sixteen-basis asset")
    body = BodyModel(args.model, pad_shape_coefficients=False)
    files = [Path(__file__).resolve(), Path(inspect.getfile(BodyModel)), Path(inspect.getfile(read_hm)),
             Path(inspect.getfile(batch_rigid_transform)), Path(inspect.getfile(smplx.SMPLH))]
    code = [{"path": str(p), "sha256": sha(p)} for p in dict.fromkeys(files)]
    records = [validate_take(args, take, raw_model, x_model, body, output) for take in args.takes]
    report = dict(passed=True, completed_at=datetime.now(timezone.utc).isoformat(),
                  elapsed_s=time.monotonic()-started, schema="smplh-hm-body-oracle-pilot-v1",
                  source="egoexo4d_hm", source_revision=args.hm_revision,
                  model=body.metadata, official_library="smplx",
                  official_library_version=importlib.metadata.version("smplx"), torch_version=torch.__version__,
                  hand_basis=dict(path=str(args.smplx_hand_basis), sha256=sha(args.smplx_hand_basis),
                                  use="existing licensed SMPL-X hand PCA/means borrowed exactly as HM wrapper; body rest joints/betas remain SMPL-H"),
                  numeric_dtype="float64", tolerance=args.tolerance,
                  native_shape_basis_count=16, published_beta_count=16,
                  padding_policy="official wrapper zero-pads16→SMPL.SHAPE_SPACE_DIM; all16 real coefficients retained; no padding needed for actual HM body FK",
                  body_joint_names=BODY_NAMES, takes=records, total_frames=sum(r["frames"] for r in records),
                  official_forward_frames=sum(r["official_forward_count"] for r in records),
                  production_archive_modified=False, camera_poses_fabricated=False,
                  world_registration_certified=False, egosuite_joint_axes_certified=False,
                  evidence_boundary="Numerical decoding agreement; does not certify fitted motion truth, joint-center equivalence across sources, or world registration", code=code)
    atomic_json(output/"reports/hm-body/summary.json", report)
    print(json.dumps(dict(passed=True, summary=str(output/"reports/hm-body/summary.json"), elapsed_s=report["elapsed_s"])), flush=True)


if __name__ == "__main__":
    main()
