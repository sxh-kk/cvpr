"""SMPL-H's first 22 kinematic joints; no mesh rendering or hand PCA decoding.

J(beta) is regressed from the shaped template, then forward kinematics uses
root_orient + the 21 pose_body rotations. This reproduces the model's body
joints, which are unaffected by skinning/hand pose. Model assets are user-owned.
"""

from pathlib import Path
import hashlib
import pickle
import numpy as np
from .geometry import axis_angle_to_quat, quat_multiply, quat_to_matrix
from .schema import PARENTS


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class BodyModel:
    def __init__(self, path, pad_shape_coefficients=False):
        path = Path(path)
        # Only licensed, trusted model assets may use pickle. Dataset NPZs do not.
        if path.suffix == ".pkl":
            with path.open("rb") as handle:
                asset = pickle.load(handle, encoding="latin1")
        else:
            with np.load(path, allow_pickle=True) as handle:
                asset = {key: handle[key] for key in handle.files}
        template = np.asarray(asset["v_template"], dtype=np.float64)
        shapedirs = np.asarray(asset["shapedirs"], dtype=np.float64)
        regressor = asset["J_regressor"]
        if isinstance(regressor, np.ndarray) and regressor.dtype == object:
            regressor = regressor.item()
        if hasattr(regressor, "toarray"):
            regressor = regressor.toarray()
        regressor = np.asarray(regressor, dtype=np.float64)
        tree = np.asarray(asset["kintree_table"])
        if (template.ndim != 2 or template.shape[1] != 3 or shapedirs.ndim != 3
                or shapedirs.shape[:2] != template.shape or regressor.ndim != 2
                or regressor.shape[0] < 22 or regressor.shape[1] != len(template)
                or tree.ndim != 2 or tree.shape[0] != 2 or tree.shape[1] < 22):
            raise ValueError("Invalid SMPL-H template/shape/regressor asset")
        vertex_count, joint_count = len(template), len(regressor)
        tree_joint_count = tree.shape[1]
        # SMPL-X and SMPL share SMPL-H's first 22 body parents, but their
        # templates/shape spaces are different. Check the full asset before
        # taking the body subset; reduced synthetic FK fixtures remain valid.
        if vertex_count == 10475 or joint_count == 55 or tree_joint_count == 55:
            raise ValueError("SMPL-X asset is incompatible with HM parameters; use matching SMPL-H assets")
        if (joint_count == 24 or tree_joint_count == 24
                or (vertex_count == 6890 and (joint_count != 52 or tree_joint_count != 52))):
            raise ValueError("Asset does not have the SMPL-H 52-joint model; use matching SMPL-H assets")
        regressor = regressor[:22]
        lookup = {int(child): idx for idx, child in enumerate(tree[1])}
        parents = np.array([-1] + [lookup[int(parent)] for parent in tree[0, 1:22]])
        if not np.array_equal(parents, PARENTS):
            raise ValueError("Model does not have the SMPL 22-joint body hierarchy")
        self.rest = regressor @ template
        self.shape_basis = np.einsum("jv,vcb->jcb", regressor, shapedirs)
        if not np.isfinite(self.rest).all() or not np.isfinite(self.shape_basis).all():
            raise ValueError("Non-finite model asset")
        self.pad = pad_shape_coefficients
        self.metadata = {"model_path": str(path.resolve()), "model_sha256": file_sha256(path),
                         "vertex_count": vertex_count, "joint_count": joint_count,
                         "tree_joint_count": tree_joint_count,
                         "shape_basis_count": shapedirs.shape[2], "decoder": "smplh_body22_fk",
                         "zero_pad_shape_basis": bool(self.pad)}

    def decode(self, trans, root_orient, pose_body, betas, valid):
        trans, root_orient, pose_body, betas = [np.asarray(x, dtype=float) for x in [trans, root_orient, pose_body, betas]]
        count = len(trans)
        if trans.shape != (count, 3) or root_orient.shape != (count, 3) or pose_body.shape != (count, 63) or betas.ndim != 2 or len(betas) != count:
            raise ValueError("HM body parameter shape mismatch")
        basis_count = self.shape_basis.shape[-1]
        if betas.shape[1] > basis_count and not self.pad:
            raise ValueError(f"Model has {basis_count} shape bases, data has {betas.shape[1]}; use matching assets or explicit upstream zero-padding")
        finite_shape = np.isfinite(betas).all(-1)
        betas = np.nan_to_num(betas[:, :basis_count], nan=0, posinf=0, neginf=0)
        rest = self.rest[None] + np.einsum("tb,jcb->tjc", betas, self.shape_basis[..., :betas.shape[-1]])
        local_aa = np.concatenate([root_orient[:, None], pose_body.reshape(count, 21, 3)], axis=1)
        local_valid = np.isfinite(local_aa).all(-1)
        local = axis_angle_to_quat(np.nan_to_num(local_aa, nan=0, posinf=0, neginf=0))
        positions = np.zeros((count, 22, 3))
        rotations = np.zeros((count, 22, 4))
        mask = np.zeros((count, 22), bool)
        positions[:, 0] = rest[:, 0] + np.nan_to_num(trans, nan=0, posinf=0, neginf=0)
        rotations[:, 0] = local[:, 0]
        mask[:, 0] = np.asarray(valid, bool) & finite_shape & np.isfinite(trans).all(-1) & local_valid[:, 0]
        for child in range(1, 22):
            parent = PARENTS[child]
            rotations[:, child] = quat_multiply(rotations[:, parent], local[:, child])
            positions[:, child] = positions[:, parent] + np.einsum("tij,tj->ti", quat_to_matrix(rotations[:, parent]), rest[:, child]-rest[:, parent])
            mask[:, child] = mask[:, parent] & local_valid[:, child]
        return positions, rotations, mask
