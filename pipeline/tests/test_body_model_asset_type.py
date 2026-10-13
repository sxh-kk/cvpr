"""Reject incompatible model families even when their body22 tree matches."""

import numpy as np
import pytest

from egomotion_pipeline.body_model import BodyModel
from egomotion_pipeline.schema import PARENTS


def write_asset(path, vertices, joints):
    parents = np.zeros(joints, dtype=np.int64)
    parents[:22] = PARENTS
    regressor = np.zeros((joints, vertices))
    regressor[np.arange(22), np.arange(22)] = 1
    np.savez(path, v_template=np.zeros((vertices, 3)),
             shapedirs=np.zeros((vertices, 3, 2)), J_regressor=regressor,
             kintree_table=np.array([parents, np.arange(joints)]))
    return path


@pytest.mark.parametrize("vertices,joints", [(10475, 55), (10475, 22), (22, 55)])
def test_rejects_recognizable_smplx_assets(tmp_path, vertices, joints):
    asset = write_asset(tmp_path/"model.npz", vertices, joints)
    with pytest.raises(ValueError, match="SMPL-X.*matching SMPL-H"):
        BodyModel(asset)


def test_rejects_smpl_without_hand_hierarchy(tmp_path):
    asset = write_asset(tmp_path/"model.npz", 6890, 24)
    with pytest.raises(ValueError, match="SMPL-H 52-joint.*matching SMPL-H"):
        BodyModel(asset)


def test_accepts_full_smplh_dimensions_and_records_counts(tmp_path):
    model = BodyModel(write_asset(tmp_path/"model.npz", 6890, 52))
    assert model.metadata["vertex_count"] == 6890
    assert model.metadata["joint_count"] == model.metadata["tree_joint_count"] == 52


def test_existing_reduced_fk_fixture_remains_usable(model_path):
    model = BodyModel(model_path)
    positions, rotations, valid = model.decode(
        np.zeros((1, 3)), np.zeros((1, 3)), np.zeros((1, 63)),
        np.zeros((1, 2)), np.ones(1, bool))
    np.testing.assert_allclose(positions[0], model.rest)
    np.testing.assert_allclose(rotations[0], np.tile([1, 0, 0, 0], (22, 1)))
    assert valid.all()
    assert model.metadata["vertex_count"] == model.metadata["joint_count"] == 22
