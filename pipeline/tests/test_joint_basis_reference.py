import importlib.util
from pathlib import Path

import numpy as np
import pytest

from egomotion_pipeline.geometry import axis_angle_to_quat, quat_to_matrix
from egomotion_pipeline.schema import BODY_NAMES


spec = importlib.util.spec_from_file_location(
    'fit_joint_basis_script', Path(__file__).parents[1]/'scripts/fit_joint_basis.py')
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)


def reference_file(path, **overrides):
    rng = np.random.default_rng(41)
    source = quat_to_matrix(axis_angle_to_quat(rng.normal(size=(32, 22, 3))))
    correction = quat_to_matrix(axis_angle_to_quat(rng.normal(size=(22, 3))))
    fit = np.zeros((32, 22), bool)
    fit[:16] = True
    payload = dict(source_rotations=source, target_rotations=source@correction,
                   fit_mask=fit, validation_mask=~fit,
                   joint_names=np.asarray(BODY_NAMES),
                   group_ids=np.asarray(['fit-person']*16+['heldout-person']*16))
    payload.update(overrides)
    np.savez(path, **payload)
    return correction


def test_numeric_success_remains_unreviewed_and_cannot_replace_report(tmp_path):
    path = tmp_path/'reference.npz'
    correction = reference_file(path)
    result = script.fit_reference(path)
    np.testing.assert_allclose(result['correction_matrices'], correction, atol=1e-12)
    assert result['numerical_pass'] == [True]*22
    assert result['verified'] == [False]*22
    assert result['reference_verified'] is False
    assert result['state'] == 'numerical_passed_reference_unverified'
    assert result['reference_groups']['fit'] == ['fit-person']
    output = tmp_path/'calibration.json'
    assert script.main(['--reference', str(path), '--output', str(output)]) == 0
    with pytest.raises(FileExistsError):
        script.main(['--reference', str(path), '--output', str(output)])


def test_reference_rejects_joint_order_and_cross_person_split_leakage(tmp_path):
    path = tmp_path/'reference.npz'
    reference_file(path, joint_names=np.asarray(BODY_NAMES[::-1]))
    with pytest.raises(ValueError, match='dense22 order'):
        script.fit_reference(path)
    reference_file(path, group_ids=np.asarray(['same-person']*32))
    with pytest.raises(ValueError, match='group_ids overlap'):
        script.fit_reference(path)


def test_reference_is_safe_npz_and_requires_defined_fields(tmp_path):
    path = tmp_path/'reference.npz'
    np.savez(path, arbitrary=np.array([1]))
    with pytest.raises(ValueError, match='missing'):
        script.fit_reference(path)
    reference_file(path, joint_names=np.asarray(BODY_NAMES, dtype=object))
    with pytest.raises(ValueError, match='Object arrays'):
        script.fit_reference(path)


def test_reference_replaced_during_fit_cannot_bind_new_hash(tmp_path, monkeypatch):
    path = tmp_path/'reference.npz'
    reference_file(path)
    original_fit = script.fit_paired_rotation_basis

    def replace_evidence(*args, **kwargs):
        result = original_fit(*args, **kwargs)
        # A different, still valid reference must not silently acquire the SHA
        # for matrices fitted from the original snapshot.
        reference_file(path, target_rotations=np.tile(np.eye(3), (32, 22, 1, 1)))
        return result

    monkeypatch.setattr(script, 'fit_paired_rotation_basis', replace_evidence)
    with pytest.raises(ValueError, match='changed while fitting'):
        script.fit_reference(path)
