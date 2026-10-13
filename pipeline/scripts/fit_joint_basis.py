"""Fit and test joint bases from independent, paired reference orientations.

NPZ inputs are numeric rotations/masks plus a Unicode joint_names array. The
source and target orientations must describe the same physical poses in the
same world frame. This tool checks numerical agreement; it deliberately leaves
reference_verified and all verified flags false pending reference review.
"""

import argparse
from datetime import datetime, timezone
import hashlib
from io import BytesIO
from pathlib import Path

import numpy as np

from egomotion_pipeline.body_model import file_sha256
from egomotion_pipeline.joint_basis import fit_paired_rotation_basis
from egomotion_pipeline.schema import BODY_NAMES
from egomotion_pipeline.storage import atomic_json


def json_values(value):
    if isinstance(value, np.ndarray):
        return json_values(value.tolist())
    if isinstance(value, np.generic):
        return json_values(value.item())
    if isinstance(value, dict):
        return {key: json_values(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_values(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError('Calibration output contains a non-finite value')
    return value


def fit_reference(path, *, target_basis='smpl_body22_axes_v1',
                  max_error_deg=5., min_fit_samples=8, min_validation_samples=8):
    path = Path(path).resolve()
    snapshot = path.read_bytes()
    reference_sha256 = hashlib.sha256(snapshot).hexdigest()
    required = {'source_rotations', 'target_rotations', 'fit_mask',
                'validation_mask', 'joint_names'}
    with np.load(BytesIO(snapshot), allow_pickle=False) as reference:
        missing = required - set(reference.files)
        if missing:
            raise ValueError('Paired reference is missing: ' + ', '.join(sorted(missing)))
        names = reference['joint_names']
        if names.dtype.kind != 'U' or names.shape != (22,) or names.tolist() != BODY_NAMES:
            raise ValueError('Reference joint_names must exactly match the Unicode dense22 order')
        source = reference['source_rotations']
        target = reference['target_rotations']
        fit = reference['fit_mask']
        validation = reference['validation_mask']
        group_evidence = None
        if 'group_ids' in reference.files:
            groups = reference['group_ids']
            if groups.dtype.kind != 'U' or groups.shape != (len(source),):
                raise ValueError('group_ids must be a Unicode array of shape (T,)')
            if fit.shape != source.shape[:2] or validation.shape != source.shape[:2]:
                raise ValueError('Masks must match paired reference frame and joint axes')
            fit_groups = set(groups[np.any(fit, axis=1)].tolist())
            validation_groups = set(groups[np.any(validation, axis=1)].tolist())
            if fit_groups & validation_groups:
                raise ValueError('Fit and validation group_ids overlap')
            group_evidence = {'fit': sorted(fit_groups), 'validation': sorted(validation_groups)}
        result = fit_paired_rotation_basis(
            source, target, fit, validation, max_error_deg=max_error_deg,
            min_fit_samples=min_fit_samples, min_validation_samples=min_validation_samples)
    if file_sha256(path) != reference_sha256:
        raise ValueError('Paired reference changed while fitting')
    result['correction_matrices'] = result.pop('matrices')
    result.update(
        schema='egomotion-joint-basis-calibration-v1', target_basis=target_basis,
        state=('numerical_passed_reference_unverified' if result['numerical_pass'].all()
               else 'numerical_validation_failed'),
        reference_kind='unreviewed_paired_orientations',
        reference_files=[{'path': str(path), 'sha256': reference_sha256}],
        reference_groups=group_evidence,
        reference_assumptions=['same physical pose per source/target pair',
                               'same world frame', 'target joint axes match target_basis'],
        created_at=datetime.now(timezone.utc).isoformat(),
        production_archive_modified=False,
        rejection_reason='Independent reference provenance has not been reviewed')
    return json_values(result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True, help='Safe paired orientation NPZ')
    parser.add_argument('--output', required=True, help='New calibration report JSON path')
    parser.add_argument('--target-basis', default='smpl_body22_axes_v1')
    parser.add_argument('--max-error-deg', type=float, default=5.)
    parser.add_argument('--min-fit-samples', type=int, default=8)
    parser.add_argument('--min-validation-samples', type=int, default=8)
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Calibration report already exists: ' + str(output))
    report = fit_reference(
        args.reference, target_basis=args.target_basis, max_error_deg=args.max_error_deg,
        min_fit_samples=args.min_fit_samples,
        min_validation_samples=args.min_validation_samples)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, report)
    print(f"{report['state']}: {sum(report['numerical_pass'])}/22 numerical passes; "
          f"0/22 reviewed axes. Report: {output}")
    return 0 if all(report['numerical_pass']) else 2


if __name__ == '__main__':
    raise SystemExit(main())
