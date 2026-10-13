"""Small v4 archives with meaningful rotations, masks and nonzero DINO bytes.

These fixtures exercise preservation rather than merely migrating all-zero
placeholders. They are independent of the v5 encoder and migration writer.
"""
from copy import deepcopy
from pathlib import Path

import numpy as np

from egomotion_pipeline import schema
from egomotion_pipeline.storage import Archive, atomic_json, quality_report, read_json, write_arrays
from test_native_dense22 import make_native_input, _axis_quaternion

V4_SCHEMA = 'egomotion-body22-camera-floor-dino5-v4'


def v4_archive(root, *, episodes=2, frames=8, missing_ground=False):
    root = Path(root)
    archive = Archive(root, {'fps': 30, 'fixture': 'independent-v4-migration'})
    provenance = read_json(root / '.pipeline/config.json')
    provenance['schema_version'] = V4_SCHEMA
    atomic_json(root / '.pipeline/config.json', provenance)
    archive.schema_version = V4_SCHEMA
    archive.specs = (schema.feature_specs_for_schema(V4_SCHEMA)
                     if hasattr(schema, 'feature_specs_for_schema') else schema.feature_specs())
    rng = np.random.default_rng(12418)
    for episode_index in range(episodes):
        arrays, record, _, rotations, floor_position = make_native_input(frames)
        for name, spec in archive.specs.items():
            if name not in arrays:
                shape = (frames,) if spec['shape'] == [1] else (frames, *spec['shape'])
                arrays[name] = np.zeros(shape, dtype=spec['dtype'])
                if name.endswith('_world_rotation'):
                    arrays[name].reshape(frames, -1, 4)[..., 0] = 1
        for name in ('index', 'episode_index'):
            arrays[name] = (np.arange(frames, dtype=np.int64) + episode_index * frames
                            if name == 'index' else np.full(frames, episode_index, dtype=np.int64))
        arrays['observation.environment.floor_height'][:] = .2
        arrays['observation.environment.floor_normal'][:] = [0, 0, 1]
        for name in ('floor', 'floor_verified', 'head_reference_basis'):
            arrays[f'observation.valid.{name}'][:] = True
        camera_position = floor_position.copy()
        camera_position[:, 2] += 1.6
        for modality in ('head', 'camera', 'head_reference'):
            arrays[f'observation.state.{modality}_world'][:] = camera_position
            arrays[f'observation.state.{modality}_world_rotation'][:] = np.stack([
                _axis_quaternion([0, 0, 1], .4 + frame * .07) for frame in range(frames)])
            arrays[f'observation.valid.{modality}_position'][:] = True
            arrays[f'observation.valid.{modality}_rotation'][:] = True
        arrays['observation.state.head_reference_height'][:] = 1.6
        arrays['observation.valid.head_reference_height'][:] = True
        body = arrays['observation.state.body_world'].reshape(frames, 22, 3)
        for side, wrist in (('left', 20), ('right', 21)):
            # Explicitly keep the measured hand wrist separate from the body wrist.
            hand = body[:, wrist, None] + rng.normal(0, .025, (frames, 21, 3)) + [0.04, -.03, .02]
            arrays[f'observation.state.hand_{side}_world'][:] = hand.reshape(frames, 63)
            arrays[f'observation.valid.hand_{side}_position'][:] = True
        arrays['observation.valid.body_position'][0, 3] = False
        body[0, 3] = 0
        arrays['observation.valid.body_rotation'][0, 4] = False
        arrays['observation.state.body_world_rotation'].reshape(frames, 22, 4)[0, 4] = [1, 0, 0, 0]
        arrays['observation.interpolated.body_position'][0, 5] = True
        arrays['observation.interpolated.body_rotation'][0, 6] = True
        arrays['observation.interpolated.hand_left_position'][0, 7] = True
        arrays['observation.valid.hand_left_position'][1, 2] = False
        arrays['observation.state.hand_left_world'].reshape(frames, 21, 3)[1, 2] = 0
        arrays['observation.valid.hand_right_position'][-1] = False
        arrays['observation.state.hand_right_world'][-1] = 0
        if missing_ground and episode_index == 0:
            record['metadata']['ground'] = {
                'height_m': 0., 'valid': False, 'verified': False,
                'mode': 'estimate', 'reason': 'no_stationary_feet',
                'window_fallback': {'mode': 'mean_valid_feet', 'offset_m': .02},
            }
            arrays['observation.valid.floor'][:] = False
            arrays['observation.valid.floor_verified'][:] = False
            arrays['observation.valid.head_reference_floor_position'][:] = False
            arrays['observation.valid.head_reference_floor_rotation'][:] = False
            arrays['observation.valid.head_reference_height'][:] = False
            arrays['observation.state.head_reference_floor_world'][:] = 0
            arrays['observation.state.head_reference_floor_world_rotation'][:] = [1, 0, 0, 0]
            arrays['observation.environment.floor_height'][:] = 0
            arrays['observation.state.head_reference_height'][:] = 0
        floor_available = (arrays['observation.valid.head_reference_floor_position'].reshape(frames) &
                           arrays['observation.valid.head_reference_floor_rotation'].reshape(frames))
        floor_body = np.einsum('tji,tkj->tki', rotations, body - floor_position[:, None])
        local_valid = arrays['observation.valid.body_position'] & floor_available[:, None]
        arrays['observation.state.body_head_floor'][:] = np.where(local_valid[:, :, None], floor_body, 0).reshape(frames, 66)
        arrays['observation.valid.body_head_floor_position'][:] = local_valid
        arrays['observation.interpolated.body_head_floor_position'][:] = (
            arrays['observation.interpolated.body_position'] & local_valid)
        tokens = np.arange(5 * 1024, dtype=np.float32).reshape(1, 5120) / 5120
        arrays['observation.features.dino'][:] = tokens + episode_index + np.arange(frames)[:, None] * .01
        arrays['observation.valid.dino'][:] = True
        arrays['observation.video_timestamp'][:] = np.arange(frames) / 30 + 10 * episode_index
        arrays['observation.video_time_error'][:] = .001
        record.update(episode_index=episode_index, source_uid=f'v4-fixture-{episode_index}',
                      source_key=f'egosuite:v4-fixture-{episode_index}', length=frames,
                      dataset_from_index=episode_index * frames,
                      dataset_to_index=(episode_index + 1) * frames,
                      input_fingerprint=f'v4-fixture-input-{episode_index}', video=None,
                      quality=quality_report(arrays), dino={'state': 'complete', 'valid_count': frames,
                                       'max_video_error_s': .01, 'encoder': schema.dino_format()})
        record['metadata'].update(operator_id=f'operator-{episode_index}',
                                  source_format='egosuite_mcap', body_basis_correction=None,
                                  rotation_basis_verified=False)
        record['metadata']['camera_pose_basis'] = 'synthetic_manual_identity'
        record['metadata']['head_reference'] = {
            'mode': 'ee4d', 'basis_policy': 'manual', 'basis_verified': True,
            'source_camera_basis': 'synthetic_manual_identity',
            # C @ EE4D_B = I, so the fixture's camera and reference agree.
            'camera_from_ee4d_camera': [[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]],
        }
        record['metadata']['ground'].setdefault('verified', True)
        write_arrays(archive.data_path(episode_index), arrays, archive.specs)
        atomic_json(archive.record_path(episode_index), deepcopy(record))
    archive._records = None
    archive.finalize()
    return Archive(root)
