"""Explicit stable-support revision of saved Bend initialization, before physics.

The original fixed30 is retained. This profile moves only target XY by the
minimum distance making its true compiled COM projection five millimetres
inside the horizontal table rectangle. It does not ensure full footprint
support, modify warmup, write live state, or certify a native AMO benchmark.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import inspect
import json
import os
from pathlib import Path

import numpy as np

PROFILE = 'stable_support_v1'
MARGIN_M = .005
GEOMETRY_CONTRACT = 'cvpr-bend-pick-support-geometry-v1'
CONTRACT = 'cvpr-bend-pick-stable-support-initialization-v1'
_WORLD_TRANSLATION = [0., 0., .405]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _json(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Path): return str(value)
    raise TypeError(type(value).__name__)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        allow_nan=False, default=_json).encode()).hexdigest()


def _identity(path):
    path = Path(path).resolve(strict=True)
    before = path.stat()
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4*1024**2), b''): h.update(block)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns, before.st_ino, before.st_dev) ==
        (after.st_size, after.st_mtime_ns, after.st_ino, after.st_dev), 'Geometry input changed while hashing')
    return {'path': str(path), 'sha256': h.hexdigest(), 'size_bytes': before.st_size}


@contextmanager
def _data_root(data_root):
    previous = os.environ.get('SIMPLE_DATA_DIR')
    os.environ['SIMPLE_DATA_DIR'] = str(Path(data_root).resolve(strict=True))
    try:
        yield
    finally:
        if previous is None: os.environ.pop('SIMPLE_DATA_DIR', None)
        else: os.environ['SIMPLE_DATA_DIR'] = previous


def _sources():
    import simple
    root = Path(inspect.getfile(simple)).parent
    return {'helper': _identity(__file__), **{name: _identity(root/name) for name in (
        'engines/mujoco.py', 'assets/graspnet.py', 'assets/asset_manager.py',
        'core/actor.py', 'core/asset.py', 'core/types.py', 'utils.py')}}


def _asset(res_id, uid, data_root):
    from simple.assets import AssetManager
    require(res_id == 'graspnet1b' and isinstance(uid, str) and uid.isdecimal(), 'Unsupported support asset identity')
    with _data_root(data_root):
        return AssetManager.create(res_id+':'+uid)


def _meshes(asset):
    paths = list(asset.collision_meshes_mujoco)
    require(paths and len(paths) == len(set(map(str, paths))), 'Nonempty unique collision meshes required')
    return [_identity(path) for path in paths]


def _compile_geometry(asset):
    """Use the exact production object builder, without constructing Simulator."""
    import mujoco
    from simple.core.actor import ObjectActor
    from simple.engines.mujoco import MujocoSimulator
    actor = ObjectActor(asset)  # Identity pose; body_ipos is independent of world placement.
    spec = mujoco.MjSpec()
    MujocoSimulator._build_object(None, spec, spec.worldbody, actor)
    model = spec.compile()
    name = asset.label
    body = model.body(name).id
    ids = np.flatnonzero(model.geom_bodyid == body)
    require(len(ids) == len(asset.collision_meshes_mujoco), 'Builder changed target convex geometry count')
    return {'body_name': name, 'local_com_m': model.body_ipos[body].tolist(),
        'mass_kg': float(model.body_mass[body]), 'geom_count': len(ids),
        'mujoco_version': mujoco.__version__, 'construction': 'unchanged MujocoSimulator._build_object; identity pose; compile only',
        'physics_steps_taken': 0, 'GPU_used': False}


def prepare_geometry_manifest(path, data_root, *, res_id='graspnet1b', uid='0'):
    """Publish one fresh CPU-only geometry manifest; never overwrite evidence."""
    path = Path(path)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Geometry preparation requires CUDA_VISIBLE_DEVICES empty')
    require(not path.exists() and not path.is_symlink(), 'Use a fresh geometry manifest output')
    sources = _sources()
    asset = _asset(res_id, uid, data_root)
    meshes = _meshes(asset)
    geometry = _compile_geometry(asset)
    require(sources == _sources() and meshes == _meshes(asset), 'Builder or mesh changed during compilation')
    manifest = {'contract': GEOMETRY_CONTRACT, 'status': 'passed', 'profile': PROFILE,
        'asset': {'res_id': res_id, 'uid': uid, 'label': asset.label}, 'geometry': geometry,
        'source_identities': sources, 'collision_meshes': meshes}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        handle.write(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
        handle.flush(); os.fsync(handle.fileno())
    return load_geometry_manifest(path, data_root)


def load_geometry_manifest(path, data_root):
    """Verify actual builder, asset/ordered meshes, version and compiled COM."""
    import mujoco
    identity = _identity(path)
    manifest = json.loads(Path(path).read_text())
    require(manifest.get('contract') == GEOMETRY_CONTRACT and manifest.get('status') == 'passed'
        and manifest.get('profile') == PROFILE, 'Wrong support geometry contract')
    sources = _sources()
    require(set(manifest.get('source_identities', {})) == set(sources), 'Geometry source closure differs')
    for name, actual in sources.items():
        expected = manifest['source_identities'][name]
        require(actual['sha256'] == expected.get('sha256') and actual['size_bytes'] == expected.get('size_bytes'),
            'Geometry source bytes differ: '+name)
    asset_info = manifest['asset']
    asset = _asset(asset_info['res_id'], asset_info['uid'], data_root)
    require(asset.label == asset_info.get('label'), 'Geometry asset label differs')
    require(_meshes(asset) == manifest.get('collision_meshes'), 'Actual ordered collision mesh identity differs')
    # Recompiling this small target-only object prevents a forged/stale COM
    # value from being treated as authoritative just because meshes match.
    geometry = _compile_geometry(asset)
    require(geometry == manifest.get('geometry') and geometry['mujoco_version'] == mujoco.__version__,
        'Actual compiled target COM/mass/builder geometry differs')
    require(_identity(path) == identity, 'Geometry manifest changed while loading')
    return {**manifest, 'manifest_identity': identity, 'validated_payload_sha256': digest(manifest)}


def _vector(value, n, label):
    array = np.asarray(value, dtype=np.float64)
    require(array.shape == (n,) and np.isfinite(array).all(), 'Invalid '+label)
    return array


def _rotation(quaternion):
    w, x, y, z = _vector(quaternion, 4, 'unit quaternion')
    require(np.isclose(w*w+x*x+y*y+z*z, 1., rtol=0, atol=1e-10), 'Non-unit support quaternion')
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def _clearances(local, half):
    return {'x_minus': float(local[0]+half[0]), 'x_plus': float(half[0]-local[0]),
        'y_minus': float(local[1]+half[1]), 'y_plus': float(half[1]-local[1])}


def derive_stable_support_reset(compat_source_config, geometry):
    """Pure minimum-XY projection. Input is the native, not translated, reset."""
    payload = {k:v for k,v in geometry.items() if k not in ('manifest_identity', 'validated_payload_sha256')}
    require(geometry.get('validated_payload_sha256') == digest(payload) and
        payload.get('contract') == GEOMETRY_CONTRACT and payload.get('profile') == PROFILE
        and payload.get('status') == 'passed', 'Use an unchanged validated geometry manifest')
    source = compat_source_config
    require(source.get('uid') == 'g1_wholebody_bend_pick_mp' and source['robot_cfg']['uid'] == 'g1_wholebody'
        and source['metadata']['physics_dt'] == .002, 'Support revision requires the original native Bend reset frame')
    actors = source['layout']['actors']; target = actors['target']; table = actors['table']
    asset = target['asset']; expected = geometry['asset']
    require(asset.get('res_id') == expected['res_id'] and str(asset.get('uid')) == expected['uid']
        and asset.get('label') == expected['label'], 'Target differs from the compiled support asset')
    uid = expected['uid']
    before = deepcopy(target['pose'])
    require(before == source['dr_state_dict']['spatial'][uid], 'Target layout/DR pose differs before correction')
    position = _vector(before['position'], 3, 'target position')
    R = _rotation(before['quaternion']); table_R = _rotation(table['pose']['quaternion'])
    require(np.allclose(table_R[:, 2], [0, 0, 1], rtol=0, atol=1e-12), 'Support table must be horizontal and upright')
    size = _vector(table['size'], 3, 'table size')
    require(np.all(size > 0) and np.all(size[:2] > 2*MARGIN_M), 'Support table is too small')
    center = _vector(table['pose']['position'], 3, 'table position')
    local_com = _vector(geometry['geometry']['local_com_m'], 3, 'compiled local COM')
    mass = geometry['geometry']['mass_kg']
    require(type(mass) in (int, float) and np.isfinite(mass) and mass > 0, 'Invalid compiled target mass')
    com_before = position + R@local_com
    local_before = table_R.T@(com_before-center)
    half = size[:2]/2
    desired = np.clip(local_before[:2], -half+MARGIN_M, half-MARGIN_M)
    local_delta = desired-local_before[:2]
    # Exact interior no-op prevents any numerical drift of all original fields.
    delta = table_R[:, :2]@local_delta
    require(abs(delta[2]) <= 1e-12, 'Support correction may not move target Z')
    delta[2] = 0.
    applied = bool(np.any(local_delta != 0))
    after = deepcopy(before)
    if applied:
        after['position'][:2] = (position[:2]+delta[:2]).tolist()
    corrected = deepcopy(source)
    corrected['layout']['actors']['target']['pose'] = deepcopy(after)
    corrected['dr_state_dict']['spatial'][uid] = deepcopy(after)
    com_after = _vector(after['position'], 3, 'corrected target position') + R@local_com
    clearances = _clearances(table_R.T@(com_after-center), half)
    require(min(clearances.values()) >= MARGIN_M-1e-12, 'Projected COM lacks requested support margin')
    changes = []
    for axis in (0, 1):
        if after['position'][axis] != before['position'][axis]:
            for prefix in (['layout', 'actors', 'target', 'pose'], ['dr_state_dict', 'spatial', uid]):
                changes.append({'path': prefix+['position', axis],
                    'before': before['position'][axis], 'after': after['position'][axis]})
    receipt = {'contract': CONTRACT, 'profile': PROFILE, 'applied': applied, 'margin_m': MARGIN_M,
        'source_config_sha256': digest(source), 'corrected_config_sha256': digest(corrected),
        'geometry_manifest_identity': deepcopy(geometry['manifest_identity']),
        'target_asset': {**deepcopy(expected), 'body_name': geometry['geometry']['body_name']},
        'target_local_com_m': local_com.tolist(), 'target_mass_kg': mass,
        'target_geom_count': geometry['geometry']['geom_count'],
        'table_source': {'size_m': size.tolist(), 'pose': deepcopy(table['pose'])},
        'before_target_pose': before, 'after_target_pose': after,
        'before_com_xyz_m': com_before.tolist(), 'after_com_xyz_m': com_after.tolist(),
        'delta_world_xyz_m': delta.tolist(), 'before_edge_clearances_m': _clearances(local_before, half),
        'after_edge_clearances_m': clearances, 'changed_fields': changes,
        'actual_world_translation_m': list(_WORLD_TRANSLATION),
        'original_fixed_initialization_modified': applied, 'live_state_writes': False,
        'warmup_modified': False, 'reward_modified': False, 'collision_modified': False,
        'boundary': 'Explicit revised saved initialization; not the unchanged original fixed30. COM margin is not full-footprint/dynamic safety.'}
    return corrected, receipt


def verify_actual_support_model(model, data, receipt):
    """Read-only full-model check after mj_forward, strictly before first step."""
    import mujoco
    from pico_bend_pick_task import WORLD_TRANSLATION_M
    require(receipt.get('contract') == CONTRACT and receipt.get('profile') == PROFILE
        and receipt.get('margin_m') == MARGIN_M, 'Wrong stable-support receipt')
    translation = _vector(WORLD_TRANSLATION_M, 3, 'Bend world translation')
    require(translation.tolist() == receipt.get('actual_world_translation_m') == _WORLD_TRANSLATION,
        'Unexpected native-to-SONIC support frame')
    require(data.model is model and abs(float(data.time)) <= 1e-12, 'Support verification must precede all physics')
    require(np.isclose(model.opt.timestep, .005, rtol=0, atol=1e-12), 'Wrong actual Bend physics timestep')
    body = model.body(receipt['target_asset']['body_name']).id
    table = model.geom('table_geom').id
    expected_pose = receipt['after_target_pose']; expected_table = receipt['table_source']
    comparisons = ((model.body_ipos[body], receipt['target_local_com_m'], 'target local COM'),
        (data.xpos[body], np.asarray(expected_pose['position'])+translation, 'target origin'),
        (data.xmat[body].reshape(3,3), _rotation(expected_pose['quaternion']), 'target rotation'),
        (data.xipos[body], np.asarray(receipt['after_com_xyz_m'])+translation, 'target actual COM'),
        (data.geom_xpos[table], np.asarray(expected_table['pose']['position'])+translation, 'table position'),
        (data.geom_xmat[table].reshape(3,3), _rotation(expected_table['pose']['quaternion']), 'table rotation'),
        (model.geom_size[table], np.asarray(expected_table['size_m'])/2, 'table half sizes'))
    for actual, expected, label in comparisons:
        require(np.allclose(actual, expected, rtol=0, atol=1e-9), 'Actual support '+label+' differs')
    require(int(model.geom_type[table]) == int(mujoco.mjtGeom.mjGEOM_BOX), 'Actual support is not a box table')
    require(np.isclose(model.body_mass[body], receipt['target_mass_kg'], rtol=0, atol=1e-12)
        and np.count_nonzero(model.geom_bodyid == body) == receipt['target_geom_count'], 'Actual target mass/geometry count differs')
    local = data.geom_xmat[table].reshape(3,3).T@(data.xipos[body]-data.geom_xpos[table])
    clearances = _clearances(local, model.geom_size[table, :2])
    require(min(clearances.values()) >= MARGIN_M-1e-9, 'Actual target COM lacks the support margin')
    return {'status': 'passed', 'profile': PROFILE, 'applied': receipt['applied'], 'read_only': True,
        'physics_steps_taken': 0, 'sim_time_s': float(data.time),
        'initialization_receipt_sha256': digest(receipt),
        'actual_target_com_world_m': data.xipos[body].tolist(),
        'actual_target_origin_world_m': data.xpos[body].tolist(),
        'actual_table_center_world_m': data.geom_xpos[table].tolist(),
        'actual_edge_clearances_m': clearances, 'geometry_manifest_identity': deepcopy(receipt['geometry_manifest_identity'])}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true', required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--asset', default='graspnet1b:0')
    args = parser.parse_args(argv)
    res_id, uid = args.asset.split(':', 1)
    geometry = prepare_geometry_manifest(args.output, args.data_root, res_id=res_id, uid=uid)
    print(json.dumps({'status': 'passed', 'manifest_identity': geometry['manifest_identity'],
        'geometry': geometry['geometry']}, indent=2))


if __name__ == '__main__':
    raise SystemExit(main())
