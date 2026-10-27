"""Independent CPU checks for original continuous-mode Mobile P&P captures.

Native task flags decide task success. This module checks their original clocks,
the current human/robot VR binding, and the actually executed 50 Hz state trace;
it never runs the SDK, ONNX, dynamics, rendering, or a geometric success test.
Planner outputs were archived as hashes, not complete numeric forecasts: their
inference cannot be independently reproduced by this evidence-only audit.
"""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

import handover_sonic_teleop_reference as assembly
import pico_mobile_vr_reference as vr
from pico_mobile_quality import MOBILE_TASK_ID, evaluate_mobile_capture


CONTRACT = 'cvpr-live-pico-mobile-pp-independent-admission-v1'
CONTROL_CONTRACT = 'cvpr-pico-mobile-continuous-vr-mode1-async-reference-v1'
TASK_SOURCE_SHA256 = 'fee537d56e134b0399ff147013438168601498ab9b3ee068658dcad5fb8ca521'
NATIVE_SUCCESS_CRITERIA = .9
PLANNER_MODEL_SHA256 = '39b553e197f62f077975ba38512bc04781a3fc37c2af7c6756e04629f760edea'
TASK_SOURCE = (Path(__file__).resolve().parents[2] / 'VLMExpert/third_party/SIMPLE/src/simple/tasks/'
               'g1_wholebody_locomotion_pick_between_tables_teleop.py')
AUDITOR_FILES = frozenset(('audit_pico_teleop_episode.py', 'pico_live_input.py',
    'pico_mobile_audit.py', 'pico_mobile_quality.py', 'pico_mobile_vr_reference.py',
    'handover_sonic_teleop_reference.py', 'handover_contact_quality.py'))
BASE_CHECKS = frozenset(('live_source_and_rates', 'complete_recording_and_writer',
    'actual_fixed_sonic_artifacts', 'raw_fullbody_and_actual_command_ledger',
    'dense_actual_rgb_state_actions', 'independent_native_mobile_task_success',
    'all_original200hz_external_forces_zero', 'independent_mobile_continuous_reference'))


def require(value, message):
    if not value:
        raise ValueError(message)


def _sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            value.update(block)
    return value.hexdigest()


def _sha_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def _array(value, shape, label):
    value = np.asarray(value, dtype=np.float64)
    require(value.shape == shape and np.isfinite(value).all(), 'Invalid finite '+label)
    return value


def _unit(value, shape, label):
    value = _array(value, shape, label)
    require(np.max(np.abs(np.linalg.norm(value, axis=-1)-1.)) <= 1e-5,
            'Nonunit quaternion: '+label)
    return value


def _near(value, expected, label, tolerance=1e-10):
    expected = np.asarray(expected, dtype=np.float64)
    value = _array(value, expected.shape, label)
    require(np.allclose(value, expected, rtol=0, atol=tolerance), 'Different '+label)


def _integer(value, label):
    require(type(value) is int and value >= 0, 'Invalid nonnegative integer '+label)
    return value


def _digest(value, label):
    require(isinstance(value, str) and len(value) == 64 and
            all(c in '0123456789abcdef' for c in value), 'Invalid SHA256 '+label)
    return value


def _roles(value):
    require(isinstance(value, dict) and bool(value), 'Missing source SHA map')
    result = {}
    for name, digest in value.items():
        require(isinstance(name, str) and Path(name).is_absolute() and
                Path(name).name not in result, 'Duplicate/relative source role')
        result[Path(name).name] = (name, _digest(digest, name))
    return result


def _bind_pinned_sources(supplied, expected, bindings):
    """Pinned roles may come from another checkout; reread original files."""
    actual, wanted = _roles(supplied), _roles(expected)
    require(set(actual) == set(wanted) and all(actual[k][1] == wanted[k][1] for k in wanted),
            'Pinned Mobile source roles/digests differ')
    for name, digest in actual.values():
        original = Path(name)
        require(original.is_file() and _trace_sha(original, _stat(original)) == digest,
                'Original pinned Mobile source bytes changed: '+name)
        bindings[name] = digest


def validate_metadata(metadata, bindings):
    require(metadata.get('task_id') == MOBILE_TASK_ID and
            metadata.get('actual_task_source_sha256') == TASK_SOURCE_SHA256 and
            type(metadata.get('actual_native_success_criteria')) in (int, float) and
            metadata['actual_native_success_criteria'] == NATIVE_SUCCESS_CRITERIA,
            'Official Mobile task source identity or exact success criterion differs')
    require(TASK_SOURCE.is_file() and _sha(TASK_SOURCE) == TASK_SOURCE_SHA256,
            'Pinned official Mobile task implementation changed')
    bindings[str(TASK_SOURCE)] = TASK_SOURCE_SHA256
    require(metadata.get('mobile_control_protocol') == CONTROL_CONTRACT and
            metadata.get('mobile_control_policy') == 'continuous_vr_mode1' and
            type(metadata.get('mobile_encoder_mode')) is int and metadata['mobile_encoder_mode'] == 1 and
            metadata.get('mobile_planner_model_sha256') == PLANNER_MODEL_SHA256,
            'Mobile requires its actual continuous VR mode1/planner protocol')
    identity = metadata.get('mobile_vr_source', {})
    require(identity.get('official_sources_rehashed') is True, 'Mobile VR source freeze is missing')
    _bind_pinned_sources(identity.get('source_sha256'),
                         vr.verify_official_sources()['source_sha256'], bindings)
    return {'task_id': MOBILE_TASK_ID, 'official_task_source_sha256': TASK_SOURCE_SHA256,
            'native_success_criteria': NATIVE_SUCCESS_CRITERIA, 'selection_policy': 'native_only',
            'geometry_or_transport_success_thresholds_added': False}


def check_historical_sources(metadata, manifest_path, bindings):
    require(manifest_path is not None, 'Mobile audit requires an explicit historical-source snapshot manifest')
    path = Path(manifest_path).resolve(strict=True)
    require(path.is_file() and not Path(manifest_path).is_symlink(), 'Unsafe historical-source manifest')
    before = _stat(path)
    encoded = path.read_bytes()
    manifest = json.loads(encoded)
    require(manifest.get('status') == 'complete' and isinstance(manifest.get('sources'), list),
            'Historical source snapshot is incomplete')
    roles = {}
    for entry in manifest['sources']:
        key = entry['filename'], entry['sha256']
        require(key not in roles, 'Repeated historical source identity')
        roles[key] = entry
    declared = metadata.get('source_code_sha256_at_session_start')
    require(isinstance(declared, dict) and bool(declared), 'Original capture source declarations are missing')
    files = []
    for filename, digest in sorted(declared.items()):
        require(Path(filename).name == filename, 'Historical source role must be a basename')
        entry = roles.get((filename, _digest(digest, 'historical '+filename)))
        require(entry is not None, 'Original capture source snapshot is missing: '+filename)
        original = Path(entry['snapshot_path'])
        require(original.is_absolute() and original.name == filename and
                not original.is_symlink() and original.is_file(), 'Unsafe original source snapshot')
        stat = _stat(original)
        require(_sha(original) == digest and _stat(original) == stat, 'Original source snapshot bytes changed: '+filename)
        bindings[str(original)] = digest
        files.append({'filename': filename, 'path': str(original), 'sha256': digest})
    require(_stat(path) == before, 'Historical-source manifest changed during audit')
    manifest_sha = hashlib.sha256(encoded).hexdigest()
    bindings[str(path)] = manifest_sha
    return {'manifest_path': str(path), 'manifest_sha256': manifest_sha, 'files': files,
            'capture_source_declarations_modified': False,
            'scope': 'Exact original declared source bytes, independently read from retained historical snapshots; not the current audit implementation'}


def native_gate(qualities, start_time):
    for index, row in enumerate(qualities):
        require(type(row.get('completed_steps')) is int and row['completed_steps'] == index+1 and
                type(row.get('sim_time_s')) in (int, float) and math.isfinite(row['sim_time_s']) and
                math.isclose(row['sim_time_s'], start_time+(index+1)/50, rel_tol=0, abs_tol=1e-6),
                'Original native50Hz source has gaps or differs from captured physical time')
    result = evaluate_mobile_capture(qualities)
    result['excluded_unsupervised_warmup_ticks50'] = 0
    return result


def check_native_source_binding(root, source_name, bindings):
    """Verify the original selection/raw acquisition identity, without geometry."""
    root = Path(root)
    record = json.loads((root/'collection_record.json').read_text())
    require(record.get('status') == 'committed' and record.get('disposition') == 'kept' and
            record.get('source_kind') == 'live_pico_human' and
            record.get('episode_json_sha256') == bindings[str(root/'episode.json')],
            'Original sealed collection episode bytes changed')
    report = record.get('physical_report', {})
    digest = bindings[str(root/source_name)]
    require(report.get('native_flag_source_file') == source_name and
            report.get('native_flag_source_sha256') == digest,
            'Original collection selection raw-native source SHA differs')
    old = root/'quality_geometry_completion.json'
    if old.exists():
        require(old.is_file() and not old.is_symlink(), 'Unsafe retained geometry receipt')
        receipt = json.loads(old.read_text())
        require(receipt.get('episode_json_sha256') == bindings[str(root/'episode.json')] and
                receipt.get('metadata_sha256') == bindings[str(root/'metadata.json')],
                'Retained old receipt refers to another physical capture')
        previous = receipt.get('quality_geometry_sources', {}).get('raw_samples_sha256')
        if previous is not None:
            require(source_name == 'raw_quality_samples.jsonl' and previous == digest,
                    'Original acquired raw-native samples changed since old receipt')
        bindings[str(old)] = _sha(old)
    return {'path': source_name, 'sha256': digest, 'original_collection_raw_sha_verified': True,
            'retained_geometry_task_gate_used': False}


def _restored_vr(calibration):
    """Validate recorded calibration algebra, then restore its pure transforms.

    Original calibration FK is serialized evidence. Later current targets can
    be recomputed against the raw packet and independently captured root.
    """
    c = deepcopy(calibration)
    require(isinstance(c, dict) and c.get('contract') == vr.CONTRACT,
            'Missing actual measured-FK Mobile calibration')
    digest = c.pop('calibration_sha256', None)
    require(_digest(digest, 'VR calibration') == _sha_json(c), 'VR calibration bytes/hash differ')
    c['calibration_sha256'] = digest
    require(c.get('point_order') == list(vr.POINT_ORDER) and
            c.get('actual_body_names') == list(vr.ACTUAL_BODY_NAMES) and
            c.get('tip_offsets_applied_once_to_actual_body_origins') is True and
            c.get('actual_state_modified') is False and
            c.get('official_identity_neck_and_fixed_005m_torso_assumed') is False and
            c.get('neck_calibration') == 'measured_actual_torso_FK_origin_and_orientation_adaptation',
            'Mobile actual-FK calibration scope changed')
    _near(c['official_body_local_tip_offsets_m'], vr.VR_OFFSETS, 'official single tip offsets', 0.)
    root = _array(c['actual_root_pose7'], (7,), 'calibration root')
    root_rotation = Rotation.from_quat(_unit(root[3:], (4,), 'calibration root'), scalar_first=True)
    origins = _array(c['actual_body_origins_world'], (3, 3), 'actual calibration FK origins')
    rotations = Rotation.from_quat(_unit(c['actual_body_quaternions_world_wxyz'], (3, 4),
                                        'actual calibration FK'), scalar_first=True)
    tips = origins+rotations.apply(vr.VR_OFFSETS)
    local_points = root_rotation.inv().apply(tips-root[:3])
    local_rotations = root_rotation.inv()*rotations
    _near(c['actual_tip_points_world'], tips, 'calibration tip world points')
    _near(c['actual_tip_points_root_local'], local_points, 'calibration tip root-local points')
    _near(c['actual_point_quaternions_root_local_wxyz'], local_rotations.as_quat(scalar_first=True),
          'calibration point orientations')
    human = _array(c['initial_human_root_local_xyz_wxyz'], (3, 7), 'initial calibrated human')
    human_rotations = Rotation.from_quat(_unit(human[:, 3:], (3, 4), 'initial human'), scalar_first=True)
    neck_inverse = human_rotations[2].inv()
    _near(c['initial_human_neck_inverse_wxyz'], neck_inverse.as_quat(scalar_first=True), 'initial neck inverse')
    wrist_offsets = neck_inverse.apply(human[:, :3])[:2]-local_points[:2]
    offsets = local_rotations*(neck_inverse*human_rotations).inv()
    torso = root_rotation.inv().apply(origins[2]-root[:3])
    _near(c['wrist_position_offsets_m'], wrist_offsets, 'calibrated wrist offsets')
    _near(c['point_rotation_offsets_wxyz'], offsets.as_quat(scalar_first=True), 'calibrated orientation offsets')
    _near(c['actual_initial_torso_origin_root_local'], torso, 'calibrated torso origin')
    helper = vr.MobileVRReference.__new__(vr.MobileVRReference)
    helper.calibration = c
    helper._neck_inverse, helper._wrist_position_offsets = neck_inverse, wrist_offsets
    helper._rotation_offsets, helper._torso_origin_local = offsets, torso
    return helper


def check_reference(frame, event, evidence, metadata, bindings, *, helper=None):
    """Independently recompute current-packet VR and validate forecast evidence."""
    require(evidence.get('contract') == CONTROL_CONTRACT and type(evidence.get('encoder_mode')) is int and
            evidence['encoder_mode'] == 1 and evidence.get('reference_policy') == 'continuous_vr_mode1' and
            evidence.get('producer') == 'planner_lower_body_plus_explicit_VR_mode1' and
            evidence.get('input_shape') == [1, 1751] and evidence.get('input_dtype') == 'float32' and
            evidence.get('future_reference_indices') == list(range(0, 46, 5)) and
            evidence.get('lower_joint_names') == list(assembly.LOWER_BODY_NAMES) and
            evidence.get('tail_clamped_by_builder') is False and
            evidence.get('inactive_encoder_branches_zeroed') is True and
            evidence.get('model_forecasts_are_training_labels') is False and
            evidence.get('actual_state_written') is False and
            evidence.get('upper_body_frozen') is False and
            evidence.get('planner_upper_body_used_as_reference') is False and
            evidence.get('synchronous_diagnostic_only') is False,
            'Mobile continuous mode1/reference/strict-future semantics differ')
    _digest(evidence.get('encoder_input_sha256'), 'recorded1751 input')
    source = evidence['source']
    require(source.get('source_sequence') == frame.sequence and
            source.get('device_timestamp_ns') == frame.device_timestamp_ns and
            source.get('source_packet_reused_as_new_sample') is False and
            type(source.get('source_packet_held_for_control_interval')) is bool and
            type(source.get('source_age_s')) in (int, float) and
            0 <= source['source_age_s'] <= .1,
            'Mobile planner used another or stale source packet')
    # A packet can already have served the immediately preceding50Hz planner
    # tick while still being new to the30Hz encoder. The common raw ledger
    # independently requires strict advancement between actual30Hz commands.
    axes = [frame.inputs[name] for name in ('left_axis_x', 'left_axis_y', 'right_axis_x', 'right_axis_y')]
    _near(source['axes_left_x_y_right_x_y'], axes, 'same-packet joystick axes', 0.)
    _integer(evidence.get('epoch'), 'planner epoch')
    tick = _integer(evidence.get('tick50'), 'reference tick')
    cursor = _integer(evidence.get('cursor50'), 'reference cursor')
    require(_integer(evidence.get('timeline_origin_tick50'), 'reference origin')+cursor == tick,
            'Planner current reference cursor is misaligned')
    inference = evidence['last_inference']
    accepted = _integer(inference['accepted_tick50'], 'accepted forecast tick')
    require(accepted <= tick and _integer(inference['generated_tick50'], 'generated forecast tick') <= accepted and
            _integer(inference['submitted_tick50'], 'submitted forecast tick') <= accepted,
            'Accepted planner timeline is out of sequence')
    planner = inference['evidence']
    require(planner.get('contract') == 'cvpr-official-sonic11-locomotion-planner-v1' and
            planner.get('official_model_sha256') == PLANNER_MODEL_SHA256 and
            planner.get('all_generated_futures_are_forecast_references_not_labels') is True and
            planner.get('physics_steps') == 0 and planner.get('GPU_used') is False and
            planner.get('reference_only') is True and planner.get('async_cpu_worker', {}).get('cpu_only') is True,
            'Original pinned asynchronous CPU planner provenance differs')
    _digest(planner.get('valid_model_output_sha256'), 'valid planner output')
    _digest(planner.get('full_padded_model_output_sha256'), 'full planner output')
    resampling = planner['resampling']
    require(resampling.get('source_rate_hz') == 30 and resampling.get('reference_rate_hz') == 50 and
            resampling.get('endpoint_holds_eligible_as_complete_reference_future') is False and
            resampling.get('future_action_or_observation_labels_claimed') is False and
            type(resampling.get('fully_interpolated_valid_forecast_frames50')) is int and
            resampling['fully_interpolated_valid_forecast_frames50'] >= 46,
            'Planner does not establish a complete nonclamped46-frame forecast')
    pending = evidence.get('pending_request_id')
    if pending is not None:
        _integer(pending, 'pending planner request')
        deadline = evidence.get('pending_deadline_monotonic')
        require(type(deadline) in (int, float) and math.isfinite(deadline) and
                deadline > frame.host_received_s+source['source_age_s'],
                'Pending planner request already expired at original reference assembly')
    _bind_pinned_sources(evidence['official_source_sha256'],
        {str(path):assembly.SOURCE_SHA256[role] for role,path in assembly.SOURCE_PATHS.items()}, bindings)
    live = evidence['live_vr']
    require(live.get('source_sequence') == frame.sequence and
            live.get('source_device_timestamp_ns') == frame.device_timestamp_ns and
            live.get('root_attachment') == 'same_tick_current_actual_full_root_translation_and_rotation',
            'Mobile upper target packet/root attachment differs')
    root = _array(event['root_pose7'], (7,), 'same captured actual root')
    _near(live['actual_root_pose7_used_for_world_lift'], root, 'same-tick VR root', 1e-12)
    if helper is None:
        helper = _restored_vr(live['calibration'])
        _bind_pinned_sources(helper.calibration['source_identities']['source_sha256'],
                             metadata['mobile_vr_source']['source_sha256'], bindings)
    require(live.get('calibration_sha256') == helper.calibration['calibration_sha256'] and
            live.get('calibration') == helper.calibration, 'VR calibration changed within sealed episode')
    points, quaternions, recomputed = helper.prepare(frame.body24x7, root)
    for name, expected in recomputed.items():
        actual = live.get(name)
        if isinstance(expected, list):
            try:
                _near(actual, expected, 'recomputed live VR '+name, 1e-10)
            except (TypeError, ValueError):
                require(actual == expected, 'Different live VR '+name)
        else:
            require(actual == expected, 'Different live VR '+name)
    upper = evidence['live_upper_reference']
    require(upper.get('source_sequence') == frame.sequence and
            upper.get('device_timestamp_ns') == frame.device_timestamp_ns and
            upper.get('point_order') == list(vr.POINT_ORDER) and
            upper.get('world_points_already_offset') is True and
            upper.get('source') == 'caller_supplied_current_VR_three_point_world_targets',
            'Mode1 upper reference is not the current body-derived VR target')
    _near(upper['world_points_m'], points, 'executed VR world points')
    _near(upper['world_quaternions_wxyz'], quaternions, 'executed VR world orientations')
    return helper


def check_hands(frame, named_targets, metadata):
    """Recompute the declared local trigger/grip policy by joint name."""
    profile = metadata['hand_control_profile']
    require(profile.get('profile_id') == 'sonic_v1_1_middle_close_local_max_trigger_grip_v1' and
            profile.get('torque_collision_physics_modified') is False,
            'Mobile hand input policy identity differs')
    for side, sign in (('left', 1.), ('right', -1.)):
        inputs = [frame.inputs[side+'_'+key] for key in ('trigger', 'grip')]
        require(all(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1 for value in inputs),
                'Invalid original Mobile trigger/grip')
        amount = max(inputs)
        for suffix, angle in zip(('thumb_0', 'thumb_1', 'thumb_2', 'index_0', 'index_1', 'middle_0', 'middle_1'),
                                 (0., .7, .7, -1., -1.5, -1., -1.5)):
            name = side+'_hand_'+suffix+'_joint'
            actual = named_targets[name]
            require(type(actual) in (int, float) and math.isfinite(actual) and
                    math.isclose(actual, sign*angle*amount, rel_tol=0, abs_tol=1e-12),
                    'Current Mobile fingers differ from raw trigger/grip: '+name)


def _stat(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


@lru_cache(maxsize=8)
def _trace_sha(path, identity):
    # Reuse only the same immutable file identity, never the pathname alone.
    digest = _sha(path)
    require(_stat(path) == identity, 'Control trace changed while hashing')
    return digest


def check_control_trace(root, metadata, outcome, events, ledger, qualities, bindings):
    """Read every original50Hz execution for this sealed episode, without replay."""
    record_path = Path(root)/'collection_record.json'
    require(record_path.is_file() and not record_path.is_symlink(), 'Mobile needs its committed session identity')
    record = json.loads(record_path.read_text())
    original = Path(record['source_episode_directory'])
    require(original.is_absolute() and original.name == metadata['human_demo_uuid'] and
            original.parent.name == 'episodes' and original.resolve(strict=True) == Path(root).resolve(strict=True),
            'Original Mobile source UUID alias/session identity differs')
    trace = original.parent.parent/'control_trace.jsonl'
    require(trace.is_file() and not trace.is_symlink(), 'Original session50Hz control trace is missing')
    stat = _stat(trace)
    digest = _trace_sha(trace, stat)
    bindings[str(trace)] = digest
    start_tick = _integer(outcome['state']['start_tick50'], 'capture start global tick')
    ticks = len(qualities)
    commands = {row['physical_tick50']: row for row in ledger[1:]}
    captures = {event['physical_tick50']: event for event in events}
    first_command = ledger[1]
    first_time = events[0]['sim_time_s']
    found, count, previous_after, reference_tick_offset = False, 0, None, None
    max_lag, held_hands = 0., None
    with trace.open() as stream:
        for line in stream:
            row = json.loads(line, parse_constant=lambda word: (_ for _ in ()).throw(ValueError('Nonfinite trace JSON '+word)))
            if not found:
                if not (row.get('phase') == 'pico_follow' and row.get('tick50') == start_tick and
                        row.get('source_sequence') == first_command['source_sequence'] and
                        row.get('source_device_timestamp_ns') == first_command['device_timestamp_ns'] and
                        math.isclose(row.get('actual_before_step', {}).get('sim_time_s', -1.), first_time,
                                     rel_tol=0, abs_tol=1e-8)):
                    continue
                found = True
            if count == ticks:
                break
            require(row.get('phase') == 'pico_follow' and type(row.get('tick50')) is int and
                    row['tick50'] == start_tick+count, 'Original50Hz trace is interrupted or not consecutive')
            before, after = row['actual_before_step'], row['actual_after_step']
            require(before.get('state_finite') is True and after.get('state_finite') is True and
                    math.isclose(before['sim_time_s'], first_time+count/50, rel_tol=0, abs_tol=1e-6) and
                    math.isclose(after['sim_time_s'], first_time+(count+1)/50, rel_tol=0, abs_tol=1e-6),
                    'Original closed-loop physics state/clock is incomplete or nonfinite')
            for state in (before, after):
                _unit(_array(state['root_pose7'], (7,), 'actual trace root')[3:], (4,), 'actual trace root')
                _array(state['body_q29'], (29,), 'actual29 body joints')
                _array(state['body_dq29'], (29,), 'actual29 joint velocities')
                _array(state['hand_q14'], (14,), 'actual14 finger joints')
                require(state['body_joint_names'] == events[0]['joint_names'][:29] and
                        state['hand_joint_names'] == events[0]['joint_names'][29:],
                        'Actual trace named joint identity differs')
            if previous_after is not None:
                for name in ('root_pose7', 'body_q29', 'body_dq29', 'hand_q14'):
                    _near(before[name], previous_after[name], 'continuous actual50Hz '+name, 1e-9)
            previous_after = after
            require(type(row.get('terminated')) is bool and row['terminated'] is qualities[count]['official_task_success'] and
                    row.get('truncated') is False, 'Native flag differs from actual original environment return or was truncated')
            lag = row['loop_wall_timing']['lag_s']
            require(type(lag) in (int, float) and math.isfinite(lag) and lag <= .1,
                    'Original50Hz control exceeded its cumulative100ms deadline')
            max_lag = max(max_lag, lag)
            command = commands.get(count)
            require(row.get('new_command30') is (command is not None), 'Actual token update grid differs from30Hz command ledger')
            if command is not None:
                require(row.get('source_sequence') == command['source_sequence'] and
                        row.get('source_device_timestamp_ns') == command['device_timestamp_ns'] and
                        row.get('reference') == command['reference_evidence'] and
                        row.get('named_hand_targets') == command['named_hand_targets'],
                        'Actual50Hz execution differs from recorded human command/reference')
                offset = row['tick50']-row['reference']['tick50']
                reference_tick_offset = offset if reference_tick_offset is None else reference_tick_offset
                require(offset == reference_tick_offset, 'Planner50Hz reference timeline shifted inside the episode')
                event = captures[count]
                _near(event['root_pose7'], before['root_pose7'], 'actual captured trace root', 1e-12)
                actual = dict(zip(before['body_joint_names'], before['body_q29']))
                actual.update(zip(before['hand_joint_names'], before['hand_q14']))
                _near(event['physical_joint_q43_float64'], [actual[name] for name in event['joint_names']],
                      'actual captured trace joints', 1e-12)
                held_hands = command['named_hand_targets']
            else:
                require(row.get('reference') is None, 'An unlogged token/reference update occurred outside30Hz')
            require(row.get('named_hand_targets') == held_hands,
                    'Actual held50Hz finger command differs from its last30Hz source')
            count += 1
    require(found and count == ticks and _stat(trace) == stat,
            'Original complete50Hz execution is missing or changed during audit')
    return {'path': str(trace), 'sha256': digest, 'first_global_tick50': start_tick,
            'control_ticks50': count, 'actual_decoder_rate_hz': 50, 'actual_command_rate_hz': 30,
            'actual_physics_rate_hz': 200, 'maximum_cumulative_lag_s': max_lag,
            'planner_to_global_tick_offset': reference_tick_offset,
            'all_original_native_flags_match_environment_return': True,
            'actual_state_continuity_verified': True, 'dynamics_steps_performed': 0,
            'scope': 'Original serialized execution/state/deadline consistency; no independent decoder inference or dynamics replay'}
