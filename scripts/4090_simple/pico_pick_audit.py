"""Independent CPU admission for three explicitly identified PICO pick tasks.

This is a new authority; it does not mutate the Handover, Mobile or Grasp
auditors. Historical recorder bytes and current auditor/RTX producer bytes
have separate roles. No SDK, ONNX session, integration step or GPU is started.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import uuid

import numpy as np
from PIL import Image

import audit_pico_teleop_episode as shared
from audit_pico_teleop_episode import (require, sha, vector, pose, close,
    _owned, _read, _read_line, _bound_identities, _check_renderer,
    _check_capture_start, STATE_NAMES, MODEL_HAND_NAMES,
    _check_preview_protocol, canonical_event_sha256, ASYNC_PREVIEW_CONTRACT,
    CAPTURE_CONTRACT, ENCODER_SHA256, capture_tick_index,
    parse_tracking, validate_frame, validate_body_pose)
import pico_mobile_audit as mobile_audit
import pico_xmove_pick_quality as xmove
import pico_bend_pick_quality as bend
import pico_xmove_bend_pick_quality as xmove_bend

CONTRACT = 'cvpr-live-pico-pick-independent-admission-v1'
BATCH_CONTRACT = 'cvpr-pico-pick-deferred-rtx-batch-v1'
FROZEN_CONTRACT = 'cvpr-pico-pick-rtx-frozen-inputs-v1'
FROZEN_CONTRACTS = frozenset((FROZEN_CONTRACT,))
COMPLETION_CONTRACT = 'cvpr-pico-pick-deferred-rtx-completion-v1'
CONTROL_CONTRACT = 'cvpr-pico-mobile-continuous-vr-mode1-async-reference-v1'
AUDITOR_FILES = frozenset(('pico_pick_audit.py', 'audit_pico_teleop_episode.py',
    'pico_live_input.py', 'pico_sonic_reference.py', 'pico_mobile_audit.py',
    'pico_mobile_quality.py', 'pico_mobile_vr_reference.py',
    'handover_sonic_teleop_reference.py', 'handover_contact_quality.py',
    'pico_xmove_pick_quality.py', 'pico_bend_pick_quality.py',
    'pico_xmove_bend_pick_quality.py', 'pico_locomotion_reference.py',
    'pico_deferred_capture.py', 'replay_handover_teacher.py'))
BASE_CHECKS = frozenset(('live_source_and_rates', 'complete_recording_and_writer',
    'actual_fixed_sonic_artifacts', 'raw_fullbody_and_actual_command_ledger',
    'dense_actual_rgb_state_actions', 'independent_native_pick_task_success',
    'independent_pick_reference', 'independent_pick_supervised_success_coverage',
    'all_original200hz_external_forces_zero'))
PROFILES = {
    xmove.TASK_ID: ('xmove_pick', xmove, 'g1_wholebody_xmove_pick_teleop.py',
        xmove.validate_xmove_pick_task_identity, xmove.evaluate_xmove_pick_capture),
    bend.TASK_ID: ('bend_pick', bend, 'g1_wholebody_bend_pick_mp.py',
        bend.validate_bend_pick_task_identity, bend.evaluate_bend_pick_capture),
    xmove_bend.TASK_ID: ('xmove_bend_pick', xmove_bend, 'g1_wholebody_xmove_bend_pick_teleop.py',
        xmove_bend.validate_xmove_bend_pick_task_identity, xmove_bend.evaluate_xmove_bend_pick_capture),
}


def task_profile(metadata):
    require(isinstance(metadata, dict) and metadata.get('task_id') in PROFILES,
            'Require an explicit supported pick task; unknown tasks never fall back')
    name, module, source_name, validator, evaluator = PROFILES[metadata['task_id']]
    return {'profile': name, 'task_id': metadata['task_id'], 'quality': module,
        'source_name': source_name, 'validator': validator, 'evaluator': evaluator}


def validate_metadata(metadata, bindings, historical_source_manifest):
    spec = task_profile(metadata)
    result = spec['validator'](metadata)
    require(result['passed'], 'Native pick identity: ' + str(result['errors']))
    require(metadata.get('operator_test_only', False) is False,
            'Operator diagnostics are not newly accepted human demonstrations')
    source = Path(__file__).resolve().parents[2] / 'VLMExpert/third_party/SIMPLE/src/simple/tasks' / spec['source_name']
    require(source.is_file() and sha(source) == spec['quality'].OFFICIAL_SOURCE_SHA256,
            'Original pinned native reward/check_success source changed')
    bindings[str(source)] = sha(source)
    historical = mobile_audit.check_historical_sources(metadata, historical_source_manifest, bindings)
    adapter_name = 'pico_' + spec['profile'] + '_task.py'
    records = [entry for entry in historical['files'] if entry['filename'] == adapter_name]
    require(len(records) == 1 and records[0]['sha256'] == result['adapter_sha256'],
            'Recorded task adapter is not its exact retained historical source')
    # Do not replace a historical capture adapter with the current checkout.
    # Its immutable original snapshot is the producer authority for raw data.
    environment = metadata.get('environment_config', {})
    marker = 'pico_bend_pick_frame_adaptation' if spec['profile'] == 'bend_pick' else 'pico_' + spec['profile'] + '_adaptation'
    receipt = environment.get(marker, {})
    keys = receipt.get('adapted_payload_scope')
    require(keys == ['uid', 'metadata', 'robot_cfg', 'sensor_cfgs', 'dr_cfgs', 'dr_state_dict'],
            'Native reset payload binding scope differs')
    payload = {key: environment[key] for key in keys}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    require(receipt.get('adapted_payload_sha256') == digest,
            'Actual recorded environment reset payload differs from its original adapter receipt')
    if spec['profile'] == 'xmove_pick':
        require(metadata.get('mobile_control_protocol') == CONTROL_CONTRACT and
                metadata.get('mobile_control_policy') == 'continuous_vr_mode1' and
                type(metadata.get('mobile_encoder_mode')) is int and metadata['mobile_encoder_mode'] == 1 and
                metadata.get('mobile_planner_model_sha256') == mobile_audit.PLANNER_MODEL_SHA256,
                'XMovePick continuous VR mode1/planner identity differs')
        identity = metadata.get('mobile_vr_source', {})
        require(identity.get('official_sources_rehashed') is True, 'Original VR source freeze missing')
        mobile_audit._bind_pinned_sources(identity.get('source_sha256'),
            mobile_audit.vr.verify_official_sources()['source_sha256'], bindings)
    elif spec['profile'] == 'xmove_bend_pick':
        require(metadata.get('mobile_control_protocol') == 'cvpr-pico-mobile-frozen-upperbody-async-reference-v2' and
                metadata.get('mobile_planner_model_sha256') == mobile_audit.PLANNER_MODEL_SHA256,
                'XMoveBendPick joystick planner identity differs')
    return {**result, 'historical_sources': historical,
        'original_adapter_source': records[0], 'current_adapter_substituted_for_capture': False}


def native_gate(qualities, start_time, metadata, events=None):
    spec = task_profile(metadata)
    for index, row in enumerate(qualities):
        require(type(row.get('completed_steps')) is int and row['completed_steps'] == index + 1 and
                type(row.get('sim_time_s')) in (int, float) and math.isfinite(row['sim_time_s']) and
                close(row['sim_time_s'], start_time + (index + 1) / 50),
                'Original native50Hz evidence clock has gaps')
        reward = row.get('native_task_reward')
        xyz = vector(row.get('actual_target_xyz_m'), 3, 'solved native target')
        baseline = row.get('native_baseline_target_z_m')
        require(type(baseline) in (float, int) and math.isfinite(baseline) and
                type(reward) in (float, int) and math.isfinite(reward), 'Missing native solved reward measurements')
        require(type(row.get('official_task_success')) is bool and
                row['official_task_success'] == (reward >= .9),
                'Original native reward/check_success boolean differs')
    matched = 0
    for event in events or ():
        tick = event['physical_tick50']
        if tick == 0:
            continue
        # Native reward is evaluated after observation refresh. A solved row's
        # xpos is from the earlier post-mj_step stage and must not substitute
        # for that measurement. The next tick's exact archived observation
        # is the same refreshed state on the independently checked 30Hz grid.
        row = qualities[tick - 1]
        target_z = pose(event['objects']['target']['pose7'], 'refreshed reward target')[2]
        expected = float(np.clip((target_z - row['native_baseline_target_z_m']) / .1, 0., 1.))
        require(close(event['sim_time_s'], row['sim_time_s']) and close(row['native_task_reward'], expected, 1e-8),
                'Native reward differs from its exact refreshed archived target state')
        matched += 1
    result = spec['evaluator'](qualities)
    result['excluded_unsupervised_warmup_ticks50'] = 0
    result['native_reward_recomputed_refreshed_archive_frames'] = matched
    result['solved_before_refresh_pose_not_substituted_for_native_reward_target'] = True
    return result


def supervised_coverage(qualities, warmup_ticks):
    require(type(warmup_ticks) is int and warmup_ticks >= 50 and len(qualities) > warmup_ticks,
            'Missing supervised pick suffix or original warmup')
    result = {'warmup_ticks50': warmup_ticks,
        'warmup_native_success_observed': any(row.get('official_task_success') is True for row in qualities[:warmup_ticks]),
        'supervised_native_success_observed': any(row.get('official_task_success') is True for row in qualities[warmup_ticks:]),
        'warmup_first_target_xyz_m': qualities[0]['actual_target_xyz_m'],
        'first_supervised_target_xyz_m': qualities[warmup_ticks]['actual_target_xyz_m'],
        'warmup_actions_excluded_from_supervision': True,
        'no_added_contact_lift_hold_or_final_state_gate': True,
        'scope': 'Native task success remains any original flag; training admission requires a successful supervised suffix. Prefix visual history carries no action labels.'}
    require(result['supervised_native_success_observed'] is True,
            'Require native success in the supervised suffix; warmup success alone is insufficient')
    return result


def _check_planner(frame, evidence, bindings):
    source = evidence['source']
    require(source.get('source_sequence') == frame.sequence and
            source.get('device_timestamp_ns') == frame.device_timestamp_ns and
            source.get('source_packet_reused_as_new_sample') is False and
            type(source.get('source_age_s')) in (int, float) and 0 <= source['source_age_s'] <= .1,
            'Planner reference used another or stale actual source packet')
    axes = [frame.inputs[name] for name in ('left_axis_x', 'left_axis_y', 'right_axis_x', 'right_axis_y')]
    mobile_audit._near(source['axes_left_x_y_right_x_y'], axes, 'same-packet joystick axes', 0.)
    _check_joystick_command(frame, evidence)
    inference = evidence['last_inference']
    accepted = mobile_audit._integer(inference['accepted_tick50'], 'accepted forecast tick')
    tick = mobile_audit._integer(evidence['tick50'], 'planner tick')
    require(mobile_audit._integer(evidence['timeline_origin_tick50'], 'planner origin') +
            mobile_audit._integer(evidence['cursor50'], 'planner cursor') == tick and accepted <= tick and
            mobile_audit._integer(inference['generated_tick50'], 'generated forecast tick') <= accepted and
            mobile_audit._integer(inference['submitted_tick50'], 'submitted forecast tick') <= accepted,
            'Planner forecast chronology differs')
    planner = inference['evidence']
    require(planner.get('contract') == 'cvpr-official-sonic11-locomotion-planner-v1' and
            planner.get('official_model_sha256') == mobile_audit.PLANNER_MODEL_SHA256 and
            planner.get('all_generated_futures_are_forecast_references_not_labels') is True and
            planner.get('physics_steps') == 0 and planner.get('GPU_used') is False and
            planner.get('reference_only') is True and planner.get('async_cpu_worker', {}).get('cpu_only') is True,
            'Pinned asynchronous CPU planner producer differs')
    resampling = planner['resampling']
    require(resampling.get('source_rate_hz') == 30 and resampling.get('reference_rate_hz') == 50 and
            resampling.get('endpoint_holds_eligible_as_complete_reference_future') is False and
            resampling.get('future_action_or_observation_labels_claimed') is False and
            type(resampling.get('fully_interpolated_valid_forecast_frames50')) is int and
            resampling['fully_interpolated_valid_forecast_frames50'] >= 46,
            'Planner lacks a complete non-clamped forecast')
    for key in ('valid_model_output_sha256', 'full_padded_model_output_sha256'):
        mobile_audit._digest(planner.get(key), key)
    # Planner Python sources can be historical; every byte must still exist
    # and be SHA-bound. Pinned official sources retain their declared digests.
    for name, digest in planner.get('source_sha256', {}).items():
        path = Path(name)
        require(path.is_absolute() and path.is_file() and mobile_audit._trace_sha(path, mobile_audit._stat(path)) == digest,
                'Original planner source changed: ' + name)
        bindings[str(path)] = digest
    for name in ('desired_command', 'applied_plan_command'):
        command = evidence[name]
        require(type(command.get('mode')) is int and command['mode'] in (0, 1), 'Unsupported joystick navigation mode')
        direction, facing = vector(command['movement_direction'], 3, 'movement'), vector(command['facing_direction'], 3, 'facing')
        require(close(direction[2], 0.) and close(facing[2], 0.) and close(np.linalg.norm(facing), 1.), 'Non-planar navigation command')
        if command['mode'] == 0:
            require(close(np.linalg.norm(direction), 0.) and command['speed'] == -1., 'Idle navigation command differs')
        else:
            require(close(np.linalg.norm(direction), 1.) and .1 <= command['speed'] <= .4,
                    'Joystick walking envelope differs')


def _check_joystick_command(frame, evidence):
    axes = [frame.inputs[name] for name in ('left_axis_x', 'left_axis_y', 'right_axis_x', 'right_axis_y')]
    lx, ly, rx, ry = axes
    source = evidence['source']
    require(source.get('axis_policy') == 'production_forward_only_with_explicit_lateral_backward_mask' and
            source.get('applied_axes_left_x_y_right_x_y') == [0., max(0., ly), rx, 0.],
            'Joystick production axis mask differs from original same-packet inputs')
    desired = evidence['desired_command']
    magnitude = min(max(0., float(ly)), 1.)
    if magnitude < .15:
        require(desired.get('mode') == 0 and desired.get('movement_direction') == [0., 0., 0.] and desired.get('speed') == -1.,
                'Released/backward/lateral joystick intent must remain an explicit idle command')
    else:
        expected_speed = min(.3, .1 + .5 * (magnitude - .15) / .85)
        require(desired.get('mode') == 1 and close(desired.get('speed'), expected_speed, 1e-12),
                'Navigation speed does not match original forward joystick amount')
        mobile_audit._near(desired['movement_direction'], desired['facing_direction'], 'joystick forward-facing movement', 1e-12)
    require(desired.get('height') == -1., 'Joystick navigation unexpectedly commands a body height')


def check_reference(frame, event, evidence, metadata, bindings, *, helper=None, checked_identity=None):
    spec = task_profile(metadata)
    state = helper if isinstance(helper, dict) else {}
    mode = evidence.get('encoder_mode')
    require(type(mode) is int, 'Executed encoder mode must be an actual integer')
    if spec['profile'] == 'xmove_pick':
        state['vr'] = mobile_audit.check_reference(frame, event, evidence, metadata, bindings, helper=state.get('vr'))
        _check_joystick_command(frame, evidence)
        return state
    if mode == 2:
        import pico_sonic_reference as smpl
        require(evidence.get('contract') == smpl.CONTRACT and evidence.get('PICO_raw_va_wva_used') is False and
                evidence.get('PICO_joint_t_used') is False and evidence.get('absolute_human_root_translation_encoded') is False and
                evidence.get('device_timestamp_ns') == frame.device_timestamp_ns and evidence.get('identities') == checked_identity,
                'Executed full-body mode2 provenance differs')
        horizon = evidence['reference_horizon']
        require(horizon.get('kind') == 'causal_constant_desired_pose_hold' and horizon.get('measured_future_frames') == 0 and
                horizon.get('not_recorded_future') is True, 'Held desired SMPL pose misrepresented as future observation')
        calibration = evidence['calibration']
        signature = json.dumps(calibration, sort_keys=True, separators=(',', ':'))
        if state.get('smpl_signature') != signature:
            humans = [Path(name) for name in checked_identity['source_sha256'] if Path(name).name == 'human_joints_info.pkl']
            require(len(humans) == 1, 'Missing unique pinned human skeleton')
            obj = smpl.PicoSonicReference.__new__(smpl.PicoSonicReference)
            obj.identities = checked_identity
            obj.rest_joints, obj.fk_parents = smpl.load_human_info(humans[0])
            obj.calibration = calibration
            obj.apply_delta_heading = vector(calibration['apply_delta_heading_wxyz'], 4, 'mode2 heading')
            require(close(np.linalg.norm(obj.apply_delta_heading), 1., 1e-6), 'Nonunit mode2 calibrated heading')
            obj.wrist_offset = vector(calibration['wrist_offset6'], 6, 'mode2 calibrated wrists')
            obj.last_body = obj.last_timestamp_ns = None
            state.update(smpl=obj, smpl_signature=signature)
        _, recomputed = state['smpl'].prepare(frame.body24x7, event['root_pose7'], frame.device_timestamp_ns)
        for key in ('reference_input_sha256', 'wrist_targets6', 'reference_root_wxyz', 'input_shape', 'input_dtype', 'layout'):
            require(shared._strict_json_equal(evidence.get(key), recomputed[key]), 'Recomputed full-body SMPL input differs: ' + key)
        return state
    require(spec['profile'] == 'xmove_bend_pick' and mode == 0, 'This task cannot use the recorded encoder branch')
    if evidence.get('producer') == 'measured_mobile_entry_pose_static_preparation_hold':
        require(evidence.get('static_desired_reference') is True and evidence.get('complete46_future') is False and
                evidence.get('planner_inference_performed') is False and evidence.get('current_human_pose_used') is False,
                'Navigation preparation is not an explicit measured static hold')
        vector(evidence['held_body_q29'], 29, 'preparation actual entry pose')
        require(type(evidence.get('pose_source_sequence')) is int and evidence['pose_source_sequence'] <= frame.sequence and
                type(evidence.get('pose_source_device_timestamp_ns')) is int and evidence['pose_source_device_timestamp_ns'] <= frame.device_timestamp_ns,
                'Static preparation pose source comes from the future')
    else:
        require(evidence.get('contract') == 'cvpr-pico-mobile-frozen-upperbody-async-reference-v2' and
                evidence.get('producer') == 'official_full_body_planner_mode0' and
                evidence.get('reference_policy') == 'locked_upper_mode0' and evidence.get('upper_body_frozen') is True and
                evidence.get('complete46_future') is True and evidence.get('actual_state_written') is False and
                evidence.get('synchronous_diagnostic_only') is False,
                'Joystick mode0 producer/frozen-upper reference differs')
        vector(evidence['frozen_upper_body_q17'], 17, 'frozen upper desired pose')
        require(evidence.get('frozen_upper_body_names') == list(STATE_NAMES[12:29]) and
                evidence.get('frozen_upper_body_velocity17') == [0.] * 17,
                'Mode0 frozen upper reference semantics differ')
        _check_planner(frame, evidence, bindings)
    require(evidence.get('actual_heading_removed') is True and evidence.get('model_forecasts_are_training_labels') is False,
            'Mode0 desired forecast/reference semantics differ')
    return state


def _bound_json(name, label, bindings):
    require(isinstance(name, str) and Path(name).is_absolute(), 'Missing absolute ' + label + ' path')
    path = Path(name)
    require(path.is_file() and not path.is_symlink(), 'Missing/unsafe ' + label)
    before = mobile_audit._stat(path)
    encoded = path.read_bytes()
    value = json.loads(encoded, parse_constant=lambda word: (_ for _ in ()).throw(ValueError('Nonfinite ' + label)))
    require(mobile_audit._stat(path) == before, label + ' changed during admission')
    digest = hashlib.sha256(encoded).hexdigest()
    require(str(path) not in bindings or bindings[str(path)] == digest, label + ' changed after an earlier binding')
    bindings[str(path)] = digest
    return path, value, digest


def _check_batch_execution(root, completion, metadata, bindings):
    execution = completion.get('batch_execution')
    require(isinstance(execution, dict), 'Pick producer requires its frozen batch execution envelope')
    script = Path(__file__).with_name('render_deferred_pick_batch.py').resolve(strict=True)
    original = Path(__file__).with_name('render_deferred_pico.py').resolve(strict=True)
    require(execution.get('contract') == BATCH_CONTRACT and execution.get('batch_code_sha256') == sha(script) and
            execution.get('batch_code_path') == str(script) and
            execution.get('original_renderer_code_path') == str(original) and
            execution.get('original_renderer_code_sha256') == sha(original) and execution.get('real_environment_count') == 1 and
            execution.get('original_reset_and_environment_binding_per_episode') is True and
            type(execution.get('environment_lease_number')) is int and execution['environment_lease_number'] >= 1,
            'Pick batch producer/environment lease differs')
    report_path, report, report_sha = _bound_json(execution.get('batch_report_path'), 'pick batch report', bindings)
    require(report.get('contract') == BATCH_CONTRACT and report.get('accepted_new_dataset') is False and
            report.get('batch_code_path') == str(script) and report.get('original_renderer_code_path') == str(original) and
            report.get('batch_code_sha256') == sha(script) and report.get('original_renderer_code_sha256') == sha(original),
            'Pick batch report names another producer')
    exit_binding = {}
    if report.get('status') == 'render_completed_pending_process_exit':
        exit_path, observed, exit_sha = _bound_json(str(report_path.with_name(report_path.stem + '_EXIT.json')), 'pick process exit', bindings)
        require(observed.get('contract') == 'cvpr-pick-rtx-observed-process-exit-v1' and
                observed.get('report_path') == str(report_path) and observed.get('report_sha256') == report_sha and
                type(observed.get('exit_code')) is int and observed['exit_code'] == 0 and
                type(observed.get('child_pid')) is int and observed['child_pid'] > 0 and
                isinstance(observed.get('argv'), list) and observed['argv'] and all(isinstance(v, str) for v in observed['argv']),
                'Independent process exit does not bind this exact completed render report')
        driver = Path(observed.get('parentdriver_path', ''))
        require(driver.is_absolute() and driver.is_file() and not driver.is_symlink() and
                sha(driver) == observed.get('parentdriver_sha256'), 'Pick exit driver source changed')
        bindings[str(driver)] = sha(driver)
        require(datetime.fromisoformat(observed['finished_beijing']) >= datetime.fromisoformat(report['render_finished_utc']),
                'Process exit predates rendering')
        require(report.get('cleanup_requested') is True and report.get('cleanup_return_verified') is False,
                'Pre-exit cleanup boundary differs')
        exit_binding = {'process_exit_path': str(exit_path), 'process_exit_sha256': exit_sha,
            'parentdriver_path': str(driver), 'parentdriver_sha256': sha(driver), 'cleanup_return_verified': False}
    else:
        require(report.get('status') == 'completed_pending_independent_audit' and
                report.get('cleanup_requested') is True and report.get('cleanup_return_verified') is True,
                'Pick batch is incomplete, rejected, or cleanup did not return')
    manifest_path, manifest, manifest_sha = _bound_json(report.get('manifest_path'), 'frozen pick manifest', bindings)
    require(manifest.get('contract') in FROZEN_CONTRACTS and
            execution.get('frozen_manifest_path') == str(manifest_path) and
            report.get('manifest_sha256') == execution.get('frozen_manifest_sha256') == manifest_sha,
            'Pick frozen manifest identity differs')
    frozen = manifest.get('episodes', [])
    require(type(manifest.get('count')) is int and manifest['count'] == len(frozen) and
            1 <= len(frozen) <= 300 and len({row['uuid'] for row in frozen}) == len(frozen),
            'Pick processing cohort UUID coverage differs')
    expected_counts = manifest.get('expected_task_counts')
    actual_counts = {PROFILES[task_id][0]: sum(row.get('task_id') == task_id for row in frozen) for task_id in PROFILES}
    require(expected_counts == actual_counts and sum(actual_counts.values()) == len(frozen) and
            all(type(value) is int and 0 <= value <= 100 for value in expected_counts.values()),
            'Explicit per-task processing counts differ')
    candidates_frozen = frozen
    if len(frozen) < 300:
        _, candidates_manifest, candidates_sha = _bound_json(manifest.get('candidate_manifest_path'), 'original candidate300 manifest', bindings)
        candidates_frozen = candidates_manifest.get('episodes', [])
        require(manifest.get('candidate_count') == 300 and manifest.get('candidate_manifest_sha256') == candidates_sha and
                candidates_manifest.get('contract') in FROZEN_CONTRACTS and candidates_manifest.get('count') == len(candidates_frozen) == 300 and
                len({row['uuid'] for row in candidates_frozen}) == 300, 'Original300 candidate cohort identity differs')
        exclusions = manifest.get('excluded_episodes')
        require(isinstance(exclusions, list) and len(exclusions) == 300 - len(frozen) and
                len({row['uuid'] for row in exclusions}) == len(exclusions) and
                all(row.get('excluded_from_processing') is True and isinstance(row.get('reason'), str) and row['reason'] for row in exclusions),
                'Processing exclusions must be explicit and distinct')
        originals = {row['uuid']: row for row in candidates_frozen}
        require(set(originals) == {row['uuid'] for row in frozen} | {row['uuid'] for row in exclusions} and
                not ({row['uuid'] for row in frozen} & {row['uuid'] for row in exclusions}) and
                all(row == originals.get(row['uuid']) for row in frozen), 'Processing cohort is not the exact candidate-minus-exclusion subset')
        for row in exclusions:
            original_row = originals[row['uuid']]
            require(all(row.get(key) == original_row.get(key) for key in ('task_id', 'directory', 'number')),
                    'Excluded candidate identity was changed')
    for task_id in PROFILES:
        task_rows = [row for row in candidates_frozen if row.get('task_id') == task_id]
        require(len(task_rows) == 100 and {row.get('cohort_slot') for row in task_rows} == set(range(100)),
                'Original candidate task slots differ; supplementary XMovePick must be explicit')
    spec = task_profile(metadata)
    subset = [row for row in frozen if row.get('profile') == spec['profile']]
    offset, selected = report.get('offset'), report.get('selected_count')
    require(report.get('profile') == spec['profile'] and report.get('task_id') == metadata['task_id'] and
            report.get('frozen_count') == len(frozen) and type(offset) is int and offset >= 0 and
            type(selected) is int and selected > 0 and offset + selected <= len(subset) and
            report.get('selected_uuids') == [row['uuid'] for row in subset[offset:offset + selected]] and
            [row.get('uuid') for row in report.get('episodes', [])] == report['selected_uuids'],
            'Completed pick batch does not cover its explicit profile/offset/limit selection')
    candidates = [row for row in frozen if row.get('uuid') == metadata['human_demo_uuid']]
    require(len(candidates) == 1 and candidates[0].get('task_id') == metadata['task_id'] and
            Path(candidates[0]['directory']).resolve(strict=True) == root.resolve(strict=True),
            'Current episode task/UUID/directory is not the frozen cohort')
    required_sources = {'metadata.json', 'episode.json', 'commands.jsonl', 'raw_quality_samples.jsonl',
        'applied_force_samples.jsonl', 'deferred_capture/capture.json', 'deferred_capture/events.jsonl',
        'deferred_capture/model.mjb', 'deferred_capture/states.npy'}
    require(required_sources <= set(candidates[0].get('source_files', {})),
            'Frozen episode inventory omitted original physical or command evidence')
    for name, expected in candidates[0]['source_files'].items():
        path = _owned(root, name)
        stat = path.stat()
        require(stat.st_size == expected['size_bytes'] and stat.st_mtime_ns == expected['mtime_ns'] and
                sha(path) == expected['sha256'], 'Frozen pick input changed: ' + name)
        require(str(path) not in bindings or bindings[str(path)] == expected['sha256'], 'Pick input changed during admission')
        bindings[str(path)] = expected['sha256']
    completed = [row for row in report.get('episodes', []) if row.get('uuid') == metadata['human_demo_uuid']]
    require(len(completed) == 1 and completed[0].get('task_id') == metadata['task_id'] and
            completed[0].get('number') == candidates[0]['number'] and
            type(completed[0].get('reused_completed_output')) is bool and
            completed[0].get('rtx_completion_sha256') == sha(root / 'rtx_completion.json') and
            completed[0].get('completed_frames') == completion['completed_frames'] and
            completed[0].get('environment_binding') == completion['environment_binding'] and
            Path(completed[0]['directory']).resolve(strict=True) == root.resolve(strict=True) and
            type(report.get('new_environment_leases')) is int and execution['environment_lease_number'] <= report['new_environment_leases'],
            'Completed pick batch does not bind this episode receipt')
    bindings[str(script)] = sha(script)
    bindings[str(original)] = sha(original)
    return {'contract': BATCH_CONTRACT, 'uuid': metadata['human_demo_uuid'], 'task_id': metadata['task_id'],
        'batch_code_path': str(script), 'batch_code_sha256': sha(script),
        'original_renderer_code_path': str(original), 'original_renderer_code_sha256': sha(original),
        'report_path': str(report_path), 'report_sha256': report_sha,
        'frozen_manifest_path': str(manifest_path), 'frozen_manifest_sha256': manifest_sha,
        'environment_lease_number': execution['environment_lease_number'],
        'completed_report_binds_this_receipt': True, **exit_binding}


def _check_deferred_bridge(root, metadata, outcome, manifest, events, ledger, read, bindings):
    """Bind zero-step RTX to original complete physical archives independently."""
    from pico_deferred_capture import PENDING, load_deferred, evidence_profile
    DEFERRED_CONTRACT = evidence_profile(metadata)[0]
    archived = load_deferred(root/"deferred_capture", verify=True)
    raw, source_events = archived["manifest"], archived["events"]
    require(raw["contract"] == DEFERRED_CONTRACT and raw["status"] == PENDING
            and raw.get("failure") is None and raw.get("requested_completed") is True
            and raw["metadata"] == metadata and outcome.get("capture") == raw
            and outcome.get("status") == "pending_rtx" and outcome.get("rtx_ready") is False,
            "Original physical capture was incomplete, changed, or not pending RTX")
    require(raw["metadata"].get("source_kind") == metadata.get("source_kind")
            and raw.get("online_operator_preview_engine") == "MuJoCo EGL"
            and raw.get("training_rgb_acquired") is False
            and raw["warmup_ticks"] >= 50 and raw["warmup_ticks"] == round(manifest["warmup_seconds"]*50),
            "Original physical preview/warmup provenance is invalid")
    completion = read("rtx_completion.json")
    require(completion.get("contract") == COMPLETION_CONTRACT
            and completion.get("status") == "deferred_rtx_completed_pending_independent_audit"
            and completion.get("accepted_new_dataset") is False
            and completion.get("eligible_for_packing") is False
            and completion.get("source_kind") == metadata.get("source_kind")
            and completion.get("archived_physics_steps") == 0 and completion.get("sonic_inferences") == 0
            and completion.get("state_event_numeric_checks_passed") is True
            and completion.get("full_integration_state_unchanged") is True
            and completion.get("environment_binding", {}).get("matched") is True,
            "Deferred RTX completion cannot certify zero-step same-scene state acquisition")
    domains = {"observation_domain_feedback":"MuJoCo EGL preview", "training_rgb_domain":"Isaac RTX deferred",
               "rgb_was_live_operator_feedback":False}
    for key, expected in domains.items():
        require(completion.get(key) == expected and manifest.get(key) == expected,
                "Deferred training RGB was misrepresented as live operator feedback")
    require(metadata.get("preview_engine") == "mujoco_egl"
            and metadata.get("preview_is_training_rgb") is False
            and metadata.get("training_rgb_engine") == "pending_offline_isaac_rtx",
            "Original MuJoCo preview was incorrectly labelled as RTX training RGB")
    require(completion.get("capture") == manifest
            and completion.get("capture_manifest_sha256") == bindings[str(root/"capture/capture.json")]
            and manifest.get("rgb_capture_mode") == "deferred_from_exact_recorded_integration_state"
            and completion.get("completed_frames") == len(events) == len(source_events)
            and completion.get("frame_bindings_count") == len(events), "Deferred RTX capture/frame counts or final manifest SHA changed")
    for kind, filename in (("manifest","capture.json"), ("events","events.jsonl"),
                           ("model","model.mjb"), ("states","states.npy")):
        path = _owned(root, "deferred_capture/"+filename)
        actual = sha(path); bindings[str(path)] = actual
        require(completion.get("source_"+kind+"_sha256") == actual, "Deferred raw "+kind+" SHA changed")
    require(manifest.get("deferred_source_manifest_sha256") == completion["source_manifest_sha256"],
            "Final RGB capture refers to another physical archive")
    renderer = Path(__file__).with_name("render_deferred_pick_batch.py").resolve(strict=True)
    require(completion.get("render_code_path") == str(renderer) and
            completion.get("render_code_sha256") == sha(renderer), "Actual pick RTX producer source changed")
    bindings[str(renderer)] = completion["render_code_sha256"]
    producers = completion.get("producer_source_sha256")
    require(isinstance(producers, dict) and producers, "Pick producer source inventory missing")
    profile = task_profile(metadata)['profile']
    required_producers = {'render_deferred_pico.py', 'pico_' + profile + '_task.py',
        'pico_' + profile + '_quality.py', 'pico_deferred_capture.py',
        'pico_simple_teleop.py', 'replay_handover_teacher.py'}
    require(len(producers) == len(required_producers) and {Path(name).name for name in producers} == required_producers,
            "Pick producer dependency inventory is incomplete or ambiguous")
    require(completion.get('task_id') == metadata['task_id'], "RTX completion task differs from raw capture")
    for name, expected in producers.items():
        path = Path(name)
        require(path.is_absolute() and path.is_file() and not path.is_symlink() and sha(path) == expected,
                "Pick renderer producer source changed: " + name)
        bindings[str(path)] = expected
    profile = task_profile(metadata)['profile']
    marker = 'pico_bend_pick_frame_adaptation' if profile == 'bend_pick' else 'pico_' + profile + '_adaptation'
    require(completion.get('task_adapter_identity') == metadata.get(profile + '_adapter_identity') and
            completion.get('task_adaptation') == metadata['environment_config'][marker] and
            completion.get('evidence_object_roles') == ['target'],
            "RTX task adapter, original scene adaptation or target roles differ")
    require(completion.get("frame_bindings_path") == "rtx_frame_bindings.jsonl", "Unexpected frame bridge path")
    bridge = read("rtx_frame_bindings.jsonl", lines=True)
    require(completion.get("frame_bindings_sha256") == bindings[str(root/"rtx_frame_bindings.jsonl")]
            and len(bridge) == len(events), "Frame bridge SHA or count changed")
    shared = ("event_index","capture_index30","physical_tick50","sim_time_s","nominal_elapsed_s",
        "actual_elapsed_s","phase","action_valid","source_command_index","joint_names","state43",
        "physical_joint_q43_float64","physical_joint_qvel43","root_pose7","root_qvel6","objects","action")
    require(len(ledger) == len(source_events)+1, "Deferred source has no corresponding live command ledger")
    asynchronous_preview = _check_preview_protocol(metadata)
    for index, (source, event, row) in enumerate(zip(source_events, events, bridge)):
        require(all(source[key] == event[key] for key in shared), "RTX changed original physical/state/action field at frame "+str(index))
        if asynchronous_preview:
            require("preview" in source and source["preview"] is None,
                    "Asynchronous preview must not attach an old/current image to a recorded training state")
        else:
            preview = source.get("preview")
            require(isinstance(preview, dict) and preview.get("training_observation") is False
                    and preview.get("engine") == "MuJoCo EGL" and preview.get("saved_as_training_rgb") is False,
                    "Original preview must be recorded explicitly as non-training feedback")
        fields, command = source["source_fields"], ledger[index+1]
        for key in ("source_raw_packet","source_host_received_s","source_sequence","device_timestamp_ns","reference_evidence"):
            require(fields[key] == command[key], "Deferred original input differs from actual command ledger: "+key)
        state = np.ascontiguousarray(archived["states"][index], dtype="<f8")
        state_hash = hashlib.sha256(state.tobytes()).hexdigest()
        require(row.get("event_index") == index and row.get("physical_tick50") == source["physical_tick50"]
                and row.get("sim_time_s") == source["sim_time_s"]
                and row.get("full_integration_state_unchanged") is True
                and row.get("integration_state_sha256") == source["integration_state_sha256"] == state_hash
                and row.get("source_event_sha256") == canonical_event_sha256(source)
                and row.get("rtx_event_sha256") == canonical_event_sha256(event)
                and row.get("rgb_pixel_sha256") == event["rgb"]["pixel_sha256"]
                and row.get("rgb_file_sha256") == event["rgb"]["file_sha256"],
                "Full-state/source/RTX frame bridge differs at frame "+str(index))
    batch_execution = _check_batch_execution(root, completion, metadata, bindings)
    first_state = np.ascontiguousarray(archived["states"][0], dtype="<f8")
    return {"raw_completed":True,"cpu_full_integration_state_verified":True,
            "batch_execution":batch_execution,
            "first_integration_state_sha256":hashlib.sha256(first_state.tobytes()).hexdigest(),
            "first_integration_state_width":int(first_state.size),
            "first_sim_time_s":source_events[0]["sim_time_s"],
            "first_physical_tick50":source_events[0]["physical_tick50"],
            "preview_protocol":ASYNC_PREVIEW_CONTRACT if asynchronous_preview else "legacy_per_event_preview_digest",
            "operator_preview_delivery_verified":False,
            "operator_saw_every_recorded_state_verified":False,
            "preview_scope":"Preview metadata only; per-command feedback/delivery and operator viewing are not certified. Every archived state remains bound to its independently checked RTX frame.",
            "source_kind":metadata.get("source_kind"),
            "observation_domain_feedback":"MuJoCo EGL preview","training_rgb_domain":"Isaac RTX deferred",
            "rgb_was_live_operator_feedback":False,"frames_bound":len(events),
            "dynamics_steps_performed":0,"sonic_inferences_performed":0}



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
    stat = mobile_audit._stat(trace)
    digest = mobile_audit._trace_sha(trace, stat)
    bindings[str(trace)] = digest
    start_tick = mobile_audit._integer(outcome['state']['start_tick50'], 'capture start global tick')
    ticks = len(qualities)
    commands = {row['physical_tick50']: row for row in ledger[1:]}
    captures = {event['physical_tick50']: event for event in events}
    first_command = ledger[1]
    first_time = events[0]['sim_time_s']
    found, count, previous_after, reference_tick_offset = False, 0, None, None
    max_lag, held_hands = 0., None
    previous_timeline_key = None
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
            if row.get('phase') == 'mobile_mode_transition':
                require(type(row.get('tick50')) is int and row['tick50'] == start_tick+count,
                        'Mode transition changed the physical tick grid')
                if previous_after is not None:
                    for name in ('root_pose7', 'body_q29', 'body_dq29', 'hand_q14'):
                        mobile_audit._near(row['actual_state'][name], previous_after[name], 'transition actual state '+name, 1e-9)
                continue
            require(row.get('phase') == 'pico_follow' and type(row.get('tick50')) is int and
                    row['tick50'] == start_tick+count, 'Original50Hz trace is interrupted or not consecutive')
            before, after = row['actual_before_step'], row['actual_after_step']
            require(before.get('state_finite') is True and after.get('state_finite') is True and
                    math.isclose(before['sim_time_s'], first_time+count/50, rel_tol=0, abs_tol=1e-6) and
                    math.isclose(after['sim_time_s'], first_time+(count+1)/50, rel_tol=0, abs_tol=1e-6),
                    'Original closed-loop physics state/clock is incomplete or nonfinite')
            for state in (before, after):
                mobile_audit._unit(mobile_audit._array(state['root_pose7'], (7,), 'actual trace root')[3:], (4,), 'actual trace root')
                mobile_audit._array(state['body_q29'], (29,), 'actual29 body joints')
                mobile_audit._array(state['body_dq29'], (29,), 'actual29 joint velocities')
                mobile_audit._array(state['hand_q14'], (14,), 'actual14 finger joints')
                require(state['body_joint_names'] == events[0]['joint_names'][:29] and
                        state['hand_joint_names'] == events[0]['joint_names'][29:],
                        'Actual trace named joint identity differs')
            if previous_after is not None:
                for name in ('root_pose7', 'body_q29', 'body_dq29', 'hand_q14'):
                    mobile_audit._near(before[name], previous_after[name], 'continuous actual50Hz '+name, 1e-9)
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
                if 'tick50' in row['reference']:
                    offset = row['tick50']-row['reference']['tick50']
                    timeline_key = (row['reference'].get('epoch'), row['reference'].get('timeline_origin_tick50'))
                    if timeline_key == previous_timeline_key:
                        require(offset == reference_tick_offset, 'Planner reference timeline shifted within one epoch')
                    previous_timeline_key, reference_tick_offset = timeline_key, offset
                event = captures[count]
                mobile_audit._near(event['root_pose7'], before['root_pose7'], 'actual captured trace root', 1e-12)
                actual = dict(zip(before['body_joint_names'], before['body_q29']))
                actual.update(zip(before['hand_joint_names'], before['hand_q14']))
                mobile_audit._near(event['physical_joint_q43_float64'], [actual[name] for name in event['joint_names']],
                      'actual captured trace joints', 1e-12)
                held_hands = command['named_hand_targets']
            else:
                require(row.get('reference') is None, 'An unlogged token/reference update occurred outside30Hz')
            require(row.get('named_hand_targets') == held_hands,
                    'Actual held50Hz finger command differs from its last30Hz source')
            count += 1
    require(found and count == ticks and mobile_audit._stat(trace) == stat,
            'Original complete50Hz execution is missing or changed during audit')
    return {'path': str(trace), 'sha256': digest, 'first_global_tick50': start_tick,
            'control_ticks50': count, 'actual_decoder_rate_hz': 50, 'actual_command_rate_hz': 30,
            'actual_physics_rate_hz': 200, 'maximum_cumulative_lag_s': max_lag,
            'planner_to_global_tick_offset': reference_tick_offset,
            'all_original_native_flags_match_environment_return': True,
            'actual_state_continuity_verified': True, 'dynamics_steps_performed': 0,
            'scope': 'Original serialized execution/state/deadline consistency; no independent decoder inference or dynamics replay'}


def audit_episode(directory, *, write_report=True, historical_source_manifest=None):
    """Recompute raw eligibility. Invalid/missing evidence yields a rejection."""
    root = Path(directory).resolve(strict=True)
    report = {"contract": CONTRACT, "generated_utc": datetime.now(timezone.utc).isoformat(),
        "episode_directory": str(root), "status": "independent_live_human_episode_audit_rejected",
        "eligible_for_packing": False, "accepted_new_dataset": False,
        "ckpt_feature_pack_ready": False, "checks": {}, "errors": [], "bindings": {},
        "scope": "Independent local raw-recording consistency and explicit native pick task flag and profile-specific reference consistency; requires separate packing, DINO, normalization, split and checkpoint-reader acceptance",
        "limitations": ["File evidence establishes local producer consistency, not authenticated proof of human identity or protection from deliberate forgery.",
            "No simulator integration replay or image re-render is performed; camera sensor/prim/current-state evidence is checked as recorded.",
            "Solved-force quality rows precede observation refresh; their derived poses are not asserted bitwise equal to refreshed RGB-frame poses.",
            "The SONIC/world-frame adaptation is explicit; no AMO benchmark or robot-physics equivalence is claimed. Native task success adds no contact, lift, hold or final-state criterion."]}
    report['auditor_source_sha256'] = {str(Path(__file__).with_name(name).resolve()):
        sha(Path(__file__).with_name(name)) for name in AUDITOR_FILES}
    bindings = report["bindings"]
    def read(name, lines=False):
        path = _owned(root, name)
        bindings[str(path)] = sha(path)
        if lines:
            rows = [_read_line(line) for line in path.read_text().splitlines() if line.strip()]
            require(all(isinstance(row, dict) for row in rows), "JSONL rows must be objects")
            return rows
        return _read(path)
    def stage(name, function):
        try:
            result = function()
            report["checks"][name] = {"passed": True}
            return result
        except Exception as error:
            message = f"{name}: {type(error).__name__}: {error}"
            report["checks"][name] = {"passed": False, "error": message}
            report["errors"].append(message)
            return None
    try:
        metadata = read("metadata.json")
        spec = task_profile(metadata)
        report.update(task_id=spec['task_id'], profile=spec['profile'], selection_policy='native_only', technical_blocked=False)
        for marker_name in ('operator_discard.json','collection_record.json'):
            marker = root/marker_name
            require(not marker.is_symlink(), 'Discard/collection record must not be a symbolic link')
            if not marker.exists():
                continue
            disposition = read(marker_name)
            require(isinstance(disposition,dict), 'Invalid discard/collection record')
            if marker_name == 'operator_discard.json':
                raise ValueError('Operator discarded this episode: '+str(disposition.get('reason','manual discard')))
            require(disposition.get('disposition') in ('kept','discarded'), 'Invalid collection disposition')
            if disposition['disposition'] == 'discarded':
                raise ValueError('Episode collection rejected this recording: '+str(disposition.get('reasons',[])))
        from pico_episode_review import discard_requested
        require(not discard_requested(root), 'Operator discard is pending in the background queue')
        outcome = read("episode.json")
        if outcome.get("geometry_pending") is True:
            submission = read("review_queue_submission.json")
            require(isinstance(submission,dict), 'Background review submission marker is missing')
            from pico_episode_review import discard_requested
            require(not discard_requested(root), 'Operator discard is pending in the background queue')
            collection = read("collection_record.json")
            require(collection.get("status") == "committed" and collection.get("disposition") == "kept",
                    "Background physical review has not retained this recording")
        manifest = read("capture/capture.json")
        deferred = metadata.get("render_mode") == "mujoco_deferred"
        async_result = None if deferred else read("capture/async_capture.json")
        events = read("capture/events.jsonl", lines=True)
        ledger = read("commands.jsonl", lines=True)
        quality_name = 'raw_quality_samples.jsonl'
        qualities = read(quality_name, lines=True)
        forces = read("applied_force_samples.jsonl", lines=True)
        require(bool(events) and bool(qualities), "Empty recorded episode")
        start_time = float(events[0]["sim_time_s"])
        ticks = len(qualities)
        identity = str(uuid.UUID(metadata["human_demo_uuid"]))
        report.update(human_demo_uuid=identity, event_count=len(events), control_ticks50=ticks)
        expected_checks = 9
        report['native_quality_source'] = {'path':quality_name, 'sha256':bindings[str(root/quality_name)]}
        if deferred:
            report.update(observation_domain_feedback="MuJoCo EGL preview", training_rgb_domain="Isaac RTX deferred",
                          rgb_was_live_operator_feedback=False)

        def source():
            for data in (metadata, outcome, manifest):
                require(data.get("source_kind") == "live_pico_human", "Recorded replay/unknown source is never eligible")
                require(data.get("accepted_new_dataset") is False, "Raw recording must not claim dataset acceptance")
                require(data.get("replay_source_sha256") is None, "Recorded replay provenance is present")
            require(metadata.get("task_id") == spec["task_id"] and metadata.get("render_mode") in ("isaac", "mujoco_deferred")
                    and metadata.get("physics_hz") == 200 and metadata.get("decoder_hz") == 50
                    and metadata.get("command_rgb_hz") == 30 and metadata.get("raw_pico_va_wva_used") is False,
                    "Task/rendering/rate/raw-derivative contract differs")
            initial = metadata["initialization"]
            require(initial.get("object_restores_during_control") is False and initial.get("external_assistance") is False,
                    "External assistance or in-episode state restore is not accepted")
            report['official_pick_identity'] = validate_metadata(metadata, bindings, historical_source_manifest)
        stage("live_source_and_rates", source)

        def completion():
            state = outcome["state"]
            require(state.get("status") == "pending_audit" and state.get("pending_audit") is True
                    and state.get("reasons") == [] and state.get("human_demo_uuid") == identity
                    and state.get("local_tick50") == ticks-1, "Episode was not completely saved pending audit")
            require(manifest.get("contract") == CAPTURE_CONTRACT
                    and manifest.get("status") == "raw_capture_completed_pending_independent_audit"
                    and manifest.get("human_demo_uuid") == identity
                    and manifest.get("events_path") == "events.jsonl"
                    and manifest.get("events_sha256") == bindings[str(root/"capture/events.jsonl")],
                    "Capture is incomplete or its event bytes changed")
            if not deferred:
                details = async_result["async_capture"]
                require(async_result.get("status") == manifest["status"]
                        and details.get("completed") is True and details.get("writer_joined") is True
                        and details.get("failure") is None and details.get("enqueued_events") == len(events)
                        and details.get("written_events") == len(events), "Async writer dropped/failed or did not drain")
                require({key:value for key,value in async_result.items() if key != "async_capture"} == manifest,
                        "Underlying capture and async final manifest disagree")
                require(outcome.get("capture") == async_result, "Episode finalization and writer receipt disagree")
            warmup = manifest.get("warmup_seconds")
            require(isinstance(warmup, (float, int)) and not isinstance(warmup, bool)
                    and warmup >= 1 and close(warmup*50, round(warmup*50))
                    and round(warmup*50) % 5 == 0, "At least one second of real aligned warmup is required")
        stage("complete_recording_and_writer", completion)
        if deferred:
            expected_checks += 1
            report["deferred_rtx_bridge"] = stage("original_full_state_to_deferred_rtx_bridge", lambda:
                _check_deferred_bridge(root, metadata, outcome, manifest, events, ledger, read, bindings))
        if metadata.get("capture_start_protocol") is not None:
            expected_checks += 1
            report["capture_start_boundary"] = stage("independent_frozen_first_capture_boundary", lambda:
                _check_capture_start(metadata, outcome, events, ledger, read, bindings, root,
                                     report.get("deferred_rtx_bridge")))
        checked_identity = stage("actual_fixed_sonic_artifacts", lambda: _bound_identities(metadata, bindings))

        def commands():
            require(len(ledger) == len(events)+1 and ledger[0].get("event_type") == "session", "Dense command ledger is missing")
            require(outcome["ledger"]["sha256"] == bindings[str(root/"commands.jsonl")]
                    and outcome["ledger"]["event_count"] == len(ledger)
                    and outcome["ledger"]["human_demo_uuid"] == identity
                    and ledger[0]["metadata"] == metadata, "Final command ledger SHA/count/identity changed")
            previous = None
            previous_ns = None
            mobile_helper = None
            for index, row in enumerate(ledger):
                require(row.get("event_index") == index and row.get("human_demo_uuid") == identity
                        and row.get("accepted_new_dataset") is False
                        and row.get("raw_pico_va_wva_validated") is False,
                        "Command ledger identity or derivative policy changed")
                recorded_ns = row["recorded_monotonic_ns"]
                require(type(recorded_ns) is int and (previous_ns is None or recorded_ns > previous_ns), "Nonmonotonic command wall clock")
                previous_ns = recorded_ns
                if index == 0:
                    continue
                event = events[index-1]
                require(row.get("event_type") == "command" and row["physical_tick50"] == event["physical_tick50"]
                        and close(row["sim_time_s"], event["sim_time_s"]), "Command and pre-step RGB/state clocks differ")
                frame = parse_tracking(row["source_raw_packet"], row["source_host_received_s"], row["source_sequence"])
                require(frame.device_timestamp_ns == row["device_timestamp_ns"], "Raw/source device time differs")
                quality = validate_frame(frame, now=recorded_ns/1e9, previous=previous)
                require(quality["recordable"], "Raw full-body packet rejected: " + ",".join(quality["reasons"]))
                validate_body_pose(frame.body24x7)
                if previous is not None:
                    require(frame.sequence > previous.sequence, "Command source callback did not advance")
                previous = frame
                token = vector(row["body_token64"], 64, "actual token")
                hands = row["named_hand_targets"]
                require(isinstance(hands, dict) and set(hands) == set(MODEL_HAND_NAMES), "Named hand14 semantics differ")
                vector([hands[name] for name in MODEL_HAND_NAMES], 14, "actual hands")
                evidence = row["reference_evidence"]
                require(evidence.get("model_inference_performed") is True
                        and evidence.get("encoder_sha256") == ENCODER_SHA256
                        and evidence.get("token_sha256") == hashlib.sha256(token.astype(np.float32).tobytes()).hexdigest(),
                        "Executed token/source/actual encoder evidence differs")
                mobile_helper = check_reference(frame, event, evidence, metadata, bindings,
                                                helper=mobile_helper, checked_identity=checked_identity)
                mobile_audit.check_hands(frame, hands, metadata)
        stage("raw_fullbody_and_actual_command_ledger", commands)

        def dense_capture():
            warmup_ticks = round(float(manifest["warmup_seconds"])*50)
            warmup_count = warmup_ticks*30//50
            require(len(events) == (ticks-1)*30//50+1 and manifest["event_count"] == len(events)
                    and manifest["warmup_event_count"] == warmup_count
                    and manifest["command_event_count"] == len(events)-warmup_count
                    and len(events) >= warmup_count+30, "Missing dense30Hz samples or complete action horizon")
            require(manifest.get("physics_hz") == 200 and manifest.get("decoder_hz") == 50
                    and manifest.get("command_rate_hz") == 30 and manifest.get("render_engine") == "Isaac RTX"
                    and manifest.get("joint_names") == list(STATE_NAMES), "Physical/rendering/state channel identity differs")
            world_reference = manifest["isaac_world_time_reference_s"]
            seen = set()
            for index, event in enumerate(events):
                tick = (index*50+29)//30
                valid = tick >= warmup_ticks
                require(event["event_index"] == event["capture_index30"] == index
                        and event["physical_tick50"] == tick and capture_tick_index(tick) == index
                        and close(event["sim_time_s"], start_time+tick/50)
                        and close(event["actual_elapsed_s"], tick/50)
                        and close(event["nominal_elapsed_s"], index/30)
                        and event["action_valid"] is valid
                        and event["phase"] == ("teacher_command" if valid else "warmup"), "Dense observation/action phase or timestamps differ")
                require(event["joint_names"] == list(STATE_NAMES), "Captured physical joint order differs")
                state = vector(event["state43"], 43, "state")
                physical = vector(event["physical_joint_q43_float64"], 43, "physical state")
                require(np.array_equal(state.astype(np.float32), physical.astype(np.float32)), "Observed and actual state differ")
                vector(event["physical_joint_qvel43"], 43, "joint velocity")
                pose(event["root_pose7"], "root")
                vector(event["root_qvel6"], 6, "root velocity")
                for role in ("target",):
                    pose(event["objects"][role]["pose7"], role)
                command = ledger[index+1]
                if valid:
                    action = event["action"]
                    require(event["source_command_index"] == tick-warmup_ticks
                            and action["hand_joint_names"] == list(MODEL_HAND_NAMES), "Captured command index/hand names differ")
                    require(np.array_equal(vector(action["body_token64"],64,"token").astype(np.float32),
                                           vector(command["body_token64"],64,"ledger token").astype(np.float32))
                            and np.array_equal(vector(action["hand14"],14,"hands").astype(np.float32),
                                np.asarray([command["named_hand_targets"][name] for name in MODEL_HAND_NAMES],np.float32)), "Captured action is not the actual logged command")
                else:
                    require(event["action"] is None and event["source_command_index"] is None, "Warmup has fabricated supervised actions")
                _check_renderer(event, world_reference)
                image = event["rgb"]
                path = _owned(root/"capture", image["path"])
                require(path.suffix == ".png" and path not in seen and sha(path) == image["file_sha256"], "Raw PNG is reused/changed")
                seen.add(path)
                with Image.open(path) as opened:
                    require(opened.mode == "RGB" and opened.size == (image["width"],image["height"]), "RTX image geometry differs")
                    pixels = np.asarray(opened)
                require(image["dtype"] == "uint8" and image["channels"] == 3
                        and hashlib.sha256(pixels.tobytes()).hexdigest() == image["pixel_sha256"], "Actual RGB pixels changed")
                bindings[str(path)] = image["file_sha256"]
        stage("dense_actual_rgb_state_actions", dense_capture)

        def pick_reference():
            historical = mobile_audit.check_historical_sources(metadata, historical_source_manifest, bindings)
            original_quality = mobile_audit.check_native_source_binding(root, quality_name, bindings)
            trace = check_control_trace(root, metadata, outcome, events, ledger, qualities, bindings)
            report['pick_reference_audit'] = {'historical_sources':historical, 'control_trace':trace,
                'original_native_source':original_quality,
                'executed_reference_semantics_checked':report['checks'].get(
                    'raw_fullbody_and_actual_command_ledger', {}).get('passed') is True,
                'mode2_reference_input_recomputed_when_present':True,
                'mode1_current_VR_recomputed_when_present':True,
                'mode0_forecast_full_numeric_input_not_serialized':True,
                'reference_numeric_inference_independently_reproduced':False,
                'actual_original_source_code_snapshots_verified':True,
                'model_forecasts_are_training_labels':False, 'dynamics_steps_performed':0}
        stage('independent_pick_reference', pick_reference)

        def native_physical():
            result = native_gate(qualities, start_time, metadata, events)
            report['physical_gate'] = result
            require(result.get('native_success_evidence_complete') is True,
                    'Original native pick native capture evidence is incomplete: '+str(result.get('capture_validation_errors')))
            require(result.get('official_task_success') is True, 'Original official pick task success flag was never true')
        stage('independent_native_pick_task_success', native_physical)

        def supervised_success():
            warmup_ticks = round(float(manifest['warmup_seconds']) * 50)
            report['pick_supervised_coverage'] = supervised_coverage(qualities, warmup_ticks)
        stage('independent_pick_supervised_success_coverage', supervised_success)

        def external_forces():
            require(len(forces) == ticks*4, "Missing original200Hz external-force measurements")
            for index, row in enumerate(forces):
                require(close(row["sim_time_s"], start_time+(index+1)/200)
                        and row.get("all_zero") is True
                        and row.get("qfrc_applied_all_zero") is True and row.get("xfrc_applied_all_zero") is True
                        and row.get("qfrc_applied_nonzero_count") == 0 and row.get("xfrc_applied_nonzero_count") == 0
                        and row.get("qfrc_applied_max_abs") == 0 and row.get("xfrc_applied_max_abs") == 0
                        and row.get("max_abs_qfrc_applied") == 0 and row.get("max_abs_xfrc_applied") == 0,
                        "External force evidence is incomplete, nonzero, or has a source-clock gap")
        stage("all_original200hz_external_forces_zero", external_forces)
    except Exception as error:
        report["errors"].append(f"evidence_load: {type(error).__name__}: {error}")
    if not report["errors"] and len(report["checks"]) == expected_checks:
        report.update(status="independent_live_human_episode_audit_passed", eligible_for_packing=True)
    if report.get('selection_policy') == 'native_only':
        result = report.get('physical_gate', {})
        native_failure = (result.get('native_success_evidence_complete') is True and
            result.get('official_task_success') is False and
            all(check.get('passed') is True for name, check in report['checks'].items()
                if name != 'independent_native_pick_task_success') and
            len(report['checks']) == expected_checks)
        technical = bool(report['errors']) and not native_failure
        report.update(technical_blocked=technical,
            failure_kind='technical_evidence_incomplete' if technical else
                'official_task_not_successful' if native_failure else None,
            official_task_success=None if technical else result.get('official_task_success'),
            official_task_success_observed=result.get('official_task_success_observed'),
            selection_status='technical_evidence_incomplete' if technical else
                'official_task_not_successful' if native_failure else 'selected')
    if write_report:
        target = root/"admission.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
        temporary.replace(target)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode_directory', type=Path)
    parser.add_argument('--historical-sources', type=Path, required=True)
    args = parser.parse_args()
    report = audit_episode(args.episode_directory, historical_source_manifest=args.historical_sources)
    print(json.dumps({k: report[k] for k in ('status', 'eligible_for_packing', 'errors')}, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report['eligible_for_packing'] else 1)


if __name__ == '__main__':
    main()
