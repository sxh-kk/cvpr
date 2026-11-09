"""Independent target-only admission for the explicit PICO/SONIC Grasp adaptation.

The original Handover/Mobile auditor remains byte-for-byte unchanged. Shared
boundary, state/RGB/command and CPU source checks are reused without running
SDK, ONNX, physics or rendering. Packaging and production-reader acceptance
remain separate. Native task success is never replaced by a geometric gate.
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
from pico_grasp_quality import (GRASP_TASK_ID, OFFICIAL_SOURCE_SHA256,
    evaluate_grasp_capture, validate_grasp_task_identity)

CONTRACT = 'cvpr-live-pico-grasp-independent-admission-v1'
CONTROL_CONTRACT = 'cvpr-pico-grasp-standing-live-vr-mode1-v1'
# The first GPU pilot exposed Isaac close() exiting the interpreter before its
# finally block could finalize the batch report. Only these reviewed producer
# bytes may use the explicit external-exit recovery below.
LEGACY_CLOSE_EXIT_WRAPPER_SHA = '0cb9a330f3d520d143222f94227a13cb8428c68d401902414e2799882dfd652f'
AUDITOR_FILES = frozenset(('pico_grasp_audit.py','audit_pico_teleop_episode.py',
    'pico_live_input.py','pico_grasp_quality.py','pico_grasp_control.py',
    'pico_mobile_audit.py','pico_mobile_quality.py','pico_mobile_vr_reference.py',
    'handover_sonic_teleop_reference.py','handover_contact_quality.py'))
BASE_CHECKS = frozenset(('live_source_and_rates','complete_recording_and_writer',
    'actual_fixed_sonic_artifacts','raw_fullbody_and_actual_command_ledger',
    'dense_actual_rgb_state_actions','independent_native_grasp_task_success',
    'independent_grasp_continuous_reference','independent_grasp_supervised_success_coverage',
    'all_original200hz_external_forces_zero'))


def validate_metadata(metadata, bindings):
    result = validate_grasp_task_identity(metadata)
    require(result['passed'], 'Original Grasp task/adapter/frame identity: '+str(result['errors']))
    source = Path(__file__).resolve().parents[2]/'VLMExpert/third_party/SIMPLE/src/simple/tasks/g1_wholebody_tabletop_grasp_mp.py'
    require(source.is_file() and sha(source) == OFFICIAL_SOURCE_SHA256, 'Original pinned Grasp implementation changed')
    bindings[str(source)] = OFFICIAL_SOURCE_SHA256
    adapter = Path(__file__).with_name('pico_grasp_task.py').resolve()
    require(sha(adapter) == result['adapter_sha256'], 'Grasp adapter differs from recorded implementation')
    bindings[str(adapter)] = result['adapter_sha256']
    require(metadata.get('mobile_control_protocol') == CONTROL_CONTRACT and
            metadata.get('mobile_control_policy') == 'continuous_vr_mode1' and
            type(metadata.get('mobile_encoder_mode')) is int and metadata['mobile_encoder_mode'] == 1 and
            metadata.get('mobile_planner_model_sha256') == mobile_audit.PLANNER_MODEL_SHA256,
            'Grasp standing mode1/planner identity differs')
    identity = metadata.get('mobile_vr_source', {})
    require(identity.get('official_sources_rehashed') is True, 'Original VR source freeze missing')
    mobile_audit._bind_pinned_sources(identity.get('source_sha256'),
        mobile_audit.vr.verify_official_sources()['source_sha256'], bindings)
    return result


def check_reference(frame, event, evidence, metadata, bindings, *, helper=None):
    require(evidence.get('contract') == CONTROL_CONTRACT, 'Original Grasp standing reference contract differs')
    source = evidence.get('source', {})
    require(source.get('applied_axes_left_x_y_right_x_y') == [0.,0.,0.,0.] and
            source.get('ignored_axes') == ['left_axis_x','left_axis_y','right_axis_x','right_axis_y'] and
            source.get('ignored_reason') == ['tabletop_grasp_standing_reference'] and
            source.get('axis_policy') == 'grasp_standing_leg_reference_all_joystick_axes_ignored',
            'Grasp must ignore all joystick axes for its standing leg reference')
    for name in ('desired_command','applied_plan_command'):
        command = evidence.get(name, {})
        require(type(command.get('mode')) is int and command['mode'] == 0 and
                command.get('movement_direction') == [0.,0.,0.] and
                command.get('speed') == -1. and command.get('height') == -1.,
                'Grasp planner command must remain the original standing idle reference')
        facing = vector(command.get('facing_direction'),3,'standing facing direction')
        require(close(float(np.linalg.norm(facing)),1.,1e-6) and close(facing[2],0.),
                'Grasp standing facing direction must be finite horizontal unit vector')
    # After explicitly checking the different standing envelope above, all
    # remaining mode1 VR/encoder/forecast algebra is shared with Mobile P&P.
    # This temporary dictionary never changes the archived serialized bytes.
    common = dict(evidence, contract=mobile_audit.CONTROL_CONTRACT)
    return mobile_audit.check_reference(frame, event, common, metadata, bindings, helper=helper)


def native_gate(qualities, start_time):
    for index,row in enumerate(qualities):
        require(type(row.get('completed_steps')) is int and row['completed_steps'] == index+1 and
                type(row.get('sim_time_s')) in (int,float) and math.isfinite(row['sim_time_s']) and
                close(row['sim_time_s'],start_time+(index+1)/50), 'Original Grasp50Hz evidence clock has gaps')
        reward = row.get('native_task_reward')
        if type(reward) in (int,float) and math.isfinite(reward) and type(row.get('official_task_success')) is bool:
            require(row['official_task_success'] == (reward >= .9), 'Original native reward and check_success boolean differ')
    result = evaluate_grasp_capture(qualities)
    result['excluded_unsupervised_warmup_ticks50'] = 0
    return result


def _check_deferred_bridge(root, metadata, outcome, manifest, events, ledger, read, bindings):
    """Bind zero-step RTX to original complete physical archives independently."""
    from pico_deferred_capture import GRASP_CONTRACT as DEFERRED_CONTRACT, PENDING, load_deferred
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
    require(completion.get("contract") == "cvpr-pico-deferred-rtx-completion-v1"
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
    renderer = Path(__file__).with_name("render_deferred_pico.py").resolve()
    require(completion.get("render_code_sha256") == sha(renderer), "Deferred renderer source changed since acquisition")
    bindings[str(renderer)] = completion["render_code_sha256"]
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


def _check_batch_execution(root, completion, metadata, bindings):
    """A leased factory is a different producer and needs its own frozen proof.

    The wrapper is producer evidence, like the single-episode renderer; it is
    SHA-bound separately from independent auditor source authorities. Original
    non-batch receipts remain valid without requiring a batch report.
    """
    execution = completion.get('batch_execution')
    if execution is None:
        return None
    require(isinstance(execution, dict), 'Invalid batch execution envelope')
    contract = 'cvpr-pico-grasp-deferred-rtx-batch-v1'
    script = Path(__file__).with_name('render_deferred_pico_batch.py').resolve(strict=True)
    current_wrapper_sha = sha(script)
    wrapper_sha = execution.get('batch_code_sha256')
    renderer = Path(__file__).with_name('render_deferred_pico.py').resolve(strict=True)
    renderer_sha = sha(renderer)
    require(execution.get('contract') == contract and
            isinstance(wrapper_sha, str) and len(wrapper_sha) == 64 and
            execution.get('original_renderer_code_sha256') == renderer_sha and
            execution.get('real_environment_count') == 1 and
            execution.get('original_reset_and_environment_binding_per_episode') is True and
            execution.get('factory_override') == 'single environment lease; defer close until batch finally' and
            type(execution.get('environment_lease_number')) is int and execution['environment_lease_number'] >= 1,
            'Batch wrapper source identity or environment lease envelope differs')

    def bound_json(name, label):
        require(isinstance(name, str) and Path(name).is_absolute(), 'Missing absolute '+label+' path')
        path = Path(name)
        require(path.is_file() and not path.is_symlink(), 'Missing/unsafe '+label)
        before = mobile_audit._stat(path)
        encoded = path.read_bytes()
        value = json.loads(encoded, parse_constant=lambda word: (_ for _ in ()).throw(ValueError('Nonfinite '+label)))
        require(mobile_audit._stat(path) == before, label+' changed during admission')
        digest = hashlib.sha256(encoded).hexdigest()
        bindings[str(path)] = digest
        return path, value, digest

    report_path, report, report_sha = bound_json(execution.get('batch_report_path'), 'batch report')
    require(report.get('contract') == contract and
            report.get('accepted_new_dataset') is False and
            report.get('batch_code_sha256') == wrapper_sha and
            report.get('original_renderer_code_sha256') == renderer_sha,
            'Batch report is incomplete/rejected or names another producer')
    recovery_binding = {}
    leases = report.get('new_environment_leases')
    if report.get('status') == 'render_completed_pending_process_exit' and wrapper_sha == current_wrapper_sha:
        require(report.get('cleanup_requested') is True and report.get('cleanup_return_verified') is False and
                type(leases) is int and leases >= 1,
                'Pre-close batch report must preserve the explicit unverified cleanup boundary')
        exit_path = report_path.with_name(report_path.stem+'_EXIT.json')
        exit_path, exit_receipt, exit_sha = bound_json(str(exit_path), 'independent process exit')
        require(exit_receipt.get('contract') == 'cvpr-grasp-rtx-observed-process-exit-v1' and
                type(exit_receipt.get('child_pid')) is int and exit_receipt['child_pid'] > 0 and
                exit_receipt.get('report_path') == str(report_path) and
                exit_receipt.get('report_sha256') == report_sha and
                type(exit_receipt.get('exit_code')) is int and exit_receipt['exit_code'] == 0 and
                isinstance(exit_receipt.get('argv'), list) and bool(exit_receipt['argv']) and
                all(isinstance(value, str) for value in exit_receipt['argv']),
                'Process exit must bind this exact completed-render report and original argv')
        driver_name = exit_receipt.get('parentdriver_path')
        require(isinstance(driver_name, str) and Path(driver_name).is_absolute(),
                'Independent process-exit driver source path is missing')
        driver_path = Path(driver_name)
        require(driver_path.is_file() and not driver_path.is_symlink(), 'Unsafe independent exit driver source')
        driver_sha = sha(driver_path)
        require(exit_receipt.get('parentdriver_sha256') == driver_sha,
                'Independent process-exit driver source bytes changed')
        bindings[str(driver_path)] = driver_sha
        finished = datetime.fromisoformat(exit_receipt['finished_beijing'])
        rendered = datetime.fromisoformat(report['render_finished_utc'])
        require(finished.tzinfo is not None and rendered.tzinfo is not None and finished >= rendered,
                'Independent process-exit timestamp predates the last rendered output')
        recovery_binding = {'process_exit_path':str(exit_path),'process_exit_sha256':exit_sha,
            'parentdriver_path':str(driver_path),'parentdriver_sha256':driver_sha,
            'cleanup_return_verified':False,
            'original_report_status_preserved':'render_completed_pending_process_exit'}
    elif report.get('status') != 'completed_pending_independent_audit' or wrapper_sha != current_wrapper_sha:
        # No generic running-report bypass: this exact first-pilot implementation
        # must have all selected rows, an immutable original report and explicit
        # observed process exit0. Cleanup return is deliberately never certified.
        require(wrapper_sha == LEGACY_CLOSE_EXIT_WRAPPER_SHA and report.get('status') == 'running' and
                report.get('selected_count') == 2 and len(report.get('episodes', [])) == 2 and
                not any(key in report for key in ('error','close_error','traceback','close_traceback')),
                'Unfinished batch needs a reviewed exact-producer recovery; never admit arbitrary running reports')
        recovery_path = report_path.with_name(report_path.stem+'.recovery.json')
        recovery_path, recovery, recovery_sha = bound_json(str(recovery_path), 'batch close-exit recovery')
        require(recovery.get('contract') == 'cvpr-pico-grasp-batch-close-exit-recovery-v1' and
                recovery.get('status') == 'rendered_outputs_completed_process_exit_zero' and
                recovery.get('cleanup_return_verified') is False and
                recovery.get('original_report') == {'path':str(report_path),'sha256':report_sha} and
                recovery.get('original_renderer_code_sha256') == renderer_sha and
                recovery.get('episodes') == report['episodes'] and
                recovery.get('new_environment_leases') == 2,
                'Legacy recovery must bind the unchanged report/producer and every completed output')
        producer = recovery.get('producer_source', {})
        require(isinstance(producer.get('path'), str) and Path(producer['path']).is_absolute(),
                'Legacy producer source snapshot is missing')
        script = Path(producer['path'])
        require(script.is_file() and not script.is_symlink() and
                producer.get('sha256') == wrapper_sha == sha(script),
                'Retained exact legacy producer bytes changed')
        external_exit = recovery.get('process_exit', {})
        exit_path, exit_receipt, exit_sha = bound_json(external_exit.get('path'), 'independent process exit')
        require(external_exit.get('sha256') == exit_sha and
                type(exit_receipt.get('exit_code')) is int and exit_receipt['exit_code'] == 0,
                'Legacy render process did not have a separately observed exit0')
        finished = datetime.fromisoformat(exit_receipt['finished_beijing'])
        created = datetime.fromisoformat(report['created_utc'])
        require(finished.tzinfo is not None and created.tzinfo is not None and finished >= created,
                'Independent process-exit timestamp predates this batch')
        recovery_binding = {'recovery_receipt_path':str(recovery_path),'recovery_receipt_sha256':recovery_sha,
            'process_exit_path':str(exit_path),'process_exit_sha256':exit_sha,
            'cleanup_return_verified':False,
            'original_report_status_preserved':'running'}
        leases = recovery['new_environment_leases']
    else:
        require(wrapper_sha == current_wrapper_sha, 'Completed batch producer implementation changed')
        if wrapper_sha != LEGACY_CLOSE_EXIT_WRAPPER_SHA:
            require(report.get('cleanup_requested') is True and report.get('cleanup_return_verified') is True,
                    'Final batch status must certify cleanup actually returned')
    manifest_path, manifest, manifest_sha = bound_json(report.get('manifest_path'), 'frozen batch manifest')
    frozen = manifest.get('episodes', [])
    require(report.get('manifest_sha256') == execution.get('frozen_manifest_sha256') == manifest_sha and
            manifest.get('contract') == 'cvpr-grasp100-rtx-admission-frozen-inputs-v1' and
            manifest.get('count') == len(frozen) == 100 and
            len({row['uuid'] for row in frozen}) == 100 and
            {row['official_episode_index'] for row in frozen} == set(range(100)),
            'Frozen batch cohort identity differs')
    if 'recovery_receipt_path' in recovery_binding:
        require(recovery.get('frozen_manifest') == {'path':str(manifest_path),'sha256':manifest_sha},
                'Legacy recovery frozen cohort differs')
    identity = metadata['human_demo_uuid']
    candidates = [row for row in frozen if row.get('uuid') == identity]
    require(len(candidates) == 1 and Path(candidates[0]['directory']).resolve(strict=True) == root.resolve(strict=True),
            'This physical episode is not the frozen batch UUID/directory')
    candidate = candidates[0]
    for name, expected in candidate['source_files'].items():
        path = _owned(root, name)
        stat = path.stat()
        require(stat.st_size == expected['size_bytes'] and stat.st_mtime_ns == expected['mtime_ns'],
                'Frozen batch source stat changed: '+name)
        digest = sha(path)
        require('sha256' not in expected or digest == expected['sha256'], 'Frozen batch source SHA changed: '+name)
        require(str(path) not in bindings or bindings[str(path)] == digest, 'Batch source changed during admission: '+name)
        bindings[str(path)] = digest
    rows = report.get('episodes', [])
    require(type(report.get('selected_count')) is int and 1 <= report['selected_count'] <= 100 and
            len(rows) == report['selected_count'] and len({row['uuid'] for row in rows}) == len(rows) and
            [row['uuid'] for row in rows] == [row['uuid'] for row in frozen[:len(rows)]],
            'Batch report episode coverage differs')
    completed = [row for row in rows if row.get('uuid') == identity]
    require(len(completed) == 1 and completed[0].get('number') == candidate.get('number') and
            Path(completed[0]['directory']).resolve(strict=True) == root.resolve(strict=True) and
            completed[0].get('rtx_completion_sha256') == sha(root/'rtx_completion.json') and
            completed[0].get('completed_frames') == completion.get('completed_frames') and
            completed[0].get('environment_binding') == completion.get('environment_binding') and
            completed[0].get('reused_completed_output') is False and
            type(leases) is int and execution['environment_lease_number'] <= leases,
            'Completed batch report does not bind this exact episode receipt')
    bindings[str(script)] = wrapper_sha
    return {'contract':contract, 'batch_code_path':str(script), 'batch_code_sha256':wrapper_sha,
        'report_path':str(report_path), 'report_sha256':report_sha,
        'frozen_manifest_path':str(manifest_path), 'frozen_manifest_sha256':manifest_sha,
        **recovery_binding,
        'uuid':identity, 'environment_lease_number':execution['environment_lease_number'],
        'completed_report_binds_this_receipt':True}


def audit_episode(directory, *, write_report=True, historical_source_manifest=None):
    """Recompute raw eligibility. Invalid/missing evidence yields a rejection."""
    root = Path(directory).resolve(strict=True)
    report = {"contract": CONTRACT, "generated_utc": datetime.now(timezone.utc).isoformat(),
        "episode_directory": str(root), "status": "independent_live_human_episode_audit_rejected",
        "eligible_for_packing": False, "accepted_new_dataset": False,
        "ckpt_feature_pack_ready": False, "checks": {}, "errors": [], "bindings": {},
        "scope": "Independent local raw-recording consistency and original Grasp native task flag and standing mode1 reference consistency; requires separate packing, DINO, normalization, split and checkpoint-reader acceptance",
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
        require(metadata.get('task_id') == GRASP_TASK_ID, 'Explicit Grasp task identity required')
        report.update(task_id=GRASP_TASK_ID, selection_policy='native_only', technical_blocked=False)
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
            require(metadata.get("task_id") == GRASP_TASK_ID and metadata.get("render_mode") in ("isaac", "mujoco_deferred")
                    and metadata.get("physics_hz") == 200 and metadata.get("decoder_hz") == 50
                    and metadata.get("command_rgb_hz") == 30 and metadata.get("raw_pico_va_wva_used") is False,
                    "Task/rendering/rate/raw-derivative contract differs")
            initial = metadata["initialization"]
            require(initial.get("object_restores_during_control") is False and initial.get("external_assistance") is False,
                    "External assistance or in-episode state restore is not accepted")
            report['official_grasp_identity'] = validate_metadata(metadata, bindings)
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
                                                helper=mobile_helper)
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

        def grasp_reference():
            historical = mobile_audit.check_historical_sources(metadata, historical_source_manifest, bindings)
            original_quality = mobile_audit.check_native_source_binding(root, quality_name, bindings)
            trace = mobile_audit.check_control_trace(root, metadata, outcome, events, ledger, qualities, bindings)
            report['grasp_reference_audit'] = {'historical_sources':historical, 'control_trace':trace,
                'original_native_source':original_quality,
                'same_packet_body_and_same_tick_root_VR_recomputed':report['checks'].get(
                    'raw_fullbody_and_actual_command_ledger', {}).get('passed') is True,
                'reference_numeric_inference_independently_reproduced':False,
                'actual_original_source_code_snapshots_verified':True,
                'model_forecasts_are_training_labels':False, 'dynamics_steps_performed':0}
        stage('independent_grasp_continuous_reference', grasp_reference)

        def native_physical():
            result = native_gate(qualities, start_time)
            report['physical_gate'] = result
            require(result.get('native_success_evidence_complete') is True,
                    'Original Grasp native capture evidence is incomplete: '+str(result.get('capture_validation_errors')))
            require(result.get('official_task_success') is True, 'Original official Grasp task success flag was never true')
        stage('independent_native_grasp_task_success', native_physical)

        def supervised_success():
            warmup_ticks = round(float(manifest['warmup_seconds'])*50)
            require(ticks > warmup_ticks, 'Missing supervised task suffix')
            result = {'warmup_ticks50':warmup_ticks,
                'warmup_native_success_observed':any(r.get('official_task_success') is True for r in qualities[:warmup_ticks]),
                'supervised_native_success_observed':any(r.get('official_task_success') is True for r in qualities[warmup_ticks:]),
                'warmup_first_target_xyz_m':qualities[0]['actual_target_xyz_m'],
                'first_supervised_target_xyz_m':qualities[warmup_ticks]['actual_target_xyz_m'],
                'warmup_actions_excluded_from_supervision':True,
                'no_added_contact_lift_hold_or_final_state_gate':True,
                'scope':'Native task success remains any original real flag; training coverage additionally requires a successful supervised suffix. Prefix motion is retained as actual visual history, never labeled actions.'}
            report['grasp_supervised_coverage'] = result
            require(not result['warmup_native_success_observed'] or result['supervised_native_success_observed'],
                    'Native success only in unlabeled warmup cannot establish supervised training coverage')
        stage('independent_grasp_supervised_success_coverage', supervised_success)

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
                if name != 'independent_native_grasp_task_success') and
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
    parser.add_argument('episode_directory',type=Path)
    parser.add_argument('--historical-sources',type=Path,required=True)
    args = parser.parse_args()
    report = audit_episode(args.episode_directory,historical_source_manifest=args.historical_sources)
    print(json.dumps({k:report[k] for k in ('status','eligible_for_packing','errors')},ensure_ascii=False,indent=2))
    raise SystemExit(0 if report['eligible_for_packing'] else 1)


if __name__ == '__main__':
    main()
