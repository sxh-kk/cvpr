"""Independent CPU admission checks for a newly recorded live PICO episode.

No SDK, ONNX session, dynamics integration or GPU is initialized. Deferred
archives are independently read with CPU MuJoCo state/FK operations. A passing report only
permits subsequent dataset packaging: DINO extraction, checkpoint statistics,
split selection and production-reader checks remain separate. Recorded-input
diagnostics can never pass, even if their physical trajectory is successful.
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

from handover_contact_quality import evaluate_capture
from pico_live_input import parse_tracking, validate_frame
from pico_sonic_reference import (CONTRACT as REFERENCE_CONTRACT, ENCODER_SHA256,
                                  validate_body_pose, verify_runtime_sources)
from replay_handover_teacher import (CAPTURE_CONTRACT, MODEL_HAND_NAMES,
    RENDERER_LINKS, RENDERER_POSE_CONTRACT, capture_tick_index)


CONTRACT = "cvpr-live-pico-handover-independent-admission-v1"
DECODER_SHA256 = "34bae8570d4a4421a5391a5c2befd745d4a02d182ec539e5f9da44c091c67509"
SONIC_REVISION = "6733128a3d8a523b1418b06bca3cdf61c8b0987f"
TASK = "simple/G1WholebodyHandoverTeleop-v0"
BODY_NAMES = tuple(
    [f"{side}_{name}_joint" for side in ("left", "right")
     for name in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
    + [f"waist_{axis}_joint" for axis in ("yaw", "roll", "pitch")]
    + [f"{side}_{name}_joint" for side in ("left", "right")
       for name in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                    "wrist_roll", "wrist_pitch", "wrist_yaw")])
STATE_NAMES = BODY_NAMES + MODEL_HAND_NAMES


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def vector(value, width, label):
    array = np.asarray(value, dtype=np.float64)
    require(array.shape == (width,) and np.isfinite(array).all(), f"Invalid finite {label}[{width}]")
    return array


def pose(value, label):
    value = vector(value, 7, label)
    require(abs(np.linalg.norm(value[3:]) - 1.) <= 1e-5, f"Nonunit {label} quaternion")
    return value


def close(a, b, tolerance=1e-6):
    return math.isfinite(float(a)) and math.isfinite(float(b)) and math.isclose(float(a), float(b), rel_tol=0, abs_tol=tolerance)


def _read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda word: (_ for _ in ()).throw(ValueError(f"Nonfinite JSON {word}")))


def _owned(root, name):
    relative = Path(name)
    require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe episode-relative path")
    path = root / relative
    require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root), "Missing or escaped episode evidence: " + str(name))
    return path


def _bound_identities(metadata, bindings):
    controller = metadata["controller_identity"]
    require(controller["body_token_version"] == "body_token_v1_1"
            and controller["decoder_sha256"] == DECODER_SHA256
            and controller["compatibility_status"] == "verified"
            and controller["observation_phase"] == "post_step_mj_forward_same_qpos"
            and controller["pd_profile"] == "official_v1_1_ankle_pitch_1_5",
            "Fixed SONIC v1.1 controller identity is missing")
    provenance = Path(controller["provenance"]).resolve(strict=True)
    require(_read(provenance).get("revision") == SONIC_REVISION, "SONIC provenance revision changed")
    resources = provenance.parent
    decoder = resources / "model_decoder.onnx"
    require(sha(decoder) == DECODER_SHA256, "Actual local decoder SHA changed")
    supplied = metadata["encoder_identity"]
    paths = supplied["source_sha256"]
    humans = [Path(name) for name in paths if Path(name).name == "human_joints_info.pkl"]
    require(len(humans) == 1, "A single fixed human FK skeleton must be bound")
    checked = verify_runtime_sources(resources, humans[0])
    require(supplied == checked, "Actual local encoder/FK/gather source identities differ")
    compatibility = Path(controller["compatibility_evidence"]).resolve(strict=True)
    evidence = _read(compatibility)
    require(evidence.get("body_token_version") == "body_token_v1_1"
            and evidence.get("decoder_sha256") == DECODER_SHA256
            and evidence.get("compatibility_verified") is True
            and evidence.get("evidence", {}).get("encoder_identity") == checked,
            "Actual encoder/decoder producer binding is incomplete")
    for path in (provenance, decoder, compatibility):
        bindings[str(path)] = sha(path)
    bindings.update(checked["source_sha256"])
    bindings.update(checked["official_gather"]["source_sha256"])
    return checked


def _check_renderer(event, world_reference):
    """Cross-check the serialized sensor/prim evidence; no implicit re-render."""
    validation = event["renderer_pose_validation"]
    require(validation.get("contract") == RENDERER_POSE_CONTRACT
            and validation.get("mujoco_clock_matches_observation_and_event") is True
            and validation.get("actual_mujoco_links_and_camera_verified") is True
            and validation.get("isaac_world_time_fixed") is True,
            "Same-frame RTX pose checks are absent")
    diagnostics = event["renderer_pose_diagnostics"]
    require(close(diagnostics["mujoco_sim_time_s"], event["sim_time_s"])
            and close(diagnostics["isaac_world_time_s"], world_reference, 1e-9),
            "RTX diagnostic clock differs from the physical event")
    for name in RENDERER_LINKS:
        entry = diagnostics["links"][name]
        vector(entry["mujoco_position_m"], 3, "renderer link position")
        q = vector(entry["mujoco_quaternion_wxyz"], 4, "renderer link rotation")
        require(abs(np.linalg.norm(q)-1.) <= 1e-5, "Invalid renderer link rotation")
        usd = entry["usd"]
        require(usd.get("valid") is True and isinstance(usd.get("path"), str) and usd["path"].startswith("/"), "Invalid actual link USD prim")
        pose([*usd["position_m"], *usd["quaternion_wxyz"]], "USD link")
    pelvis = diagnostics["links"]["pelvis"]
    actual = pose(event["root_pose7"], "physical root")
    reported = np.r_[pelvis["mujoco_position_m"], pelvis["mujoco_quaternion_wxyz"]]
    require(np.allclose(actual[:3], reported[:3], rtol=0, atol=1e-9)
            and min(np.max(np.abs(actual[3:]-reported[3:])), np.max(np.abs(actual[3:]+reported[3:]))) <= 1e-9,
            "RTX reported pelvis differs from the actual captured root")
    camera = diagnostics["cameras"]["head_stereo_left"]
    require(not camera.get("mujoco_camera_missing") and camera["usd"].get("valid") is True,
            "Actual rendered head camera is missing")
    sensor = pose([*camera["sensor_position_m"], *camera["sensor_quaternion_wxyz_usd_axes"]], "RTX sensor")
    usd = pose([*camera["usd"]["position_m"], *camera["usd"]["quaternion_wxyz"]], "USD camera")
    require(np.allclose(sensor[:3], usd[:3], rtol=0, atol=1e-5)
            and min(np.max(np.abs(sensor[3:]-usd[3:])), np.max(np.abs(sensor[3:]+usd[3:]))) <= 1e-5,
            "Rendered sensor and actual USD camera pose disagree")
    rotation = vector(camera["mujoco_rotation_usd_axes"], 9, "MuJoCo camera rotation").reshape(3, 3)
    vector(camera["mujoco_position_m"], 3, "MuJoCo camera position")
    require(np.allclose(rotation.T@rotation, np.eye(3), rtol=0, atol=1e-5)
            and close(np.linalg.det(rotation), 1., 1e-5), "Invalid physical camera orientation")


def canonical_event_sha256(event):
    encoded = json.dumps(event, ensure_ascii=False, allow_nan=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _check_deferred_bridge(root, metadata, outcome, manifest, events, ledger, read, bindings):
    """Bind zero-step RTX to original complete physical archives independently."""
    from pico_deferred_capture import CONTRACT as DEFERRED_CONTRACT, PENDING, load_deferred
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
    for index, (source, event, row) in enumerate(zip(source_events, events, bridge)):
        require(all(source[key] == event[key] for key in shared), "RTX changed original physical/state/action field at frame "+str(index))
        require(source.get("preview", {}).get("training_observation") is False
                and source["preview"].get("engine") == "MuJoCo EGL"
                and source["preview"].get("saved_as_training_rgb") is False,
                "Original preview must be recorded explicitly as non-training feedback")
        fields, command = source["source_fields"], ledger[index+1]
        for key in ("source_raw_packet","source_host_received_s","source_sequence","device_timestamp_ns","reference_evidence"):
            require(fields[key] == command[key], "Deferred original input differs from actual command ledger: "+key)
        state = np.ascontiguousarray(archived["states"][index])
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
    return {"raw_completed":True,"cpu_full_integration_state_verified":True,
            "source_kind":metadata.get("source_kind"),
            "observation_domain_feedback":"MuJoCo EGL preview","training_rgb_domain":"Isaac RTX deferred",
            "rgb_was_live_operator_feedback":False,"frames_bound":len(events),
            "dynamics_steps_performed":0,"sonic_inferences_performed":0}


def _check_deferred_quality(metadata, outcome, qualities, read, bindings, root):
    from pico_simple_teleop import complete_quality_geometry
    raw = read("raw_quality_samples.jsonl", lines=True)
    cache = read("container_geometry_cache.json")
    require(metadata.get("container_geometry_cache_sha256") == bindings[str(root/"container_geometry_cache.json")],
            "Original immutable geometry cache SHA changed")
    sources = outcome["quality_geometry_sources"]
    require(sources.get("raw_samples_sha256") == bindings[str(root/"raw_quality_samples.jsonl")]
            and sources.get("cache_sha256") == bindings[str(root/"container_geometry_cache.json")]
            and sources.get("completed_samples_sha256") == bindings[str(root/"quality_samples.jsonl")],
            "Original solved-quality/cache/completion SHA changed")
    reconstructed = complete_quality_geometry(raw, cache)
    require(reconstructed == qualities, "Deferred geometry differs from original solved poses/support")
    for before, after in zip(raw, qualities):
        for key,value in before["original_live_support"].items():
            require(after["container_geometry"].get(key) == value,
                    "Original solved force was changed or reconstructed offline: "+key)
    return {"original_solved_force_retained":True,"deferred_geometry_reproduced":True,"rows":len(raw)}


def audit_episode(directory, *, write_report=True):
    """Recompute raw eligibility. Invalid/missing evidence yields a rejection."""
    root = Path(directory).resolve(strict=True)
    report = {"contract": CONTRACT, "generated_utc": datetime.now(timezone.utc).isoformat(),
        "episode_directory": str(root), "status": "independent_live_human_episode_audit_rejected",
        "eligible_for_packing": False, "accepted_new_dataset": False,
        "ckpt_feature_pack_ready": False, "checks": {}, "errors": [], "bindings": {},
        "scope": "Independent local raw-recording consistency and unchanged handover-physics-v2 gates; requires separate packing, DINO, normalization, split and checkpoint-reader acceptance",
        "limitations": ["File evidence establishes local producer consistency, not authenticated proof of human identity or protection from deliberate forgery.",
            "No simulator integration replay or image re-render is performed; camera sensor/prim/current-state evidence is checked as recorded.",
            "Solved-force quality rows precede observation refresh; their derived poses are not asserted bitwise equal to refreshed RGB-frame poses.",
            "The physical placement gate does not certify exact inner-cavity containment or official benchmark success."]}
    report["auditor_source_sha256"] = {str(path): sha(path) for path in (
        Path(__file__).resolve(), Path(__file__).with_name("pico_live_input.py").resolve(),
        Path(__file__).with_name("handover_contact_quality.py").resolve(),
        Path(__file__).with_name("handover_geometry_quality.py").resolve())}
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
        outcome = read("episode.json")
        manifest = read("capture/capture.json")
        deferred = metadata.get("render_mode") == "mujoco_deferred"
        async_result = None if deferred else read("capture/async_capture.json")
        events = read("capture/events.jsonl", lines=True)
        ledger = read("commands.jsonl", lines=True)
        qualities = read("quality_samples.jsonl", lines=True)
        forces = read("applied_force_samples.jsonl", lines=True)
        require(bool(events) and bool(qualities), "Empty recorded episode")
        start_time = float(events[0]["sim_time_s"])
        ticks = len(qualities)
        identity = str(uuid.UUID(metadata["human_demo_uuid"]))
        report.update(human_demo_uuid=identity, event_count=len(events), control_ticks50=ticks)
        expected_checks = 7
        if deferred:
            report.update(observation_domain_feedback="MuJoCo EGL preview", training_rgb_domain="Isaac RTX deferred",
                          rgb_was_live_operator_feedback=False)

        def source():
            for data in (metadata, outcome, manifest):
                require(data.get("source_kind") == "live_pico_human", "Recorded replay/unknown source is never eligible")
                require(data.get("accepted_new_dataset") is False, "Raw recording must not claim dataset acceptance")
                require(data.get("replay_source_sha256") is None, "Recorded replay provenance is present")
            require(metadata.get("task_id") == TASK and metadata.get("render_mode") in ("isaac", "mujoco_deferred")
                    and metadata.get("physics_hz") == 200 and metadata.get("decoder_hz") == 50
                    and metadata.get("command_rgb_hz") == 30 and metadata.get("raw_pico_va_wva_used") is False,
                    "Task/rendering/rate/raw-derivative contract differs")
            initial = metadata["initialization"]
            require(initial.get("object_restores_during_control") is False and initial.get("external_assistance") is False,
                    "External assistance or in-episode state restore is not accepted")
            initial_z = metadata.get("initial_target_z_m")
            require(type(initial_z) in (int, float) and math.isfinite(initial_z)
                    and type(manifest.get("initial_target_z_m")) in (int, float)
                    and close(initial_z, manifest["initial_target_z_m"], 1e-9)
                    and close(initial_z, pose(events[0]["objects"]["target"]["pose7"], "initial target")[2], 1e-9),
                    "Initial target height must be finite and match actual first-frame and capture metadata")
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
        if metadata.get("quality_geometry_mode") == "deferred_geometry_with_original_solved_support":
            expected_checks += 1
            report["deferred_quality_geometry"] = stage("original_solved_support_and_deferred_geometry", lambda:
                _check_deferred_quality(metadata, outcome, qualities, read, bindings, root))
        checked_identity = stage("actual_fixed_sonic_artifacts", lambda: _bound_identities(metadata, bindings))

        def commands():
            require(len(ledger) == len(events)+1 and ledger[0].get("event_type") == "session", "Dense command ledger is missing")
            require(outcome["ledger"]["sha256"] == bindings[str(root/"commands.jsonl")]
                    and outcome["ledger"]["event_count"] == len(ledger)
                    and outcome["ledger"]["human_demo_uuid"] == identity
                    and ledger[0]["metadata"] == metadata, "Final command ledger SHA/count/identity changed")
            previous = None
            previous_ns = None
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
                require(evidence.get("contract") == REFERENCE_CONTRACT and evidence.get("encoder_mode") == 2
                        and evidence.get("model_inference_performed") is True
                        and evidence.get("encoder_sha256") == ENCODER_SHA256
                        and evidence.get("PICO_raw_va_wva_used") is False
                        and evidence.get("device_timestamp_ns") == frame.device_timestamp_ns
                        and evidence.get("token_sha256") == hashlib.sha256(token.astype(np.float32).tobytes()).hexdigest()
                        and evidence.get("identities") == checked_identity,
                        "Executed token/source/actual encoder evidence differs")
                horizon = evidence["reference_horizon"]
                require(horizon.get("kind") == "causal_constant_desired_pose_hold"
                        and horizon.get("measured_future_frames") == 0, "Live desired reference is misrepresented as measured future")
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
                for role in ("target", "container"):
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

        def physical():
            for index, row in enumerate(qualities):
                require(row["completed_steps"] == index+1 and close(row["sim_time_s"], start_time+(index+1)/50), "Physical50Hz evidence has gaps")
            warmup_ticks = round(float(manifest["warmup_seconds"])*50)
            require(ticks > warmup_ticks, "No physical task evidence after warmup")
            # Warmup has no action labels. A grasp/transfer hidden in that
            # prefix must never satisfy task success for the supervised suffix.
            initial_target = vector(events[0]["objects"]["target"]["pose7"][:3], 3, "initial target")
            for row in qualities[:warmup_ticks]:
                contacts = row["contacts"]
                require(contacts.get("left_hand") is False and contacts.get("right_hand") is False
                        and np.linalg.norm(vector(row["actual_target_xyz_m"], 3, "warmup target")-initial_target) <= .01,
                        "Task/grasp began inside the unsupervised warmup prefix")
            supervised = [{**row, "completed_steps": index+1}
                          for index, row in enumerate(qualities[warmup_ticks:])]
            result = evaluate_capture(supervised, initial_target_z_m=metadata["initial_target_z_m"])
            result["excluded_unsupervised_warmup_ticks50"] = warmup_ticks
            report["physical_gate"] = result
            require(result.get("eligible_physics") is True, "Unchanged handover-physics-v2 gate failed")
        stage("independent_physical_handover_gate", physical)

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
    if write_report:
        target = root/"admission.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
        temporary.replace(target)
    return report


def _read_line(line):
    return json.loads(line, parse_constant=lambda word: (_ for _ in ()).throw(ValueError(f"Nonfinite JSON {word}")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode_directory", type=Path)
    args = parser.parse_args()
    report = audit_episode(args.episode_directory)
    print(json.dumps({key: report[key] for key in ("status", "eligible_for_packing", "accepted_new_dataset", "errors")}, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["eligible_for_packing"] else 1)


if __name__ == "__main__":
    main()
