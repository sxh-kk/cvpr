"""Lossless CPU packaging of a frozen, independently admitted live-PICO list.

No token encoding, simulation, model inference, new statistics, split selection
or training is performed.  This adapter reuses the established numeric/video
conversion, without claiming the older teacher collection's ancestry contract.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import uuid

import numpy as np

import pack_handover_quality100 as legacy
from pico_episode_review import discard_requested


CONTRACT = "cvpr-pico-handover-lerobot-pack-v1"
SELECTION_CONTRACT = "cvpr-pico-handover-admissions-selection-v1"
ADMISSION_CONTRACT = "cvpr-live-pico-handover-independent-admission-v1"
HANDOVER_TASK_ID = "simple/G1WholebodyHandoverTeleop-v0"
MOBILE_TASK_ID = "simple/G1WholebodyLocomotionPickBetweenTablesTeleop-v0"
MOBILE_CONTRACT = "cvpr-pico-mobile-pp-lerobot-pack-v1"
MOBILE_SELECTION_CONTRACT = "cvpr-pico-mobile-pp-admissions-selection-v1"
MOBILE_ADMISSION_CONTRACT = "cvpr-live-pico-mobile-pp-independent-admission-v1"
SOURCE_FILES = ("source_initialization_receipt.json", "source_run_config.json",
                "source_dataset_statistics.json", "source_state_info.json")
AUDITOR_FILES = frozenset(("audit_pico_teleop_episode.py", "pico_live_input.py",
                          "handover_contact_quality.py", "handover_geometry_quality.py"))
BASE_CHECKS = frozenset(("live_source_and_rates", "complete_recording_and_writer",
    "actual_fixed_sonic_artifacts", "raw_fullbody_and_actual_command_ledger",
    "dense_actual_rgb_state_actions", "independent_physical_handover_gate",
    "all_original200hz_external_forces_zero"))
MOBILE_AUDITOR_FILES = frozenset(("audit_pico_teleop_episode.py", "pico_live_input.py",
    "pico_mobile_audit.py", "pico_mobile_quality.py", "pico_mobile_vr_reference.py",
    "handover_sonic_teleop_reference.py", "handover_contact_quality.py"))
MOBILE_BASE_CHECKS = (BASE_CHECKS - {"independent_physical_handover_gate"}) | {
    "independent_native_mobile_task_success", "independent_mobile_continuous_reference"}
require, sha = legacy.require, legacy.sha


def task_protocol(task_id):
    """Explicit task envelopes; old Handover artifacts keep their contracts."""
    require(task_id in (HANDOVER_TASK_ID, MOBILE_TASK_ID), "Unsupported live-PICO task identity")
    mobile = task_id == MOBILE_TASK_ID
    return {"task_id": task_id, "mobile": mobile,
        "pack_contract": MOBILE_CONTRACT if mobile else CONTRACT,
        "selection_contract": MOBILE_SELECTION_CONTRACT if mobile else SELECTION_CONTRACT,
        "admission_contract": MOBILE_ADMISSION_CONTRACT if mobile else ADMISSION_CONTRACT,
        "auditor_files": MOBILE_AUDITOR_FILES if mobile else AUDITOR_FILES,
        "base_checks": MOBILE_BASE_CHECKS if mobile else BASE_CHECKS}


def artifact_protocol(value, role):
    """Missing task IDs remain compatible only for the original Handover schema."""
    mobile_contract = {"pack": MOBILE_CONTRACT, "selection": MOBILE_SELECTION_CONTRACT}[role]
    task = value.get("task_id", HANDOVER_TASK_ID)
    protocol = task_protocol(task)
    require(value.get("contract") == protocol[role + "_contract"]
            and (value.get("contract") != mobile_contract or value.get("task_id") == MOBILE_TASK_ID),
            "Task identity and " + role + " contract differ")
    return protocol


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda word:
        (_ for _ in ()).throw(ValueError("Nonfinite JSON: " + word)))


def jsonl(path):
    rows = [json.loads(line, parse_constant=lambda word:
        (_ for _ in ()).throw(ValueError("Nonfinite JSON: " + word)))
        for line in Path(path).read_text().splitlines() if line.strip()]
    require(all(isinstance(row, dict) for row in rows), "JSONL rows must be objects")
    return rows


def valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


class Bindings:
    """Check every authority byte once; refuse changes before publication."""
    def __init__(self):
        self.files = {}

    @staticmethod
    def stat(path):
        s = path.stat()
        return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns

    def verify(self, path, expected):
        path = Path(path)
        require(path.is_absolute() and path.is_file() and not path.is_symlink()
                and valid_sha(expected), "Missing/unsafe SHA-bound evidence: " + str(path))
        path = path.resolve(strict=True)
        before = self.stat(path)
        previous = self.files.get(path)
        require(previous is None or previous[0] == expected, "Conflicting evidence SHA")
        if previous != (expected, before):
            require(sha(path) == expected and self.stat(path) == before,
                    "Evidence SHA changed: " + str(path))
        self.files[path] = expected, before
        return path

    def verify_all(self, values):
        require(isinstance(values, dict) and values, "Missing independent evidence bindings")
        for path, expected in values.items():
            self.verify(path, expected)

    def unchanged(self):
        for path, (_, before) in self.files.items():
            require(path.is_file() and not path.is_symlink() and self.stat(path) == before,
                    "Evidence changed while packaging: " + str(path))


def _collection(directory, identity, outcome):
    require(not (directory/"operator_discard.json").exists()
            and not (directory/"operator_discard.json").is_symlink(), "Operator discarded episode")
    require(not discard_requested(directory), "Operator discard intent is pending")
    path = legacy.owned(directory, "collection_record.json")
    record = read(path)
    require(record.get("contract") == "cvpr-pico-episode-collection-record-v1"
            and record.get("status") == "committed" and record.get("disposition") == "kept"
            and record.get("human_demo_uuid") == identity and record.get("source_kind") == "live_pico_human"
            and record.get("episode_json_sha256") == sha(directory/"episode.json"),
            "Require the same committed-kept live-PICO collection record")
    require(outcome.get("state", {}).get("status") == "pending_audit"
            and outcome["state"].get("human_demo_uuid") == identity
            and outcome["state"].get("pending_audit") is True,
            "Episode was not sealed for independent audit")
    source = Path(record.get("source_episode_directory", ""))
    require(source.name == identity and source.parent.name == "episodes"
            and source.parent.parent.name, "Collection lacks its original session identity")
    return record, source.parent.parent.name


def load_selection(selection, bindings):
    path = Path(selection).resolve(strict=True)
    bindings.verify(path, sha(path)); chosen = read(path)
    protocol = artifact_protocol(chosen, "selection")
    require(isinstance(chosen.get("episodes"), list) and chosen["episodes"],
            "Require a frozen nonempty PICO admissions selection")
    items, identities, resolved = [], set(), set()
    for entry in chosen["episodes"]:
        identity = str(uuid.UUID(entry["human_demo_uuid"]))
        require(identity == entry["human_demo_uuid"] and identity not in identities,
                "Repeated or noncanonical live-PICO UUID")
        directory = Path(entry["episode_dir"]).resolve(strict=True)
        require(directory.is_dir() and directory not in resolved, "Repeated physical episode directory")
        identities.add(identity); resolved.add(directory)
        admission = legacy.owned(directory, "admission.json")
        require(Path(entry["admission_path"]).resolve(strict=True) == admission,
                "Selected admission is not inside its episode")
        bindings.verify(admission, entry["admission_sha256"])
        audit = read(admission)
        require(audit.get("contract") == protocol["admission_contract"]
                and audit.get("status") == "independent_live_human_episode_audit_passed"
                and audit.get("eligible_for_packing") is True and audit.get("accepted_new_dataset") is False
                and audit.get("ckpt_feature_pack_ready") is False and audit.get("errors") == []
                and audit.get("human_demo_uuid") == identity,
                "Only an independently admitted actual live-PICO episode may be packaged")
        if protocol["mobile"]:
            require(audit.get("task_id") == MOBILE_TASK_ID and audit.get("selection_policy") == "native_only"
                    and audit.get("technical_blocked") is False,
                    "Mobile audit must bind its explicit native-only task identity without technical blockers")
            trace = audit.get("mobile_reference_audit", {}).get("control_trace", {})
            require(isinstance(trace, dict) and isinstance(trace.get("path"), str)
                    and valid_sha(trace.get("sha256"))
                    and audit.get("bindings", {}).get(trace["path"]) == trace["sha256"],
                    "Mobile audit must bind its original shared session control trace")
            historical = audit.get("mobile_reference_audit", {}).get("historical_sources", {})
            require(isinstance(historical, dict) and isinstance(historical.get("manifest_path"), str)
                    and valid_sha(historical.get("manifest_sha256"))
                    and audit.get("bindings", {}).get(historical["manifest_path"]) == historical["manifest_sha256"]
                    and isinstance(historical.get("files"), list) and historical["files"],
                    "Mobile audit must bind its capture-session source snapshots and manifest")
            for source in historical["files"]:
                require(isinstance(source, dict) and isinstance(source.get("path"), str)
                        and isinstance(source.get("filename"), str)
                        and Path(source["filename"]).name == source["filename"]
                        and source["filename"] not in ("", ".", "..")
                        and valid_sha(source.get("sha256"))
                        and audit.get("bindings", {}).get(source["path"]) == source["sha256"],
                        "Mobile original capture implementation is missing or not SHA-bound")
        try:
            stamp = datetime.fromisoformat(audit["generated_utc"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Missing independent audit generation time") from error
        require(stamp.tzinfo is not None, "Audit generation time must be timezone-aware")
        bindings.verify_all(audit.get("bindings"))
        authorities = audit.get("auditor_source_sha256")
        bindings.verify_all(authorities)
        require({Path(p).name for p in authorities} == protocol["auditor_files"],
                "Independent audit lacks the complete auditor source authority")
        for original, expected in authorities.items():
            local = Path(__file__).with_name(Path(original).name)
            require(local.is_file() and sha(local) == expected,
                    "Fresh audit must use this frozen auditor implementation: " + local.name)

        def bound(name):
            owned = legacy.owned(directory, name)
            require(audit["bindings"].get(str(owned)) == sha(owned), "Unbound episode authority: " + name)
            return owned

        metadata, outcome = read(bound("metadata.json")), read(bound("episode.json"))
        quality_name = "raw_quality_samples.jsonl" if protocol["mobile"] else "quality_samples.jsonl"
        for name in ("commands.jsonl", quality_name, "applied_force_samples.jsonl"):
            bound(name)
        require(metadata.get("source_kind") == "live_pico_human"
                and outcome.get("source_kind") == "live_pico_human"
                and metadata.get("replay_source_sha256") is None
                and metadata.get("human_demo_uuid") == identity
                and metadata.get("task_id") == protocol["task_id"],
                "Replay/unknown/mismatched human source is forbidden")
        record, session = _collection(directory, identity, outcome)
        require(entry.get("source_session", session) == session, "Selected human session differs")
        # Collection is an additional gate, independent of the full data audit.
        collection_path = legacy.owned(directory, "collection_record.json")
        bindings.verify(collection_path, sha(collection_path))
        checks = audit.get("checks", {})
        required = set(protocol["base_checks"])
        if metadata.get("render_mode") == "mujoco_deferred":
            required.add("original_full_state_to_deferred_rtx_bridge")
            for name in ("rtx_completion.json", "rtx_frame_bindings.jsonl",
                         "deferred_capture/capture.json", "deferred_capture/events.jsonl",
                         "deferred_capture/states.npy", "deferred_capture/model.mjb"):
                bound(name)
        if not protocol["mobile"] and metadata.get("quality_geometry_mode") == "deferred_geometry_with_original_solved_support":
            required.add("original_solved_support_and_deferred_geometry")
        if metadata.get("capture_start_protocol") is not None:
            required.add("independent_frozen_first_capture_boundary")
            bound("capture_start_receipt.json")
        require(isinstance(checks, dict) and required <= set(checks)
                and all(isinstance(c, dict) and c.get("passed") is True for c in checks.values()),
                "Independent admission checks are missing or failed")
        manifest = read(bound("capture/capture.json"))
        require(manifest.get("contract") == legacy.CAPTURE_CONTRACT
                and manifest.get("status") == "raw_capture_completed_pending_independent_audit"
                and manifest.get("source_kind") == "live_pico_human"
                and manifest.get("human_demo_uuid") == identity
                and manifest.get("warmup_seconds") == 1., "Require complete actual RTX with one real second warmup")
        if protocol["mobile"]:
            require(manifest.get("task_id") == MOBILE_TASK_ID, "Mobile RTX must retain its actual task identity")
        events_path = bound("capture/events.jsonl")
        require(manifest.get("events_sha256") == sha(events_path), "Capture event SHA differs")
        events = jsonl(events_path)
        require(len(events) == manifest.get("event_count") == audit.get("event_count")
                and len(events) >= 60, "Too few actual aligned history/future frames")
        instruction = metadata.get("instruction")
        require(isinstance(instruction, str) and instruction.strip()
                and manifest.get("instruction") == instruction, "Original task instruction is absent or changed")
        start = events[0]["sim_time_s"]
        for index, event in enumerate(events):
            tick = (index*50+29)//30
            require(type(event.get("physical_tick50")) is int and event["physical_tick50"] == tick
                    and event.get("event_index") == index and event.get("capture_index30") == index
                    and abs(event["nominal_elapsed_s"]-index/30) < 1e-9
                    and abs(event["actual_elapsed_s"]-tick/50) < 1e-9
                    and abs(event["sim_time_s"]-(start+tick/50)) < 1e-8,
                    "Actual PICO capture clock is not the dense ceil50Hz/native30Hz grid")
            image = legacy.owned(directory/"capture", event["rgb"]["path"])
            require(audit["bindings"].get(str(image)) == event["rgb"]["file_sha256"],
                    "RTX PNG is not bound by independent admission")
        eligible, histories = legacy.eligible_currents(events)
        items.append(dict(identity=identity, directory=directory, admission=admission,
            admission_sha256=entry["admission_sha256"], audit=audit, metadata=metadata,
            outcome=outcome, collection=record, source_session=session, manifest=manifest,
            capture=directory/"capture", events=events, eligible=eligible, histories=histories,
            instruction=instruction, protocol=protocol))
    return path, items


def _copy(source, destination, expected=None):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    require(sha(destination) == (expected or sha(source)), "Copied evidence bytes changed")


def _shared_copy(source, staging, relative, expected):
    destination = staging/relative
    if destination.exists():
        require(not destination.is_symlink() and sha(destination) == expected,
                "Shared original Mobile evidence changed")
    else:
        _copy(source, destination, expected)
    return {"original_path":str(source), "path":str(relative), "sha256":expected}


def copy_evidence(item, staging):
    directory = item["directory"]
    prefix = staging/"accepted_evidence"/item["identity"]
    destination = prefix/"admission.json"
    _copy(item["admission"], destination, item["admission_sha256"])
    evidence, external = {}, {}
    trace_file = historical_files = None
    copied_shared_sources = set()
    if item["protocol"]["mobile"]:
        trace = item["audit"]["mobile_reference_audit"]["control_trace"]
        trace_file = _shared_copy(trace["path"], staging,
            Path("accepted_evidence/session_traces")/(trace["sha256"]+".jsonl"), trace["sha256"])
        copied_shared_sources.add(trace["path"])
        historical = item["audit"]["mobile_reference_audit"]["historical_sources"]
        manifest = _shared_copy(historical["manifest_path"], staging,
            Path("accepted_evidence/historical_sources")/historical["manifest_sha256"]/"manifest.json",
            historical["manifest_sha256"])
        copied_shared_sources.add(historical["manifest_path"])
        sources = []
        for source in historical["files"]:
            sources.append({"filename":source["filename"], **_shared_copy(source["path"], staging,
                Path("accepted_evidence/historical_sources")/source["sha256"]/source["filename"], source["sha256"])})
            copied_shared_sources.add(source["path"])
        historical_files = {"manifest":manifest, "files":sources}
    sources = dict(item["audit"]["bindings"])
    sources[str(directory/"collection_record.json")] = sha(directory/"collection_record.json")
    for original, expected in sorted(sources.items()):
        if original in copied_shared_sources:
            continue  # One immutable shared copy, never one giant trace per episode.
        path = Path(original)
        if path.is_relative_to(directory):
            if path.suffix.lower() == ".png":
                continue  # Lossless MP4 holds every original RGB byte.
            relative = str(path.relative_to(directory))
            target = prefix/"run"/relative
            _copy(path, target, expected)
            evidence[relative] = str(target.relative_to(staging))
        else:
            if trace_file is None or original != trace_file["original_path"]:
                external[original] = expected  # Model/resource authorities are not duplicated weights.
    authorities = {}
    for original, expected in item["audit"]["auditor_source_sha256"].items():
        target = prefix/"auditor_source"/Path(original).name
        _copy(original, target, expected)
        authorities[original] = {"path": str(target.relative_to(staging)), "sha256": expected}
    return str(destination.relative_to(staging)), evidence, external, authorities, trace_file, historical_files


def verify_raw_values(table, events, state_names, hand_names):
    """Read the produced rows against actual commands and named observations."""
    for index, event in enumerate(events):
        measured = dict(zip(event["joint_names"], event["state43"]))
        expected = np.asarray([measured[name] for name in state_names], np.float32)
        require(np.array_equal(np.asarray(table["observation.state"][index].as_py(), np.float32), expected),
                "Packed current state changed a measured joint")
        if index < 30:
            for key in ("action", "action.body_token_v1_1", "action.neck"):
                require(np.isnan(table[key][index].as_py()).all(), "Warmup gained invented action labels")
            require(not any(table["action_mask"][index].as_py())
                    and table["action_valid"][index].as_py() is False, "Warmup gained action validity")
        else:
            action = event["action"]
            hands = dict(zip(action["hand_joint_names"], action["hand14"]))
            require(np.array_equal(np.asarray(table["action"][index].as_py(), np.float32),
                                   np.asarray([hands[name] for name in hand_names], np.float32))
                    and np.array_equal(np.asarray(table["action.body_token_v1_1"][index].as_py(), np.float32),
                                       np.asarray(action["body_token64"], np.float32)),
                    "Packed labels changed actual executed token or named hand commands")
            require(table["action_mask"][index].as_py() == [True]*78+[False]*2,
                    "Uncontrolled padding gained supervision")


def pack(template, selection, output):
    """Create one new variable-size train-only pack; leave every input intact."""
    import pyarrow.parquet as pq
    template = Path(template).resolve(strict=True)
    output = Path(output)
    require(not output.is_symlink(), "Output must not be an alias")
    output = output.resolve()
    require(not output.exists() and not any((p/".git").exists() for p in (output, *output.parents)),
            "Use a new output directory outside Git")
    bindings = Bindings()
    for name in SOURCE_FILES:
        path = legacy.owned(template, name)
        bindings.verify(path, sha(path))
    identity, statistics = legacy.checkpoint_contract(*(template/name for name in SOURCE_FILES))
    selected_path, items = load_selection(selection, bindings)
    protocol = items[0]["protocol"]
    require(not output.is_relative_to(template)
            and all(not output.is_relative_to(item["directory"]) for item in items),
            "Output must remain outside original episode and template evidence")
    require(len(items) <= 1000, "One PICO pack supports at most one 1000-episode chunk")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name("."+output.name+".partial-"+uuid.uuid4().hex)
    staging.mkdir()
    episodes, receipts, offset, shape, tasks = [], [], 0, None, {}
    try:
        _copy(selected_path, staging/"selection.json", sha(selected_path))
        for name in SOURCE_FILES:
            _copy(template/name, staging/name)
        for index, item in enumerate(items):
            events = item["events"]; identity_uuid = item["identity"]
            values = legacy.episode_values(events, index, offset,
                identity["state_joint_names"], identity["hand_joint_names"])
            task = tasks.setdefault(item["instruction"], len(tasks))
            # This integer is a reader compatibility identifier. UUID and its
            # namespace, rather than this index, define human source lineage.
            source_id = 1_000_000+index
            values.update(task_index=np.full(len(events), task, np.int64),
                task_description=[item["instruction"]]*len(events),
                source_episode_index=np.full(len(events), source_id, np.int64))
            table = legacy.arrow_table(values)
            parquet = staging/f"train/data/chunk-000/episode_{index:06d}.parquet"
            parquet.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, parquet, compression="zstd")
            reread = pq.read_table(parquet)
            legacy.check_parquet_roundtrip(table, reread)
            verify_raw_values(reread, events, identity["state_joint_names"], identity["hand_joint_names"])
            video_path = staging/f"train/videos/chunk-000/{legacy.VIDEO_KEY}/episode_{index:06d}.mp4"
            video = legacy.encode_pngs(item["capture"], events, video_path)
            video["path"] = str(video_path.relative_to(staging))
            if shape is None:
                shape = video["shape"]
            require(shape == video["shape"], "One pack requires consistent actual camera dimensions")
            admission, evidence, external, authorities, trace_file, historical_files = copy_evidence(item, staging)
            source = {"source_kind":"live_pico_human", "source_namespace":"live_pico_human",
                "human_demo_uuid":identity_uuid, "source_session":item["source_session"]}
            episode = {"episode_index":index, "candidate_id":identity_uuid, "human_demo_uuid":identity_uuid,
                "task_id":protocol["task_id"],
                **({"selection_policy":"native_only"} if protocol["mobile"] else {}),
                "tasks":[item["instruction"]], "instruction":item["instruction"], "length":len(events),
                "frames":len(events), "source_kind":"live_pico_human", "source_session":item["source_session"],
                "source_episode_index":source_id, "source_dataset_key":"live_pico_human",
                "source_identity":source, "lineage_group_sources":[source], "lineage_complete":True,
                "split":"train", "split_restriction":"train_only", "history_only_prefix_rows":30,
                "eligible_current_rows":item["eligible"], "label_origin":"actual_live_pico_execution",
                "new_human_demonstration_claim":True}
            episodes.append(episode)
            receipts.append({**episode, "history_rows":item["histories"],
                "original_episode_directory":str(item["directory"]),
                "admission_path":admission, "admission_sha256":item["admission_sha256"],
                "parquet_path":str(parquet.relative_to(staging)), "parquet_sha256":sha(parquet),
                "video":video, "evidence_files":evidence, "external_authority_bindings":external,
                "auditor_source_files":authorities,
                **({"mobile_control_trace_file":trace_file,
                    "mobile_historical_source_files":historical_files} if trace_file is not None else {})})
            offset += len(events)
        meta = staging/"train/meta"
        info = {"codebase_version":"v2.0", "robot_type":"g1_sonic", "fps":30,
            "total_episodes":len(episodes), "total_frames":offset, "total_tasks":len(tasks),
            "total_videos":len(episodes), "total_chunks":1, "chunks_size":1000,
            "splits":{"train":f"0:{len(episodes)}"},
            "data_path":"data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path":"videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features":legacy.features(identity["state_joint_names"], identity["hand_joint_names"], shape),
            "warmup_label_policy":"First30 actual RGB/state remain history; NaN labels/all-false action mask",
            "normalization":{"refitted":False,"inherited_statistics_sha256":identity["statistics"]["sha256"]}}
        legacy.write(meta/"info.json", info)
        for name in ("stats.json", "stats_psi0.json"):
            legacy.write(meta/name, statistics)
        legacy.write(meta/"modality.json", {"state":{"joint_positions":{"start":0,"end":43}},
            "action":{"joint_positions":{"start":0,"end":14}},
            "video":{"egocentric":{"original_key":legacy.VIDEO_KEY}}})
        for name, rows in (("episodes.jsonl", episodes), ("tasks.jsonl",
                [{"task_index":i,"task":task} for task,i in tasks.items()])):
            (meta/name).write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n" for row in rows))
        reader = legacy.verify_lerobot_reader(staging/"train", episodes, [item["events"] for item in items])
        for item in items:
            _collection(item["directory"], item["identity"], item["outcome"])
        bindings.unchanged()
        report = {"contract":protocol["pack_contract"], "task_id":protocol["task_id"],
            **({"selection_policy":"native_only"} if protocol["mobile"] else {}),
            "status":"cpu_lossless_pico_pack_complete_dino_and_production_reader_pending",
            "created_utc":datetime.now(timezone.utc).isoformat(), "source_kind":"live_pico_human",
            "train_payload_packed":True, "final_ready":False, "accepted_new_dataset":False,
            "train_episodes":len(episodes), "total_frames":offset,
            "train_windows":sum(len(e["eligible_current_rows"]) for e in episodes),
            "episodes":receipts, "checkpoint_contract":identity, "normalization_refitted":False,
            "source_contract_files":{name:sha(staging/name) for name in SOURCE_FILES},
            "selection":{"path":"selection.json","sha256":sha(staging/"selection.json")},
            "cpu_reader_validation":reader, "DINO_cache_generated":False,
            "production_reader_verified":False, "training_started":False, "GPU_used":False,
            "physics_steps_performed":0, "tokens_reencoded":False, "existing_VAL_modified":False,
            "source_episode_index_scope":"1000000+local index is reader compatibility only; lineage requires namespace/session/UUID",
            "clock_policy":"Nominal30Hz row/30; physical/source_frame_index=ceil(row*50/30). DINO sidecar source_frame_index must be decoded ordinal row.",
            "final_ready_blockers":["fresh clean DINO cache with inherited encoder identity",
                "actual complete-history/future joint production-reader checks", "remote byte/path/reader checks before A100 use"],
            "packer_sha256":sha(Path(__file__)),
            "payload_sha256":{str(p.relative_to(staging)):sha(p) for p in sorted(staging.rglob("*")) if p.is_file()}}
        legacy.write(staging/"PACK_READY.json", report)
        require(not output.exists(), "Output appeared while packaging")
        staging.rename(output)
        return report
    except BaseException as error:
        legacy.write(staging/"PACK_FAILED.json", {"status":"failed", "error":repr(error),
            "final_ready":False,"accepted_new_dataset":False,"original_data_modified":False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("template", "selection", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    result = pack(args.template, args.selection, args.output)
    print(json.dumps({key:result[key] for key in ("contract", "status", "train_episodes", "total_frames", "train_windows", "final_ready")}))


if __name__ == "__main__":
    main()
