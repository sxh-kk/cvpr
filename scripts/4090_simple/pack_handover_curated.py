"""CPU-only, variable-size curation of admitted Handover captures.

All qualified new captures precede the explicitly selected legacy episodes.
Legacy video bytes and all Parquet columns except three dataset indices stay
unchanged. This contract makes no exact100, minimum50, or readiness claim.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import uuid

import numpy as np

import pack_handover_quality100 as legacy
import pack_handover_random100 as random100

CONTRACT = "cvpr-handover-curated-lerobot-pack-v1"
SELECTION_CONTRACT = "cvpr-handover-curated-selection-v1"
QUEUE_CONTRACT = "cvpr-handover-random100-collection-queue-v1"
REINDEX_COLUMNS = frozenset(("episode_index", "index", "task_index"))
INHERITED_FILES = (*random100.CONTRACT_FILES, "train/meta/stats.json",
                   "train/meta/stats_psi0.json", "train/meta/modality.json")
require = legacy.require


class Bindings:
    """Verify bytes once per unchanged stat identity; recheck after packaging."""

    def __init__(self):
        self.files = {}

    @staticmethod
    def identity(path):
        s = path.stat()
        return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)

    def verify(self, path, expected):
        path = Path(path).resolve(strict=True)
        require(path.is_file() and random100.valid_sha(expected), "Invalid source binding")
        current = self.identity(path)
        previous = self.files.get(str(path))
        require(previous is None or previous[0] == expected, "Conflicting source SHA")
        if previous != (expected, current):
            require(legacy.sha(path) == expected, "Bound bytes changed: " + str(path))
        self.files[str(path)] = (expected, current)
        return path

    def verify_all(self, bindings):
        require(isinstance(bindings, dict) and bindings, "Missing evidence bindings")
        for path, expected in bindings.items():
            self.verify(path, expected)

    def unchanged(self):
        for path, (expected, identity) in list(self.files.items()):
            require(self.identity(Path(path)) == identity, "Input changed during packaging: " + path)
            self.verify(path, expected)


def queue_path(path):
    path = Path(path).resolve(strict=True)
    return path / "queue.json" if path.is_dir() else path


def safe_candidate(candidate):
    require(isinstance(candidate, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,128}", candidate),
            "Unsafe candidate identity")
    return candidate


def validate_selection(selection, entries, old_ready):
    """Curation count is explicit and variable; original exact100 is untouched."""
    require(selection.get("contract") == SELECTION_CONTRACT, "Wrong curation selection contract")
    indices = selection.get("old_episode_indices")
    require(isinstance(indices, list) and all(type(i) is int and 0 <= i < len(old_ready["episodes"]) for i in indices)
            and len(set(indices)) == len(indices), "Repeated or invalid legacy episode")
    require(bool(entries) and len({e["candidate_id"] for e in entries}) == len(entries),
            "Missing or repeated qualified new capture")
    selected_new = selection.get("new14", [])
    require(len(selected_new) == len(entries) and all(all(a.get(k) == b.get(k)
            for k in ("candidate_id", "run", "admission_path", "admission_sha256"))
            for a, b in zip(entries, selected_new)), "Every qualified capture must be retained in queue order")
    old = [old_ready["episodes"][i] for i in indices]
    expected = [("new14", e["candidate_id"]) for e in entries] + [("original100", e["candidate_id"]) for e in old]
    require([(e.get("origin"), e.get("candidate_id")) for e in selection.get("retained", [])] == expected,
            "Selected candidate order/identity differs")
    require([e.get("old_episode_index") for e in selection["retained"] if e["origin"] == "original100"] == indices,
            "Legacy candidate-to-episode mapping differs")
    for key, count in (("new_count", len(entries)), ("old_count", len(old)), ("total_count", len(expected))):
        require(selection.get(key) == count, "Selection count differs: " + key)
    require(len({e[1] for e in expected}) == len(expected), "Repeated capture candidate across origins")
    return indices


def load_new_items(path, bindings):
    from handover_random100_sources import SOURCE_KEY, ARCHIVE_SHA256
    path = queue_path(path); bindings.verify(path, legacy.sha(path)); queue = legacy.read(path)
    require(queue.get("contract") == QUEUE_CONTRACT, "Require the real formal collection queue")
    entries = queue.get("qualified", [])
    require(bool(entries), "No independently qualified new captures")
    for key in ("candidate_id", "run", "admission_path", "admission_sha256", "capture_content_sha256"):
        require(all(row.get(key) for row in entries) and len({row[key] for row in entries}) == len(entries),
                "Repeated/missing qualified " + key)
    items = []
    for entry in entries:
        safe_candidate(entry["candidate_id"])
        require(type(entry.get("source_episode_index")) is int and entry["source_episode_index"] not in random100.EXCLUDED
                and 0 <= entry["source_episode_index"] < 100, "Non-TRAIN raw source")
        admission = bindings.verify(entry["admission_path"], entry["admission_sha256"])
        audit = legacy.read(admission); random100.validate_admission(audit, entry)
        bindings.verify_all(audit["bindings"]); bindings.verify_all(audit["source_sha256"])
        run = Path(entry["run"]).resolve(strict=True)

        def bound(name):
            p = legacy.owned(run, name)
            require(audit["bindings"].get(str(p)) == legacy.sha(p), "Actual run evidence is unbound: " + name)
            return p

        evidence = legacy.read(bound("source_evidence.json")); args = legacy.read(bound("execution_args.json"))
        require(evidence.get("source_dataset_key") == SOURCE_KEY and evidence.get("archive_sha256") == ARCHIVE_SHA256
                and evidence.get("source_id") == entry["source_episode_index"] and evidence.get("split") == "train",
                "Wrong raw namespace/archive/source")
        require(args["source_episode"] == entry["source_episode_index"] and args["seed"] == entry["seed"]
                and args["hand_condition"] == entry["hand_condition"] and args["render_mode"] == "isaac",
                "Qualified actual execution args differ")
        manifest_path = bound("capture/capture.json"); manifest = legacy.read(manifest_path)
        events_path = legacy.owned(manifest_path.parent, manifest["events_path"])
        require(audit["bindings"].get(str(events_path)) == legacy.sha(events_path), "Unbound raw event stream")
        events = random100.jsonl(events_path); rgb_audit = audit["RGB_state_action_history_audit"]
        require(len(events) == manifest["event_count"] == rgb_audit["event_count"]
                and manifest.get("execution_contract") == "cvpr-handover-random100-online-execution-v1"
                and manifest.get("commands_generated_online") is True
                and manifest.get("source_episode_index") == entry["source_episode_index"]
                and manifest.get("scene_proposal_sha256") == entry["scene_proposal_sha256"]
                and rgb_audit["capture_content_sha256"] == entry["capture_content_sha256"], "Actual new capture identity differs")
        eligible, histories = legacy.eligible_currents(events)
        require(eligible == rgb_audit["eligible_current_rows"] and histories == rgb_audit["history_rows"],
                "Admitted new history/future rows changed")
        source = dict(source_dataset_key=SOURCE_KEY, source_episode_index=entry["source_episode_index"],
                      source_parquet_sha256=evidence["parquet_sha256"], archive_sha256=ARCHIVE_SHA256)
        items.append(dict(kind="new_random", entry=entry, audit=audit, admission=admission, run=run,
            capture=manifest_path.parent, events=events, manifest=manifest, source=source,
            eligible=eligible, histories=histories, source_evidence=evidence))
    return path, entries, items


def inherited_contract(old_pack, ready, bindings):
    for name in (*INHERITED_FILES, "train/meta/info.json", "train/meta/episodes.jsonl"):
        bindings.verify(old_pack / name, ready["payload_sha256"].get(name))
    identity, statistics = legacy.checkpoint_contract(*(old_pack / name for name in random100.CONTRACT_FILES))
    require(legacy.read(old_pack / "train/meta/stats.json") == statistics
            and legacy.read(old_pack / "train/meta/stats_psi0.json") == statistics,
            "Legacy package statistics are not the inherited38k statistics")
    return identity


def load_old_items(old_pack, ready, selection, bindings):
    episodes = random100.jsonl(old_pack / "train/meta/episodes.jsonl"); items = []
    for index in selection["old_episode_indices"]:
        receipt, episode = ready["episodes"][index], episodes[index]
        require(receipt["episode_index"] == episode["episode_index"] == index
                and receipt["candidate_id"] == episode["candidate_id"], "Legacy index/candidate differs")
        candidate = safe_candidate(receipt["candidate_id"])
        require(episode.get("split") == "train" and episode.get("split_restriction") == "train_only"
                and episode.get("lineage_complete") is True
                and episode["source_episode_index"] not in random100.HELDOUT,
                "Legacy TRAIN lineage must exclude protected VAL")
        report_path = Path(selection["old_quality_reports"][str(index)]).resolve(strict=True)
        bindings.verify(report_path, selection["source_sha256"].get(str(report_path)))
        report = legacy.read(report_path)
        require(report["episode_index"] == index and report["candidate_id"] == candidate
                and report["source_identity"] == episode["source_identity"]
                and report.get("old_v2_recomputed_exact") is True and report.get("old_v2_passed") is True
                and report.get("unified_v2_and_actual_full50_passed") is True
                and report.get("physics_steps_performed") == 0, "Selected old capture failed unified strict quality")
        random100.validate_full_footprint(report["actual_full_outer_footprint_final50"])
        bindings.verify_all(report["bound_core_evidence_sha256"])
        relative_pq = f"train/data/chunk-000/episode_{index:06d}.parquet"
        relative_video = f"train/videos/chunk-000/{legacy.VIDEO_KEY}/episode_{index:06d}.mp4"
        parquet = bindings.verify(old_pack / relative_pq, ready["payload_sha256"].get(relative_pq))
        video = bindings.verify(old_pack / relative_video, ready["payload_sha256"].get(relative_video))
        require(legacy.sha(parquet) == receipt["parquet_sha256"] == report["parquet_sha256"]
                and legacy.sha(video) == receipt["video"]["sha256"], "Legacy published payload identity differs")
        evidence = old_pack / "accepted_evidence" / candidate
        for name in ("admission.json", "capture.json", "events.jsonl"):
            bindings.verify(evidence / name, ready["payload_sha256"].get(str((evidence / name).relative_to(old_pack))))
        admission = evidence / "admission.json"; audit = legacy.read(admission)
        require(legacy.sha(admission) == receipt["admission_sha256"] == report["admission_sha256"]
                and audit.get("eligible_dataset") is True, "Legacy admitted capture differs")
        events = random100.jsonl(evidence / "events.jsonl")
        eligible, histories = legacy.eligible_currents(events)
        require(eligible == receipt["eligible_current_rows"] == episode["eligible_current_rows"]
                and histories == receipt["history_rows"] and len(events) == receipt["frames"] == episode["length"],
                "Legacy complete history/future changed")
        items.append(dict(kind="legacy_quality100", entry=receipt, episode=episode, audit=audit,
            admission=admission, run=Path(report["actual_RTX_run"]).resolve(strict=True),
            events=events, manifest=legacy.read(evidence / "capture.json"), source=episode["source_identity"],
            eligible=eligible, histories=histories, parquet=parquet, video=video,
            quality_report=report_path, quality=report, old_index=index))
    return items


def reindex_table(table, episode_index, offset, task_index):
    """Only three LeRobot dataset indices may change, including with NaN labels."""
    import pyarrow as pa
    result = table
    for key, values in (("episode_index", np.full(table.num_rows, episode_index, np.int64)),
                        ("index", np.arange(table.num_rows, dtype=np.int64) + offset),
                        ("task_index", np.full(table.num_rows, task_index, np.int64))):
        index = table.schema.get_field_index(key)
        require(index >= 0 and pa.types.is_int64(table.schema.field(index).type), "Dataset index must be int64")
        result = result.set_column(index, table.schema.field(index), pa.array(values))
    check_unmodified_columns(table, result)
    return result


def check_unmodified_columns(original, target):
    require(original.schema == target.schema and original.num_rows == target.num_rows, "Legacy schema/length changed")
    legacy.check_parquet_roundtrip(original.drop_columns(list(REINDEX_COLUMNS)), target.drop_columns(list(REINDEX_COLUMNS)))


def checked_copy(source, destination, expected=None):
    source, destination = Path(source), Path(destination)
    expected = expected or legacy.sha(source)
    destination.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, destination)
    require(legacy.sha(destination) == expected, "Byte-copy changed: " + str(source))
    return str(destination)


def copy_evidence(item, staging):
    candidate = item["entry"]["candidate_id"]
    root = staging / "accepted_evidence" / candidate; root.mkdir(parents=True)
    checked_copy(item["admission"], root / "admission.json", item["entry"]["admission_sha256"])
    files = {}
    # Historical global source paths need not remain current. Actual bound run
    # evidence is copied verbatim; absolute originals stay in the raw admission.
    maps = dict(item["audit"]["bindings"])
    if item["kind"] == "legacy_quality100":
        maps.update(item["quality"]["bound_core_evidence_sha256"])
    for name, expected in maps.items():
        path = Path(name).resolve(strict=True)
        if path.is_relative_to(item["run"]) and path.suffix.lower() != ".png":
            relative = path.relative_to(item["run"])
            destination = root / "run" / relative
            checked_copy(path, destination, expected)
            files[str(relative)] = str(destination.relative_to(staging))
    # Packed legacy raw files are also retained even if its original run copy
    # lacked one of these entries in an older admission revision.
    for name in ("capture.json", "events.jsonl"):
        if item["kind"] == "legacy_quality100":
            source = item["admission"].parent / name
        else:
            source = item["capture"] / name
        destination = root / "run/capture" / name
        checked_copy(source, destination)
        files["capture/" + name] = str(destination.relative_to(staging))
    quality = None
    if item["kind"] == "legacy_quality100":
        destination = root / "quality_report.json"; checked_copy(item["quality_report"], destination)
        quality = str(destination.relative_to(staging))
    # Preserve original raw human payload where it is explicitly SHA-bound,
    # without duplicating archives, rendered PNGs, or model weights.
    raw_sources = []
    raw_sha = item["source"].get("source_parquet_sha256")
    for name, expected in item["audit"].get("source_sha256", {}).items():
        if expected == raw_sha and Path(name).suffix == ".parquet":
            destination = staging / "raw_sources" / f"{expected}.parquet"
            if not destination.exists(): checked_copy(name, destination, expected)
            raw_sources.append(dict(original_path=name, path=str(destination.relative_to(staging)), sha256=expected))
    return dict(admission_path=str((root / "admission.json").relative_to(staging)),
                evidence_files=files, quality_report_path=quality, raw_source_payloads=raw_sources)


def inspect(pack, *, verify_payload=True):
    """Portable structural/label/evidence audit; no external original required."""
    import pyarrow.parquet as pq
    pack = Path(pack).resolve(strict=True); ready = legacy.read(pack / "PACK_READY.json")
    require(ready.get("contract") == CONTRACT and ready.get("train_payload_packed") is True
            and ready.get("normalization_refitted") is False and ready.get("final_ready") is False
            and ready.get("quantity_claim") == "selected_variable_count_not_exact100_or_minimum50",
            "Wrong curated package contract")
    if verify_payload:
        for relative, expected in ready["payload_sha256"].items():
            require(legacy.sha(legacy.owned(pack, relative)) == expected, "Curated payload SHA changed: " + relative)
    episodes = random100.jsonl(pack / "train/meta/episodes.jsonl")
    require(len(episodes) == len(ready["episodes"]) == ready["train_episodes"]
            and [e["episode_index"] for e in episodes] == list(range(len(episodes))), "Curated episode coverage differs")
    identity, _ = legacy.checkpoint_contract(*(pack / name for name in random100.CONTRACT_FILES))
    require(identity["bounds_sha256"] == ready["checkpoint_contract"]["bounds_sha256"], "Normalization bounds changed")
    for name, expected in ready["inherited_file_sha256"].items():
        require(legacy.sha(legacy.owned(pack, name)) == expected, "Inherited statistics/modality changed")
    selection_path = legacy.owned(pack, ready["curation_selection"]["frozen_path"])
    require(legacy.sha(selection_path) == ready["curation_selection"]["sha256"], "Frozen selection changed")
    offset = 0; windows = 0
    for index, (episode, receipt) in enumerate(zip(episodes, ready["episodes"])):
        require(receipt["episode_index"] == index and episode["candidate_id"] == receipt["candidate_id"]
                and episode["source_identity"] == receipt["source_identity"]
                and episode["source_kind"] == receipt["source_kind"], "Curated receipt identity differs")
        admission = legacy.owned(pack, receipt["admission_path"])
        require(legacy.sha(admission) == receipt["admission_sha256"], "Portable admission SHA differs")
        audit = legacy.read(admission)
        if receipt["source_kind"] == "new_random":
            random100.validate_admission(audit, receipt["original_entry"])
        else:
            quality_path = legacy.owned(pack, receipt["quality_report_path"])
            require(legacy.sha(quality_path) == receipt["quality_report_sha256"], "Portable strict quality report changed")
            quality = legacy.read(quality_path)
            require(quality.get("old_v2_passed") is True and quality.get("unified_v2_and_actual_full50_passed") is True,
                    "Legacy unified strict gate failed")
            random100.validate_full_footprint(quality["actual_full_outer_footprint_final50"])
        events = random100.jsonl(legacy.owned(pack, receipt["evidence_files"]["capture/events.jsonl"]))
        eligible, histories = legacy.eligible_currents(events)
        require(eligible == receipt["eligible_current_rows"] == episode["eligible_current_rows"]
                and histories == receipt["history_rows"], "Portable history/future coverage changed")
        parquet = pack / f"train/data/chunk-000/episode_{index:06d}.parquet"
        table = pq.read_table(parquet)
        require(table.num_rows == episode["length"] == receipt["frames"] == len(events), "Packed length differs")
        require(table["episode_index"].to_pylist() == [index] * len(events)
                and table["index"].to_pylist() == list(range(offset, offset + len(events))), "Global/local indices differ")
        expected = legacy.episode_values(events, index, offset, identity["state_joint_names"], identity["hand_joint_names"])
        # This checks exact executed tokens, hand order, physical state/vel/root,
        # NaN warmup, action masks, all clocks and the original future boundary.
        expected = legacy.arrow_table(expected)
        legacy.check_parquet_roundtrip(expected, table.select(expected.column_names))
        offset += len(events); windows += len(eligible)
    require(offset == ready["total_frames"] and windows == ready["train_windows"], "Curated frame/window totals differ")
    return ready, episodes


def pack(new_queue, old_pack, selection_path, output, *, video_workers=1):
    import pyarrow.parquet as pq
    output = Path(output).absolute(); old_pack = Path(old_pack).resolve(strict=True)
    require(not output.exists() and not output.is_symlink() and not output.resolve().is_relative_to(legacy.REPO)
            and not output.resolve().is_relative_to(old_pack), "Use a fresh external curated output")
    require(type(video_workers) is int and 1 <= video_workers <= 4, "Use one to four CPU video workers")
    bindings = Bindings(); selection_path = Path(selection_path).resolve(strict=True)
    bindings.verify(selection_path, legacy.sha(selection_path)); selection = legacy.read(selection_path)
    old_ready_path = old_pack / "PACK_READY.json"; bindings.verify(old_ready_path, legacy.sha(old_ready_path))
    old_ready = legacy.read(old_ready_path)
    require(old_ready.get("contract") == legacy.CONTRACT and old_ready.get("train_payload_packed") is True,
            "Require the immutable legacy quality100 package")
    bindings.verify_all(selection["source_sha256"])
    require(Path(selection["old_pack"]).resolve() == old_pack and queue_path(selection["new_queue"]) == queue_path(new_queue),
            "Selection belongs to different source roots")
    queue_file, entries, new_items = load_new_items(new_queue, bindings)
    validate_selection(selection, entries, old_ready)
    identity = inherited_contract(old_pack, old_ready, bindings)
    old_items = load_old_items(old_pack, old_ready, selection, bindings); items = new_items + old_items
    require(all(not output.resolve().is_relative_to(item["run"]) for item in items), "Output overlaps an actual capture")
    staging = output.parent / f".{output.name}.partial-{uuid.uuid4().hex}"; staging.mkdir(parents=True)
    root = staging / "train"; episodes, receipts, tasks = [], [], {}; offset = 0
    try:
        checked_copy(selection_path, staging / "selection.json")
        checked_copy(queue_file, staging / "provenance/new_queue.json")
        checked_copy(old_ready_path, staging / "provenance/old_PACK_READY.json")
        checked_copy(selection["old_quality_audit"], staging / "provenance/old_quality_audit.json",
                     selection["old_quality_audit_sha256"])
        videos = {}
        with ThreadPoolExecutor(max_workers=video_workers) as pool:
            futures = {number: pool.submit(legacy.encode_pngs, item["capture"], item["events"],
                root / f"videos/chunk-000/{legacy.VIDEO_KEY}/episode_{number:06d}.mp4")
                for number, item in enumerate(new_items)}
            for number, future in futures.items():
                videos[number] = future.result()
                print(json.dumps(dict(stage="new_video_verified", episode_index=number, frames=videos[number]["frames"])), flush=True)
        for number, item in enumerate(items):
            entry, events = item["entry"], item["events"]
            instruction = item["manifest"].get("instruction")
            require(isinstance(instruction, str) and instruction.strip(), "Missing actual task instruction")
            task_index = tasks.setdefault(instruction, len(tasks))
            if item["kind"] == "new_random":
                values = legacy.episode_values(events, number, offset, identity["state_joint_names"], identity["hand_joint_names"])
                values.update(task_index=np.full(len(events), task_index, np.int64), task_description=[instruction] * len(events),
                              source_episode_index=np.full(len(events), entry["source_episode_index"], np.int64))
                table = legacy.arrow_table(values); video = videos[number]
                episode = dict(tasks=[instruction], candidate_id=entry["candidate_id"],
                    source_episode_index=entry["source_episode_index"], source_dataset_key=item["source"]["source_dataset_key"],
                    source_identity=item["source"], lineage_group_sources=[item["source"]], lineage_complete=True,
                    split="train", split_restriction="train_only", history_only_prefix_rows=30,
                    seed=entry["seed"], hand_condition=entry["hand_condition"], scene_proposal_sha256=entry["scene_proposal_sha256"],
                    label_origin="actual_new_online_expert_execution", new_human_demonstration_claim=False)
                origin = dict(episode_index=number, run=str(item["run"]), admission_path=str(item["admission"]))
            else:
                original = pq.read_table(item["parquet"]); table = reindex_table(original, number, offset, task_index)
                video = deepcopy(entry["video"])
                checked_copy(item["video"], root / f"videos/chunk-000/{legacy.VIDEO_KEY}/episode_{number:06d}.mp4", video["sha256"])
                episode = deepcopy(item["episode"])
                origin = dict(episode_index=item["old_index"], pack_root=str(old_pack), parquet_path=str(item["parquet"]),
                              video_path=str(item["video"]), source_parquet_sha256=legacy.sha(item["parquet"]),
                              source_video_sha256=legacy.sha(item["video"]))
            parquet = root / f"data/chunk-000/episode_{number:06d}.parquet"; parquet.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, parquet, compression="zstd"); legacy.check_parquet_roundtrip(table, pq.read_table(parquet))
            if item["kind"] == "legacy_quality100": check_unmodified_columns(original, pq.read_table(parquet))
            evidence = copy_evidence(item, staging)
            episode.update(episode_index=number, length=len(events), source_kind=item["kind"], origin=origin,
                           eligible_current_rows=item["eligible"])
            receipt = dict(episode_index=number, candidate_id=entry["candidate_id"], source_kind=item["kind"],
                source_identity=episode["source_identity"], source_dataset_key=episode["source_dataset_key"],
                source_episode_index=episode["source_episode_index"], origin=origin, original_entry=entry,
                admission_sha256=entry["admission_sha256"], capture_content_sha256=entry["capture_content_sha256"],
                frames=len(events), eligible_current_rows=item["eligible"], history_rows=item["histories"],
                video=video, parquet_sha256=legacy.sha(parquet), **evidence)
            if item["kind"] == "legacy_quality100":
                receipt["quality_report_sha256"] = legacy.sha(item["quality_report"])
                receipt["unchanged_legacy_columns"] = sorted(set(table.column_names) - REINDEX_COLUMNS)
            episodes.append(episode); receipts.append(receipt); offset += len(events)
            print(json.dumps(dict(stage="episode_packed", episode_index=number, source_kind=item["kind"])), flush=True)
        meta = root / "meta"; meta.mkdir(parents=True)
        info = legacy.read(old_pack / "train/meta/info.json")
        require(all(e["video"]["shape"] == info["features"][legacy.VIDEO_KEY]["shape"] for e in receipts), "Actual RGB shape differs")
        count = len(episodes)
        info.update(total_episodes=count, total_frames=offset, total_videos=count, total_tasks=len(tasks), splits={"train": f"0:{count}"})
        legacy.write(meta / "info.json", info)
        for name in INHERITED_FILES: checked_copy(old_pack / name, staging / name, old_ready["payload_sha256"][name])
        (meta / "episodes.jsonl").write_text("".join(json.dumps(e, ensure_ascii=False, allow_nan=False) + "\n" for e in episodes))
        (meta / "tasks.jsonl").write_text("".join(json.dumps({"task_index": i, "task": t}) + "\n" for t, i in tasks.items()))
        reader = legacy.verify_lerobot_reader(root, episodes, [item["events"] for item in items])
        bindings.unchanged()
        report = dict(contract=CONTRACT, status="cpu_lossless_variable_curated_pack_complete",
            created_utc=datetime.now(timezone.utc).isoformat(), train_episodes=count, new_episodes=len(new_items),
            legacy_episodes=len(old_items), total_frames=offset, train_windows=sum(len(i["eligible"]) for i in items),
            episodes=receipts, checkpoint_contract=identity, normalization_refitted=False,
            inherited_file_sha256={name: old_ready["payload_sha256"][name] for name in INHERITED_FILES},
            curation_selection=dict(path=str(selection_path), sha256=legacy.sha(selection_path), frozen_path="selection.json"),
            new_queue_binding=dict(path=str(queue_file), sha256=legacy.sha(queue_file), qualified_count=len(new_items)),
            old_pack_binding=dict(path=str(old_pack), pack_ready_sha256=legacy.sha(old_ready_path),
                                  selected_episode_indices=selection["old_episode_indices"]),
            cpu_reader_validation=reader, train_payload_packed=True, final_ready=False,
            final_ready_blockers=["real_DINO_cache_pending", "production_training_reader_preflight_pending"],
            quantity_claim="selected_variable_count_not_exact100_or_minimum50", independent_policy_TEST_certified=False,
            cross_namespace_unique_ancestor_count_claimed=False, new_human_demonstration_claim=False,
            DINO_cache_generated=False, training_started=False, physics_steps_performed=0, GPU_used=False,
            packer_sha256=legacy.sha(__file__))
        report["payload_sha256"] = {str(p.relative_to(staging)): legacy.sha(p) for p in sorted(staging.rglob("*")) if p.is_file()}
        legacy.write(staging / "PACK_READY.json", report)
        inspect(staging, verify_payload=False)
        staging.rename(output)
        return report
    except Exception as error:
        legacy.write(staging / "FAILED.json", dict(error=str(error), final_ready=False)); raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("new-queue", "old-pack", "selection", "output"):
        parser.add_argument("--" + flag, required=True, type=Path)
    parser.add_argument("--video-workers", type=int, default=1)
    args = parser.parse_args()
    report = pack(args.new_queue, args.old_pack, args.selection, args.output, video_workers=args.video_workers)
    print(json.dumps({key: report[key] for key in ("status", "train_episodes", "total_frames", "train_windows", "final_ready")}))


if __name__ == "__main__":
    main()
