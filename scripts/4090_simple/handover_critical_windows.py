"""Build a TRAIN-only, immutable Handover critical-transition sampling sidecar.

This indexes existing successful demonstrations. It creates neither new expert
actions nor recovery demonstrations and never changes a running training job.
Window identities are (episode_index, current_row), NOT Dataset/Parquet indices.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


CONTRACT = "cvpr-handover-critical-windows-v2"
FAILURES = {"lift": 8, "handover": 1, "placement": 17}
FOCUS_POLICY_REVISION = "critical-conversions-v3"
FOCUS_EVENTS = {
    "lift": ("right_lift", "lift_10cm"),
    "handover": ("both_hands", "left_only"),
    "placement": ("placement_alignment", "container_supported", "released_supported"),
}
FAILURE_COUNT_PROVENANCE = {
    "source": "user_reported_aggregate",
    "split": "unverified",
    "trace_bindings_available": False,
    "used_for_train_sampling_priorities": True,
    "independent_blind_test_claim": False,
}
HELDOUT = frozenset({39, 65, 80, 92, 94})
REPO = Path(__file__).resolve().parents[2]
EPS = 1e-7


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def finite(value, name):
    require(type(value) in (int, float) and math.isfinite(value), f"Invalid finite {name}")
    return float(value)


def _jsonl(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _inside(root, relative):
    relative = Path(relative)
    require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe packed relative path")
    target = (root / relative).resolve(strict=True)
    require(target.is_file() and target.is_relative_to(root), "Packed input escapes dataset root")
    return target


def _bound(path, expected, receipts):
    path = Path(path).resolve(strict=True)
    require(path.is_file() and sha256(path) == expected, f"Bound source SHA changed: {path}")
    receipts[str(path)] = expected
    return path


def classify_quality(samples, initial_target_z_m):
    """Causally annotate real 50 Hz observations and confirmed transitions.

    A transition is timestamped at its confirming observation, never backdated.
    Five consecutive observations confirm stages/contacts; 50 confirm stability.
    No future sample participates in any returned observation's state label.
    These descriptive annotations do not replace/redefine the admission gate.
    """
    initial_z = finite(initial_target_z_m, "initial target height")
    phases, events = [], []
    streak = Counter()
    emitted = set()
    right_seen = both_seen = left_seen = False
    stable_history = []
    previous_time = None
    for index, sample in enumerate(samples):
        require(sample.get("completed_steps") == index + 1, "Quality steps must be consecutive from1")
        time = finite(sample.get("sim_time_s"), "quality time")
        if previous_time is not None:
            require(abs(time - previous_time - .02) <= EPS, "Quality clock is not native50Hz")
        previous_time = time
        require(sample.get("solver_stage") == "post_mj_step_before_observation_refresh",
                "Require recorded solved-state physical measurements")
        contacts = sample.get("contacts", {})
        require(all(type(contacts.get(k)) is bool for k in ("left_hand", "right_hand", "container")),
                "Quality contact measurements are incomplete")
        xyz = sample.get("actual_target_xyz_m")
        require(isinstance(xyz, list) and len(xyz) == 3, "Quality target XYZ is missing")
        xyz = [finite(v, "target XYZ") for v in xyz]
        left, right = contacts["left_hand"], contacts["right_hand"]
        geometry = sample.get("container_geometry", {})
        for key in ("inside_container_footprint", "collision_vertical_ranges_overlap"):
            require(type(geometry.get(key)) is bool, "Quality3D geometry measurements are incomplete")
        up = finite(geometry.get("container_upright_cosine"), "container upright cosine")
        support = finite(geometry.get("container_support_weight_fraction"), "solved support fraction")
        require(geometry.get("support_measurement") == "live_solved_mujoco_contact_force",
                "Support must be a recorded solved force, not reconstructed pose evidence")
        lifted = xyz[2] - initial_z >= .05
        aligned = geometry["inside_container_footprint"] and geometry["collision_vertical_ranges_overlap"] and up >= math.cos(math.radians(30))
        supported = aligned and contacts["container"] and support >= .5
        released = supported and not left and not right
        conditions = {
            "right_lift": right and not left and lifted,
            "lift_10cm": right and not left and xyz[2] - initial_z >= .1,
            "both_hands": right_seen and left and right and lifted,
            "left_only": both_seen and left and not right and lifted,
            "placement_alignment": left_seen and aligned,
            "container_supported": left_seen and supported,
            "released_supported": left_seen and released,
        }
        for key, active in conditions.items():
            streak[key] = streak[key] + 1 if active else 0
        for category, keys in FOCUS_EVENTS.items():
            for key in keys:
                if streak[key] >= 5 and key not in emitted:
                    events.append({"stage": category, "event": key, "sim_time_s": time,
                                   "completed_steps": index + 1, "sampling_focus": True})
                    emitted.add(key)
        right_seen |= streak["right_lift"] >= 5
        both_seen |= streak["both_hands"] >= 5
        left_seen |= streak["left_only"] >= 5
        if left_seen and aligned and not left and not right:
            stable_history.append((xyz, supported, contacts["container"]))
            stable_history = stable_history[-50:]
        else:
            stable_history = []
        stable_positions = [row[0] for row in stable_history]
        stable = (len(stable_history) == 50
                  and sum(row[1] for row in stable_history) >= 49
                  and sum(row[2] for row in stable_history) >= 49
                  and max(math.dist(a, b) for i, a in enumerate(stable_positions)
                          for b in stable_positions[i + 1:]) < .005)
        if stable and "stable_placement" not in emitted:
            events.append({"stage": "placement", "event": "stable_placement", "sim_time_s": time,
                           "completed_steps": index + 1, "sampling_focus": False})
            emitted.add("stable_placement")
        phase = ("stable_placement" if stable else "released_supported" if left_seen and released
                 else "supported_held" if left_seen and supported else "post_handover" if left_seen
                 else "both_hands" if both_seen else "lifted_right" if right_seen else "pre_lift")
        phases.append({"sim_time_s": time, "observed_stage": phase, "completed_steps": index + 1})
    require(bool(phases), "No recorded quality measurements")
    require(all(k in emitted for k in ("right_lift", "both_hands", "left_only", "released_supported", "stable_placement")),
            "Admitted episode lacks confirmed lifted handover/released placement transitions")
    return phases, events


def annotate_windows(episode_index, source_episode_index, eligible, capture_events, samples,
                     initial_target_z_m, *, lead_seconds=.5, trail_seconds=.5):
    """Return immutable keys and offline sampling tags; never model inputs."""
    lead, trail = finite(lead_seconds, "transition lead"), finite(trail_seconds, "transition trail")
    require(0 <= lead <= 1 and 0 <= trail <= 1, "Transition context must lie within0..1seconds")
    phases, transitions = classify_quality(samples, initial_target_z_m)
    times = [p["sim_time_s"] for p in phases]
    output = []
    for current in eligible:
        time = finite(capture_events[current]["sim_time_s"], "current time")
        observed = bisect_right(times, time + EPS) - 1
        phase = phases[observed]["observed_stage"] if observed >= 0 else "warmup_complete"
        tags = sorted({e["stage"] for e in transitions
                       if e["event"] in FOCUS_EVENTS.get(e["stage"], ()) and e["sampling_focus"]
                       if e["sim_time_s"] - lead - EPS <= time <= e["sim_time_s"] + trail + EPS})
        output.append({"episode_index": episode_index, "current_row": current,
                       "source_episode_index": source_episode_index, "sim_time_s": time,
                       "observed_stage": phase, "critical_stages": tags})
    return output, transitions


def sampling_weights(rows, *, ancestor_weights=None, full_trajectory_fraction=.5,
                     failure_counts=None):
    """50% full TRAIN trajectory +50% critical events; balance ancestry in each.

    Category mixture uses split-unverified user-reported aggregate counts8/1/17.
    Their evaluation split is unbound, so this is no independent blind-TEST claim
    or claim that these successful demonstrations have those failure rates.
    Overlap is additive; its component probabilities are explicitly retained.
    """
    require(bool(rows), "Sampling requires TRAIN windows")
    fraction = finite(full_trajectory_fraction, "full-trajectory mixture")
    require(0 < fraction < 1, "Both full trajectory and focus must have positive mass")
    failures = dict(FAILURES if failure_counts is None else failure_counts)
    require(set(failures) == set(FAILURES) and all(type(v) is int and v > 0 for v in failures.values()),
            "All three failure categories need positive integer counts")
    ancestors = sorted({r["source_episode_index"] for r in rows})
    require(not set(ancestors) & HELDOUT, "Held-out ancestry cannot supply TRAIN weights")
    requested = {a: 1.0 for a in ancestors} if ancestor_weights is None else {int(a): v for a, v in ancestor_weights.items()}
    require(set(requested) == set(ancestors), "Ancestor weights must exactly cover TRAIN ancestors")
    for value in requested.values():
        require(finite(value, "ancestor weight") > 0, "Ancestor weights must be positive")
    total = sum(requested.values())
    require(math.isfinite(total), "Ancestor weight sum must be finite")
    requested = {a: float(v) / total for a, v in requested.items()}
    all_counts = Counter(r["source_episode_index"] for r in rows)
    critical = {stage: Counter(r["source_episode_index"] for r in rows if stage in r["critical_stages"])
                for stage in failures}
    require(all(critical.values()), "Missing critical category; do not silently renormalize failure priorities")
    category_mass = {stage: (1 - fraction) * count / sum(failures.values()) for stage, count in failures.items()}
    normalizers = {stage: sum(requested[a] for a in counts) for stage, counts in critical.items()}
    weighted = []
    identities = set()
    for row in rows:
        identity = (row["episode_index"], row["current_row"])
        require(identity not in identities, "Duplicate episode/current sampling identity")
        identities.add(identity)
        ancestor = row["source_episode_index"]
        require(set(row["critical_stages"]) <= set(failures), "Unknown critical stage")
        base = fraction * requested[ancestor] / all_counts[ancestor]
        focus = {stage: category_mass[stage] * requested[ancestor] / normalizers[stage] / critical[stage][ancestor]
                 for stage in row["critical_stages"]}
        weight = base + sum(focus.values())
        require(math.isfinite(weight) and weight > 0, "Every sampling weight must be finite positive")
        weighted.append({**row, "weight": weight, "base_weight": base, "focus_weights": focus})
    require(math.isclose(sum(r["weight"] for r in weighted), 1, abs_tol=1e-10), "Sampling probabilities do not sum to1")
    return weighted, {
        "full_trajectory_fraction": fraction, "critical_fraction": 1 - fraction,
        "user_reported_failure_counts": failures, "failure_count_provenance": dict(FAILURE_COUNT_PROVENANCE),
        "critical_category_probability_mass": category_mass,
        "ancestor_policy": "weighted ancestry balance within full trajectory and independently within each available critical category",
        "normalized_requested_ancestor_weights": {str(k): v for k, v in requested.items()},
        "effective_ancestor_probability_mass": {str(a): sum(r["weight"] for r in weighted if r["source_episode_index"] == a) for a in ancestors},
        "critical_windows_by_stage": {k: sum(v.values()) for k, v in critical.items()},
        "critical_ancestors_by_stage": {k: sorted(v) for k, v in critical.items()},
        "overlap_policy": "one window may receive additive focus probability from several transition categories",
    }


def _verify_events(events, expected_count):
    require(len(events) == expected_count and len(events) >= 60, "Capture length differs from packed episode")
    origin = finite(events[0].get("sim_time_s"), "capture time origin")
    for index, event in enumerate(events):
        tick = (index * 50 + 29) // 30
        require(event.get("capture_index30") == index and event.get("event_index") == index
                and event.get("physical_tick50") == tick, "Capture30Hz/50Hz row mapping changed")
        require(abs(finite(event.get("sim_time_s"), "capture time") - origin - tick / 50) <= EPS,
                "Capture time is inconsistent with actual physics tick")
        require(abs(finite(event.get("nominal_elapsed_s"), "nominal time") - index / 30) <= 1e-5,
                "Nominal30Hz clock changed")
        require(type(event.get("action_valid")) is bool and event["action_valid"] == (index >= 30),
                "Warmup/real-action validity changed")
        if index < 30:
            require(event.get("action") is None, "Warmup must not invent action labels")
        else:
            action = event.get("action", {})
            for key, width in (("body_token64", 64), ("hand14", 14)):
                vector = action.get(key)
                require(isinstance(vector, list) and len(vector) == width, "Real teacher action layout changed")
                for value in vector:
                    finite(value, key)


def build_sidecar(dataset_root, output, *, ancestor_weights=None, lead_seconds=.5, trail_seconds=.5):
    """Verify source hashes, production windows and clocks before writing outside source/repo."""
    dataset_root, output = Path(dataset_root).resolve(strict=True), Path(output).resolve()
    train = dataset_root if (dataset_root / "meta/episodes.jsonl").exists() else dataset_root / "train"
    pack = train.parent
    require(not output.is_relative_to(pack) and not output.is_relative_to(REPO),
            "Output must be outside the original dataset and repository")
    require(not output.exists() or (output.is_dir() and not any(output.iterdir())), "Use a fresh output directory")
    ready_path = pack / "PACK_READY.json"
    ready = read_json(ready_path)
    require(ready.get("contract") == "cvpr-handover-quality100-lerobot-pack-v1"
            and ready.get("train_target_satisfied") is True, "Require a complete admitted TRAIN pack")
    require(set(ready.get("existing_VAL_sources_excluded", [])) >= HELDOUT, "Original VAL exclusion contract missing")
    require(ready.get("boundary_policy", {}).get("tail_padding_permitted") is False,
            "Complete real future/no-padding contract missing")
    payload = ready["payload_sha256"]
    receipts = {str(ready_path): sha256(ready_path)}
    def packed(relative):
        require(relative in payload, f"Packed input is not hash-bound: {relative}")
        return _bound(_inside(pack, relative), payload[relative], receipts)
    metadata_path = packed("train/meta/episodes.jsonl")
    episodes = _jsonl(metadata_path)
    require(len(episodes) == ready["train_episodes"] == len(ready["episodes"]), "TRAIN episode count changed")
    entries = {e["episode_index"]: e for e in ready["episodes"]}
    require(len(entries) == len(episodes), "Duplicate packed episode identity")
    import pack_handover_quality100 as packer
    rows, episode_reports = [], []
    for episode in episodes:
        index, ancestor, candidate = episode["episode_index"], episode["source_episode_index"], episode["candidate_id"]
        lineage = episode.get("lineage_group_sources", [])
        require(episode.get("split") == "train" and episode.get("split_restriction") == "train_only"
                and episode.get("lineage_complete") is True and ancestor not in HELDOUT,
                "Only complete TRAIN ancestry may be indexed")
        require(len(lineage) == 1 and lineage[0].get("source_episode_index") == ancestor,
                "Ambiguous/held-out mixed ancestry is not admitted")
        entry = entries[index]
        require(candidate == entry["candidate_id"] and episode["eligible_current_rows"] == entry["eligible_current_rows"],
                "Packed metadata/accepted windows differ")
        prefix = f"accepted_evidence/{candidate}"
        admission_path, capture_path, events_path = (packed(prefix + "/" + name)
                                                    for name in ("admission.json", "capture.json", "events.jsonl"))
        admission, capture, events = read_json(admission_path), read_json(capture_path), _jsonl(events_path)
        require(admission.get("eligible_dataset") is True and admission.get("candidate_id") == candidate
                and admission.get("source_episode_index") == ancestor and sha256(admission_path) == entry["admission_sha256"],
                "Accepted admission identity changed")
        require(capture.get("candidate_id") == candidate and capture.get("source_episode_index") == ancestor
                and capture.get("events_sha256") == receipts[str(events_path)]
                and admission.get("capture_content_sha256") == entry.get("capture_content_sha256"),
                "Original capture identity/events hash changed")
        bindings = admission["bindings"]
        require(receipts[str(capture_path)] in [v for p, v in bindings.items() if p.endswith("/capture/capture.json")],
                "Original capture SHA was not independently bound")
        quality_bindings = [(Path(p), digest) for p, digest in bindings.items() if p.endswith("/quality_samples.jsonl")]
        require(len(quality_bindings) == 1, "Require one independently bound native50Hz quality stream")
        quality_path = _bound(*quality_bindings[0], receipts)
        samples = _jsonl(quality_path)
        gate = admission["physical_gate"]
        require(gate.get("verified_handover_success") is True
                and gate.get("final_placement_geometric_validation", {}).get("require_3d_evidence") is True,
                "Require accepted lifted handover and actual3D placement evidence")
        _verify_events(events, episode["length"])
        require(len(samples) == gate["contact_evidence"]["sample_count"], "Original50Hz sample count changed")
        require(abs(samples[0]["sim_time_s"] - events[30]["sim_time_s"] - .02) <= EPS,
                "Teacher quality and captured warmup clocks differ")
        eligible, histories = packer.eligible_currents(events)
        require(eligible == episode["eligible_current_rows"] and histories == entry["history_rows"],
                "Production full-history/complete-future window admission changed")
        episode_rows, transitions = annotate_windows(index, ancestor, eligible, events, samples,
            capture["initial_target_z_m"], lead_seconds=lead_seconds, trail_seconds=trail_seconds)
        rows.extend(episode_rows)
        episode_reports.append({"episode_index": index, "candidate_id": candidate,
            "source_episode_index": ancestor, "eligible_windows": len(eligible), "transitions": transitions,
            "admission_sha256": receipts[str(admission_path)], "capture_sha256": receipts[str(capture_path)],
            "quality_source": {"path": str(quality_path), "sha256": receipts[str(quality_path)]}})
    weighted, mixture = sampling_weights(rows, ancestor_weights=ancestor_weights)
    # Close the source snapshot before emitting a usable sidecar.
    require(all(sha256(path) == digest for path, digest in receipts.items()), "Source changed during indexing; retry")
    output.mkdir(parents=True, exist_ok=True)
    rows_path = output / "rows.jsonl"
    with rows_path.open("w") as handle:
        for row in weighted:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
    weights_digest = hashlib.sha256(json.dumps(
        [[r["episode_index"], r["current_row"], r["weight"]] for r in weighted],
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    metadata = {
        "contract": CONTRACT, "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "TRAIN-only reindexing of existing admitted successful trajectories, not new demonstrations or recovery data",
        "status": "cpu_verified_sidecar_ready_for_explicit_new_training_integration",
        "source_dataset": str(train), "source_pack_ready_sha256": receipts[str(ready_path)],
        "source_metadata_sha256": receipts[str(metadata_path)], "window_count": len(weighted),
        "rows_sha256": sha256(rows_path), "weights_digest": weights_digest,
        "window_identity": ["episode_index", "current_row"],
        "production_dataset_index_mapping_required": True,
        "time_semantics": {"capture": "same native30Hz RGB/state/command event, actual tick=ceil(row*50/30)",
            "quality": "real solved-state50Hz observations after teacher steps; current observed_stage uses latest measurement at/before current event only",
            "critical_sampling": "offline TRAIN annotations may use transitions up to lead_seconds in the future, never model inputs",
            "lead_seconds": lead_seconds, "trail_seconds": trail_seconds,
            "history_future": "original production six-frame real-history and30-real-action-future selection re-executed; no padding"},
        "critical_focus_policy_revision": FOCUS_POLICY_REVISION,
        "critical_focus_events": FOCUS_EVENTS,
        "critical_focus_excluded_events": ["stable_placement"],
        "critical_focus_scope": "Focus alignment, real container support and release conversions; stable placement remains a causal observed stage and admission requirement, sampled through the full-trajectory component unless near another allowed conversion",
        "user_reported_failure_counts": FAILURES,
        "failure_count_provenance": dict(FAILURE_COUNT_PROVENANCE),
        "mixture": mixture, "episodes": episode_reports,
        "source_receipts": receipts, "script_sha256": sha256(__file__),
        "source_data_modified": False, "running_training_modified": False,
        "new_demonstrations": False, "recovery_demonstrations": False,
        "validation_test_reweighted": False, "model_input_fields_added": False,
        "quality_gate_redefined": False,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ancestor-weights", type=Path)
    parser.add_argument("--lead-seconds", type=float, default=.5)
    parser.add_argument("--trail-seconds", type=float, default=.5)
    args = parser.parse_args()
    metadata = build_sidecar(args.dataset_root, args.output,
        ancestor_weights=read_json(args.ancestor_weights) if args.ancestor_weights else None,
        lead_seconds=args.lead_seconds, trail_seconds=args.trail_seconds)
    print(json.dumps({"status": metadata["status"], "output": str(args.output),
                      "windows": metadata["window_count"], "mixture": metadata["mixture"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
