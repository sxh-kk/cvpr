"""Select existing strict successes without collecting or simulating new data."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from diversity_handover_random100 import measured_descriptor
from pack_handover_random100 import validate_admission, validate_full_footprint

CONTRACT = "cvpr-handover-curated-selection-v1"
JOINT_RMS = .015
FK_RMS = .008


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_old_pack_binding(summary, old_pack):
    """Never combine an audit for package A with package B's episode IDs."""
    old_pack = Path(old_pack).resolve(strict=True)
    for relative in ("PACK_READY.json", "train/meta/episodes.jsonl"):
        path = old_pack / relative
        require(summary["source_sha256"].get(str(path)) == sha(path),
                "Original quality audit does not bind this old package: " + relative)
    ready = read(old_pack / "PACK_READY.json")
    metadata = [json.loads(line) for line in (old_pack / "train/meta/episodes.jsonl").read_text().splitlines() if line]
    return ready, {e["episode_index"]: e for e in metadata}


def distance(a, b):
    joint = float(np.sqrt(np.mean((a["joints"] - b["joints"]) ** 2)))
    fk = float(np.sqrt(np.mean((a["fk"] - b["fk"]) ** 2)))
    require(np.isfinite(joint) and np.isfinite(fk), "Nonfinite measured-motion distance")
    return {"joint_RMS_rad": joint, "FK_RMS_m": fk,
            "near_duplicate": joint <= JOINT_RMS and fk <= FK_RMS}


def quality_key(row):
    margin = float(row["actual_full_outer_footprint_final50"]["minimum_outer_hull_margin_m"])
    displacement = float(row["final_stability_max_pairwise_displacement_m"])
    support = float(row["final_support_weight_fraction"]["min"])
    require(np.isfinite([margin, displacement, support]).all(), "Nonfinite quality ranking")
    return (-margin, displacement, -support, row["episode_index"])


def choose(old_rows, new_entries, descriptors):
    """Mandatory new14 first, then strict old successes in quality order."""
    require(len(new_entries) == 14, "Preserve exactly the already admitted new14")
    retained = [{"origin": "new14", "candidate_id": e["candidate_id"]} for e in new_entries]
    require(len({r["candidate_id"] for r in retained}) == 14, "Repeated new14 identity")
    rejected, pairs = [], []
    for i, a in enumerate(retained):
        for b in retained[i + 1:]:
            metric = distance(descriptors[a["candidate_id"]], descriptors[b["candidate_id"]])
            if metric["near_duplicate"]:
                pairs.append({"first": a["candidate_id"], "second": b["candidate_id"], **metric})
    require(not pairs, "The mandatory new14 contain measured near duplicates; do not silently drop them")
    for row in sorted(old_rows, key=quality_key):
        identity = row["candidate_id"]
        duplicate = None
        for keeper in retained:
            metric = distance(descriptors[identity], descriptors[keeper["candidate_id"]])
            if metric["near_duplicate"]:
                duplicate = {"old_episode_index": row["episode_index"], "candidate_id": identity,
                             "kept_candidate_id": keeper["candidate_id"], **metric}
                break
        if duplicate:
            rejected.append(duplicate)
        else:
            retained.append({"origin": "original100", "old_episode_index": row["episode_index"],
                             "candidate_id": identity})
    return retained, rejected


def select(new_queue, old_audit, old_pack, output):
    new_queue, old_audit, old_pack = (Path(p).resolve(strict=True) for p in (new_queue, old_audit, old_pack))
    output = Path(output).absolute()
    require(not output.exists(), "Selection output must be new")
    queue_path = new_queue / "queue.json"
    summary_path = old_audit / "summary.json"
    bindings = {str(p): sha(p) for p in (queue_path, summary_path, old_pack / "PACK_READY.json")}
    queue, summary = read(queue_path), read(summary_path)
    require(summary["old_v2_pass_count"] == summary["recomputed_episodes"] == summary["episodes"] == 100
            and not summary["unverifiable_episodes"] and summary["physics_steps_performed"] == 0,
            "Original100 retrospective quality audit is incomplete")
    old_ready, old_metadata = validate_old_pack_binding(summary, old_pack)
    old_receipts = {e["episode_index"]: e for e in old_ready["episodes"]}
    require(set(old_metadata) == set(old_receipts) == set(range(100)), "Old package episode identities differ")
    new_entries = queue["qualified"]
    require(len(new_entries) == 14 and not any(a["status"] == "running" for a in queue["attempts"]),
            "Use the stopped, completed new14 queue")
    descriptors, new_audits, old_rows = {}, [], []
    for e in new_entries:
        admission_path = Path(e["admission_path"]).resolve(strict=True)
        require(sha(admission_path) == e["admission_sha256"], "New14 admission changed")
        audit = read(admission_path)
        validate_admission(audit, e)
        run = Path(e["run"]).resolve(strict=True)
        for name in ("physics_model.mjb", "trajectory.npz"):
            p = run / name
            require(sha(p) == audit["bindings"].get(str(p)), "Measured new14 motion changed")
            bindings[str(p)] = sha(p)
        bindings[str(admission_path)] = e["admission_sha256"]
        descriptors[e["candidate_id"]] = measured_descriptor(run)
        new_audits.append({"candidate_id": e["candidate_id"], "run": str(run),
                           "admission_path": str(admission_path), "admission_sha256": e["admission_sha256"]})
    old_report_bindings = {r["episode_index"]: r for r in summary["row_reports"]}
    require(len(old_report_bindings) == 100, "Missing original100 row report identities")
    for index in summary["unified_pass_episode_indices"]:
        p = old_audit / "episodes" / f"episode_{index:06d}.json"
        require(Path(old_report_bindings[index]["path"]).resolve() == p.resolve()
                and sha(p) == old_report_bindings[index]["sha256"], "Old quality row changed")
        row = read(p)
        require(row["episode_index"] == index and row["old_v2_recomputed_exact"] is True
                and row["unified_v2_and_actual_full50_passed"] is True, "Old quality decision changed")
        require(row["candidate_id"] == old_metadata[index]["candidate_id"] == old_receipts[index]["candidate_id"]
                and row["source_identity"] == old_metadata[index]["source_identity"]
                and row["parquet_sha256"] == old_receipts[index]["parquet_sha256"]
                and row["admission_sha256"] == old_receipts[index]["admission_sha256"],
                "Old quality row belongs to a different packed episode")
        require(row["actual_full_outer_footprint_final50"]["minimum_outer_hull_margin_m"] >= .005,
                "Old sample does not meet the common complete-footprint gate")
        validate_full_footprint(row["actual_full_outer_footprint_final50"])
        run = Path(row["actual_RTX_run"]).resolve(strict=True)
        for name in ("physics_model.mjb", "trajectory.npz"):
            source = run / name
            require(sha(source) == row["bound_core_evidence_sha256"][str(source)], "Old measured motion changed")
            bindings[str(source)] = sha(source)
        bindings[str(p)] = sha(p)
        descriptors[row["candidate_id"]] = measured_descriptor(run)
        old_rows.append(row)
    retained, duplicates = choose(old_rows, new_entries, descriptors)
    selected = [r["old_episode_index"] for r in retained if r["origin"] == "original100"]
    ranked = sorted(old_rows, key=quality_key)
    report = {"contract": CONTRACT, "created_utc": datetime.now(timezone.utc).isoformat(),
        "new_queue": str(new_queue), "old_pack": str(old_pack), "old_quality_audit": str(summary_path),
        "old_quality_audit_sha256": bindings[str(summary_path)],
        "new14": new_audits, "old_episode_indices": selected,
        "old_quality_reports": {str(r["episode_index"]): str(old_audit / "episodes" / f"episode_{r['episode_index']:06d}.json")
                                for r in old_rows if r["episode_index"] in selected},
        "retained": retained, "new_count": 14, "old_count": len(selected), "total_count": len(retained),
        "old_strict_pass_count_before_dedup": len(old_rows), "rejected_near_duplicates": duplicates,
        "old_quality_ranking": [{"old_episode_index": r["episode_index"], "candidate_id": r["candidate_id"],
            "minimum_complete_footprint_margin_m": -quality_key(r)[0],
            "final_max_pairwise_displacement_m": quality_key(r)[1],
            "minimum_support_weight_fraction": -quality_key(r)[2]} for r in ranked],
        "selection_policy": "Keep all new14; old100 must meet unchanged physics-v2 plus actual complete collision footprint final50>=5mm. Rank eligible old by larger complete-footprint margin, lower final displacement, greater minimum support; retain only if not near any prior keeper.",
        "near_duplicate_rule": {"joint_RMS_rad": JOINT_RMS, "FK_RMS_m": FK_RMS,
            "rule": "both thresholds inclusive", "normalized_time_samples": 64,
            "global_initial_pelvis_SE3_removed": True, "measured_state43_and_pelvis_two_wrists": True},
        "source_identity_policy": "Preserve full original namespace and payload identities; numeric source IDs alone do not establish cross-archive ancestry.",
        "collection_started": False, "physics_steps_performed": 0, "GPU_used": False,
        "source_sha256": bindings, "selector_sha256": sha(__file__)}
    require(all(sha(p) == digest for p, digest in bindings.items()), "Inputs changed during selection")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("new-queue", "old-audit", "old-pack", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    report = select(**vars(parser.parse_args()))
    print(json.dumps({k: report[k] for k in ("new_count", "old_count", "total_count")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
