#!/usr/bin/env python3
"""Evaluate registered expanded gates before starting full candidate fitting."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from egomotion_pipeline.smplh_jobs import assert_plan_current, file_identity
from egomotion_pipeline.storage import Archive, atomic_json


def file_sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def summary(values):
    values = np.asarray(values, dtype=float)
    if not values.size:
        return dict(count=0, mean=None, p95=None, max=None)
    return dict(count=int(values.size), mean=float(values.mean()),
                p95=float(np.percentile(values, 95)), max=float(values.max()))


def guard_inputs(job):
    before = json.loads((job/"production-before.json").read_text())
    failures = []
    for path, expected in before["identities"].items():
        if not Path(path).exists() or file_identity(path) != expected:
            failures.append("Original archive file identity changed: "+path)
    hashes = dict(before["content_sha256"])
    hashes.update(json.loads((job/"motionexpert-before.json").read_text()))
    for path, expected in hashes.items():
        if not Path(path).exists() or file_sha(path) != expected:
            failures.append("Original metadata/reader/training asset changed: "+path)
    return dict(passed=not failures, failures=failures,
                original_file_identities_checked=len(before["identities"]),
                metadata_and_motionexpert_hashes_checked=len(hashes),
                preservation_evidence="file identities for all original Parquet/records; content hashes for metadata and MotionExpert; fresh motion values in independent audit")


def evaluate_gates(policy, audit, preservation, body, wrists, jump_count, pair_count,
                   expected_episodes, expected_frames):
    failures = []
    if not audit.get("passed"):
        failures.append("Independent numerical/source/train-prior audit failed")
    if not preservation.get("passed"):
        failures.append("Original v5 or MotionExpert preservation check failed")
    if audit.get("episodes_checked") != expected_episodes or audit.get("frames_checked") != expected_frames:
        failures.append("Expanded episode/frame population is incomplete")
    for stats, field, threshold in (
        (body, "mean", policy["body_observation_mean_max_m"]),
        (body, "p95", policy["body_observation_p95_max_m"]),
        (wrists, "p95", policy["native_hand_to_fitted_wrist_p95_max_m"]),
    ):
        label = "body" if stats is body else "hand-wrist"
        if not stats["count"] or stats[field] is None or not np.isfinite(stats[field]) or stats[field] > threshold:
            failures.append(f"{label} {field} exceeds registered {threshold:g} m gate or has no observations")
    fraction = jump_count/pair_count if pair_count else None
    if fraction is None or fraction > policy["continuous_local_step_over45deg_fraction_max"]:
        failures.append("Continuous local rotation jumps exceed registered fraction gate or have no pairs")
    return failures, fraction


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--job",type=Path,required=True)
    p.add_argument("--scope",choices=("expanded","full"),default="expanded")
    args=p.parse_args(); job=args.job
    plan=json.loads((job/"plan.json").read_text()); assert_plan_current(plan)
    audit_path=job/"reports"/("sidecar-audit.json" if args.scope=="expanded" else "full-sidecar-audit.json")
    audit=json.loads(audit_path.read_text())
    execution=json.loads((job/(args.scope+"-execution.json")).read_text())
    policy_path=job/"quality-policy.json";policy=json.loads(policy_path.read_text())
    registry=json.loads((job/"quality-policy-registration.json").read_text())
    if (registry["policy_sha256"] != file_sha(policy_path) or registry.get("policy_predates_first_output") is not True
            or registry["job_signature"] != plan["job_signature"] or registry["policy_identity"] != file_identity(policy_path)):
        raise ValueError("Quality policy differs from its pre-output registration")
    if execution["state"]!="complete" or execution["job_signature"]!=plan["job_signature"]:
        raise ValueError("Expanded fitting is not complete under the frozen job")
    root=Path(execution["output_root"])
    if execution.get("scope") != args.scope:
        raise ValueError("Execution belongs to a different validation scope")
    manifest_time=file_identity(root/"manifest.json")["ctime_ns"]
    if (file_identity(policy_path)["ctime_ns"] >= manifest_time
            or (registry.get("registered_at_unix_ns") is not None and
                (registry["registered_at_unix_ns"] >= manifest_time
                 or file_identity(job/"quality-policy-registration.json")["ctime_ns"] >= manifest_time))):
        raise ValueError("Quality policy was not registered before the first sidecar output")
    if Path(audit["sidecar_root"]).resolve()!=root.resolve():
        raise ValueError("Independent audit belongs to different sidecars")
    if audit.get("auditor_sha256") != file_sha(Path(__file__).with_name("audit_smplh_sidecars.py")):
        raise ValueError("Independent auditor changed after the audit")
    if not (audit.get("shared_train_prior_audit") or {}).get("passed"):
        raise ValueError("Independent audit did not verify the train-only prior")
    if args.scope=="expanded":
        if audit.get("selection_sha256") != file_sha(job/"selection.json"):
            raise ValueError("Independent audit did not verify this exact selection")
        expected=plan["selection"]["validation_records"]
    else:
        expected=[dict(episode_index=r["episode_index"],source_key=r["source_key"],length=r["length"])
                  for r in Archive(plan["source_archive"]).records()]
    expected_join={r["episode_index"]:(r["source_key"],r["length"]) for r in expected}
    actual_join={r["episode_index"]:(r.get("source_key"),r.get("frames")) for r in audit["results"]}
    execution_join={r["episode_index"]:(r["source_key"],r["frames"]) for r in execution["receipts"]}
    if (actual_join!=expected_join or execution_join!=expected_join or len(audit["results"])!=len(expected)
            or len(execution["receipts"])!=len(expected)):
        raise ValueError("Audited/executed episode identities differ from exact frozen selection")
    bodies=[];wrists=[];per_episode=[];jump_count=pair_count=0
    for row in audit["results"]:
        index=int(row["episode_index"])
        folder=root/"episodes"/f"{index:08d}"
        receipt=json.loads((folder/"receipt.json").read_text())
        if row.get("numeric_sha256") != receipt["numeric_sha256"] or file_sha(folder/"motion.parquet")!=receipt["parquet_sha256"]:
            raise ValueError("An audited sidecar changed before quality acceptance")
        columns=["quality.observation_residual_m","quality.observation_residual_valid",
                 "fitted.body_world_rot6d_xyz","estimated.valid.body_world_position"]
        for side in ("left","right"):
            columns += [f"source.observation.state.hand_{side}_world", f"source.observation.valid.hand_{side}_position"]
        table=pq.read_table(folder/"motion.parquet",columns=columns)
        a={k:table[k].combine_chunks().values.to_numpy(zero_copy_only=False).reshape(len(table),-1) for k in columns}
        residual=a[columns[0]][a[columns[1]]]
        bodies.append(residual)
        pos=a["fitted.body_world_rot6d_xyz"].reshape(-1,22,9)[...,6:]
        valid=a["estimated.valid.body_world_position"]
        episode_wrists=[]
        for side,joint in (("left",20),("right",21)):
            hand=a[f"source.observation.state.hand_{side}_world"].reshape(-1,21,3)
            good=a[f"source.observation.valid.hand_{side}_position"][:,0]&valid[:,joint]
            episode_wrists.append(np.linalg.norm(hand[:,0]-pos[:,joint],axis=-1)[good])
        merged=np.concatenate(episode_wrists);wrists.append(merged)
        metrics=row["metrics"]
        jump_count+=metrics["continuous_local_step_above45deg_count"]
        pair_count+=metrics["continuous_rotation_pair_count"]
        per_episode.append(dict(episode_index=index,frames=len(table),body=summary(residual),
            hand_wrist=summary(merged),continuous_local_step_above45deg_count=metrics["continuous_local_step_above45deg_count"],
            continuous_rotation_pair_count=metrics["continuous_rotation_pair_count"]))
    body=summary(np.concatenate(bodies));hand=summary(np.concatenate(wrists))
    preservation=guard_inputs(job)
    failures,fraction=evaluate_gates(policy,audit,preservation,body,hand,jump_count,pair_count,
        len(expected),sum(r["length"] for r in expected))
    report=dict(format="smplh-candidate-acceptance-v2",scope=args.scope,passed=not failures,
        full_candidate_processing_allowed=(not failures and args.scope=="expanded"),
        full_candidate_quality_passed=(not failures and args.scope=="full"),formal_publication_allowed=False,
        motion_truth_certified=False,absolute_rotation_accuracy_established=False,
        native_axis_certified=False,failures=failures,job_signature=plan["job_signature"],
        policy=policy,policy_sha256=file_sha(policy_path),independent_audit_sha256=file_sha(audit_path),
        body_observation_residual_m=body,native_hand_to_fitted_wrist_m=hand,
        continuous_local_step_above45deg_count=jump_count,continuous_rotation_pair_count=pair_count,
        continuous_local_step_above45deg_fraction=fraction,preservation=preservation,
        episodes=len(per_episode),frames=sum(r["frames"] for r in per_episode),per_episode=per_episode,
        evaluation_kind="offline fitted-label reconstruction; group calibration may use same recording; not prediction/generalization accuracy")
    atomic_json(job/"reports"/(args.scope+"-acceptance.json"),report)
    print(json.dumps({k:v for k,v in report.items() if k not in ("per_episode","preservation")},indent=2))
    return 0 if report["passed"] else 1


if __name__=="__main__":
    raise SystemExit(main())
