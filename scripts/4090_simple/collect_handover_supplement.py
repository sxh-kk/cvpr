"""Versioned, TRAIN-only supplementary Handover execution and capture.

The legacy quality100 queue and its bounds remain immutable. This independent
contract permits bounded 2cm single-axis proposals, runs CPU physics first, and
admits only newly rendered captures passing the unchanged full physics/RGB
auditor. These are teacher variants, never policy recoveries or TEST examples.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import math
import os
from pathlib import Path
import shutil
import subprocess

import collect_handover_quality100 as quality100
import plan_handover_quality100 as planner

PLAN_CONTRACT = "cvpr-handover-train-supplement-plan-v1"
QUEUE_CONTRACT = "cvpr-handover-train-supplement-queue-v1"
BOUNDS = {"maximum_abs_xy_m": .03, "maximum_abs_yaw_rad": .10}
SCOPE = "TRAIN_teacher_variants_not_policy_recoveries"
REPO = Path(__file__).resolve().parents[2]
REPLAY = Path(__file__).with_name("replay_handover_teacher.py")
SPEC_KEYS = quality100.SPEC_KEYS
read, write, sha256 = quality100.read, quality100.write, quality100.sha256


def _integer(value, minimum, maximum):
    return type(value) is int and minimum <= value <= maximum


def validate_candidate(candidate, ancestors):
    source = candidate.get("ancestor", {}).get("source_episode_index")
    if (source in planner.HELDOUT_VAL or source not in ancestors
            or candidate.get("ancestor") != ancestors[source]
            or candidate.get("split") != "train" or candidate.get("accepted") is not False
            or candidate.get("status") != "planned_not_collected"
            or not _integer(candidate.get("variant"), 0, 999999)):
        raise ValueError("Supplement candidates require an unchanged original TRAIN ancestor")
    spec = {key: candidate[key] for key in SPEC_KEYS}
    identity = planner.digest(spec)
    expected = f"handover-supplement-source{source:03d}-variant{candidate['variant']:02d}-{identity[:16]}"
    if candidate.get("candidate_spec_sha256") != identity or candidate.get("candidate_id") != expected:
        raise ValueError("Supplement proposal identity differs from its registered bytes")
    for key in ("physics_seed", "material_seed", "generation_seed"):
        if not _integer(candidate.get(key), 1, planner.TRAIN_SEED_MAX):
            raise ValueError("Supplement seeds must remain in the TRAIN seed range")
    deltas = candidate.get("initial_object_deltas", {})
    if set(deltas) != {"target", "container"}:
        raise ValueError("Supplement proposals must bind both initial object deltas")
    changed = False
    for value in deltas.values():
        if set(value) != {"xy_delta_m", "world_z_yaw_delta_rad"}:
            raise ValueError("Only one-time initial XY/yaw perturbations are supported")
        xy = value["xy_delta_m"]
        yaw = value["world_z_yaw_delta_rad"]
        if (not isinstance(xy, list) or len(xy) != 2
                or any(type(x) not in (int, float) or not math.isfinite(x) or abs(x) > .03 for x in xy)
                or type(yaw) not in (int, float) or not math.isfinite(yaw) or abs(yaw) > .10):
            raise ValueError("Supplement reset proposal exceeds its fixed3cm/0.10rad bounds")
        changed |= any(x != 0 for x in (*xy, yaw))
    if not changed:
        raise ValueError("A supplementary execution must propose an actual physical reset change")


def validate_plan(plan):
    """Pure structural check; file identities are checked separately by load_plan."""
    if (plan.get("contract") != PLAN_CONTRACT or plan.get("scope") != SCOPE
            or plan.get("status") != "planned_not_collected"
            or plan.get("perturbation_bounds") != BOUNDS
            or plan.get("physical_acceptance_gate") != "handover-physics-v2"
            or plan.get("existing_validation_sources") != list(planner.HELDOUT_VAL)
            or plan.get("policy_recovery_data") is not False
            or not _integer(plan.get("target_accepted_count"), 6, 12)
            or plan.get("maximum_accepted_per_ancestor") != 2
            or plan.get("minimum_distinct_accepted_ancestors") != 6):
        raise ValueError("Unknown or weakened supplementary TRAIN protocol")
    originals = plan.get("original_ancestors", [])
    ancestors = {r["source_episode_index"]: r for r in originals}
    selected = plan.get("selected_source_ids", [])
    if (len(originals) != 24 or len(ancestors) != 24 or set(ancestors) & set(planner.HELDOUT_VAL)
            or len(selected) != len(set(selected)) or not 6 <= len(selected) <= 24
            or not set(selected) <= set(ancestors)
            or plan["target_accepted_count"] > 2 * len(selected)):
        raise ValueError("Supplement must bind24 eligible ancestors and select at least6 TRAIN sources")
    controls = plan.get("warmup_controls", {})
    if set(controls) != {str(n) for n in selected} or any(v not in quality100.WARMUP_KINDS for v in controls.values()):
        raise ValueError("Supplement needs a frozen TRAIN-only warmup control for every selected source")
    candidates = plan.get("candidates", [])
    ids, specs, covered = set(), set(), set()
    for candidate in candidates:
        validate_candidate(candidate, ancestors)
        source = candidate["ancestor"]["source_episode_index"]
        if source not in selected or candidate["candidate_id"] in ids or candidate["candidate_spec_sha256"] in specs:
            raise ValueError("Supplement candidates repeat or escape the selected TRAIN sources")
        ids.add(candidate["candidate_id"]); specs.add(candidate["candidate_spec_sha256"]); covered.add(source)
    if covered != set(selected) or plan.get("candidate_count") != len(candidates):
        raise ValueError("Supplement candidates must cover all declared selected TRAIN sources")
    return plan


def build_plan(manifest, selection, *, manifest_sha256, selection_sha256,
               warmup_controls, source_ids, target=6, seed=20261024, xy_m=.02, yaw_rad=.075):
    ancestors, _ = planner.eligible_ancestors(manifest, selection)
    if manifest.get("selection_sha256") != selection_sha256:
        raise ValueError("Source release/selection identity mismatch")
    if (not _integer(seed, 0, 2**63 - 1) or not 0 < xy_m <= .03 or not 0 < yaw_rad <= .10):
        raise ValueError("Use a finite bounded supplementary reset proposal")
    selected = sorted(source_ids)
    candidates = []
    # Round-robin by intervention, keeping object roles separate and zeroing
    # untouched axes explicitly. No trajectory values are edited afterwards.
    patterns = (("target", 0, 1), ("container", 1, 1), ("target", 0, -1), ("container", 1, -1))
    by_source = {row["source_episode_index"]: row for row in ancestors}
    for variant, (role, axis, sign) in enumerate(patterns):
        for n in selected:
            if n not in by_source or n in planner.HELDOUT_VAL:
                raise ValueError("Requested supplementary source is not an original TRAIN ancestor")
            deltas = {key: {"xy_delta_m": [0., 0.], "world_z_yaw_delta_rad": 0.} for key in ("target", "container")}
            deltas[role]["xy_delta_m"][axis] = sign * float(xy_m)
            deltas[role]["world_z_yaw_delta_rad"] = sign * float(yaw_rad)
            proposal = {"ancestor": deepcopy(by_source[n]), "variant": variant,
                        **{key + "_seed": planner.seed_for(seed, n, variant, "supplement." + key)
                           for key in ("physics", "material", "generation")},
                        "initial_object_deltas": deltas}
            identity = planner.digest(proposal)
            candidates.append({**proposal, "candidate_id": f"handover-supplement-source{n:03d}-variant{variant:02d}-{identity[:16]}",
                               "candidate_spec_sha256": identity, "split": "train", "accepted": False,
                               "status": "planned_not_collected"})
    plan = {"contract": PLAN_CONTRACT, "scope": SCOPE, "status": "planned_not_collected",
            "task": planner.TASK, "source_manifest_sha256": manifest_sha256, "selection_sha256": selection_sha256,
            "seed": seed, "target_accepted_count": target, "maximum_accepted_per_ancestor": 2,
            "minimum_distinct_accepted_ancestors": 6, "original_ancestors": ancestors,
            "selected_source_ids": selected, "warmup_controls": {str(n): warmup_controls[str(n)] for n in selected},
            "existing_validation_sources": list(planner.HELDOUT_VAL), "perturbation_bounds": dict(BOUNDS),
            "physical_acceptance_gate": "handover-physics-v2", "policy_recovery_data": False,
            "material_seed_applied": False, "generation_seed_applied": False,
            "candidate_count": len(candidates), "candidates": candidates,
            "source_diversity_note": "New physical variants of existing TRAIN motions; no new human motion families or certified TEST."}
    return validate_plan(plan)


def load_plan(path):
    plan = validate_plan(read(path))
    files = plan["source_files"]
    for receipt in files.values():
        if sha256(receipt["path"]) != receipt["sha256"]:
            raise ValueError("Supplement source metadata changed after preregistration")
    originals, _ = planner.eligible_ancestors(read(files["manifest"]["path"]), read(files["selection"]["path"]))
    manifest = read(files["manifest"]["path"])
    if originals != plan["original_ancestors"] or manifest.get("selection_sha256") != plan["selection_sha256"]:
        raise ValueError("Supplement declared ancestry differs from the actual source release")
    if (files["manifest"]["sha256"] != plan["source_manifest_sha256"]
            or files["selection"]["sha256"] != plan["selection_sha256"]):
        raise ValueError("Supplement source metadata binding differs")
    schedule = read(files["warmup_schedule"]["path"])
    if (schedule.get("contract") != quality100.WARMUP_SCHEDULE_CONTRACT
            or schedule.get("scope") != quality100.WARMUP_SCOPE
            or any(schedule["controls"].get(str(n)) != value for n, value in plan["warmup_controls"].items())):
        raise ValueError("Supplement warmup schedule differs from the bound TRAIN schedule")
    return plan


def code_identities():
    paths = {Path(__file__), REPLAY, *quality100.CORE_FILES, quality100.FRAME_CALIBRATION,
             quality100.LIVE_WARMUP_SOURCE,
             REPO / "VLMExpert/third_party/SIMPLE/src/simple/baselines/sonic_v1_1.py",
             REPO / "VLMExpert/third_party/SIMPLE/src/simple/core/action.py",
             REPO / "VLMExpert/third_party/SIMPLE/src/simple/engines/isaacsim.py",
             REPO / "VLMExpert/third_party/SIMPLE/src/simple/engines/mujoco.py",
             REPO / "VLMExpert/third_party/SIMPLE/src/simple/envs/sonic_loco_manip.py"}
    return {str(p): sha256(p) for p in sorted(paths)}


def queue_summary(state, plan):
    accepted = [x for x in state["attempts"] if x["status"] == "accepted"]
    counts = Counter(x["source_episode_index"] for x in accepted)
    if len(accepted) > plan["target_accepted_count"] or max(counts.values(), default=0) > 2:
        raise ValueError("Supplement acceptance exceeds its frozen total/source quota")
    complete = len(accepted) == plan["target_accepted_count"] and len(counts) >= 6
    state.update(status="accepted_target_reached_pack_pending" if complete else "incomplete",
                 accepted_count=len(accepted), accepted_per_ancestor={str(n): v for n, v in sorted(counts.items())},
                 minimum_distinct_sources_satisfied=len(counts) >= 6, capture_target_satisfied=complete,
                 dataset_pack_complete=False, DINO_cache_generated=False, final_ready=False,
                 policy_recovery_data=False, independent_TEST_certified=False, updated_utc=quality100.now())
    return state


def prepare(args):
    output = args.output.resolve()
    inputs = {key: str(getattr(args, key).resolve(strict=True)) for key in ("snapshot_root", "data_root", "sonic_resources", "python")}
    files = {key: {"path": str(getattr(args, key).resolve(strict=True)), "sha256": sha256(getattr(args, key))}
             for key in ("manifest", "selection", "warmup_schedule")}
    if (not args.output.is_absolute() or output.exists() or output.is_relative_to(REPO)
            or any(output.is_relative_to(Path(v)) for k, v in inputs.items() if k != "python")
            or any(output.is_relative_to(Path(v["path"]).parent) for v in files.values())):
        raise ValueError("Use a fresh supplementary output outside all source/resource trees")
    schedule = read(args.warmup_schedule)
    if schedule.get("contract") != quality100.WARMUP_SCHEDULE_CONTRACT or schedule.get("scope") != quality100.WARMUP_SCOPE:
        raise ValueError("Use the already registered TRAIN-only warmup schedule")
    plan = build_plan(read(args.manifest), read(args.selection), manifest_sha256=files["manifest"]["sha256"],
                      selection_sha256=files["selection"]["sha256"], warmup_controls=schedule["controls"],
                      source_ids=args.sources, target=args.target, seed=args.seed, xy_m=args.xy_m, yaw_rad=args.yaw_rad)
    plan["source_files"] = files
    plan["runtime_inputs"] = inputs
    plan["created_utc"] = quality100.now()
    output.mkdir(parents=True, exist_ok=False)
    write(output / "candidate_plan.json", plan)
    code = code_identities()
    for path, identity in code.items():
        archived = output / "frozen-code" / Path(path).relative_to(REPO)
        archived.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, archived)
        if sha256(archived) != identity: raise ValueError("Supplement source changed while freezing")
    state = {"contract": QUEUE_CONTRACT, "scope": SCOPE, "plan_sha256": sha256(output / "candidate_plan.json"),
             "code_sha256": code, "attempts": [], "created_utc": quality100.now()}
    write(output / "queue.json", queue_summary(state, plan))
    return state


def verified_queue(root):
    root = Path(root).resolve(strict=True)
    plan_path = root / "candidate_plan.json"
    plan = load_plan(plan_path); state = read(root / "queue.json")
    if (state.get("contract") != QUEUE_CONTRACT or state.get("scope") != SCOPE
            or state.get("plan_sha256") != sha256(plan_path) or state.get("code_sha256") != code_identities()):
        raise ValueError("Supplement queue plan or executable identity changed")
    by_id = {c["candidate_id"]: c for c in plan["candidates"]}
    seen = set()
    for index, attempt in enumerate(state["attempts"]):
        cid = attempt.get("candidate_id")
        if cid not in by_id or cid in seen or attempt.get("directory") != f"attempt-{index:05d}":
            raise ValueError("Supplement attempt repeats or invents a candidate")
        candidate = by_id[cid];seen.add(cid)
        if (attempt.get("candidate_spec_sha256") != candidate["candidate_spec_sha256"]
                or attempt.get("source_episode_index") != candidate["ancestor"]["source_episode_index"]):
            raise ValueError("Supplement attempt source/proposal differs from the frozen plan")
        if attempt.get("status", "").endswith("_running"):
            raise RuntimeError("Prior supplementary child has not been finalized; inspect it before continuing")
        if attempt["status"] in ("screen_passed", "accepted"):
            receipt = root / attempt["screen_audit_path"]
            if sha256(receipt) != attempt["screen_audit_sha256"]: raise ValueError("Supplement screen receipt changed")
            screen = read(receipt)
            if (screen.get("status") != "independent_physics_audit_passed"
                    or screen.get("candidate_id") != cid or screen.get("plan_sha256") != state["plan_sha256"]):
                raise ValueError("Supplement screen receipt does not bind this physical candidate")
            for path, identity in screen["bindings"].items():
                if sha256(path) != identity: raise ValueError("Supplement screen payload changed")
        if attempt["status"] == "accepted":
            receipt = root / attempt["audit_path"]
            if sha256(receipt) != attempt["audit_sha256"]: raise ValueError("Supplement RTX receipt changed")
            audit = read(receipt)
            if audit.get("status") != "independent_capture_audit_passed" or audit.get("eligible_dataset") is not True:
                raise ValueError("Supplement capture has not passed independent admission")
            for group in ("bindings", "source_sha256"):
                for path, identity in audit[group].items():
                    if sha256(path) != identity: raise ValueError("Supplement admitted capture bytes changed")
            actual = quality100.audit_capture(root / attempt["directory"] / "rtx",
                root / attempt["directory"] / "rtx/capture", candidate, state["plan_sha256"],
                expected_warmup=plan["warmup_controls"][str(attempt["source_episode_index"])])
            stable = lambda x: {k: v for k, v in x.items() if k != "generated_utc"}
            if stable(actual) != stable(audit):
                raise ValueError("Supplement admission no longer matches independent capture recomputation")
    return state, plan


def _execute(command, env, log, state, attempt, phase, root, plan, timeout):
    attempt.update(status=phase + "_running", command=command, phase_started_utc=quality100.now())
    write(root / "queue.json", queue_summary(state, plan))
    with log.open("x") as stream:
        result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout)
    if result.returncode: raise RuntimeError(f"Owned {phase} child returned {result.returncode}; partial files retained")


def process(root, *, phase, maximum=1, candidate_id=None, timeout=900):
    root = Path(root).resolve(strict=True)
    if phase not in ("screen", "capture") or not _integer(maximum, 1, 200) or timeout <= 0:
        raise ValueError("Use screen/capture with a positive bounded attempt budget")
    with quality100.queue_lock(root):
        state, plan = verified_queue(root)
        candidates = {c["candidate_id"]: c for c in plan["candidates"]}
        if candidate_id is not None and (candidate_id not in candidates or maximum != 1):
            raise ValueError("Select exactly one existing supplementary candidate")
        processed = 0
        for candidate in plan["candidates"]:
            cid = candidate["candidate_id"];source = candidate["ancestor"]["source_episode_index"]
            if candidate_id is not None and cid != candidate_id: continue
            if processed >= maximum or state["capture_target_satisfied"]: break
            previous = next((x for x in state["attempts"] if x["candidate_id"] == cid), None)
            if phase == "screen":
                if previous is not None: continue
                attempt = {"candidate_id": cid, "candidate_spec_sha256": candidate["candidate_spec_sha256"],
                           "plan_sha256": state["plan_sha256"], "source_episode_index": source,
                           "directory": f"attempt-{len(state['attempts']):05d}", "status": "registered",
                           "warmup_control": plan["warmup_controls"][str(source)]}
                state["attempts"].append(attempt);(root / attempt["directory"]).mkdir(exist_ok=False)
            else:
                if previous is None or previous["status"] != "screen_passed": continue
                attempt = previous
                counts = Counter(x["source_episode_index"] for x in state["attempts"] if x["status"] == "accepted")
                missing = max(0, 6 - len(counts))
                if counts[source] >= 2 or (plan["target_accepted_count"]-sum(counts.values()) <= missing and source in counts): continue
            processed += 1; directory = root / attempt["directory"];inputs = plan["runtime_inputs"]
            output = directory / ("screen" if phase == "screen" else "rtx")
            command = [inputs["python"], str(REPLAY), "--snapshot-root", inputs["snapshot_root"],
                       "--data-root", inputs["data_root"], "--sonic-resources", inputs["sonic_resources"],
                       "--candidate-plan", str(root / "candidate_plan.json"), "--candidate-id", cid,
                       "--command-rate", "30", "--warmup-control", attempt["warmup_control"], "--warmup-seconds", "1",
                       "--pd-profile", "fixed_cpp_baseline", "--hand-torque-profile", "recording_0p7_nm",
                       "--render-mode", "none" if phase == "screen" else "isaac", "--output", str(output)]
            if phase == "capture": command.extend(["--capture-training", str(output / "capture")])
            env = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1")
            if phase == "screen": env["CUDA_VISIBLE_DEVICES"] = ""
            else: env.update(OMNI_KIT_ACCEPT_EULA="YES", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
            try:
                _execute(command, env, directory / (phase + ".log"), state, attempt, phase, root, plan, timeout)
                if phase == "screen":
                    audit = quality100.audit_physics(output, candidate, state["plan_sha256"], expected_warmup=attempt["warmup_control"])
                    path = directory / "screen_audit.json";write(path, audit)
                    attempt.update(status="screen_passed", screen_audit_path=str(path.relative_to(root)), screen_audit_sha256=sha256(path))
                else:
                    audit = quality100.audit_capture(output, output / "capture", candidate, state["plan_sha256"], expected_warmup=attempt["warmup_control"])
                    contents = [read(root / x["audit_path"])["capture_content_sha256"] for x in state["attempts"] if x["status"] == "accepted"]
                    if audit["capture_content_sha256"] in contents: raise ValueError("Duplicate actual supplementary capture")
                    path = directory / "independent_capture_audit.json";write(path, audit)
                    attempt.update(status="accepted", audit_path=str(path.relative_to(root)), audit_sha256=sha256(path),
                                   rtx_capture_completed=True, diversity_receipt=audit["diversity_receipt"])
            except Exception as error:
                attempt.update(status=phase + "_rejected", error=repr(error))
                if isinstance(error, quality100.CaptureRejected): write(directory / "independent_rejection.json", error.receipt)
            attempt.update(ended_utc=quality100.now())
            write(root / "queue.json", queue_summary(state, plan))
        return {**queue_summary(state, plan), "processed_in_call": processed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    p = modes.add_parser("prepare")
    for key in ("manifest", "selection", "warmup-schedule", "snapshot-root", "data-root", "sonic-resources", "python", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--sources", type=lambda v: [int(x) for x in v.split(",")], required=True)
    p.add_argument("--target", type=int, default=6);p.add_argument("--seed", type=int, default=20261024)
    p.add_argument("--xy-m", type=float, default=.02);p.add_argument("--yaw-rad", type=float, default=.075)
    for name in ("screen", "capture"):
        p = modes.add_parser(name);p.add_argument("--queue", type=Path, required=True)
        p.add_argument("--max-candidates", type=int, default=1);p.add_argument("--candidate-id")
        p.add_argument("--timeout", type=float, default=900)
    p=modes.add_parser("status");p.add_argument("--queue", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "prepare": result = prepare(args)
    elif args.mode == "status": result = queue_summary(*verified_queue(args.queue))
    else: result = process(args.queue, phase=args.mode, maximum=args.max_candidates, candidate_id=args.candidate_id, timeout=args.timeout)
    import json
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
