"""Recoverable independent random100 queue; only strict RTX admissions count.

Preparation freezes sources/code/models and continuous scene proposals. Formal
execution requires a current full-RTX successful bootstrap. CPU passes are only
screening, every failed/interrupted attempt remains in the denominator, and
neither this collector nor collection readiness uploads data or starts training.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import time

import pack_handover_random100 as packer

REPO = Path(__file__).resolve().parents[2]
DIRECTORY = Path(__file__).resolve().parent
RUNNER = DIRECTORY / "replay_handover_random100.py"
PLAN_CONTRACT = "cvpr-handover-random100-collection-plan-v1"
QUEUE_CONTRACT = "cvpr-handover-random100-collection-queue-v1"
REQUIREMENTS_CONTRACT = "cvpr-handover-random100-rebuild-requirements-v1"
FORMAL = {"xy_max_m": .05, "yaw_max_rad": .25, "max_seconds": 30.}
REQUIRED_RUN_CODE = ("replay_handover_random100.py", "handover_random100_sources.py",
                     "handover_random100_scene.py", "handover_random100_expert.py", "sonic_v1_1.py",
                     "g1_sonic.py", "g1_29dof_sonic_model12.yaml")
read, sha = packer.legacy.read, packer.legacy.sha


def require(value, message):
    if not value: raise ValueError(message)


def now(): return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def proposal_file_sha(proposal):
    import replay_handover_teacher as base
    return hashlib.sha256((base.json_text(proposal, indent=2) + "\n").encode()).hexdigest()


def write(path, value, *, exclusive=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if exclusive:
        with path.open("x") as stream: stream.write(payload)
    else:
        temporary = path.with_name(path.name + ".pending")
        with temporary.open("w") as stream: stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        temporary.replace(path)


@contextmanager
def lock(root):
    with (Path(root) / "queue.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try: yield
        finally: fcntl.flock(stream, fcntl.LOCK_UN)


def train_seed(master, source, scene_round):
    require(type(master) is int and 0 <= master < 2**63 and type(source) is int
            and source not in packer.EXCLUDED and 0 <= source < 100
            and type(scene_round) is int and scene_round >= 0, "Invalid TRAIN seed derivation")
    return 1 + int(digest(["random100-TRAIN", master, source, scene_round])[:16], 16) % (2**30 - 1)


def source_order(eligible_sources, old_sources):
    eligible, old = set(eligible_sources), set(old_sources)
    require(len(eligible) == 94 and not eligible & packer.EXCLUDED and old <= eligible and len(old) == 24,
            "Require94 complete TRAIN candidates and protected legacy24 priority identities")
    return sorted(eligible - old) + sorted(old)


def code_identities():
    from handover_sonic_teleop_reference import SOURCE_PATHS
    names = ("collect_handover_random100.py", "replay_handover_random100.py", "audit_handover_random100.py",
        "diversity_handover_random100.py", "handover_random100_sources.py", "handover_random100_scene.py",
        "handover_random100_expert.py", "handover_sonic_latent_projector.py", "handover_sonic_teleop_reference.py", "feedback_wrist_ik_v11.py",
        "prepare_carrybox_v11.py", "handover_eval_runtime.py", "handover_contact_quality.py", "handover_geometry_quality.py",
        "replay_handover_teacher.py", "prepare_handover_eval.py", "collect_handover_quality100.py",
        "pack_handover_quality100.py", "pack_handover_random100.py")
    simple = REPO / "VLMExpert/third_party/SIMPLE"
    paths = [*(DIRECTORY / name for name in names), *(simple / "src/simple" / name for name in
        ("core/action.py", "engines/isaacsim.py", "engines/mujoco.py", "sensors/head_camera_mount.py",
         "envs/sonic_loco_manip.py", "robots/g1_sonic.py", "baselines/sonic_v1_1.py", "baselines/live_pose_sonic_warmup.py")),
        simple / "third_party/gear_sonic/utils/mujoco_sim/configs.py",
        simple / "third_party/gear_sonic/utils/mujoco_sim/wbc_configs/g1_29dof_sonic_model12.yaml"]
    paths.extend(Path(path).resolve(strict=True) for path in SOURCE_PATHS.values()
                 if Path(path).resolve(strict=True).is_relative_to(REPO))
    return {str(path.resolve(strict=True)): sha(path) for path in sorted(set(paths))}


def official_placement_sources():
    from handover_sonic_teleop_reference import verify_official_sources
    evidence = verify_official_sources()
    require(evidence.get("official_sources_rehashed") is True, "Official teleop sources were not verified")
    return evidence["source_sha256"]


def stat_identity(path):
    value = Path(path).stat()
    return [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns]


def validate_plan(plan):
    require(plan.get("contract") == PLAN_CONTRACT and plan.get("formal") == FORMAL
            and plan.get("target") == 100 and plan.get("minimum_distinct_sources") == 50
            and plan.get("maximum_registered_per_source") == 2 and plan.get("TRAIN_seed_upper_exclusive") == 2**30,
            "Unknown or weakened formal random100 protocol")
    require(plan.get("TRAIN_seed_minimum") == 1 and plan.get("additional_dataset_quality") == packer.ADDITIONAL_DATASET_QUALITY,
            "Strict TRAIN seed range and actual final50 collision-footprint requirements missing")
    require(plan["runtime"].get("placement_encoder_mode") in ("g1", "teleop"), "Unknown frozen placement encoder mode")
    require(plan["runtime"].get("placement_locomotion") in ("source_future", "actual_hold"),
            "Unknown frozen placement locomotion")
    phases = plan["runtime"]["latent_projection_phases"].split(",")
    require(phases and len(phases) == len(set(phases))
            and all(name in ("approach", "right_lift", "handover", "left_transfer", "placement", "release", "hold")
                    for name in phases), "Invalid frozen optional projection phases")
    order = source_order(plan["source_order"], plan["legacy24_source_ids"])
    require(order == plan["source_order"] and len(plan["sources"]) == 94, "Incomplete94-source inventory")
    candidates = plan["candidates"]; identities = set()
    rounds = plan["candidates_per_source"]
    require(type(rounds) is int and 2 <= rounds <= 20 and len(candidates) == 94 * rounds,
            "Use bounded candidate rounds covering every original TRAIN source")
    for index, candidate in enumerate(candidates):
        round_index, source_offset = divmod(index, 94)
        spec = candidate["spec"]; source = order[source_offset]
        require(spec["source_episode_index"] == source and spec["round"] == round_index
                and spec["seed"] == train_seed(plan["master_seed"], source, round_index // 2)
                and spec["hand_condition"] == ("declared" if round_index % 2 == 0 else "inferred_effective")
                and spec["formal"] == FORMAL and spec["code_sha256_digest"] == digest(plan["code_sha256"])
                and spec["additional_dataset_quality"] == plan["additional_dataset_quality"]
                and spec["source"] == plan["sources"][str(source)]
                and spec["scene_proposal_sha256"] == proposal_file_sha(spec["scene_proposal"])
                and spec["latent_projector_batch_view"] == plan["runtime"]["latent_projector_batch_view"]
                and spec["latent_projection_phases"] == plan["runtime"]["latent_projection_phases"]
                and spec["placement_encoder_mode"] == plan["runtime"]["placement_encoder_mode"]
                and spec["placement_locomotion"] == plan["runtime"]["placement_locomotion"]
                and candidate["candidate_spec_sha256"] == digest(spec)
                and candidate["candidate_id"] == f"random100-source{source:03d}-{digest(spec)[:24]}"
                and candidate["candidate_id"] not in identities, "Candidate spec escaped its immutable TRAIN protocol")
        identities.add(candidate["candidate_id"])
    return plan


def prepare(args):
    from handover_random100_sources import RawSourceCatalog, inventory, load_raw_source
    from handover_random100_scene import randomize_config, MATERIAL_CATALOG
    from prepare_carrybox_v11 import ENCODER_SHA, OBSERVATION_SHA
    from handover_sonic_latent_projector import OFFICIAL_DECODER_SHA, DerivedBatchJacobianView
    import replay_handover_teacher as base
    output = Path(args.output).absolute()
    roots = {key: str(Path(getattr(args, key)).resolve(strict=True))
             for key in ("raw_root", "data_root", "sonic_resources", "python")}
    require(not output.exists() and not output.is_symlink() and not output.resolve().is_relative_to(REPO)
            and all(not output.resolve().is_relative_to(Path(value)) for key, value in roots.items() if key != "python"),
            "Use a fresh external queue outside original data/assets/code")
    requirements = read(args.requirements); packer.validate_requirements(requirements)
    require(requirements.get("minimum_distinct_accepted_original_sources") == 50
            and requirements["formal_scene_sampling"]["XY_max_m"] == .05
            and requirements["formal_scene_sampling"]["yaw_max_rad"] == .25,
            "Frozen formal scene bounds differ")
    require(type(args.candidates_per_source) is int and 2 <= args.candidates_per_source <= 20,
            "Use2..20 bounded candidates per source")
    catalog = RawSourceCatalog(args.raw_root); stock = inventory(catalog)
    eligible = [record["source_id"] for record in stock["records"] if record["status"] == "TRAIN_raw_reference_candidate"]
    old = read(Path(args.old_pack) / "PACK_READY.json"); order = source_order(eligible, old["covered_ancestors"])
    require(old.get("train_episodes") == 100 and old.get("train_target_satisfied") is True,
            "Priority list must come from the immutable original accepted100 pack")
    code = code_identities(); bindings = dict(stock["sources"])
    bindings.update(code)
    bindings.update(official_placement_sources())
    for path in (Path(args.requirements).resolve(strict=True), (Path(args.old_pack) / "PACK_READY.json").resolve(strict=True),
                 Path(roots["python"])): bindings[str(path)] = sha(path)
    sonic = Path(roots["sonic_resources"])
    for name, expected in (("model_encoder.onnx", ENCODER_SHA), ("model_decoder.onnx", OFFICIAL_DECODER_SHA),
                           ("observation_config.yaml", OBSERVATION_SHA)):
        require(sha(sonic / name) == expected, "Official SONIC identity mismatch: " + name)
        bindings[str(sonic / name)] = expected
    provenance = sonic / "PROVENANCE.json"
    require(read(provenance).get("revision") == "6733128a3d8a523b1418b06bca3cdf61c8b0987f", "Official SONIC provenance mismatch")
    bindings[str(provenance)] = sha(provenance)
    _, calibration = base.bound_calibration_assets(Path(roots["data_root"]), base.calibration_source_identities())
    bindings.update({str(Path(roots["data_root"]) / name): value for name, value in calibration.items()})
    view = str(Path(args.latent_projector_batch_view).resolve(strict=True)) if args.latent_projector_batch_view else None
    if view:
        DerivedBatchJacobianView(view)
        for path in (Path(view), Path(view + ".json")): bindings[str(path)] = sha(path)
    bindings[str(MATERIAL_CATALOG.resolve())] = sha(MATERIAL_CATALOG)
    sources, proposals = {}, {}
    for source_id in order:
        source = load_raw_source(source_id, catalog=catalog, asset_root=Path(roots["data_root"]))
        bindings.update(source["sources"])
        sources[str(source_id)] = {"source_dataset_key": source["evidence"]["source_dataset_key"],
            "source_episode_index": source_id, "frames": source["evidence"]["source_frames"],
            "parquet_sha256": source["evidence"]["parquet_sha256"], "evidence_sha256": digest(source["evidence"]),
            "source_files_sha256": source["sources"]}
        source_config = deepcopy(source["config"]); source_config["uid"] = "g1_wholebody_handover_teleop"
        source_config = base.remap_paths(source_config, Path(roots["data_root"]), Path(roots["data_root"]))
        for scene_round in range((args.candidates_per_source + 1) // 2):
            _, proposal = randomize_config(source_config, train_seed(args.seed, source_id, scene_round),
                Path(roots["data_root"]), xy_max_m=.05, yaw_max_rad=.25)
            proposals[source_id, scene_round] = proposal
            bindings[str(Path(proposal["material"]["table_material"]["path"]).resolve())] = proposal["material"]["table_mdl_sha256"]
    plan = {"contract": PLAN_CONTRACT, "created_utc": now(), "master_seed": args.seed,
        "target": 100, "minimum_distinct_sources": 50, "maximum_registered_per_source": 2,
        "TRAIN_seed_upper_exclusive": 2**30, "TRAIN_seed_minimum": 1,
        "additional_dataset_quality": deepcopy(packer.ADDITIONAL_DATASET_QUALITY),
        "formal": dict(FORMAL), "candidates_per_source": args.candidates_per_source,
        "source_order": order, "legacy24_source_ids": sorted(old["covered_ancestors"]), "sources": sources,
        "runtime": {**roots, "latent_projector_batch_view": view, "latent_projection_phases": args.latent_projection_phases,
                    "placement_encoder_mode": args.placement_encoder_mode,
                    "placement_locomotion": args.placement_locomotion}, "code_sha256": code,
        "source_bindings": bindings, "requirements_sha256": sha(args.requirements),
        "raw_inventory_sha256": digest(stock), "new_independent_human_sessions_claimed": False,
        "bootstrap_required_before_formal_run": True, "candidates": []}
    for round_index in range(args.candidates_per_source):
        for source_id in order:
            proposal = proposals[source_id, round_index // 2]
            spec = {"source_episode_index": source_id, "round": round_index,
                "seed": train_seed(args.seed, source_id, round_index // 2),
                "hand_condition": "declared" if round_index % 2 == 0 else "inferred_effective",
                "additional_dataset_quality": deepcopy(packer.ADDITIONAL_DATASET_QUALITY),
                "formal": dict(FORMAL), "source": sources[str(source_id)], "code_sha256_digest": digest(code),
                "scene_proposal_sha256": proposal_file_sha(proposal), "scene_proposal": proposal,
                "latent_projector_batch_view": view, "latent_projection_phases": args.latent_projection_phases,
                "placement_encoder_mode": args.placement_encoder_mode, "placement_locomotion": args.placement_locomotion}
            identity = digest(spec)
            plan["candidates"].append({"candidate_id": f"random100-source{source_id:03d}-{identity[:24]}",
                "candidate_spec_sha256": identity, "spec": spec})
    validate_plan(plan)
    plan["source_stat_identity"] = {path: stat_identity(path) for path in bindings}
    output.mkdir(parents=True)
    write(output / "candidate_plan.json", plan, exclusive=True)
    shutil.copyfile(args.requirements, output / "REBUILD_REQUIREMENTS.json")
    write(output / "raw_source_inventory.json", stock, exclusive=True)
    for path, expected in code.items():
        destination = output / "frozen-code" / Path(path).relative_to(REPO)
        destination.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, destination)
        require(sha(destination) == expected, "Executable changed while freezing")
    require(all(sha(path) == expected for path, expected in code.items()), "Code drifted during preparation")
    require(sha(output / "REBUILD_REQUIREMENTS.json") == plan["requirements_sha256"], "Frozen requirements copy changed")
    verify_frozen(plan)
    state = {"contract": QUEUE_CONTRACT, "plan_sha256": sha(output / "candidate_plan.json"), "created_utc": now(),
        "status": "pilot_pending", "attempts": [], "qualified": [], "diversity_excluded_candidates": [],
        "registry_versions": [], "bootstrap": None, "collection_ready": False}
    write(output / "queue.json", state, exclusive=True)
    return summary(state, plan)


def verify_frozen(plan, *, full=False):
    require(all(sha(path) == expected for path, expected in plan["code_sha256"].items()),
            "Frozen executable drift: stop new dispatch immediately")
    require(all(stat_identity(path) == expected for path, expected in plan["source_stat_identity"].items()),
            "Frozen source/model/resource identity drift: stop new dispatch immediately")
    if full: packer.verify_bindings(plan["source_bindings"])


def verified_queue(root, *, full=False):
    root = Path(root).resolve(strict=True); plan = validate_plan(read(root / "candidate_plan.json")); state = read(root / "queue.json")
    require(state.get("contract") == QUEUE_CONTRACT and state["plan_sha256"] == sha(root / "candidate_plan.json")
            and sha(root / "REBUILD_REQUIREMENTS.json") == plan["requirements_sha256"]
            and digest(read(root / "raw_source_inventory.json")) == plan["raw_inventory_sha256"],
            "Queue/plan/requirements/raw inventory identity differs")
    verify_frozen(plan, full=full)
    for path, expected in plan["code_sha256"].items():
        require(sha(root / "frozen-code" / Path(path).relative_to(REPO)) == expected, "Frozen code snapshot changed")
    candidates = {candidate["candidate_id"]: candidate for candidate in plan["candidates"]}
    captures, all_bindings = {}, {}
    for index, attempt in enumerate(state["attempts"]):
        candidate = candidates.get(attempt.get("candidate_id")); phase = attempt.get("phase")
        require(candidate is not None and phase in ("screen", "capture") and attempt["attempt_id"] == index
                and attempt["directory"] == f"attempts/{index:06d}-{phase}"
                and attempt["candidate_spec_sha256"] == candidate["candidate_spec_sha256"]
                and attempt["source_episode_index"] == candidate["spec"]["source_episode_index"]
                and attempt["status"] in ("running", "passed", "rejected", "interrupted"),
                "Persisted attempt escaped its frozen candidate/owned directory")
        if attempt["status"] != "passed": continue
        audit_path = Path(attempt["admission_path"]).resolve(strict=True)
        require(audit_path == (root / attempt["directory"] / "admission.json").resolve()
                and sha(audit_path) == attempt["admission_sha256"], "Prior owned admission changed")
        admission = read(audit_path); validate_phase_audit(admission, phase)
        outcome_path = root / attempt["outcome_path"]
        require(outcome_path == root / attempt["directory"] / "outcome.json"
                and sha(outcome_path) == attempt["outcome_sha256"], "Prior owned outcome changed")
        if phase == "capture": captures[attempt["candidate_id"]] = attempt
        if full:
            for group in ("bindings", "source_sha256"):
                for path, expected in admission[group].items():
                    require(path not in all_bindings or all_bindings[path] == expected, "Conflicting prior admission identities")
                    all_bindings[path] = expected
    require(len({row["candidate_id"] for row in state["qualified"]}) == len(state["qualified"]), "Qualified pool repeats a candidate")
    for entry in state["qualified"]:
        attempt = captures.get(entry["candidate_id"])
        require(attempt is not None and entry["admission_path"] == attempt["admission_path"]
                and entry["admission_sha256"] == attempt["admission_sha256"] and entry["run"] == attempt["run"],
                "Qualified pool has no strict owned RTX attempt")
        admission = read(entry["admission_path"])
        packer.validate_admission(admission, entry)
        require(entry["capture_content_sha256"] == admission["RGB_state_action_history_audit"]["capture_content_sha256"],
                "Qualified pool capture identity differs from actual strict admission")
    if full and all_bindings: packer.verify_bindings(all_bindings)
    return state, plan


def summary(state, plan):
    counts = Counter(row["source_episode_index"] for row in state["qualified"])
    outcomes = Counter(f"{row['phase']}_{row['status']}" for row in state["attempts"])
    return {"contract": QUEUE_CONTRACT, "status": state["status"], "collection_ready": state["collection_ready"],
        "planned_candidates": len(plan["candidates"]), "all_attempts_including_failures": len(state["attempts"]),
        "outcomes": dict(outcomes), "strict_RTX_qualified_pool": len(state["qualified"]),
        "distinct_strict_RTX_sources": len(counts), "registered_episodes": len(state.get("registry_entries", [])),
        "screen_passes_are_new_data": False, "ready_for_A100_sync": False,
        "placement_encoder_mode": plan["runtime"]["placement_encoder_mode"],
        "placement_locomotion": plan["runtime"]["placement_locomotion"],
        "DINO_and_production_preflight_still_required": True, "bootstrap_validated": bool(state.get("bootstrap"))}


def validate_result(result, args, candidate, plan, phase):
    spec = candidate["spec"]
    require(result.get("contract") == "cvpr-handover-random100-online-execution-v1"
            and result.get("status") == "execution_complete"
            and result.get("code_unchanged_during_run") is True
            and result.get("expert_dependencies_unchanged_during_run") is True
            and result.get("calibration_source_assets_unchanged") is True
            and result.get("commands_generated_online") is True and result.get("source_tokens_reencoded") is True
            and result.get("accepted_new_dataset") is False and result.get("object_restores_during_control") is False
            and result.get("source_split") == "train" and result.get("source_episode_index") == spec["source_episode_index"]
            and result.get("hand_command_condition") == spec["hand_condition"],
            "Child exit code is insufficient: actual complete frozen online execution evidence missing")
    quality = result.get("physical_quality", {})
    require(quality.get("acceptance_gate_version") == "handover-physics-v2"
            and quality.get("eligible_physics") is True and quality.get("verified_handover_success") is True,
            "Actual child physical gate failed")
    code = result.get("capture_code_sha256", {})
    frozen_execution = {**plan.get("source_bindings", {}), **plan["code_sha256"]}
    require(set(REQUIRED_RUN_CODE) <= {Path(path).name for path in code}
            and all(frozen_execution.get(path) == expected for path, expected in code.items()),
            "Actual capture did not execute the frozen source")
    mode = plan["runtime"]["placement_encoder_mode"]
    require(args.get("placement_encoder_mode", "g1") == result.get("placement_encoder_mode", "g1") == mode,
            "Actual placement encoder mode differs from the frozen bootstrap/candidate")
    require(args.get("placement_locomotion", "source_future") == result.get("placement_locomotion", "source_future")
            == plan["runtime"]["placement_locomotion"], "Actual placement locomotion differs from frozen bootstrap/candidate")
    if mode == "teleop":
        helper = str((DIRECTORY / "handover_sonic_teleop_reference.py").resolve(strict=True))
        required_mode1 = {helper: plan["code_sha256"].get(helper), **official_placement_sources()}
        require(all(value is not None and frozen_execution.get(path) == value and code.get(path) == value
                    for path, value in required_mode1.items()),
                "Teleop actual execution must snapshot helper and all four frozen official sources")
    for key in ("source_episode", "seed", "hand_condition"):
        expected = spec["source_episode_index"] if key == "source_episode" else spec[key]
        require(args.get(key) == expected, "Actual source/seed/hand condition differs")
    require(all(Path(args.get(key, "")).resolve() == Path(plan["runtime"][key]).resolve()
                for key in ("raw_root", "data_root", "sonic_resources"))
            and result.get("source_sha256") == spec["source"]["source_files_sha256"],
            "Actual raw source/assets/model roots differ from the frozen candidate")
    view = plan["runtime"]["latent_projector_batch_view"]
    expected_dependencies = {name: plan["source_bindings"][name] for name in (view, view + ".json")} if view else {}
    require(result.get("expert_dependency_sha256") == expected_dependencies, "Actual optional projection dependencies differ")
    require(all(args.get(key) == value for key, value in FORMAL.items())
            and args.get("scene_only") is False
            and args.get("render_mode") == ("none" if phase == "screen" else "isaac")
            and args.get("latent_projector_batch_view") == plan["runtime"]["latent_projector_batch_view"],
            "Actual runtime formal bounds/mode/projection differs")
    require(args.get("latent_projection_phases") == plan["runtime"]["latent_projection_phases"],
            "Actual latent projection phases differ from the frozen candidate")
    require(result.get("scene_proposal_sha256") == spec["scene_proposal_sha256"], "Actual scene changed between screening and RTX")


def validate_phase_audit(audit, phase):
    packer.validate_full_footprint(audit.get("actual_full_outer_footprint_final50", {}))
    require(audit.get("contract") == packer.ADMISSION_CONTRACT and audit.get("replay_expert") is True
            and audit.get("expert_reproduction", {}).get("status") == "all_generated_commands_reproduced_on_CPU"
            and audit.get("physical_gate", {}).get("eligible_physics") is True
            and audit.get("physical_gate", {}).get("verified_handover_success") is True,
            "Independent actual saved-prestate expert/physics reproduction failed")
    if phase == "screen":
        require(audit.get("status") == "physics_passed_full_dataset_admission_pending"
                and audit.get("eligible_dataset") is False and audit.get("require_rgb") is False,
                "CPU screening cannot certify a new training demonstration")
    else:
        require(audit.get("status") == "independent_capture_audit_passed"
                and audit.get("eligible_dataset") is True and audit.get("require_rgb") is True,
                "RTX did not pass strict complete dataset admission")


def validate_bootstrap(path, plan):
    from audit_handover_random100 import audit as audit_run
    path = Path(path).resolve(strict=True); previous = read(path)
    require(previous.get("eligible_dataset") is True and previous.get("require_rgb") is True
            and previous.get("replay_expert") is True, "A full strict accepted RTX bootstrap is required before formal dispatch")
    run = Path(previous["run"]).resolve(strict=True)
    args, result = read(run / "execution_args.json"), read(run / "result.json")
    source = args["source_episode"]
    require(source in plan["source_order"] and 1 <= args["seed"] < 2**30
            and all(args.get(key) == value for key, value in FORMAL.items()), "Bootstrap must use complete TRAIN and formal bounds")
    candidate = {"spec": {"source_episode_index": source, "seed": args["seed"], "hand_condition": args["hand_condition"],
                           "source": plan["sources"][str(source)],
                           "scene_proposal_sha256": result["scene_proposal_sha256"]}}
    validate_result(result, args, candidate, plan, "capture")
    expected_auditor = plan["code_sha256"][str((DIRECTORY / "audit_handover_random100.py").resolve())]
    require(previous.get("auditor_sha256") == expected_auditor, "Bootstrap used a different auditor version")
    actual = audit_run(run, require_rgb=True, replay_expert=True); validate_phase_audit(actual, "capture")
    stable = lambda value: {key: item for key, item in value.items() if key != "generated_utc"}
    require(stable(previous) == stable(actual), "Bootstrap no longer matches full strict independent recomputation")
    verify_frozen(plan)
    return {"admission_path": str(path), "admission_sha256": sha(path), "run": str(run),
            "source_episode_index": source, "validated_utc": now(), "counted_as_formal_data": False}


def execution_command(plan, candidate, run, phase):
    runtime, spec = plan["runtime"], candidate["spec"]
    command = [runtime["python"], str(RUNNER), "--raw-root", runtime["raw_root"], "--data-root", runtime["data_root"],
        "--sonic-resources", runtime["sonic_resources"], "--source-episode", str(spec["source_episode_index"]),
        "--seed", str(spec["seed"]), "--hand-condition", spec["hand_condition"], "--xy-max-m", ".05",
        "--yaw-max-rad", ".25", "--max-seconds", "30", "--render-mode", "none" if phase == "screen" else "isaac",
        "--latent-projection-phases", runtime["latent_projection_phases"],
        "--placement-encoder-mode", runtime["placement_encoder_mode"],
        "--placement-locomotion", runtime["placement_locomotion"], "--output", str(run)]
    if runtime["latent_projector_batch_view"]:
        command += ["--latent-projector-batch-view", runtime["latent_projector_batch_view"]]
    return command


def child_environment(phase, gpu_device):
    simple = REPO / "VLMExpert/third_party/SIMPLE"
    paths = [DIRECTORY, simple / "src", simple / "third_party", REPO]
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1", OMP_NUM_THREADS="1",
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMNI_KIT_ACCEPT_EULA="YES",
        PYTHONPATH=os.pathsep.join(map(str, paths)), MUJOCO_GL="egl")
    environment["CUDA_VISIBLE_DEVICES"] = "" if phase == "screen" else str(gpu_device)
    return environment


def worker(root, candidate_id, attempt_id, phase, timeout):
    """One owned child; retain its result and independent rejection even exit0."""
    root = Path(root).resolve(strict=True); plan = validate_plan(read(root / "candidate_plan.json"))
    candidate = next(item for item in plan["candidates"] if item["candidate_id"] == candidate_id)
    directory = root / "attempts" / f"{attempt_id:06d}-{phase}"
    require(directory.is_dir() and not (directory / "outcome.json").exists(), "Invalid owned attempt directory")
    outcome = {"candidate_id": candidate_id, "attempt_id": attempt_id, "phase": phase, "passed": False, "started_utc": now()}
    run = directory / "run"; audit_path = directory / "admission.json"
    try:
        # The parent persists this PID before authorizing execution. A parent
        # crash in Popen's tiny registration window can leave only an idle
        # worker, never an unregistered RTX renderer running concurrently.
        identity = {"pid": os.getpid(), "process_start_ticks": process_identity(os.getpid()),
                    "candidate_id": candidate_id, "phase": phase}
        write(directory / "worker_identity.json", identity, exclusive=True)
        deadline = time.monotonic() + 30
        authorization = directory / "dispatch_authorized.json"
        while not authorization.is_file() and time.monotonic() < deadline: time.sleep(.1)
        require(authorization.is_file() and read(authorization) == identity, "Parent did not authorize this registered owned worker")
        verify_frozen(plan)
        with (directory / "execution.log").open("x") as stream:
            result = subprocess.run(execution_command(plan, candidate, run, phase), env=dict(os.environ),
                stdout=stream, stderr=subprocess.STDOUT, timeout=timeout, cwd=REPO)
        outcome["child_returncode"] = result.returncode
        # Always read persisted execution authority. Isaac shutdown can turn a
        # Python exception into exit0; a status/result/quality check is required.
        actual_result, args = read(run / "result.json"), read(run / "execution_args.json")
        validate_result(actual_result, args, candidate, plan, phase)
        require(result.returncode == 0, "Owned child exited nonzero despite its result ledger")
        from audit_handover_random100 import audit as audit_run
        admission = audit_run(run, require_rgb=phase == "capture", replay_expert=True)
        write(audit_path, admission, exclusive=True); validate_phase_audit(admission, phase)
        verify_frozen(plan)
        outcome.update(passed=True, run=str(run), admission_path=str(audit_path), admission_sha256=sha(audit_path),
                       scene_proposal_sha256=actual_result["scene_proposal_sha256"])
    except Exception as error:
        outcome["error"] = f"{type(error).__name__}: {error}"
    outcome["ended_utc"] = now(); write(directory / "outcome.json", outcome, exclusive=True)
    return outcome


def process_identity(pid):
    try:
        # /proc comm can contain spaces/parentheses: fields start after its last).
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]
    except (OSError, IndexError): return None


def finalize_attempt(root, state, attempt, plan):
    directory = root / attempt["directory"]; outcome_path = directory / "outcome.json"
    if not outcome_path.is_file():
        if process_identity(attempt.get("pid", -1)) == attempt.get("process_start_ticks") and attempt.get("process_start_ticks") is not None:
            return False
        attempt.update(status="interrupted", ended_utc=now(), error="Owned worker ended without complete outcome; files retained")
        return True
    outcome = read(outcome_path)
    require(outcome["candidate_id"] == attempt["candidate_id"] and outcome["attempt_id"] == attempt["attempt_id"]
            and outcome["phase"] == attempt["phase"], "Owned outcome identity differs")
    attempt.update(status="passed" if outcome["passed"] else "rejected", ended_utc=now(),
                   outcome_path=str(outcome_path.relative_to(root)), outcome_sha256=sha(outcome_path))
    if outcome.get("error"): attempt["error"] = outcome["error"]
    if outcome["passed"]:
        candidate = next(c for c in plan["candidates"] if c["candidate_id"] == attempt["candidate_id"])
        audit = read(outcome["admission_path"])
        require(sha(outcome["admission_path"]) == outcome["admission_sha256"], "Owned admission changed")
        validate_phase_audit(audit, attempt["phase"])
        attempt.update(admission_path=outcome["admission_path"], admission_sha256=outcome["admission_sha256"], run=outcome["run"])
        if attempt["phase"] == "capture":
            spec = candidate["spec"]
            entry = {"candidate_id": candidate["candidate_id"], "source_episode_index": spec["source_episode_index"],
                "seed": spec["seed"], "hand_condition": spec["hand_condition"], "run": outcome["run"],
                "admission_path": outcome["admission_path"], "admission_sha256": outcome["admission_sha256"],
                "scene_proposal_sha256": outcome["scene_proposal_sha256"]}
            packer.validate_admission(audit, entry)
            identity = audit["RGB_state_action_history_audit"]["capture_content_sha256"]
            if (candidate["candidate_id"] in {e["candidate_id"] for e in state["qualified"]}
                    or identity in {e["capture_content_sha256"] for e in state["qualified"]}):
                attempt.update(status="rejected", error="Duplicate actual RTX capture content; strict report retained, not counted")
                return True
            state["qualified"].append({**entry, "capture_content_sha256": identity})
    return True


def last_attempts(state, phase):
    return {row["candidate_id"]: row for row in state["attempts"] if row["phase"] == phase}


def choose_screen(plan, state):
    previous = last_attempts(state, "screen")
    tried_sources = {row["source_episode_index"] for row in previous.values()}
    covered = {row["source_episode_index"] for row in state["qualified"]
               if row["candidate_id"] not in state["diversity_excluded_candidates"]}
    available = [c for c in plan["candidates"] if c["candidate_id"] not in previous
                 or previous[c["candidate_id"]]["status"] == "interrupted"]
    if len(tried_sources) < 94:
        available = [c for c in available if c["spec"]["source_episode_index"] not in tried_sources]
    return sorted(available, key=lambda c: (c["spec"]["source_episode_index"] in covered, c["spec"]["round"],
                                           plan["source_order"].index(c["spec"]["source_episode_index"])))


def choose_capture(plan, state):
    screens, captures = last_attempts(state, "screen"), last_attempts(state, "capture")
    counts = Counter(row["source_episode_index"] for row in state["qualified"]
                     if row["candidate_id"] not in state["diversity_excluded_candidates"])
    source_by_candidate = {c["candidate_id"]: c for c in plan["candidates"]}
    used_scenes = {row["scene_proposal_sha256"] for row in state["qualified"]
                   if row["candidate_id"] not in state["diversity_excluded_candidates"]}
    available = [source_by_candidate[cid] for cid, row in screens.items() if row["status"] == "passed"
        and (cid not in captures or captures[cid]["status"] == "interrupted")
        and source_by_candidate[cid]["spec"]["scene_proposal_sha256"] not in used_scenes]
    missing = [c for c in available if c["spec"]["source_episode_index"] not in counts]
    if missing: return missing
    # Finish each original source's first real screening, then exhaust remaining
    # opportunities for uncovered ancestors before spending RTX on repeats.
    if len(counts) < 50 or len({row["source_episode_index"] for row in screens.values()}) < 94:
        return []
    if any(c["spec"]["source_episode_index"] not in counts for c in choose_screen(plan, state)):
        return []
    return sorted(available, key=lambda c: (counts[c["spec"]["source_episode_index"]], c["spec"]["round"], c["candidate_id"]))


def select_registry(qualified, excluded=()):
    excluded = set(excluded); available = [e for e in qualified if e["candidate_id"] not in excluded]
    selected, used, seen, scenes = [], Counter(), set(), set()
    # First pass maximizes original sources. Only the second pass adds repeats.
    for pass_index in (0, 1):
        for entry in available:
            source = entry["source_episode_index"]
            if len(selected) == 100: break
            if used[source] != pass_index or entry["candidate_id"] in seen or entry["scene_proposal_sha256"] in scenes: continue
            selected.append({key: value for key, value in entry.items() if key != "capture_content_sha256"}); used[source] += 1
            seen.add(entry["candidate_id"])
            scenes.add(entry["scene_proposal_sha256"])
    return selected


def update_registry(root, state, plan):
    if state["collection_ready"]: return True
    entries = select_registry(state["qualified"], state["diversity_excluded_candidates"])
    state["registry_entries"] = entries
    selected_ids = {row["candidate_id"] for row in entries}
    selected_scenes = {row["scene_proposal_sha256"] for row in entries}
    state["excluded_same_scene"] = [row["candidate_id"] for row in state["qualified"]
                                    if row["candidate_id"] not in selected_ids and row["scene_proposal_sha256"] in selected_scenes]
    if len(entries) != 100:
        return False
    requirements = read(root / "REBUILD_REQUIREMENTS.json"); packer.validate_selection(entries, requirements)
    selection_sha = digest(entries)
    if any(row["selection_sha256"] == selection_sha for row in state["registry_versions"]):
        return False
    from diversity_handover_random100 import audit as audit_diversity
    version = len(state["registry_versions"])
    while any((root / "registry_versions" / f"{kind}-{version:04d}.json").exists() for kind in ("registry", "diversity")):
        version += 1
    path = root / "registry_versions" / f"registry-{version:04d}.json"
    registry = {"contract": packer.REGISTRY_CONTRACT, "requirements_path": str(root / "REBUILD_REQUIREMENTS.json"),
        "requirements_sha256": plan["requirements_sha256"], "admissions": entries,
        "plan_sha256": state["plan_sha256"], "all_attempts_including_failures": len(state["attempts"])}
    write(path, registry, exclusive=True)
    report = audit_diversity(path)
    report_path = path.with_name(f"diversity-{version:04d}.json"); write(report_path, report, exclusive=True)
    record = {"registry_path": str(path), "registry_sha256": sha(path), "selection_sha256": selection_sha,
              "diversity_path": str(report_path), "diversity_sha256": sha(report_path), "passed": report.get("passed") is True}
    state["registry_versions"].append(record)
    # A display/current registry may change after a rejection; every diversity
    # report binds its own immutable version, preserving all earlier evidence.
    write(root / "registry.json", registry)
    if report.get("passed") is True:
        packer.validate_diversity(report, plan["requirements_sha256"], sha(path), entries)
        packer.verify_bindings(report["bindings"]); verify_frozen(plan)
        state.update(collection_ready=True, status="quality_and_diversity_passed_pack_and_DINO_pending")
        write(root / "COLLECTION_READY.json", {"contract": "cvpr-handover-random100-collection-ready-v1",
            **record, "episodes": 100, "quality_and_diversity_passed": True,
            "ready_for_pack": True, "ready_for_A100_sync": False,
            "DINO_and_production_preflight_pending": True, "upload_performed": False}, exclusive=True)
        return True
    replacements = report.get("replacement_candidate_ids", [])
    require(isinstance(replacements, list) and set(replacements) <= {e["candidate_id"] for e in entries},
            "Diversity replacement advice must reference current registered candidates")
    state["diversity_excluded_candidates"] = sorted(set(state["diversity_excluded_candidates"]) | set(replacements))
    state["status"] = "diversity_rejected_additional_strict_captures_required"
    return False


def restore_collection_ready(root, state, plan):
    path = root / "COLLECTION_READY.json"
    if not path.exists(): return False
    ready = read(path)
    require(ready.get("contract") == "cvpr-handover-random100-collection-ready-v1"
            and ready.get("quality_and_diversity_passed") is True and ready.get("episodes") == 100,
            "Incomplete collection-ready receipt cannot resume as success")
    registry_path, diversity_path = Path(ready["registry_path"]).resolve(strict=True), Path(ready["diversity_path"]).resolve(strict=True)
    require(registry_path.is_relative_to(root / "registry_versions") and diversity_path.is_relative_to(root / "registry_versions")
            and sha(registry_path) == ready["registry_sha256"] and sha(diversity_path) == ready["diversity_sha256"],
            "Immutable final registry/diversity identity changed")
    registry, diversity = read(registry_path), read(diversity_path)
    entries = registry["admissions"]; pool = {row["candidate_id"]: row for row in state["qualified"]}
    packer.validate_selection(entries, read(root / "REBUILD_REQUIREMENTS.json"))
    require(all(all(pool.get(row["candidate_id"], {}).get(key) == value for key, value in row.items()) for row in entries),
            "Final registry escapes the queue's real strict RTX qualified pool")
    packer.validate_diversity(diversity, plan["requirements_sha256"], sha(registry_path), entries)
    packer.verify_bindings(diversity["bindings"])
    state.update(collection_ready=True, status="quality_and_diversity_passed_pack_and_DINO_pending", registry_entries=entries)
    return True


def launch(root, state, plan, candidate, phase, timeout, gpu_device):
    verify_frozen(plan)
    index = len(state["attempts"])
    relative = f"attempts/{index:06d}-{phase}"; directory = root / relative; directory.mkdir(parents=True, exist_ok=False)
    command = [plan["runtime"]["python"], str(Path(__file__).resolve()), "_worker", "--queue", str(root),
        "--candidate-id", candidate["candidate_id"], "--attempt-id", str(index), "--phase", phase, "--timeout", str(timeout)]
    attempt = {"attempt_id": index, "candidate_id": candidate["candidate_id"],
        "candidate_spec_sha256": candidate["candidate_spec_sha256"], "source_episode_index": candidate["spec"]["source_episode_index"],
        "phase": phase, "status": "running", "directory": relative, "started_utc": now(),
        "pid": None, "process_start_ticks": None, "timeout_s": timeout, "worker_command": command}
    state["attempts"].append(attempt); write(root / "queue.json", state)
    stream = (directory / "worker.log").open("x")
    try:
        process = subprocess.Popen(command, env=child_environment(phase, gpu_device), stdout=stream, stderr=subprocess.STDOUT,
                                   cwd=REPO, start_new_session=True)
    except Exception:
        attempt.update(status="interrupted", ended_utc=now(), error="Owned worker could not be launched")
        write(root / "queue.json", state)
        raise
    finally: stream.close()
    attempt.update(pid=process.pid, process_start_ticks=process_identity(process.pid)); write(root / "queue.json", state)
    write(directory / "dispatch_authorized.json", {"pid": process.pid, "process_start_ticks": attempt["process_start_ticks"],
          "candidate_id": candidate["candidate_id"], "phase": phase}, exclusive=True)
    return process, attempt, time.monotonic()


def wait_owned(root, state, plan, active):
    drift = None
    while active:
        try: verify_frozen(plan)
        except Exception as error:
            drift = str(error)
            # Do not dispatch anything else. Existing owned work may finish and
            # remains preserved; unchanged-code validation prevents admission.
        remaining = []
        for process, attempt, started in active:
            # Worker includes execution and CPU independent audit; allow the
            # same bounded budget again for audit, then terminate only its owned
            # process group. Never search/kill another training process.
            if process.poll() is None and time.monotonic() - started > 2 * attempt["timeout_s"] + 60:
                os.killpg(process.pid, signal.SIGTERM)
                try: process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL); process.wait()
                attempt["error"] = "Owned worker exceeded execution-plus-audit budget; retained incomplete evidence"
            if process.poll() is not None:
                finalize_attempt(root, state, attempt, plan)
                write(root / "queue.json", state)
                print(json.dumps({"candidate": attempt["candidate_id"], "phase": attempt["phase"],
                                  "status": attempt["status"], "strict_RTX_pool": len(state["qualified"])}), flush=True)
            else: remaining.append((process, attempt, started))
        active = remaining
        if active: time.sleep(.25)
    if drift:
        state.update(status="frozen_source_drift_dispatch_stopped", dispatch_error=drift)
        write(root / "queue.json", state)
        raise RuntimeError(drift)


def run(root, *, bootstrap_admission=None, phase="all", workers=4, maximum_screen=564, maximum_capture=120,
        timeout=900, gpu_device="0"):
    require(phase in ("all", "screen", "capture") and type(workers) is int and 1 <= workers <= 8
            and type(maximum_screen) is int and 0 <= maximum_screen <= 2000
            and type(maximum_capture) is int and 0 <= maximum_capture <= 2000
            and 10 <= timeout <= 7200, "Use bounded owned worker budgets")
    root = Path(root).resolve(strict=True)
    with lock(root):
        state, plan = verified_queue(root, full=True)
        for attempt in state["attempts"]:
            if attempt["status"] == "running": finalize_attempt(root, state, attempt, plan)
        if any(row["status"] == "running" for row in state["attempts"]):
            state["status"] = "prior_owned_workers_still_running_no_new_dispatch"
            write(root / "queue.json", state); return summary(state, plan)
        if restore_collection_ready(root, state, plan):
            write(root / "queue.json", state); return summary(state, plan)
        require(state["collection_ready"] is False, "Queue claims ready without immutable strict collection receipt")
        bootstrap = bootstrap_admission or (state.get("bootstrap") or {}).get("admission_path")
        require(bootstrap is not None, "Formal dispatch blocked: prepare is pilot_pending until a current full strict RTX bootstrap passes")
        state["bootstrap"] = validate_bootstrap(bootstrap, plan)
        state["status"] = "collecting_new_executions"; write(root / "queue.json", state)
        screened = captured = 0
        while not state["collection_ready"]:
            try: verify_frozen(plan)
            except Exception as error:
                state.update(status="frozen_source_drift_dispatch_stopped", dispatch_error=str(error))
                write(root / "queue.json", state); raise
            if update_registry(root, state, plan): break
            captures = choose_capture(plan, state) if phase != "screen" and captured < maximum_capture else []
            if captures:
                active = [launch(root, state, plan, captures[0], "capture", timeout, gpu_device)]
                captured += 1; wait_owned(root, state, plan, active); continue
            screens = choose_screen(plan, state) if phase != "capture" and screened < maximum_screen else []
            if screens:
                active = []
                for candidate in screens[:min(workers, maximum_screen - screened)]:
                    try:
                        active.append(launch(root, state, plan, candidate, "screen", timeout, gpu_device)); screened += 1
                    except Exception as error:
                        state.update(status="new_dispatch_stopped", dispatch_error=str(error)); write(root / "queue.json", state)
                        if active: wait_owned(root, state, plan, active)
                        raise
                wait_owned(root, state, plan, active); continue
            break
        update_registry(root, state, plan)
        if not state["collection_ready"] and state["status"] == "collecting_new_executions":
            state["status"] = "budget_or_candidate_pool_exhausted_not_ready"
        state["last_call"] = {"screen_attempts": screened, "RTX_attempts": captured, "ended_utc": now()}
        write(root / "queue.json", state)
        return summary(state, plan)


def status(root):
    root = Path(root).resolve(strict=True)
    try:
        state, plan = verified_queue(root)
        return summary(state, plan)
    except Exception as error:
        # Read-only status remains useful after source drift; it never quietly
        # resumes the queue or rewrites its old evidence.
        state, plan = read(root / "queue.json"), read(root / "candidate_plan.json")
        return {**summary(state, plan), "collection_ready": False,
                "recorded_collection_ready_before_identity_failure": state["collection_ready"],
                "frozen_identity_valid": False, "error": str(error)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    p = modes.add_parser("prepare")
    for flag in ("requirements", "old-pack", "raw-root", "data-root", "sonic-resources", "python", "output"):
        p.add_argument("--" + flag, required=True, type=Path)
    p.add_argument("--seed", type=int, default=20261024)
    p.add_argument("--candidates-per-source", type=int, default=6)
    p.add_argument("--latent-projector-batch-view", type=Path)
    p.add_argument("--latent-projection-phases", default="placement")
    p.add_argument("--placement-encoder-mode", choices=("g1", "teleop"), default="g1")
    p.add_argument("--placement-locomotion", choices=("source_future", "actual_hold"), default="source_future")
    p = modes.add_parser("run"); p.add_argument("--queue", required=True, type=Path)
    p.add_argument("--bootstrap-admission", type=Path)
    p.add_argument("--phase", choices=("all", "screen", "capture"), default="all")
    p.add_argument("--workers", type=int, default=4); p.add_argument("--max-screen", type=int, default=564)
    p.add_argument("--max-capture", type=int, default=120); p.add_argument("--timeout", type=float, default=900)
    p.add_argument("--gpu-device", default="0")
    p = modes.add_parser("status"); p.add_argument("--queue", required=True, type=Path)
    p = modes.add_parser("_worker", help=argparse.SUPPRESS)
    p.add_argument("--queue", required=True, type=Path); p.add_argument("--candidate-id", required=True)
    p.add_argument("--attempt-id", required=True, type=int); p.add_argument("--phase", choices=("screen", "capture"), required=True)
    p.add_argument("--timeout", type=float, required=True)
    args = parser.parse_args()
    if args.mode == "prepare": result = prepare(args)
    elif args.mode == "status": result = status(args.queue)
    elif args.mode == "_worker": result = worker(args.queue, args.candidate_id, args.attempt_id, args.phase, args.timeout)
    else:
        result = run(args.queue, bootstrap_admission=args.bootstrap_admission, phase=args.phase, workers=args.workers,
            maximum_screen=args.max_screen, maximum_capture=args.max_capture, timeout=args.timeout, gpu_device=args.gpu_device)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__": main()
