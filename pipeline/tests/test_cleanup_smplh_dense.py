"""Deletion gates exercised against real files, Parquet footers and durable receipts."""

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


SCRIPT = Path(__file__).parents[1] / "scripts/cleanup_smplh_dense.py"
spec = importlib.util.spec_from_file_location("cleanup_smplh_dense_under_test", SCRIPT)
cleanup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cleanup)


def put(path, value):
    cleanup.write(path, value)


def blob(path, value=b"preserved binary"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return path


def parquet(path, column="observation.state.body_world_rot6d_xyz"):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({column: [[1.0, 2.0, 3.0]]}), path)
    return path


@pytest.fixture
def accepted(tmp_path, monkeypatch):
    """Small two-source population with the same full-gate shape as production."""
    monkeypatch.setattr(cleanup, "EXPECTED_POPULATION", {
        "egosuite": {"episodes": 2, "frames": 8},
        "egoexo4d_hm": {"episodes": 1, "frames": 4}})
    monkeypatch.setattr(cleanup, "live_processes", lambda: [])
    mix = tmp_path / "MixEgoMotion"
    root = mix / "smplh-parameters-v1"
    job = mix / ".pipeline/jobs/parameters"
    fitjob = mix / ".pipeline/jobs/fit"
    fitted = mix / "derived/smplh-body22-sequence-v2/egosuite"
    job.mkdir(parents=True); fitjob.mkdir(parents=True)
    model = blob(root / "weights/smplh/male/model.npz", b"actual16-test-model")
    monkeypatch.setattr(cleanup, "MODEL_SHA256", cleanup.sha(model))
    hand = blob(root / "weights/smplh/hand_basis.npz", b"hand basis")
    raw = blob(tmp_path / "raw/sequence.mcap", b"raw mcap")
    hmraw = blob(tmp_path / "hm/raw.npz", b"original hm")
    hmdino = blob(tmp_path / "hm/DINO/source.parquet", b"original hm dino")
    evidence = blob(mix / ".pipeline/jobs/research/fitted.npz", b"research evidence")
    oldmeta = parquet(mix / "data/metadata.parquet", "metadata_only")
    oldjson = blob(mix / "derived/egosuite-native-dense22-v1/meta/info.json", b"{}"); oldjson.parent.mkdir(exist_ok=True)
    mixmodel = blob(mix / "weights/model.npz", b"immutable original model")
    extra = parquet(mix / "derived/egosuite-native-dense22-v1/body.parquet")
    smoke = parquet(mix / "derived/egosuite-native-dense22-export-smoke-agent-20261011/body.parquet")
    pilot = parquet(mix / "derived/egosuite-native-dense22-pilot-20261011T045036Z/body.parquet")
    info = {"model": {"path": "weights/smplh/male/model.npz", "model_sha256": cleanup.MODEL_SHA256},
            "hand_basis": {"path": "weights/smplh/hand_basis.npz", "sha256": cleanup.sha(hand)}}
    put(root / "meta/info.json", info)
    put(root / "meta/splits.json", {"train": ["egosuite:0", "egoexo4d_hm:take"], "val": ["egosuite:1"], "test": []})
    blob(root / "meta/episodes.parquet", b"owned episode table")
    rows, records, reports, oldfiles, audit_results = [], [], [], [extra, smoke, pilot], []
    for index in range(3):
        source = "egosuite" if index < 2 else "egoexo4d_hm"
        key = "egosuite:" + str(index) if index < 2 else "egoexo4d_hm:take"
        split = "val" if index == 1 else "train"
        row = {"episode_index": index, "source": source, "source_key": key, "length": 4,
               "split": split, "dataset_from_index": 4 * index, "dataset_to_index": 4 * (index + 1),
               "split_group_ids": [source + ":person" + str(index)], "operator_id": str(index)}
        provenance = {}
        if index < 2:
            original = parquet(mix / "data" / f"episode_{index}.parquet")
            sidecar = parquet(fitted / "episodes" / f"{index:08d}/motion.parquet", "fitted.body_world_rot6d_xyz")
            provenance = {"source_path": str(original), "source_identity": cleanup.identity(original),
                          "fitted_path": str(sidecar), "fitted_identity": cleanup.identity(sidecar)}
            row.update(source_episode_index=index, source_path=str(original), fitted_path=str(sidecar))
            oldfiles += [original, sidecar]
            audit_results.append({"episode_index": index, "source_key": key, "frames": 4, "passed": True})
        else:
            provenance = {"source_path": str(hmraw), "source_identity": cleanup.identity(hmraw)}
        rows.append(row)
        folder = root / "episodes" / f"{index:08d}"
        motion, vision = blob(folder / "motion.parquet", b"parameter motion" + bytes([index])), blob(folder / "vision.parquet", b"owned dino" + bytes([index]))
        record = {**row, "data_path": str(motion.relative_to(root)), "vision_path": str(vision.relative_to(root)),
                  "motion_sha256": cleanup.sha(motion), "vision_sha256": cleanup.sha(vision), "source_provenance": provenance,
                  "metadata": {"note": "独立数据集；不认证原生旋转"}}
        receipt = {"record": record, "source_provenance": provenance,
                   "output_identities": {"motion": cleanup.identity(motion), "vision": cleanup.identity(vision)},
                   "motion": {"sha256": record["motion_sha256"]}, "vision": {"sha256": record["vision_sha256"]}}
        put(folder / "receipt.json", receipt)
        put(root / ".pipeline/episodes" / f"{index:08d}.json", record)
        records.append(record)
        reports.append({"episode_index": index, "source": source, "source_key": key, "frames": 4,
                        "split": split, "receipt_sha256": cleanup.sha(folder / "receipt.json")})
    plan = {"egosuite_archive": str(mix), "output_root": str(root), "egosuite_fitting_job": str(fitjob),
            "egosuite_fitted_root": str(fitted), "hm_root": str(hmraw.parent)}
    put(job / "plan.json", plan)
    put(job / "inventory/combined.json", {"rows": rows})
    fitplan = {"job_signature": "fixed full fitting job"}
    put(fitjob / "plan.json", fitplan)
    put(fitjob / "status.json", {**fitplan, "state": "full_candidate_complete", "independent_acceptance_pending": False})
    put(fitjob / "active-child.json", {"pid": 99999998})
    put(fitjob / "full-process.json", {"pid": 99999999})
    put(fitjob / "quality-policy.json", {"body_mean_m": .01, "rotation_jump_fraction": .001})
    put(fitjob / "quality-policy-registration.json", {**fitplan, "policy_predates_first_output": True,
          "policy_sha256": cleanup.sha(fitjob / "quality-policy.json"), "policy_identity": cleanup.identity(fitjob / "quality-policy.json")})
    # This file is created after the registered policy, as required by acceptance.
    put(fitted / "manifest.json", {"model_defined_axes": True})
    put(fitjob / "full-execution.json", {**fitplan, "scope": "full", "state": "complete", "output_root": str(fitted), "episodes": 2, "frames": 8})
    ftaudit = {"passed": True, "episodes_checked": 2, "frames_checked": 8,
               "auditor_sha256": cleanup.sha(cleanup.FIT_AUDITOR), "sidecar_root": str(fitted), "results": audit_results,
               "shared_train_prior_audit": {"passed": True}, "failures": []}
    put(fitjob / "reports/full-sidecar-audit.json", ftaudit)
    fitgate = {"format": "smplh-candidate-acceptance-v2", **fitplan, "scope": "full", "passed": True,
               "full_candidate_quality_passed": True, "episodes": 2, "frames": 8, "failures": [],
               "preservation": {"passed": True}, "policy_sha256": cleanup.sha(fitjob / "quality-policy.json"),
               "independent_audit_sha256": cleanup.sha(fitjob / "reports/full-sidecar-audit.json")}
    put(fitjob / "reports/full-acceptance.json", fitgate)
    probe = job / "reports/standalone-fresh-reader.json"
    probe_report = {"passed": True, "episodes": 3, "fresh_process": True, "all_population_interfaces_checked": True,
                    "old_source_open_attempts": [], "blocked_roots": [str(mix), str(fitted), str(hmraw.parent)],
                    "rows": [{k: record[k] for k in ("episode_index", "source_key", "length", "split")} for record in records]}
    put(probe, probe_report)
    binding = {"plan_sha256": cleanup.sha(job / "plan.json"), "combined_inventory_sha256": cleanup.sha(job / "inventory/combined.json"),
               "auditor_sha256": cleanup.sha(cleanup.AUDITOR), "reader_sha256": cleanup.sha(cleanup.READER),
               "info_sha256": cleanup.sha(root / "meta/info.json"), "episodes_metadata_sha256": cleanup.sha(root / "meta/episodes.parquet"),
               "fresh_reader_sha256": cleanup.sha(probe), "split_metadata_sha256": {"meta/splits.json": cleanup.sha(root / "meta/splits.json")},
               "records_manifest_sha256": cleanup.digest(records), "model_sha256": cleanup.MODEL_SHA256,
               "hand_basis_sha256": cleanup.sha(hand), "original_full_fit_acceptance_sha256": cleanup.sha(fitjob / "reports/full-acceptance.json"),
               "original_full_fit_audit_sha256": cleanup.sha(fitjob / "reports/full-sidecar-audit.json"), "original_full_fit_signature": fitplan["job_signature"]}
    report = {"format": "independent-standalone-smplh-parameter-full-audit-v1", "scope": "full", "population": cleanup.population(rows),
              "audited_episodes": 3, "audited_frames": 12, "failures": [], "binding": binding, "episode_reports": reports,
              "fresh_reader_report": str(probe), **dict.fromkeys(("passed", "cleanup_allowed", "parameter_and_source_integrity_passed",
              "original_egosuite_full_fit_acceptance_passed", "old_dense_dependency_absent"), True)}
    put(job / "reports/full-parameter-acceptance.json", report)
    preserved = [raw, hmraw, hmdino, evidence, oldmeta, oldjson, mixmodel, model, hand]
    def fake_probe(context):
        path = context["job"] / "reports/post-cleanup-standalone.json"
        put(path, probe_report)
        return {"passed": True, "path": str(path), "sha256": cleanup.sha(path)}
    monkeypatch.setattr(cleanup, "post_cleanup_probe", fake_probe)
    return {"job": job, "fitjob": fitjob, "mix": mix, "root": root, "rows": rows, "records": records,
            "report": report, "oldfiles": oldfiles, "preserved": preserved, "fitted": fitted}


def mutate(path, key, value):
    document = cleanup.read(path); document[key] = value; put(path, document)


def test_full_gate_manifest_is_idempotent_and_deletes_only_old_dense(accepted):
    before = {str(p): cleanup.sha(p) for p in accepted["preserved"]}
    manifest = cleanup.prepare_manifest(accepted["job"])
    assert cleanup.prepare_manifest(accepted["job"]) == manifest
    assert {Path(r["path"]) for r in manifest["files"]} == set(accepted["oldfiles"])
    assert manifest["file_count"] == 7 and manifest["delete_directories"] is False
    receipt = cleanup.execute_cleanup(accepted["job"])
    assert receipt["state"] == "complete" and receipt["deleted_files"] == 7
    assert receipt["post_cleanup_probe"]["passed"]
    assert all(not p.exists() for p in accepted["oldfiles"])
    assert all(cleanup.sha(Path(p)) == checksum for p, checksum in before.items())
    assert all(Path(p).exists() for p in manifest["protected_files"])
    assert cleanup.execute_cleanup(accepted["job"])["deleted_files"] == 7


@pytest.mark.parametrize("relative", ["reports/full-parameter-acceptance.json", "reports/standalone-fresh-reader.json"])
def test_missing_parameter_acceptance_or_probe_preserves_all(accepted, relative):
    (accepted["job"] / relative).unlink()
    with pytest.raises(ValueError, match="missing"):
        cleanup.prepare_manifest(accepted["job"])
    assert all(p.exists() for p in accepted["oldfiles"])


@pytest.mark.parametrize("name", ["full-acceptance.json", "full-sidecar-audit.json"])
def test_missing_original_full_fit_gate_refuses(accepted, name):
    (accepted["fitjob"] / "reports" / name).unlink()
    with pytest.raises(ValueError, match="missing"):
        cleanup.prepare_manifest(accepted["job"])


@pytest.mark.parametrize("key,value", [("episodes", 1), ("frames", 4), ("scope", "expanded"),
                                       ("passed", False), ("preservation", {"passed": False})])
def test_partial_or_failed_old_fit_refuses(accepted, key, value):
    mutate(accepted["fitjob"] / "reports/full-acceptance.json", key, value)
    with pytest.raises(ValueError, match="original full fitting"):
        cleanup.prepare_manifest(accepted["job"])


def test_wrong_split_group_and_population_rejects(accepted):
    rows = deepcopy(accepted["rows"])
    rows[1]["split_group_ids"] = rows[0]["split_group_ids"]
    put(accepted["job"] / "inventory/combined.json", {"rows": rows})
    with pytest.raises(ValueError, match="leaks across splits"):
        cleanup.prepare_manifest(accepted["job"])
    rows[1]["split_group_ids"] = ["separate"]
    rows.pop()
    put(accepted["job"] / "inventory/combined.json", {"rows": rows})
    with pytest.raises(ValueError, match="population is incomplete"):
        cleanup.prepare_manifest(accepted["job"])


def test_changed_original_and_changed_owned_outputs_refuse(accepted):
    original = next(p for p in accepted["oldfiles"] if p.parent == accepted["mix"] / "data")
    parquet(original, "observation.state.body_world_rot6d_xyz")
    with pytest.raises(ValueError, match="original dense source was replaced"):
        cleanup.prepare_manifest(accepted["job"])
    # A mutation of owned output is independently fatal before any deletion.
    record = accepted["records"][0]
    blob(accepted["root"] / record["data_path"], b"changed parameter output")
    with pytest.raises(ValueError, match="owned motion/vision identity changed"):
        cleanup.prepare_manifest(accepted["job"])


def test_changed_gate_or_model_after_plan_refuses(accepted):
    cleanup.prepare_manifest(accepted["job"])
    mutate(accepted["fitjob"] / "status.json", "state", "fitting_full")
    with pytest.raises(ValueError, match="has not completed"):
        cleanup.execute_cleanup(accepted["job"])
    assert all(p.exists() for p in accepted["oldfiles"])


@pytest.mark.parametrize("key,value", [("fresh_process", False), ("blocked_roots", []),
                                       ("old_source_open_attempts", ["old/data.parquet"]), ("rows", [])])
def test_incomplete_or_old_dependent_probe_cannot_authorize_deletion(accepted, key, value):
    mutate(accepted["job"] / "reports/standalone-fresh-reader.json", key, value)
    with pytest.raises(ValueError, match="Standalone probe lacks"):
        cleanup.prepare_manifest(accepted["job"])
    assert all(p.exists() for p in accepted["oldfiles"])


def test_protected_hardlink_is_never_deleted(accepted):
    record = accepted["records"][0]
    target = accepted["root"] / record["data_path"]
    linked = accepted["mix"] / "data/linked-parameter.parquet"
    # A valid old dense footer is linked into the protected root. Update its
    # accepted fixture receipt before planning; inode sharing remains forbidden.
    target.unlink()
    parquet(target)
    linked.hardlink_to(target)
    record["motion_sha256"] = cleanup.sha(target)
    folder = target.parent
    receipt = cleanup.read(folder / "receipt.json")
    receipt["record"] = record
    receipt["motion"]["sha256"] = record["motion_sha256"]
    receipt["output_identities"]["motion"] = cleanup.identity(target)
    put(folder / "receipt.json", receipt)
    put(accepted["root"] / ".pipeline/episodes/00000000.json", record)
    report = cleanup.read(accepted["job"] / "reports/full-parameter-acceptance.json")
    report["binding"]["records_manifest_sha256"] = cleanup.digest(accepted["records"])
    report["episode_reports"][0]["receipt_sha256"] = cleanup.sha(folder / "receipt.json")
    put(accepted["job"] / "reports/full-parameter-acceptance.json", report)
    with pytest.raises(ValueError, match="hard-linked to a protected"):
        cleanup.prepare_manifest(accepted["job"])
    assert target.exists() and linked.exists()


def test_owned_model_or_binary_change_after_plan_refuses(accepted):
    manifest = cleanup.prepare_manifest(accepted["job"])
    target = accepted["root"] / accepted["records"][0]["vision_path"]
    blob(target, b"different owned DINO")
    with pytest.raises(ValueError, match="Protected owned artifact changed"):
        cleanup.execute_cleanup(accepted["job"])
    assert all(Path(entry["path"]).exists() for entry in manifest["files"])


def test_symlink_and_traversal_are_rejected(accepted, tmp_path):
    link = accepted["mix"] / "data/external.parquet"
    link.symlink_to(accepted["preserved"][1])
    with pytest.raises(ValueError, match="Symlink"):
        cleanup.prepare_manifest(accepted["job"])
    assert accepted["preserved"][1].exists()
    with pytest.raises(ValueError, match="traversal"):
        cleanup.nonsymlink(accepted["mix"] / "data/../weights/model.npz")


def test_live_writer_and_unreadable_pid_block_but_unrelated_pid_reuse_does_not(accepted, monkeypatch):
    monkeypatch.setattr(cleanup, "live_processes", lambda: [{"pid": 9, "command": "python process_smplh_dataset.py --job " + str(accepted["fitjob"])}])
    with pytest.raises(ValueError, match="remains active"):
        cleanup.prepare_manifest(accepted["job"])
    monkeypatch.setattr(cleanup, "live_processes", lambda: [{"pid": 99999998, "command": None}])
    with pytest.raises(ValueError, match="demonstrably exited"):
        cleanup.prepare_manifest(accepted["job"])
    monkeypatch.setattr(cleanup, "live_processes", lambda: [{"pid": 99999998, "command": "python unrelated_service.py"}])
    assert cleanup.prepare_manifest(accepted["job"])["file_count"] == 7


def test_preflight_all_files_before_first_unlink(accepted):
    manifest = cleanup.prepare_manifest(accepted["job"])
    path = Path(manifest["files"][-1]["path"])
    blob(path, b"changed old file")
    with pytest.raises(ValueError, match="remaining dense file changed"):
        cleanup.execute_cleanup(accepted["job"])
    assert all(p.exists() for p in accepted["oldfiles"])


def test_partial_deletion_resumes_only_from_durable_pending(accepted, monkeypatch):
    manifest = cleanup.prepare_manifest(accepted["job"])
    original_unlink = cleanup.secure_unlink
    calls = []
    def interrupted_unlink(mix, entry):
        original_unlink(mix, entry)
        calls.append(entry["path"])
        if len(calls) == 2:
            raise RuntimeError("simulated crash after unlink before receipt update")
    monkeypatch.setattr(cleanup, "secure_unlink", interrupted_unlink)
    with pytest.raises(RuntimeError, match="simulated crash"):
        cleanup.execute_cleanup(accepted["job"])
    receipt = cleanup.read(accepted["job"] / "reports/dense-cleanup-execution.json")
    assert len(receipt["deleted"]) == 1 and receipt["pending"] == calls[-1]
    assert not Path(receipt["pending"]).exists()
    monkeypatch.setattr(cleanup, "secure_unlink", original_unlink)
    receipt = cleanup.execute_cleanup(accepted["job"])
    assert receipt["state"] == "complete" and receipt["deleted_files"] == manifest["file_count"]


def test_unrecorded_disappearance_never_authorizes_other_deletions(accepted):
    manifest = cleanup.prepare_manifest(accepted["job"])
    Path(manifest["files"][-1]["path"]).unlink()
    with pytest.raises(ValueError, match="disappeared without durable"):
        cleanup.execute_cleanup(accepted["job"])
    assert all(Path(row["path"]).exists() for row in manifest["files"][:-1])


def test_unicode_records_digest_matches_independent_auditor(accepted):
    auditor_spec = importlib.util.spec_from_file_location("parameter_auditor_for_cleanup", cleanup.AUDITOR)
    auditor = importlib.util.module_from_spec(auditor_spec); auditor_spec.loader.exec_module(auditor)
    assert cleanup.canonical(accepted["records"]) == auditor.canonical(accepted["records"])
    assert cleanup.digest(accepted["records"]) == hashlib.sha256(auditor.canonical(accepted["records"])).hexdigest()
    assert cleanup.population(accepted["rows"]) == auditor.population_records(accepted["records"], accepted["rows"])
    assert cleanup.prepare_manifest(accepted["job"])["file_count"] == 7


def test_real_incomplete_job_refuses_without_modifying_sources():
    job = Path("/gaozt-test1/datasets/MixEgoMotion/.pipeline/jobs/smplh-parameters-v1-20261012T032915Z")
    if not (job / "plan.json").exists():
        pytest.skip("This environment has no real migration job")
    if (job / "reports/full-parameter-acceptance.json").exists():
        pytest.skip("The real migration has already reached its acceptance phase")
    plan = cleanup.read(job / "plan.json")
    candidates = sorted((Path(plan["egosuite_archive"]) / "data").rglob("*.parquet"))[:3]
    before = {str(path): cleanup.identity(path) for path in candidates}
    with pytest.raises(ValueError, match="missing"):
        cleanup.gate_context(job)
    assert before == {str(path): cleanup.identity(path) for path in candidates}
    assert not cleanup.manifest_path(job).exists()
