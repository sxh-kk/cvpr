"""CPU checks for causal labels, immutable provenance and probability mixtures."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import handover_critical_windows as critical


def quality_samples():
    samples = []
    for step in range(1, 301):
        t = step / 50
        right, left = .8 <= t < 2.2, 1.8 <= t < 4
        lifted = .8 <= t < 3.4
        aligned, supported = t >= 3.2, t >= 3.4
        samples.append({"completed_steps": step, "sim_time_s": 1.005 + t,
            "actual_target_xyz_m": [0., 0., .8 if lifted else .65 if supported else .6],
            "pelvis_z_m": .75, "contacts": {"left_hand": left, "right_hand": right, "container": supported},
            "solver_stage": "post_mj_step_before_observation_refresh",
            "container_geometry": {"inside_container_footprint": aligned,
                "collision_vertical_ranges_overlap": aligned, "container_upright_cosine": 1.,
                "container_support_weight_fraction": 1. if supported else 0.,
                "support_measurement": "live_solved_mujoco_contact_force"}})
    return samples


def capture_events():
    events = []
    for row in range(210):
        tick = (row * 50 + 29) // 30
        events.append({"event_index": row, "capture_index30": row, "physical_tick50": tick,
            "sim_time_s": .005 + tick / 50, "nominal_elapsed_s": row / 30,
            "action_valid": row >= 30, "action": None if row < 30 else {"body_token64": [0.] * 64, "hand14": [0.] * 14}})
    return events


def sampling_rows():
    return [{"episode_index": a, "current_row": i, "source_episode_index": a,
             "critical_stages": tags} for a in (0, 9)
            for i, tags in enumerate((["lift"], ["handover"], ["placement"], ["lift", "placement"], []))]


def test_observation_phase_is_causal_but_sampling_tags_are_explicit_future_annotations():
    samples, events = quality_samples(), capture_events()
    rows, transitions = critical.annotate_windows(0, 0, [30, 50, 60, 90, 150, 180], events, samples, .6)
    assert rows[0]["observed_stage"] == "warmup_complete"
    assert rows[1]["observed_stage"] == "pre_lift"
    assert "lift" in rows[1]["critical_stages"]
    assert rows[-1]["observed_stage"] == "stable_placement"
    modified = deepcopy(samples)
    for sample in modified:
        if sample["sim_time_s"] > events[50]["sim_time_s"]:
            sample["actual_target_xyz_m"][0] = .1
    changed, _ = critical.annotate_windows(0, 0, [50], events, modified, .6)
    assert changed[0]["observed_stage"] == rows[1]["observed_stage"]
    assert {e["stage"] for e in transitions} == set(critical.FAILURES)


def test_contacts_without_real_support_cannot_be_placement_annotations():
    samples = quality_samples()
    for sample in samples:
        sample["container_geometry"]["container_support_weight_fraction"] = 0.
    with pytest.raises(ValueError, match="released placement transitions"):
        critical.classify_quality(samples, .6)


def test_stability_annotation_preserves_existing_49_of_50_support_rule():
    samples = quality_samples()[:250]
    samples[-10]["container_geometry"]["container_support_weight_fraction"] = 0.
    phases, events = critical.classify_quality(samples, .6)
    assert phases[-1]["observed_stage"] == "stable_placement"
    stable = next(event for event in events if event["event"] == "stable_placement")
    assert stable["sampling_focus"] is False


def test_stable_placement_keeps_causal_label_and_admission_but_never_adds_focus():
    samples, events = quality_samples(), capture_events()
    rows, transitions = critical.annotate_windows(0, 0, range(30, 181), events, samples, .6)
    stable = next(event for event in transitions if event["event"] == "stable_placement")
    placement = [event for event in transitions
                 if event["stage"] == "placement" and event["sampling_focus"]]
    assert {event["event"] for event in placement} == {
        "placement_alignment", "container_supported", "released_supported"}
    stable_only = [row for row in rows if abs(row["sim_time_s"] - stable["sim_time_s"]) <= .5
                   and all(abs(row["sim_time_s"] - event["sim_time_s"]) > .5
                           for event in placement)]
    assert stable_only and all("placement" not in row["critical_stages"] for row in stable_only)
    assert rows[-1]["observed_stage"] == "stable_placement"
    weighted, report = critical.sampling_weights(rows)
    tail = weighted[-1]
    assert tail["observed_stage"] == "stable_placement" and tail["critical_stages"] == []
    assert tail["focus_weights"] == {} and tail["weight"] == tail["base_weight"] > 0
    assert sum(row["focus_weights"].get("placement", 0) for row in weighted) == pytest.approx(.5 * 17 / 26)
    assert report["critical_windows_by_stage"]["placement"] < len(rows)
    with pytest.raises(ValueError, match="released placement transitions"):
        critical.classify_quality(samples[:230], .6)


def test_handover_must_follow_right_lift_and_both_hands():
    samples = quality_samples()
    for sample in samples:
        if sample["contacts"]["right_hand"] and sample["contacts"]["left_hand"]:
            sample["contacts"]["left_hand"] = False
    with pytest.raises(ValueError, match="lifted handover"):
        critical.classify_quality(samples, .6)


@pytest.mark.parametrize("mutation", ["clock", "solver", "missing_support", "nonfinite"])
def test_incomplete_or_nonphysical_quality_is_rejected(mutation):
    samples = quality_samples()
    if mutation == "clock": samples[20]["sim_time_s"] += .02
    if mutation == "solver": samples[20]["solver_stage"] = "offline_mj_forward"
    if mutation == "missing_support": del samples[20]["container_geometry"]["support_measurement"]
    if mutation == "nonfinite": samples[20]["actual_target_xyz_m"][0] = float("nan")
    with pytest.raises(ValueError): critical.classify_quality(samples, .6)


def test_mixture_has_exact_full_and_failure_category_mass_with_positive_weights():
    weighted, report = critical.sampling_weights(sampling_rows())
    assert sum(r["weight"] for r in weighted) == pytest.approx(1)
    assert sum(r["base_weight"] for r in weighted) == pytest.approx(.5)
    for stage, count in critical.FAILURES.items():
        assert sum(r["focus_weights"].get(stage, 0) for r in weighted) == pytest.approx(.5 * count / 26)
    assert all(math.isfinite(r["weight"]) and r["weight"] > 0 for r in weighted)
    assert report["effective_ancestor_probability_mass"] == {"0": pytest.approx(.5), "9": pytest.approx(.5)}


def test_ancestry_is_controllable_and_missing_category_is_an_error():
    weighted, report = critical.sampling_weights(sampling_rows(), ancestor_weights={0: 3, 9: 1})
    assert report["effective_ancestor_probability_mass"]["0"] == pytest.approx(.75)
    rows = sampling_rows()
    for row in rows: row["critical_stages"] = [s for s in row["critical_stages"] if s != "handover"]
    with pytest.raises(ValueError, match="Missing critical category"):
        critical.sampling_weights(rows)
    assert sum(r["base_weight"] for r in weighted if r["source_episode_index"] == 0) == pytest.approx(.375)


@pytest.mark.parametrize("weights", [{0: 0, 9: 1}, {0: float("inf"), 9: 1}, {0: 1}, {0: -1, 9: 1}])
def test_invalid_ancestor_control_is_rejected(weights):
    with pytest.raises(ValueError): critical.sampling_weights(sampling_rows(), ancestor_weights=weights)


def test_validation_ancestry_and_duplicate_windows_are_rejected():
    rows = sampling_rows()
    with pytest.raises(ValueError, match="Duplicate"):
        critical.sampling_weights(rows + [rows[0]])
    rows[0]["source_episode_index"] = 39
    with pytest.raises(ValueError, match="Held-out"):
        critical.sampling_weights(rows)


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, allow_nan=False) + "\n")


def admitted_fixture(tmp_path):
    pack = tmp_path / "old_pack"
    evidence = pack / "accepted_evidence/one"
    events = capture_events(); samples = quality_samples()
    event_path = evidence / "events.jsonl"; event_path.parent.mkdir(parents=True)
    event_path.write_text("".join(json.dumps(x) + "\n" for x in events))
    quality_path = tmp_path / "original_capture/quality_samples.jsonl"
    quality_path.parent.mkdir(parents=True)
    quality_path.write_text("".join(json.dumps(x) + "\n" for x in samples))
    capture = {"candidate_id": "one", "source_episode_index": 0, "events_sha256": critical.sha256(event_path),
               "initial_target_z_m": .6}
    _write(evidence / "capture.json", capture)
    admission = {"eligible_dataset": True, "candidate_id": "one", "source_episode_index": 0,
        "capture_content_sha256": "fixture-content-id",
        "bindings": {str(quality_path): critical.sha256(quality_path),
                     str(tmp_path / "original_capture/capture/capture.json"): critical.sha256(evidence / "capture.json")},
        "physical_gate": {"verified_handover_success": True,
            "contact_evidence": {"sample_count": len(samples)},
            "final_placement_geometric_validation": {"require_3d_evidence": True}}}
    _write(evidence / "admission.json", admission)
    import pack_handover_quality100 as packer
    eligible, histories = packer.eligible_currents(events)
    episode = {"episode_index": 0, "source_episode_index": 0, "candidate_id": "one", "length": len(events),
        "split": "train", "split_restriction": "train_only", "lineage_complete": True,
        "lineage_group_sources": [{"source_episode_index": 0}], "eligible_current_rows": eligible}
    _write(pack / "train/meta/episodes.jsonl", episode)
    ready = {"contract": "cvpr-handover-quality100-lerobot-pack-v1", "train_target_satisfied": True,
        "existing_VAL_sources_excluded": list(critical.HELDOUT), "boundary_policy": {"tail_padding_permitted": False},
        "train_episodes": 1, "episodes": [{"episode_index": 0, "candidate_id": "one",
            "eligible_current_rows": eligible, "history_rows": histories,
            "admission_sha256": critical.sha256(evidence / "admission.json"), "capture_content_sha256": "fixture-content-id"}],
        "payload_sha256": {str(p.relative_to(pack)): critical.sha256(p) for p in pack.rglob("*") if p.is_file()}}
    _write(pack / "PACK_READY.json", ready)
    return pack, quality_path


def test_real_sidecar_build_preserves_source_bytes_and_production_window_ids(tmp_path):
    pack, quality = admitted_fixture(tmp_path)
    before = {str(p): critical.sha256(p) for p in tmp_path.rglob("*") if p.is_file()}
    result = critical.build_sidecar(pack, tmp_path / "new_sidecar")
    rows = [json.loads(x) for x in (tmp_path / "new_sidecar/rows.jsonl").read_text().splitlines()]
    assert rows[0]["current_row"] == 30 and rows[-1]["current_row"] == 180
    assert result["window_count"] == 151
    assert result["new_demonstrations"] is result["recovery_demonstrations"] is False
    assert result["validation_test_reweighted"] is result["model_input_fields_added"] is False
    assert result["critical_focus_policy_revision"] == "critical-conversions-v3"
    assert result["critical_focus_excluded_events"] == ["stable_placement"]
    assert result["critical_focus_events"]["placement"] == (
        "placement_alignment", "container_supported", "released_supported")
    assert result["failure_count_provenance"] == result["mixture"]["failure_count_provenance"] == {
        "source": "user_reported_aggregate", "split": "unverified", "trace_bindings_available": False,
        "used_for_train_sampling_priorities": True, "independent_blind_test_claim": False}
    assert all(critical.sha256(p) == digest for p, digest in before.items())
    assert result["source_receipts"][str(quality)] == before[str(quality)]


@pytest.mark.parametrize("target", ["quality", "capture", "events"])
def test_changed_raw_quality_or_capture_is_rejected(tmp_path, target):
    pack, quality = admitted_fixture(tmp_path)
    path = quality if target == "quality" else pack / "accepted_evidence/one" / (target + (".jsonl" if target == "events" else ".json"))
    with path.open("a") as f: f.write(" ")
    with pytest.raises(ValueError, match="SHA changed"):
        critical.build_sidecar(pack, tmp_path / "new_sidecar")
    assert not (tmp_path / "new_sidecar").exists()


def test_source_output_and_reuse_of_existing_artifacts_are_rejected(tmp_path):
    pack, _ = admitted_fixture(tmp_path)
    with pytest.raises(ValueError, match="outside"):
        critical.build_sidecar(pack, pack / "sidecar")
    target = tmp_path / "occupied"; target.mkdir(); (target / "keep").write_text("existing")
    with pytest.raises(ValueError, match="fresh"):
        critical.build_sidecar(pack, target)


def test_clock_gaps_and_fake_warmup_labels_are_rejected():
    events = capture_events(); events[40]["physical_tick50"] += 1
    with pytest.raises(ValueError, match="mapping"):
        critical._verify_events(events, len(events))
    events = capture_events(); events[0]["action"] = {"body_token64": [0.] * 64}
    with pytest.raises(ValueError, match="Warmup"):
        critical._verify_events(events, len(events))
