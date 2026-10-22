"""CPU packaging/reader tests with synthetic admitted-input doubles, no GPU."""
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pack_handover_quality100 as packer

STATE = [f"body{i}" for i in range(29)] + [f"left_hand_{i}" for i in range(7)] + [f"right_hand_{i}" for i in range(7)]
HANDS = STATE[-14:]


def fixture_events(root, count=65):
    root.mkdir(parents=True, exist_ok=True)
    events = []
    names = list(reversed(STATE)); hands = list(reversed(HANDS))
    for row in range(count):
        tick = (row * 50 + 29) // 30
        pixels = (np.arange(8 * 10 * 3).reshape(8, 10, 3) + row * 7).astype(np.uint8)
        image = root / f"{row:05d}.png"; Image.fromarray(pixels).save(image)
        physical = np.asarray([STATE.index(name) + row / 100 for name in names], np.float32)
        events.append({"event_index": row, "capture_index30": row, "physical_tick50": tick,
            "sim_time_s": 5.125 + tick / 50, "nominal_elapsed_s": row / 30, "actual_elapsed_s": tick / 50,
            "phase": "warmup" if row < 30 else "teacher_command", "action_valid": row >= 30,
            "source_command_index": None if row < 30 else ((row - 30) * 50 + 29) // 30,
            "joint_names": names, "state43": physical.tolist(), "physical_joint_q43_float64": physical.astype(float).tolist(),
            "physical_joint_qvel43": [float(STATE.index(name)) for name in names],
            "root_pose7": [0, 0, .75, 1, 0, 0, 0], "root_qvel6": [0.] * 6,
            "objects": {"target": {"pose7": [0, 0, .6, 1, 0, 0, 0]}, "container": {"pose7": [0, .3, .6, 1, 0, 0, 0]}},
            "action": None if row < 30 else {"body_token64": [row / 100] * 64,
                "hand14": [HANDS.index(name) + row / 100 for name in hands], "hand_joint_names": hands},
            "rgb": {"path": image.name, "file_sha256": packer.sha(image), "pixel_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
                    "width": 10, "height": 8, "channels": 3, "dtype": "uint8"}})
    return events


def checkpoint_fixture(root):
    root.mkdir(parents=True, exist_ok=True)
    stats = {"observation.state": {"min": [-1.] * 43, "max": [1.] * 43, "count": [10]},
             "action": {"min": [-2.] * 36, "max": [2.] * 36, "count": [10]},
             "action.body_token_v1_1": {"min": [-.5] * 64, "max": [.5] * 64, "count": [10]}}
    info = {"features": {"observation.state": {"dtype": "float32", "shape": [43], "names": STATE},
                          "action": {"dtype": "float32", "shape": [36], "names": HANDS + [f"extra{i}" for i in range(22)]}}}
    field = {"stat_action_keys": ["action.body_token_v1_1", "action[:14]"], "stat_state_keys": ["observation.state"],
        "pad_action_dim": 80, "pad_state_dim": 45, "normalize_state": True, "action_norm_type": "bounds",
        "action_min": [-.5] * 64 + [-2.] * 14 + [0.] * 2, "action_max": [.5] * 64 + [2.] * 14 + [0.] * 2,
        "state_min": [-1.] * 43 + [0.] * 2, "state_max": [1.] * 43 + [0.] * 2}
    config = {"data": {"transform": {"field": field}}}
    config_path, stats_path, info_path, receipt_path = [root / f for f in ("run_config.json", "dataset_statistics.json", "info.json", "preparation.json")]
    for p, d in ((config_path, config), (stats_path, stats), (info_path, info)): packer.write(p, d)
    bounds = {key: hashlib.sha256(json.dumps(field[key], separators=(",", ":"), allow_nan=False).encode()).hexdigest()
              for key in ("action_min", "action_max", "state_min", "state_max")}
    receipt = {"source_step": 38000, "source_checkpoint_sha256": packer.SOURCE38_SHA,
        "config_sha256": packer.sha(config_path), "normalization": {"stat_file_sha256": packer.sha(stats_path), "bounds_sha256": bounds},
        "state_contract": "/original/info.json", "source_metadata_sha256": {"/original/info.json": packer.sha(info_path)}}
    packer.write(receipt_path, receipt)
    return receipt_path, config_path, stats_path, info_path


def test_named_values_keep_real_warmup_and_unavailable_targets(tmp_path):
    events = fixture_events(tmp_path / "capture")
    before = deepcopy(events)
    values = packer.episode_values(events, 2, 100, STATE, HANDS)
    np.testing.assert_array_equal(values["observation.state"][0], np.arange(43))
    np.testing.assert_allclose(values["action"][30], np.arange(14) + .3)
    assert np.isnan(values["action"][:30]).all() and np.isnan(values["action.body_token_v1_1"][:30]).all()
    assert not values["action_mask"][:30].any() and not values["action_valid"][:30].any()
    assert values["action_mask"][30:, :78].all() and not values["action_mask"][:, 78:].any()
    np.testing.assert_array_equal(values["source_frame_index"], [(i * 50 + 29) // 30 for i in range(65)])
    assert values["source_timestamp"][0] == 5.125 and values["timestamp"][0] == 0.
    assert events == before and all(e["action"] is None for e in events[:30])


def test_production_history_and_future_filters_keep_warmup_pixels_only(tmp_path):
    events = fixture_events(tmp_path / "capture")
    current, rows = packer.eligible_currents(events)
    assert current == list(range(30, 36))
    assert rows["30"] == [0, 6, 12, 18, 24, 30]
    bad = deepcopy(events); bad[40]["action_valid"] = False
    with pytest.raises(ValueError, match="complete30-action"):
        packer.eligible_currents(bad)
    bad = deepcopy(events); bad[35]["physical_tick50"] += 1
    with pytest.raises(ValueError, match="complete30-action"):
        packer.eligible_currents(bad)


def test_actual_motion_history_index_and_training_metadata_loader(tmp_path, monkeypatch):
    events = fixture_events(tmp_path / "capture")
    packer.eligible_currents(events)
    history = importlib.import_module("_handover_pack_history.motion_history")
    n = len(events); frame = np.arange(n); timestamp = (frame / 30).astype(np.float32)
    values = {"episode_index": np.zeros(n, np.int64), "frame_index": frame, "timestamp": timestamp,
        "source_frame_index": np.asarray([e["physical_tick50"] for e in events]), "source_episode_index": np.zeros(n, np.int64),
        "action_valid": np.asarray([e["action_valid"] for e in events])}
    class Table:
        column_names = list(values)
        def __getitem__(self, key): return values[key]
        def __len__(self): return n
        def with_format(self, *_): return self
    class Dataset(SimpleNamespace):
        def __len__(self): return n
    episode = SimpleNamespace(frame_index=frame, timestamp=timestamp, source_frame_index=frame,
        video_timestamp=frame / 30, valid=np.ones(n, bool),
        features=np.broadcast_to(frame[:, None, None], (n, 5, 1024)).astype(np.float32))
    monkeypatch.setattr(history, "load_cache_manifest", lambda *a, **k: {"source": {"video_key": packer.VIDEO_KEY}})
    monkeypatch.setattr(history, "load_episode_cache", lambda *a, **k: episode)
    dataset = Dataset(meta=SimpleNamespace(fps=30), hf_dataset=Table())
    index = history.MotionHistoryIndex(tmp_path, [dataset], ["train"], [tmp_path], packer.VIDEO_KEY)
    np.testing.assert_array_equal(index.get_history(0)[:, 0, 0], [0, 6, 12, 18, 24, 30])
    from finetune_carrybox import eligible_dataset_rows
    wrapped = SimpleNamespace(raw_dataset=SimpleNamespace(motion_history=index, base_dataset=dataset))
    candidates, _, _ = eligible_dataset_rows(wrapped, 1, source_frame_rule="ceil")
    assert [int(index.original_indices[i]) for i in candidates[0]] == list(range(30, 36))
    index.close()


def test_checkpoint_normalization_is_copied_projected_not_refitted(tmp_path):
    paths = checkpoint_fixture(tmp_path / "ckpt")
    identity, stats = packer.checkpoint_contract(*paths)
    assert len(stats["action"]["min"]) == 14 and stats["action"]["min"] == [-2.] * 14
    assert identity["bounds_refitted"] is False
    changed = packer.read(paths[2]); changed["action"]["min"][0] = -3; packer.write(paths[2], changed)
    with pytest.raises(ValueError, match="normalization bytes changed"):
        packer.checkpoint_contract(*paths)


def test_real_lossless_png_video_roundtrip_and_changed_png_rejected(tmp_path):
    events = fixture_events(tmp_path / "capture", 60)
    report = packer.encode_pngs(tmp_path / "capture", events, tmp_path / "video.mp4")
    assert report["pixels_exact"] and report["frames"] == 60 and report["raw_rgb_sha256"] == report["decoded_rgb_sha256"]
    image = tmp_path / "capture" / events[0]["rgb"]["path"]
    Image.new("RGB", (10, 8), "red").save(image)
    with pytest.raises(ValueError, match="PNG bytes changed"):
        packer.encode_pngs(tmp_path / "capture", events, tmp_path / "bad.mp4")


def test_parquet_roundtrip_preserves_float32_and_nan_label_absence(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    values = packer.episode_values(fixture_events(tmp_path / "capture"), 0, 0, STATE, HANDS)
    original = packer.arrow_table(values); path = tmp_path / "e.parquet"; pq.write_table(original, path)
    actual = pq.read_table(path); packer.check_parquet_roundtrip(original, actual)
    assert actual.schema.field("action").type == pa.list_(pa.float32())
    assert np.isnan(np.asarray(actual["action"].slice(0, 30).to_pylist())).all()


@pytest.mark.parametrize("audit_change", [
    {"eligible_dataset": False},
    {"contract": "cvpr-handover-independent-rtx-capture-audit-v1"},
    {"authority_revision": "superseded-authority"},
])
def test_physics_only_or_superseded_audit_cannot_be_admitted(tmp_path, monkeypatch, audit_change):
    import collect_handover_quality100 as collector
    audit = tmp_path / "audit.json"
    packer.write(audit, {"contract": collector.AUDIT_CONTRACT,
        "authority_revision": collector.AUTHORITY_REVISION,
        "status": "independent_capture_audit_passed", "eligible_dataset": True, **audit_change})
    ancestor_ids = [i for i in range(30) if i not in packer.HELDOUT][:24]
    ancestor = {"source_episode_index": 0}
    candidate = {"candidate_id": "one", "candidate_spec_sha256": "s", "ancestor": ancestor}
    attempt = {"status": "accepted", "source_episode_index": 0, "candidate_id": "one", "candidate_spec_sha256": "s",
               "plan_sha256": "p", "audit_path": "audit.json", "audit_sha256": packer.sha(audit)}
    state = {"contract": collector.QUEUE_CONTRACT,
             "authority_revision": collector.AUTHORITY_REVISION, "attempts": [attempt],
             "recipe": {"warmup_schedule": {"controls": {str(i): "measured_pose_pd" for i in ancestor_ids}}}}
    packer.write(tmp_path / "queue.json", state)
    monkeypatch.setattr(collector, "verified_queue", lambda _: (state, {"original_ancestors": [{"source_episode_index": i} for i in ancestor_ids]}, [(candidate, "unused", "p")]))
    with pytest.raises(ValueError, match="not dataset admissions"):
        packer.collect_admitted(tmp_path)


def frozen_admission_fixture(tmp_path, monkeypatch):
    """An admission double tests authority selection, never actual physics/RGB."""
    import collect_handover_quality100 as collector
    source = 31
    ancestors = [i for i in range(40) if i not in packer.HELDOUT and i != source][:23] + [source]
    candidate = {"candidate_id": "unit-live", "candidate_spec_sha256": "s", "ancestor": {"source_episode_index": source}}
    run = tmp_path / "attempt-00000/rtx"; capture = run / "capture"; capture.mkdir(parents=True)
    packer.write(run / "result.json", {"warmup_control": {"kind": "measured_pose_pd"}})
    packer.write(capture / "capture.json", {"contract": packer.CAPTURE_CONTRACT, "events_path": "events.jsonl"})
    (capture / "events.jsonl").write_text("")
    audit = {"contract": collector.AUDIT_CONTRACT, "authority_revision": collector.AUTHORITY_REVISION,
             "status": "independent_capture_audit_passed", "eligible_dataset": True,
             "bindings": {}, "source_sha256": {}, "capture_content_sha256": "synthetic-unit-double"}
    packer.write(tmp_path / "audit.json", audit)
    attempt = {"status": "accepted", "source_episode_index": source, "candidate_id": candidate["candidate_id"],
               "candidate_spec_sha256": "s", "plan_sha256": "p", "audit_path": "audit.json",
               "audit_sha256": packer.sha(tmp_path / "audit.json"), "directory": "attempt-00000"}
    schedule_path = tmp_path / "warmup_schedule.json"
    controls = {str(i): "live_pose_sonic_v1_1" if i == source else "measured_pose_pd" for i in ancestors}
    packer.write(schedule_path, {"unit_test_double": True, "controls": controls})
    schedule = {"path": str(schedule_path), "sha256": packer.sha(schedule_path), "controls": controls}
    state = {"contract": collector.QUEUE_CONTRACT, "authority_revision": collector.AUTHORITY_REVISION,
             "attempts": [attempt], "recipe": {"warmup_schedule": schedule}}
    packer.write(tmp_path / "queue.json", state)
    plan = {"original_ancestors": [{"source_episode_index": i} for i in ancestors]}
    monkeypatch.setattr(collector, "verified_queue", lambda _: (state, plan, [(candidate, "unused", "p")]))
    return collector, state, audit


def test_admission_recheck_uses_frozen_schedule_even_when_result_claims_other_control(tmp_path, monkeypatch):
    collector, state, audit = frozen_admission_fixture(tmp_path, monkeypatch)
    called = []
    def check(*args, expected_warmup):
        called.append(expected_warmup)
        return deepcopy(audit)
    monkeypatch.setattr(collector, "audit_capture", check)
    items, _, binding = packer.collect_admitted(tmp_path)
    assert len(items) == 1 and called == ["live_pose_sonic_v1_1"]
    assert binding["warmup_schedule"] == state["recipe"]["warmup_schedule"]


def test_default_all_measured_frozen_schedule_needs_no_external_file(tmp_path, monkeypatch):
    collector, state, audit = frozen_admission_fixture(tmp_path, monkeypatch)
    controls = state["recipe"]["warmup_schedule"]["controls"]
    state["recipe"]["warmup_schedule"] = {"path": None, "sha256": None,
        "controls": {source: "measured_pose_pd" for source in controls}}
    packer.write(tmp_path / "queue.json", state)
    called = []
    def check(*args, expected_warmup):
        called.append(expected_warmup)
        return deepcopy(audit)
    monkeypatch.setattr(collector, "audit_capture", check)
    _, _, binding = packer.collect_admitted(tmp_path)
    assert called == ["measured_pose_pd"] and binding["warmup_schedule"]["path"] is None


@pytest.mark.parametrize("missing", ["recipe", "warmup_schedule", "controls", "source"])
def test_missing_frozen_warmup_authority_never_falls_back_to_result(tmp_path, monkeypatch, missing):
    _, state, _ = frozen_admission_fixture(tmp_path, monkeypatch)
    if missing == "recipe":
        del state["recipe"]
    elif missing == "warmup_schedule":
        del state["recipe"]["warmup_schedule"]
    elif missing == "controls":
        del state["recipe"]["warmup_schedule"]["controls"]
    else:
        del state["recipe"]["warmup_schedule"]["controls"]["31"]
    packer.write(tmp_path / "queue.json", state)
    with pytest.raises(ValueError, match="[Ff]rozen warmup"):
        packer.collect_admitted(tmp_path)


@pytest.mark.parametrize("queue_change", [
    {"contract": "cvpr-handover-quality100-serial-queue-v1"},
    {"authority_revision": "superseded-authority"},
])
def test_superseded_registry_cannot_be_admitted(tmp_path, monkeypatch, queue_change):
    import collect_handover_quality100 as collector
    state = {"contract": collector.QUEUE_CONTRACT,
             "authority_revision": collector.AUTHORITY_REVISION, "attempts": [], **queue_change}
    packer.write(tmp_path / "queue.json", state)
    monkeypatch.setattr(collector, "verified_queue", lambda _: (state, {}, []))
    with pytest.raises(ValueError, match="obsolete audit authority"):
        packer.collect_admitted(tmp_path)


def test_full_synthetic_unit_pack_has_actual_cpu_lerobot_read_and_stays_incomplete(tmp_path, monkeypatch):
    """Admission is a test double; this creates no claimed real demonstration."""
    import collect_handover_quality100 as collector
    events = fixture_events(tmp_path / "capture")
    paths = checkpoint_fixture(tmp_path / "ckpt")
    manifest = {"events_path": "events.jsonl", "source_sha256": {}, "plan_sha256": "p", "candidate_spec_sha256": "s", "instruction": "Hand over the test object."}
    packer.write(tmp_path / "capture/capture.json", manifest)
    (tmp_path / "capture/events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    packer.write(tmp_path / "audit.json", {"unit_test_double": True})
    warmup = {"control": "measured_pose_pd", "actual_receipt_kind": "measured_pose_pd", "seconds": 1.,
              "scope": collector.WARMUP_SCOPE, "warmup_supervision": "none", "evaluation_oracle_allowed": False}
    item = {"events": events, "capture": tmp_path / "capture", "candidate": {"candidate_id": "unit-synthetic",
        "ancestor": {"source_episode_index": 0, "source_dataset_key": "unit-test"}}, "manifest": manifest,
        "result": {}, "audit": {"capture_content_sha256": "unit-test", "teacher_collection_warmup": warmup},
        "audit_path": tmp_path / "audit.json"}
    queue = tmp_path / "queue"; queue.mkdir()
    monkeypatch.setattr(packer, "collect_admitted", lambda *a: ([item], {}, {"unit_test_double": True}))
    report = packer.pack(queue, tmp_path / "packed", *paths)
    assert report["train_episodes"] == 1 and report["final_ready"] is False and report["train_target_satisfied"] is False
    assert report["cpu_reader_validation"]["status"] == "actual_cpu_LeRobot_numeric_video_queries_passed"
    assert (tmp_path / "packed/source_dataset_statistics.json").read_bytes() == paths[2].read_bytes()
    metadata = json.loads((tmp_path / "packed/train/meta/episodes.jsonl").read_text().splitlines()[0])
    assert metadata["teacher_collection_warmup"] == warmup
    assert report["episodes"][0]["teacher_collection_warmup"] == warmup


def multi_queue_fixture(tmp_path, monkeypatch):
    """Fake admissions isolate cross-queue identity checks; no real data claim."""
    plan = {"original_ancestors": [{"source_episode_index": i} for i in range(24)],
            "source_manifest_sha256": "release", "selection_sha256": "selection"}
    sources = []
    for index in range(2):
        queue = tmp_path / f"queue-{index}"; queue.mkdir()
        executable = tmp_path / f"code-{index}/scripts/4090_simple/collect_handover_quality100.py"
        executable.parent.mkdir(parents=True)
        executable.write_text(f"AUTHORITY_REVISION = 'unit-auditor-{index}'\n")
        frozen = queue / "frozen-code/scripts/4090_simple/collect_handover_quality100.py"
        frozen.parent.mkdir(parents=True); frozen.write_bytes(executable.read_bytes())
        packer.write(queue / "queue.json", {"recipe": {"code_sha256": {
            "scripts/4090_simple/collect_handover_quality100.py": packer.sha(executable)}}})
        sources.append({"queue_root": str(queue), "collector_path": str(executable),
            "collector_sha256": packer.sha(executable), "candidate_ids": [f"candidate-{index}"]})
    def fake_collect(root, limit=None, *, collector, candidate_ids):
        index = int(Path(root).name.split('-')[-1])
        assert collector.AUTHORITY_REVISION == f"unit-auditor-{index}"
        return [{"candidate": {"candidate_id": value, "ancestor": {"source_episode_index": index}},
                 "audit": {"capture_content_sha256": f"content-{value}"}} for value in candidate_ids], deepcopy(plan), {
            "queue_path": str(Path(root) / "queue.json"), "_snapshot_utf8": "unit", "queue_sha256": "unit"}
    monkeypatch.setattr(packer, "collect_admitted", fake_collect)
    document = {"contract": "cvpr-handover-multi-queue-selection-v1", "scope": "TRAIN_only", "sources": sources}
    path = tmp_path / "selection.json"; packer.write(path, document)
    return path, document


def test_multiple_queues_use_their_own_frozen_auditor(tmp_path, monkeypatch):
    path, _ = multi_queue_fixture(tmp_path, monkeypatch)
    items, _, binding = packer.collect_admitted_sources(path)
    assert len(items) == 2 and len(binding["_snapshots_utf8"]) == 2
    assert binding["sources"][0]["collector_sha256"] != binding["sources"][1]["collector_sha256"]


@pytest.mark.parametrize("tamper", ["executable", "frozen", "recipe", "repeated_queue", "repeated_candidate", "cap"])
def test_multiple_queue_selection_rejects_identity_and_global_limit_violations(tmp_path, monkeypatch, tamper):
    path, document = multi_queue_fixture(tmp_path, monkeypatch)
    row = document["sources"][0]; queue = Path(row["queue_root"])
    if tamper == "executable":
        Path(row["collector_path"]).write_text("raise AssertionError('must never execute changed code')\n")
    elif tamper == "frozen":
        (queue / "frozen-code/scripts/4090_simple/collect_handover_quality100.py").write_text("changed")
    elif tamper == "recipe":
        packer.write(queue / "queue.json", {"recipe": {"code_sha256": {}}})
    elif tamper == "repeated_queue":
        document["sources"][1] = deepcopy(row)
    elif tamper == "repeated_candidate":
        document["sources"][1]["candidate_ids"] = row["candidate_ids"]
    elif tamper == "cap":
        row["candidate_ids"] = [f"more-{i}" for i in range(6)]
    packer.write(path, document)
    with pytest.raises(ValueError):
        packer.collect_admitted_sources(path)


def test_live_registry_may_append_but_may_not_rewrite_admitted_captures():
    snapshot = {"contract":"unit", "attempts":[{"candidate_id":"old","status":"accepted","audit_sha256":"fixed"},
        {"candidate_id":"running","status":"capture_running"}]}
    live = deepcopy(snapshot)
    live["attempts"][1].update(status="accepted", audit_sha256="new")
    live["attempts"].append({"candidate_id":"next","status":"screen_running"})
    packer.verify_append_only_registry(snapshot,live)
    live["attempts"][0]["audit_sha256"]="changed"
    with pytest.raises(ValueError,match="previously admitted"):
        packer.verify_append_only_registry(snapshot,live)


def test_snapshot_audit_changes_no_module_global_or_actual_payload_read(tmp_path):
    from types import SimpleNamespace
    root=tmp_path; packer.write(root/"queue.json",{"attempts":[],"version":"live"})
    packer.write(root/"payload.json",{"actual":"payload"})
    namespace={"read":packer.read}
    exec("def verified_queue(root):\n    return read(root/'queue.json'), read(root/'payload.json'), []\n",namespace)
    frozen=json.dumps({"attempts":[],"version":"snapshot"}).encode()
    module=SimpleNamespace(verified_queue=namespace['verified_queue'])
    registry,payload,_=packer.verify_registry_snapshot(module,root,frozen)
    assert registry["version"]=="snapshot" and payload=={"actual":"payload"}
    assert namespace["read"] is packer.read and packer.read(root/'queue.json')["version"]=="live"
