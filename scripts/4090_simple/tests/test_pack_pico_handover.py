"""CPU converter tests using explicitly synthetic admitted-input doubles.

These fixtures exercise packaging of attestations, not human/physics admission.
The positive test runs actual lossless encoding and actual LeRobot CPU queries.
"""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import uuid

import numpy as np
import pyarrow.parquet as pq
import tempfile
import unittest
from unittest.mock import patch
import hashlib
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pack_pico_handover as packer


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
    for p, d in ((config_path, config), (stats_path, stats), (info_path, info)): packer.legacy.write(p, d)
    bounds = {key: hashlib.sha256(json.dumps(field[key], separators=(",", ":"), allow_nan=False).encode()).hexdigest()
              for key in ("action_min", "action_max", "state_min", "state_max")}
    receipt = {"source_step": 38000, "source_checkpoint_sha256": packer.legacy.SOURCE38_SHA,
        "config_sha256": packer.sha(config_path), "normalization": {"stat_file_sha256": packer.sha(stats_path), "bounds_sha256": bounds},
        "state_contract": "/original/info.json", "source_metadata_sha256": {"/original/info.json": packer.sha(info_path)}}
    packer.legacy.write(receipt_path, receipt)
    return receipt_path, config_path, stats_path, info_path



def write(path, value):
    packer.legacy.write(path, value)


def make_case(root, *, count=65, deferred=False):
    original = root/"source_contract_fixture"
    paths = checkpoint_fixture(original)
    template = root/"template"; template.mkdir()
    for path, name in zip(paths, packer.SOURCE_FILES):
        shutil.copyfile(path, template/name)
    identity = str(uuid.uuid4())
    directory = root/"collection/kept/001"
    capture = directory/"capture"
    events = fixture_events(capture, count)
    events_file = capture/"events.jsonl"
    events_file.write_text("".join(json.dumps(row)+"\n" for row in events))
    instruction = "  Hand Over THE cracker box, then place it.  "
    metadata = {"human_demo_uuid":identity,"source_kind":"live_pico_human",
        "replay_source_sha256":None,"task_id":"simple/G1WholebodyHandoverTeleop-v0",
        "instruction":instruction,"render_mode":"mujoco_deferred" if deferred else "isaac"}
    outcome = {"source_kind":"live_pico_human","accepted_new_dataset":False,
        "status":"pending_rtx" if deferred else "pending_audit",
        "state":{"status":"pending_audit","pending_audit":True,"human_demo_uuid":identity}}
    write(directory/"metadata.json", metadata); write(directory/"episode.json", outcome)
    write(capture/"capture.json", {**metadata,"contract":packer.legacy.CAPTURE_CONTRACT,
        "status":"raw_capture_completed_pending_independent_audit", "warmup_seconds":1.,
        "event_count":count,"events_sha256":packer.sha(events_file)})
    for name in ("commands.jsonl","quality_samples.jsonl","applied_force_samples.jsonl"):
        (directory/name).write_text('{"synthetic_pack_unit_fixture_only":true}\n')
    record = {"contract":"cvpr-pico-episode-collection-record-v1", "status":"committed",
        "disposition":"kept","source_kind":"live_pico_human","human_demo_uuid":identity,
        "source_episode_directory":str(root/"collection/live_human_fixture/episodes"/identity),
        "episode_json_sha256":packer.sha(directory/"episode.json")}
    write(directory/"collection_record.json", record)
    checks = {name:{"passed":True} for name in packer.BASE_CHECKS}
    if deferred:
        checks["original_full_state_to_deferred_rtx_bridge"] = {"passed":True}
        # Byte-only stand-ins: no MJB is ever constructed or executed here.
        for name in ("rtx_completion.json", "deferred_capture/capture.json"):
            write(directory/name, {"synthetic_pack_unit_fixture_only":True})
        for name in ("rtx_frame_bindings.jsonl","deferred_capture/events.jsonl"):
            path = directory/name; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('{"synthetic_pack_unit_fixture_only":true}\n')
        np.save(directory/"deferred_capture/states.npy", np.zeros((count, 12), np.float64))
        (directory/"deferred_capture/model.mjb").write_bytes(b"NOT AN MJB: CPU pack fixture only")
    audit = {"contract":packer.ADMISSION_CONTRACT,
        "status":"independent_live_human_episode_audit_passed","eligible_for_packing":True,
        "accepted_new_dataset":False,"ckpt_feature_pack_ready":False,"errors":[],
        "human_demo_uuid":identity,"event_count":count,"checks":checks,
        "generated_utc":datetime.now(timezone.utc).isoformat(),
        "auditor_source_sha256":{str(Path(packer.__file__).with_name(name)):
            packer.sha(Path(packer.__file__).with_name(name)) for name in packer.AUDITOR_FILES}}
    selection = root/"selection.json"
    case = dict(root=root, template=template, directory=directory, events=events,
        identity=identity, metadata=metadata, audit=audit, selection=selection,
        instruction=instruction, output=root/"packed")
    refresh(case)
    return case


def refresh(case):
    directory = case["directory"]
    case["audit"]["bindings"] = {str(p.resolve()):packer.sha(p)
        for p in directory.rglob("*") if p.is_file() and p.name != "admission.json"}
    case["audit"]["bindings"].update(case.get("external_bindings", {}))
    write(directory/"admission.json", case["audit"])
    protocol = packer.task_protocol(case["metadata"]["task_id"])
    write(case["selection"], {"contract":protocol["selection_contract"],
        **({"task_id":packer.MOBILE_TASK_ID} if protocol["mobile"] else {}),"episodes":[{
        "human_demo_uuid":case["identity"],"episode_dir":str(directory),
        "admission_path":str(directory/"admission.json"),
        "admission_sha256":packer.sha(directory/"admission.json"),"source_session":"live_human_fixture"}]})


def make_mobile_case(root):
    case = make_case(root)
    directory = case['directory']
    case['metadata'].update(task_id=packer.MOBILE_TASK_ID,
        quality_geometry_mode='deferred_geometry_with_original_solved_support',
        instruction=' Pick up the cracker box, walk to table2, and place it. ')
    case['instruction'] = case['metadata']['instruction']
    write(directory/'metadata.json',case['metadata'])
    manifest = packer.read(directory/'capture/capture.json')
    manifest.update(task_id=packer.MOBILE_TASK_ID,instruction=case['instruction'])
    write(directory/'capture/capture.json',manifest)
    (directory/'quality_samples.jsonl').rename(directory/'raw_quality_samples.jsonl')
    audit = case['audit']
    audit.update(contract=packer.MOBILE_ADMISSION_CONTRACT,task_id=packer.MOBILE_TASK_ID,
        selection_policy='native_only',technical_blocked=False,
        checks={name:{'passed':True} for name in packer.MOBILE_BASE_CHECKS},
        auditor_source_sha256={str(Path(packer.__file__).with_name(name)):
            packer.sha(Path(packer.__file__).with_name(name)) for name in packer.MOBILE_AUDITOR_FILES})
    trace = root/'original_session_control_trace.jsonl'
    trace.write_text('{"synthetic_control_trace_fixture_only":true}\n')
    source = root/'historical_pico_simple_teleop.py'
    source.write_text('# Historical capture implementation CPU fixture only\n')
    historical = root/'HISTORICAL_SOURCE_IDENTITIES.json'
    write(historical,{'files':[{'filename':source.name,'path':str(source),'sha256':packer.sha(source)}]})
    audit['mobile_reference_audit'] = {'control_trace':{'path':str(trace),'sha256':packer.sha(trace)},
        'historical_sources':{'manifest_path':str(historical),'manifest_sha256':packer.sha(historical),
            'files':[{'filename':source.name,'path':str(source),'sha256':packer.sha(source)}]}}
    case['external_bindings'] = {str(path):packer.sha(path) for path in (trace,source,historical)}
    refresh(case)
    return case


def selected(case):
    return packer.load_selection(case["selection"], packer.Bindings())



def rejected_case(test, tmp_path, change):
    case = make_case(tmp_path)
    directory = case["directory"]
    if change == "replay":
        write(directory/"metadata.json", {**case["metadata"],"source_kind":"recorded_pico_diagnostic"})
    elif change == "discarded":
        record = packer.read(directory/"collection_record.json"); record["disposition"] = "discarded"
        write(directory/"collection_record.json", record)
    elif change == "missing_check":
        case["audit"]["checks"].pop("raw_fullbody_and_actual_command_ledger")
    elif change == "bad_auditor":
        key = next(iter(case["audit"]["auditor_source_sha256"]))
        case["audit"]["auditor_source_sha256"][key] = "f"*64
    elif change == "missing_authority":
        (directory/"commands.jsonl").unlink()
    elif change == "bad_instruction":
        manifest = packer.read(directory/"capture/capture.json"); manifest["instruction"] = "a different task"
        write(directory/"capture/capture.json", manifest)
    elif change == "pending_collection":
        record = packer.read(directory/"collection_record.json"); record["status"] = "pending"
        write(directory/"collection_record.json", record)
    elif change == "missing_warmup":
        manifest = packer.read(directory/"capture/capture.json"); manifest["warmup_seconds"] = 0
        write(directory/"capture/capture.json", manifest)
    elif change == "clock_gap":
        rows = packer.jsonl(directory/"capture/events.jsonl"); rows[35]["physical_tick50"] += 1
        path = directory/"capture/events.jsonl"; path.write_text("".join(json.dumps(r)+"\n" for r in rows))
        manifest = packer.read(directory/"capture/capture.json"); manifest["events_sha256"] = packer.sha(path)
        write(directory/"capture/capture.json", manifest)
    elif change in ("nonfinite_state", "nonfinite_token", "nonfinite_hand"):
        rows = packer.jsonl(directory/"capture/events.jsonl")
        if change == "nonfinite_state":
            rows[35]["state43"][0] = "NaN"
        else:
            key = "body_token64" if change == "nonfinite_token" else "hand14"
            rows[35]["action"][key][0] = "NaN"
        path = directory/"capture/events.jsonl"
        path.write_text("".join(json.dumps(r)+"\n" for r in rows))
        manifest = packer.read(directory/"capture/capture.json")
        manifest["events_sha256"] = packer.sha(path)
        write(directory/"capture/capture.json", manifest)
    elif change == "missing_start_receipt":
        write(directory/"metadata.json", {**case["metadata"],"capture_start_protocol":"cvpr-pico-frozen-capture-start-v1"})
    refresh(case)
    selection = packer.read(case["selection"])
    if change == "bad_admission_sha":
        selection["episodes"][0]["admission_sha256"] = "f"*64
    elif change == "bad_bound_sha":
        (directory/"commands.jsonl").write_text("modified after audit\n")
    elif change == "duplicate_uuid":
        selection["episodes"].append(deepcopy(selection["episodes"][0]))
    write(case["selection"], selection)
    with test.assertRaises((ValueError, KeyError)):
        packer.pack(case["template"], case["selection"], case["output"])
    assert not (case["output"]/"PACK_READY.json").exists()




class PackerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_unqualified_or_changed_evidence_never_enters_pack(self):
        for change in ("replay", "discarded", "bad_admission_sha", "bad_bound_sha", "duplicate_uuid",
                "missing_check", "bad_auditor", "missing_authority", "bad_instruction",
                "pending_collection", "missing_warmup", "clock_gap", "missing_start_receipt",
                "nonfinite_state", "nonfinite_token", "nonfinite_hand"):
            with self.subTest(change=change):
                root = self.root/change
                root.mkdir()
                rejected_case(self, root, change)

    def test_original_evidence_directories_cannot_be_used_as_output(self):
        case = make_case(self.root)
        for output in (case["directory"]/"packed", case["template"]/"packed"):
            with self.subTest(output=output):
                with self.assertRaisesRegex(ValueError, "outside original"):
                    packer.pack(case["template"], case["selection"], output)
                self.assertFalse(output.exists())
        alias = self.root/"alias-output"
        alias.symlink_to(self.root/"missing-output")
        with self.assertRaisesRegex(ValueError, "alias"):
            packer.pack(case["template"], case["selection"], alias)

    def test_actual_cpu_conversion_preserves_commands_joint_order_masks_time_and_pixels(self):
        tmp_path = self.root
        case = make_case(tmp_path)
        original = {str(p):packer.sha(p) for p in case["directory"].rglob("*") if p.is_file()}
        report = packer.pack(case["template"], case["selection"], case["output"])
        assert report["contract"] == packer.CONTRACT
        assert report["train_episodes"] == 1 and report["total_frames"] == 65 and report["train_windows"] == 6
        assert report["final_ready"] is False and report["DINO_cache_generated"] is False
        assert report["normalization_refitted"] is False and report["tokens_reencoded"] is False
        assert report["accepted_new_dataset"] is False
        assert report["cpu_reader_validation"]["status"] == "actual_cpu_LeRobot_numeric_video_queries_passed"
        receipt = report["episodes"][0]
        assert receipt["video"]["pixels_exact"] is True
        assert receipt["video"]["raw_rgb_sha256"] == receipt["video"]["decoded_rgb_sha256"]
        assert receipt["instruction"] == case["instruction"]
        assert receipt["source_identity"]["human_demo_uuid"] == case["identity"]
        assert receipt["source_identity"]["source_namespace"] == "live_pico_human"
        assert receipt["source_episode_index"] == 1_000_000
        assert receipt["split_restriction"] == "train_only"
        assert receipt["eligible_current_rows"] == list(range(30, 36))
        assert receipt["history_rows"]["30"] == [0, 6, 12, 18, 24, 30]
        table = pq.read_table(case["output"]/receipt["parquet_path"])
        np.testing.assert_array_equal(table["observation.state"][0].as_py(), np.arange(43))
        np.testing.assert_array_equal(np.asarray(table["action"][30].as_py(), np.float32),
                                     (np.arange(14)+.3).astype(np.float32))
        np.testing.assert_array_equal(table["action.body_token_v1_1"][30].as_py(),
                                     np.full(64, .3, dtype=np.float32))
        assert np.isnan(table["action.body_token_v1_1"][:30].to_pylist()).all()
        assert np.isnan(table["action"][:30].to_pylist()).all()
        assert table["action_mask"][29].as_py() == [False]*80
        assert table["action_mask"][30].as_py() == [True]*78+[False]*2
        assert table["action_valid"][:31].to_pylist() == [False]*30+[True]
        assert table["source_frame_index"].to_pylist() == [(i*50+29)//30 for i in range(65)]
        np.testing.assert_array_equal(table["timestamp"].to_pylist(), (np.arange(65)/30).astype(np.float32))
        assert table["task_description"].to_pylist() == [case["instruction"]]*65
        for name in packer.SOURCE_FILES:
            assert (case["output"]/name).read_bytes() == (case["template"]/name).read_bytes()
        for relative, expected in report["payload_sha256"].items():
            assert packer.sha(case["output"]/relative) == expected
        assert {str(p):packer.sha(p) for p in case["directory"].rglob("*") if p.is_file()} == original
        assert not (case["output"]/"val").exists()


    def test_deferred_evidence_and_declared_start_receipt_are_copied(self):
        tmp_path = self.root
        case = make_case(tmp_path, deferred=True)
        metadata = dict(case["metadata"], capture_start_protocol="cvpr-pico-frozen-capture-start-v1")
        write(case["directory"]/"metadata.json", metadata)
        write(case["directory"]/"capture_start_receipt.json", {"synthetic_pack_unit_fixture_only":True})
        case["audit"]["checks"]["independent_frozen_first_capture_boundary"] = {"passed":True}
        refresh(case)
        report = packer.pack(case["template"], case["selection"], case["output"])
        evidence = report["episodes"][0]["evidence_files"]
        for name in ("metadata.json","commands.jsonl","collection_record.json",
                     "capture_start_receipt.json","rtx_completion.json","deferred_capture/states.npy",
                     "deferred_capture/model.mjb","capture/events.jsonl"):
            assert (case["output"]/evidence[name]).read_bytes() == (case["directory"]/name).read_bytes()
        assert not any(name.endswith(".png") for name in evidence)

    def test_pending_discard_and_operator_discard_block_even_after_good_admission(self):
        tmp_path = self.root
        case = make_case(tmp_path)
        directory = case["directory"]
        spool = directory.parent.parent/".pico_episode_review/discard_requests"
        spool.mkdir(parents=True)
        write(spool/(case["identity"]+".json"), {"uuid":case["identity"],"operator_discard_requested":True})
        with self.assertRaisesRegex(ValueError, "discard intent"):
            selected(case)
        (spool/(case["identity"]+".json")).unlink()
        write(directory/"operator_discard.json", {"reason":"operator discard"})
        with self.assertRaisesRegex(ValueError, "Operator discarded"):
            selected(case)


    def test_bound_inputs_are_rechecked_before_publication(self):
        tmp_path = self.root
        case = make_case(tmp_path)
        real_reader = packer.legacy.verify_lerobot_reader
        def change_after_conversion(*args):
            result = real_reader(*args)
            (case["directory"]/"commands.jsonl").write_text("changed during packing\n")
            return result
        with patch.object(packer.legacy, "verify_lerobot_reader", change_after_conversion):
            with self.assertRaisesRegex(ValueError, "Evidence changed while packaging"):
                packer.pack(case["template"], case["selection"], case["output"])
        assert not case["output"].exists()
        assert len(list(tmp_path.glob(".packed.partial-*/PACK_FAILED.json"))) == 1

    def test_mobile_actual_lossless_conversion_keeps_task_labels_and_portable_original_evidence(self):
        import verify_pico_handover as verifier
        case = make_mobile_case(self.root)
        report = packer.pack(case['template'],case['selection'],case['output'])
        self.assertEqual(report['contract'],packer.MOBILE_CONTRACT)
        self.assertEqual(report['task_id'],packer.MOBILE_TASK_ID)
        self.assertFalse(report['tokens_reencoded'])
        self.assertFalse(report['normalization_refitted'])
        episode = report['episodes'][0]
        self.assertTrue(episode['video']['pixels_exact'])
        self.assertEqual(episode['instruction'],case['instruction'])
        self.assertEqual(episode['task_id'],packer.MOBILE_TASK_ID)
        self.assertEqual(episode['eligible_current_rows'],list(range(30,36)))
        self.assertIn('raw_quality_samples.jsonl',episode['evidence_files'])
        self.assertNotIn('quality_samples.jsonl',episode['evidence_files'])
        table = pq.read_table(case['output']/episode['parquet_path'])
        np.testing.assert_array_equal(table['action.body_token_v1_1'][30].as_py(),np.full(64,.3,np.float32))
        self.assertTrue(np.isnan(table['action.body_token_v1_1'][:30].to_pylist()).all())
        self.assertEqual(table['task_description'].to_pylist(),[case['instruction']]*65)
        trace = episode['mobile_control_trace_file']
        self.assertEqual((case['output']/trace['path']).read_bytes(),Path(trace['original_path']).read_bytes())
        historical = episode['mobile_historical_source_files']
        for value in [historical['manifest'],*historical['files']]:
            self.assertEqual((case['output']/value['path']).read_bytes(),Path(value['original_path']).read_bytes())
        self.assertFalse(set(case['external_bindings']) & set(episode['external_authority_bindings']))
        verifier.inspect_pack(case['output'],expected_episodes=1)

    def test_mobile_shared_trace_and_historical_source_are_copied_once_per_sha(self):
        first_root,second_root = self.root/'first',self.root/'second'
        first_root.mkdir();second_root.mkdir()
        first,second = make_mobile_case(first_root),make_mobile_case(second_root)
        selection = packer.read(first['selection'])
        selection['episodes'] += packer.read(second['selection'])['episodes']
        write(first['selection'],selection)
        report = packer.pack(first['template'],first['selection'],first['output'])
        a,b = report['episodes']
        self.assertEqual(a['mobile_control_trace_file']['path'],b['mobile_control_trace_file']['path'])
        self.assertEqual(a['mobile_historical_source_files']['files'][0]['path'],
                         b['mobile_historical_source_files']['files'][0]['path'])
        self.assertEqual(len(list((first['output']/'accepted_evidence/session_traces').glob('*.jsonl'))),1)
        self.assertEqual(len(list((first['output']/'accepted_evidence/historical_sources').rglob('*.py'))),1)

    def test_mobile_refuses_handover_contract_missing_native_authority_and_unbound_original_sources(self):
        for change in ('selection','admission','task','policy','technical','native_check','continuous_check','auditor',
                       'raw_samples','trace','historical_sources'):
            with self.subTest(change=change):
                root = self.root/change;root.mkdir();case = make_mobile_case(root)
                if change == 'selection':
                    value=packer.read(case['selection']);value['contract']=packer.SELECTION_CONTRACT;write(case['selection'],value)
                elif change == 'admission':
                    case['audit']['contract']=packer.ADMISSION_CONTRACT;refresh(case)
                elif change == 'task':
                    case['metadata']['task_id']=packer.HANDOVER_TASK_ID
                    write(case['directory']/'metadata.json',case['metadata'])
                    case['metadata']['task_id']=packer.MOBILE_TASK_ID;refresh(case)
                elif change in ('policy','technical'):
                    case['audit']['selection_policy' if change=='policy' else 'technical_blocked'] = (
                        'extra_geometry' if change=='policy' else True)
                    refresh(case)
                elif change in ('native_check','continuous_check'):
                    key='independent_native_mobile_task_success' if change=='native_check' else 'independent_mobile_continuous_reference'
                    case['audit']['checks'].pop(key);refresh(case)
                elif change == 'auditor':
                    case['audit']['auditor_source_sha256'].pop(next(iter(case['audit']['auditor_source_sha256'])));refresh(case)
                elif change == 'raw_samples':
                    (case['directory']/'raw_quality_samples.jsonl').unlink();refresh(case)
                else:
                    case['audit']['mobile_reference_audit'].pop('control_trace' if change=='trace' else change);refresh(case)
                with self.assertRaises((ValueError,KeyError)):
                    packer.pack(case['template'],case['selection'],case['output'])
                self.assertFalse(case['output'].exists())


if __name__ == "__main__":
    unittest.main()
