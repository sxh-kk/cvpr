"""CPU contracts for paired reporting; fixtures are not physical acceptance runs."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import compare_liftbox_eval as compare


def rng(number):
    digest = hashlib.sha256(str(number).encode()).hexdigest()
    return {"python": digest, "numpy": digest, "torch_cpu": digest, "torch_cuda": [digest]}


def physical_fixture(success=False):
    rows = []
    count = 51 if success else 3
    for index in range(count):
        now, nxt = .005 + index * .02, .025 + index * .02
        hold = index * .02 if success else 0.
        p = {"metric": compare.METRIC, "measurement_hz": 50, "continuity_gap": False,
             "sim_time_s": nxt, "box_lowest_z_m": .15 if success else 0., "floor_z_m": 0.,
             "hand_normal_force_n": 2. if success else 0., "hold_sim_s": hold,
             "clearance_ge_10cm": success, "loaded_hand_contact": success,
             "floor_contact": not success, "table_contact": False,
             "condition": success, "success": success and index == 50}
        rows.append({"sim_time_s": now, "next_sim_time_s": nxt, "pelvis_height_m": .8,
                     "pelvis_tilt_degrees": 0., "fell": False, "task_progress": p,
                     "terminated": p["success"]})
    episode = {"status": "task_terminated" if success else "sim_time_limit", "finite_states": True,
               "diagnostic_loop_executed": True, "simulator_unchanged_during_http": True,
               "decoder_calls": count, "policy_frames": count - 1, "warmup_frames": 1,
               "fell": False, "simulated_seconds": count * .02, "frames": count,
               "final_task_progress": rows[-1]["task_progress"], "task_success_observed": success,
               "task_success_attributed_to_policy": False}
    return episode, rows


def service_fixture():
    startup = {"weight_sha256": compare.BASE_SHA, "sampling_seed": 5, "rng_sha256": rng(0),
               "host": "127.0.0.1", "Ta": 30, "rtc": False, "inference_steps": 10}
    rows = [{"ordinal": index + 1, "episode_id": "one", "session_id": "session-a", "sim_time_s": str(index),
             "wall_time_s": .1, "status": 425 if index == 0 else 200,
             "rng_before": rng(max(index - 1, 0)), "rng_after": rng(index)} for index in range(3)]
    episode = {"episode_id": "one", "http_counts": {"/act": 3}, "policy_chunks": 2,
               "http_425_count": 1, "successful_action_http_latencies_s": [.1, .1]}
    return startup, rows, episode


def paired_fixture():
    episode, trace = physical_fixture()
    startup, requests, client = service_fixture()
    episode.update(client)
    episode.update(instruction=compare.INSTRUCTION, seed=10005, local_environment_config={"task": "fixed"},
                   physics_dt_s=.005, render_hz=50, action_fps=30, sonic_decoder_hz=50,
                   execute_steps=30, warmup_phase_counts={"real": 1}, initial_pelvis_position_m=[0, 0, .8],
                   initial_pelvis_quaternion_wxyz=[1, 0, 0, 0], wall_seconds=1.,
                   warmup_control={"helper_source_sha256": "a" * 64, "reference": "live", "events": [1]},
                   initial_physics={key: "b" * 64 for key in ("source_result_sha256", "source_mjb_sha256",
                       "source_trajectory_sha256", "initial_state_sha256", "restored_state_sha256")})
    episode["initial_physics"].update(source_seed=10005, state_spec=8191, state_size=548, source_row=0, source_sim_time_s=.005)
    result = {"controller_identity": {"decoder_sha256": "c" * 64, "compatibility_evidence": "per-run"},
              "render_contract": {"renderer": "fixed"}, "observation_phase": "synced", "server_info": {"history": 6},
              "formal_evaluation_eligible": False}
    return {"result": result, "episode": episode, "startup": startup, "successful": requests[1:],
            "metrics": compare.physics_metrics(episode, trace, .06), "artifacts": {}}


def add_projection(case, mode):
    count = case["episode"]["policy_frames"]
    fsq = mode == "fsq_v1_1"
    receipt = {"mode":mode,"consumer_source_sha256":"a"*64,
        "step":1/16 if fsq else None,"range":[-1.,15/16] if fsq else None,
        "levels":32 if fsq else None,"encoder_sha256":compare.V11_ENCODER_SHA if fsq else None,
        "rounding":"nearest ties-to-even" if fsq else None,"command_samples":count,
        "elements_compared":64*count,"changed_elements":64*count if fsq else 0,
        "range_clipped_elements":0,"absolute_change_sum":.01*64*count if fsq else 0.,
        "mean_absolute_change":.01 if fsq else 0.,"maximum_absolute_change":.02 if fsq else 0.}
    case["episode"]["body_token_projection"] = receipt
    case["result"]["body_token_projection"] = copy.deepcopy(receipt)
    return case


def write_fit(root):
    def write(name, value):
        (root / name).write_text(json.dumps(value))
    weights = root / "model.safetensors"
    weights.write_bytes(b"CPU unit test placeholder, never loaded")
    stats = root / "statistics.json"
    stats.write_text('{}')
    field = {"stat_path": str(stats), "action_min": [-1.] * 80, "action_max": [1.] * 80,
             "state_min": [-1.] * 45, "state_max": [1.] * 45, "use_norm_mask": False, "action_norm_type": "bounds"}
    normalization = {"stat_path": str(stats), "stat_file_sha256": compare.sha256(stats),
                     "bounds_sha256": {k: hashlib.sha256(json.dumps(field[k], separators=(',', ':')).encode()).hexdigest()
                                       for k in ("action_min", "action_max", "state_min", "state_max")}}
    write("run_config.json", {"data": {"transform": {"field": field}}})
    write("training_result.json", {"status": "completed", "comparison_complete": True, "completed_updates": 300,
          "maximum_updates": 300, "initial_global_step": 0, "frozen_unchanged": True, "action_changed": True,
          "source_joint_sha256": compare.BASE_SHA, "final_checkpoint_sha256": "d" * 64,
          "final_checkpoint": str(weights), "normalization": normalization})
    write("preparation.json", {"source_step": 28000, "source_joint_sha256": compare.BASE_SHA,
          "initial_global_step": 0, "optimizer_scheduler_rng_restored": False, "normalization": normalization,
          "launcher_arguments": {"seed": 7}})
    windows = [{"ordinal": i, "episode": 0, "source_episode": 4, "frame": 30 + i,
                "last_future_frame": 59 + i, "history_frames": [i + k * 6 for k in range(6)],
                "input_sha256": hashlib.sha256(str(i).encode()).hexdigest(),
                "fm_seed": 10007 + i * 2, "inference_seed": 10008 + i * 2} for i in range(20)]
    write("test_windows.json", {"split": "test", "no_aug": True, "windows": windows})
    for stage, updates, prediction in (("before", 0, .2), ("after", 300, .1)):
        predictions = np.full((20, 30, 80), prediction, np.float32)
        target = np.zeros_like(predictions)
        denormalized = .5 * (predictions + 1) * 2 - 1
        values = {"fm_loss": 1.}
        for name, sl in (("body", slice(0, 64)), ("hand", slice(64, 78))):
            values[f"{name}_normalized_action_l1"] = float(np.abs(predictions[:, :, sl]).mean())
            values[f"{name}_denormalized_action_l1"] = float(np.abs(denormalized[:, :, sl]).mean())
        rows = [{"ordinal": i, "episode": 0, **values} for i in range(20)]
        write(f"test_{stage}.json", {"stage": stage, "optimizer_updates_completed": updates,
              "window_count": 20, "mean": values, "windows": rows})
        np.savez(root / f"test_{stage}_actions.npz", predictions=predictions, normalized_targets=target, raw_targets=target)


def write_joint_fit(root, source_step=31000):
    """Synthetic JSON/arrays only, never an actual training witness."""
    write_fit(root)
    source = root / "source" / "checkpoints" / f"ckpt_{source_step}" / "model.safetensors"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"CPU joint fixture source placeholder")
    dino = root / "dino-fixture.bin"
    dino.write_bytes(b"frozen DINO fixture")
    config = json.loads((root / "run_config.json").read_text())
    config["model"] = {"tune_vlm":True,"tune_motion":True,"pretrained_joint_checkpoint":str(source)}
    (root / "run_config.json").write_text(json.dumps(config))
    preparation = json.loads((root / "preparation.json").read_text())
    source_sha = hashlib.sha256(b"independent31k").hexdigest()
    expected = {name:{"trainable_parameters":index+10,"peak_lr":1e-5}
                for index,name in enumerate(compare.JOINT_GROUPS)}
    preparation.update(training_mode="joint_vlm_motion_action",source_step=source_step,source_joint_sha256=source_sha,
                       source_checkpoint=str(source),expected_groups=expected,dino_file_sha256={str(dino):compare.sha256(dino)})
    (root / "preparation.json").write_text(json.dumps(preparation))
    def optimizer(steps):
        groups = [{"group_name":name,"lr":0.,"numel":expected[name]["trainable_parameters"],
                   "parameter_tensors":3,"public_devices":["cuda:0"],"master_devices":["cpu"],
                   "state_devices":["cpu"] if steps else [],"initialized_states":2 if steps else 0,
                   "parameter_step_min":17 if steps else 0,"parameter_step_max":steps} for name in compare.JOINT_GROUPS]
        return {"completed_steps":steps,"master_elements":sum(v["trainable_parameters"] for v in expected.values()),
                "master_device_types":["cpu"],"native_adamw_state_entries":10 if steps else 0,
                "actual_state_devices":["cpu"] if steps else [],
                "wrapper_receipt":{"schema":"cvpr_cpu_master_adamw_v1","native_optimizer":"torch.optim.AdamW",
                    "completed_steps":steps,"model_and_master_dtype":"float32","copy_mode":"synchronous","groups":groups}}
    training = json.loads((root / "training_result.json").read_text())
    training.pop("frozen_unchanged");training.pop("action_changed")
    training.update(training_mode="joint_vlm_motion_action",normalization=preparation["normalization"],
        source_joint_sha256=source_sha,all_five_groups_changed=True,dino_files_unchanged=True,
        before_group_sha256={name:hashlib.sha256((name+"before").encode()).hexdigest() for name in compare.JOINT_GROUPS},
        after_group_sha256={name:hashlib.sha256((name+"after").encode()).hexdigest() for name in compare.JOINT_GROUPS},
        five_group_peak_contract=[{"name":name,"trainable_parameters":v["trainable_parameters"],"lr":v["peak_lr"]} for name,v in expected.items()],
        initial_optimizer=optimizer(0),final_optimizer=optimizer(300),
        updates=[{"completed_step":step,"loss_last_microbatch":1.,"pre_clip_grad_norms":{name:.1 for name in compare.JOINT_GROUPS},
                  "used_group_lrs":{f"lr_{name}":0. if step==1 else 1e-5 for name in compare.JOINT_GROUPS},
                  "next_group_lrs":{f"lr_{name}":1e-5 for name in compare.JOINT_GROUPS}} for step in range(1,301)])
    (root / "training_result.json").write_text(json.dumps(training))
    return training,preparation


class PairedReportingTests(unittest.TestCase):
    def test_prefix_v2_curated_offline_test_keeps_original_physical_failures(self):
        gate = {"dataset_contract": "cvpr-simple-lift-real-warmup-prefix-v2",
                "test_source_episodes": [9],
                "test_demo_curation": {"selected_test_source": 9, "candidate_order": [4, 9, 10, 11]}}
        self.assertEqual(compare.offline_test_source({"quality_gate": gate}, "joint_vlm_motion_action"), 9)
        self.assertEqual(compare.physical_case_ids({"dataset_contract": gate["dataset_contract"], "source_episode": 9}), [4, 8, 9])
        self.assertEqual(compare.physical_case_ids({"source_episode": 4}), [4, 8])
        self.assertEqual(compare.offline_test_source({"quality_gate": {"test_source_episodes": [9]}}, "joint_vlm_motion_action"), 4)
        for change in ({"test_source_episodes": [3]}, {"test_demo_curation": {}},
                       {"test_source_episodes": [True]}, {"test_source_episodes": [9, 10]}):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "explicit, audited"):
                compare.offline_test_source({"quality_gate": {**gate, **change}}, "joint_vlm_motion_action")
        with self.assertRaises(ValueError):
            compare.offline_test_source({"quality_gate": gate}, "action_only")
        with self.assertRaises(ValueError):
            compare.physical_case_ids({"source_episode": 9})

    def test_completed_nofall_smoke_is_not_task_success(self):
        episode, rows = physical_fixture()
        metrics = compare.physics_metrics(episode, rows, .06)
        self.assertTrue(metrics["completed_physical_attempt"])
        self.assertTrue(metrics["no_fall"])
        self.assertFalse(metrics["physical_task_success"])

    def test_true_terminal_hold_can_succeed_without_formal_policy_attribution(self):
        episode, rows = physical_fixture(True)
        metrics = compare.physics_metrics(episode, rows, 25)
        self.assertTrue(metrics["physical_task_success"])
        self.assertFalse(metrics["producer_policy_attribution"])
        self.assertEqual(metrics["measurement_hz"], 50)

    def test_truncated_clock_and_fabricated_hold_rejected(self):
        episode, rows = physical_fixture(True)
        with self.assertRaisesRegex(ValueError, "trace continuity"):
            compare.physics_metrics(episode, rows[:3] + rows[4:], 25)
        rows[2]["task_progress"]["hold_sim_s"] = float('nan')
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            compare.physics_metrics(episode, rows, 25)

    def test_sum_of_subthreshold_contacts_cannot_invent_loaded_contact(self):
        episode, rows = physical_fixture()
        for row in rows:
            row["task_progress"]["hand_normal_force_n"] = .019
        self.assertFalse(compare.physics_metrics(episode, rows, .06)["physical_task_success"])

    def test_service_warmup_rng_and_chain_checked(self):
        startup, rows, episode = service_fixture()
        self.assertEqual(len(compare.service_audit(startup, rows, episode, compare.BASE_SHA)), 2)
        rows[0]["rng_after"] = rng(8)
        with self.assertRaisesRegex(ValueError, "Warmup request consumed RNG"):
            compare.service_audit(startup, rows, episode, compare.BASE_SHA)

    def test_successful_rng_prefix_allows_early_end_but_detects_changed_noise(self):
        before = paired_fixture()
        after = copy.deepcopy(before)
        after["successful"] = after["successful"][:1]
        self.assertEqual(compare.pair_cases(4, before, after)["rng_successful_request_prefix_matched"], 1)
        after["successful"][0]["rng_after"] = rng(40)
        with self.assertRaisesRegex(ValueError, "RNG differs"):
            compare.pair_cases(4, before, after)

    def test_changed_initial_state_warmup_or_controller_rejected(self):
        before = paired_fixture()
        for name in ("initial", "warmup", "controller"):
            after = copy.deepcopy(before)
            if name == "initial":
                after["episode"]["initial_physics"]["initial_state_sha256"] = "f" * 64
            elif name == "warmup":
                after["episode"]["warmup_control"]["reference"] = "different"
            else:
                after["result"]["controller_identity"]["decoder_sha256"] = "e" * 64
            with self.subTest(name=name), self.assertRaises(ValueError):
                compare.pair_cases(4, before, after)

    def test_projection_change_cannot_be_called_same_protocol_pair(self):
        for before in (paired_fixture(), add_projection(paired_fixture(), "none")):
            after = add_projection(paired_fixture(), "fsq_v1_1")
            with self.subTest(legacy="body_token_projection" not in before["episode"]), self.assertRaisesRegex(ValueError, "Paired body-token projection"):
                compare.pair_cases(4, before, after)

    def test_same_projection_matches_recipe_not_outcome_counters(self):
        before = add_projection(paired_fixture(), "fsq_v1_1")
        after = copy.deepcopy(before)
        receipt = after["episode"]["body_token_projection"]
        receipt.update(mean_absolute_change=.012, absolute_change_sum=.012*receipt["elements_compared"], maximum_absolute_change=.025)
        result = compare.pair_cases(4, before, after)
        self.assertEqual(result["body_token_projection"]["mode"], "fsq_v1_1")
        self.assertEqual(compare.pair_cases(4, paired_fixture(), paired_fixture())["body_token_projection"]["mode"], "none")

    def test_wrong_fsq_range_declared_mode_or_command_counts_rejected(self):
        for corruption in ("legacy_range", "declared_mode", "counts", "consumer_source"):
            case = add_projection(paired_fixture(), "fsq_v1_1")
            if corruption == "legacy_range":case["episode"]["body_token_projection"]["range"] = [-.625,.625]
            elif corruption == "declared_mode":case["result"]["body_token_projection"]["mode"] = "none"
            elif corruption == "counts":case["episode"]["body_token_projection"]["command_samples"] += 1
            else:case["episode"]["body_token_projection"]["consumer_source_sha256"] = "not_a_sha"
            with self.subTest(corruption=corruption), self.assertRaisesRegex(ValueError, "projection"):
                compare.body_token_projection_contract(case["result"],case["episode"])

    def test_offline_comparison_independently_recomputes_raw_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_fit(root)
            _, report = compare.offline_comparison(root)
            self.assertLess(report["after_minus_before"]["body_denormalized_action_l1"], 0)
            path = root / "test_after.json"
            bad = json.loads(path.read_text())
            bad["mean"]["body_denormalized_action_l1"] = 0.
            for row in bad["windows"]:
                row["body_denormalized_action_l1"] = 0.
            path.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, "raw/denormalized L1"):
                compare.offline_comparison(root)

    def test_changed_test_seed_or_target_is_not_paired(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_fit(root)
            path = root / "test_windows.json"
            data = json.loads(path.read_text())
            data["windows"][0]["fm_seed"] += 1
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "input/seed/history"):
                compare.offline_comparison(root)
            write_fit(root)
            path = root / "test_after_actions.npz"
            with np.load(path) as archive:
                arrays = {k: archive[k] for k in archive.files}
            arrays["raw_targets"][0, 0, 0] = 1.
            np.savez(path, **arrays)
            with self.assertRaisesRegex(ValueError, "TEST targets changed"):
                compare.offline_comparison(root)

    def test_joint31k_mode_requires_its_own_actual_update_witness(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);training,preparation=write_joint_fit(root)
            _,report=compare.offline_comparison(root)
            self.assertEqual(report["training_mode"],"joint_vlm_motion_action")
            self.assertEqual(report["source_step"],31000)
            self.assertEqual(report["source_joint_sha256"],preparation["source_joint_sha256"])
            self.assertNotEqual(report["source_joint_sha256"],compare.BASE_SHA)
            self.assertTrue(report["update_witness"]["actual_cpu_adamw_receipts_verified"])

    def test_joint_label_cannot_borrow_action_frozen_or_incomplete_witness(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);training,preparation=write_joint_fit(root)
            for corruption in ("missing_sha","unchanged_group","missing_cpu_receipt","cuda_moment","bad_step","missing_grad","nan_grad","dino_changed","empty_dino_identity","missing_update"):
                bad=copy.deepcopy(training);prep=copy.deepcopy(preparation)
                bad.update(frozen_unchanged=True,action_changed=True)
                if corruption=="missing_sha":bad.pop("before_group_sha256")
                elif corruption=="unchanged_group":bad["after_group_sha256"]["motion_expert"]=bad["before_group_sha256"]["motion_expert"]
                elif corruption=="missing_cpu_receipt":bad.pop("final_optimizer")
                elif corruption=="cuda_moment":bad["final_optimizer"]["actual_state_devices"]=["cuda:0"]
                elif corruption=="bad_step":bad["final_optimizer"]["wrapper_receipt"]["groups"][0]["parameter_step_max"]=299
                elif corruption=="missing_grad":bad["updates"][10]["pre_clip_grad_norms"].pop("motion_expert")
                elif corruption=="nan_grad":bad["updates"][20]["pre_clip_grad_norms"]["lang_backbone"]=float('nan')
                elif corruption=="dino_changed":bad["dino_files_unchanged"]=False
                elif corruption=="empty_dino_identity":prep["dino_file_sha256"]={}
                else:bad["updates"].pop()
                with self.subTest(corruption=corruption),self.assertRaises(ValueError):
                    compare.joint_update_witness(bad,prep,300)

    def test_joint_lr_receipt_requires_actual_trainer_metric_names(self):
        with tempfile.TemporaryDirectory() as directory:
            training, preparation = write_joint_fit(Path(directory))
            # Captured schema of the production Trainer.group_lrs() receipt,
            # distinct from bare group names in gradient/hash receipts.
            actual = {"lr_action_header": 5e-6, "lr_lang_backbone": 5e-8,
                      "lr_vision_tower": 5e-7, "lr_mm_projector": 5e-6,
                      "lr_motion_expert": 5e-7}
            training["updates"][0]["next_group_lrs"] = actual
            compare.joint_update_witness(training, preparation, 300)
            for invalid in ({key.removeprefix("lr_"): value for key, value in actual.items()},
                            {key: value for key, value in actual.items() if key != "lr_motion_expert"},
                            {**actual, "lr_motion_expert": float("nan")}):
                bad = copy.deepcopy(training)
                bad["updates"][0]["next_group_lrs"] = invalid
                with self.subTest(keys=list(invalid)), self.assertRaisesRegex(ValueError, "LR|finite"):
                    compare.joint_update_witness(bad, preparation, 300)

    def test_legacy_action_frozen_and28k_checks_are_not_relaxed(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);write_fit(root)
            training=json.loads((root/"training_result.json").read_text())
            for field,value in (("frozen_unchanged",False),("action_changed",False),("training_mode","not_a_mode")):
                bad=copy.deepcopy(training);bad[field]=value
                (root/"training_result.json").write_text(json.dumps(bad))
                with self.subTest(field=field),self.assertRaises(ValueError):compare.offline_comparison(root)
            (root/"training_result.json").write_text(json.dumps(training))
            prep=json.loads((root/"preparation.json").read_text());prep["source_step"]=31000
            (root/"preparation.json").write_text(json.dumps(prep))
            with self.assertRaisesRegex(ValueError,"Action-only"):compare.offline_comparison(root)


if __name__ == "__main__":
    unittest.main()
