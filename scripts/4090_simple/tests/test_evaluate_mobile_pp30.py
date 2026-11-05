"""CPU-only campaign identity, ownership and actual-evidence acceptance tests."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evaluate_mobile_pp30 as evaluate


def encoder():
    value = {key: "fixture" for key in ("torch_version", "transformers_version", "preprocessing", "image_process",
                                       "normalization", "inference_dtype", "storage_dtype", "token_order")}
    value.update(shape=[5, 1024], local_file_sha256={"model": "weights"}, optimization={key: "fixture" for key in
        ("inference_precision", "compute_precision", "attention_resolved", "output_storage_dtype", "compile_model", "preprocessor", "pillow_version")})
    value["weights_sha256"] = hashlib.sha256(json.dumps(value["local_file_sha256"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return value


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.args = SimpleNamespace(deployment=self.root / "deployment", data_root=self.root / "data",
            training_manifest=self.root / "training_manifest.json", output=self.root / "campaign", runtime_root=self.root / "runtime",
            checkpoint_step=6000, port=21075, sampling_seed=20261105)
        model = self.args.deployment / "checkpoints/ckpt_6000/model.safetensors"
        model.parent.mkdir(parents=True)
        model.write_bytes(b"tiny exact checkpoint identity")
        self.args.expected_model_sha256 = evaluate.sha(model)
        self.write(self.args.training_manifest, {"encoder": encoder()})
        self.ready = {"contract": evaluate.READY_CONTRACT, "status": evaluate.READY_STATUS,
            "task_id": "simple/" + evaluate.TASK, "task_step": 6000, "step": 6000,
            "joint_sha256": self.args.expected_model_sha256, "training_completed": False,
            "checkpoint_save_completed": True, "stopped_by_user": True,
            "component_load_validation": {"status": "passed"}, "training_manifest_sha256": evaluate.sha(self.args.training_manifest)}
        self.ready["deployment_artifacts"] = {}
        for name in evaluate.DEPLOYMENT_ARTIFACTS:
            path = self.args.deployment / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"frozen deployed runtime artifact")
            self.ready["deployment_artifacts"][name] = {"sha256": evaluate.sha(path), "bytes": path.stat().st_size}
        self.write(self.args.deployment / "LOCAL_RUN_READY.json", self.ready)
        self.metadata = []
        for level in range(3):
            path = self.args.data_root / "eval" / evaluate.TASK / f"dr-level-{level}/meta/episodes.jsonl"
            path.parent.mkdir(parents=True)
            rows = []
            for case in range(10):
                config = {"uid": evaluate.TASK_UID, "metadata": {"physics_dt": .005, "render_hz": 50,
                    "max_episode_steps": 1200, "success_criteria": .4}, "dr_state_dict": {"level": level, "case": case},
                    "asset": "/home/official/SIMPLE/data/asset.usd"}
                rows.append({"episode_index": case, "environment_config": json.dumps(config)})
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            self.metadata.append(path)
        self.metadata_patch = patch.object(evaluate, "OFFICIAL_METADATA_SHA256", tuple(evaluate.sha(path) for path in self.metadata))
        self.metadata_patch.start()
        self.addCleanup(self.metadata_patch.stop)

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def test_freeze_exact_thirty_original_cases_and_separate_sampling_seed(self):
        frozen = evaluate.freeze_inputs(self.args)
        self.assertEqual([(case["level"], case["case_index"]) for case in frozen["cases"]],
                         [(level, case) for level in range(3) for case in range(10)])
        self.assertEqual([case["sampling_seed"] for case in frozen["cases"]], list(range(20261105, 20261135)))
        self.assertEqual({case["environment_seed"] for case in frozen["cases"]}, {0})
        self.assertEqual(frozen["cases"][20]["local_environment_config"]["dr_state_dict"], {"level": 2, "case": 0})
        self.assertIn("/home/official/SIMPLE/data", frozen["cases"][0]["source_row"]["environment_config"])
        self.assertEqual(frozen["cases"][0]["local_environment_config"]["asset"], str(self.args.data_root / "asset.usd"))
        self.assertFalse(frozen["formal_evaluation_eligible"])

    def test_refuse_pending_ready_wrong_step_model_or_manifest(self):
        for field, value in (("status", "pending"), ("contract", "old-complete10000"), ("task_step", 10000),
                             ("joint_sha256", "0" * 64), ("checkpoint_save_completed", False),
                             ("training_manifest_sha256", "0" * 64), ("component_load_validation", {"status": "pending"})):
            with self.subTest(field=field):
                ready = dict(self.ready, **{field: value})
                self.write(self.args.deployment / "LOCAL_RUN_READY.json", ready)
                with self.assertRaises(ValueError):
                    evaluate.freeze_inputs(self.args)
        self.write(self.args.deployment / "LOCAL_RUN_READY.json", self.ready)
        (self.args.deployment / "checkpoints/ckpt_6000/model.safetensors").write_bytes(b"different bytes")
        with self.assertRaisesRegex(ValueError, "Actual checkpoint"):
            evaluate.freeze_inputs(self.args)

    def test_refuse_missing_duplicate_or_tampered_official_metadata(self):
        self.metadata[1].write_text(self.metadata[1].read_text().replace('"case": 0', '"case": 99'))
        # Serialized inner JSON uses escapes; use a guaranteed changed source byte.
        self.metadata[1].write_text(self.metadata[1].read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "frozen official metadata"):
            evaluate.freeze_inputs(self.args)
        rows = [json.loads(line) for line in self.metadata[1].read_text().splitlines() if line.strip()]
        rows[-1]["episode_index"] = 0
        self.metadata[1].write_text("\n".join(json.dumps(row) for row in rows))
        with patch.object(evaluate, "OFFICIAL_METADATA_SHA256", tuple(evaluate.sha(path) for path in self.metadata)):
            with self.assertRaisesRegex(ValueError, "exactly the ten"):
                evaluate.freeze_inputs(self.args)

    def test_command_contains_native_budget_all_levels_and_no_teacher_or_fall_gate(self):
        case = evaluate.freeze_inputs(self.args)["cases"][29]
        command = evaluate.commands(self.args, case, self.root / "case")
        simulator = command["simulator"]
        for option, value in (("--dr-level", "2"), ("--episode-start", "9"), ("--seconds", "24"),
                              ("--seed", "0"), ("--expected-checkpoint-step", "6000")):
            self.assertEqual(simulator[simulator.index(option) + 1], value)
        for option in ("--mobile-official-protocol", "--video", "--sync-observations", "--isaac-observation-only"):
            self.assertIn(option, simulator)
        self.assertFalse(any(option in simulator for option in ("--stop-on-fall", "--continue-after-task-success", "--mobile-initialization-manifest")))
        self.assertIn("20261134", command["server"])
        self.assertEqual(simulator[simulator.index("--worker-task") + 1], evaluate.TASK)

    def test_changed_runtime_component_or_unbound_artifact_is_refused(self):
        (self.args.deployment / "motion/joint_motion.ckpt").write_bytes(b"wrong new Motion")
        with self.assertRaisesRegex(ValueError, "Deployment artifact differs"):
            evaluate.freeze_inputs(self.args)
        del self.ready["deployment_artifacts"]["motion/joint_motion.ckpt"]
        self.write(self.args.deployment / "LOCAL_RUN_READY.json", self.ready)
        with self.assertRaisesRegex(ValueError, "seven complete"):
            evaluate.freeze_inputs(self.args)

    def test_runtime_gate_uses_actual_encoder_and_fails_any_contract_difference(self):
        expected = encoder()
        startup = {"weight_sha256": self.args.expected_model_sha256, "pid": 123, "sampling_seed": 44,
                   "Ta": 30, "rtc": False, "inference_steps": 10, "online_dino_runtime": deepcopy(expected)}
        self.assertEqual(evaluate.runtime_gate(startup, expected, self.args.expected_model_sha256, 44, 123)["status"], "passed")
        for key in ("shape", "normalization", "torch_version", "local_file_sha256"):
            changed = deepcopy(startup)
            changed["online_dino_runtime"][key] = "wrong"
            with self.assertRaises(ValueError):
                evaluate.runtime_gate(changed, expected, self.args.expected_model_sha256, 44, 123)
        with self.assertRaisesRegex(ValueError, "server identity"):
            evaluate.runtime_gate(startup, expected, self.args.expected_model_sha256, 44, 999)

    def test_rtc_protocol_has_real_overlap_and_exact_control_clock(self):
        self.args.rtc_mode = "test_time"
        policy = evaluate.evaluation_policy(self.args)
        self.assertEqual((policy["execute_steps"], policy["control_ticks_per_window"]), (15, 25))
        self.assertEqual((policy["rtc_inference_delay"], policy["rtc_min_exec_horizon"]), (0, 15))
        frozen = evaluate.freeze_inputs(self.args)
        self.assertEqual(frozen["evaluation_policy"], policy)
        for command in evaluate.commands(self.args, frozen["cases"][0], self.root / "case").values():
            self.assertEqual(command[command.index("--rtc-mode") + 1], "test_time")
            self.assertEqual(command[command.index("--execute-steps") + 1], "15")
        for steps in (0, 10, 30):
            self.args.execute_steps = steps
            with self.assertRaises(ValueError):
                evaluate.evaluation_policy(self.args)

    def test_rtc_startup_must_match_actual_guidance_and_horizon(self):
        self.args.rtc_mode = "test_time"
        policy = evaluate.evaluation_policy(self.args)
        startup = {"weight_sha256": self.args.expected_model_sha256, "pid": 123, "sampling_seed": 44,
                   "Ta": 15, "Tp": 30, "rtc": True, "inference_steps": 10, "online_dino_runtime": encoder(),
                   "parameter_tensor_count": 12, "trainable_parameter_tensor_count": 0, "parameter_gradient_tensor_count": 0}
        startup.update({k: policy[k] for k in ("rtc_mode", "rtc_exec_shift", "rtc_inference_delay",
                       "rtc_min_exec_horizon", "pig_mask_schedule", "pig_guidance_alpha")})
        evaluate.runtime_gate(startup, encoder(), self.args.expected_model_sha256, 44, 123, policy)
        for key, value in (("rtc_mode", "train"), ("rtc_exec_shift", 10), ("rtc_inference_delay", 15),
                           ("pig_guidance_alpha", 0.), ("Tp", 15), ("rtc", False), ("parameter_gradient_tensor_count", 1)):
            changed = dict(startup, **{key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                evaluate.runtime_gate(changed, encoder(), self.args.expected_model_sha256, 44, 123, policy)

    def rtc_ledger(self):
        policy = evaluate.evaluation_policy(SimpleNamespace(rtc_mode="test_time"))
        info = {"rtc_enabled": True, "rtc_mode": "test_time", "action": {"action_exec_horizon": 15}}
        info.update({k: policy[k] for k in ("rtc_exec_shift", "rtc_inference_delay", "rtc_min_exec_horizon",
                                         "pig_mask_schedule", "pig_guidance_alpha")})
        ticks, requests = [], [{"status": 425, "sampling_path": None, "sampler_calls": {"plain": 0, "test_time": 0, "train": 0},
                               "parameter_gradient_tensor_count": 0}]
        for i in range(51):
            t = 1.005 + i * .02
            start = 1.005 + (i // 25) * .5
            ticks.append({"phase": "policy_latent", "sim_time_s": t, "chunk_start_sim_time_s": start,
                          "action_index_at_30hz": int((t - start) * 30 + 1e-7)})
            if i % 25 == 0:
                path = "plain" if i == 0 else "test_time"
                requests.append({"status": 200, "sim_time_s": str(t), "sampling_path": path,
                    "sampler_calls": {"plain": int(i == 0), "test_time": int(i != 0), "train": 0},
                    "response_action_shape": [15, 80], "parameter_gradient_tensor_count": 0})
        return {"server_info": info}, ticks, requests, policy

    def test_rtc_acceptance_proves_real_calls_response_rows_and_consumed_windows(self):
        result, ticks, requests, policy = self.rtc_ledger()
        accepted = evaluate.validate_rtc_execution(result, ticks, requests, policy)
        self.assertEqual((accepted["plain_requests"], accepted["test_time_requests"]), (1, 2))
        mutations = [lambda r, t, q: q[2].update(sampling_path="plain"),
                     lambda r, t, q: q[2].update(response_action_shape=[30, 80]),
                     lambda r, t, q: q[2].update(sim_time_s="1.345"),
                     lambda r, t, q: q[2].update(sampler_calls={"plain": 1, "test_time": 1}),
                     lambda r, t, q: q[0].update(sampling_path="test_time", sampler_calls={"test_time": 1}),
                     lambda r, t, q: t[25].update(action_index_at_30hz=15),
                     lambda r, t, q: t[26].update(chunk_start_sim_time_s=1.005),
                     lambda r, t, q: q[2].update(parameter_gradient_tensor_count=1),
                     lambda r, t, q: q[0].pop("sampler_calls"),
                     lambda r, t, q: q.pop()]
        for mutate in mutations:
            r, t, q = deepcopy((result, ticks, requests))
            mutate(r, t, q)
            with self.assertRaises(ValueError):
                evaluate.validate_rtc_execution(r, t, q, policy)

    def test_rtc_audit_preserves_native_success_during_warmup_or_first_window(self):
        result, ticks, requests, policy = self.rtc_ledger()
        warmup = evaluate.validate_rtc_execution(result, [], requests[:1], policy)
        self.assertEqual((warmup["plain_requests"], warmup["test_time_requests"]), (0, 0))
        first_window = evaluate.validate_rtc_execution(result, ticks[:1], requests[:2], policy)
        self.assertEqual((first_window["plain_requests"], first_window["test_time_requests"]), (1, 0))

    def test_frozen_model_and_source_change_is_not_silently_accepted(self):
        frozen = evaluate.freeze_inputs(self.args)
        path = Path(frozen["model"]["path"])
        path.write_bytes(b"new")
        with self.assertRaisesRegex(ValueError, "Frozen input changed"):
            evaluate.unchanged(frozen["model"])

    def test_campaign_zero_success_is_complete_but_technical_is_incomplete(self):
        state = {"cases": [{"level": level, "case_index": case, "status": "complete", "native_success": False}
                           for level in range(3) for case in range(10)]}
        self.assertEqual(evaluate.finish(state), 0)
        self.assertEqual(state["native_successes"], 0)
        state["cases"][10].update(status="technical_error", error="retained")
        self.assertEqual(evaluate.finish(state), 1)
        self.assertEqual(state["technically_valid_cases"], 29)
        self.assertEqual(state["levels"][1]["technically_valid_cases"], 9)
        self.assertEqual(state["technical_errors"], 1)
        state["cases"][10] = deepcopy(state["cases"][0])
        self.assertEqual(evaluate.finish(state), 1)  # Thirty rows are not thirty unique fixed cases.

    def test_pid_reuse_is_refused_without_signals(self):
        process = SimpleNamespace(pid=123, owned_identity={"group": 123, "session": 123, "start": 50})
        with patch.object(evaluate, "process_identity", return_value={"start": 51}), patch.object(evaluate.os, "killpg") as kill:
            with self.assertRaisesRegex(ValueError, "reused"):
                evaluate.stop_owned(process)
            kill.assert_not_called()

    def test_pending_preparation_never_creates_campaign_or_launches(self):
        self.write(self.args.deployment / "LOCAL_RUN_READY.json", dict(self.ready, status="pending"))
        argv = []
        for name in ("deployment", "data-root", "training-manifest", "output", "runtime-root"):
            path = getattr(self.args, name.replace("-", "_"))
            if name == "runtime-root":
                path.mkdir()
            argv.extend(["--" + name, str(path)])
        argv.extend(["--expected-model-sha256", self.args.expected_model_sha256])
        with patch.object(evaluate.subprocess, "Popen") as launch:
            with self.assertRaises(ValueError):
                evaluate.main(argv)
            launch.assert_not_called()
        self.assertFalse(self.args.output.exists())

    def evidence_fixture(self, native=False, count=1200, directory=None):
        frozen = evaluate.freeze_inputs(self.args)
        case = frozen["cases"][0]
        directory = directory or self.root / "actual_case"
        epdir = directory / "simulation" / evaluate.TASK / "episode-0"
        epdir.mkdir(parents=True)
        ticks, requests, http = [], [], []
        eid, sid = "one-episode", "one-session"
        reset_history = {"episode_id": eid, "session_id": sid}
        http.append({"endpoint": "/reset", "http_status": 200, "history": reset_history,
                     "simulator_before_s": .005, "simulator_after_s": .005,
                     "reset_response": {"history_frames": 0, "rtc_cleared": True}})
        for index in range(count):
            sim_time = .005 + .02 * index
            warmup = index < 50
            ticks.append({"sim_time_s": sim_time, "next_sim_time_s": .005 + .02 * (index+1),
                          "phase": "warmup_measured_pose_pd" if index < 10 else "warmup_live_pose_sonic_v1_1" if warmup else "policy_latent",
                          "terminated": native and index == count-1, "truncated": not native and index == count-1})
            history = {"episode_id": eid, "session_id": sid, "sim_time_s": sim_time}
            common = {"history": history, "simulator_before_s": sim_time, "simulator_after_s": sim_time}
            http.append(dict(common, endpoint="/observe", http_status=200))
            http.append(dict(common, endpoint="/act", http_status=425 if warmup else 200))
            requests.append({"ordinal": index+1, "episode_id": eid, "session_id": sid,
                "sim_time_s": str(sim_time), "status": 425 if warmup else 200, "rng_before": "same", "rng_after": "same"})
        video = epdir / "rollout.mp4"
        video.write_bytes(b"CPU fixture; ffprobe subprocess is mocked")
        self.write(epdir / "http.json", http)
        (epdir / "trace.jsonl").write_text("\n".join(json.dumps(row) for row in ticks))
        (directory / "server").mkdir(exist_ok=True)
        (directory / "server/requests.jsonl").write_text("\n".join(json.dumps(row) for row in requests))
        if not (directory / "server/startup.json").exists():
            self.write(directory / "server/startup.json", {"CPUfixture": True})
        self.write(directory / "ONLINE_DINO_RUNTIME_ACCEPTANCE.json", {"status": "passed"})
        episode = {"episode_index": 0, "dr_level": 0, "seed": 0, "source_scene_reset_dr_level": None,
            "source_metadata_sha256": case["metadata_sha256"], "local_environment_config": case["local_environment_config"],
            "mobile_protocol": {"effective_success_criteria": .9, "source_metadata_success_criteria": .4,
                                "max_episode_steps": 1200, "reset_dr_level": None},
            "frames": count, "warmup_frames": min(count, 50), "policy_frames": max(0, count-50), "episode_id": eid,
            "status": "task_terminated" if native else "env_truncated", "finite_states": True,
            "diagnostic_loop_executed": count > 50, "decoder_calls": max(0, count-50),
            "simulator_unchanged_during_http": True, "task_success_observed": native, "simulated_seconds": count*.02,
            "policy_chunks": max(0, count-50), "http_425_count": min(count, 50),
            "http_counts": {"/reset": 1, "/observe": count, "/act": count},
            "video": {"path": str(video), "frames": count+1, "ffmpeg_returncode": 0},
            "fell": True, "fell_before_first_policy_command": True}
        result = {"status": "bounded_smoke_finished", "episodes": [episode], "http_request_count": len(http),
            "render_contract": {"render_engine": "Isaac RTX", "authoritative_physics": "MuJoCo",
                                "sync_no_time_advance": True, "isaac_observation_only": True,
                                "environment_source_sha256": frozen["sources"]["SIMPLE/envs/sonic_loco_manip.py"]["sha256"]}}
        self.write(epdir.parent / "result.json", result)
        probe = {"streams": [{"width": 640, "height": 360, "r_frame_rate": "50/1", "nb_read_frames": str(count+1)}]}
        return directory, case, frozen, result, probe

    def test_actual_evidence_full_false_retains_fall_without_extra_quality_gate(self):
        directory, case, frozen, result, probe = self.evidence_fixture()
        with patch.object(evaluate.subprocess, "check_output", return_value=json.dumps(probe).encode()):
            row = evaluate.validate_case(directory, case, frozen)
        self.assertFalse(row["native_success"])
        self.assertTrue(row["fell"])
        self.assertEqual(row["ticks"], 1200)

    def test_native_success_early_including_warmup_is_preserved(self):
        directory, case, frozen, result, probe = self.evidence_fixture(native=True, count=8)
        with patch.object(evaluate.subprocess, "check_output", return_value=json.dumps(probe).encode()):
            self.assertTrue(evaluate.validate_case(directory, case, frozen)["native_success"])

    def test_execute_direct_worker_preserves_early_native_success_returncode(self):
        frozen = evaluate.freeze_inputs(self.args)
        case = frozen["cases"][0]
        self.args.output.mkdir()
        state, launched = {"cases": []}, []
        directory = self.args.output / "level0/case_00"
        probe = {"streams": [{"width": 640, "height": 360, "r_frame_rate": "50/1", "nb_read_frames": "9"}]}

        def popen(command, **kwargs):
            launched.append(command)
            process = SimpleNamespace(pid=120+len(launched), poll=lambda: None, wait=lambda timeout: 0)
            if len(launched) == 1:
                self.write(directory / "server/startup.json", {"weight_sha256": self.args.expected_model_sha256,
                    "pid": process.pid, "sampling_seed": case["sampling_seed"], "Ta": 30, "rtc": False,
                    "inference_steps": 10, "online_dino_runtime": encoder()})
            else:
                self.evidence_fixture(native=True, count=8, directory=directory)
            return process

        response = unittest.mock.MagicMock()
        response.__enter__.return_value.status = 200
        opener = SimpleNamespace(open=lambda *args, **kwargs: response)
        with patch.object(evaluate, "preflight"), patch.object(evaluate.subprocess, "Popen", side_effect=popen), \
                patch.object(evaluate, "mark_owned", side_effect=lambda process: setattr(process, "owned_identity", {"start": 1}) or process), \
                patch.object(evaluate, "stop_owned") as cleanup, patch.object(evaluate, "build_opener", return_value=opener), \
                patch.object(evaluate.subprocess, "check_output", return_value=json.dumps(probe).encode()):
            evaluate.execute(self.args, case, frozen, state)
        self.assertEqual(state["cases"][0]["status"], "complete")
        self.assertTrue(state["cases"][0]["native_success"])
        self.assertEqual(state["cases"][0]["policy_ticks"], 0)
        self.assertIn("--worker-task", launched[1])
        self.assertEqual(cleanup.call_count, 2)

    def test_all_thirty_are_attempted_even_when_one_technical_case_is_retained(self):
        self.args.runtime_root.mkdir()
        argv = []
        for name in ("deployment", "data-root", "training-manifest", "output", "runtime-root"):
            argv.extend(["--" + name, str(getattr(self.args, name.replace("-", "_")))])
        argv.extend(["--expected-model-sha256", self.args.expected_model_sha256])

        def execute(args, case, frozen, state):
            state["cases"].append({"level": case["level"], "case_index": case["case_index"],
                "status": "technical_error" if case["flat_index"] == 0 else "complete", "native_success": False})

        with patch.object(evaluate, "execute", side_effect=execute) as operation:
            self.assertEqual(evaluate.main(argv), 1)
        self.assertEqual(operation.call_count, 30)
        final = evaluate.read(self.args.output / "FINAL_EVALUATION_ACCEPTANCE.json")
        self.assertEqual((final["status"], final["technically_valid_cases"], final["native_successes"]), ("incomplete", 29, 0))

    def test_invalid_video_clock_native_or_http_is_technical(self):
        directory, case, frozen, result, probe = self.evidence_fixture(native=True, count=61)
        with patch.object(evaluate.subprocess, "check_output", return_value=json.dumps(dict(streams=[dict(probe["streams"][0], nb_read_frames="61")])).encode()):
            with self.assertRaisesRegex(ValueError, "Decoded video"):
                evaluate.validate_case(directory, case, frozen)
        trace = directory / "simulation" / evaluate.TASK / "episode-0/trace.jsonl"
        ticks = evaluate.json_lines(trace)
        ticks[-1]["terminated"] = False
        trace.write_text("\n".join(json.dumps(row) for row in ticks))
        with self.assertRaisesRegex(ValueError, "Native success mismatch"):
            evaluate.validate_case(directory, case, frozen)
        ticks[-1]["terminated"] = True
        ticks[20]["next_sim_time_s"] += .001
        trace.write_text("\n".join(json.dumps(row) for row in ticks))
        with self.assertRaisesRegex(ValueError, "Wrong simulation tick"):
            evaluate.validate_case(directory, case, frozen)


if __name__ == "__main__":
    unittest.main()
