"""The pause guard must refuse ambiguous optimizer progress and always release.

Only freshly spawned CPU sleep processes are signaled by these tests.
"""
import copy
import importlib.util
import json
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock
import zipfile

SCRIPT = Path(__file__).resolve().parents[1] / "checkpoint_pause_guard.py"
REPO = SCRIPT.parents[2]
SPEC = importlib.util.spec_from_file_location("checkpoint_pause_guard", SCRIPT)
guard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(guard)


def launched_source(relative):
    return subprocess.check_output(["git", "show", f"{guard.LAUNCH_COMMIT}:{relative}"], cwd=REPO).decode()


def production_policy():
    return guard.code_policy(launched_source(guard.TRAIN_REL), launched_source(guard.POSTTRAIN_REL),
        train_path=REPO / guard.TRAIN_REL, posttrain_path=REPO / guard.POSTTRAIN_REL,
        task_path=REPO / guard.TASK_REL, trainer_path=REPO / guard.TRAINER_REL)


def local(name, value):
    return {"name": name, "repr": str(value)}


def snapshot(policy, *, step=4000, line=None, inner_line=None, pid=12345):
    frames = [{"name": "train", "filename": policy["train_path"],
               "line": policy["batch_fetch_line"] if line is None else line,
               "locals": [local("global_step", step), local("completed_step", 4000)]}]
    if inner_line is not None:
        frames.insert(0, {"name": "_training_step", "filename": policy["posttrain_path"],
                          "line": inner_line, "locals": []})
        frames.insert(1, {"name": "step", "filename": policy["trainer_path"], "line": 300,
                          "locals": [local("global_step", step)]})
    return [{"pid": pid, "thread_id": 1, "frames": frames}]


class StackGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = production_policy()

    def check(self, value):
        return guard.validate_stack(value, pid=12345, checkpoint_step=4000, policy=self.policy)

    def test_real_launch_source_is_bound_to_expected_optimizer_and_context(self):
        self.assertEqual(self.policy["train_step_call_line"], 330)
        self.assertEqual(self.policy["optimizer_call_line"], 853)
        self.assertIn(813, self.policy["ambiguous_context_lines"])
        self.assertNotIn(813, self.policy["pre_optimizer_lines"])

    def test_completed_checkpoint_before_counter_increment_is_accepted(self):
        row = self.check(snapshot(self.policy, step=3999, line=345))
        self.assertEqual(row["global_step"], 3999)
        self.assertIn("completed", row["phase"])

    def test_next_batch_fetch_and_pre_optimizer_work_are_accepted(self):
        self.assertIn("fetching", self.check(snapshot(self.policy))["phase"])
        for line in (818, 823, 828, 830, 852):
            with self.subTest(line=line):
                row = self.check(snapshot(self.policy, line=330, inner_line=line))
                self.assertIn("before_optimizer", row["phase"])

    def test_context_exit_can_follow_optimizer_and_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "may have started"):
            self.check(snapshot(self.policy, line=330, inner_line=813))

    def test_optimizer_line_and_all_later_work_are_rejected(self):
        for line in (853, 854, 856, 859, 870):
            with self.subTest(line=line), self.assertRaises(ValueError):
                self.check(snapshot(self.policy, line=330, inner_line=line))

    def test_counter_alone_never_proves_boundary(self):
        for step, line, inner in ((4001, 321, None), (4000, 335, None),
                                  (3999, 330, 818), (3999, 321, None), (3999, 345, 818)):
            with self.subTest(step=step, line=line), self.assertRaises(ValueError):
                self.check(snapshot(self.policy, step=step, line=line, inner_line=inner))

    def test_wrong_completed_step_is_rejected(self):
        value = snapshot(self.policy, step=3999, line=345)
        value[0]["frames"][0]["locals"][1]["repr"] = "3999"
        with self.assertRaises(ValueError):
            self.check(value)

    def test_optimizer_in_any_thread_or_unknown_step_is_rejected(self):
        for filename, name in (("/env/torch/optim/adamw.py", "_single_tensor_adamw"),
                               ("/env/accelerate/optimizer.py", "step")):
            value = snapshot(self.policy)
            value.append({"pid": 12345, "frames": [{"filename": filename, "name": name}]})
            with self.subTest(filename=filename), self.assertRaises(ValueError):
                self.check(value)

    def test_ambiguous_or_expression_locals_are_never_evaluated(self):
        for value in ("4000.0", "int(4000)", "True", "__import__('os').getpid()", "-1"):
            row = snapshot(self.policy)
            row[0]["frames"][0]["locals"][0]["repr"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.check(row)
        row = snapshot(self.policy)
        row[0]["frames"][0]["locals"].append(local("global_step", 4000))
        with self.assertRaises(ValueError):
            self.check(row)

    def test_wrong_pid_duplicate_train_and_disagreeing_wrapper_are_rejected(self):
        for value in (snapshot(self.policy, pid=54321), [], {}):
            with self.assertRaises(ValueError):
                self.check(value)
        value = snapshot(self.policy)
        value.append(copy.deepcopy(value[0]))
        with self.assertRaises(ValueError):
            self.check(value)
        value = snapshot(self.policy, line=330, inner_line=818)
        value[0]["frames"][1]["locals"][0]["repr"] = "3999"
        with self.assertRaisesRegex(ValueError, "counter disagrees"):
            self.check(value)


def tiny_checkpoint(root):
    root.mkdir()
    header = json.dumps({"a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    (root / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 4)
    for name in ("optimizer.bin", "scheduler.bin"):
        with zipfile.ZipFile(root / name, "w") as archive:
            archive.writestr("archive/data.pkl", b"state")
    for rank in range(8):
        (root / f"random_states_{rank}.pkl").write_bytes(b"rng")
    log = root.parent / "train.log"
    log.write_text(f"Saved state to {root.resolve()}\n")
    return log


class CheckpointAndSourceTests(unittest.TestCase):
    def test_prepared_binding_requires_original_config_and_optimizer(self):
        with tempfile.TemporaryDirectory() as directory:
            prepared = Path(directory)
            state = {"preparation": str(prepared / "preparation.json"), "timestamp": "launch"}
            config = {"train": {"data_parallel": "ddp", "gradient_accumulation_steps": 1,
                "train_batch_size": 4, "optimizer_foreach": False, "resume_from_checkpoint": None},
                "model": {"optim": "adamw_torch"}}
            path = prepared / "run_config.json"
            guard.atomic_json(path, config)
            plan = {"timestamp": "launch", "config_sha256": guard.digest(path), "source_step": 38000,
                    "gpu_count": 8, "effective_global_batch": 32, "checkpointing_steps": 4000,
                    "max_optimizer_steps": 20000}
            guard.atomic_json(Path(state["preparation"]), plan)
            self.assertEqual(guard.prepared_bindings(state)["config_sha256"], guard.digest(path))
            config["train"]["optimizer_foreach"] = True
            guard.atomic_json(path, config)
            with self.assertRaises(ValueError):
                guard.prepared_bindings(state)
            plan["config_sha256"] = guard.digest(path)
            guard.atomic_json(Path(state["preparation"]), plan)
            with self.assertRaisesRegex(ValueError, "ordinary-AdamW"):
                guard.prepared_bindings(state)

    def test_complete_small_checkpoint_and_missing_rank_rng(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "ckpt_4000"
            log = tiny_checkpoint(root)
            self.assertEqual(len(guard.checkpoint_complete(root, log)["payloads"]), 11)
            (root / "random_states_7.pkl").unlink()
            with self.assertRaises(OSError):
                guard.checkpoint_complete(root, log)

    def test_truncated_model_missing_marker_and_incomplete_archive_are_rejected(self):
        for changed in ("model", "marker", "zip"):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "ckpt_4000"
                log = tiny_checkpoint(root)
                if changed == "model":
                    path = root / "model.safetensors"
                    path.write_bytes(path.read_bytes()[:-1])
                elif changed == "marker":
                    log.write_text("Saving checkpoint\n")
                else:
                    (root / "optimizer.bin").write_bytes(b"partial")
                with self.assertRaises((ValueError, zipfile.BadZipFile)):
                    guard.checkpoint_complete(root, log)

    def test_changed_source_is_rejected_before_signaling(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            path = repo / guard.POSTTRAIN_REL
            path.parent.mkdir(parents=True)
            path.write_text("changed")
            receipt = repo / "launch.json"
            guard.atomic_json(receipt, {"git_commit": guard.LAUNCH_COMMIT, "git_status": "",
                "files_sha256": {guard.POSTTRAIN_REL: "0" * 64, guard.TASK_REL: "0" * 64}})
            with mock.patch.object(guard.os, "kill") as kill:
                with self.assertRaisesRegex(ValueError, "differs"):
                    guard.bound_policy(repo, receipt)
                kill.assert_not_called()


class CpuProcessTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.children = []

    def tearDown(self):
        for child in self.children:
            if child.poll() is None:
                os.kill(child.pid, signal.SIGCONT)
                child.terminate()
            child.wait(timeout=5)
        self.directory.cleanup()

    def sleep_process(self):
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.children.append(child)
        return child

    def wait_for(self, predicate, seconds=5):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(.01)
        self.fail("Owned CPU child did not reach the expected state")

    def launch_watchdog(self, target, *, seconds=.25, controller=None):
        identity = guard.proc(target.pid)
        controller = guard.proc(os.getpid()) if controller is None else guard.proc(controller.pid)
        lease = self.root / "lease.json"
        guard.atomic_json(lease, {"active": True, "controller": controller, "ranks": [identity],
                                 "expires_monotonic": time.monotonic() + seconds})
        os.kill(target.pid, signal.SIGSTOP)
        self.wait_for(lambda: guard.all_threads_stopped(target.pid))
        child = subprocess.Popen([sys.executable, str(SCRIPT), "_watchdog", str(lease)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.children.append(child)
        self.wait_for(lambda: (self.root / "watchdog_ready.json").exists())
        return child, lease

    def test_independent_watchdog_resumes_owned_cpu_target_when_lease_expires(self):
        target = self.sleep_process()
        watchdog, _ = self.launch_watchdog(target)
        watchdog.wait(timeout=5)
        self.assertFalse(guard.all_threads_stopped(target.pid))
        result = guard.read_json(self.root / "watchdog_result.json")
        self.assertEqual(result["continued_pids"], [target.pid])
        self.assertEqual(result["continue_errors"], [])

    def test_controller_death_and_corrupt_lease_both_resume_original_owned_target(self):
        for mode in ("controller_death", "corrupt"):
            with self.subTest(mode=mode):
                target, controller = self.sleep_process(), self.sleep_process()
                watchdog, lease = self.launch_watchdog(target, seconds=5, controller=controller)
                if mode == "controller_death":
                    controller.terminate()
                    controller.wait(timeout=5)
                else:
                    lease.write_text("bad json")
                watchdog.wait(timeout=5)
                self.assertFalse(guard.all_threads_stopped(target.pid))
                self.assertEqual(guard.read_json(self.root / "watchdog_result.json")["state"], "auto_continued")
                (self.root / "watchdog_ready.json").unlink()

    def test_stack_rejection_finally_resumes_all_eight_owned_cpu_processes(self):
        targets = [self.sleep_process() for _ in range(8)]
        identities = [{**guard.proc(child.pid), "rank": rank} for rank, child in enumerate(targets)]
        binding = {"ranks": identities}
        state = {"timestamp": "test", "run_directory": str(self.root), "log": str(self.root / "train.log")}
        state_path = self.root / "state.json"
        guard.atomic_json(state_path, state)
        args = SimpleNamespace(output=self.root, hold_seconds=10, stop_timeout=2,
            dump_timeout=2, checkpoint_step=4000, repo=REPO, run_state=state_path)
        def reject(*_):
            self.assertTrue(all(guard.all_threads_stopped(child.pid) for child in targets))
            raise ValueError("synthetic uncertain optimizer state")
        with mock.patch.object(guard, "verify_live_binding", return_value=identities), \
             mock.patch.object(guard, "checkpoint_complete", return_value={}), \
             mock.patch.object(guard, "inspect_paused", side_effect=reject):
            receipt = guard.leased_pause(args, binding, state, {})
        self.assertEqual(receipt["state"], "rejected_resumed")
        self.assertEqual(set(receipt["continued_pids"]), {child.pid for child in targets})
        self.assertTrue(all(not guard.all_threads_stopped(child.pid) for child in targets))
        self.assertFalse(guard.read_json(self.root / "lease.json")["active"])

    def test_reused_pid_identity_never_receives_a_signal(self):
        target = self.sleep_process()
        wrong = {**guard.proc(target.pid), "start_ticks": "0"}
        with mock.patch.object(guard.os, "kill") as kill:
            self.assertEqual(guard.continue_identities([wrong]), [])
            kill.assert_not_called()

    def test_one_signal_error_does_not_skip_other_bound_targets(self):
        identities = [{"pid": 1, "start_ticks": "1"}, {"pid": 2, "start_ticks": "2"}]
        errors = []
        with mock.patch.object(guard, "same_process", return_value=True), \
             mock.patch.object(guard.os, "kill", side_effect=[PermissionError("blocked"), None]) as kill:
            self.assertEqual(guard.continue_identities(identities, errors), [2])
        self.assertEqual(kill.call_count, 2)
        self.assertEqual(errors[0]["pid"], 1)

    def test_previously_stopped_binding_is_rejected_before_watchdog_or_signals(self):
        args = SimpleNamespace(repo=REPO)
        with mock.patch.object(guard, "verify_live_binding", side_effect=ValueError("already stopped")), \
             mock.patch.object(guard.subprocess, "Popen") as launch, mock.patch.object(guard.os, "kill") as kill:
            with self.assertRaisesRegex(ValueError, "already stopped"):
                guard.leased_pause(args, {}, {}, {})
            launch.assert_not_called()
            kill.assert_not_called()

    def test_expired_or_resumed_receipt_cannot_authorize_termination(self):
        targets = [self.sleep_process() for _ in range(8)]
        identities = [{**guard.proc(child.pid), "rank": rank} for rank, child in enumerate(targets)]
        controller = guard.proc(os.getpid())
        watchdog = guard.proc(self.sleep_process().pid)
        lease = self.root / "lease.json"
        receipt = self.root / "receipt.json"
        lease_value = {"active": True, "controller": controller, "ranks": identities,
                       "expires_monotonic": time.monotonic() + 10}
        receipt_value = {"state": "held_verified", "lease_path": str(lease), "boot_id": guard.boot_id(),
                         "controller": controller, "ranks": identities, "watchdog": watchdog,
                         "expires_monotonic": lease_value["expires_monotonic"]}
        guard.atomic_json(lease, lease_value)
        guard.atomic_json(receipt, receipt_value)
        with self.assertRaisesRegex(ValueError, "resumed"):
            guard.validate_held(receipt)
        for child in targets:
            os.kill(child.pid, signal.SIGSTOP)
        self.wait_for(lambda: all(guard.all_threads_stopped(child.pid) for child in targets))
        self.assertEqual(guard.validate_held(receipt)["state"], "held_verified")
        lease_value["expires_monotonic"] = time.monotonic() - 1
        receipt_value["expires_monotonic"] = lease_value["expires_monotonic"]
        guard.atomic_json(lease, lease_value)
        guard.atomic_json(receipt, receipt_value)
        with self.assertRaisesRegex(ValueError, "expired"):
            guard.validate_held(receipt)


if __name__ == "__main__":
    unittest.main()
