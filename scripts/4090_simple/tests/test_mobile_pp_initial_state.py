"""Real CPU MuJoCo first-row fidelity and frozen Mobile split/provenance gates."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import uuid

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mobile_pp_initial_state as runtime


JOINTS = ["j" + str(i) for i in range(43)]
HINGES = "".join(f'<body pos="0 0 .01"><joint name="{n}" range="-2 2"/>'
    '<geom type="sphere" size=".01" mass=".01"/></body>' for n in JOINTS)
XML = ('<mujoco><compiler angle="radian"/><option timestep=".005"/><worldbody>'
    '<body name="pelvis" pos="0 0 1"><freejoint name="root"/>'
    '<geom type="sphere" size=".1" mass="1"/><camera name="head"/>' + HINGES + '</body>'
    '<body name="target" pos="1 0 1"><freejoint/><geom type="box" size=".1 .1 .1" mass=".1"/></body>'
    '<body name="container" pos="2 0 1"><freejoint/><geom type="box" size=".1 .1 .1" mass=".1"/></body>'
    '</worldbody><actuator>' + ''.join(f'<motor joint="{n}"/>' for n in JOINTS) + '</actuator></mujoco>')


def save(path, data):
    path.write_text(json.dumps(data))


class Fixture:
    def __init__(self, root):
        self.root, self.pack = root, root / "packed"
        self.pack.mkdir()
        self.identities = [str(uuid.UUID(int=i + 1)) for i in range(100)]
        self.uid = self.identities[0]
        self.source = self.pack / "evidence" / self.uid / "run"
        (self.source / "deferred_capture").mkdir(parents=True)
        self.original = root / "original" / self.uid
        self.model = mujoco.MjModel.from_xml_string(XML)
        self.data = mujoco.MjData(self.model)
        self.data.time = 2.205
        self.data.qpos[0] = -.3
        self.data.qpos[int(self.model.joint("j42").qposadr[0])] = .25
        self.data.qvel[:] = np.arange(self.model.nv) * .001
        self.data.ctrl[:] = np.arange(self.model.nu) * .002
        self.data.qacc_warmstart[:] = .42
        self.model_path = self.source / "deferred_capture/model.mjb"
        mujoco.mj_saveModel(self.model, str(self.model_path))
        self.state_path = self.source / "deferred_capture/states.npy"
        self.capture_path = self.source / "deferred_capture/capture.json"
        self.start_path = self.source / "capture_start_receipt.json"
        self.audit_path = self.pack / "evidence" / self.uid / "admission.json"
        self.split_path, self.manifest_path = root / "split.json", root / "initialization.json"
        self.capture = {"contract": runtime.CAPTURE_CONTRACT, "requested_completed": True,
            "failure": None, "state_spec": runtime.SPEC, "state_spec_name": "mjSTATE_INTEGRATION",
            "states_dtype": "float64", "states_path": "states.npy", "model_path": "model.mjb",
            "joint_names": JOINTS, "event_count": 2, "state_width": mujoco.mj_stateSize(self.model, runtime.SPEC),
            "metadata": {"human_demo_uuid": self.uid, "task_id": runtime.TASK_ID,
                "source_kind": "live_pico_human", "replay_source_sha256": None,
                "actual_native_success_criteria": .9,
                "environment_config": {"uid": runtime.TASK_UID, "metadata": {
                    "max_episode_steps": 1200, "render_hz": 50, "physics_dt": .005, "split": "train"}},
                "warmup": {"source_reference_tokens_must_not_be_carried": [1, 2, 3]},
                "physics_seed": 20261026, "mujoco_object_body_names": {"target": "target", "container": "container"}}}
        self.start = {"contract": runtime.START_CONTRACT, "human_demo_uuid": self.uid,
            "frozen_before_first_capture": True, "single_use": True, "captured_grid_samples_before_commit": 0,
            "integration_state_width": self.capture["state_width"],
            "integration_state_encoding": "little-endian float64 contiguous bytes"}
        self.audit = {"contract": runtime.ADMISSION_CONTRACT, "task_id": runtime.TASK_ID,
            "human_demo_uuid": self.uid, "selection_policy": "native_only",
            "status": "independent_live_human_episode_audit_passed", "eligible_for_packing": True,
            "technical_blocked": False, "official_task_success": True, "selection_status": "selected",
            "checks": {n: {"passed": True} for n in
                ("independent_native_mobile_task_success", "all_original200hz_external_forces_zero")}}
        self.entries = [{"episode_index": i, "human_demo_uuid": u} for i, u in enumerate(self.identities)]
        self.entries[0].update(original_episode_directory=str(self.original),
            admission_path=str(self.audit_path.relative_to(self.pack)),
            evidence_files={n: str((self.source / n).relative_to(self.pack)) for n in
                ("deferred_capture/capture.json", "deferred_capture/model.mjb", "deferred_capture/states.npy", "capture_start_receipt.json")})
        self.split = {"task_id": runtime.TASK_ID}
        for name, indices in (("train", list(range(95))), ("eval", list(range(95, 100))), ("train_probe", list(range(5)))):
            self.split[name + "_episode_indices"] = indices
            self.split[name + "_uuids"] = [self.identities[i] for i in indices]
        self.manifest = {"contract": runtime.CONTRACT, "task_id": runtime.TASK_ID,
            "source_pack_root": str(self.pack), "episodes": {}}
        for i, source in enumerate([*range(5), *range(95, 100)]):
            self.manifest["episodes"][str(i)] = {"role": "train_probe" if i < 5 else "heldout",
                "source_episode_index": source, "human_demo_uuid": self.identities[source]}
        self.write_states()
        self.seal()

    def write_states(self, *, dtype=np.float64, fortran=False):
        self.state = np.empty(self.capture["state_width"])
        mujoco.mj_getState(self.model, self.data, self.state, runtime.SPEC)
        # Malformed future states must never be read or even validated here.
        values = np.stack([self.state, np.full_like(self.state, np.nan)]).astype(dtype)
        np.save(self.state_path, np.asfortranarray(values) if fortran else values)
        self.capture["start_time"] = float(self.data.time)
        self.start.update(integration_state_sha256_before=hashlib.sha256(self.state.astype("<f8").tobytes()).hexdigest(),
            integration_state_sha256_after=hashlib.sha256(self.state.astype("<f8").tobytes()).hexdigest(),
            sim_time_s_before=float(self.data.time), sim_time_s_after=float(self.data.time))

    def seal(self):
        self.capture["states_sha256"] = runtime._sha(self.state_path)
        self.capture["model_sha256"] = runtime._sha(self.model_path)
        save(self.capture_path, self.capture)
        save(self.start_path, self.start)
        self.audit["bindings"] = {str(self.original / n): runtime._sha(self.source / n)
            for n in self.entries[0]["evidence_files"]}
        save(self.audit_path, self.audit)
        self.entries[0]["admission_sha256"] = runtime._sha(self.audit_path)
        save(self.pack / "PACK_READY.json", {"contract": runtime.PACK_CONTRACT, "task_id": runtime.TASK_ID,
            "selection_policy": "native_only", "episodes": self.entries})
        self.pack_sha = runtime._sha(self.pack / "PACK_READY.json")
        save(self.split_path, self.split)
        self.manifest.update(source_pack_ready_sha256=self.pack_sha,
            split_manifest={"path": str(self.split_path), "sha256": runtime._sha(self.split_path)})
        self.manifest["episodes"]["0"]["states_identity"] = runtime.file_identity(self.state_path)
        save(self.manifest_path, self.manifest)

    def load(self, index=0):
        with mock.patch.object(runtime, "SOURCE_PACK_SHA256", self.pack_sha):
            return runtime.load_initialization(self.manifest_path, index)


def fresh_raw(*, xml=XML, isaac=True):
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    calls = []
    raw = SimpleNamespace(mujoco=SimpleNamespace(mjModel=model, mjData=data,
        mj_objects={n: data.body(n) for n in ("target", "container")}),
        task=SimpleNamespace(uid=runtime.TASK_UID, success_criteria=.9,
            reward=8., _init_target_height=.7, _contact_started=True),
        step_count=0, _success=True, last_reward=3., sim_thread=None,
        sync_mujoco_observations=True, isaac_observation_only=True)
    raw.mujoco.get_robot_qpos = lambda: {n: float(data.joint(n).qpos[0]) for n in JOINTS}
    def refresh(env, *, update_kinematics):
        calls.append(("zero_time_rtx", update_kinematics, float(env.mjData.time)))
    raw.isaac = SimpleNamespace(refresh_from_mujoco=refresh) if isaac else None
    def render():
        calls.append(("new_rgb", float(data.time)))
        return {"head_stereo_left": np.zeros((2, 3, 3), np.uint8)}
    raw._render_frame = render
    raw._get_obs = mock.Mock(side_effect=AssertionError("_get_obs calls mj_forward"))
    raw._get_info = lambda: {"proprio": {"time": float(data.time), "body_dq": data.qvel.copy()}}
    return raw, calls


class MobileInitialStateTests(unittest.TestCase):
    def test_zero_step_exact_complete_state_fresh_observation_and_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(Path(tmp)); init = f.load(); raw, calls = fresh_raw()
            model, data = raw.mujoco.mjModel, raw.mujoco.mjData
            cached = data.joint("j42").qpos
            with mock.patch.object(mujoco, "mj_forward", side_effect=AssertionError("no forward")), \
                    mock.patch.object(mujoco, "mj_step", side_effect=AssertionError("no step")):
                obs, info, receipt = init.restore(raw)
            self.assertIs(raw.mujoco.mjModel, model); self.assertIs(raw.mujoco.mjData, data)
            self.assertTrue(np.shares_memory(cached, data.qpos)); self.assertEqual(cached[0], .25)
            actual = np.empty_like(f.state); mujoco.mj_getState(model, data, actual, runtime.SPEC)
            np.testing.assert_array_equal(actual, f.state)
            self.assertEqual(obs["sim_time_s"], f.data.time)
            self.assertEqual(info["proprio"]["time"], f.data.time)
            self.assertEqual(tuple(obs["joint_names"]), tuple(JOINTS))
            self.assertGreater(np.abs(data.cvel).max(), 0.)
            self.assertEqual([x[0] for x in calls], ["zero_time_rtx", "new_rgb"])
            self.assertFalse(raw._success); self.assertEqual(raw.task.reward, 0.)
            self.assertIsNone(raw.task._init_target_height); self.assertFalse(raw.task._contact_started)
            self.assertEqual(receipt["initial_state_sha256"], receipt["restored_state_sha256"])
            self.assertFalse(receipt["teacher_actions_used"]); self.assertFalse(receipt["future_states_read"])
            self.assertNotIn("warmup", init.capture["metadata"])
            self.assertFalse(init.state.flags.writeable)
            self.assertEqual(receipt["physics_steps"], 0)
            self.assertIn("not certified", receipt["initial_info_boundary"])
            with self.assertRaisesRegex(RuntimeError, "once"): init.restore(raw)
            with self.assertRaisesRegex(RuntimeError, "once"): f.load().restore(raw)

    def test_first_row_only_no_states_hash_no_np_load_and_invalid_future_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(Path(tmp)); original = runtime._sha
            seen, positions = [], []
            original_open = Path.open
            class Tracked:
                def __init__(self, stream): self.stream = stream
                def __enter__(self): return self
                def __exit__(self, *args): return self.stream.__exit__(*args)
                def __getattr__(self, key): return getattr(self.stream, key)
                def read(self, count=-1):
                    value = self.stream.read(count); positions.append(self.stream.tell()); return value
            def tracked_open(path, *args, **kwargs):
                stream = original_open(path, *args, **kwargs)
                return Tracked(stream) if path == f.state_path else stream
            def checked_sha(path):
                seen.append(Path(path).name)
                if Path(path) == f.state_path: raise AssertionError("must not scan future state rows")
                return original(path)
            with mock.patch.object(runtime, "_sha", checked_sha), mock.patch.object(Path, "open", tracked_open), \
                    mock.patch.object(np, "load", side_effect=AssertionError("no np.load")):
                init = f.load()
            np.testing.assert_array_equal(init.state, f.state)
            self.assertNotIn("states.npy", seen)
            with f.state_path.open("rb") as stream:
                version = np.lib.format.read_magic(stream)
                np.lib.format.read_array_header_1_0(stream)
                expected_end = stream.tell() + f.state.size * 8
            self.assertEqual(version, (1, 0)); self.assertEqual(max(positions), expected_end)
            self.assertLess(expected_end, f.state_path.stat().st_size)
            self.assertTrue(np.isnan(np.load(f.state_path)[1]).all())

    def test_nonintegration_wrong_dtype_fortran_and_shape_rejected(self):
        for change in ("spec", "float32", "fortran", "width"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                f = Fixture(Path(tmp))
                if change == "spec": f.capture["state_spec"] = 1
                elif change == "width": f.capture["state_width"] += 1
                else: f.write_states(dtype=np.float32 if change == "float32" else np.float64, fortran=change == "fortran")
                f.seal()
                with self.assertRaises(ValueError): f.load()

    def test_uuid_task_native_admission_and_start_sha_are_required(self):
        for change in ("uuid", "task", "blocked", "native", "forceproof", "startsha", "starttime"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                f = Fixture(Path(tmp))
                if change == "uuid": f.capture["metadata"]["human_demo_uuid"] = str(uuid.UUID(int=200))
                elif change == "task": f.capture["metadata"]["task_id"] = "handover"
                elif change == "blocked": f.audit["technical_blocked"] = True
                elif change == "native": f.audit["official_task_success"] = False
                elif change == "forceproof": f.audit["checks"]["all_original200hz_external_forces_zero"]["passed"] = False
                elif change == "startsha": f.start.update(integration_state_sha256_before="f" * 64, integration_state_sha256_after="f" * 64)
                else: f.start.update(sim_time_s_before=0., sim_time_s_after=0.)
                f.seal()
                with self.assertRaises(ValueError): f.load()

    def test_split_overlap_role_order_and_foreign_uuid_refused(self):
        for change in ("overlap", "role", "order", "uuid"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                f = Fixture(Path(tmp))
                if change == "overlap":
                    f.split["eval_episode_indices"][0] = 0; f.split["eval_uuids"][0] = f.identities[0]
                elif change == "role": f.manifest["episodes"]["0"]["role"] = "heldout"
                elif change == "order": f.manifest["episodes"]["0"]["source_episode_index"] = 1
                else: f.split["train_probe_uuids"][0] = "foreign"
                f.seal()
                with self.assertRaises(ValueError): f.load()

    def test_states_identity_capture_start_or_model_tampering_rejected(self):
        for path_name in ("state_path", "capture_path", "start_path", "model_path"):
            with self.subTest(path=path_name), tempfile.TemporaryDirectory() as tmp:
                f = Fixture(Path(tmp)); path = getattr(f, path_name)
                with path.open("ab") as stream: stream.write(b"X")
                with self.assertRaises((ValueError, json.JSONDecodeError)): f.load()

    def test_initial_external_force_refused_without_clearing_or_replaying(self):
        for name in ("qfrc_applied", "xfrc_applied"):
            with self.subTest(force=name), tempfile.TemporaryDirectory() as tmp:
                f = Fixture(Path(tmp)); getattr(f.data, name).flat[0] = .01
                f.write_states(); f.seal()
                with self.assertRaisesRegex(ValueError, "external forces"): f.load()

    def test_live_physics_actor_joint_membership_task_and_clock_rejected_before_assignment(self):
        for change in ("mass", "actor", "jointmembership", "jointvalue", "task", "threshold", "already_control", "background", "rtx_steps"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                f = Fixture(Path(tmp)); init = f.load(); raw, _ = fresh_raw()
                if change == "mass": raw.mujoco.mjModel.body_mass[1] += .1
                elif change == "actor": raw.mujoco.mj_objects["target"] = raw.mujoco.mjData.body("container")
                elif change == "jointmembership": raw.mujoco.get_robot_qpos = lambda: {**{n: 0. for n in JOINTS[:-1]}, "foreign": 0.}
                elif change == "jointvalue": raw.mujoco.get_robot_qpos = lambda: {n: .125 for n in JOINTS}
                elif change == "task": raw.task.uid = "handover"
                elif change == "threshold": raw.task.success_criteria = .4
                elif change == "already_control": raw.step_count = 1
                elif change == "background": raw.sim_thread = object()
                else: raw.isaac_observation_only = False
                before = raw.mujoco.mjData.qpos.copy()
                with self.assertRaises(ValueError): init.restore(raw)
                np.testing.assert_array_equal(raw.mujoco.mjData.qpos, before)
                self.assertFalse(init.applied)

    def test_observation_and_canonical_capture_orders_use_identical_named_addresses(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(Path(tmp)); init = f.load(); raw, _ = fresh_raw()
            actual_order = tuple(reversed(JOINTS))
            raw.mujoco.get_robot_qpos = lambda: {n: float(raw.mujoco.mjData.joint(n).qpos[0]) for n in actual_order}
            obs, _, receipt = init.restore(raw)
            self.assertEqual(tuple(obs["joint_names"]), actual_order)
            self.assertEqual(obs["joint_qpos"][0], .25)
            self.assertEqual(receipt["source_capture_joint_names"], list(JOINTS))
            self.assertEqual(receipt["source_to_observation_indices"], list(reversed(range(43))))
            self.assertTrue(receipt["named_joint_addresses_equal"])
            self.assertEqual(receipt["initial_state_sha256"], receipt["restored_state_sha256"])

    def test_observation_mutation_or_history_time_override_fails_and_cannot_retry(self):
        for change in ("state", "clock"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                f = Fixture(Path(tmp)); init = f.load(); raw, _ = fresh_raw()
                if change == "state":
                    def render(): raw.mujoco.mjData.time += .005; return {}
                else:
                    def render(): return {"sim_time_s": 0.}
                raw._render_frame = render
                with self.assertRaisesRegex(ValueError, "integration state|clock"): init.restore(raw)
                with self.assertRaisesRegex(RuntimeError, "once"): init.restore(raw)

    def test_pack_split_binding_and_unsafe_source_paths_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(Path(tmp))
            with mock.patch.object(runtime, "SOURCE_PACK_SHA256", "0" * 64):
                with self.assertRaisesRegex(ValueError, "PACK_READY"): runtime.load_initialization(f.manifest_path, 0)
            f.split_path.write_text(f.split_path.read_text() + " ")
            with self.assertRaisesRegex(ValueError, "split manifest"): f.load()
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(Path(tmp)); alias = f.pack / "state_alias.npy"; alias.symlink_to(f.state_path)
            f.entries[0]["evidence_files"]["deferred_capture/states.npy"] = "state_alias.npy"
            f.seal()
            with self.assertRaisesRegex(ValueError, "Symlink"): f.load()

    def test_matched_cli_requires_explicit_pair_and_preserves_old_protocol_isolation(self):
        from run_closed_loop import MOBILE_PP_TASK, parse_args
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(Path(tmp))
            base = ["run_closed_loop.py", "--data-root", tmp, "--sonic-resources", tmp,
                "--output", tmp + "/out", "--tasks", MOBILE_PP_TASK, "--seconds", "24",
                "--sync-observations", "--sim-mode", "mujoco_isaac", "--isaac-observation-only"]
            pair = ["--mobile-initialization-manifest", str(f.manifest_path), "--mobile-initialization-case", "0",
                "--expected-mobile-split-sha256", runtime._sha(f.split_path)]
            with mock.patch.object(sys, "argv", base + pair):
                args = parse_args()
            self.assertEqual(args.mobile_initialization_manifest, f.manifest_path)
            self.assertFalse(args.mobile_official_protocol)
            for extra in (["--mobile-official-protocol"], ["--stop-on-fall"], ["--episodes-per-task", "2"],
                    ["--handover-initialization", str(f.manifest_path)],
                    ["--initial-physics-run", str(f.source)], ["--sim-mode", "mujoco"],
                    ["--worker-task", "G1WholebodyHandoverTeleop-v0"]):
                with self.subTest(extra=extra), mock.patch.object(sys, "argv", base + pair + extra), mock.patch("sys.stderr"):
                    with self.assertRaises(SystemExit): parse_args()
            for single in (pair[:2], pair[2:4], pair[:4], pair[4:], pair[:4] + ["--expected-mobile-split-sha256", "bad"]):
                with mock.patch.object(sys, "argv", base + single), mock.patch("sys.stderr"):
                    with self.assertRaises(SystemExit): parse_args()

    def test_matched_reset_wiring_restores_before_server_history_and_saves_receipt_first(self):
        from run_closed_loop import bind_mobile_training_split, mobile_matched_protocol, reset_mobile_matched_episode
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(Path(tmp)); init = f.load(); raw, calls = fresh_raw()
            with self.assertRaisesRegex(ValueError, "actual model training split"):
                bind_mobile_training_split(init, "0" * 64)
            expected_split = runtime._sha(f.split_path)
            bind_mobile_training_split(init, expected_split)
            receipt_path = Path(tmp) / "actual_initialization.json"
            def reset(**kwargs):
                calls.append(("environment_reset", kwargs))
                raw.mujoco.mjData.time = .005
                return {"discard_this_initial_observation": True}, {}
            env = SimpleNamespace(reset=reset, unwrapped=raw)
            def agent_reset(*, episode_id):
                self.assertTrue(receipt_path.is_file())
                self.assertEqual(raw.mujoco.mjData.time, f.data.time)
                calls.append(("new_server_history", episode_id))
            agent = SimpleNamespace(reset=agent_reset)
            obs, _, receipt = reset_mobile_matched_episode(env, init, agent, "new-episode", receipt_path)
            self.assertEqual([x[0] for x in calls], ["environment_reset", "zero_time_rtx", "new_rgb", "new_server_history"])
            self.assertEqual(obs["sim_time_s"], receipt["source_sim_time_s"])
            self.assertEqual(receipt["expected_mobile_split_sha256"], expected_split)
            self.assertNotIn("discard_this_initial_observation", obs)
            options = calls[0][1]
            self.assertEqual(options["seed"], init.seed); self.assertIsNone(options["options"]["dr_level"])
            protocol = mobile_matched_protocol(init)
            self.assertEqual(protocol["effective_success_criteria"], .9)
            self.assertEqual(protocol["max_episode_steps"], 1200)
            self.assertEqual(protocol["experiment_role"], "train_probe")
            self.assertFalse(protocol["formal_evaluation_eligible"])
            self.assertIn("not an official", protocol["boundary"])


class MobileMeshRelocationTests(unittest.TestCase):
    def models(self, root, *, relative=False):
        source_dir, live_dir = root / "source_assets_with_long_path", root / "live"
        source_dir.mkdir(); live_dir.mkdir()
        obj = "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n"
        for folder in (source_dir, live_dir):
            (folder / "part.obj").write_text(obj)
            (folder / "relative.obj").write_text(obj)
        def compile_model(folder):
            assets = f'<mesh name="shape" file="{folder / "part.obj"}"/>'
            geoms = '<geom type="mesh" mesh="shape" mass="1"/>'
            if relative:
                assets += '<mesh name="relative" file="relative.obj"/>'
                geoms += '<geom type="mesh" mesh="relative" mass=".2" pos="2 0 0"/>'
            return mujoco.MjModel.from_xml_string(f'<mujoco><compiler meshdir="{folder}"/>'
                f'<asset>{assets}</asset><worldbody><body name="root"><freejoint/>{geoms}</body></worldbody></mujoco>')
        return compile_model(source_dir), compile_model(live_dir), source_dir, live_dir

    def test_only_verified_mesh_paths_and_aligned_buffer_storage_can_differ(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, live, left, right = self.models(Path(tmp), relative=True)
            receipt = runtime.compare_mobile_models(source, live, source_mesh_root=left, live_mesh_root=right)
            self.assertTrue(receipt["compatible"]); self.assertTrue(receipt["path_relocation_used"])
            self.assertEqual(receipt["source_normalized_sha256"], receipt["target_normalized_sha256"])
            self.assertEqual(receipt["file_backed_meshes_verified"], 2)
            proof = receipt["buffer_path_storage_proof"]
            self.assertEqual(proof["expected_buffer_delta"], live.nbuffer - source.nbuffer)
            self.assertLessEqual(set(receipt["different_fields"]), runtime.PATH_FIELDS)
            for mesh in receipt["mesh_resource_identities"]:
                self.assertEqual(mesh["source"]["sha256"], mesh["live"]["sha256"])
                self.assertTrue(Path(mesh["source"]["resolved_path"]).is_file())
            with self.assertRaisesRegex(ValueError, "compiler mesh root"):
                runtime.compare_mobile_models(source, live)
            # Same embedded geometry is insufficient: actual file bytes must
            # agree even when the differing bytes are only a harmless comment.
            with (right / "relative.obj").open("a") as stream: stream.write("# changed bytes\n")
            with self.assertRaisesRegex(ValueError, "Mesh resource bytes differ"):
                runtime.compare_mobile_models(source, live, source_mesh_root=left, live_mesh_root=right)

    def test_geometry_physics_and_resource_order_are_never_normalized(self):
        for change in ("geometry", "mass", "pathorder"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                source, live, left, right = self.models(Path(tmp), relative=True)
                if change == "geometry": live.mesh_vert[0, 0] += .001
                elif change == "mass": live.body_mass[1] += .01
                else: live.mesh_pathadr[:] = live.mesh_pathadr[::-1]
                with self.assertRaisesRegex(ValueError, "physics/geometry|path table"):
                    runtime.compare_mobile_models(source, live, source_mesh_root=left, live_mesh_root=right)

    def test_unexplained_buffer_delta_or_unknown_runtime_fails_closed(self):
        proof = runtime.path_buffer_proof(9721, 9163, 135463404, 135462892)
        self.assertEqual(proof["expected_buffer_delta"], -512)
        with self.assertRaisesRegex(ValueError, "not exclusively"):
            runtime.path_buffer_proof(9721, 9163, 135463404, 135462891)
        with mock.patch.object(mujoco, "__version__", "future"):
            with self.assertRaisesRegex(ValueError, "verified MuJoCo"):
                runtime.path_buffer_proof(9721, 9163, 135463404, 135462892)


if __name__ == "__main__":
    unittest.main()
