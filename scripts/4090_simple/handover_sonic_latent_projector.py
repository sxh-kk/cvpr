"""Optional CPU latent residual experiment through the unchanged SONIC decoder.

Only a returned token may be executed. The projector never steps physics,
overrides joints/torques/gains, appends controller history, or certifies task
success. Its objective concerns decoder joint targets, not measured next-step
wrist motion. It is deliberately independent of the current expert.
"""
from __future__ import annotations

import argparse
from collections import deque
from copy import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[2]
CONTRACT = "cvpr-handover-sonic-latent-residual-v1"
BATCH_CONTRACT = "cvpr-sonic-derived-metadata-only-jacobian-view-v1"
OFFICIAL_DECODER_SHA = "34bae8570d4a4421a5391a5c2befd745d4a02d182ec539e5f9da44c091c67509"


def require(value, message):
    if not value:
        raise ValueError(message)


def sonic_module():
    path = REPO / "VLMExpert/third_party/SIMPLE/src"
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    from simple.baselines import sonic_v1_1
    return sonic_v1_1


def finite(value, shape, name):
    array = np.asarray(value, dtype=np.float64)
    require(array.shape == shape and np.isfinite(array).all(), f"Require finite {name}{shape}")
    return array.copy()


@dataclass(frozen=True)
class ProjectionConfig:
    iterations: int = 2
    finite_difference_step: float = .005
    damping: float = .01
    protected_nullspace_rtol: float = 1e-5
    max_latent_per_coordinate: float = .08
    max_latent_l2: float = .25
    max_step_l2: float = .15
    max_protected_target_delta_rad: float = .002
    max_arm_target_delta_rad: float = .20
    minimum_arm_rms_improvement_rad: float = 1e-7
    line_search_steps: int = 8
    latent_min: float = -.96875
    latent_max: float = .96875
    control_scope: str = "arms"

    def __post_init__(self):
        require(self.control_scope in ("arms", "waist_arms"), "Use explicit arms or waist_arms control scope")
        require(type(self.iterations) is int and 1 <= self.iterations <= 3
                and type(self.line_search_steps) is int and 1 <= self.line_search_steps <= 12,
                "Use bounded iteration/line-search budgets")
        for key, value in asdict(self).items():
            if key not in ("control_scope", "iterations", "line_search_steps", "latent_min", "latent_max"):
                require(type(value) in (int, float) and math.isfinite(value) and value > 0,
                        f"Positive finite projector setting required: {key}")
        require(math.isfinite(self.latent_min) and math.isfinite(self.latent_max)
                and -.96875 <= self.latent_min < self.latent_max <= .96875, "Invalid official latent numeric range")
        require(self.max_protected_target_delta_rad <= .002, "Protected decoder target changes cannot exceed0.002rad")

    @property
    def protected_joint_count(self):
        return 15 if self.control_scope == "arms" else 12

    @classmethod
    def waist_arms_experiment(cls):
        """Optional larger bounded residual; no claim of physical reachability."""
        return cls(control_scope="waist_arms", iterations=3, max_latent_per_coordinate=.16,
                   max_latent_l2=.5, max_step_l2=.2)


def optimize_latent(token, desired_q29, predict_targets, config=ProjectionConfig(), *, predict_targets_batch=None):
    """Bounded DLS in the selected protected-joint Jacobian nullspace.

    ``predict_targets`` is stateless and returns BODY_JOINT_NAMES order. Each
    accepted step must improve the true nonlinear controlled objective and obey all
    global bounds relative to the original decoder output/token.
    """
    original = finite(token, (64,), "token64")
    desired = finite(desired_q29, (29,), "desired body targets")
    require(np.all(original >= config.latent_min) and np.all(original <= config.latent_max),
            "Original encoder token lies outside the configured official numeric range")
    evaluations = 0
    batch_evaluations = 0

    def predict(z):
        nonlocal evaluations
        evaluations += 1
        return finite(predict_targets(np.asarray(z, np.float32)), (29,), "official decoded body targets")

    baseline = predict(original)
    z, output = original.copy(), baseline.copy()
    protected_count = config.protected_joint_count
    controlled_count = 29 - protected_count
    rms = lambda output, first, stop=29: float(np.sqrt(np.mean((output[first:stop] - desired[first:stop]) ** 2)))
    initial_error = rms(baseline, protected_count)
    iterations = []
    for iteration in range(config.iterations):
        trials = np.repeat(z[None], 64, axis=0)
        steps = np.empty(64, np.float64)
        for dimension in range(64):
            step = min(config.finite_difference_step, config.latent_max - z[dimension])
            if step < config.finite_difference_step / 2:
                step = -min(config.finite_difference_step, z[dimension] - config.latent_min)
            require(abs(step) > 0, "No finite-difference direction is available")
            trials[dimension, dimension] += step
            steps[dimension] = step
        if predict_targets_batch is None:
            decoded_trials = np.stack([predict(trial) for trial in trials])
        else:
            batch_evaluations += 1
            decoded_trials = finite(predict_targets_batch(trials.astype(np.float32)), (64, 29),
                                    "derived batch finite-difference targets")
        jacobian = ((decoded_trials - output) / steps[:, None]).T
        protected = jacobian[:protected_count]
        _, singular, right = np.linalg.svd(protected, full_matrices=True)
        rank = int(np.count_nonzero(singular > config.protected_nullspace_rtol * max(float(singular[0]), 1e-12)))
        nullspace = right[rank:].T
        arm_map = jacobian[protected_count:] @ nullspace
        residual = desired[protected_count:] - output[protected_count:]
        delta = nullspace @ arm_map.T @ np.linalg.solve(
            arm_map @ arm_map.T + config.damping ** 2 * np.eye(controlled_count), residual)
        delta *= min(1., config.max_step_l2 / max(float(np.linalg.norm(delta)), 1e-15))
        old_error = rms(output, protected_count)
        accepted = False
        for backtrack in range(config.line_search_steps):
            candidate = z + delta * (.5 ** backtrack)
            candidate = np.clip(candidate, np.maximum(original - config.max_latent_per_coordinate, config.latent_min),
                                np.minimum(original + config.max_latent_per_coordinate, config.latent_max))
            change = candidate - original
            candidate = original + change * min(1., config.max_latent_l2 / max(float(np.linalg.norm(change)), 1e-15))
            # Validate the float32 command that will actually be returned.
            candidate = candidate.astype(np.float32).astype(np.float64)
            decoded = predict(candidate)
            new_error = rms(decoded, protected_count)
            protected_change = float(np.max(np.abs(decoded[:protected_count] - baseline[:protected_count])))
            arm_change = float(np.max(np.abs(decoded[protected_count:] - baseline[protected_count:])))
            if (new_error < old_error - config.minimum_arm_rms_improvement_rad
                    and protected_change <= config.max_protected_target_delta_rad
                    and arm_change <= config.max_arm_target_delta_rad
                    and np.max(np.abs(candidate - original)) <= config.max_latent_per_coordinate + 1e-7
                    and np.linalg.norm(candidate - original) <= config.max_latent_l2 + 1e-7):
                old_arm_error, new_arm_error = rms(output, 15), rms(decoded, 15)
                z, output, accepted = candidate, decoded, True
                iterations.append({"iteration": iteration, "protected_jacobian_rank": rank,
                    "line_search_halvings": backtrack, "controlled_rms_before_rad": old_error,
                    "controlled_rms_after_rad": new_error, "arm_rms_before_rad": old_arm_error,
                    "arm_rms_after_rad": new_arm_error, "max_protected_target_change_rad": protected_change})
                break
        if not accepted:
            iterations.append({"iteration": iteration, "protected_jacobian_rank": rank,
                               "accepted": False, "reason": "no_nonlinear_improvement_within_all_hard_bounds"})
            break
    final_error = rms(output, protected_count)
    applied = bool(final_error < initial_error - config.minimum_arm_rms_improvement_rad)
    result = z.astype(np.float32) if applied else original.astype(np.float32)
    return result, {"applied": applied, "decoder_evaluations": evaluations,
        "derived_batch_evaluations": batch_evaluations,
        "control_scope": config.control_scope, "protected_joint_count": protected_count,
        "controlled_joint_count": controlled_count,
        "controlled_rms_before_rad": initial_error, "controlled_rms_after_rad": final_error,
        "arm_rms_before_rad": rms(baseline, 15), "arm_rms_after_rad": rms(output, 15),
        "waist_rms_before_rad": rms(baseline, 12, 15), "waist_rms_after_rad": rms(output, 12, 15),
        "max_protected_target_change_rad": float(np.max(np.abs(output[:protected_count] - baseline[:protected_count]))),
        "max_leg_target_change_rad": float(np.max(np.abs(output[:12] - baseline[:12]))),
        "max_waist_target_change_rad": float(np.max(np.abs(output[12:15] - baseline[12:15]))),
        "max_arm_target_change_rad": float(np.max(np.abs(output[15:] - baseline[15:]))),
        "latent_residual_linf": float(np.max(np.abs(result.astype(float) - original))),
        "latent_residual_l2": float(np.linalg.norm(result.astype(float) - original)),
        "original_decoder_targets_q29": baseline.tolist(), "projected_decoder_targets_q29": output.tolist(),
        "desired_targets_q29": desired.tolist(), "iterations": iterations}


def build_batch_view(official_path, derived_path):
    """Use installed ONNX tooling to edit only input/output dim0 metadata.

    This derived artifact is exclusively a finite-difference accelerator. No
    official file, initializer, graph operation, attribute or constant is edited.
    Run this optional builder in an environment which already contains ONNX.
    """
    import onnx
    official, derived = Path(official_path).resolve(strict=True), Path(derived_path).resolve()
    receipt_path = derived.with_suffix(derived.suffix + ".json")
    require(not derived.exists() and not receipt_path.exists() and not derived.is_relative_to(REPO)
            and not derived.is_relative_to(official.parent), "Use fresh external derived artifacts")
    original_bytes = official.read_bytes()
    require(hashlib.sha256(original_bytes).hexdigest() == OFFICIAL_DECODER_SHA, "Official decoder SHA mismatch")
    original = onnx.load_from_string(original_bytes)
    modified = onnx.ModelProto(); modified.CopyFrom(original)
    changed = []
    for value in [*modified.graph.input, *modified.graph.output]:
        dims = value.type.tensor_type.shape.dim
        require(len(dims) == 2 and dims[0].dim_value == 1, "Unexpected official batch metadata")
        dims[0].ClearField("dim_value"); dims[0].dim_param = "jacobian_batch"
        changed.append(value.name)
    # Reverse only those edits and compare the complete model protobuf; stronger
    # than checking weights and op names alone.
    restored = onnx.ModelProto(); restored.CopyFrom(modified)
    for value in [*restored.graph.input, *restored.graph.output]:
        value.type.tensor_type.shape.dim[0].ClearField("dim_param")
        value.type.tensor_type.shape.dim[0].dim_value = 1
    require(restored.SerializeToString() == original.SerializeToString(), "A non-batch model field changed")
    def digest_parts(values):
        h = hashlib.sha256()
        for value in values:
            payload = value.SerializeToString(); h.update(len(payload).to_bytes(8, "big")); h.update(payload)
        return h.hexdigest()
    onnx.checker.check_model(modified)
    payload = modified.SerializeToString()
    receipt = {"contract": BATCH_CONTRACT, "official_decoder_sha256": OFFICIAL_DECODER_SHA,
        "derived_view_sha256": hashlib.sha256(payload).hexdigest(),
        "only_input_output_dim0_changed": True, "changed_value_names": changed,
        "initializer_sha256": digest_parts(original.graph.initializer),
        "complete_nodes_with_attributes_sha256": digest_parts(original.graph.node),
        "restored_complete_model_equal": True, "official_execution_claim": False,
        "purpose": "Jacobian finite differences only; all candidate acceptance uses original official B1",
        "builder_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    derived.parent.mkdir(parents=True, exist_ok=True)
    with derived.open("xb") as stream: stream.write(payload)
    with receipt_path.open("x") as stream: stream.write(json.dumps(receipt, indent=2) + "\n")
    require(official.read_bytes() == original_bytes, "Official source changed during derived build")
    return receipt


class DerivedBatchJacobianView:
    """Metadata-only derived ONNX, never used for actual command acceptance."""
    def __init__(self, path):
        import onnxruntime as ort
        self.path = Path(path).resolve(strict=True)
        self.receipt = json.loads(self.path.with_suffix(self.path.suffix + ".json").read_text())
        r = self.receipt
        require(r.get("contract") == BATCH_CONTRACT and r.get("official_decoder_sha256") == OFFICIAL_DECODER_SHA
                and r.get("only_input_output_dim0_changed") is True
                and r.get("restored_complete_model_equal") is True
                and r.get("official_execution_claim") is False
                and r.get("derived_view_sha256") == hashlib.sha256(self.path.read_bytes()).hexdigest(),
                "Derived Jacobian artifact/receipt identity mismatch")
        options = ort.SessionOptions(); options.intra_op_num_threads = 1; options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(self.path), sess_options=options, providers=["CPUExecutionProvider"])
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        require(len(inputs) == len(outputs) == 1 and inputs[0].shape == ["jacobian_batch", 994]
                and outputs[0].shape == ["jacobian_batch", 29], "Unexpected derived-view IO")
        self.input_name, self.output_name = inputs[0].name, outputs[0].name

    def predict_actions(self, actor_inputs):
        actor_inputs = finite(actor_inputs, (64, 994), "derived batch actor inputs").astype(np.float32)
        return finite(self.session.run([self.output_name], {self.input_name: actor_inputs})[0],
                      (64, 29), "derived batch decoder actions")


def controller_fingerprint(controller):
    digest = hashlib.sha256()
    for row in controller._history:
        for term in row:
            array = np.asarray(term)
            digest.update(str((array.shape, str(array.dtype))).encode()); digest.update(array.tobytes())
    for name in ("_last_tick", "_last_time", "_last_raw_action", "_body_target", "decoder_calls",
                 "clipped_torque_elements", "body_kp", "body_kd"):
        value = getattr(controller, name, None)
        digest.update(name.encode())
        if value is None: digest.update(b"None")
        else:
            array = np.asarray(value); digest.update(str((array.shape, str(array.dtype))).encode()); digest.update(array.tobytes())
    digest.update(json.dumps(controller.identity, sort_keys=True, allow_nan=False).encode())
    return digest.hexdigest()


def live_fingerprint(model, data):
    import mujoco
    spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
    state = np.empty(mujoco.mj_stateSize(model, spec)); mujoco.mj_getState(model, data, state, spec)
    digest = hashlib.sha256(state.tobytes())
    for value in (data.qacc, data.xmat, data.cvel, data.sensordata, model.body_mass,
                  model.body_inertia, model.actuator_ctrlrange, model.opt.gravity, model.qpos0):
        digest.update(np.asarray(value).tobytes())
    return digest.hexdigest()


class SonicLatentProjector:
    def __init__(self, config=ProjectionConfig(), *, jacobian_batch_view=None, capture_batch_audit=False):
        self.config = config
        self._model = self._scratch = None
        self.batch_view = DerivedBatchJacobianView(jacobian_batch_view) if jacobian_batch_view else None
        self.capture_batch_audit = capture_batch_audit
        self.batch_audit_inputs = []

    def project(self, token, desired_q29, raw, controller):
        import mujoco
        sonic = sonic_module()
        token = finite(token, (64,), "token64").astype(np.float32)
        desired = finite(desired_q29, (29,), "desired body targets")
        model, data = raw.mujoco.mjModel, raw.mujoco.mjData
        baseline = {"contract": CONTRACT, "applied": False, "physical_success_claim": False,
            "joints_torques_gains_overridden": False, "history_padded_or_appended": False,
            "configuration": asdict(self.config), "decoder_evaluations": 0}
        require(controller.identity.get("decoder_sha256") == sonic.OFFICIAL_DECODER_SHA256
                and controller.sync_mujoco_observations is True
                and controller.session.get_providers() == ["CPUExecutionProvider"],
                "Require the unchanged official CPU decoder and synchronized real proprioception")
        t, previous = float(data.time), controller._last_tick
        history = list(controller._history)
        if previous is None or len(history) < 9 or controller._body_target is None:
            return token.copy(), {**baseline, "reason": "insufficient_real_controller_history"}
        if (not math.isclose(t - previous, .02, rel_tol=0, abs_tol=1e-6)
                or not math.isclose(float(history[-1][0]), previous, rel_tol=0, abs_tol=1e-8)):
            return token.copy(), {**baseline, "reason": "not_the_exact_next_50Hz_decoder_tick", "actual_time_s": t,
                                  "last_controller_tick_s": previous}
        selected = history[-9:]
        if not np.allclose(np.diff([float(row[0]) for row in selected]), .02, rtol=0, atol=1e-6):
            return token.copy(), {**baseline, "reason": "noncontiguous_real_controller_history"}
        require(0 < float(model.opt.timestep) <= .02
                and math.isclose(.02 / model.opt.timestep, round(.02 / model.opt.timestep), abs_tol=1e-6),
                "Physics timestep must divide50Hz exactly")
        before_live, before_controller = live_fingerprint(model, data), controller_fingerprint(controller)
        if self._model is not model:
            self._model, self._scratch = model, mujoco.MjData(model)
        spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
        state = np.empty(mujoco.mj_stateSize(model, spec)); mujoco.mj_getState(model, data, state, spec)
        mujoco.mj_setState(model, self._scratch, state, spec)
        mujoco.mj_forward(model, self._scratch)
        body_q = np.array([self._scratch.joint(name).qpos[0] for name in sonic.BODY_JOINT_NAMES])
        body_dq = np.array([self._scratch.joint(name).qvel[0] for name in sonic.BODY_JOINT_NAMES])
        pelvis = model.body("pelvis").id
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(model, self._scratch, mujoco.mjtObj.mjOBJ_BODY, pelvis, velocity, 1)
        gravity = self._scratch.xmat[pelvis].reshape(3, 3).T @ np.array([0., 0., -1.])
        current = (t, velocity[:3].copy(), (body_q - sonic.DEFAULT_ANGLES)[sonic.POLICY_TO_BODY],
                   body_dq[sonic.POLICY_TO_BODY], np.array(controller._last_raw_action, copy=True), gravity)
        actual_history = [(float(row[0]), *(np.array(term, copy=True) for term in row[1:])) for row in selected] + [current]
        template = sonic.decoder_input(token, actual_history)

        def predict(z):
            value = template.copy(); value[0, :64] = z
            action = controller.session.run([controller.output_name], {controller.input_name: value})[0]
            return sonic.body_targets(np.asarray(action)[0])

        self.batch_audit_inputs = []
        def predict_batch(tokens):
            actor_inputs = np.repeat(template, 64, axis=0); actor_inputs[:, :64] = tokens
            actions = self.batch_view.predict_actions(actor_inputs)
            if self.capture_batch_audit:
                self.batch_audit_inputs.append((actor_inputs.copy(), actions.copy()))
            return np.stack([sonic.body_targets(row) for row in actions])

        started = time.perf_counter()
        new_token, diagnostic = optimize_latent(token, desired, predict, self.config,
            predict_targets_batch=predict_batch if self.batch_view else None)
        require(live_fingerprint(model, data) == before_live and controller_fingerprint(controller) == before_controller,
                "Live state/controller changed while projecting; refuse the stale proposal")
        diagnostic.update(baseline | diagnostic)
        diagnostic.update(reason="bounded_latent_residual_accepted" if diagnostic["applied"] else "original_token_retained",
            CPU_seconds=time.perf_counter() - started, real_history_times_s=[float(row[0]) for row in actual_history],
            predicted_decoder_time_s=t, actor_nonlatent_sha256=hashlib.sha256(template[0, 64:].tobytes()).hexdigest(),
            projected_actor_input_sha256=hashlib.sha256(sonic.decoder_input(new_token, actual_history).tobytes()).hexdigest(),
            live_state_history_unchanged=True, decoder_sha256=sonic.OFFICIAL_DECODER_SHA256,
            jacobian_evaluator="metadata-only derived B64 view" if self.batch_view else "official B1",
            candidate_acceptance_evaluator="unchanged original official B1",
            derived_view_receipt=self.batch_view.receipt if self.batch_view else None,
            latent_semantics="bounded continuous residual in official numeric range; not requantized to FSQ grid",
            scope="Predict the next50Hz decoder joint targets from actual current proprioception and9 prior real samples; not future physical joint/wrist motion or closed-loop success")
        return new_token, diagnostic


def archived_comparison(pilot_root, sonic_resources, teacher_ticks=(50, 150, 250), *, jacobian_batch_view=None,
                        config=ProjectionConfig(), compare_default_scope=False):
    """Use recorded real integration states/commands, never synthesize history.

    Prior raw decoder actions are deterministically reproduced with the same
    official controller on each recorded50Hz state. No physics step is called.
    The IK desired q is reconstructed from the archived expert's raw source
    frame + recorded bounded arm delta, with the original joint-limit clipping.
    """
    import mujoco
    from handover_random100_sources import load_raw_source, sha256
    from prepare_carrybox_v11 import TRAIN_STATE_NAMES
    sonic = sonic_module()
    root = Path(pilot_root).resolve(strict=True)
    model = mujoco.MjModel.from_binary_path(str(root / "physics_model.mjb"))
    result = json.loads((root / "result.json").read_text())
    source = None
    trace = [json.loads(line) for line in (root / "trace.jsonl").read_text().splitlines() if line.strip()]
    commands = {row["teacher_tick50"]: row for row in (json.loads(line)
        for line in (root / "expert_commands.jsonl").read_text().splitlines() if line.strip())}
    with np.load(root / "trajectory.npz", allow_pickle=False) as values:
        states, spec = np.array(values["physics_state_integration"]), int(values["physics_state_spec"])
        times = np.array(values["sim_time_s"])
    require(spec == int(mujoco.mjtState.mjSTATE_INTEGRATION), "Need complete actual recorded integration states")
    require(len(states) == len(times) == len(trace) + 1
            and np.allclose(np.diff(times), .02, atol=1e-6, rtol=0)
            and all(row["tick50"] == i and abs(row["sim_time_s"] - times[i]) < 1e-9
                    and abs(row["next_sim_time_s"] - times[i + 1]) < 1e-9 for i, row in enumerate(trace)),
            "Archived integration/trace clock mismatch")
    comparisons = []
    projector = SonicLatentProjector(config, jacobian_batch_view=jacobian_batch_view, capture_batch_audit=True)
    for teacher_tick in teacher_ticks:
        require(teacher_tick in commands and teacher_tick + 50 < len(trace), "Choose an actual recorded new-command tick")
        index = teacher_tick + 50
        data = mujoco.MjData(model); robot = SimpleNamespace(mjModel=model, mjData=data)
        controller = sonic.SonicV11OnnxController(sonic_resources, allow_unverified_compatibility=True,
            pd_profile="fixed_cpp_baseline", hand_torque_profile="recording_0p7_nm", sync_mujoco_observations=True)
        for previous in range(index):
            mujoco.mj_setState(model, data, states[previous], spec)
            controller.apply_physics(robot, trace[previous]["actual_body_token64"], trace[previous]["actual_hand_targets"])
        mujoco.mj_setState(model, data, states[index], spec); mujoco.mj_forward(model, data)
        raw = SimpleNamespace(mujoco=robot)
        metadata = commands[teacher_tick]["diagnostics"]
        if "desired_body_q29" in metadata:
            require(metadata.get("desired_body_q29_names") == list(sonic.BODY_JOINT_NAMES),
                    "Archived IK desired q must preserve all29 named body joints")
            desired = finite(metadata["desired_body_q29"], (29,), "archived expert IK desired targets")
            desired_origin = "actual archived named expert desired_body_q29, including waist IK"
        else:
            require(config.control_scope == "arms", "Waist experiment requires the real archived named29-joint IK target")
            if source is None: source = load_raw_source(result["source_episode_index"])
            desired = source["values"]["reference_full_state43"][metadata["actual_reference_frame"], :29].copy()
            desired[15:] += np.asarray(metadata["current_arm_joint_delta"])
            for joint_index in range(15, 29):
                joint = model.joint(TRAIN_STATE_NAMES[joint_index])
                if joint.limited[0]: desired[joint_index] = np.clip(desired[joint_index], *joint.range)
            desired_origin = "archived source frame plus actual recorded arm delta, legacy diagnostic reconstruction"
        actual_token = np.asarray(commands[teacher_tick]["body_token64"], np.float32)
        original = np.asarray(metadata.get("unprojected_body_token64", actual_token), np.float32)
        require(abs(commands[teacher_tick]["sim_time_s"] - times[index]) < 1e-9
                and np.array_equal(actual_token, np.asarray(trace[index]["actual_body_token64"], np.float32)),
                "Archived command is not bound to this actual control tick")
        before = (live_fingerprint(model, data), controller_fingerprint(controller))
        projected, diagnostic = projector.project(original, desired, raw, controller)
        require(before == (live_fingerprint(model, data), controller_fingerprint(controller)), "Projector modified archived live state")
        require(diagnostic["applied"] and diagnostic["controlled_rms_after_rad"] < diagnostic["controlled_rms_before_rad"]
                and diagnostic["max_protected_target_change_rad"] <= config.max_protected_target_delta_rad
                and diagnostic["max_leg_target_change_rad"] <= .002,
                "Actual official decoder did not improve controlled targets within the protected bound")
        # Independently run the actual unchanged apply_physics ordering on a
        # separate controller/data copy; the projector prediction must match.
        probe = copy(controller)
        probe._history = deque([(row[0], *(np.array(term, copy=True) for term in row[1:])) for row in controller._history], maxlen=60)
        probe._last_raw_action = controller._last_raw_action.copy(); probe._body_target = controller._body_target.copy()
        probe_data = mujoco.MjData(model); mujoco.mj_setState(model, probe_data, states[index], spec)
        probe.apply_physics(SimpleNamespace(mjModel=model, mjData=probe_data), projected, trace[index]["actual_hand_targets"])
        require(np.array_equal(probe._body_target, diagnostic["projected_decoder_targets_q29"]),
                "Prediction differs from actual official apply_physics history/order")
        diagnostic.update(teacher_tick50=teacher_tick, global_control_tick50=index,
                          actual_controller_apply_physics_prediction_exact=True, archived_expert_phase=metadata.get("phase"),
                          desired_target_origin=desired_origin, evaluated_configuration=asdict(config))
        if projector.batch_view:
            started = time.perf_counter()
            errors = []
            for actor_inputs, batch_actions in projector.batch_audit_inputs:
                official_actions = np.concatenate([controller.session.run([controller.output_name],
                    {controller.input_name: row[None]})[0] for row in actor_inputs])
                errors.append(float(np.max(np.abs(batch_actions - official_actions))))
                require(np.allclose(batch_actions, official_actions, rtol=2e-5, atol=2e-5),
                        "Derived batch row differs from original official B1 beyond the declared tolerance")
            diagnostic.update(derived_all_FD_rows_compared_with_original_B1=True,
                derived_compared_rows=64 * len(errors), derived_actions_max_abs_error=max(errors),
                independent_row_validation_CPU_seconds=time.perf_counter() - started,
                row_validation_excluded_from_projection_timing=True,
                derived_row_tolerance={"rtol": 2e-5, "atol": 2e-5})
            # Timed original B1 alternative on the very same state/history and
            # target, independent of the batch audit overhead.
            _, b1 = SonicLatentProjector(config).project(original, desired, raw, controller)
            diagnostic["official_B1_projection_comparison"] = {key: b1[key] for key in
                ("CPU_seconds", "arm_rms_before_rad", "arm_rms_after_rad", "decoder_evaluations",
                 "controlled_rms_before_rad", "controlled_rms_after_rad", "waist_rms_before_rad", "waist_rms_after_rad",
                 "max_protected_target_change_rad", "max_leg_target_change_rad")}
            diagnostic["projection_speedup_vs_B1"] = b1["CPU_seconds"] / diagnostic["CPU_seconds"]
        if compare_default_scope:
            _, default = SonicLatentProjector(ProjectionConfig(), jacobian_batch_view=jacobian_batch_view).project(
                original, desired, raw, controller)
            diagnostic["default_arms_scope_comparison"] = {key: default[key] for key in
                ("CPU_seconds", "arm_rms_before_rad", "arm_rms_after_rad", "waist_rms_before_rad", "waist_rms_after_rad",
                 "max_protected_target_change_rad", "max_leg_target_change_rad", "max_waist_target_change_rad")}
        require(before == (live_fingerprint(model, data), controller_fingerprint(controller)),
                "Independent comparisons modified actual archived live state/history")
        comparisons.append(diagnostic)
    return {"contract": CONTRACT, "status": "official_CPU_decoder_comparison_passed", "comparisons": comparisons,
        "source_sha256": {str(root / name): sha256(root / name) for name in
            ("physics_model.mjb", "trajectory.npz", "trace.jsonl", "expert_commands.jsonl", "result.json")},
        "projector_sha256": sha256(__file__), "GPU_used": False, "physics_stepped": False,
        "closed_loop_task_success_claim": False, "current_expert_or_run_modified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-batch-view", type=Path, help="Create fresh derived ONNX+receipt instead of comparing")
    parser.add_argument("--jacobian-batch-view", type=Path, help="Optional derived view for FD only")
    parser.add_argument("--control-scope", choices=("arms", "waist_arms"), default="arms",
        help="waist_arms selects the optional3-iteration0.16/0.5/0.2 bounded experiment")
    parser.add_argument("--compare-default-scope", action="store_true")
    parser.add_argument("--pilot-root", type=Path)
    parser.add_argument("--sonic-resources", required=True, type=Path)
    parser.add_argument("--teacher-ticks50", type=int, nargs="+", default=[50, 150, 250])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.build_batch_view:
        require(args.output is None and args.pilot_root is None and args.jacobian_batch_view is None,
                "Do not combine building with comparison arguments")
        print(json.dumps(build_batch_view(args.sonic_resources / "model_decoder.onnx", args.build_batch_view)))
        return
    require(args.output is not None and args.pilot_root is not None, "Comparison needs --output and --pilot-root")
    output = args.output.resolve()
    require(not output.exists() and not output.is_relative_to(REPO)
            and not output.is_relative_to(args.pilot_root.resolve())
            and not output.is_relative_to(args.sonic_resources.resolve()), "Use a fresh external comparison report")
    report = archived_comparison(args.pilot_root, args.sonic_resources, tuple(args.teacher_ticks50),
        jacobian_batch_view=args.jacobian_batch_view, compare_default_scope=args.compare_default_scope,
        config=ProjectionConfig.waist_arms_experiment() if args.control_scope == "waist_arms" else ProjectionConfig())
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream: stream.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "output": str(output), "comparisons": [
        {key: row[key] for key in ("teacher_tick50", "arm_rms_before_rad", "arm_rms_after_rad",
            "max_protected_target_change_rad", "decoder_evaluations", "CPU_seconds")} for row in report["comparisons"]]}))


if __name__ == "__main__":
    main()
