"""Isolated CPU-only physical locomotion diagnostic, without SDK or renderer.

This opt-in executable does not connect to, send commands to, or restart the
current collector. Every output is diagnostic and forbidden for training.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SIMPLE = REPO/'VLMExpert/third_party/SIMPLE'
sys.path[:0] = [str(SIMPLE/'src'), str(SIMPLE/'third_party')]

DIRECTIONS = {'forward': (1., 0., 0.), 'backward': (-1., 0., 0.),
              'left': (0., 1., 0.), 'right': (0., -1., 0.)}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def model_input_files(xml_path):
    """Bind the selected XML, included XMLs and referenced mesh/texture bytes."""
    seen = set()

    def visit(path):
        path = Path(path).resolve(strict=True)
        if path in seen:
            return
        seen.add(path)
        tree = ET.parse(path).getroot()
        for include in tree.iter('include'):
            visit(path.parent/include.attrib['file'])
        compiler = tree.find('compiler')
        meshdir = compiler.get('meshdir', '') if compiler is not None else ''
        texturedir = compiler.get('texturedir', '') if compiler is not None else ''
        for kind, directory in (('mesh', meshdir), ('texture', texturedir)):
            for asset in tree.findall('./asset/'+kind):
                if 'file' in asset.attrib:
                    seen.add((path.parent/directory/asset.attrib['file']).resolve(strict=True))

    visit(xml_path)
    return sorted(seen)


def fixed_command(tick, direction, speed):
    walking = 100 <= tick < 400
    return {'mode': 1 if walking else 0, 'movement_direction': list(direction) if walking else [0., 0., 0.],
            'facing_direction': [1., 0., 0.], 'speed': speed if walking else -1., 'height': -1.}


class RecordedIntent:
    """Causal offline command playback with a 200ms upper freshness bound.

    A fresh command can cover intervening 50Hz ticks; it is never counted as a
    new PICO sample. A source gap or end emits explicit IDLE. The synchronous
    planner receives that stop rather than silently retaining an old walk.
    """
    def __init__(self, path):
        from pico_locomotion_reference import OfflinePlannerReference
        self.path = Path(path).resolve(strict=True)
        receipt = json.loads(self.path.with_name('receipt.json').read_text())
        if (receipt.get('contract') != 'cvpr-pico-root-planner-offline-replay-v1'
                or receipt.get('intents_sha256') != sha(self.path)
                or receipt.get('training_eligible') is not False):
            raise ValueError('Bound diagnostic replay receipt required')
        with self.path.open() as stream:
            self.rows = [json.loads(line) for line in stream]
        self.times = np.asarray([r['elapsed_device_s'] for r in self.rows], dtype=float)
        if (not len(self.times) or not np.isfinite(self.times).all()
                or self.times[0] < 0 or np.any(np.diff(self.times) <= 0)):
            raise ValueError('Finite strictly increasing intent times required')
        for row in self.rows:
            OfflinePlannerReference._command(row['planner'])
            if row.get('diagnostic_only') is not True or row.get('training_eligible') is not False:
                raise ValueError('Replayed intent must remain diagnostic')
        self.receipt = receipt

    def at(self, elapsed_s, actual_root_pose7):
        if not np.isfinite(elapsed_s) or elapsed_s < 0:
            raise ValueError('Nonnegative finite replay time required')
        index = int(np.searchsorted(self.times, elapsed_s, side='right'))-1
        if index < 0 or elapsed_s-self.times[index] > .2:
            from handover_sonic_teleop_reference import finite, quaternion, heading_wxyz, rotation
            root = finite(actual_root_pose7, (7,), 'actual robot root')
            heading = heading_wxyz(quaternion(root[3:], (4,), 'actual root quaternion'))
            command = fixed_command(0, (0., 0., 0.), .3)
            command['facing_direction'] = rotation(heading)[:, 0].tolist()
            return command, {'reason': 'no_fresh_recorded_intent', 'sample_reused_as_new': False,
                             'facing_source': 'current_actual_robot_heading'}
        row = self.rows[index]
        return dict(row['planner']), {'source_sequence': row['source_sequence'],
            'device_timestamp_ns': row['device_timestamp_ns'], 'intent_row': index,
            'age_device_s': elapsed_s-self.times[index], 'sample_reused_as_new': False}


def run(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Explicit CUDA_VISIBLE_DEVICES="" required for isolated CPU diagnostic')
    if not .1 <= args.speed <= .4:
        raise ValueError('Diagnostic SLOW_WALK speed must be .1.. .4 m/s')
    # Resource-heavy imports occur only after opt-in, never when inspecting CLI.
    import mujoco
    import onnxruntime as ort
    from simple.baselines.sonic_v1_1 import SonicV11OnnxController, BODY_JOINT_NAMES, SIMPLE_HAND_JOINT_NAMES, DEFAULT_ANGLES
    from simple.baselines.live_pose_sonic_warmup import LivePoseSonicWarmup
    from pico_locomotion_reference import OfflinePlannerReference
    from sonic_locomotion_physics import build_model, initialize, foot_geometry, audit_locomotion, model_evidence
    from handover_sonic_teleop_reference import rotation
    from pico_sonic_reference import ENCODER_SHA256

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    resources = args.sonic_resources.resolve(strict=True)
    sources = [Path(__file__), Path(__file__).with_name('pico_locomotion_reference.py'),
               Path(__file__).with_name('sonic_locomotion_physics.py'),
               Path(__file__).with_name('sonic_locomotion_planner.py'),
               SIMPLE/'src/simple/baselines/sonic_v1_1.py',
               SIMPLE/'src/simple/baselines/live_pose_sonic_warmup.py',
               Path(__file__).with_name('prepare_carrybox_v11.py'),
               Path(__file__).with_name('handover_sonic_teleop_reference.py'),
               resources/'model_encoder.onnx', resources/'model_decoder.onnx',
               resources/'observation_config.yaml', args.planner_model.resolve(strict=True)]
    model_path = args.official_scene if args.environment == 'official' else args.simple_robot
    if model_path is None:
        raise ValueError('Explicit selected model XML path required')
    model_sources = model_input_files(model_path)
    sources += model_sources
    intent_source = RecordedIntent(args.intent_trace) if args.intent_trace is not None else None
    if intent_source:
        sources += [intent_source.path, intent_source.path.with_name('receipt.json')]
    bindings = {str(p): sha(p) for p in sources}
    if sha(resources/'model_encoder.onnx') != ENCODER_SHA256:
        raise ValueError('Pinned SONIC v1.1 encoder identity mismatch')
    model = build_model(args.environment, official_scene_path=args.official_scene,
                        simple_robot_path=args.simple_robot)
    data, initial = initialize(model, DEFAULT_ANGLES)
    mujoco.mj_saveModel(model, str(output/'physics_model.mjb'), None)
    controller = SonicV11OnnxController(resources, allow_unverified_compatibility=True,
        providers=['CPUExecutionProvider'], pd_profile='official_v1_1_ankle_pitch_1_5',
        hand_torque_profile='mjcf_limits', sync_mujoco_observations=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    encoder = ort.InferenceSession(str(resources/'model_encoder.onnx'), sess_options=options,
                                  providers=['CPUExecutionProvider'])
    encoder_name = encoder.get_inputs()[0].name
    reference = OfflinePlannerReference(args.planner_model, seed=args.seed)
    warmup = LivePoseSonicWarmup(resources, hand_policy='measured')
    private = mujoco.MjData(model)
    robot = SimpleNamespace(mjModel=model, mjData=data)
    actual_robot = SimpleNamespace(mjModel=model, mjData=private)
    spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
    root_adr = int(model.joint('floating_base_joint').qposadr[0])
    all_names = tuple(BODY_JOINT_NAMES) + tuple(SIMPLE_HAND_JOINT_NAMES)
    states, samples, forces, commands, encoder_inputs, tokens = [], [], [], [], [], []
    warmup_commands = []
    times = []

    def state():
        value = np.empty(mujoco.mj_stateSize(model, spec))
        mujoco.mj_getState(model, data, value, spec)
        return value

    def force_check():
        value = {'sim_time_s': float(data.time),
                 'maximum_xfrc_applied': float(np.abs(data.xfrc_applied).max()),
                 'maximum_qfrc_applied': float(np.abs(data.qfrc_applied).max()),
                 'neq': int(model.neq), 'nmocap': int(model.nmocap)}
        if (value['maximum_xfrc_applied'] != 0 or value['maximum_qfrc_applied'] != 0
                or model.neq or model.nmocap):
            raise ValueError('External force, constraint, or mocap intervention is forbidden')
        if not all(np.isfinite(v).all() for v in (data.qpos, data.qvel, data.ctrl)):
            raise ValueError('Nonfinite actual state')
        return value

    result = {'contract': 'cvpr-isolated-pico-locomotion-physical-diagnostic-v1',
              'started_utc': datetime.now(timezone.utc).isoformat(),
              'environment': args.environment, 'direction': args.direction, 'speed_m_s': args.speed,
              'physics_hz': 200, 'control_hz': 50, 'GPU_used': False, 'SDK_started': False,
              'production_collector_contacted': False, 'rendered_RGB': False,
              'diagnostic_only': True, 'accepted_new_dataset': False, 'human_demonstration': False,
              'training_eligible': False, 'automatic_resets': False,
              'state_restored_after_first_physics_step': False,
              'controller_identity': controller.identity, 'initialization': initial,
              'input_source': 'archived diagnostic PICO intent' if intent_source else 'explicit fixed walking command',
              'physical_live_human_following_validated': False,
              'intent_replay_receipt': intent_source.receipt if intent_source else None,
              'plan': ('1s measured warmup; 10s causal archived intent playback' if intent_source
                       else '1s measured warmup; IDLE2s, SLOW_WALK6s, IDLE2s'),
              'gait_acceptance_scope': ('not applicable to arbitrary intent trace; no live human following acceptance'
                                       if intent_source else 'fixed requested direction and 2s/6s/2s local diagnostic gates'),
              'source_bindings': bindings, 'cpu_affinity': sorted(os.sched_getaffinity(0)),
              'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              'model_source_xml': str(model_path.resolve(strict=True)),
              'model_source_bindings': {str(p): bindings[str(p)] for p in model_sources},
              'compiled_model_evidence': model_evidence(model),
              'nice': os.getpriority(os.PRIO_PROCESS, 0)}
    states.append(state()); times.append(float(data.time))
    early_stop = None
    try:
        for global_tick in range(550):
            root = data.qpos[root_adr:root_adr+7].copy()
            body = np.asarray([data.joint(n).qpos[0] for n in BODY_JOINT_NAMES])
            if global_tick < 50:
                mujoco.mj_setState(model, private, states[-1], spec)
                mujoco.mj_forward(model, private)
                observation = {'joint_names': list(all_names),
                    'joint_qpos': [float(private.joint(n).qpos[0]) for n in all_names],
                    'sim_time_s': float(private.time)}
                command = warmup.command(actual_robot, controller, observation)
                token, hands = command['body_token'], command['hand_targets']
                warmup_commands.append({'global_tick50': global_tick,
                    'physics_prestate_row': len(states)-1, 'sim_time_s': float(data.time),
                    'actual_observation43': observation, 'actual_root_wxyz': private.body('pelvis').xquat.tolist(),
                    'token64': None if token is None else np.asarray(token).tolist(),
                    'hand_targets': hands, 'phase': warmup.phase,
                    'history_length_before_step': len(controller._history)})
            else:
                tick = global_tick - 50
                if intent_source:
                    command, source_evidence = intent_source.at(tick/50., root)
                else:
                    command, source_evidence = fixed_command(tick, DIRECTIONS[args.direction], args.speed), None
                value, evidence = reference.step(tick, root, body, command)
                evidence['input_source_evidence'] = source_evidence
                token = np.asarray(encoder.run(None, {encoder_name: value})[0][0], np.float32)
                if token.shape != (64,) or not np.isfinite(token).all():
                    raise ValueError('Invalid encoded SONIC v1.1 latent')
                hands = {n: 0. for n in SIMPLE_HAND_JOINT_NAMES}
                encoder_inputs.append(value[0]); tokens.append(token)
                commands.append({'tick50': tick, 'sim_time_s': float(data.time), **evidence})
            for _ in range(4):
                before = force_check()
                controller.apply_physics(robot, token, hands)
                mujoco.mj_step(model, data)
                after = force_check()
                forces.append({'physics_step200': len(forces)+1, 'before': before, 'after': after})
                states.append(state()); times.append(float(data.time))
            mujoco.mj_setState(model, private, states[-1], spec)
            mujoco.mj_kinematics(model, private)
            mujoco.mj_comPos(model, private)
            root = private.qpos[root_adr:root_adr+7].copy()
            sample = {'global_tick50': global_tick, 'post_warmup_tick50': global_tick-50,
                'physics_state_row': len(states)-1,
                'sim_time_s': float(private.time), 'actual_root_pose7': root.tolist(),
                'actual_root_qvel6': private.qvel[:6].tolist(),
                'actual_BODY29': [float(private.joint(n).qpos[0]) for n in BODY_JOINT_NAMES],
                'actual_decoded_PD_target29': controller._body_target.tolist(),
                'pelvis_tilt_rad': float(np.arccos(np.clip(rotation(root[3:])[2, 2], -1., 1.))),
                'feet': {s: foot_geometry(model, private, s) for s in ('left', 'right')}}
            samples.append(sample)
            if root[2] < .4:
                early_stop = 'pelvis_below_0p4m'
                break
        gait_audit = (audit_locomotion(samples, forces, movement_world=DIRECTIONS[args.direction])
                      if not intent_source else {'passed': False, 'applicable': False,
                        'scope': 'arbitrary trace execution without a trajectory-following acceptance protocol',
                        'human_following_validated': False, 'accepted_new_dataset': False,
                        'reasons': ['Fixed direction/time gait gates cannot certify arbitrary archived intents']})
        result.update(status='execution_complete', early_stop=early_stop,
                      completed_physics_steps=len(forces), completed_intervals50=len(samples),
                      warmup=warmup.receipt,
                      source_bindings_unchanged=all(sha(p) == v for p, v in bindings.items()),
                      audit=gait_audit)
    except Exception as error:
        result.update(status='execution_failed', error=repr(error))
    finally:
        np.savez_compressed(output/'trajectory.npz', physics_state_spec=spec,
            physics_state_integration=np.asarray(states), sim_time_s=np.asarray(times),
            encoder_inputs=np.asarray(encoder_inputs), token64=np.asarray(tokens))
        for name, rows in [('samples50.jsonl', samples), ('force_guard200.jsonl', forces),
                           ('commands50.jsonl', commands), ('warmup_commands50.jsonl', warmup_commands)]:
            with (output/name).open('x') as stream:
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
        result['ended_utc'] = datetime.now(timezone.utc).isoformat()
        result['saved_payload_sha256'] = {p.name: sha(p) for p in output.iterdir() if p.is_file()}
        write_json(output/'result.json', result)
    print(json.dumps({k: result.get(k) for k in ('status', 'error', 'direction', 'audit')}, ensure_ascii=False))
    return 0 if (result.get('audit', {}).get('passed') and result.get('source_bindings_unchanged')) else 1


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument('--environment', choices=['official', 'simple'], required=True)
    value.add_argument('--official-scene', type=Path)
    value.add_argument('--simple-robot', type=Path)
    value.add_argument('--sonic-resources', type=Path, required=True)
    value.add_argument('--planner-model', type=Path, required=True)
    value.add_argument('--output', type=Path, required=True)
    value.add_argument('--direction', choices=DIRECTIONS, default='forward')
    value.add_argument('--intent-trace', type=Path, help='Optional offline intents.jsonl with matching diagnostic receipt')
    value.add_argument('--speed', type=float, default=.3)
    value.add_argument('--seed', type=int, default=20261025)
    return value


if __name__ == '__main__':
    raise SystemExit(run(parser().parse_args()))
