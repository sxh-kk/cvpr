"""Small synthetic CPU fixtures test admission, tampering and queue recovery.

These fixtures are unit-test data, never live RTX demonstrations or evidence of
task success. No subprocess, model, renderer or GPU is started in this file.
"""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import mujoco
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import collect_handover_quality100 as collector
import handover_contact_quality as quality
from handover_geometry_quality import build_container_geometry_cache

# Tiny asset stand-ins isolate hashing/path logic from multi-megabyte installed
# robot assets. Only the test double changes the expected asset identities;
# real captures always import and enforce the source helper's pinned hashes.
FIXTURE_ASSETS = {name: hashlib.sha256(("CPU fixture calibration:"+name).encode()).hexdigest() for name in (
    'robots/g1/g1_29dof_wholebody_dex3.urdf','robots/g1/g1_29dof_wholebody_dex3.usd',
    'robots/g1/g1_29dof_with_dex3_base_fix.usd','robots/g1/meshes/head_link.STL')}
ORIGINAL_FRAME_LOAD = collector.load_frame_calibration
def fixture_frame_load(model,result):
    module=ORIGINAL_FRAME_LOAD(model,result)
    module.SOURCE_IDENTITIES=dict(FIXTURE_ASSETS)
    return module

spec = importlib.util.spec_from_file_location("planner_test_fixture", Path(__file__).with_name("test_plan_handover_quality100.py"))
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)


def plan_file(root):
    manifest, selection = fixtures.fixture()
    plan = fixtures.build(manifest, selection)
    path = root / "plans/plan.json"
    path.parent.mkdir(exist_ok=True)
    collector.write(path, plan)
    return path, plan


def quality_rows():
    rows = []
    for i in range(65):
        left, right = (False, True) if i < 5 else (True, True) if i < 10 else (True, False) if i < 15 else (False, False)
        rows.append({"completed_steps": i+1, "sim_time_s": 1.005+(i+1)/50,
            "actual_target_xyz_m": [0, 0, .7 if i < 15 else .61], "pelvis_z_m": .75,
            "contacts": {"left_hand": left, "right_hand": right, "container": i >= 15},
            "solver_stage": quality.SOLVED_STATE_STAGE,
            "container_geometry": {"inside_container_footprint": True, "signed_margin_m": .02,
                "collision_vertical_ranges_overlap": True, "container_upright_cosine": 1.,
                "container_support_weight_fraction": 1., "support_measurement": "live_solved_mujoco_contact_force"}})
    return rows


def fixture_capture(root, candidate, plan_sha):
    run, capture, source = root / "run", root / "run/capture", root / "source"
    capture.mkdir(parents=True)
    source.mkdir()
    hand_names = [f"{side}_hand_{finger}_{j}_joint" for side in ("left", "right")
                  for finger, count in (("thumb", 3), ("middle", 2), ("index", 2)) for j in range(count)]
    names = [f"j{i}" for i in range(29)] + hand_names
    hinges = ''.join(f'<body><joint name="{name}" type="hinge"/><geom type="sphere" size=".01" mass=".01" contype="0" conaffinity="0"/></body>' for name in names if name not in ('j12','j13','j14'))
    inertial='<geom type="sphere" size=".01" mass=".01" contype="0" conaffinity="0"/>'
    arms=''.join(f'<body name="{side}_shoulder_pitch_link" pos=".0039563 {y} .23778"><body name="{side}_wrist_yaw_link" pos=".2 0 0"/></body>' for side,y in (('left',.10022),('right',-.10021)))
    waist=('<body name="waist_yaw_link"><joint name="j12" axis="0 0 1"/>'+inertial+
        '<body name="waist_roll_link" pos="-.0039635 0 .035"><joint name="j13" axis="1 0 0"/>'+inertial+
        '<body name="torso_link" pos="0 0 .019"><joint name="j14" axis="0 1 0"/>'+inertial+
        arms+'<camera name="head_stereo_left" pos="0 0 .1"/></body></body></body>')
    xml = '<mujoco><option timestep=".005"/><worldbody><body name="pelvis" pos="0 0 .75"><freejoint/><geom type="sphere" size=".1" mass="1"/>' + hinges + waist + '</body>'
    for role, x in (("target", 0), ("container", .3)):
        xml += f'<body name="{role}" pos="{x} 0 .55"><freejoint/><geom type="box" size=".02 .02 .02" mass=".1"/></body>'
    model = mujoco.MjModel.from_xml_string(xml + '</worldbody></mujoco>')
    data = mujoco.MjData(model)
    base_joints=np.zeros(43);base_joints[12:15]=[.2,.3,-.1]
    for name,q in zip(names,base_joints):data.qpos[model.jnt_qposadr[model.joint(name).id]]=q
    receipt = {"apply_count": 1, "phase": "before_warmup", "objects": {},
               "material_seed_applied": False, "generation_seed_applied": False}
    for role, x in (("target", 0), ("container", .3)):
        delta = candidate["initial_object_deltas"][role]
        source_pose = np.array([x, 0, .55, 1, 0, 0, 0], dtype=np.float64)
        applied = source_pose.copy(); applied[:2] += delta["xy_delta_m"]
        applied[3:] = [np.cos(delta["world_z_yaw_delta_rad"]/2), 0, 0, np.sin(delta["world_z_yaw_delta_rad"]/2)]
        joint = int(model.body_jntadr[model.body(role).id])
        data.qpos[int(model.jnt_qposadr[joint]):int(model.jnt_qposadr[joint])+7] = applied
        receipt["objects"][role] = {"source_pose7": source_pose.tolist(), "applied_pose7": applied.tolist(), **delta}
    mujoco.mj_forward(model, data)
    mujoco.mj_saveModel(model, str(run / "physics_model.mjb"))
    collector.write(run / "container_geometry_cache.json", build_container_geometry_cache(model, data,
        container_body="container", target_body="target"))
    state_spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
    states, times = [], []
    for tick in range(116):
        data.time = .005 + tick/50
        state = np.empty(mujoco.mj_stateSize(model, state_spec))
        mujoco.mj_getState(model, data, state, state_spec)
        states.append(state); times.append(float(data.time))
    joints = np.tile(base_joints,(116,1)); roots = np.tile([0, 0, .75, 1, 0, 0, 0], (116, 1)); velocities = np.zeros((116, model.nv))
    np.savez_compressed(run / "trajectory.npz", physics_state_spec=np.asarray(state_spec),
        physics_state_integration=np.asarray(states), sim_time_s=np.asarray(times), joint_names=np.asarray(names),
        actual_joint_q43=joints, root_pose7=roots, actual_qvel=velocities)
    forces = [{"physics_step": i+1, "sim_time_s": .005+(i+1)/200,
               "qfrc_applied_all_zero": True, "xfrc_applied_all_zero": True,
               "max_abs_qfrc_applied": 0, "max_abs_xfrc_applied": 0} for i in range(460)]
    (run / "applied_force_samples.jsonl").write_text(''.join(json.dumps(x)+'\n' for x in forces))
    rows = quality_rows()
    report = quality.evaluate_capture(rows, initial_target_z_m=.55)
    collector.write(run / "physical_quality.json", report)
    (run / "quality_samples.jsonl").write_text(''.join(json.dumps(x)+'\n' for x in rows))
    train = source / "training/Handover-sonic11-train"
    (train / "meta").mkdir(parents=True)
    (train / "data/chunk-000").mkdir(parents=True)
    tokens = np.arange(65*64, dtype=np.float32).reshape(65,64)/1000
    hands = np.arange(65*14, dtype=np.float32).reshape(65,14)/10000
    parquet = train / "data/chunk-000/episode_000000.parquet"
    replay_names = [f"{side}_hand_{finger}_{j}_joint" for side in ("left", "right")
                    for finger, count in (("thumb", 3), ("index", 2), ("middle", 2)) for j in range(count)]
    source_objects = [term for role in ("target", "container") for term in receipt["objects"][role]["source_pose7"]]
    pq.write_table(pa.table({"action.body_token_v1_1": tokens.tolist(), "action": hands.tolist(),
        "replay_action78": np.concatenate([tokens,hands[:,[hand_names.index(n) for n in replay_names]]],axis=1).tolist(),
        "replay.initial_state43": [base_joints.tolist()]*65, "observation.base_pose": roots[:65].tolist(),
        "observation.base_vel": [[0.]*6]*65, "observation.object_poses": [source_objects]*65,
        "timestamp": (np.arange(65)/50).tolist()}), parquet)
    info = train / "meta/info.json"
    object_terms = [f"{role}.{term}" for role in ("target", "container") for term in ("pos_x","pos_y","pos_z","quat_w","quat_x","quat_y","quat_z")]
    collector.write(info, {"fps": 50, "sonic_version": "1.1", "features": {"action": {"names": hand_names},
        "observation.object_poses": {"names": object_terms}}})
    selection = source / "training/selection.json"
    n, source_key = candidate["ancestor"]["source_episode_index"], candidate["ancestor"]["source_dataset_key"]
    collector.write(selection, {"episodes": [{"source_scene_episode_index": n, "split": "train", "data_category": "original_or_repaired"}]})
    episodes = train / "meta/episodes.jsonl"
    episodes.write_text(json.dumps({"episode_index":0, "source_scene_episode_index":n, "data_category":"original_or_repaired",
        "split_restriction":"unrestricted", "lineage_complete":True, "source_dataset_key":source_key,
        "lineage_group_sources":[{"source_dataset_key":source_key,"source_episode_index":n}],
        "environment_config": json.dumps({"dr_state_dict":{role:{"label":role} for role in ("target","container")}})})+'\n')
    source_physics = source/f'source_episode_{n:06d}_physics.json'
    collector.write(source_physics, {"reset_joint_names":names,"reset_joint_state43":base_joints.tolist(),"gravity":[0.,0.,-9.81],
        "objects":{role:{"mass":.1} for role in ("target","container")}})
    source_sha = {str(p): collector.sha256(p) for p in (parquet, info, selection, episodes, source_physics)}
    assets=root/'assets';assets.mkdir()
    for relative in FIXTURE_ASSETS:
        path=assets/relative;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(('CPU fixture calibration:'+relative).encode())
    code_paths = [SCRIPTS / name for name in ("replay_handover_teacher.py", "handover_eval_runtime.py", "handover_contact_quality.py", "handover_geometry_quality.py")]
    simple=SCRIPTS.parents[1]/'VLMExpert/third_party/SIMPLE/src/simple'
    code_paths.extend(simple/name for name in ('engines/isaacsim.py','envs/sonic_loco_manip.py','core/action.py'))
    code_paths.append(collector.FRAME_CALIBRATION)
    snapshots = []
    for i, code in enumerate(code_paths):
        destination = run / f"code/{i}-{code.name}"
        destination.parent.mkdir(exist_ok=True)
        shutil.copy2(code, destination)
        snapshots.append({"original_path":str(code), "path":str(destination.relative_to(run)), "sha256":collector.sha256(code)})
    result = {"status":"teacher_diagnostic_completed", "candidate_id":candidate["candidate_id"],
        "candidate_spec_sha256":candidate["candidate_spec_sha256"], "plan_sha256":plan_sha,
        "source_episode_index":n, "source_split":"train", "source_commands":65,
        "command_rate_hz":30,"physics_hz":200,"decoder_hz":50,"accepted_new_dataset":False,
        "finite_physics":True,"fell":False,"render_mode":"isaac","rgb_collected":True,
        "controller_identity":{"pd_profile":"fixed_cpp_baseline","hand_torque_profile":"recording_0p7_nm","body_token_version":"body_token_v1_1"},
        "warmup_control":{"kind":"measured_pose_pd","seconds":1.},"warmup_ticks":50,"decoder_calls":65,"initial_target_z_m":.55,
        "initial_object_deltas":candidate["initial_object_deltas"],"perturbation_receipt":receipt,
        "object_restores_during_control":False,"source_sha256":source_sha,
        'resource_data_root':str(assets.resolve()),'calibration_source_asset_sha256':dict(FIXTURE_ASSETS),
        'calibration_source_assets_unchanged':True,
        "source_provenance":{"source_record":{"source_episode_index":n,"source_dataset_key":source_key,
            "train_episode_index":0,"source_rows":65,
            "parquet":{"path":"training/Handover-sonic11-train/data/chunk-000/episode_000000.parquet","sha256":collector.sha256(parquet)},
            "required_files":{"parquet":{"path":"training/Handover-sonic11-train/data/chunk-000/episode_000000.parquet","sha256":collector.sha256(parquet)}},
            "plan_declared_source_parquet_sha256":candidate["ancestor"]["source_parquet_sha256"]}},
        "initialization":{"sim_time_s":.005},
        "capture_code_sha256":{str(p):collector.sha256(p) for p in code_paths},"code_snapshots":snapshots,"code_unchanged_during_run":True,
        "physical_quality":report, "physics_model":{"sha256":collector.sha256(run/"physics_model.mjb")},
        "trajectory":{"sha256":collector.sha256(run/"trajectory.npz")},
        "force_guard":{"path":"applied_force_samples.jsonl","sha256":collector.sha256(run/"applied_force_samples.jsonl"),
            "steps":460,"equality_count":0,"target_robot_attachment":False,"all_applied_forces_zero":True}}
    trace = []
    for tick in range(115):
        index = (((tick-50)*30//50)*50+29)//30 if tick>=50 else None
        measured = {f'{role}_pose7':receipt['objects'][role]['applied_pose7'] for role in ('target','container')}
        trace.append({'tick50':tick,'phase':'teacher_command' if tick>=50 else 'warmup_measured_pose_pd',
            'decoder_calls_before':max(tick-50,0),'decoder_calls_after':max(tick-49,0),
            'sim_time_s':.005+tick/50,'next_sim_time_s':.005+(tick+1)/50,'elapsed_s':tick/50,
            'physics_state_row_before':tick,'physics_state_row_after':tick+1,
            'measured_before':measured,'measured_after':measured,'source_command_index':index,
            'actual_body_token64':tokens[index].tolist() if index is not None else None,
            'actual_hand_targets':dict(zip(hand_names,hands[index].tolist() if index is not None else [0.]*14))})
    (run/'trace.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in trace))
    events = []
    diagnostics={'mujoco_sim_time_s':.005,'isaac_world_time_s':0.,'links':{},'cameras':{}}
    def usd_pose(position,quaternion,path):
        return {'valid':True,'path':path,'position_m':position.tolist(),'quaternion_wxyz':quaternion.tolist()}
    for name in ('pelvis','torso_link','left_wrist_yaw_link','right_wrist_yaw_link'):
        body=model.body(name).id
        diagnostics['links'][name]={'usd':usd_pose(data.xpos[body],data.xquat[body],'/fixture/'+name),
            'mujoco_position_m':data.xpos[body].tolist(),'mujoco_quaternion_wxyz':data.xquat[body].tolist()}
    # Independent known source joint offsets, deliberately at nonzero yaw/roll.
    torso,yaw,roll=(model.body(n).id for n in ('torso_link','waist_yaw_link','waist_roll_link'))
    usd_torso=(data.xpos[torso]+data.xmat[yaw].reshape(3,3)@np.array([2.04354525e-10,0,.00899999976158142])
               +data.xmat[roll].reshape(3,3)@np.array([0,0,-.019]))
    diagnostics['links']['torso_link']['usd']['position_m']=usd_torso.tolist()
    for side,y in (('left',.10022),('right',-.10021)):
        wrist=model.body(side+'_wrist_yaw_link').id;shoulder=model.body(side+'_shoulder_pitch_link').id
        source_shoulder=np.asarray([.0039563,y,.24778],dtype=np.float32).astype(np.float64)
        position=data.xpos[wrist]+usd_torso-data.xpos[torso]+data.xmat[torso].reshape(3,3)@(source_shoulder-model.body_pos[shoulder])
        diagnostics['links'][side+'_wrist_yaw_link']['usd']['position_m']=position.tolist()
    camera=model.camera('head_stereo_left').id;quat=np.empty(4);mujoco.mju_mat2Quat(quat,data.cam_xmat[camera])
    diagnostics['cameras']['head_stereo_left']={'usd':usd_pose(data.cam_xpos[camera],quat,'/fixture/camera'),
        'sensor_position_m':data.cam_xpos[camera].tolist(),'sensor_quaternion_wxyz_usd_axes':quat.tolist(),
        'mujoco_position_m':data.cam_xpos[camera].tolist(),'mujoco_rotation_usd_axes':data.cam_xmat[camera].tolist()}
    usd_camera=usd_torso+data.xmat[torso].reshape(3,3)@np.array([.05762349767610431,.01752999983727932,.42987000942230225])
    diagnostics['cameras']['head_stereo_left']['usd']['position_m']=usd_camera.tolist()
    diagnostics['cameras']['head_stereo_left']['sensor_position_m']=usd_camera.tolist()
    for k in range(69):
        tick=(k*50+29)//30; warm=tick<50; index=tick-50
        action=None if warm else {"body_token64":tokens[index].tolist(),"hand14":hands[index].tolist(),"hand_joint_names":hand_names}
        pixels=np.empty((8,16,3),np.uint8);pixels[:]=[k,n%256,17]
        path=capture/f'rgb/frame_{k:06d}.png';path.parent.mkdir(exist_ok=True);Image.fromarray(pixels).save(path)
        events.append({"event_index":k,"capture_index30":k,"physical_tick50":tick,"sim_time_s":.005+tick/50,
            "actual_elapsed_s":tick/50,"nominal_elapsed_s":k/30,"phase":"warmup" if warm else "teacher_command",
            "action_valid":not warm,"source_command_index":None if warm else index,"joint_names":names,"state43":base_joints.astype(np.float32).tolist(),
            "physical_joint_q43_float64":base_joints.tolist(),"physical_joint_qvel43":[0.]*43,"root_pose7":roots[tick].tolist(),"root_qvel6":[0.]*6,
            "objects":{role:{"pose7":receipt['objects'][role]['applied_pose7']} for role in ['target','container']},
            'renderer_camera':'head_stereo_left','renderer_pose_diagnostics':copy.deepcopy(diagnostics),
            "action":action,"rgb":{"path":str(path.relative_to(capture)),"file_sha256":collector.sha256(path),
                "pixel_sha256":hashlib.sha256(pixels.tobytes()).hexdigest(),"width":16,"height":8,"channels":3,"dtype":"uint8"}})
        events[-1]['renderer_pose_diagnostics']['mujoco_sim_time_s']=.005+tick/50
    events_path=capture/'events.jsonl';events_path.write_text(''.join(json.dumps(x)+'\n' for x in events))
    meta={key:copy.deepcopy(result[key]) for key in ['plan_sha256','candidate_id','candidate_spec_sha256','source_episode_index',
        'initial_target_z_m','source_sha256','initial_object_deltas','perturbation_receipt','code_unchanged_during_run',
        'resource_data_root','calibration_source_asset_sha256','calibration_source_assets_unchanged','warmup_control']}
    meta.update(contract='cvpr-handover-teacher-rtx-capture-v1',status='raw_capture_completed_pending_independent_audit',
        command_rate_hz=30,physics_hz=200,decoder_hz=50,render_engine='Isaac RTX',warmup_seconds=1.,
        renderer_pose_contract=collector.RENDERER_POSE_CONTRACT,isaac_world_time_reference_s=0.,render_mount_equivalence_verified=False,
        object_restores_during_control=False,accepted_new_dataset=False,events_path='events.jsonl',events_sha256=collector.sha256(events_path),event_count=69)
    collector.write(capture/'capture.json',meta);collector.write(run/'result.json',result)
    return run,capture,result,events,meta


def convert_fixture_to_live(run,capture,result,meta):
    """Synthetic live receipt tests; no encoder or model is executed."""
    code=collector.LIVE_WARMUP_SOURCE;destination=run/'code/live-warmup.py';shutil.copyfile(code,destination)
    identity=collector.sha256(code);result['capture_code_sha256'][str(code)]=identity
    result['code_snapshots'].append({'original_path':str(code),'path':str(destination.relative_to(run)),'sha256':identity})
    with np.load(run/'trajectory.npz',allow_pickle=False) as values:
        q=values['actual_joint_q43'][0];names=values['joint_names'].tolist();root=values['root_pose7'][0]
    hand_names=names[-14:]
    warmup={'kind':'live_initial_pose_sonic_v1_1','helper_source_sha256':identity,'hand_policy':'measured',
        'state_forcing':False,'history_injection':False,'initialization_count':1,'encoder_mode':0,
        'encoder_calls':40,'first_sim_time_s':.005,'reference_body_q29':q[:29].astype(np.float32).tolist(),
        'reference_root_wxyz':root[3:].tolist(),'reference_hands':dict(zip(hand_names,q[-14:].astype(np.float32).tolist())), 'events':[]}
    path=run/'trace.jsonl';trace=[json.loads(x) for x in path.read_text().splitlines()]
    calls=0
    for tick,row in enumerate(trace):
        row['decoder_calls_before']=calls
        if 10<=tick<50:
            token=np.full(64,(tick+1)/1000,dtype=np.float32).tolist()
            row['phase']='warmup_live_pose_sonic_v1_1';row['actual_body_token64']=token
            warmup['events'].append({'sim_time_s':row['sim_time_s'],'token64':token,'state_forcing':False,'history_injection':False})
        if tick>=10:calls+=1
        row['decoder_calls_after']=calls
    path.write_text(''.join(json.dumps(x)+'\n' for x in trace))
    result['warmup_control']=warmup;result['decoder_calls']=calls
    meta['warmup_control']={'kind':'live_pose_sonic_v1_1','seconds':1.}
    collector.write(run/'result.json',result);collector.write(capture/'capture.json',meta)


class IndependentCaptureTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        patcher=patch.object(collector,'load_frame_calibration',side_effect=fixture_frame_load)
        patcher.start();self.addCleanup(patcher.stop)
        _,plan=plan_file(self.root)
        self.candidate=next(c for c in plan['candidates'] if c['ancestor']['source_episode_index']==0)
        self.run,self.capture,self.result,self.events,self.meta=fixture_capture(self.root,self.candidate,'a'*64)

    def audit(self):
        return collector.audit_capture(self.run,self.capture,self.candidate,'a'*64)

    def save_events(self):
        path=self.capture/'events.jsonl';path.write_text(''.join(json.dumps(x)+'\n' for x in self.events))
        self.meta['events_sha256']=collector.sha256(path);collector.write(self.capture/'capture.json',self.meta)

    def save_trajectory(self, change):
        path=self.run/'trajectory.npz'
        with np.load(path,allow_pickle=False) as data: values={k:data[k].copy() for k in data.files}
        model=mujoco.MjModel.from_binary_path(str(self.run/'physics_model.mjb'))
        change(values,model)
        np.savez_compressed(path,**values)
        self.result['trajectory']['sha256']=collector.sha256(path)
        collector.write(self.run/'result.json',self.result)

    def save_trace(self, change):
        path=self.run/'trace.jsonl';rows=[json.loads(x) for x in path.read_text().splitlines()]
        change(rows);path.write_text(''.join(json.dumps(x)+'\n' for x in rows))

    def test_full_fixture_requires_physics_pixels_source_actions_and_real_clock(self):
        audit=self.audit()
        self.assertTrue(audit['eligible_dataset']);self.assertEqual(audit['warmup_events'],30);self.assertEqual(audit['teacher_events'],39)
        self.assertTrue(audit['physical_gate']['verified_handover_success'])
        self.assertGreater(len(audit['bindings']),69)

    def test_self_reported_success_cannot_override_failed_measured_release(self):
        rows=quality_rows();rows[-1]['contacts']['left_hand']=True
        (self.run/'quality_samples.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
        with self.assertRaisesRegex(ValueError,'recomputed'):self.audit()

    def test_camera_free_success_never_admitted(self):
        self.result['render_mode']='none';self.result['rgb_collected']=False;collector.write(self.run/'result.json',self.result)
        with self.assertRaisesRegex(ValueError,'Camera-free'):self.audit()

    def test_png_pixel_corruption_and_path_escape_rejected(self):
        self.events[1]['rgb']['pixel_sha256']='0'*64;self.save_events()
        with self.assertRaisesRegex(ValueError,'pixels'):self.audit()
        self.events[1]['rgb']['path']='../physics_model.mjb';self.save_events()
        with self.assertRaisesRegex(ValueError,'relative'):self.audit()

    def test_fabricated_warmup_labels_or_wrong_source_actions_rejected(self):
        self.events[0]['action']={'fake':True};self.save_events()
        with self.assertRaisesRegex(ValueError,'Warmup'):self.audit()
        self.events[0]['action']=None;self.events[31]['action']['body_token64'][0]+=1;self.save_events()
        with self.assertRaisesRegex(ValueError,'source teacher command'):self.audit()

    def test_wrong_event_clock_or_fabricated_physical_state_rejected(self):
        self.events[2]['sim_time_s']+=.005;self.save_events()
        with self.assertRaisesRegex(ValueError,'clock'):self.audit()
        self.events[2]['sim_time_s']-=.005;self.events[2]['state43'][0]=1;self.events[2]['physical_joint_q43_float64'][0]=1;self.save_events()
        with self.assertRaisesRegex(ValueError,'integration-history'):self.audit()

    def test_mid_substep_external_force_cannot_hide_in_zero50hz_states(self):
        path=self.run/'applied_force_samples.jsonl';rows=[json.loads(x) for x in path.read_text().splitlines()]
        rows[3]['max_abs_xfrc_applied']=.01;path.write_text(''.join(json.dumps(x)+'\n' for x in rows))
        self.result['force_guard']['sha256']=collector.sha256(path);collector.write(self.run/'result.json',self.result)
        with self.assertRaisesRegex(ValueError,'Applied forces/clock'):self.audit()

    def test_source_or_snapshot_tampering_is_rejected(self):
        p=Path(next(iter(self.result['source_sha256'])));p.write_bytes(p.read_bytes()+b' ')
        with self.assertRaisesRegex(ValueError,'source file identity'):self.audit()

    def test_integration_joint_change_cannot_hide_behind_unchanged_auxiliary_arrays(self):
        def change(values,model):
            data=mujoco.MjData(model);spec=int(values['physics_state_spec'])
            for state in values['physics_state_integration']:
                mujoco.mj_setState(model,data,state,spec)
                data.qpos[model.jnt_qposadr[model.joint('j0').id]]=.2
                mujoco.mj_getState(model,data,state,spec)
        self.save_trajectory(change)
        with self.assertRaisesRegex(ValueError,'authoritative integration'):self.audit()

    def test_root_velocity_fake_is_rejected_without_changing_measured_joints(self):
        self.events[5]['root_qvel6'][0]=.25;self.save_events()
        with self.assertRaisesRegex(ValueError,'integration-history'):self.audit()

    def test_noninitial_object_event_and_integration_trace_tampering_are_rejected(self):
        self.events[5]['objects']=copy.deepcopy(self.events[5]['objects'])
        self.events[5]['objects']['target']['pose7'][0]+=.1;self.save_events()
        with self.assertRaisesRegex(ValueError,'object pose.*integration'):self.audit()
        self.events[5]['objects']['target']['pose7'][0]-=.1;self.save_events()
        def change(values,model):
            data=mujoco.MjData(model);spec=int(values['physics_state_spec']);state=values['physics_state_integration'][9]
            mujoco.mj_setState(model,data,state,spec)
            joint=model.body_jntadr[model.body('container').id]
            data.qpos[model.jnt_qposadr[joint]]+=.1
            mujoco.mj_getState(model,data,state,spec)
        self.save_trajectory(change)
        with self.assertRaisesRegex(ValueError,'trace object pose'):self.audit()

    def test_absolute_event_clock_translation_is_rejected(self):
        for event in self.events:event['sim_time_s']+=3
        self.save_events()
        with self.assertRaisesRegex(ValueError,'clock'):self.audit()

    def test_all_clock_fields_translated_together_still_fail_fixed_reset_anchor(self):
        def change(values,model):
            values['sim_time_s']+=3
            data=mujoco.MjData(model);spec=int(values['physics_state_spec'])
            for state in values['physics_state_integration']:
                mujoco.mj_setState(model,data,state,spec);data.time+=3;mujoco.mj_getState(model,data,state,spec)
        self.save_trajectory(change)
        self.result['initialization']['sim_time_s']+=3
        def shift_trace(rows):
            for row in rows:row.update(sim_time_s=row['sim_time_s']+3,next_sim_time_s=row['next_sim_time_s']+3)
        self.save_trace(shift_trace)
        guards=self.run/'applied_force_samples.jsonl';rows=[json.loads(x) for x in guards.read_text().splitlines()]
        for row in rows:row['sim_time_s']+=3
        guards.write_text(''.join(json.dumps(x)+'\n' for x in rows));self.result['force_guard']['sha256']=collector.sha256(guards)
        samples=self.run/'quality_samples.jsonl';rows=[json.loads(x) for x in samples.read_text().splitlines()]
        for row in rows:row['sim_time_s']+=3
        samples.write_text(''.join(json.dumps(x)+'\n' for x in rows))
        self.result['physical_quality']=quality.evaluate_capture(rows,initial_target_z_m=.55)
        collector.write(self.run/'physical_quality.json',self.result['physical_quality'])
        collector.write(self.run/'result.json',self.result)
        for event in self.events:
            event['sim_time_s']+=3;event['renderer_pose_diagnostics']['mujoco_sim_time_s']+=3
        self.save_events()
        with self.assertRaisesRegex(ValueError,'Absolute reset'):self.audit()

    def test_source_and_applied_receipts_cannot_be_shifted_together(self):
        receipt=self.result['perturbation_receipt']['objects']['target']
        receipt['source_pose7'][0]+=3;receipt['applied_pose7'][0]+=3
        self.meta['perturbation_receipt']=copy.deepcopy(self.result['perturbation_receipt'])
        def change(values,model):
            data=mujoco.MjData(model);spec=int(values['physics_state_spec'])
            joint=model.body_jntadr[model.body('target').id];adr=model.jnt_qposadr[joint]
            for state in values['physics_state_integration']:
                mujoco.mj_setState(model,data,state,spec);data.qpos[adr]+=3;mujoco.mj_getState(model,data,state,spec)
        self.save_trajectory(change)
        collector.write(self.run/'result.json',self.result);collector.write(self.capture/'capture.json',self.meta)
        with self.assertRaisesRegex(ValueError,'Source-row0'):self.audit()

    def test_wrong_command_on_unsampled50hz_hold_tick_is_rejected(self):
        def change(rows):
            # tick51 is not an RTX30Hz event; both it and tick50 must hold source0.
            rows[51]['source_command_index']=1
            rows[51]['actual_body_token64']=rows[52]['actual_body_token64']
        self.save_trace(change)
        with self.assertRaisesRegex(ValueError,'floor30/ceil50'):self.audit()

    def test_cache_role_swap_is_rejected(self):
        path=self.run/'container_geometry_cache.json';cache=collector.read(path)
        cache['target_root_body_name']='container';collector.write(path,cache)
        with self.assertRaisesRegex(ValueError,'object role'):self.audit()

    def test_matching_integration_and_auxiliary_initial_joints_still_require_source_row0(self):
        def change(values,model):
            data=mujoco.MjData(model);spec=int(values['physics_state_spec'])
            mujoco.mj_setState(model,data,values['physics_state_integration'][0],spec)
            data.qpos[model.jnt_qposadr[model.joint('j0').id]]=.2
            mujoco.mj_getState(model,data,values['physics_state_integration'][0],spec)
            values['actual_joint_q43'][0,0]=.2
        self.save_trajectory(change)
        with self.assertRaisesRegex(ValueError,'SHA-bound source row0'):self.audit()

    def test_renderer_absolute_tick_and_nonadvancing_world_clock_are_required(self):
        self.events[4]['renderer_pose_diagnostics']['mujoco_sim_time_s']+=.02;self.save_events()
        with self.assertRaisesRegex(ValueError,'Renderer diagnostic absolute time'):self.audit()
        self.events[4]['renderer_pose_diagnostics']['mujoco_sim_time_s']-=.02
        self.events[4]['renderer_pose_diagnostics']['isaac_world_time_s']+=.02;self.save_events()
        with self.assertRaisesRegex(ValueError,'Isaac world clock'):self.audit()

    def test_renderer_self_reported_mujoco_pose_must_match_integration_kinematics(self):
        self.events[1]['renderer_pose_diagnostics']['links']['left_wrist_yaw_link']['mujoco_position_m'][0]+=.01
        self.save_events()
        with self.assertRaisesRegex(ValueError,'authoritative integration kinematics'):self.audit()

    def test_real_usd_and_sensor_offset_is_measured_and_rejected_at_declared1mm(self):
        camera=self.events[5]['renderer_pose_diagnostics']['cameras']['head_stereo_left']
        camera['usd']['position_m'][0]+=.01;camera['sensor_position_m'][0]+=.01
        self.save_events()
        with self.assertRaises(collector.CaptureRejected) as caught:self.audit()
        report=caught.exception.receipt['renderer_pose_audit']
        self.assertEqual(report['events_checked'],69)
        self.assertAlmostEqual(report['maxima']['camera_sensor_position_error_m'],.01)
        self.assertEqual(report['position_tolerance_m'],.001)
        self.assertFalse(caught.exception.receipt['eligible_dataset'])

    def test_renderer_rotation_tolerance_is_point1degree_and_invalid_prims_fail(self):
        camera=self.events[6]['renderer_pose_diagnostics']['cameras']['head_stereo_left']
        rotation=[np.cos(np.radians(.2)/2),0,0,np.sin(np.radians(.2)/2)]
        camera['usd']['quaternion_wxyz']=rotation;camera['sensor_quaternion_wxyz_usd_axes']=rotation
        self.save_events()
        with self.assertRaises(collector.CaptureRejected) as caught:self.audit()
        self.assertAlmostEqual(caught.exception.receipt['renderer_pose_audit']['rotation_tolerance_degrees'],.1)
        camera['usd']['valid']=False;self.save_events()
        with self.assertRaisesRegex(ValueError,'absent/invalid'):self.audit()

    def test_new_admission_requires_persisted_renderer_evidence(self):
        self.events[0].pop('renderer_pose_diagnostics');self.save_events()
        with self.assertRaises(KeyError):self.audit()

    def test_torso_mapping_at_nonzero_waist_angles_is_not_a_fitted_constant_offset(self):
        torso=self.events[5]['renderer_pose_diagnostics']['links']['torso_link']
        torso['usd']['position_m']=(np.asarray(torso['mujoco_position_m'])+[0,0,-.01]).tolist()
        self.save_events()
        with self.assertRaises(collector.CaptureRejected) as caught:self.audit()
        self.assertGreater(caught.exception.receipt['renderer_pose_audit']['maxima']['max_link_position_error_m'],.001)
        self.assertIn('never first-frame fitted',caught.exception.receipt['renderer_pose_audit']['frame_mapping']['kind'])

    def test_new_waist_joint_frame_assumptions_require_a_reviewed_mapping(self):
        model=mujoco.MjModel.from_binary_path(str(self.run/'physics_model.mjb'))
        model.body_quat[model.body('waist_roll_link').id]=[np.cos(.1),np.sin(.1),0,0]
        with self.assertRaisesRegex(ValueError,'identity body rotations'):
            collector.load_frame_calibration(model,self.result)

    def test_actual_calibration_asset_missing_or_changed_cannot_self_redeclare_identity(self):
        relative=next(iter(FIXTURE_ASSETS));path=Path(self.result['resource_data_root'])/relative
        path.write_bytes(b'changed asset')
        with self.assertRaisesRegex(ValueError,'source asset is missing or changed'):self.audit()
        self.result['calibration_source_asset_sha256'][relative]=collector.sha256(path)
        self.meta['calibration_source_asset_sha256']=copy.deepcopy(self.result['calibration_source_asset_sha256'])
        collector.write(self.run/'result.json',self.result);collector.write(self.capture/'capture.json',self.meta)
        with self.assertRaisesRegex(ValueError,'source identities'):self.audit()

    def test_installed_robot_asset_symlinks_bind_actual_resolved_payloads(self):
        root=Path(self.result['resource_data_root']);shared=self.root/'shared-robot-assets'
        (root/'robots').rename(shared);(root/'robots').symlink_to(shared,target_is_directory=True)
        audit=self.audit()
        for relative,identity in FIXTURE_ASSETS.items():
            self.assertEqual(audit['bindings'][str((root/relative).resolve())],identity)

    def test_wrist_source_chain_mapping_is_required_at_nonzero_waist_angles(self):
        wrist=self.events[7]['renderer_pose_diagnostics']['links']['left_wrist_yaw_link']
        wrist['usd']['position_m']=copy.deepcopy(wrist['mujoco_position_m']);self.save_events()
        with self.assertRaises(collector.CaptureRejected) as caught:self.audit()
        self.assertGreater(caught.exception.receipt['renderer_pose_audit']['maxima']['max_link_position_error_m'],.001)

    def test_declared_live_warmup_uses_real_trace_but_never_supervises_prefix(self):
        convert_fixture_to_live(self.run,self.capture,self.result,self.meta)
        audit=collector.audit_capture(self.run,self.capture,self.candidate,'a'*64,expected_warmup='live_pose_sonic_v1_1')
        self.assertTrue(audit['eligible_dataset']);self.assertEqual(audit['warmup_events'],30)
        self.assertEqual(audit['teacher_collection_warmup']['control'],'live_pose_sonic_v1_1')
        self.assertFalse(audit['teacher_collection_warmup']['evaluation_oracle_allowed'])
        self.assertTrue(all(event['action'] is None and event['action_valid'] is False for event in self.events[:30]))
        with self.assertRaisesRegex(ValueError,'preregistered control schedule'):self.audit()

    def test_live_receipt_tokens_counters_and_no_forcing_are_not_self_authorized(self):
        convert_fixture_to_live(self.run,self.capture,self.result,self.meta)
        self.result['warmup_control']['events'][0]['token64'][0]+=1
        collector.write(self.run/'result.json',self.result)
        with self.assertRaisesRegex(ValueError,'bound real-history receipt'):
            collector.audit_physics(self.run,self.candidate,'a'*64,expected_warmup='live_pose_sonic_v1_1')
        self.result['warmup_control']['events'][0]['token64'][0]-=1
        self.result['decoder_calls']-=40;collector.write(self.run/'result.json',self.result)
        with self.assertRaisesRegex(ValueError,'Final SONIC decoder count'):
            collector.audit_physics(self.run,self.candidate,'a'*64,expected_warmup='live_pose_sonic_v1_1')


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.plan_path,self.plan=plan_file(self.root)
        self.live_source=next(a['source_episode_index'] for a in self.plan['original_ancestors'] if a['source_episode_index']!=0)
        patcher=patch.object(collector,'load_frame_calibration',side_effect=fixture_frame_load)
        patcher.start();self.addCleanup(patcher.stop)
        self.queue=self.root/'queue';collector.initialize_queue(self.plan_path,self.queue)
        self.inputs={}
        for key in ['snapshot_root','data_root','sonic_resources']:
            p=self.root/key;p.mkdir();self.inputs[key]=p
        self.inputs['python']=Path(sys.executable)

    def schedule_file(self):
        document={'contract':collector.WARMUP_SCHEDULE_CONTRACT,'scope':collector.WARMUP_SCOPE,
            'plan_sha256':collector.sha256(self.plan_path),
            'controls':{str(a['source_episode_index']):'live_pose_sonic_v1_1' if a['source_episode_index']==self.live_source else 'measured_pose_pd'
                        for a in self.plan['original_ancestors']}}
        path=self.root/'train-schedule.json';collector.write(path,document)
        return path,document

    def install_admitted_fixture(self,source_id=0,warmup='measured_pose_pd',schedule_path=None):
        candidate=next(c for c in self.plan['candidates'] if c['ancestor']['source_episode_index']==source_id)
        with patch.object(collector,'execute_phase',side_effect=RuntimeError('unit-test freeze only, no child')):
            collector.collect(self.queue,**self.inputs,candidate_id=candidate['candidate_id'],warmup_schedule=schedule_path)
        raw=self.root/'raw-fixture';raw.mkdir()
        run,capture,result,_,meta=fixture_capture(raw,candidate,collector.sha256(self.plan_path))
        if warmup=='live_pose_sonic_v1_1':convert_fixture_to_live(run,capture,result,meta)
        attempt=self.queue/'attempt-00000'
        shutil.move(run,attempt/'rtx')
        audit=collector.audit_capture(attempt/'rtx',attempt/'rtx/capture',candidate,collector.sha256(self.plan_path),expected_warmup=warmup)
        audit_path=attempt/'independent_capture_audit.json';collector.write(audit_path,audit)
        state=collector.read(self.queue/'queue.json')
        state['attempts']=[{'status':'accepted','candidate_id':candidate['candidate_id'],
            'candidate_spec_sha256':candidate['candidate_spec_sha256'],'plan_path':str(self.plan_path.resolve()),
            'plan_sha256':collector.sha256(self.plan_path),'source_episode_index':source_id,'directory':'attempt-00000',
            'warmup_control':warmup,
            'rtx_capture_completed':True,'audit_path':str(audit_path.relative_to(self.queue)),
            'audit_sha256':collector.sha256(audit_path)}]
        collector.write(self.queue/'queue.json',collector.queue_summary(state))
        return state,audit_path,audit

    def update_admission(self,state,audit_path,audit):
        collector.write(audit_path,audit)
        state['attempts'][0]['audit_sha256']=collector.sha256(audit_path)
        collector.write(self.queue/'queue.json',state)

    def test_resume_reaudits_admitted_fixture_and_refuses_missing_png(self):
        state,path,audit=self.install_admitted_fixture()
        self.assertEqual(collector.verified_queue(self.queue)[0]['accepted_count'],1)
        png=next(Path(p) for p in audit['bindings'] if p.endswith('.png'));png.unlink()
        with self.assertRaisesRegex(ValueError,'payload/source is missing'):collector.verified_queue(self.queue)

    def test_updated_receipt_sha_does_not_authorize_nonexistent_payload(self):
        state,path,audit=self.install_admitted_fixture()
        audit['bindings'][str(self.root/'nonexistent.png')]='0'*64
        self.update_admission(state,path,audit)
        with self.assertRaisesRegex(ValueError,'payload/source is missing'):collector.verified_queue(self.queue)

    def test_updated_receipt_cannot_change_candidate_or_acceptance_authority(self):
        state,path,audit=self.install_admitted_fixture()
        audit['source_episode_index']=9;self.update_admission(state,path,audit)
        with self.assertRaisesRegex(ValueError,'candidate/plan/source'):collector.verified_queue(self.queue)
        audit['source_episode_index']=0;audit['teacher_events']=999;self.update_admission(state,path,audit)
        with self.assertRaisesRegex(ValueError,'independent.*revalidation'):collector.verified_queue(self.queue)

    def test_duplicate_candidate_cannot_increment_accepted_count(self):
        state,path,audit=self.install_admitted_fixture()
        duplicate=copy.deepcopy(state['attempts'][0]);duplicate['directory']='attempt-00001'
        state['attempts'].append(duplicate);collector.write(self.queue/'queue.json',state)
        with self.assertRaisesRegex(ValueError,'repeats.*candidate'):collector.verified_queue(self.queue)

    def test_old_queue_authority_requires_a_fresh_queue(self):
        state=collector.read(self.queue/'queue.json');state['contract']='cvpr-handover-quality100-serial-queue-v1'
        collector.write(self.queue/'queue.json',state)
        with self.assertRaisesRegex(ValueError,'fresh queue'):collector.verified_queue(self.queue)

    def test_accepted_registry_cannot_drop_preexecution_frozen_recipe(self):
        state,path,audit=self.install_admitted_fixture();state.pop('recipe')
        collector.write(self.queue/'queue.json',state)
        with self.assertRaisesRegex(ValueError,'pre-execution'):collector.verified_queue(self.queue)

    def test_failed_render_numeric_gate_keeps_its_independent_rejection(self):
        receipt={'renderer_pose_audit':{'position_tolerance_m':.001,'observed_m':.01},'eligible_dataset':False}
        with patch.object(collector,'execute_phase'),patch.object(collector,'audit_physics',return_value={'eligible_dataset':False}), \
                patch.object(collector,'audit_capture',side_effect=collector.CaptureRejected('fixture rejection',receipt)):
            collector.collect(self.queue,**self.inputs,max_candidates=1)
        state=collector.read(self.queue/'queue.json');attempt=state['attempts'][0]
        self.assertEqual(attempt['status'],'capture_rejected');self.assertEqual(state['accepted_count'],0)
        path=self.queue/attempt['rejection_path'];self.assertEqual(collector.read(path),receipt)
        self.assertEqual(collector.sha256(path),attempt['rejection_sha256'])

    def test_schedule_is_complete_train_only_and_base_plan_bound(self):
        path,document=self.schedule_file()
        expected=collector.load_warmup_schedule(self.plan,collector.sha256(self.plan_path),path)
        self.assertEqual(expected['controls'][str(self.live_source)],'live_pose_sonic_v1_1');self.assertFalse(expected['evaluation_oracle_allowed'])
        default=collector.load_warmup_schedule(self.plan,collector.sha256(self.plan_path))
        self.assertEqual(set(default['controls'].values()),{'measured_pose_pd'})
        for mutation in ('val','missing','badmode','wrongplan','evaluationscope'):
            value=copy.deepcopy(document)
            if mutation=='val':value['controls']['39']='measured_pose_pd'
            if mutation=='missing':value['controls'].pop(str(self.live_source))
            if mutation=='badmode':value['controls'][str(self.live_source)]='automatic_success_oracle'
            if mutation=='wrongplan':value['plan_sha256']='0'*64
            if mutation=='evaluationscope':value['scope']='VAL_TEST_oracle'
            collector.write(path,value)
            with self.assertRaisesRegex(ValueError,'exactly its24'):collector.load_warmup_schedule(self.plan,collector.sha256(self.plan_path),path)

    def test_schedule_is_frozen_before_both_children_and_resume_cannot_change_it(self):
        path,document=self.schedule_file()
        candidate=next(c for c in self.plan['candidates'] if c['ancestor']['source_episode_index']==self.live_source)
        calls=[]
        def execute(command,env,log,state,attempt,phase,root,timeout):
            calls.append((phase,command[command.index('--warmup-control')+1]))
            frozen=collector.read(root/'frozen_recipe.json')
            self.assertEqual(frozen['warmup_schedule']['controls'][str(self.live_source)],'live_pose_sonic_v1_1')
        with patch.object(collector,'execute_phase',side_effect=execute),patch.object(collector,'audit_physics',return_value={'eligible_dataset':False}), \
                patch.object(collector,'audit_capture',side_effect=ValueError('fixture preserves rejection')) as audit:
            collector.collect(self.queue,**self.inputs,candidate_id=candidate['candidate_id'],warmup_schedule=path)
            self.assertEqual(audit.call_args.kwargs['expected_warmup'],'live_pose_sonic_v1_1')
        self.assertEqual(calls,[('screen','live_pose_sonic_v1_1'),('capture','live_pose_sonic_v1_1')])
        state=collector.verified_queue(self.queue)[0];self.assertEqual(state['attempts'][0]['warmup_control'],'live_pose_sonic_v1_1')
        document['controls'][str(self.live_source)]='measured_pose_pd';collector.write(path,document)
        with self.assertRaisesRegex(ValueError,'schedule changed'):collector.verified_queue(self.queue)

    def test_attempt_cannot_silently_switch_away_from_frozen_source_control(self):
        with patch.object(collector,'execute_phase',side_effect=RuntimeError('fixture no child')):
            collector.collect(self.queue,**self.inputs,max_candidates=1)
        state=collector.read(self.queue/'queue.json');state['attempts'][0]['warmup_control']='live_pose_sonic_v1_1'
        collector.write(self.queue/'queue.json',state)
        with self.assertRaisesRegex(ValueError,'warmup differs'):collector.verified_queue(self.queue)

    def test_accepted_live_receipt_is_independently_recomputed_from_frozen_schedule(self):
        path,document=self.schedule_file()
        self.install_admitted_fixture(self.live_source,'live_pose_sonic_v1_1',path)
        state,_,_=collector.verified_queue(self.queue)
        self.assertEqual(state['accepted_count'],1)
        self.assertEqual(state['recipe']['warmup_schedule']['controls'][str(self.live_source)],'live_pose_sonic_v1_1')
        snapshot=self.queue/'frozen_warmup_schedule.json';snapshot.write_text(snapshot.read_text()+' ')
        with self.assertRaisesRegex(ValueError,'schedule snapshot changed'):collector.verified_queue(self.queue)

    def test_extension_is_preregistered_unique_and_does_not_increase_acceptance_cap(self):
        first=collector.register_extension(self.queue,variants_per_ancestor=1)
        self.assertEqual(first['registered_candidates'],24)
        state,plan,candidates=collector.verified_queue(self.queue)
        self.assertEqual(len(candidates),144);self.assertEqual(state['accepted_count'],0)
        self.assertEqual(plan['acceptance_limits']['maximum_accepted_per_ancestor'],5)
        self.assertEqual(len({c['candidate_spec_sha256'] for c,_,_ in candidates}),144)
        with self.assertRaisesRegex(ValueError,'attempt budget'):
            collector.register_extension(self.queue,variants_per_ancestor=5,maximum_total_candidates=150)

    def test_complete_requires_exact100_all24_and_per_family_cap5(self):
        attempts=[{'status':'accepted','source_episode_index':i} for i in range(24) for _ in range(4)]
        attempts += [{'status':'accepted','source_episode_index':i} for i in range(4)]
        state=collector.queue_summary({'attempts':copy.deepcopy(attempts)})
        self.assertTrue(state['train_capture_target_satisfied']);self.assertFalse(state['final_ready'])
        self.assertFalse(collector.queue_summary({'attempts':attempts[:-1]})['train_capture_target_satisfied'])
        bad=[{'status':'accepted','source_episode_index':0}]*6
        with self.assertRaisesRegex(ValueError,'quotas'):collector.queue_summary({'attempts':bad})

    def test_failed_attempts_retained_resume_never_repeats_output_and_pilot_selects_existing(self):
        calls=[]
        def execute(command,env,log,state,attempt,phase,root,timeout):
            calls.append((attempt['candidate_id'],phase));Path(log).write_text('unit-test no child process\n')
            if len(calls)==1:raise RuntimeError('fixture screen failure')
        def audit(run,capture,candidate,identity,**kwargs):
            return {'eligible_dataset':True,'capture_content_sha256':candidate['candidate_spec_sha256']}
        pilot=next(c for c in self.plan['candidates'] if c['ancestor']['source_episode_index']==0)
        with patch.object(collector,'execute_phase',side_effect=execute),patch.object(collector,'audit_physics',return_value={'eligible_dataset':False}),patch.object(collector,'audit_capture',side_effect=audit):
            collector.collect(self.queue,**self.inputs,candidate_id=pilot['candidate_id'])
            collector.collect(self.queue,**self.inputs,candidate_id=pilot['candidate_id'])
            self.assertEqual(len(calls),1)
            collector.collect(self.queue,**self.inputs,max_candidates=1)
        state=collector.read(self.queue/'queue.json')
        self.assertEqual([a['status'] for a in state['attempts']],['screen_rejected','accepted'])
        self.assertTrue((self.queue/'attempt-00000/screen.log').exists())
        self.assertEqual(state['accepted_count'],1)

    def test_changed_auditor_recipe_or_modified_plan_blocks_resume(self):
        with patch.object(collector,'execute_phase',side_effect=RuntimeError('unit-test stop before subprocess')):
            collector.collect(self.queue,**self.inputs)
        state=collector.read(self.queue/'queue.json')
        self.assertIn('collect_handover_quality100.py',' '.join(state['recipe']['code_sha256']))
        frozen=self.queue/'frozen_recipe.json';frozen.write_text(frozen.read_text()+' ')
        with self.assertRaisesRegex(ValueError,'recipe changed'):collector.verified_queue(self.queue)


if __name__=='__main__':unittest.main()
