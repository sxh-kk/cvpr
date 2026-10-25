"""CPU fixtures exercise raw admission; no SDK, model inference or simulation."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import uuid

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import audit_pico_teleop_episode as audit
from pico_sonic_reference import CONTRACT as REFERENCE_CONTRACT, ENCODER_SHA256


def write(path, value):
    Path(path).write_text(json.dumps(value)+"\n")


def write_rows(path, rows):
    Path(path).write_text("".join(json.dumps(row)+"\n" for row in rows))


def raw_packet(timestamp):
    value = {"Input":1,"timeStampNs":timestamp,"appState":{"focus":True},
        "Head":{"pose":".1,1.6,.2,0,0,0,1","status":3},
        "Controller":{side:{"pose":f"{x},1.2,.4,0,0,0,1","trigger":.2,"grip":.1}
                      for side,x in (("left",-.3),("right",.3))},
        "Body":{"joints":[{"p":f"{i*.02},1,.1,0,0,0,1","va":"bad-sdk-evidence","wva":"-3.29E27"}
                            for i in range(24)]}}
    return {"functionName":"Tracking","value":json.dumps(value)}


def renderer(timestamp):
    links = {name:{"mujoco_position_m":[0,0,.75],"mujoco_quaternion_wxyz":[1,0,0,0],
                   "usd":{"valid":True,"path":"/robot/"+name,"position_m":[0,0,.75],"quaternion_wxyz":[1,0,0,0]}}
             for name in audit.RENDERER_LINKS}
    return {"mujoco_sim_time_s":timestamp,"isaac_world_time_s":0.,"links":links,
        "cameras":{"head_stereo_left":{"sensor_position_m":[0,0,1.5],
            "sensor_quaternion_wxyz_usd_axes":[1,0,0,0],"mujoco_position_m":[0,0,1.5],
            "mujoco_rotation_usd_axes":np.eye(3).reshape(-1).tolist(),
            "usd":{"valid":True,"path":"/robot/camera","position_m":[0,0,1.5],"quaternion_wxyz":[1,0,0,0]}}}}


def make_episode(root):
    """Explicit synthetic evidence for validation tests, never an actual admission.

    The fixed-artifact resolver is mocked in each test and the decoder identity
    is patched to the small CPU fixture's real SHA. Production retains pinned
    official hashes and rehashes the real files without model inference.
    """
    (root/"capture/rgb").mkdir(parents=True)
    resources = root/"fixtures"; resources.mkdir()
    decoder = resources/"model_decoder.onnx"; decoder.write_bytes(b"CPU-test-decoder")
    encoder = resources/"model_encoder.onnx"; encoder.write_bytes(b"CPU-test-encoder")
    human = resources/"human_joints_info.pkl"; human.write_bytes(b"CPU-test-human")
    provenance = resources/"PROVENANCE.json"; write(provenance,{"revision":audit.SONIC_REVISION})
    decoder_sha = audit.sha(decoder)
    identities = {"source_sha256":{str(encoder):audit.sha(encoder),str(human):audit.sha(human)},
                  "official_gather":{"source_sha256":{}}}
    compatibility = resources/"binding.json"
    write(compatibility,{"body_token_version":"body_token_v1_1","decoder_sha256":decoder_sha,
        "compatibility_verified":True,"evidence":{"encoder_identity":identities}})
    identity = str(uuid.uuid4())
    metadata = {"human_demo_uuid":identity,"source_kind":"live_pico_human","accepted_new_dataset":False,
        "replay_source_sha256":None,"task_id":audit.TASK,"render_mode":"isaac",
        "physics_hz":200,"decoder_hz":50,"command_rgb_hz":30,"raw_pico_va_wva_used":False,
        "initial_target_z_m":.55,"initialization":{"object_restores_during_control":False,"external_assistance":False},
        "encoder_identity":identities,"controller_identity":{"body_token_version":"body_token_v1_1",
            "decoder_sha256":decoder_sha,"provenance":str(provenance),"compatibility_status":"verified",
            "observation_phase":"post_step_mj_forward_same_qpos","pd_profile":"official_v1_1_ankle_pitch_1_5",
            "compatibility_evidence":str(compatibility)}}
    write(root/"metadata.json",metadata)
    ticks, start = 150, 3.
    qualities = []
    for index in range(ticks):
        phase = index-50
        left,right = ((False,False) if phase<0 else (False,True) if phase<5 else
                      (True,True) if phase<10 else (True,False) if phase<15 else (False,False))
        height = .55 if phase<0 else .7 if phase<15 else .61
        qualities.append({"completed_steps":index+1,"sim_time_s":start+(index+1)/50,
            "actual_target_xyz_m":[0,0,height],"pelvis_z_m":.75,
            "contacts":{"left_hand":left,"right_hand":right,"container":phase>=15},
            "container_geometry":{"inside_container_footprint":True,"signed_margin_m":.02,
                "collision_vertical_ranges_overlap":True,"container_upright_cosine":1.,
                "container_support_weight_fraction":1.,"support_measurement":"live_solved_mujoco_contact_force"},
            "solver_stage":"post_mj_step_before_observation_refresh"})
    write_rows(root/"quality_samples.jsonl",qualities)
    forces = [{"sim_time_s":start+(index+1)/200,"all_zero":True,"qfrc_applied_all_zero":True,
        "xfrc_applied_all_zero":True,"qfrc_applied_nonzero_count":0,"xfrc_applied_nonzero_count":0,
        "qfrc_applied_max_abs":0.,"xfrc_applied_max_abs":0.,"max_abs_qfrc_applied":0.,"max_abs_xfrc_applied":0.}
        for index in range(ticks*4)]
    write_rows(root/"applied_force_samples.jsonl",forces)
    events=[]
    ledger=[{"event_index":0,"event_type":"session","human_demo_uuid":identity,
             "recorded_monotonic_ns":99_000_000_000,"accepted_new_dataset":False,
             "raw_pico_va_wva_validated":False,"metadata":metadata}]
    for index in range((ticks-1)*30//50+1):
        tick=(index*50+29)//30; timestamp=start+tick/50; valid=tick>=50
        pixels=np.full((3,4,3),index,dtype=np.uint8)
        relative=f"rgb/frame_{index:06d}.png"; image=root/"capture"/relative
        Image.fromarray(pixels).save(image)
        token=[index/100]*64; hands={name:.2 for name in audit.MODEL_HAND_NAMES}
        device_timestamp=1_000_000_000+tick*20_000_000
        evidence={"contract":REFERENCE_CONTRACT,"encoder_mode":2,"model_inference_performed":True,
            "encoder_sha256":ENCODER_SHA256,"PICO_raw_va_wva_used":False,"device_timestamp_ns":device_timestamp,
            "token_sha256":hashlib.sha256(np.asarray(token,np.float32).tobytes()).hexdigest(),"identities":identities,
            "reference_horizon":{"kind":"causal_constant_desired_pose_hold","measured_future_frames":0}}
        ledger.append({"event_index":index+1,"event_type":"command","human_demo_uuid":identity,
            "recorded_monotonic_ns":round((100+tick/50+.001)*1e9),"accepted_new_dataset":False,
            "raw_pico_va_wva_validated":False,"physical_tick50":tick,"sim_time_s":timestamp,
            "source_sequence":index+1,"device_timestamp_ns":device_timestamp,"source_host_received_s":100+tick/50,
            "source_raw_packet":raw_packet(device_timestamp),"body_token64":token,
            "named_hand_targets":hands,"reference_evidence":evidence})
        target=qualities[tick-1]["actual_target_xyz_m"] if tick else [0,0,.55]
        events.append({"event_index":index,"capture_index30":index,"physical_tick50":tick,
            "sim_time_s":timestamp,"actual_elapsed_s":tick/50,"nominal_elapsed_s":index/30,
            "phase":"teacher_command" if valid else "warmup","action_valid":valid,
            "source_command_index":tick-50 if valid else None,"joint_names":list(audit.STATE_NAMES),
            "state43":[0.]*43,"physical_joint_q43_float64":[0.]*43,"physical_joint_qvel43":[0.]*43,
            "root_pose7":[0,0,.75,1,0,0,0],"root_qvel6":[0.]*6,
            "objects":{"target":{"pose7":[*target,1,0,0,0]},"container":{"pose7":[0,0,.6,1,0,0,0]}},
            "action":{"body_token64":token,"hand14":[.2]*14,"hand_joint_names":list(audit.MODEL_HAND_NAMES)} if valid else None,
            "renderer_pose_validation":{"contract":audit.RENDERER_POSE_CONTRACT,
                "mujoco_clock_matches_observation_and_event":True,"actual_mujoco_links_and_camera_verified":True,
                "isaac_world_time_fixed":True},"renderer_pose_diagnostics":renderer(timestamp),
            "rgb":{"path":relative,"file_sha256":audit.sha(image),"pixel_sha256":hashlib.sha256(pixels.tobytes()).hexdigest(),
                "width":4,"height":3,"channels":3,"dtype":"uint8"}})
    write_rows(root/"commands.jsonl",ledger); write_rows(root/"capture/events.jsonl",events)
    manifest={**metadata,"contract":audit.CAPTURE_CONTRACT,"status":"raw_capture_completed_pending_independent_audit",
        "events_path":"events.jsonl","events_sha256":audit.sha(root/"capture/events.jsonl"),
        "event_count":len(events),"warmup_event_count":30,"command_event_count":len(events)-30,
        "warmup_seconds":1.,"command_rate_hz":30,"render_engine":"Isaac RTX","joint_names":list(audit.STATE_NAMES),
        "isaac_world_time_reference_s":0.}
    write(root/"capture/capture.json",manifest)
    async_result={**manifest,"async_capture":{"completed":True,"writer_joined":True,"failure":None,
        "enqueued_events":len(events),"written_events":len(events)}}
    write(root/"capture/async_capture.json",async_result)
    outcome={"source_kind":"live_pico_human","accepted_new_dataset":False,
        "state":{"status":"pending_audit","pending_audit":True,"reasons":[],
                 "human_demo_uuid":identity,"local_tick50":ticks-1},
        "capture":async_result,"ledger":{"sha256":audit.sha(root/"commands.jsonl"),"event_count":len(ledger),
                                          "human_demo_uuid":identity}}
    write(root/"episode.json",outcome)
    return identities,decoder_sha


def add_capture_start(root):
    """Produce boundary evidence only; all files remain explicit CPU fixtures."""
    from pico_capture_start import CaptureStartGate
    metadata=json.loads((root/"metadata.json").read_text())
    metadata["capture_start_protocol"]=audit.CAPTURE_START_CONTRACT
    write(root/"metadata.json",metadata)
    ledger=[json.loads(row) for row in (root/"commands.jsonl").read_text().splitlines()]
    ledger[0]["metadata"]=metadata
    command=ledger[1]
    before=audit.parse_tracking(raw_packet(900_000_000),99.9,0)
    after=audit.parse_tracking(command["source_raw_packet"],command["source_host_received_s"],command["source_sequence"])
    clock=iter((99.9,100.))
    state=np.arange(12,dtype=np.float64)
    gate=CaptureStartGate(metadata["human_demo_uuid"],50,3.,400,state,before,clock=lambda:next(clock))
    _,receipt=gate.finish(99.82,after_global_tick=50,after_sim_time=3.,after_substeps=400,
        after_state=state,frame=after,quality=audit.validate_frame(after,now=100.,previous=before),
        event_count=0,ledger_event_count=1,quality_bytes=0,force_bytes=0)
    command["capture_start_receipt"]=receipt
    write_rows(root/"commands.jsonl",ledger)
    write(root/"capture_start_receipt.json",receipt)
    for name in ("capture/capture.json","capture/async_capture.json"):
        manifest=json.loads((root/name).read_text()); manifest.update(metadata); write(root/name,manifest)
    outcome=json.loads((root/"episode.json").read_text())
    outcome["state"]["start_tick50"]=50
    outcome["capture"]=json.loads((root/"capture/async_capture.json").read_text())
    outcome["ledger"]["sha256"]=audit.sha(root/"commands.jsonl")
    outcome["capture_start_receipt"]={"contract":audit.CAPTURE_START_CONTRACT,
        "path":"capture_start_receipt.json","sha256":audit.sha(root/"capture_start_receipt.json")}
    write(root/"episode.json",outcome)
    return receipt


def convert_to_deferred(root, *, asynchronous_preview=False, preview_rgb_absent=False):
    """Archive a real CPU MjModel/full states, without any dynamics or RTX."""
    import mujoco
    from pico_deferred_capture import DeferredStateCapture
    from render_deferred_pico import write_frame_bindings
    class Command:
        def __init__(self,kind,**parameters): self.type,self.parameters=kind,parameters
    metadata=json.loads((root/"metadata.json").read_text())
    metadata.update(render_mode="mujoco_deferred",preview_engine="mujoco_egl",preview_is_training_rgb=False,
        training_rgb_engine="pending_offline_isaac_rtx",object_body_names={"target":"target","container":"container"})
    if asynchronous_preview:
        metadata["preview_protocol"]={"contract":audit.ASYNC_PREVIEW_CONTRACT,"independent_process":True,
            "physics_steps":0,"forward_dynamics_calls":0,"latest_only_feedback":True,
            "training_observation":False,"maximum_frame_delivery_age_s":.1}
    write(root/"metadata.json",metadata)
    events=[json.loads(row) for row in (root/"capture/events.jsonl").read_text().splitlines()]
    ledger=[json.loads(row) for row in (root/"commands.jsonl").read_text().splitlines()]
    ledger[0]["metadata"]=metadata
    write_rows(root/"commands.jsonl",ledger)
    children="".join(f'<body name="joint_link_{i}" pos="0 0 .01"><joint name="{name}"/><geom type="sphere" size=".01" mass=".01"/></body>'
                     for i,name in enumerate(audit.STATE_NAMES))
    model=mujoco.MjModel.from_xml_string('<mujoco><option timestep=".005"/><worldbody>'
        '<body name="pelvis" pos="0 0 .75"><freejoint name="root"/><geom type="sphere" size=".05"/>'+children+'</body>'
        '<body name="target" pos="0 0 .55"><freejoint name="target_root"/><geom type="sphere" size=".02"/></body>'
        '<body name="container" pos="0 0 .6"><freejoint name="container_root"/><geom type="sphere" size=".03"/></body>'
        '</worldbody></mujoco>')
    data=mujoco.MjData(model)
    capture=DeferredStateCapture(root/"deferred_capture",metadata=metadata,joint_names=list(audit.STATE_NAMES),
        warmup_ticks=50,start_time=events[0]["sim_time_s"],model=model)
    spec=mujoco.mjtState.mjSTATE_INTEGRATION
    for index,event in enumerate(events):
        data.time=event["sim_time_s"]
        data.joint("root").qpos[:]=event["root_pose7"]
        for role in ("target","container"):
            data.joint(role+"_root").qpos[:]=event["objects"][role]["pose7"]
        data.qacc_warmstart[:]=np.arange(model.nv)+index
        mujoco.mj_kinematics(model,data); mujoco.mj_camlight(model,data)
        state=np.empty(mujoco.mj_stateSize(model,spec)); mujoco.mj_getState(model,data,state,spec)
        action=event["action"]
        command=None if action is None else Command("sonic_v1_1",controller=object(),
            body_token=np.asarray(action["body_token64"],np.float64),
            hand_targets=dict(zip(action["hand_joint_names"],action["hand14"])))
        row=ledger[index+1]
        fields={key:row[key] for key in ("source_raw_packet","source_host_received_s","source_sequence",
                                       "device_timestamp_ns","reference_evidence")}
        fields["observation_sim_time_s"]=event["sim_time_s"]
        capture.append(tick50=event["physical_tick50"],sim_time_s=event["sim_time_s"],
            observed_joint_names=event["joint_names"],observed_qpos=event["state43"],
            physical_qpos=event["physical_joint_q43_float64"],physical_qvel=event["physical_joint_qvel43"],
            root_pose7=event["root_pose7"],root_qvel6=event["root_qvel6"],objects=event["objects"],
            command=command,integration_state=state,source_fields=fields,
            preview_rgb=None if asynchronous_preview or preview_rgb_absent else np.zeros((3,4,3),np.uint8))
    original=capture.close(completed=True)
    outcome=json.loads((root/"episode.json").read_text())
    outcome.update(capture=original,status="pending_rtx",rtx_ready=False)
    outcome["ledger"]["sha256"]=audit.sha(root/"commands.jsonl")
    write(root/"episode.json",outcome)
    manifest=json.loads((root/"capture/capture.json").read_text())
    manifest.update(metadata)
    manifest.update(status="raw_capture_completed_pending_independent_audit",render_engine="Isaac RTX",
        rgb_capture_mode="deferred_from_exact_recorded_integration_state",observation_domain_feedback="MuJoCo EGL preview",
        training_rgb_domain="Isaac RTX deferred",rgb_was_live_operator_feedback=False,
        deferred_source_manifest_sha256=audit.sha(root/"deferred_capture/capture.json"))
    write(root/"capture/capture.json",manifest)
    (root/"capture/async_capture.json").unlink()
    source_events=[json.loads(line) for line in (root/"deferred_capture/events.jsonl").read_text().splitlines()]
    completion={"contract":"cvpr-pico-deferred-rtx-completion-v1","status":"deferred_rtx_completed_pending_independent_audit",
        "accepted_new_dataset":False,"eligible_for_packing":False,"source_kind":metadata["source_kind"],
        "observation_domain_feedback":"MuJoCo EGL preview","training_rgb_domain":"Isaac RTX deferred",
        "rgb_was_live_operator_feedback":False,"archived_physics_steps":0,"sonic_inferences":0,
        "completed_frames":len(events),"state_event_numeric_checks_passed":True,"environment_binding":{"matched":True},
        "capture":manifest,"capture_manifest_sha256":audit.sha(root/"capture/capture.json"),
        "render_code_sha256":audit.sha(Path(audit.__file__).with_name("render_deferred_pico.py"))}
    for kind,name in (("manifest","capture.json"),("events","events.jsonl"),("model","model.mjb"),("states","states.npy")):
        completion["source_"+kind+"_sha256"]=audit.sha(root/"deferred_capture"/name)
    completion.update(write_frame_bindings(root/"rtx_frame_bindings.jsonl",source_events,root/"capture/events.jsonl"))
    write(root/"rtx_completion.json",completion)


def add_deferred_quality(root):
    from pico_simple_teleop import complete_quality_geometry
    qualities=[json.loads(row) for row in (root/"quality_samples.jsonl").read_text().splitlines()]
    cube=np.asarray([[x,y,z] for x in (-.2,.2) for y in (-.2,.2) for z in (-.1,.1)]).tolist()
    target_cube=(np.asarray(cube)*.1).tolist()
    cache={"format":"rigid-container-collision-vertices-v1","root_local_collision_vertices_m":cube,
        "collision_geom_count":1,"collision_mesh_geom_count":0,"container_root_body_name":"container",
        "target_root_body_name":"target","target_com_offset_local_m":[0,0,0],
        "target_root_local_collision_vertices_m":target_cube,"upright_axis_local":[0,0,1],"upright_reference":"initial"}
    raw=[]
    for row in qualities:
        item={key:value for key,value in row.items() if key!="container_geometry"}
        item.update(target_pose7=[*row["actual_target_xyz_m"],1,0,0,0],container_pose7=[0,0,.6,1,0,0,0],
            target_body_name="target",container_body_name="container",original_live_support={
                "support_measurement":"live_solved_mujoco_contact_force","target_mass_kg":.1,"target_weight_n":.981,
                "container_target_contact_count":1,"container_force_on_target_world_n":[0,0,.981],
                "container_upward_support_force_n":.981,"container_support_weight_fraction":1.})
        raw.append(item)
    write(root/"container_geometry_cache.json",cache)
    write_rows(root/"raw_quality_samples.jsonl",raw)
    write_rows(root/"quality_samples.jsonl",complete_quality_geometry(raw,cache))
    metadata=json.loads((root/"metadata.json").read_text())
    metadata.update(quality_geometry_mode="deferred_geometry_with_original_solved_support",
                    container_geometry_cache_sha256=audit.sha(root/"container_geometry_cache.json"))
    write(root/"metadata.json",metadata)
    for name in ("capture/capture.json","capture/async_capture.json"):
        if (root/name).exists():
            value=json.loads((root/name).read_text()); value.update(metadata); write(root/name,value)
    ledger=[json.loads(row) for row in (root/"commands.jsonl").read_text().splitlines()]
    ledger[0]["metadata"]=metadata; write_rows(root/"commands.jsonl",ledger)
    outcome=json.loads((root/"episode.json").read_text())
    outcome["ledger"]["sha256"]=audit.sha(root/"commands.jsonl")
    outcome["quality_geometry_sources"]={"raw_samples_sha256":audit.sha(root/"raw_quality_samples.jsonl"),
        "cache_sha256":audit.sha(root/"container_geometry_cache.json"),
        "completed_samples_sha256":audit.sha(root/"quality_samples.jsonl")}
    if (root/"capture/async_capture.json").exists():
        outcome["capture"]=json.loads((root/"capture/async_capture.json").read_text())
    write(root/"episode.json",outcome)


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)
        self.identities,self.decoder_sha=make_episode(self.root)
        self.patches=[patch.object(audit,"DECODER_SHA256",self.decoder_sha),
                      patch.object(audit,"verify_runtime_sources",return_value=self.identities)]
        for item in self.patches: item.start(); self.addCleanup(item.stop)

    def rewrite(self,name,mutate):
        path=self.root/name; value=json.loads(path.read_text()); mutate(value); write(path,value)

    def rewrite_rows(self,name,mutate,refresh=False):
        path=self.root/name; rows=[json.loads(line) for line in path.read_text().splitlines()]
        mutate(rows); write_rows(path,rows)
        if refresh:
            if name=="commands.jsonl":
                self.rewrite("episode.json",lambda v:v["ledger"].update(sha256=audit.sha(path),event_count=len(rows)))
            elif name=="capture/events.jsonl":
                for relative in ("capture/capture.json","capture/async_capture.json"):
                    self.rewrite(relative,lambda v:v.update(events_sha256=audit.sha(path)))
                self.rewrite("episode.json",lambda v:v["capture"].update(events_sha256=audit.sha(path)))

    def check_rejected(self):
        result=audit.audit_episode(self.root)
        self.assertFalse(result["eligible_for_packing"]); self.assertFalse(result["accepted_new_dataset"])
        self.assertTrue(result["errors"])
        return " ".join(result["errors"])

    def test_consistent_fixture_only_allows_later_packing(self):
        result=audit.audit_episode(self.root)
        self.assertEqual(result["errors"],[])
        self.assertTrue(result["eligible_for_packing"])
        self.assertFalse(result["accepted_new_dataset"])
        self.assertFalse(result["ckpt_feature_pack_ready"])
        self.assertTrue(result["physical_gate"]["eligible_physics"])
        self.assertEqual(json.loads((self.root/"admission.json").read_text()),result)

    def test_operator_discard_blocks_even_previously_eligible_recording(self):
        write(self.root/'operator_discard.json',{'reason':'operator requested previous episode discard'})
        self.assertIn('Operator discarded',self.check_rejected())

    def test_collection_physics_rejection_blocks_admission(self):
        write(self.root/'collection_record.json',{'number':'002','disposition':'discarded',
            'reasons':['未完成交接']})
        self.assertIn('collection rejected',self.check_rejected())

    def test_background_pending_is_not_eligible_before_numbered_review(self):
        self.rewrite('episode.json',lambda value:value.update(geometry_pending=True))
        self.assertIn('队列登记标记',self.check_rejected())

    def test_completed_background_geometry_uses_bound_receipt_without_rewriting_outcome(self):
        add_deferred_quality(self.root)
        path=self.root/'episode.json'
        outcome=json.loads(path.read_text())
        sources=outcome.pop('quality_geometry_sources')
        outcome['geometry_pending']=True
        outcome['quality_geometry_expected_cache_sha256']=sources['cache_sha256']
        write(path,outcome)
        sealed=audit.sha(path)
        metadata=json.loads((self.root/'metadata.json').read_text())
        spool=self.root/'.review'
        (spool/'discard_requests').mkdir(parents=True)
        write(self.root/'review_queue_submission.json',{
            'contract':'cvpr-pico-review-submission-v1','uuid':metadata['human_demo_uuid'],
            'episode_json_sha256':sealed,'sequence':1,'collection_root':str(self.root),
            'spool_directory':str(spool),'collection_enabled':True,'accepted_new_dataset':False})
        write(self.root/'quality_geometry_completion.json',{
            'contract':'cvpr-pico-quality-geometry-completion-v1','status':'completed',
            'episode_json_sha256':sealed,'metadata_sha256':audit.sha(self.root/'metadata.json'),
            'human_demo_uuid':metadata['human_demo_uuid'],'source_kind':metadata['source_kind'],
            'quality_geometry_sources':sources,'accepted_new_dataset':False})
        write(self.root/'collection_record.json',{'status':'committed','disposition':'kept'})
        result=audit.audit_episode(self.root)
        self.assertTrue(result['eligible_for_packing'],result['errors'])
        self.assertEqual(audit.sha(path),sealed)
        write(spool/'discard_requests'/(metadata['human_demo_uuid']+'.json'),{
            'uuid':metadata['human_demo_uuid'],'operator_discard_requested':True})
        self.assertIn('Operator discard is pending',self.check_rejected())

    def test_recorded_replay_never_qualifies_despite_successful_physics(self):
        self.rewrite("metadata.json",lambda v:v.update(source_kind="recorded_pico_diagnostic"))
        self.assertIn("never eligible",self.check_rejected())

    def test_missing_body_or_input_mode_cannot_be_replaced_with_cached_state(self):
        def bad(rows):
            value=json.loads(rows[10]["source_raw_packet"]["value"]); value.pop("Body")
            rows[10]["source_raw_packet"]["value"]=json.dumps(value)
        self.rewrite_rows("commands.jsonl",bad,refresh=True)
        self.assertIn("body_missing",self.check_rejected())

    def test_source_clock_gap_is_rejected(self):
        self.rewrite_rows("commands.jsonl",lambda rows:rows[10].update(device_timestamp_ns=rows[9]["device_timestamp_ns"]),refresh=True)
        self.assertIn("Raw/source device time differs",self.check_rejected())

    def test_unfocused_raw_packet_is_not_rescued_by_valid_encoder_evidence(self):
        def bad(rows):
            value=json.loads(rows[10]["source_raw_packet"]["value"])
            value["appState"]["focus"]=False
            rows[10]["source_raw_packet"]["value"]=json.dumps(value)
        self.rewrite_rows("commands.jsonl",bad,refresh=True)
        self.assertIn("app_not_focused",self.check_rejected())

    def test_actual_local_decoder_mutation_is_rejected(self):
        (self.root/"fixtures/model_decoder.onnx").write_bytes(b"changed")
        self.assertIn("Actual local decoder SHA changed",self.check_rejected())

    def test_different_checkout_paths_keep_recorded_identity_for_compatibility_and_commands(self):
        current=self.root/'current-checkout'; current.mkdir()
        checked=deepcopy(self.identities)
        checked['source_sha256']={}
        for original,digest in self.identities['source_sha256'].items():
            path=current/Path(original).name; path.write_bytes(Path(original).read_bytes())
            checked['source_sha256'][str(path)]=digest
        self.assertNotEqual(checked,self.identities)
        with patch.object(audit,'verify_runtime_sources',return_value=checked):
            result=audit.audit_episode(self.root)
        self.assertTrue(result['eligible_for_packing'],result['errors'])
        for original,digest in self.identities['source_sha256'].items():
            self.assertEqual(result['bindings'][original],digest)
        # A receipt from a different producer is still wrong even if its
        # corresponding files have the same pinned bytes in another checkout.
        self.rewrite('fixtures/binding.json',lambda v:v['evidence'].update(encoder_identity=checked))
        with patch.object(audit,'verify_runtime_sources',return_value=checked):
            self.assertIn('producer binding is incomplete',self.check_rejected())

    def test_stale_initial_height_cannot_exaggerate_measured_lift(self):
        self.rewrite("metadata.json",lambda v:v.update(initial_target_z_m=.1))
        self.assertIn("Initial target height",self.check_rejected())

    def test_capture_initial_height_must_match_actual_first_frame(self):
        self.rewrite("capture/capture.json",lambda v:v.update(initial_target_z_m=.1))
        self.assertIn("Initial target height",self.check_rejected())

    def test_initial_height_cannot_be_boolean_or_nonfinite(self):
        self.rewrite("metadata.json",lambda v:v.update(initial_target_z_m=True))
        self.assertIn("Initial target height",self.check_rejected())

    def test_missing_rgb_frame_and_mutated_actual_action_are_rejected(self):
        (self.root/"capture/rgb/frame_000010.png").unlink()
        self.assertIn("Missing or escaped",self.check_rejected())

    def test_captured_action_must_match_the_actual_logged_token(self):
        self.rewrite_rows("capture/events.jsonl",lambda rows:rows[-1]["action"]["body_token64"].__setitem__(0,9),refresh=True)
        self.assertIn("actual logged command",self.check_rejected())

    def test_missing_physical_or_force_samples_are_not_interpolated(self):
        self.rewrite_rows("applied_force_samples.jsonl",lambda rows:rows.pop())
        self.assertIn("Missing original200Hz",self.check_rejected())

    def test_nonzero_force_and_unreleased_hand_reject(self):
        self.rewrite_rows("applied_force_samples.jsonl",lambda rows:rows[-1].update(qfrc_applied_max_abs=.1))
        self.rewrite_rows("quality_samples.jsonl",lambda rows:rows[-1]["contacts"].update(left_hand=True))
        errors=self.check_rejected()
        self.assertIn("gate failed",errors); self.assertIn("External force evidence",errors)

    def test_warmup_cannot_hide_the_grasp_or_transfer(self):
        self.rewrite_rows("quality_samples.jsonl",lambda rows:rows[20]["contacts"].update(right_hand=True))
        self.assertIn("unsupervised warmup",self.check_rejected())

    def test_incomplete_async_writer_cannot_complete_raw_recording(self):
        self.rewrite("capture/async_capture.json",lambda v:v["async_capture"].update(failure="queue full",completed=False))
        self.assertIn("Async writer dropped",self.check_rejected())

    def test_image_path_cannot_escape_episode_directory(self):
        self.rewrite_rows("capture/events.jsonl",lambda rows:rows[0]["rgb"].update(path="../../elsewhere.png"),refresh=True)
        self.assertIn("Unsafe episode-relative path",self.check_rejected())

    def test_deferred_real_cpu_full_state_bridge_passes_without_async_receipt(self):
        convert_to_deferred(self.root)
        self.assertFalse((self.root/"capture/async_capture.json").exists())
        result=audit.audit_episode(self.root)
        self.assertEqual(result["errors"],[])
        self.assertTrue(result["eligible_for_packing"])
        self.assertTrue(result["deferred_rtx_bridge"]["cpu_full_integration_state_verified"])
        self.assertEqual(result["observation_domain_feedback"],"MuJoCo EGL preview")
        self.assertEqual(result["training_rgb_domain"],"Isaac RTX deferred")
        self.assertFalse(result["rgb_was_live_operator_feedback"])
        self.assertFalse(result["accepted_new_dataset"])

    def test_deferred_incomplete_raw_archive_cannot_be_promoted_by_rtx_receipt(self):
        convert_to_deferred(self.root)
        self.rewrite("deferred_capture/capture.json",lambda v:v.update(status="partial_capture_rejected"))
        self.assertIn("Original physical capture was incomplete",self.check_rejected())

    def test_synthetic_deferred_bridge_can_pass_but_human_admission_still_rejects(self):
        kind="synthetic_neutral_diagnostic"
        self.rewrite("metadata.json",lambda v:v.update(source_kind=kind))
        self.rewrite("episode.json",lambda v:v.update(source_kind=kind))
        convert_to_deferred(self.root)
        result=audit.audit_episode(self.root)
        self.assertTrue(result["checks"]["original_full_state_to_deferred_rtx_bridge"]["passed"])
        self.assertEqual(result["deferred_rtx_bridge"]["source_kind"],kind)
        self.assertFalse(result["checks"]["live_source_and_rates"]["passed"])
        self.assertFalse(result["eligible_for_packing"])
        self.assertFalse(result["accepted_new_dataset"])

    def test_deferred_saved_full_state_mutation_is_independently_rejected(self):
        convert_to_deferred(self.root)
        path=self.root/"deferred_capture/states.npy"
        matrix=np.load(path,allow_pickle=False).copy(); matrix[0,-1]+=.1
        np.save(path,matrix,allow_pickle=False)
        self.assertIn("states archive SHA256 mismatch",self.check_rejected())

    def test_deferred_final_action_cannot_change_even_if_rendered_hashes_are_refreshed(self):
        convert_to_deferred(self.root)
        path=self.root/"capture/events.jsonl"
        events=[json.loads(row) for row in path.read_text().splitlines()]
        events[-1]["action"]["body_token64"][0]+=1.
        write_rows(path,events)
        manifest=json.loads((self.root/"capture/capture.json").read_text())
        manifest["events_sha256"]=audit.sha(path); write(self.root/"capture/capture.json",manifest)
        rows=[json.loads(row) for row in (self.root/"rtx_frame_bindings.jsonl").read_text().splitlines()]
        rows[-1]["rtx_event_sha256"]=audit.canonical_event_sha256(events[-1])
        write_rows(self.root/"rtx_frame_bindings.jsonl",rows)
        self.rewrite("rtx_completion.json",lambda v:v.update(capture=manifest,
            capture_manifest_sha256=audit.sha(self.root/"capture/capture.json"),
            frame_bindings_sha256=audit.sha(self.root/"rtx_frame_bindings.jsonl")))
        self.assertIn("RTX changed original physical/state/action",self.check_rejected())

    def test_deferred_frame_bridge_cannot_claim_modified_integration_state(self):
        convert_to_deferred(self.root)
        self.rewrite("rtx_completion.json",lambda v:v.update(full_integration_state_unchanged=False))
        self.assertIn("zero-step same-scene",self.check_rejected())

    def test_deferred_relabelled_feedback_cannot_be_admitted(self):
        convert_to_deferred(self.root)
        self.rewrite("rtx_completion.json",lambda v:v.update(rgb_was_live_operator_feedback=True))
        self.assertIn("misrepresented as live",self.check_rejected())

    def test_deferred_quality_geometry_recomputes_original_force_without_dynamics(self):
        add_deferred_quality(self.root)
        convert_to_deferred(self.root)
        import mujoco
        with patch.object(mujoco,"mj_step",side_effect=AssertionError("No physics integration")), \
             patch.object(mujoco,"mj_forward",side_effect=AssertionError("No solver warmstart overwrite")):
            result=audit.audit_episode(self.root)
        self.assertEqual(result["errors"],[])
        self.assertTrue(result["eligible_for_packing"])
        self.assertTrue(result["deferred_quality_geometry"]["original_solved_force_retained"])

    def test_deferred_quality_force_cannot_be_replaced_in_completed_geometry(self):
        add_deferred_quality(self.root)
        self.rewrite_rows("quality_samples.jsonl",lambda rows:rows[-1]["container_geometry"].update(container_support_weight_fraction=999.))
        self.rewrite("episode.json",lambda v:v["quality_geometry_sources"].update(
            completed_samples_sha256=audit.sha(self.root/"quality_samples.jsonl")))
        self.assertIn("Deferred geometry differs",self.check_rejected())

    def test_deferred_quality_original_raw_or_cache_sha_cannot_change(self):
        add_deferred_quality(self.root)
        self.rewrite("container_geometry_cache.json",lambda v:v.update(collision_geom_count=2))
        self.assertIn("immutable geometry cache SHA changed",self.check_rejected())

    def test_capture_start_independently_binds_file_first_command_and_preserved_lag(self):
        receipt=add_capture_start(self.root)
        result=audit.audit_episode(self.root)
        self.assertTrue(result["eligible_for_packing"],result["errors"])
        boundary=result["capture_start_boundary"]
        self.assertTrue(boundary["endpoint_packets_independently_revalidated"])
        self.assertAlmostEqual(boundary["preserved_lateness_s"],.08)
        self.assertFalse(boundary["all_intermediate_sdk_packets_inspected"])
        self.assertFalse(boundary["first_full_integration_state_independently_verified"])
        self.assertEqual(result["bindings"][str(self.root/"capture_start_receipt.json")],
                         audit.sha(self.root/"capture_start_receipt.json"))
        self.rewrite("capture_start_receipt.json",lambda v:v.update(single_use=False))
        self.assertIn("receipt SHA changed",self.check_rejected())
        write(self.root/"capture_start_receipt.json",receipt)
        self.rewrite_rows("commands.jsonl",lambda rows:rows[1]["capture_start_receipt"].update(single_use=False),refresh=True)
        self.assertIn("First command does not bind",self.check_rejected())

    def test_capture_start_rejects_clock_exclusion_after_samples_and_nonfrozen_state(self):
        original=add_capture_start(self.root)
        metadata=json.loads((self.root/"metadata.json").read_text())
        outcome=json.loads((self.root/"episode.json").read_text())
        ledger=[json.loads(row) for row in (self.root/"commands.jsonl").read_text().splitlines()]
        events=[json.loads(row) for row in (self.root/"capture/events.jsonl").read_text().splitlines()]
        mutations={
            "overlong setup":lambda v:v.update(setup_finished_monotonic_s=100.11,setup_wall_s=.21),
            "old lag lost":lambda v:v.update(shifted_deadline_monotonic_s=100.,preserved_lateness_s=0.),
            "original budget exceeded":lambda v:v.update(maximum_lateness_s=.101),
            "old lag already invalid":lambda v:v.update(old_deadline_monotonic_s=99.79,old_lateness_s=.11,
                shifted_deadline_monotonic_s=99.89,preserved_lateness_s=.11),
            "started samples":lambda v:v["writer_counts_before_first_capture"].update(event_count=1),
            "boolean sample count":lambda v:v["writer_counts_before_first_capture"].update(event_count=False),
            "missing writer":lambda v:v["writer_counts_before_first_capture"].pop("force_bytes"),
            "physics moved":lambda v:v.update(sim_time_s_after=3.005),
            "tick moved":lambda v:v.update(global_tick50_after=55),
            "solver advanced":lambda v:v.update(substeps_after=404),
            "state changed":lambda v:v.update(integration_state_sha256_after="f"*64),
            "unaligned boundary":lambda v:v.update(global_tick50_before=51,global_tick50_after=51),
            "false callback coverage":lambda v:v.update(all_intermediate_sdk_packets_inspected=True),
            "packet hash changed":lambda v:v["source_after"].update(raw_packet_sha256="f"*64),
            "sequence delta lied":lambda v:v.update(source_sequence_delta=2),
            "device delta lied":lambda v:v.update(source_device_delta_s=.05),
            "quality lied":lambda v:v["source_endpoint_quality"].update(age_s=.01),
        }
        for label,mutate in mutations.items():
            with self.subTest(label=label):
                receipt=deepcopy(original); mutate(receipt)
                write(self.root/"capture_start_receipt.json",receipt)
                data=deepcopy(outcome); rows=deepcopy(ledger); rows[1]["capture_start_receipt"]=receipt
                data["capture_start_receipt"]["sha256"]=audit.sha(self.root/"capture_start_receipt.json")
                bindings={}
                def read(name):
                    path=self.root/name; bindings[str(path)]=audit.sha(path)
                    return json.loads(path.read_text())
                with self.assertRaises(ValueError):
                    audit._check_capture_start(metadata,data,events,rows,read,bindings,self.root)

    def test_capture_start_reparses_endpoint_continuity_even_when_receipt_claims_valid(self):
        add_capture_start(self.root)
        def corrupt(rows):
            receipt=rows[1]["capture_start_receipt"]
            endpoint=receipt["source_after"]
            packet=endpoint["raw_packet"]
            body=json.loads(packet["value"])
            body["Body"]["joints"][0]["p"]="10,1,.1,0,0,0,1"
            packet["value"]=json.dumps(body)
            frame=audit.parse_tracking(packet,endpoint["host_received_s"],endpoint["sequence"])
            endpoint["raw_packet_sha256"]=hashlib.sha256(frame._raw_json.encode()).hexdigest()
            write(self.root/"capture_start_receipt.json",receipt)
        self.rewrite_rows("commands.jsonl",corrupt,refresh=True)
        self.rewrite("episode.json",lambda v:v["capture_start_receipt"].update(
            sha256=audit.sha(self.root/"capture_start_receipt.json")))
        self.assertIn("endpoint tracking continuity",self.check_rejected())

    def test_capture_start_post_setup_packet_must_be_first_command_source(self):
        add_capture_start(self.root)
        self.rewrite_rows("commands.jsonl",lambda rows:rows[1].update(source_sequence=2),refresh=True)
        self.assertIn("First command reused another source",self.check_rejected())

    def test_capture_start_deferred_binds_actual_first_full_integration_state(self):
        add_capture_start(self.root)
        convert_to_deferred(self.root,asynchronous_preview=True)
        first=np.ascontiguousarray(np.load(self.root/"deferred_capture/states.npy",allow_pickle=False)[0],dtype="<f8")
        digest=hashlib.sha256(first.tobytes()).hexdigest()
        def bind_actual(rows):
            receipt=rows[1]["capture_start_receipt"]
            receipt.update(integration_state_sha256_before=digest,integration_state_sha256_after=digest,
                           integration_state_width=int(first.size))
            write(self.root/"capture_start_receipt.json",receipt)
        self.rewrite_rows("commands.jsonl",bind_actual,refresh=True)
        self.rewrite("episode.json",lambda v:v["capture_start_receipt"].update(
            sha256=audit.sha(self.root/"capture_start_receipt.json")))
        result=audit.audit_episode(self.root)
        self.assertTrue(result["eligible_for_packing"],result["errors"])
        self.assertTrue(result["capture_start_boundary"]["first_full_integration_state_independently_verified"])
        # A mutually consistent receipt/command hash is insufficient: bind to
        # independently re-read states[0], not the producer's two equal strings.
        def wrong_hash(rows):
            receipt=rows[1]["capture_start_receipt"]
            receipt.update(integration_state_sha256_before="f"*64,integration_state_sha256_after="f"*64)
            write(self.root/"capture_start_receipt.json",receipt)
        self.rewrite_rows("commands.jsonl",wrong_hash,refresh=True)
        self.rewrite("episode.json",lambda v:v["capture_start_receipt"].update(
            sha256=audit.sha(self.root/"capture_start_receipt.json")))
        self.assertIn("First archived full integration state differs",self.check_rejected())

    def test_unknown_capture_start_protocol_cannot_silently_use_legacy_admission(self):
        self.rewrite("metadata.json",lambda v:v.update(capture_start_protocol="unknown"))
        self.assertIn("Unknown capture-start protocol",self.check_rejected())

    def test_asynchronous_preview_none_preserves_full_state_rtx_binding_and_limited_scope(self):
        convert_to_deferred(self.root,asynchronous_preview=True)
        result=audit.audit_episode(self.root)
        self.assertTrue(result["eligible_for_packing"],result["errors"])
        bridge=result["deferred_rtx_bridge"]
        self.assertEqual(bridge["preview_protocol"],audit.ASYNC_PREVIEW_CONTRACT)
        self.assertFalse(bridge["operator_preview_delivery_verified"])
        self.assertFalse(bridge["operator_saw_every_recorded_state_verified"])
        self.assertEqual(bridge["frames_bound"],90)
        self.assertTrue(bridge["cpu_full_integration_state_verified"])

    def test_legacy_preview_none_is_rejected_without_asynchronous_contract(self):
        convert_to_deferred(self.root,preview_rgb_absent=True)
        self.assertIn("Original preview must be recorded explicitly",self.check_rejected())

    def test_asynchronous_preview_contract_does_not_allow_mislabelled_or_dynamic_preview(self):
        base={"contract":audit.ASYNC_PREVIEW_CONTRACT,"independent_process":True,"physics_steps":0,
              "forward_dynamics_calls":0,"latest_only_feedback":True,"training_observation":False}
        for key,value in (("contract","unknown"),("independent_process",False),("physics_steps",1),
                          ("physics_steps",False),("forward_dynamics_calls",1),
                          ("latest_only_feedback",False),("training_observation",True)):
            with self.subTest(key=key,value=value):
                with self.assertRaisesRegex(ValueError,"preview protocol is invalid"):
                    audit._check_preview_protocol({"preview_protocol":{**base,key:value}})
        self.assertFalse(audit._check_preview_protocol({}))
        self.assertTrue(audit._check_preview_protocol({"preview_protocol":base}))


class SourceIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)
        self.checked={'source_sha256':{},'official_gather':{'source_sha256':{},'official_sources_rehashed':True}}
        self.recorded=deepcopy(self.checked)
        for group,roles in ((None,('model_encoder.onnx','human_joints_info.pkl')),
                            ('official_gather',('g1_deploy.cpp','math_utils.hpp'))):
            for role in roles:
                for checkout,identity in (('current',self.checked),('recorded',self.recorded)):
                    path=self.root/checkout/role; path.parent.mkdir(exist_ok=True)
                    path.write_bytes(('pinned-fixture-bytes:'+role).encode())
                    parent=identity if group is None else identity[group]
                    parent['source_sha256'][str(path)]=audit.sha(path)

    def test_same_path_and_different_checkout_both_rehash_original_source_groups(self):
        for supplied in (self.checked,self.recorded):
            with self.subTest(same_path=supplied is self.checked):
                bindings={}
                result=audit._recorded_source_identity(supplied,self.checked,bindings)
                self.assertIs(result,supplied)
                expected={**supplied['source_sha256'],**supplied['official_gather']['source_sha256']}
                self.assertEqual(bindings,expected)

    def test_original_missing_or_changed_cannot_fall_back_to_current_checkout(self):
        path=Path(next(iter(self.recorded['source_sha256'])))
        original=path.read_bytes()
        path.write_bytes(b'mutated-recorded-file')
        with self.assertRaisesRegex(ValueError,'Original recorded source bytes changed'):
            audit._recorded_source_identity(self.recorded,self.checked,{})
        path.write_bytes(original); path.unlink()
        with self.assertRaises(FileNotFoundError):
            audit._recorded_source_identity(self.recorded,self.checked,{})

    def test_fake_sha_missing_duplicate_or_relative_file_role_is_rejected(self):
        for group in (None,'official_gather'):
            for variant in ('fake-sha','missing','duplicate','relative'):
                with self.subTest(group=group,variant=variant):
                    supplied=deepcopy(self.recorded)
                    parent=supplied if group is None else supplied[group]
                    paths=parent['source_sha256']; name=next(iter(paths)); digest=paths[name]
                    if variant=='fake-sha': paths[name]='f'*64
                    if variant=='missing': del paths[name]
                    if variant=='duplicate': paths[str(self.root/'extra'/Path(name).name)]=digest
                    if variant=='relative': del paths[name]; paths[Path(name).name]=digest
                    with self.assertRaises(ValueError):
                        audit._recorded_source_identity(supplied,self.checked,{})

    def test_all_nonpath_metadata_and_scalar_types_remain_exact(self):
        variants=[lambda v:v.update(unrecognized_flag=True),
                  lambda v:v['official_gather'].update(unrecognized_flag=True),
                  lambda v:v['official_gather'].update(official_sources_rehashed=False),
                  lambda v:v['official_gather'].update(official_sources_rehashed=1),
                  lambda v:v['official_gather'].pop('official_sources_rehashed')]
        for index,mutate in enumerate(variants):
            with self.subTest(variant=index):
                supplied=deepcopy(self.recorded); mutate(supplied)
                with self.assertRaisesRegex(ValueError,'non-path identity fields differ'):
                    audit._recorded_source_identity(supplied,self.checked,{})


if __name__=="__main__":
    unittest.main()
