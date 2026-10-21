"""CPU audit fixtures; synthetic rows never authorize physical collection/training."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/"VLMExpert/third_party/SIMPLE/src"))
import prepare_liftbox_prefix_v2 as packing
import collect_liftbox_prefix_v2 as collection


def event_fixture(count=60):
    rows=[]
    for index in range(count):
        tick=collection.native_tick(index); teacher=index>=30
        source=tick-50 if teacher else None
        rows.append({"event_index":index,"observation_hz":30,"native_frame_index":tick,
            "source_episode":0,"phase":"oracle_teacher_token" if teacher else "warmup_observation_prefix",
            "action_valid":teacher,"source_ref_frame":source,
            "teacher_source_time_s":source/50 if teacher else None,
            "observation_phase":packing.common.OBSERVATION_PHASE,"sim_time_s":.005+tick/50,
            "collection_elapsed_s":tick/50,"nominal_timestamp_s":index/30,
            "state43":[0.]*43,"base_pose7":[0,0,.8,1,0,0,0],"box_pose7":[0,0,.217,1,0,0,0],
            "executed_token64":[.25]*64 if teacher else None,"hand_targets14":[.5]*14 if teacher else None,
            "command_hz":30 if teacher else None,
            "runtime_encoder":{"mode":"feedback-heading-v11","source_frame_index":source,
                "changed_input_slice":[584,644],"encoder_input_sha256":"a"*64} if teacher else None})
    return rows


def clock_fixture():
    events=event_fixture();trace=[]
    for index in range(100):
        plan=collection.tick_plan(index); event=events[plan["event_index"]]
        trace.append({"native_frame_index":index,"phase":plan["phase"],"source_ref_frame":plan["source_ref_frame"],
            "teacher_source_time_s":plan["teacher_source_time_s"],
            "observation_event_index":plan["event_index"] if plan["is_event"] else None,
            "command_updated":plan["is_event"] and plan["action_valid"],
            "observation_phase":packing.common.OBSERVATION_PHASE,
            "sim_time_s":.005+index/50,"next_sim_time_s":.005+(index+1)/50,
            "decoder_calls_before_step":max(0,index-10),"decoder_calls_after_step":max(0,index-9),
            "applied_body_token":list(event["executed_token64"]) if index>=50 else None,
            "applied_hand_targets":dict(zip(packing.TRAIN_HAND_NAMES,[.5]*14)),
            "runtime_encoder":event["runtime_encoder"] if index>=50 and plan["is_event"] else None})
    trajectory={"sim_time_s":.005+np.arange(101)/50,"native_frame_index":np.arange(101),
        "phase":np.array(["warmup_observation_prefix"]*50+["oracle_teacher_token"]*50+["task_terminal"])}
    result={"initial_sim_time_s":.005,"teacher_start_sim_time_s":1.005,"actual_sim_seconds":2.,"decoder_calls":90}
    return trajectory,trace,events,result


class PrefixPackContracts(unittest.TestCase):
    def test_real_symlink_entry_is_owned_but_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);physical=base/"physical";physical.mkdir()
            artifact=physical/"model.mjb";artifact.write_bytes(b"fixture")
            alias=base/"runtime";alias.symlink_to(physical,target_is_directory=True)
            self.assertEqual(packing.declared_file(physical,alias/"model.mjb"),artifact)
            self.assertEqual(packing.declared_file(alias,physical/"model.mjb"),artifact)
            self.assertTrue(packing.same_directory(str(alias),physical))
            outside=base/"outside.mjb";outside.write_bytes(b"outside")
            (physical/"escape.mjb").symlink_to(outside)
            with self.assertRaisesRegex(ValueError,"escapes"):
                packing.declared_file(alias,alias/"escape.mjb")

    def test_first_current_has_real_prefix_and_complete_teacher_future(self):
        events=event_fixture()
        self.assertEqual(packing.validate_events(events),0)
        self.assertEqual(events[30]["native_frame_index"],50)
        self.assertEqual(events[30]["source_ref_frame"],0)
        self.assertFalse(collection.eligible_window(events,29))
        self.assertTrue(collection.eligible_window(events,30))
        np.testing.assert_allclose([events[i]["sim_time_s"] for i in [0,6,12,18,24,30]],
                                   [.005,.205,.405,.605,.805,1.005],rtol=0,atol=1e-12)

    def test_fake_prefix_labels_retiming_missing_future_and_nan_rejected(self):
        for kind in ("prefix_label","valid_prefix","teacher_shift","floor_clock","nan","missing_future"):
            events=event_fixture()
            if kind=="prefix_label":events[0]["executed_token64"]=[0.]*64
            elif kind=="valid_prefix":events[0]["action_valid"]=True
            elif kind=="teacher_shift":events[30]["source_ref_frame"]=50
            elif kind=="floor_clock":events[1]["native_frame_index"]=1
            elif kind=="nan":events[8]["sim_time_s"]=float("nan")
            else:events.pop()
            with self.subTest(kind=kind),self.assertRaises(ValueError):packing.validate_events(events)

    def test_placeholders_are_explicit_invalid_and_never_enter_action_stats(self):
        values=packing.episode_values(event_fixture(),0,0)
        np.testing.assert_array_equal(values["action.body_token_v1_1"][:30],0)
        np.testing.assert_array_equal(values["action"][:30],0)
        np.testing.assert_array_equal(values["action_valid"],[False]*30+[True]*30)
        np.testing.assert_array_equal(values["source_reference_frame_index"][:30],-1)
        self.assertEqual(values["source_reference_frame_index"][30],0)
        stats=packing.episode_stats(values)
        self.assertEqual(stats["action.body_token_v1_1"]["count"],[30])
        self.assertEqual(stats["action.body_token_v1_1"]["min"],[.25]*64)
        self.assertEqual(stats["action"]["min"],[.5]*14)
        self.assertEqual(stats["observation.state"]["count"],[60])

    def test_50hz_decoder_clock_and_actual_zoh_are_not_30hz_subsampling(self):
        trajectory,trace,events,result=clock_fixture()
        packing.audit_clock(trajectory,trace,events,result)
        for kind in ("gap","reset","terminal","zoh","regen"):
            t,r,e,s=copy.deepcopy((trajectory,trace,events,result))
            if kind=="gap":t["sim_time_s"][20]+=.005
            elif kind=="reset":r[50]["decoder_calls_before_step"]=0
            elif kind=="terminal":t["phase"][-1]="oracle_teacher_token"
            elif kind=="zoh":r[51]["applied_body_token"][0]=.5
            else:r[51]["runtime_encoder"]={"new":"token"}
            with self.subTest(kind=kind),self.assertRaises(ValueError):packing.audit_clock(t,r,e,s)

    def test_fixed_five_sources_compare_semantics_not_per_run_paths(self):
        audited={i:([],[],{"semantic_contract":{"controller":"same"},"frozen_contract":{"sha256":str(i)}}) for i in range(5)}
        self.assertEqual(packing.validate_pilot(audited),{"train":[0,1,2],"val":[3],"test":[4]})
        audited[4][2]["semantic_contract"]={"controller":"changed"}
        with self.assertRaises(ValueError):packing.validate_pilot(audited)
        audited.pop(4)
        with self.assertRaises(ValueError):packing.validate_pilot(audited)

    def test_curated_train016_val3_test9_keep_actual_disjoint_source_ids(self):
        audited={i:([],[],{"semantic_contract":{"controller":"same"}}) for i in (0,1,6,3,9)}
        self.assertEqual(packing.validate_pilot(audited,[0,1,6],9),{"train":[0,1,6],"val":[3],"test":[9]})
        for train in ([0,1,1],[0,1,3],[0,1,4],[0,1,8],[0,1,9],[0,True,6],[0,1,12]):
            with self.subTest(train=train), self.assertRaises(ValueError):packing.train_sources(train)
        for heldout in (3,5,8,12,True):
            with self.subTest(test=heldout), self.assertRaises(ValueError):packing.test_source(heldout)

    def test_prospective_candidate_order_requires_real_prior_physical_failures(self):
        facts=[];attempts=[]
        for source,success in ((2,False),(5,False),(6,True),(4,False),(9,True)):
            attempt=f"case{source}";attempts.append({"attempt_id":attempt,"source_episode":source,"phase":"started"})
            attempts.append({"attempt_id":attempt,"source_episode":source,"phase":"finished"})
            facts.append({"source_episode":source,"attempt_id":attempt,"finished_record_index":len(attempts)-1,
                "task_success_observed":success,"protocol_comparable":True,
                "status":"task_terminated" if success else "source_end",
                "physical_outcome_status":"task_terminated" if success else "source_end",
                "trajectory_sha256":"a"*64,"trace_sha256":"b"*64})
        ledger=(attempts,facts)
        train=packing.train_curation({},[0,1,6],ledger=ledger)
        test=packing.test_curation({},9,ledger=ledger)
        self.assertEqual([r["source_episode"] for r in train["failed_candidates_retained"]],[2,5])
        self.assertEqual([r["source_episode"] for r in test["failed_candidates_retained"]],[4])
        self.assertEqual(test["required_additional_physical_before_after_source_episodes"],[4,8])
        for corruption in ("missing","earlier_success","infra_only","out_of_order","hidden_failure"):
            report=copy.deepcopy(train)
            if corruption=="missing":report["preserved_attempts"].pop(1)
            elif corruption=="earlier_success":report["preserved_attempts"][1]["task_success_observed"]=True
            elif corruption=="infra_only":report["preserved_attempts"][1]["physical_outcome_status"]="error"
            elif corruption=="out_of_order":report["preserved_attempts"][1]["finished_record_index"]=9
            else:report["failed_candidates_retained"]=[]
            with self.subTest(corruption=corruption),self.assertRaises(ValueError):packing.validate_curation(report,[0,1,6])
        changed=copy.deepcopy(test);changed["preserved_attempts"][3]["task_success_observed"]=True
        with self.assertRaises(ValueError):packing.validate_test_curation(changed,9)

    def test_retained_numerical_report_requires_exact_verifier_and_full_witness(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);snapshot=root/"verifier.py";snapshot.write_bytes(Path(packing.__file__).read_bytes())
            report=root/"accepted.json"
            receipt={"run_root":str(root),"status":"task_terminated","valid_teacher_rows":30,
                "reconstructed_lift":{"measurement_hz":50,"integration_state_spec":8191,
                    "frames_reconstructed":101,"final_reconstructed_hold_s":1.},
                "warmup_reproduction":{"real_pd_samples":10,"live_encoder_commands":40,"tokens_bitwise_equal":True,"receipt_recomputed":True},
                "encoder_reproduction":{"input_sha256_matches":True,"tokens_bitwise_equal":True,"events_reproduced":30}}
            accepted={"contract":packing.CONTRACT,"status":"single_run_cpu_audited","training_authorized":False,
                      "packer_sha256":packing.common.sha256(snapshot),"event_count":60,"pngs":60,"receipt":receipt}
            report.write_text(json.dumps(accepted))
            identity={"files":{"actual.png":"a"*64},"attempt_facts_sha256":"b"*64}
            with patch.object(packing,"accepted_identity",return_value=(event_fixture(),[],identity)),patch.object(
                    packing,"audit_run",side_effect=AssertionError("Accepted numerics must never run again")):
                self.assertEqual(packing.bind_accepted_run(report,snapshot,expected_identity=identity)[3],identity)
                with self.assertRaisesRegex(ValueError,"identity changed"):
                    packing.bind_accepted_run(report,snapshot,expected_identity={"different":"bytes"})
                bad=copy.deepcopy(accepted);bad["receipt"]["warmup_reproduction"]["receipt_recomputed"]=False
                report.write_text(json.dumps(bad))
                with self.assertRaisesRegex(ValueError,"full completed"):packing.bind_accepted_run(report,snapshot)
                report.write_text(json.dumps(accepted));snapshot.write_bytes(b"changed verifier")
                with self.assertRaisesRegex(ValueError,"snapshot identity"):packing.bind_accepted_run(report,snapshot)
            with self.assertRaises(FileNotFoundError):packing.bind_accepted_run(root/"missing.json",snapshot)

    def test_verify_root_binds_retained_reports_and_never_falls_back_to_numerical_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);events=event_fixture();episodes=[];audited={}
            for source in range(5):
                audit={"source_episode":source,"semantic_contract":{"controller":"same"}}
                report=root/f"accepted{source}.json";report.write_text("CPU fixture")
                snapshot=root/f"verifier{source}.py";snapshot.write_text("CPU fixture")
                split="train" if source<3 else ("val" if source==3 else "test")
                episodes.append({"split":split,"episode_index":source if source<3 else 0,"source_episode_index":source,
                    "length":60,"audit":audit,"accepted_numerical":{"report":report.name,"report_sha256":packing.common.sha256(report),
                    "verifier_snapshot":snapshot.name,"verifier_sha256":packing.common.sha256(snapshot),
                    "binder_sha256":packing.common.sha256(packing.__file__),"identity":{"fixture":source}}})
                audited[source]=(events,[],audit)
            ledger=([],[])
            summary={"contract":packing.CONTRACT,"task_id":packing.lift.TASK_ID,"instruction":packing.lift.INSTRUCTION,
                "prefix_rows":30,"training_authorized":False,"packer_sha256":packing.common.sha256(packing.__file__),
                "splits":packing.SPLITS,"episodes":episodes,"packed_files":{},"attempt_log_snapshots":[],
                "train_demo_curation":packing.train_curation({},[0,1,2],ledger=ledger),
                "test_demo_curation":packing.test_curation({},4,ledger=ledger)}
            ready=root/"LIFT_COMMANDED_ROLLOUT_READY.json";ready.write_text(json.dumps(summary))
            def bind(report,snapshot,*,expected_identity):
                source=int(report.stem.removeprefix("accepted"))
                self.assertEqual(expected_identity,{"fixture":source})
                return (*audited[source],expected_identity)
            with patch.object(packing,"audit_run",side_effect=AssertionError("No force/encoder rerun")),patch.object(
                    packing,"bind_accepted_run",side_effect=bind) as binder,patch.object(packing,"verify_packed_episode") as transformed,patch.object(
                    packing,"curation_facts",return_value=ledger):
                self.assertEqual(packing.verify_pack(root),summary)
                self.assertEqual(binder.call_count,5);self.assertEqual(transformed.call_count,5)
                (root/"accepted0.json").unlink()
                with self.assertRaises(FileNotFoundError):packing.verify_pack(root)

    def test_partial_pack_preserves_each_accepted_run_before_later_numerics_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);runs=[];base=root/"base";base.mkdir();source=root/"protected";source.mkdir()
            (base/"dataset_statistics.json").write_text("accepted normalization bytes")
            (source/"dataset_statistics.json").write_text("accepted normalization bytes")
            (base/"LOCAL_RUN_READY.json").write_text(json.dumps({"status":"local_joint_run_prepared",
                "optimizer_rng_restored":False,"source_snapshot":str(source),"joint_sha256":"a"*64}))
            for index in range(5):
                run=root/f"run{index}";run.mkdir();runs.append(run)
                (run/"result.json").write_text(json.dumps({"teacher":{"source_episode_index":index},"attempt_log":{"path":"fixture"}}))
            events=event_fixture();audit={"source_episode":0,"run_root":str(runs[0]),"semantic_contract":{"same":True}}
            with patch.object(packing,"validate_result",return_value={"same":True}),patch.object(
                    packing,"curation_facts",return_value=([],[])),patch.object(packing,"audit_run",side_effect=[
                    (events,[root/"PNG fixture"]*60,audit),ValueError("new source failed actual numerical audit")]),patch.object(
                    packing,"accepted_identity",return_value=(events,[],{"files":{}})):
                output=root/"newpack"
                with self.assertRaisesRegex(ValueError,"new source failed"):packing.pack(runs,output,base)
                accepted=packing.read(output/"provenance/source0_accepted_numerics.json")
                self.assertEqual(accepted["status"],"single_run_cpu_audited")
                self.assertEqual(accepted["receipt"],audit)
                self.assertEqual(packing.common.sha256(output/"provenance/source0_accepted_verifier.py"),accepted["packer_sha256"])
                self.assertFalse((output/"LIFT_COMMANDED_ROLLOUT_READY.json").exists())

    def test_old_v1_status_or_changed_success_claim_cannot_unlock_v2(self):
        for result in ({"contract":"cvpr-simple-lift-box-hold-commanded-rollout-v1"},
                       {"contract":packing.CONTRACT,"status":"source_end","task_success_observed":True}):
            with self.assertRaisesRegex(ValueError,"newly executed"):packing.validate_result(Path("/tmp"),result)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"LIFT_COMMANDED_ROLLOUT_READY.json"
            path.write_text(json.dumps({"contract":"cvpr-simple-lift-box-hold-commanded-rollout-v1"}))
            with self.assertRaisesRegex(ValueError,"real-prefix-v2"):packing.verify_pack(path.parent)

    def test_real_png_parquet_and_lossless_video_roundtrip_and_tamper_rejection(self):
        import pyarrow.parquet as pq
        import pyarrow as pa
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);events=event_fixture();pngs=[]
            for index,row in enumerate(events):
                rgb=np.full((16,16,3),index,np.uint8);path=root/f"frame{index}.png";Image.fromarray(rgb).save(path)
                row["rgb_sha256"]=packing.common.raw_rgb_sha(rgb);pngs.append(path)
            entry,stats,receipt=packing.pack_episode(root/"train",0,0,events,pngs,
                {"source_episode":0,"instruction":packing.lift.INSTRUCTION})
            receipt["split"]="train"
            (root/"train/meta").mkdir()
            (root/"train/meta/episodes_stats.jsonl").write_text(json.dumps(stats)+"\n")
            packing.verify_packed_episode(root,receipt,events)
            self.assertEqual(entry["length"],60)
            path=root/"train/data/chunk-000/episode_000000.parquet"
            table=pq.read_table(path);index=table.column_names.index("action_valid")
            table=table.set_column(index,"action_valid",pa.array([True]*60));pq.write_table(table,path)
            with self.assertRaisesRegex(ValueError,"packed action_valid"):packing.verify_packed_episode(root,receipt,events)

    def test_reconstruct_requires_every548_integration_state_and_true_loaded_hold(self):
        import mujoco
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            bodies="".join(f'<body><joint name="{name}"/><geom type="sphere" size=".001" mass=".001" contype="0" conaffinity="0"/></body>' for name in packing.TRAIN_STATE_NAMES)
            motors="".join(f'<motor joint="{name}"/>' for name in packing.TRAIN_STATE_NAMES)
            xml='<mujoco><option timestep=".005"/><worldbody><body name="pelvis"><freejoint/><geom type="sphere" size=".01" mass="1" contype="0" conaffinity="0"/>'+bodies+'</body><body name="left_hand_thumb_0_link" pos="0 0 .15"><geom type="box" size=".3 .3 .05"/></body><body name="cube"><freejoint name="cube_joint"/><geom type="box" size=".1 .1 .1" mass="1"/></body></worldbody><actuator>'+motors+'</actuator></mujoco>'
            model=mujoco.MjModel.from_xml_string(xml);data=mujoco.MjData(model)
            data.qpos[:7]=[0,0,.8,1,0,0,0];data.joint("cube_joint").qpos[:]=[0,0,.2999,1,0,0,0]
            spec=int(mujoco.mjtState.mjSTATE_INTEGRATION);self.assertEqual(mujoco.mj_stateSize(model,spec),548)
            path=root/"model.mjb";mujoco.mj_saveModel(model,str(path));states=[]
            for time in np.arange(61)/50:
                data.time=time;mujoco.mj_forward(model,data);state=np.empty(548);mujoco.mj_getState(model,data,state,spec);states.append(state)
            trajectory={"sim_time_s":np.arange(61)/50,"physics_state_integration":np.stack(states),"physics_state_spec":np.asarray(spec),
                "phase":np.array(["oracle_teacher_token"]*60+["task_terminal"]),"state43_joint_names":np.array(packing.TRAIN_STATE_NAMES),
                "actual_state43":np.zeros((61,43)),"actual_base_pose":np.tile(data.qpos[:7],(61,1)),
                "actual_box_pose":np.tile(data.joint("cube_joint").qpos,(61,1))}
            physical=packing.lift.reconstruct_lift(path,trajectory,{},[{"next_sim_time_s":1.2}])
            self.assertGreaterEqual(physical["final_reconstructed_hold_s"],1.)
            trajectory["physics_state_integration"][20,0]=np.nan
            with self.assertRaises(ValueError):packing.lift.reconstruct_lift(path,trajectory,{},[{"next_sim_time_s":1.2}])


if __name__=="__main__":unittest.main()
