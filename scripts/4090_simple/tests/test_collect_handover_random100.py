"""Synthetic queue fixtures; never execute physics, ONNX, GPU or real capture."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import collect_handover_random100 as collector
from test_handover_random100_pack import footprint


def plan_fixture(rounds=2):
    eligible = [n for n in range(100) if n not in collector.packer.EXCLUDED]
    old = eligible[:24]; order = collector.source_order(eligible, old)
    plan = {"contract": collector.PLAN_CONTRACT, "formal": dict(collector.FORMAL), "target": 100,
        "minimum_distinct_sources": 50, "maximum_registered_per_source": 2, "TRAIN_seed_upper_exclusive": 2**30,
        "TRAIN_seed_minimum":1,"additional_dataset_quality":deepcopy(collector.packer.ADDITIONAL_DATASET_QUALITY),
        "source_order": order, "legacy24_source_ids": old, "candidates_per_source": rounds, "master_seed": 13,
        "sources": {str(source): {"source_episode_index": source, "parquet_sha256": f"{source:064x}",
                                  "source_files_sha256":{"/synthetic/raw":"f"*64}} for source in order},
        "code_sha256": {"/synthetic/script": "a"*64},
        "runtime": {"python": "/synthetic/python", "raw_root": "/raw", "data_root": "/assets", "sonic_resources": "/sonic",
                    "latent_projector_batch_view": None, "latent_projection_phases": "placement",
                    "placement_encoder_mode":"g1","placement_locomotion":"source_future"}, "candidates": []}
    for round_index in range(rounds):
        for source in order:
            proposal = {"synthetic_test_only": True, "source": source, "scene_round": round_index//2}
            spec = {"source_episode_index": source, "round": round_index,
                "seed": collector.train_seed(13, source, round_index//2),
                "hand_condition": "declared" if round_index%2==0 else "inferred_effective",
                "formal": dict(collector.FORMAL), "source": plan["sources"][str(source)],
                "additional_dataset_quality":deepcopy(collector.packer.ADDITIONAL_DATASET_QUALITY),
                "code_sha256_digest": collector.digest(plan["code_sha256"]), "scene_proposal": proposal,
                "scene_proposal_sha256": collector.proposal_file_sha(proposal),
                "latent_projector_batch_view": None, "latent_projection_phases": "placement","placement_encoder_mode":"g1",
                "placement_locomotion":"source_future"}
            identity=collector.digest(spec)
            plan["candidates"].append({"candidate_id": f"random100-source{source:03d}-{identity[:24]}",
                "candidate_spec_sha256": identity,"spec":spec})
    return plan


def state_fixture():
    return {"status":"pilot_pending","collection_ready":False,"bootstrap":None,"attempts":[],
            "qualified":[],"registry_versions":[],"diversity_excluded_candidates":[]}


def attempt(candidate, phase="screen", status="passed"):
    return {"candidate_id":candidate["candidate_id"],"source_episode_index":candidate["spec"]["source_episode_index"],
            "phase":phase,"status":status}


def qualified(candidate):
    spec=candidate["spec"]; cid=candidate["candidate_id"]
    return {"candidate_id":cid,"source_episode_index":spec["source_episode_index"],"seed":spec["seed"],
        "hand_condition":spec["hand_condition"],"run":"/synthetic/"+cid,"admission_path":"/synthetic/"+cid+".json",
        "admission_sha256":collector.digest(cid),"scene_proposal_sha256":spec["scene_proposal_sha256"],
        "capture_content_sha256":collector.digest([cid,"pixels"])}


class CollectorTests(unittest.TestCase):
    def test_frozen_94_source_plan_covers_first70_before_legacy24(self):
        plan=plan_fixture();collector.validate_plan(plan)
        self.assertEqual(len(plan["source_order"]),94)
        self.assertTrue(set(plan["source_order"][:70]).isdisjoint(plan["legacy24_source_ids"]))
        self.assertEqual({c["spec"]["source_episode_index"] for c in plan["candidates"][:94]},set(plan["source_order"]))
        self.assertTrue(all(c["spec"]["round"]==0 for c in plan["candidates"][:94]))

    def test_all_seeds_strictly_TRAIN_and_reproducible_without_global_rng(self):
        import random
        before=random.getstate()
        seeds=[collector.train_seed(13,s,r) for s in plan_fixture()["source_order"] for r in range(10)]
        self.assertTrue(all(1<=value<2**30 for value in seeds));self.assertEqual(len(seeds),len(set(seeds)))
        self.assertEqual(seeds,[collector.train_seed(13,s,r) for s in plan_fixture()["source_order"] for r in range(10)])
        self.assertEqual(before,random.getstate())
        for source in (*collector.packer.EXCLUDED,-1,100,True):
            with self.assertRaises(ValueError):collector.train_seed(1,source,0)
        with patch.object(collector,"digest",return_value="0"*64):
            self.assertEqual(collector.train_seed(1,1,0),1)

    def test_hand_conditions_are_explicit_paired_control_for_same_scene(self):
        plan=plan_fixture(4)
        for offset in range(94):
            first,second,third=(plan["candidates"][offset+i*94] for i in range(3))
            self.assertEqual(first["spec"]["hand_condition"],"declared")
            self.assertEqual(second["spec"]["hand_condition"],"inferred_effective")
            self.assertEqual(first["spec"]["seed"],second["spec"]["seed"])
            self.assertNotEqual(first["candidate_id"],second["candidate_id"])
            self.assertNotEqual(first["spec"]["seed"],third["spec"]["seed"])

    def test_tampered_bound_or_source_spec_or_projection_phase_rejected(self):
        for key,value in (("formal",dict(collector.FORMAL,xy_max_m=.02)),("target",99),("maximum_registered_per_source",3)):
            plan=plan_fixture();plan[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):collector.validate_plan(plan)
        plan=plan_fixture();plan["candidates"][0]["spec"]["latent_projection_phases"]="approach"
        with self.assertRaises(ValueError):collector.validate_plan(plan)

    def test_candidate_hash_excludes_execution_output_and_rendering_mode(self):
        candidate=plan_fixture()["candidates"][0]
        self.assertNotIn("output",candidate["spec"]);self.assertNotIn("render_mode",candidate["spec"])
        plan=plan_fixture();runtime=plan["runtime"]
        cpu=collector.execution_command(plan,candidate,Path("/fresh/cpu"),"screen")
        rtx=collector.execution_command(plan,candidate,Path("/fresh/rtx"),"capture")
        for option in ("--source-episode","--seed","--hand-condition","--xy-max-m","--yaw-max-rad","--latent-projection-phases","--placement-encoder-mode","--placement-locomotion"):
            self.assertEqual(cpu[cpu.index(option)+1],rtx[rtx.index(option)+1])
        self.assertEqual(cpu[cpu.index("--render-mode")+1],"none")
        self.assertEqual(rtx[rtx.index("--render-mode")+1],"isaac")

    def test_optional_view_and_filter_propagated_to_real_entrypoint(self):
        plan=plan_fixture();plan["runtime"]["latent_projector_batch_view"]="/view.onnx"
        plan["runtime"]["latent_projection_phases"]="placement,release"
        command=collector.execution_command(plan,plan["candidates"][0],Path("/fresh"),"capture")
        self.assertEqual(command[command.index("--latent-projector-batch-view")+1],"/view.onnx")
        self.assertEqual(command[command.index("--latent-projection-phases")+1],"placement,release")
        plan["runtime"]["placement_encoder_mode"]="teleop"
        command=collector.execution_command(plan,plan["candidates"][0],Path("/fresh"),"capture")
        self.assertEqual(command[command.index("--placement-encoder-mode")+1],"teleop")
        plan["runtime"]["placement_locomotion"]="actual_hold"
        command=collector.execution_command(plan,plan["candidates"][0],Path("/fresh"),"capture")
        self.assertEqual(command[command.index("--placement-locomotion")+1],"actual_hold")

    def test_no_repeat_screen_before_all94_original_sources_attempted(self):
        plan=plan_fixture();state=state_fixture()
        state["attempts"]=[attempt(c,status="rejected") for c in plan["candidates"][:70]]
        pending=collector.choose_screen(plan,state)
        self.assertEqual(len(pending),24*2)
        self.assertTrue(all(c["spec"]["source_episode_index"] in plan["source_order"][70:] for c in pending))
        self.assertTrue(all(c["spec"]["round"]==0 for c in pending[:24]))

    def test_screening_pass_never_enters_qualified_or_registry(self):
        plan=plan_fixture();state=state_fixture();state["attempts"]=[attempt(c) for c in plan["candidates"][:94]]
        report=collector.summary(state,plan)
        self.assertEqual(report["strict_RTX_qualified_pool"],0);self.assertFalse(report["screen_passes_are_new_data"])
        self.assertEqual(collector.select_registry(state["qualified"]),[])

    def test_RTX_repeats_wait_for_min50_and_uncovered_source_opportunities(self):
        plan=plan_fixture(4);state=state_fixture();first=plan["candidates"][:94]
        state["qualified"]=[qualified(c) for c in first[:49]]
        state["attempts"]=[attempt(c,status="passed" if i<49 else "rejected") for i,c in enumerate(first)]
        state["attempts"] += [attempt(c,"capture") for c in first[:49]]
        state["attempts"].append(attempt(plan["candidates"][94]))
        self.assertEqual(collector.choose_capture(plan,state),[])
        state["qualified"].append(qualified(first[49]));state["attempts"].append(attempt(first[49],"capture"))
        self.assertEqual(collector.choose_capture(plan,state),[]) # uncovered second scenes remain
        state["attempts"] += [attempt(c,status="rejected") for c in plan["candidates"][94+50:]]
        state["attempts"].append(attempt(plan["candidates"][188]))
        self.assertEqual(collector.choose_capture(plan,state)[0]["candidate_id"],plan["candidates"][188]["candidate_id"])

    def test_registry_maximizes_sources_then_caps_two_without_duplicate_IDs(self):
        plan=plan_fixture(4);pool=[qualified(c) for c in plan["candidates"]]
        selected=collector.select_registry(pool)
        self.assertEqual(len(selected),100);self.assertEqual(len({r["candidate_id"] for r in selected}),100)
        self.assertEqual(len({r["source_episode_index"] for r in selected}),94)
        counts=collector.Counter(r["source_episode_index"] for r in selected);self.assertLessEqual(max(counts.values()),2)
        excluded={selected[0]["candidate_id"]}
        replacement=collector.select_registry(pool,excluded)
        self.assertTrue(excluded.isdisjoint({r["candidate_id"] for r in replacement}))
        self.assertEqual(len(replacement),100)

    def test_paired_AB_same_scene_never_counts_twice_in_formal_registry(self):
        plan=plan_fixture();pool=[qualified(c) for c in plan["candidates"]]
        selected=collector.select_registry(pool)
        self.assertEqual(len(selected),94)
        self.assertEqual(len({row["scene_proposal_sha256"] for row in selected}),94)
        state=state_fixture();first=plan["candidates"][0]
        state["qualified"]=[qualified(first)]
        state["attempts"]=[attempt(first),attempt(first,"capture"),attempt(plan["candidates"][94])]
        self.assertEqual(collector.choose_capture(plan,state),[])

    def test_missing_strict_bootstrap_blocks_all_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            state=state_fixture();plan=plan_fixture()
            with patch.object(collector,"verified_queue",return_value=(state,plan)),\
                 patch.object(collector,"restore_collection_ready",return_value=False),\
                 patch.object(collector.subprocess,"Popen") as popen,\
                 self.assertRaisesRegex(ValueError,"bootstrap"):
                collector.run(directory,maximum_screen=4,maximum_capture=1)
            popen.assert_not_called()

    def test_partial_or_CPU_bootstrap_rejected_without_audit_or_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"bootstrap.json";path.write_text(json.dumps({"eligible_dataset":False,"require_rgb":False}))
            with patch("audit_handover_random100.audit") as audit,self.assertRaises(ValueError):
                collector.validate_bootstrap(path,plan_fixture())
            audit.assert_not_called()

    def phase_audit(self,phase):
        return {"contract":collector.packer.ADMISSION_CONTRACT,"replay_expert":True,
            "expert_reproduction":{"status":"all_generated_commands_reproduced_on_CPU"},
            "physical_gate":{"eligible_physics":True,"verified_handover_success":True},
            "actual_full_outer_footprint_final50":footprint(),
            "status":"physics_passed_full_dataset_admission_pending" if phase=="screen" else "independent_capture_audit_passed",
            "require_rgb":phase=="capture","eligible_dataset":phase=="capture"}

    def test_phase_specific_independent_audit_cannot_be_swapped(self):
        for phase in ("screen","capture"):
            collector.validate_phase_audit(self.phase_audit(phase),phase)
            with self.assertRaises(ValueError):collector.validate_phase_audit(self.phase_audit(phase),"capture" if phase=="screen" else "screen")

    def test_exit_zero_error_result_cannot_pass(self):
        with self.assertRaisesRegex(ValueError,"exit code is insufficient"):
            collector.validate_result({"status":"error"},{},plan_fixture()["candidates"][0],plan_fixture(),"capture")

    def test_complete_result_must_match_actual_frozen_source_roots_and_gate(self):
        plan=plan_fixture();candidate=plan["candidates"][0];spec=candidate["spec"]
        plan["code_sha256"]={str(Path("/synthetic")/name):"a"*64 for name in collector.REQUIRED_RUN_CODE}
        result={"contract":"cvpr-handover-random100-online-execution-v1","status":"execution_complete",
            "code_unchanged_during_run":True,"expert_dependencies_unchanged_during_run":True,
            "calibration_source_assets_unchanged":True,"commands_generated_online":True,"source_tokens_reencoded":True,
            "accepted_new_dataset":False,"object_restores_during_control":False,"source_split":"train",
            "source_episode_index":spec["source_episode_index"],"hand_command_condition":spec["hand_condition"],
            "capture_code_sha256":dict(plan["code_sha256"]),"source_sha256":spec["source"]["source_files_sha256"],
            "expert_dependency_sha256":{},"scene_proposal_sha256":spec["scene_proposal_sha256"],
            "physical_quality":{"acceptance_gate_version":"handover-physics-v2","eligible_physics":True,"verified_handover_success":True}}
        args={**collector.FORMAL,"source_episode":spec["source_episode_index"],"seed":spec["seed"],
            "hand_condition":spec["hand_condition"],"scene_only":False,"render_mode":"isaac",
            **{key:plan["runtime"][key] for key in ("raw_root","data_root","sonic_resources",
                "latent_projector_batch_view","latent_projection_phases","placement_encoder_mode","placement_locomotion")}}
        collector.validate_result(result,args,candidate,plan,"capture")
        for key,value in (("source_sha256",{}),("code_unchanged_during_run",False),("expert_dependency_sha256",{"wrong":"x"}),
                          ("scene_proposal_sha256","d"*64),("status","error")):
            changed=deepcopy(result);changed[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):collector.validate_result(changed,args,candidate,plan,"capture")
        for key,value in (("data_root","/different/assets"),("raw_root","/different/raw"),("latent_projection_phases","approach"),
                          ("xy_max_m",.02),("render_mode","none")):
            changed=deepcopy(args);changed[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):collector.validate_result(result,changed,candidate,plan,"capture")
        for missing in ("g1_sonic.py","g1_29dof_sonic_model12.yaml"):
            changed=deepcopy(result);del changed["capture_code_sha256"][str(Path('/synthetic')/missing)]
            with self.subTest(missing=missing),self.assertRaisesRegex(ValueError,"frozen source"):
                collector.validate_result(changed,args,candidate,plan,"capture")
        changed=deepcopy(args);changed["placement_locomotion"]="actual_hold"
        with self.assertRaisesRegex(ValueError,"locomotion"):
            collector.validate_result(result,changed,candidate,plan,"capture")
        plan["runtime"]["placement_locomotion"]="actual_hold"
        result["placement_locomotion"]=args["placement_locomotion"]="actual_hold"
        collector.validate_result(result,args,candidate,plan,"capture")
        plan["runtime"]["placement_locomotion"]="source_future"
        result["placement_locomotion"]=args["placement_locomotion"]="source_future"
        args["placement_encoder_mode"]="teleop"
        with self.assertRaisesRegex(ValueError,"placement encoder mode"):
            collector.validate_result(result,args,candidate,plan,"capture")
        plan["runtime"]["placement_encoder_mode"]="teleop";result["placement_encoder_mode"]="teleop"
        helper=str((collector.DIRECTORY/'handover_sonic_teleop_reference.py').resolve())
        bindings={helper:collector.sha(helper),**collector.official_placement_sources()}
        plan["code_sha256"][helper]=bindings[helper];plan["source_bindings"]=collector.official_placement_sources()
        result["capture_code_sha256"].update(bindings)
        collector.validate_result(result,args,candidate,plan,"capture")
        for missing in bindings:
            changed=deepcopy(result);del changed["capture_code_sha256"][missing]
            with self.subTest(missing=missing),self.assertRaisesRegex(ValueError,"all four"):
                collector.validate_result(changed,args,candidate,plan,"capture")

    def test_real_runtime_robot_and_loaded_WBC_YAML_are_explicitly_frozen(self):
        identities=collector.code_identities()
        simple=collector.REPO/'VLMExpert/third_party/SIMPLE'
        for relative in ('src/simple/robots/g1_sonic.py',
            'third_party/gear_sonic/utils/mujoco_sim/configs.py',
            'third_party/gear_sonic/utils/mujoco_sim/wbc_configs/g1_29dof_sonic_model12.yaml'):
            path=(simple/relative).resolve(strict=True)
            self.assertIn(str(path),identities);self.assertEqual(identities[str(path)],collector.sha(path))

    def test_mode1_headers_are_repo_code_and_external_inputs_are_separately_bound(self):
        from handover_sonic_teleop_reference import SOURCE_PATHS
        identities=collector.code_identities();official=collector.official_placement_sources()
        self.assertEqual(len(official),4)
        self.assertIn(str((collector.DIRECTORY/'handover_sonic_teleop_reference.py').resolve()),identities)
        for name,path in SOURCE_PATHS.items():
            path=path.resolve(strict=True)
            self.assertEqual(official[str(path)],collector.sha(path))
            if path.is_relative_to(collector.REPO):self.assertIn(str(path),identities)
            else:self.assertNotIn(str(path),identities)

    def test_plan_encoder_mode_is_hashed_and_cannot_be_switched_after_freeze(self):
        plan=plan_fixture();plan["runtime"]["placement_encoder_mode"]="teleop"
        with self.assertRaises(ValueError):collector.validate_plan(plan)
        plan=plan_fixture();plan["runtime"]["placement_encoder_mode"]="unknown"
        with self.assertRaises(ValueError):collector.validate_plan(plan)
        plan=plan_fixture();plan["runtime"]["placement_locomotion"]="actual_hold"
        with self.assertRaises(ValueError):collector.validate_plan(plan)
        plan=plan_fixture();plan["runtime"]["placement_locomotion"]="unknown"
        with self.assertRaises(ValueError):collector.validate_plan(plan)

    def test_live_owned_worker_is_preserved_and_dead_one_is_interrupted(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);state=state_fixture();row={"directory":"attempts/000000-screen","pid":123,
                "process_start_ticks":"unique","status":"running"};state["attempts"]=[row]
            with patch.object(collector,"process_identity",return_value="unique"):
                self.assertFalse(collector.finalize_attempt(root,state,row,plan_fixture()))
            self.assertEqual(row["status"],"running")
            with patch.object(collector,"process_identity",return_value=None):
                self.assertTrue(collector.finalize_attempt(root,state,row,plan_fixture()))
            self.assertEqual(row["status"],"interrupted");self.assertEqual(len(state["attempts"]),1)

    def test_frozen_code_drift_refuses_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"code.py";path.write_text("initial")
            plan={"code_sha256":{str(path):collector.sha(path)},"source_stat_identity":{str(path):collector.stat_identity(path)}}
            collector.verify_frozen(plan);path.write_text("changed")
            with self.assertRaisesRegex(ValueError,"executable drift"):collector.verify_frozen(plan)

    def test_source_stat_drift_cannot_silently_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"source";path.write_text("raw")
            plan={"code_sha256":{},"source_stat_identity":{str(path):collector.stat_identity(path)}}
            path.write_text("other")
            with self.assertRaisesRegex(ValueError,"source/model/resource"):collector.verify_frozen(plan)

    def test_partial_pool_writes_no_ready_or_registry(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);state=state_fixture();state["qualified"]=[qualified(c) for c in plan_fixture()["candidates"][:94]]
            self.assertFalse(collector.update_registry(root,state,plan_fixture()))
            self.assertEqual(list(root.iterdir()),[]);self.assertFalse(state["collection_ready"])

    def test_CPU_environment_hides_GPU_and_RTX_selects_one_device(self):
        cpu=collector.child_environment("screen","0");rtx=collector.child_environment("capture","2")
        self.assertEqual(cpu["CUDA_VISIBLE_DEVICES"],"");self.assertEqual(rtx["CUDA_VISIBLE_DEVICES"],"2")
        self.assertEqual(cpu["HF_HUB_OFFLINE"],"1");self.assertEqual(cpu["OMP_NUM_THREADS"],"1")

    def test_proposal_SHA_is_actual_pretty_JSON_bytes_not_internal_digest(self):
        proposal={"objects":{"target":[.0123,-.0456]},"proposal_sha256":"a"*64}
        with tempfile.TemporaryDirectory() as directory:
            import replay_handover_teacher as base
            path=Path(directory)/"proposal.json";base.write_json(path,proposal)
            self.assertEqual(collector.proposal_file_sha(proposal),collector.sha(path))
            self.assertNotEqual(collector.proposal_file_sha(proposal),proposal["proposal_sha256"])


if __name__=="__main__":unittest.main()
