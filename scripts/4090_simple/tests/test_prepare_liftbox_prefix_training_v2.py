"""Model311 CPU fixtures (LeRobot history reader); never authorize real training."""
from contextlib import redirect_stderr
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO/'scripts/4090_simple'), str(REPO/'VLMExpert/src'), str(REPO)]
import prepare_liftbox_prefix_training_v2 as preparation


def fixture(count=110):
    frames=np.arange(count,dtype=np.int64)
    valid=frames>=30
    ref=np.where(valid,((frames-30)*50+29)//30,-1)
    columns={'frame_index':frames,'timestamp':frames.astype(np.float32)/30,
             'source_frame_index':(frames*50+29)//30,'source_reference_frame_index':ref,
             'source_episode_index':np.zeros(count,dtype=np.int64),
             'source_elapsed_timestamp':((frames*50+29)//30)/50,
             'teacher_source_timestamp':np.where(valid,ref/50,-1.),
             'action_valid':valid,'phase':np.where(valid,'oracle_teacher_token','warmup_observation_prefix')}
    tokens=np.zeros((count,64),np.float32);hands=np.zeros((count,14),np.float32)
    tokens[30:]=.25;hands[30:]=.4
    return columns,tokens,hands


class PrefixPreparationCPU(unittest.TestCase):
    def test_real_first_history_and_all_future_labels_start_at_teacher_row30(self):
        columns,tokens,hands=fixture()
        report,candidates=preparation.validate_episode_arrays(columns,tokens,hands,0)
        self.assertEqual(report['history_only_rows'],30)
        self.assertEqual(report['valid_teacher_rows'],80)
        self.assertEqual(candidates[0],30)
        self.assertEqual(candidates[-1],80)
        self.assertEqual(report['future_tail_rows_excluded'],29)
        self.assertTrue(report['first_future_action_valid'])
        # Invoke the production history selector on real30Hz cache ordinals;
        # physical native50Hz tick IDs deliberately remain a separate column.
        from psi.data.lerobot.motion_history import history_rows
        frames=columns['frame_index'];times=columns['timestamp']
        current,history=history_rows(frames,times,frames,times,np.ones(len(frames),bool))
        selected={int(row):past for row,past in zip(current,history)}
        np.testing.assert_array_equal(selected[30],[0,6,12,18,24,30])
        np.testing.assert_array_equal(columns['source_frame_index'][selected[30]],[0,10,20,30,40,50])
        self.assertEqual(columns['source_reference_frame_index'][30],0)

    def test_float32_late_episode_clock_and_global_ceil_are_valid(self):
        columns,tokens,hands=fixture(730)
        report,candidates=preparation.validate_episode_arrays(columns,tokens,hands,0)
        self.assertGreater(candidates[-1]/30,16)
        self.assertEqual(report['first_eligible_frame'],30)
        self.assertEqual(columns['source_frame_index'][31],52)
        self.assertEqual(columns['source_reference_frame_index'][31],2)

    def test_numeric_validity_prefix_label_or_teacher_gap_are_rejected(self):
        for corruption in ('numeric_bool','prefix_valid','missing_label','nonzero_prefix','short_future'):
            with self.subTest(corruption=corruption):
                columns,tokens,hands=fixture(59 if corruption=='short_future' else 110)
                if corruption=='numeric_bool':columns['action_valid']=columns['action_valid'].astype(np.int64)
                elif corruption=='prefix_valid':columns['action_valid'][29]=True
                elif corruption=='missing_label':columns['action_valid'][45]=False
                elif corruption=='nonzero_prefix':tokens[0,1]=.01
                with self.assertRaises(ValueError):preparation.validate_episode_arrays(columns,tokens,hands,0)

    def test_clock_phase_or_source_identity_corruption_is_rejected(self):
        for corruption in ('old_floor','teacher_reference_global','phase','source','clock','teacher_time'):
            with self.subTest(corruption=corruption):
                columns,tokens,hands=fixture()
                if corruption=='old_floor':columns['source_frame_index']=columns['frame_index']*50//30
                elif corruption=='teacher_reference_global':columns['source_reference_frame_index'][30:]=columns['source_frame_index'][30:]
                elif corruption=='phase':columns['phase'][30]='warmup_observation_prefix'
                elif corruption=='source':columns['source_episode_index'][20]=8
                elif corruption=='clock':columns['source_elapsed_timestamp'][20]+=.02
                else:columns['teacher_source_timestamp'][30]=1.
                with self.assertRaises(ValueError):preparation.validate_episode_arrays(columns,tokens,hands,0)

    def test_train_bounds_exclude_prefix_placeholders_without_refitting(self):
        columns,tokens,hands=fixture()
        field=SimpleNamespace(action_min=[.1]*80,action_max=[1.]*80)
        values=np.concatenate((tokens,hands),axis=1)
        report=preparation.valid_train_bounds([(values,columns['action_valid'])]*3,field)
        self.assertEqual(report['unique_command_events'],240)
        self.assertEqual(report['excluded_history_only_train_rows'],90)
        self.assertEqual(report['groups']['body64']['outside_fraction'],0)
        self.assertEqual(report['groups']['hand14']['outside_fraction'],0)
        self.assertFalse(report['bounds_reestimated'] or report['test_used_for_statistics'])
        self.assertEqual(field.action_min,[.1]*80)
        with self.assertRaisesRegex(ValueError,'bool'):
            preparation.valid_train_bounds([(values,columns['action_valid'].astype(int))],field)

    def test_actual_sim_verifier_execution_is_required_and_v1_status_claims_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);script=root/'verifier.py';script.write_text('# CPU verifier fixture only')
            audit_python=root/'python';audit_python.write_text('fixture')
            summary={'contract':preparation.CONTRACT,'task_id':preparation.TASK_ID,'splits':preparation.SPLITS,
                     'episodes':[{'episode_index':i} for i in range(5)],'packed_files':{'data':'signature'},
                     'train_demo_curation':{'selected_train_sources':[0,1,2],'scope':'CPU fixture only'},
                     'test_demo_curation':{'selected_test_source':4,'scope':'CPU fixture only'}}
            (root/'LIFT_COMMANDED_ROLLOUT_READY.json').write_text(json.dumps(summary))
            with patch.object(preparation,'VERIFIER',script),patch.object(preparation.subprocess,'run') as run:
                run.return_value=SimpleNamespace(returncode=0,stdout=json.dumps(summary),stderr='')
                result=preparation.audit_quality(root,audit_python)
                self.assertEqual(result['command'],[str(audit_python),str(script),'--verify-root',str(root)])
                self.assertEqual(run.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'],'')
                self.assertEqual(run.call_args.kwargs['env']['PYTHONNOUSERSITE'],'1')
                self.assertEqual(result['verifier_script_sha256'],preparation.shared.sha256(script))
                self.assertEqual(result['scope'], 'accepted numerical evidence retained + actual raw/packed byte binding/new transform verification')
                self.assertIn('not repeated', result['execution_scope'])
                for wrong in ({'status':'passed'}, {**summary,'contract':'cvpr-simple-lift-commanded-rollout-v1'}):
                    run.return_value=SimpleNamespace(returncode=0,stdout=json.dumps(wrong),stderr='')
                    with self.assertRaises(ValueError):preparation.audit_quality(root,audit_python)
                run.return_value=SimpleNamespace(returncode=1,stdout=json.dumps(summary),stderr='actual force mismatch')
                with self.assertRaisesRegex(RuntimeError,'actual force mismatch'):preparation.audit_quality(root,audit_python)

    def test_actual_train015_identity_is_retained_and_heldout_or_duplicate_ids_fail(self):
        pack={"splits":{"train":[0,1,5],"val":[3],"test":[4]}}
        self.assertEqual(preparation.validated_splits(pack)["train"],[0,1,5])
        columns,tokens,hands=fixture()
        columns["source_episode_index"][:]=5
        report,_=preparation.validate_episode_arrays(columns,tokens,hands,5)
        self.assertEqual(report["first_eligible_frame"],30)
        for train in ([0,1,3],[0,1,4],[0,1,8],[0,1,1],[0,True,5],[0,1]):
            with self.subTest(train=train), self.assertRaises(ValueError):
                preparation.validated_splits({"splits":{"train":train,"val":[3],"test":[4]}})
        with self.assertRaisesRegex(ValueError,"curation"):
            preparation.quality_gate_report({**pack,"train_demo_curation":{"selected_train_sources":[0,1,2]}})

    def test_curated_test9_10_11_keeps_actual_ids_and_full_failed_case_provenance(self):
        for source in (4,9,10,11):
            pack={"splits":{"train":[0,1,5],"val":[3],"test":[source]},
                  "train_demo_curation":{"selected_train_sources":[0,1,5]},
                  "test_demo_curation":{"selected_test_source":source,"candidate_order":[4,9,10,11],
                                        "failed_source4_result_sha256":"b"*64}}
            with self.subTest(source=source):
                gate=preparation.quality_gate_report(pack)
                self.assertEqual(gate["test_source_episodes"],[source])
                self.assertEqual(gate["test_demo_curation"],pack["test_demo_curation"])
                self.assertEqual(gate["required_physical_before_after_source_episodes"],sorted({4,8,source}))
                columns,tokens,hands=fixture();columns["source_episode_index"][:]=source
                preparation.validate_episode_arrays(columns,tokens,hands,source)
                bad=copy.deepcopy(pack);bad["test_demo_curation"]["selected_test_source"]=8
                with self.assertRaisesRegex(ValueError,"curation"):preparation.quality_gate_report(bad)
                for order in (None, [4,10,9,11], [4,9,10], [9,10,11]):
                    bad=copy.deepcopy(pack);bad["test_demo_curation"]["candidate_order"]=order
                    with self.assertRaisesRegex(ValueError,"curation"):preparation.quality_gate_report(bad)
        for source in (3,5,8,12,True):
            with self.subTest(rejected=source),self.assertRaises(ValueError):
                preparation.validated_splits({"splits":{"train":[0,1,5],"val":[3],"test":[source]}})
        with self.assertRaises(ValueError):
            preparation.validated_splits({"splits":{"train":[0,1,9],"val":[3],"test":[10]}})

    def test_only31k_and_identical_statistics_can_back_the_v2_pack(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);base=root/'base';data=root/'data';base.mkdir();data.mkdir()
            for path in (base/'dataset_statistics.json',data/'source_dataset_statistics.json'):path.write_text('{"frozen":true}')
            ready={'status':'local_joint_run_prepared','step':31000,'joint_sha256':preparation.SOURCE_SHA,'optimizer_rng_restored':False}
            path=base/'LOCAL_RUN_READY.json';path.write_text(json.dumps(ready))
            pack={'contract':preparation.CONTRACT,'task_id':preparation.TASK_ID,'splits':preparation.SPLITS,
                  'source_joint_sha256':preparation.SOURCE_SHA,'source_statistics_sha256':preparation.shared.sha256(base/'dataset_statistics.json')}
            preparation.validate_base_and_pack(base,pack,data)
            preparation.validate_base_and_pack(base,{**pack,'splits':{'train':[0,1,5],'val':[3],'test':[4]}},data)
            for wrong in ({**pack,'contract':'v1'}, {**pack,'source_joint_sha256':'a'*64},
                          {**pack,'splits':{'train':[0,1,2],'val':[8],'test':[4]}}):
                with self.assertRaises(ValueError):preparation.validate_base_and_pack(base,wrong,data)
            (data/'source_dataset_statistics.json').write_text('{"reestimated":true}')
            with self.assertRaisesRegex(ValueError,'statistics'):preparation.validate_base_and_pack(base,pack,data)

    def test_heldout_scope_records_previous_diagnostics_without_claiming_a_blind_test(self):
        pack={'splits':{'train':[0,1,5],'val':[3],'test':[4]},
              'train_demo_curation':{'selected_train_sources':[0,1,5], 'candidate_order':[2,5,6,7],
                                    'failed_source2_result_sha256':'a'*64},
              'test_demo_curation':{'selected_test_source':4,'candidate_order':[4,9,10,11]}}
        gate=preparation.quality_gate_report(pack)
        self.assertEqual(gate['training_source_episodes'],[0,1,5])
        self.assertEqual(gate['train_demo_curation'],pack['train_demo_curation'])
        self.assertEqual(gate['status'],'actual_new_task_evidence_recomputed')
        self.assertEqual(gate['previously_diagnosed_sources'],[4,8])
        self.assertTrue(gate['not_blind_test'])
        self.assertFalse(gate['test_used_for_control_selection'])
        self.assertIn('current v2',gate['test_used_for_control_selection_scope'])
        self.assertIn('not a statement',gate['test_used_for_control_selection_scope'])
        self.assertIn('no training gradients',gate['test_data_policy'])
        self.assertIn('not a claim to repeat',gate['status_scope'])

    def test_cli_is_prepare_only_and_cannot_invoke_action_training(self):
        args=['--base-run','/base','--dataset-root','/data','--dino-cache-root','/dino','--output','/output']
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):preparation.parser().parse_args([*args,'--prepare-only','--train'])
        parsed=preparation.parser().parse_args([*args,'--prepare-only'])
        self.assertTrue(parsed.prepare_only)
        self.assertFalse(hasattr(parsed,'train'))
        # The actual new entrypoint must never call an existing Action-only train function.
        source=Path(preparation.__file__).read_text()
        self.assertNotIn('shared._train_with_comparison(',source)
        self.assertNotIn('train_joint(',source)


if __name__=='__main__':unittest.main()
