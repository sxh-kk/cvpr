"""CPU contract regressions for the149 BS4 milestone; no model or CUDA import."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SCRIPTS))
spec = importlib.util.spec_from_file_location('prepare_handover149_eval',SCRIPTS/'prepare_handover149_eval.py')
m = importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class SourceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name);self.source = self.root/'ckpt_5000';self.source.mkdir()
        self.release = self.root/'release';(self.release/'packed').mkdir(parents=True);(self.release/'dino').mkdir()
        self.config = {'model':{'action_architecture':'dual_source_v1','num_blocks':6,'action_dim':80,'odim':45,
            'action_chunk_size':30,'state_drop_prob':0.,'state_null_token':True,'tune_vlm':True,'tune_motion':True,
            'rtc':False,'pretrained_joint_checkpoint':'/accepted/checkpoints/ckpt_38000/model.safetensors'},
            'wandb':deepcopy(m.EXPECTED_TRACKER),
            'train':{'hf_token':"PosixPath('.hf_token')",'train_batch_size':4,'gradient_accumulation_steps':1,'max_training_steps':10000,'resume_from_checkpoint':None},
            'data':{'transform':{'repack':{'action_keys':['action.body_token_v1_1','action[:14]'],
                'state_keys':['observation.state'],'pad_action_dim':80,'pad_state_dim':45,'action_mask_key':'action_mask'},
                'model':{'shared_image_profile':'256x320','motion_image_policy':'vlm_color_clean_motion'},
                'field':{'stat_path':'/accepted/release/packed/source_dataset_statistics.json',
                    'action_min':[0,1],'action_max':[1,2],'state_min':[0],'state_max':[1]}}}}
        (self.source/'model.safetensors').write_bytes(b'fixture-only-executed-weights')
        (self.source/'dataset_statistics.json').write_text('{"fixed_original_stats":true}')
        (self.source/'argv.txt').write_text('train.py posttrain_sonic_psi0_config\n')
        self.stat = m.sha256(self.source/'dataset_statistics.json')
        stat_patch = patch.object(m,'STAT_SHA',self.stat);stat_patch.start();self.addCleanup(stat_patch.stop)
        self.pack={'contract':m.PACK_CONTRACT,'train_episodes':149,'total_frames':68259,'train_windows':59468,
            'component_counts':{'original45':45,'new_random14':14,'human90':90},'normalization_refitted':False,
            'final_ready':False,'checkpoint_contract':{'source_step':38000,'source_checkpoint_sha256':m.SOURCE_SHA,
                'bounds_refitted':False,'statistics':{'sha256':self.stat},
                'bounds_sha256':{n:m.digest(self.config['data']['transform']['field'][n]) for n in
                    ('action_min','action_max','state_min','state_max')}}}
        self.manifest={'state':'complete','contract':'unifolm-dino-native-frames-v1','episode_indices':list(range(149)),
            'source':{'total_episodes':149,'total_frames':68259,'fps':30},'encoder':{'fixture':True}}
        self.ready={'status':'ready_for_training','contract':'handover-train149-a100-ready-v1',
            'train_episodes':149,'total_frames':68259,'complete_history_future_windows':59468,
            'component_counts':{'original_train':45,'new_random':14,'live_pico_human':90},
            'normalization_refitted':False,'validation_included':False}
        self.reindex={'contract':m.PACK_CONTRACT+'/DINO-reindex','status':'passed','train_episodes':149,
            'all_other_encoder_fields_exact':True}
        self.preparation={'contract':m.PREPARATION_CONTRACT,'fresh_run':True,'initial_global_step':0,
            'optimizer_scheduler_rng_restored':False,'source_step':38000,'source_checkpoint_sha256':m.SOURCE_SHA,
            'source_checkpoint':self.config['model']['pretrained_joint_checkpoint'],'gpu_count':8,'per_gpu_batch':4,
            'gradient_accumulation_steps':1,'effective_global_batch':32,'train_episodes':149,'train_windows':59468,
            'max_optimizer_steps':10000,'release':'/accepted/release','bindings_sha256':{
                '/accepted/release/packed/source_dataset_statistics.json':self.stat}}
        self.receipt={'contract':m.BACKUP_CONTRACT,'step':5000,'optimizer_copied':False,
            'completed_retention_marker':{'all_models_preserved':True,'all_saved_model_steps':[1000,2000,3000,4000,5000],
                'saved_update':5000},'run_identity':{'run_directory':m.EXPECTED_RUN},'model':{
                'source_path':m.EXPECTED_RUN+'/checkpoints/ckpt_5000/model.safetensors',
                'bytes':(self.source/'model.safetensors').stat().st_size,
                **{k:m.sha256(self.source/'model.safetensors')for k in ('source_sha256_before','source_sha256_after','local_sha256')}}}
        model_patch = patch.object(m,'EXPECTED_MODEL_SHA',m.sha256(self.source/'model.safetensors'))
        model_patch.start();self.addCleanup(model_patch.stop)
        self.seal()

    def put(self,path,value):path.write_text(json.dumps(value,sort_keys=True))

    def move_to_milestone(self, step, *, change_weights=True):
        """Fixture models have distinct bytes; changing only receipt labels is insufficient."""
        target=self.root/f'ckpt_{step}'
        if target != self.source:
            self.source.rename(target);self.source=target
        if change_weights:
            (self.source/'model.safetensors').write_bytes(f'fixture-saved-model-step-{step}'.encode())
        self.receipt['step']=step
        self.receipt['completed_retention_marker'].update(
            saved_update=step,all_saved_model_steps=list(range(1000,step+1,1000)))
        model=self.receipt['model']
        model.update(source_path=f'{m.EXPECTED_RUN}/checkpoints/ckpt_{step}/model.safetensors',
            bytes=(self.source/'model.safetensors').stat().st_size)
        actual=m.sha256(self.source/'model.safetensors')
        model.update({key:actual for key in ('source_sha256_before','source_sha256_after','local_sha256')})
        self.seal()
        return actual

    def seal(self, *, refresh_reindex=True, refresh_prepared=True):
        self.put(self.release/'packed/PACK_READY.json',self.pack)
        self.put(self.release/'dino/manifest.json',self.manifest)
        if refresh_reindex:
            self.reindex.update(pack_ready_sha256=m.sha256(self.release/'packed/PACK_READY.json'),
                manifest_sha256=m.sha256(self.release/'dino/manifest.json'))
        self.put(self.release/'dino/MERGED149_REINDEX.json',self.reindex)
        self.put(self.release/'TRAINING_DATASET_READY.json',self.ready)
        for name in m.RELEASE_METADATA:
            self.preparation['bindings_sha256']['/accepted/release/'+name]=m.sha256(self.release/name)
        self.put(self.source/'run_config.json',self.config)
        if refresh_prepared:
            prepared=deepcopy(self.config);prepared['train']['hf_token']='.hf_token'
            prepared['wandb'].update({key:None for key in m.EXPECTED_TRACKER})
            (self.source/'evidence').mkdir(exist_ok=True)
            self.put(self.source/'evidence/prepared_run_config.json',prepared)
            (self.source/'evidence/prepared_argv.txt').write_text('train.py\nposttrain_sonic_psi0_config\n')
        self.preparation.update(config_sha256=m.sha256(self.source/'evidence/prepared_run_config.json'),
            argv_sha256=m.sha256(self.source/'evidence/prepared_argv.txt'))
        self.put(self.source/'preparation.json',self.preparation)
        self.receipt['metadata']={n:m.sha256(self.source/n)for n in m.METADATA}
        self.receipt['run_identity'].update(run_config_sha256=self.receipt['metadata']['run_config.json'],
            preparation_sha256=self.receipt['metadata']['preparation.json'])
        self.put(self.source/'backup_receipt.json',self.receipt)

    def test_completed_milestone_is_valid_without_complete_training_flag(self):
        _,config,preparation,receipt=m.validate_source(self.source,self.release)
        self.assertEqual(config['model']['state_drop_prob'],0.)
        self.assertEqual((preparation['effective_global_batch'],receipt['step']),(32,5000))
        self.assertFalse(self.pack['final_ready'])

    def test_requested_milestones_require_their_actual_saved_model_and_explicit_sha(self):
        for step in (4000,6000,7000,8000,9000,10000):
            source_sha=self.move_to_milestone(step)
            with self.subTest(step=step):
                _,config,preparation,receipt=m.validate_source(self.source,self.release,
                    expected_step=step,expected_model_sha256=source_sha)
                self.assertEqual(receipt['step'],step)
                self.assertEqual(config['model']['num_blocks'],6)
                self.assertEqual(preparation['source_step'],38000)
                self.assertFalse(self.pack['final_ready'])

    def test_additional_milestone_cannot_use_receipt_sha_as_its_only_authority(self):
        self.move_to_milestone(6000)
        with self.assertRaisesRegex(ValueError,'separate A100 source'):
            m.validate_source(self.source,self.release,expected_step=6000)

    def test_task5000_pin_cannot_be_overridden(self):
        with self.assertRaisesRegex(ValueError,'cannot be overridden'):
            m.validate_source(self.source,self.release,expected_model_sha256='1'*64)

    def test_metadata_only_relabel_of5000_is_rejected(self):
        source_sha=self.move_to_milestone(6000,change_weights=False)
        with self.assertRaisesRegex(ValueError,'cannot be relabeled'):
            m.validate_source(self.source,self.release,expected_step=6000,expected_model_sha256=source_sha)

    def test_separately_pinned_source_sha_rejects_resealed_alternate_weights(self):
        source_sha=self.move_to_milestone(6000)
        (self.source/'model.safetensors').write_bytes(b'different-saved-weight-values')
        self.move_to_milestone(6000,change_weights=False)
        with self.assertRaisesRegex(ValueError,'source-before/source-after/local SHA differs'):
            m.validate_source(self.source,self.release,expected_step=6000,expected_model_sha256=source_sha)

    def test_new_step_checks_actual_local_bytes_after_matching_all_receipt_hashes(self):
        source_sha=self.move_to_milestone(6000)
        weights=self.source/'model.safetensors'
        weights.write_bytes(b'x'*weights.stat().st_size)
        with self.assertRaisesRegex(ValueError,'Actual local milestone model SHA differs'):
            m.validate_source(self.source,self.release,expected_step=6000,expected_model_sha256=source_sha)

    def test_expected_step_cannot_be_taken_from_another_milestone_receipt(self):
        source_sha=self.move_to_milestone(6000)
        with self.assertRaisesRegex(ValueError,'task7000 model-only'):
            m.validate_source(self.source,self.release,expected_step=7000,expected_model_sha256=source_sha)

    def test_new_step_needs_completed_checkpoint_save_inclusion(self):
        source_sha=self.move_to_milestone(10000)
        for marker in ({'saved_update':9000,'all_saved_model_steps':[10000]},
                       {'saved_update':10000,'all_saved_model_steps':[9000]}):
            self.receipt['completed_retention_marker'].update(marker);self.seal()
            with self.subTest(marker=marker),self.assertRaisesRegex(ValueError,'retention authority'):
                m.validate_source(self.source,self.release,expected_step=10000,expected_model_sha256=source_sha)

    def test_unauthorized_steps_and_malformed_external_pins_are_rejected(self):
        for step in (1000,3000,11000,True,6000.):
            with self.subTest(step=step),self.assertRaisesRegex(ValueError,'Unsupported149'):
                m.validate_source(self.source,self.release,expected_step=step,expected_model_sha256='1'*64)
        for value in (None,'','a'*63,'g'*64,'A'*64):
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,'separate A100 source'):
                m.validate_source(self.source,self.release,expected_step=6000,expected_model_sha256=value)

    def test_actual_runtime_metadata_serialization_bridge_is_explicit(self):
        proof=m.validate_runtime_metadata(self.source,self.config,self.preparation)
        self.assertTrue(proof['all_other_config_values_exact'])
        self.assertTrue(proof['argv_tokens_exact'])
        self.assertEqual(len(proof['allowed_runtime_metadata']),5)
        self.assertNotEqual(proof['prepared_config_sha256'],proof['runtime_config_sha256'])

    def test_non_metadata_runtime_change_is_rejected_even_when_backup_resealed(self):
        self.config['model']['state_drop_prob']=.5;self.seal(refresh_prepared=False)
        with self.assertRaisesRegex(ValueError,'beyond exact Path/tracker'):m.validate_source(self.source,self.release)

    def test_tracker_values_cannot_be_arbitrary_strings(self):
        self.config['wandb']['name']='another-model';self.seal()
        with self.assertRaisesRegex(ValueError,'beyond exact Path/tracker'):m.validate_source(self.source,self.release)

    def test_final_argv_parameter_change_is_rejected(self):
        (self.source/'argv.txt').write_text('train.py posttrain_sonic_psi0_config --model.state-drop-prob=0.5')
        self.seal()
        with self.assertRaisesRegex(ValueError,'semantic tokens'):m.validate_source(self.source,self.release)

    def test_old114_or_failed_bs8_run_is_rejected(self):
        for wrong in (m.EXPECTED_RUN.replace('bs4_restart_001/',''),'/accepted/old114'):
            self.receipt['run_identity']['run_directory']=wrong;self.seal()
            with self.subTest(wrong=wrong),self.assertRaisesRegex(ValueError,'current149 BS4'):m.validate_source(self.source,self.release)

    def test_partial_or_changed_local_model_is_rejected(self):
        (self.source/'model.safetensors').write_bytes(b'x'*(self.source/'model.safetensors').stat().st_size)
        with self.assertRaisesRegex(ValueError,'SHA differs'):m.validate_source(self.source,self.release)

    def test_saved_checkpoint_authority_must_include5000(self):
        self.receipt['completed_retention_marker']['all_saved_model_steps']=[1000,2000,3000,4000];self.seal()
        with self.assertRaisesRegex(ValueError,'retention authority'):m.validate_source(self.source,self.release)

    def test_source_run_cannot_relabel_another_model_milestone(self):
        self.receipt['model']['source_path']=m.EXPECTED_RUN+'/checkpoints/ckpt_4000/model.safetensors';self.seal()
        with self.assertRaisesRegex(ValueError,'source/run identity'):m.validate_source(self.source,self.release)

    def test_resealed_changed_state_drop_or_block_count_is_rejected(self):
        for field,value in [('state_drop_prob',.5),('num_blocks',12)]:
            old=self.config['model'][field];self.config['model'][field]=value;self.seal()
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'measured state retained'):m.validate_source(self.source,self.release)
            self.config['model'][field]=old

    def test_training_bs8_provenance_is_rejected(self):
        self.preparation.update(per_gpu_batch=8,effective_global_batch=64);self.seal()
        with self.assertRaisesRegex(ValueError,'BS4 batch32 provenance'):m.validate_source(self.source,self.release)

    def test_resealed_changed_inline_normalization_is_rejected(self):
        self.config['data']['transform']['field']['state_max']=[5];self.seal()
        with self.assertRaisesRegex(ValueError,'Inline inherited bounds'):m.validate_source(self.source,self.release)

    def test_old_sonic_token_key_is_rejected(self):
        self.config['data']['transform']['repack']['action_keys']=['action.body_token','action[:14]'];self.seal()
        with self.assertRaisesRegex(ValueError,'body-token'):m.validate_source(self.source,self.release)

    def test_original45_random14_human90_composition_required(self):
        self.pack['component_counts']={'original100':100,'new_random14':14,'human35':35};self.seal()
        with self.assertRaisesRegex(ValueError,'original45'):m.validate_source(self.source,self.release)

    def test_unbound_release_metadata_changed_is_rejected(self):
        self.put(self.release/'dino/manifest.json',{'state':'complete','encoder':'different'})
        with self.assertRaisesRegex(ValueError,'metadata changed'):m.validate_source(self.source,self.release)

    def test_missing_dino_episode_is_rejected_even_if_metadata_resealed(self):
        self.manifest['episode_indices'].pop();self.seal()
        with self.assertRaisesRegex(ValueError,'native30Hz DINO'):m.validate_source(self.source,self.release)

    def test_reindex_cannot_bind_another_pack(self):
        self.reindex['pack_ready_sha256']='0'*64;self.seal(refresh_reindex=False)
        with self.assertRaisesRegex(ValueError,'reindex source linkage'):m.validate_source(self.source,self.release)

    def test_historical_pack_flag_is_not_current_readiness_authority(self):
        self.ready['status']='production_pending';self.seal()
        with self.assertRaisesRegex(ValueError,'actual149 training readiness'):m.validate_source(self.source,self.release)


class SourceCodeAndEncoderTests(unittest.TestCase):
    def test_exported_contract_gets_actual_training_policy_not_legacy_default(self):
        config={'data':{'transform':{'model':{'motion_image_policy':'vlm_color_clean_motion'}}}}
        for reference in ({'encoder':{'identity':'unchanged'}},
                          {'encoder':{'identity':'unchanged'},'motion_image_policy':'shared_rgb'}):
            before=deepcopy(reference)
            contract=m.bind_training_motion_image_policy(reference,config)
            self.assertEqual(contract['motion_image_policy'],'vlm_color_clean_motion')
            self.assertEqual(contract['encoder'],before['encoder'])
            self.assertEqual(reference,before)

    def test_export_cannot_invent_policy_when_source_is_missing_or_different(self):
        for model in ({},{'motion_image_policy':None},{'motion_image_policy':'shared_rgb'}):
            config={'data':{'transform':{'model':model}}}
            with self.subTest(model=model),self.assertRaisesRegex(ValueError,'source run configuration'):
                m.bind_training_motion_image_policy({'motion_image_policy':'vlm_color_clean_motion'},config)

    def test_frozen_inference_model_code_change_is_rejected(self):
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory);bindings={}
            for prefix in m.CRITICAL_PREFIXES:
                p=root/prefix/'fixture.py';p.parent.mkdir(parents=True,exist_ok=True);p.write_text('# frozen\n')
                bindings['/remote/632968d/'+str(p.relative_to(root))]=m.sha256(p)
            preparation={'code_sha256':bindings}
            self.assertEqual(m.validate_frozen_model_code(root,preparation)['matched_files'],6)
            (root/'MotionExpert/fixture.py').write_text('# changed architecture\n')
            with self.assertRaisesRegex(ValueError,'differs from frozen'):m.validate_frozen_model_code(root,preparation)

    def test_runtime_transition_preserves_weights_and_preprocessing(self):
        encoder={'checkpoint':'/reference/dino','local_file_sha256':{'model.safetensors':'a'*64},
            'torch_version':'2.6.0+cu118','transformers_version':'4.49.0','preprocessing':{'do_resize':False},'shape':[5,1024]}
        actual=deepcopy(encoder);actual.update(checkpoint='/local/dino',torch_version='2.7.0+cu126',transformers_version='4.57.1')
        self.assertFalse(m.recorded_encoder_transition(encoder,actual)['numerical_bitwise_equivalence_claimed'])
        actual['preprocessing']['do_resize']=True
        with self.assertRaisesRegex(ValueError,'processor'):m.recorded_encoder_transition(encoder,actual)


class CommandLineTests(unittest.TestCase):
    def test_cli_passes_requested_step_and_independent_sha_to_preparation(self):
        argv=['prepare_handover149_eval.py','--source','/source/ckpt_4000','--resources','/resources',
            '--output','/new/output','--release','/release','--step','4000',
            '--expected-model-sha256','a'*64]
        with patch.object(sys,'argv',argv),patch.object(m,'prepare',return_value={'status':'fixture'})as prepare,\
             patch('builtins.print'):
            m.main()
        self.assertEqual(prepare.call_args.kwargs,{'expected_step':4000,'expected_model_sha256':'a'*64})

    def test_cli_defaults_preserve_pinned5000_preparation(self):
        argv=['prepare_handover149_eval.py','--source','/source/ckpt_5000','--resources','/resources',
            '--output','/new/output','--release','/release']
        with patch.object(sys,'argv',argv),patch.object(m,'prepare',return_value={'status':'fixture'})as prepare,\
             patch('builtins.print'):
            m.main()
        self.assertEqual(prepare.call_args.kwargs,{'expected_step':5000,'expected_model_sha256':None})


if __name__=='__main__':unittest.main()
