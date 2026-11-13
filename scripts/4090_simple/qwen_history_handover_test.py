"""Focused CPU boundaries for the new Handover history ablation launcher."""
from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('history_handover_train',HERE/'qwen_history_handover_train.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
BASE=Path('/home/zxc/CVPR_Evaluation_runtime/handover_qwen_history6_20261113/baseline')


class HistoryLauncherBoundaries(unittest.TestCase):
    def setUp(self):
        self.original=m.read(BASE/'prepared_config.json')
        self.paths={name:'new-'+name for name in m.PATH_FIELDS}
        self.actual=deepcopy(self.original)
        m.put(self.actual,m.SEMANTIC_CHANGE,6)
        for name,value in self.paths.items():m.put(self.actual,name,value)

    def test_real_original_full_configuration_only_one_semantic_leaf_changes(self):
        receipt=m.validate_single_factor(self.original,self.actual,expected_paths=self.paths)
        self.assertTrue(receipt['all_other_config_values_exact'])
        self.assertEqual(receipt['only_semantic_change'],m.SEMANTIC_CHANGE)
        self.assertEqual(self.original['seed'],20261025)
        self.assertEqual(self.actual['train']['checkpointing_steps'],1000)
        self.assertEqual(self.actual['train']['early_checkpoint_steps'],[])

    def test_optimizer_seed_data_split_augmentation_and_model_drift_refused(self):
        for key,value in [('train.learning_rate',2e-5),('train.train_batch_size',8),('seed',1),
            ('model.num_blocks',5),('model.state_drop_prob',.5),('model.img_chunk',6),
            ('model.n_cams',6),('model.observation_horizon',6),('data.transform.repack.num_past_frames',5),
            ('data.train_repo_ids',['val']),('data.transform.model.img_aug',False),
            ('data.transform.model.motion_image_policy','shared_rgb'),('train.early_checkpoint_steps',[5500])]:
            with self.subTest(field=key):
                bad=deepcopy(self.actual);m.put(bad,key,value)
                with self.assertRaises(ValueError):m.validate_single_factor(self.original,bad,expected_paths=self.paths)

    def test_six_must_be_exact_integer_not_bool_float_or_other_count(self):
        for value in (True,6.,1,5,7):
            bad=deepcopy(self.actual);m.put(bad,m.SEMANTIC_CHANGE,value)
            with self.assertRaises(ValueError):m.validate_single_factor(self.original,bad,expected_paths=self.paths)

    def test_all_paths_must_be_explicitly_pinned(self):
        for bad in (None,{k:v for k,v in self.paths.items()if k!='timestamp'}):
            with self.assertRaises(ValueError):m.validate_single_factor(self.original,self.actual,expected_paths=bad)
        bad=deepcopy(self.actual);m.put(bad,'data.root_dir','foreign-data')
        with self.assertRaises(ValueError):m.validate_single_factor(self.original,bad,expected_paths=self.paths)

    def test_original_reference_cannot_already_be_six_frames(self):
        bad=deepcopy(self.original);m.put(bad,m.SEMANTIC_CHANGE,6)
        with self.assertRaises(ValueError):m.validate_single_factor(bad,self.actual,expected_paths=self.paths)

    def test_frozen_baseline_twenty_original_files_pass(self):
        path,manifest=m.baseline(BASE)
        self.assertEqual(path,BASE);self.assertEqual(len(manifest['files']),20)
        self.assertTrue(manifest['historical_5000_binding']['current_server_stat_exactly_matches_original_hashed_receipt'])

    def test_original_adapter_change_is_only_one_dependency_root(self):
        raw=(BASE/'quality100_bs4_shape_adapter.py').read_bytes()
        result=m.upstream_source(raw,Path('/new/reviewed-code'))
        old=('CODE=Path("'+m.ORIGINAL_CODE+'")').encode()
        self.assertEqual(result.replace(b'CODE=Path("/new/reviewed-code")',old,1),raw)
        with self.assertRaises(ValueError):m.upstream_source(raw+b'\n',Path('/new/reviewed-code'))

    def test_original_training_functions_are_inherited_unchanged(self):
        runtime=m.inherited({'baseline':str(BASE)})
        original=type(runtime)('reference_unmodified')
        original.__file__=str(BASE/'a100_handover149_bs4.py')
        exec(compile((BASE/'a100_handover149_bs4.py').read_bytes(),original.__file__,'exec'),original.__dict__)
        for name in ('datasets','map_accepted_rows','trainer_class','retain_model_weights','train','run','load'):
            with self.subTest(function=name):
                self.assertEqual(getattr(runtime,name).__code__.co_code,getattr(original,name).__code__.co_code)
        self.assertEqual(runtime.__file__,str(HERE/'qwen_history_handover_train.py'))
        self.assertEqual(runtime.CONTRACT,m.CONTRACT)

    def test_code_review_cannot_be_pending_or_single_frame(self):
        from unittest.mock import patch
        for value in ({'status':'pending'},dict(contract=m.CODE_CONTRACT,status='passed',
                single_factor='model.img_chunk',baseline_manifest_sha256=m.BASE_MANIFEST_SHA)):
            with patch.object(m,'read',return_value=value),self.assertRaises(ValueError):
                m.code_acceptance(BASE/'SOURCE_MANIFEST.json',m.REPO)

    def test_real_probe_must_match_current_formal_config_plan_code_and_source(self):
        from unittest.mock import patch
        plan=dict(config_sha256='a'*64,code_acceptance_sha256='b'*64)
        accepted=dict(status='passed',user_authorized=True,prepared_config_sha256='a'*64,
            preparation_sha256='c'*64,source_checkpoint_sha256=m.SOURCE_SHA,
            fresh_formal_load_from_joint38000=True,single_gpu_probe_path='/probe/PROBE_RESULT.json',
            single_gpu_probe_sha256='d'*64)
        probe=dict(status='passed',actual_batch_size=4,qwen_images_per_sample=6,motion_history_frames=6,
            forward_backward_completed_steps=2,full_vlm_motion_gradients_finite=True,
            probe_weights_reused_for_formal_training=False,config_sha256='a'*64,plan_sha256='c'*64,
            code_acceptance_sha256='b'*64,source_checkpoint_sha256=m.SOURCE_SHA)
        def check(value):
            def read(path):
                return {'preparation.json':plan,'ROOT_TRAINING_ADMISSION.json':accepted,
                        'PROBE_RESULT.json':value}[Path(path).name]
            def sha(path):return 'c'*64 if Path(path).name=='preparation.json' else 'd'*64
            with patch.object(m,'read',side_effect=read),patch.object(m,'sha',side_effect=sha):
                m.require_gpu_admission(Path('/formal'))
        check(probe)
        for name in ('config_sha256','plan_sha256','code_acceptance_sha256','source_checkpoint_sha256'):
            with self.subTest(identity=name):
                bad=dict(probe,**{name:'0'*64})
                with self.assertRaisesRegex(ValueError,'another formal'):check(bad)


if __name__=='__main__':
    unittest.main(verbosity=2)
