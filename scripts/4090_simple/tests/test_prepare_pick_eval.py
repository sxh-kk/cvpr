"""Explicit Pick authority,13 checkpoints, provenance and tiny actual CPU export."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import prepare_pick_eval as deploy
import prepare_bend_pick_eval as old_deploy
import test_prepare_bend_pick_eval as old_tests


class PickDeploymentTests(unittest.TestCase):
    def start(self, patcher):
        value = patcher.start(); self.addCleanup(patcher.stop); return value

    def fixture_for(self, profile='bend_pick', step=5000):
        f = old_tests.BendDeploymentTests(); f.setUp(); self.addCleanup(f.doCleanups)
        source, release = f.source, f.release
        item = deepcopy(deploy.PROFILES[profile]); counts = item['counts']
        prepared_root = deploy.QUEUE_ROOT+'/runs/'+profile
        project_name = item['project_name'] or ('xmove_bend_pick_joint38000_statekeep_bs4_8gpu.us.flow1000.cosine.lr1.0e-05.b32.gpus8.261112012345')
        project = prepared_root+'/training/posttrain/'+project_name
        remote_release = deploy.SERVER_BASE+'/'+item['release_name']
        replacements = [(old_deploy.EXPECTED_PROJECT, project), (old_deploy.EXPECTED_ROOT, prepared_root),
            (old_deploy.EXPECTED_RELEASE, remote_release), (old_deploy.TASK_ID, item['task_id'])]
        def transform(value, key=None):
            if key in ('train_episodes', 'total_episodes'): return counts[0]
            if key == 'total_frames': return counts[1]
            if key == 'train_windows': return counts[2]
            if key in ('episode_indices', 'train_episode_indices'): return list(range(counts[0]))
            if key == 'profile' and value == 'bend_pick': return profile
            if isinstance(value, dict): return {transform(k):transform(v, k) for k,v in value.items()}
            if isinstance(value, list): return [transform(v) for v in value]
            if isinstance(value, str):
                for before,after in replacements: value = value.replace(before,after)
            return value
        for name in old_deploy.METADATA:
            path = source/name
            if name.endswith('.json'): deploy.write_json(path, transform(deploy.read(path)))
            else:path.write_text(transform(path.read_text()))
        pack = deploy.read(source/'release_metadata/packed/PACK_READY.json')
        deploy.write_json(release/'packed/PACK_READY.json', pack)
        original_manifest = transform(deploy.read(release/'dino/manifest.json'))
        deploy.write_json(release/'dino/manifest.json', original_manifest)
        pack_sha = deploy.sha256(source/'release_metadata/packed/PACK_READY.json')
        manifest_sha = deploy.sha256(source/'release_metadata/dino/manifest.json')
        relocation = deploy.read(source/'release_metadata/dino/REMOTE_DINO_RELOCATION.json')
        relocation.update(pack_ready_sha256=pack_sha, manifest_sha256=manifest_sha,
            original_manifest_sha256=deploy.sha256(release/'dino/manifest.json'))
        deploy.write_json(source/'release_metadata/dino/REMOTE_DINO_RELOCATION.json', relocation)
        ready = deploy.read(source/'release_metadata/A100_PICO_DATASET_READY.json')
        ready.update(pack_ready_sha256=pack_sha, cache_manifest_sha256=manifest_sha,
            remote_relocation_binding={'sha256': deploy.sha256(source/'release_metadata/dino/REMOTE_DINO_RELOCATION.json')})
        deploy.write_json(source/'release_metadata/A100_PICO_DATASET_READY.json',ready)
        prep = deploy.read(source/'preparation.json')
        prep.update(expected_ready_sha256=deploy.sha256(source/'release_metadata/A100_PICO_DATASET_READY.json'),
            config_sha256=deploy.sha256(source/'prepared_run_config.json'),argv_sha256=deploy.sha256(source/'prepared_argv.txt'))
        deploy.write_json(source/'preparation.json',prep)
        state = deploy.read(source/'training_state.json');state['expected_ready_sha256']=prep['expected_ready_sha256']
        deploy.write_json(source/'training_state.json',state)
        profiles = deepcopy(deploy.PROFILES);profiles[profile]['pack_sha256']=pack_sha
        self.start(patch.object(deploy,'PROFILES',profiles))
        plan = {'contract':'cvpr-pico-pick-three-model-a800-serial-queue-v1','training_authorized':True,
            'remote_root':deploy.QUEUE_ROOT,'checkpoint_steps':list(deploy.SAVE_STEPS),'per_task_optimizer_steps':10000,
            'source_checkpoint_sha256':deploy.SOURCE_SHA,'gpu_count':8,'tasks':[{'profile':profile,'task_id':item['task_id'],
                'release':remote_release,'episodes':counts[0],'frames':counts[1],'windows':counts[2],'pack_ready_sha256':pack_sha}]}
        queue_path = source.parent/(profile+'_authority.json');deploy.write_json(queue_path,plan)
        self.start(patch.object(deploy,'QUEUE_PLAN_SHA',deploy.sha256(queue_path)))
        self.start(patch.object(deploy,'EXPECTED_MODEL_BYTES',f.model_bytes))
        self.start(patch.object(deploy,'STAT_SHA',f.fixture.stat_sha))
        authority = deploy.PickDeployment(profile,step,source,queue_path)
        receipt = transform(deepcopy(f.receipt))
        receipt.update(contract=deploy.BACKUP_CONTRACT,profile=profile,step=step,
            run_identity={'run_directory':project,'prepared_root':prepared_root,
                'run_config_sha256':deploy.sha256(source/'run_config.json'),'preparation_sha256':deploy.sha256(source/'preparation.json')})
        receipt['model']['source_path']=project+f'/checkpoints/ckpt_{step}/model.safetensors'
        receipt['metadata']={name:{'source_path':remote,'bytes':(source/name).stat().st_size,'sha256':deploy.sha256(source/name)}
            for name,remote in authority.METADATA_SOURCES.items()}
        deploy.write_json(source/'backup_receipt.json',receipt)
        return authority,f,receipt,queue_path

    def rebind(self, authority, f, receipt, name):
        path=f.source/name
        receipt['metadata'][name]={'source_path':authority.METADATA_SOURCES[name], 'bytes':path.stat().st_size,
            'sha256':deploy.sha256(path)}
        if name in ('run_config.json','preparation.json'):
            receipt['run_identity']['run_config_sha256' if name=='run_config.json' else 'preparation_sha256']=deploy.sha256(path)
        deploy.write_json(f.source/'backup_receipt.json',receipt)

    def validate(self, authority, f):
        return authority.validate_source(f.source,f.release,authority.EXPECTED_STEP,f.model_sha)

    def test_all_profiles_bind_complete_run_dataset_and_twentyfour_exact_source_paths(self):
        for profile in deploy.PROFILES:
            with self.subTest(profile=profile):
                authority,f,receipt,_=self.fixture_for(profile)
                result=self.validate(authority,f)
                self.assertEqual(result[4]['profile'],profile)
                self.assertEqual(result[5]['status'],'passed')
                self.assertEqual(len(authority.METADATA),24)
                self.assertEqual(authority.COUNTS,deploy.PROFILES[profile]['counts'])
                self.assertEqual(receipt['model']['source_path'],authority.EXPECTED_PROJECT+'/checkpoints/ckpt_5000/model.safetensors')

    def test_all_thirteen_saves_select_exact_checkpoint_path_without_changing_training_recipe(self):
        for step in deploy.SAVE_STEPS:
            with self.subTest(step=step):
                authority,f,receipt,_=self.fixture_for(step=step)
                self.assertEqual(self.validate(authority,f)[4]['step'],step)
                self.assertEqual(deploy.read(f.source/'checkpoint_retention.json')['all_saved_model_steps'],list(deploy.SAVE_STEPS))
        for bad in [True,False,0,4999,12000]:
            with self.assertRaises(ValueError):deploy.PickDeployment('bend_pick',bad)
        with self.assertRaises(ValueError):deploy.PickDeployment('foreign',5000)

    def test_cross_task_profile_receipt_and_metadata_rebinding_are_rejected(self):
        authority,f,receipt,_=self.fixture_for('xmove_pick')
        for field,value in [('profile','bend_pick'),('task_id',old_deploy.TASK_ID),
            ('contract',old_deploy.BACKUP_CONTRACT),('training_completed',False),('checkpoint_save_completed',False)]:
            changed=deepcopy(receipt);changed[field]=value;deploy.write_json(f.source/'backup_receipt.json',changed)
            with self.subTest(field=field),self.assertRaises(ValueError):self.validate(authority,f)
        changed=deepcopy(receipt);changed['metadata']['run_config.json']['source_path']=old_deploy.EXPECTED_PROJECT+'/run_config.json'
        deploy.write_json(f.source/'backup_receipt.json',changed)
        with self.assertRaisesRegex(ValueError,'metadata path/bytes/SHA'):self.validate(authority,f)

    def test_queue_authority_and_generated_namespace_cannot_self_authorize_a_different_run(self):
        authority,f,receipt,queue=self.fixture_for('xmove_bend_pick')
        before=queue.read_bytes();queue.write_bytes(before+b' ')
        with self.assertRaisesRegex(ValueError,'queue plan SHA'):
            deploy.PickDeployment('xmove_bend_pick',5000,f.source,queue)
        queue.write_bytes(before)
        state=deploy.read(f.source/'training_state.json')
        for project in ['/elsewhere/run',authority.EXPECTED_PROJECT.replace('xmove_bend_pick_joint','xmove_pick_joint'),
            authority.EXPECTED_PROJECT+'/../escape',authority.EXPECTED_PROJECT.replace('lr1.0e-05','lr9.0e-01')]:
            changed=dict(state,run_directory=project);deploy.write_json(f.source/'training_state.json',changed)
            with self.subTest(project=project),self.assertRaisesRegex(ValueError,'namespace'):
                deploy.PickDeployment('xmove_bend_pick',5000,f.source,queue)

    def test_running_or_incomplete_final_health_and_retention_do_not_deploy_intermediate_saves(self):
        authority,f,receipt,_=self.fixture_for('xmove_pick',4500)
        for name,key,value in [('training_state.json','state','running'),('checkpoint_retention.json','all_models_preserved',False),
            ('gpu_check_update10000.json','all_rank_probe_values_equal',False),
            ('state_policy_migration_rank_7.json','unchanged_learned_parameters',False)]:
            before=deploy.read(f.source/name);changed=deepcopy(before);changed[key]=value
            deploy.write_json(f.source/name,changed);self.rebind(authority,f,receipt,name)
            with self.subTest(name=name),self.assertRaises(ValueError):self.validate(authority,f)
            deploy.write_json(f.source/name,before);self.rebind(authority,f,receipt,name)

    def test_sha_source_stability_and_actual_training_rtc_false_remain_mandatory(self):
        authority,f,receipt,_=self.fixture_for()
        for mutate in [lambda r:r['model'].update(source_sha256='0'*64),
            lambda r:r['model']['source_stat_after'].update(st_mtime_ns=r['model']['source_stat_after']['st_mtime_ns']+1),
            lambda r:r['run_identity'].update(prepared_root='/other')]:
            changed=deepcopy(receipt);mutate(changed);deploy.write_json(f.source/'backup_receipt.json',changed)
            with self.assertRaises(ValueError):self.validate(authority,f)
        config=deploy.read(f.source/'run_config.json');config['model']['rtc']=True
        with self.assertRaises(ValueError):authority.validate_training(f.source,config,deploy.read(f.source/'preparation.json'))

    def test_actual_tiny_cpu_strict_export_for_xmove_intermediate_and_bend_final(self):
        for profile,step in [('xmove_pick',5000),('bend_pick',10000)]:
            with self.subTest(profile=profile,step=step):
                authority,f,_,queue=self.fixture_for(profile,step)
                with patch.dict(sys.modules,f.runtime_modules()),patch.object(deploy.snapshot,'validate_config'), \
                        patch.object(deploy.generic,'validate_source',side_effect=AssertionError('Old Mobile gate invoked')):
                    report=authority.prepare(f.source,f.resources,f.output,f.repo,f.release,step,f.model_sha)
                torch=old_tests.fixture_module.torch
                motion=torch.load(f.output/'motion/joint_motion.ckpt',map_location='cpu',weights_only=False)
                self.assertTrue(torch.equal(motion['state_dict']['weight'],torch.full((2,2),9.)))
                self.assertFalse(torch.equal(motion['state_dict']['weight'],torch.full((2,2),-7.)))
                target=f.output/f'checkpoints/ckpt_{step}/model.safetensors'
                self.assertEqual(target.stat().st_ino,(f.source/'model.safetensors').stat().st_ino)
                self.assertEqual((report['contract'],report['status'],report['profile']),
                    (deploy.CONTRACT,deploy.READY_STATUS,profile))
                self.assertEqual(report['selected_checkpoint_is_final'],step==10000)
                self.assertEqual(report['joint_sha256'],f.model_sha)
                self.assertEqual(report['authorized_queue_plan_sha256'],deploy.sha256(queue))
                self.assertFalse(report['training_rtc_enabled']);self.assertFalse(report['GPU_used'])
                self.assertFalse(deploy.read(f.output/'run_config.json')['model']['rtc'])
                self.assertEqual(deploy.read(f.output/'dino_training_contract.json')['motion_image_policy'],'vlm_color_clean_motion')
                self.assertEqual(set(report['deployment_artifacts']),set(deploy.DEPLOYMENT_ARTIFACTS))
                self.assertTrue(report['component_load_validation']['strict_VLM_load'])
                self.assertTrue(report['component_load_validation']['strict_action_load'])
                self.assertFalse(report['previous_motion_parameters_used'])

    def test_existing_output_is_never_overwritten(self):
        authority,f,_,_=self.fixture_for()
        f.output.mkdir();marker=f.output/'retained';marker.write_bytes(b'retain')
        with self.assertRaisesRegex(ValueError,'fresh independent'):
            authority.prepare(f.source,f.resources,f.output,f.repo,f.release,5000,f.model_sha)
        self.assertEqual(marker.read_bytes(),b'retain')


if __name__=='__main__':unittest.main()
