"""Local joint pilot CPU tests: actual budget, accumulated updates and model-only IO."""
from pathlib import Path
import importlib.util
import copy
import json
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import check_liftbox_joint_gpu as check
import finetune_lift_box_hold_joint as pilot
import finetune_carrybox as shared


def alignment():
    model = dict(action_architecture="dual_source_v1", tune_vlm=True, tune_motion=True,
                 tune_mm_llm=False, tune_mm_mlp=False, tune_mm_vision=False,
                 lang_backbone_lr=1e-6, vision_tower_lr=1e-5, mm_projector_lr=1e-4, motion_expert_lr=1e-5)
    return {"configuration": {"model":model, "train":{"learning_rate":1e-4}},
            "optimizer_groups":[{"name":name,"peak_lr":lr,"trainable_parameters":i+2}
                                for i,(name,lr) in enumerate(zip(check.GROUPS,[1e-4,1e-6,1e-5,1e-4,1e-5]))]}


class ModelOnlyWriter:
    def __init__(self):
        self.calls=[]

    def save_model(self, model, destination, **kwargs):
        from safetensors.torch import save_file
        self.calls.append((destination,kwargs))
        path=Path(destination);path.mkdir(parents=True)
        save_file(model.state_dict(),str(path/'model.safetensors'))

    def save_state(self,*args,**kwargs):
        raise AssertionError("Optimizer/scheduler/RNG state must never be saved")


class JointPilotCPU(unittest.TestCase):
    def base_run_fixture(self, root, step):
        run=root/f'run{step}'; source=root/f'source{step}'
        (run/'provenance').mkdir(parents=True); source.mkdir()
        qwen=root/'qwen';qwen.mkdir(exist_ok=True)
        original={'model':{'model_name_or_path':'/remote/qwen','motion_checkpoint':'/remote/motion','action_dim':80},
                  'data':{'root_dir':'/remote/data','transform':{'field':{'stat_path':'/remote/statistics'}}}}
        def write(path,value):
            path.write_text(json.dumps(value,sort_keys=True))
        write(source/'run_config.json',original);write(run/'provenance/run_config.json',original)
        local=copy.deepcopy(original);local['model'].update(model_name_or_path=str(qwen),motion_checkpoint=str(run/'motion/joint_motion.ckpt'))
        local['data']['transform']['field']['stat_path']=str(run/'dataset_statistics.json')
        write(run/'run_config.json',local)
        for path in (source/'dataset_statistics.json',run/'dataset_statistics.json'):
            path.write_bytes(b'{"same_statistics":true}\n')
        (run/'dino_training_contract.json').write_bytes(b'{"same_dino_contract":true}\n')
        weights=run/'checkpoints'/f'ckpt_{step}'/'model.safetensors';weights.parent.mkdir(parents=True)
        weights.write_bytes(f'actual-source-model-{step}'.encode())
        (source/'model.safetensors').write_bytes(weights.read_bytes())
        snapshot={'step':step,'model_sha256':shared.sha256(weights),'model_bytes':weights.stat().st_size,
                  'save_completion_marker_verified':True,'training_modified':False}
        write(source/'SNAPSHOT_READY.json',snapshot);write(run/'provenance/SNAPSHOT_READY.json',snapshot)
        ready={'status':'local_joint_run_prepared','step':step,'joint_sha256':snapshot['model_sha256'],
               'source_snapshot':str(source),'optimizer_rng_restored':False,'strict_motion_load':'passed',
               'motion_weights':f'joint_posttrain_step_{step}'}
        write(run/'LOCAL_RUN_READY.json',ready)
        (run/'motion').mkdir();(run/'motion/joint_motion.ckpt').write_bytes(f'extracted-{step}'.encode())
        write(run/'motion/export_validation.json',{'weights':ready['motion_weights'],'strict_meta_load':'passed',
                                                  'exact_serialization_comparison':'passed'})
        return run

    def test_newer_weight_identity_preserves_explicit_older_data_anchor(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);old=self.base_run_fixture(root,28000);new=self.base_run_fixture(root,31000)
            replacements,receipt=pilot.validate_base_run(new,old)
            self.assertEqual(receipt['source_step'],31000)
            self.assertEqual(receipt['source_joint_sha256'],shared.sha256(new/'checkpoints/ckpt_31000/model.safetensors'))
            self.assertEqual(set(replacements),{'model.pretrained_joint_checkpoint','model.motion_checkpoint',
                                                'model.model_name_or_path','data.transform.field.stat_path'})
            self.assertTrue(receipt['original_source_model_and_data_equal'])
            self.assertTrue(receipt['statistics_bytes_equal'] and receipt['dino_contract_bytes_equal'])

    def test_new_base_rejects_changed_model_data_stats_dino_or_actual_weights(self):
        for changed in ('model','data','statistics','dino','weights','step','motion'):
            with self.subTest(changed=changed),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);old=self.base_run_fixture(root,28000);new=self.base_run_fixture(root,31000)
                if changed in ('model','data'):
                    path=new/'provenance/run_config.json';value=json.loads(path.read_text());value[changed]['unexpected_change']=True;path.write_text(json.dumps(value))
                elif changed=='statistics':(new/'dataset_statistics.json').write_bytes(b'different statistics')
                elif changed=='dino':(new/'dino_training_contract.json').write_bytes(b'different encoder')
                elif changed=='weights':(new/'checkpoints/ckpt_31000/model.safetensors').write_bytes(b'different weights')
                elif changed=='step':
                    path=new/'LOCAL_RUN_READY.json';value=json.loads(path.read_text());value['step']=28000;path.write_text(json.dumps(value))
                else:
                    path=new/'motion/export_validation.json';value=json.loads(path.read_text());value['weights']='joint_posttrain_step_28000';path.write_text(json.dumps(value))
                with self.assertRaises(ValueError):pilot.validate_base_run(new,old)

    def args(self, **overrides):
        fields=dict(max_steps=300,batch_size=1,grad_accum=4,warmup_steps=20,test_windows=20,seed=20261021)
        fields.update(overrides)
        return SimpleNamespace(**fields)

    def test_explicit_pilot_budget_overrides_two_step_check_without_changing_data(self):
        for batch,accum in ((1,4),(4,4),(2,8)):
            args=self.args(batch_size=batch,grad_accum=accum)
            result=pilot.pilot_overrides(args,alignment(),Path('/tmp/pilot'),'fixed')
            self.assertEqual((result['train.train_batch_size'],result['train.gradient_accumulation_steps'],
                              result['train.max_training_steps'],result['train.warmup_steps']), (batch,accum,300,20))
            self.assertEqual(result['train.lr_scheduler_type'],'cosine')
            self.assertEqual(result['train.checkpointing_steps'],301)
            self.assertIsNone(result['train.resume_from_checkpoint'])
            self.assertTrue(result['model.tune_vlm'] and result['model.tune_motion'])
            self.assertTrue(result['train.training_safety_checks'])
            self.assertFalse(any(key.startswith('data.') for key in result))
            for row in alignment()['optimizer_groups']:
                self.assertEqual(result[check.LR_FIELDS[row['name']]],row['peak_lr'])

    def test_invalid_budget_or_test_reselection_is_rejected(self):
        for changes in (dict(batch_size=0),dict(grad_accum=0),dict(warmup_steps=300),dict(test_windows=40)):
            with self.assertRaises(ValueError):
                pilot.pilot_overrides(self.args(**changes),alignment(),Path('/tmp/pilot'),'fixed')

    def test_accumulated_microbatches_do_not_increment_actual_update_count(self):
        lifecycle=shared.ComparisonLifecycle(2)
        lifecycle.before(lambda:None)
        for update in range(2):
            for micro in range(4):
                lifecycle.update(update,micro==3)
                self.assertEqual(lifecycle.completed,update+(micro==3))
        lifecycle.after(lambda:None)
        self.assertTrue(lifecycle.after_done)

    def test_cpu_optimizer_and_scheduler_start_fresh_with_zero_lr_warmup(self):
        import torch
        from cpu_master_adamw import CpuMasterAdamW
        from transformers.optimization import get_scheduler
        parameter=torch.nn.Parameter(torch.ones(2,dtype=torch.float32))
        native=torch.optim.AdamW([{'params':[parameter],'group_name':'action_header','lr':1e-4}])
        optimizer=CpuMasterAdamW.from_optimizer(native)
        scheduler=get_scheduler('cosine',optimizer,num_warmup_steps=20,num_training_steps=300)
        receipt=check.cpu_state_receipt(optimizer)
        self.assertEqual(receipt['completed_steps'],0)
        self.assertEqual(receipt['native_adamw_state_entries'],0)
        self.assertEqual(receipt['master_device_types'],['cpu'])
        self.assertEqual(optimizer.param_groups[0]['lr'],0)
        before=parameter.detach().clone()
        parameter.grad=torch.ones_like(parameter)
        optimizer.step();scheduler.step();optimizer.zero_grad()
        self.assertTrue(torch.equal(parameter,before))
        self.assertEqual(optimizer.completed_steps,1)
        self.assertGreater(optimizer.param_groups[0]['lr'],0)
        self.assertEqual(check.cpu_state_receipt(optimizer)['actual_state_devices'],['cpu'])

    def test_final_single_model_only_file_roundtrips_and_nonfinal_writes_nothing(self):
        import torch
        from safetensors.torch import load_file
        with tempfile.TemporaryDirectory() as directory:
            writer=ModelOnlyWriter()
            model=torch.nn.Linear(3,2)
            trainer=SimpleNamespace(project_dir=directory,accelerator=writer,model=model)
            self.assertIsNone(pilot.save_final_model_only(trainer,1,300))
            self.assertFalse((Path(directory)/'checkpoints').exists())
            saved=Path(pilot.save_final_model_only(trainer,300,300))
            self.assertEqual(sorted(p.name for p in saved.iterdir()),['model.safetensors'])
            self.assertEqual(writer.calls[0][1],dict(max_shard_size='100GB',safe_serialization=True))
            loaded=load_file(saved/'model.safetensors')
            self.assertTrue(all(torch.equal(value,loaded[name]) for name,value in model.state_dict().items()))
            with self.assertRaises(FileExistsError):
                pilot.save_final_model_only(trainer,300,300)

    def test_sharded_or_optimizer_output_is_not_mislabeled_single_model(self):
        class WrongWriter(ModelOnlyWriter):
            def save_model(self,model,destination,**kwargs):
                p=Path(destination);p.mkdir(parents=True)
                (p/'model.safetensors').write_bytes(b'fixture')
                (p/'optimizer.bin').write_bytes(b'forbidden')
        with tempfile.TemporaryDirectory() as directory:
            trainer=SimpleNamespace(project_dir=directory,accelerator=WrongWriter(),model=None)
            with self.assertRaisesRegex(RuntimeError,'one safetensors'):
                pilot.save_final_model_only(trainer,300,300)
            self.assertFalse((Path(directory)/'checkpoints/ckpt_300').exists())

    def test_real_accelerate_writer_shared_tensors_restore_strictly_on_cpu(self):
        import torch
        from accelerate import Accelerator
        from safetensors.torch import load_file
        class SharedModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight=torch.nn.Parameter(torch.arange(6,dtype=torch.float32).reshape(2,3))
                self.weight_alias=self.weight
                self.register_buffer('position',torch.arange(4,dtype=torch.float32))
                self.register_buffer('position_alias',self.position)
        accelerator=Accelerator(cpu=True,mixed_precision='no')
        # A model-only save must not invoke the training-state API.
        accelerator.save_state=lambda *a,**k: self.fail('save_state would serialize CPU optimizer moments')
        model=SharedModel()
        spec=importlib.util.spec_from_file_location('pilot_shared_alias_fixture',shared.REPO/'VLMExpert/src/psi/models/checkpoint_aliases.py')
        aliases=importlib.util.module_from_spec(spec);spec.loader.exec_module(aliases)
        with tempfile.TemporaryDirectory() as directory:
            trainer=SimpleNamespace(project_dir=directory,accelerator=accelerator,model=model)
            final=Path(pilot.save_final_model_only(trainer,300,300))
            self.assertEqual(sorted(p.name for p in final.iterdir()),['model.safetensors'])
            saved=load_file(final/'model.safetensors')
            self.assertEqual(len(saved),2)
            restored=SharedModel()
            restored.load_state_dict(aliases.restore_shared_tensor_aliases(restored,saved),strict=True)
            self.assertTrue(all(torch.equal(value,restored.state_dict()[name]) for name,value in model.state_dict().items()))


if __name__ == '__main__':
    unittest.main()
