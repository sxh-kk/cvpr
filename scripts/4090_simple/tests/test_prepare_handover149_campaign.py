"""CPU-only checkpoint provenance and exact framework-adaptation regression."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

FILE = Path(__file__).resolve().parents[1] / 'prepare_handover149_campaign.py'
SPEC = importlib.util.spec_from_file_location('campaign_prepare', FILE)
p = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(p)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class IdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.source, self.local = root / 'ckpt_4000', root / 'actual_cpu_export'
        self.source.mkdir(); self.local.mkdir()
        weight = self.source / 'model.safetensors'
        weight.write_bytes(b'checkpoint-specific CPU fixture, never a model')
        self.digest = p.sha(weight)
        deployed = self.local / 'checkpoints/ckpt_4000/model.safetensors'
        deployed.parent.mkdir(parents=True); os.link(weight, deployed)
        (self.source / 'run_config.json').write_text('{"original":true}')
        self.backup = {
            'contract': 'cvpr-handover149-local-checkpoint-backup-v1', 'step': 4000,
            'optimizer_copied': False, 'run_identity': {'run_directory': p.SOURCE_RUN},
            'model': {'source_sha256_before': self.digest, 'source_sha256_after': self.digest,
                'local_sha256': self.digest, 'source_path': p.SOURCE_RUN + '/checkpoints/ckpt_4000/model.safetensors',
                'bytes': weight.stat().st_size},
            'completed_retention_marker': {'all_models_preserved': True, 'saved_update': 5000,
                'all_saved_model_steps': [1000, 2000, 3000, 4000, 5000]},
            'metadata': {'run_config.json': p.sha(self.source / 'run_config.json')}}
        write(self.source / 'backup_receipt.json', self.backup)
        self.ready = {'contract': 'cvpr-handover149-local-deployment-v1',
            'status': 'local_handover149_run_prepared', 'task_step': 4000, 'step': 4000,
            'joint_sha256': self.digest, 'source_backup_receipt_sha256': p.sha(self.source / 'backup_receipt.json'),
            'source_model_bytes_verified': True, 'checkpoint_save_completed': True,
            'GPU_used': False, 'policy_forward_performed': False,
            'train_episodes': 149, 'train_windows': 59468, 'source_global_batch': 32,
            'num_blocks': 6, 'state_drop_prob': 0., 'strict_motion_load': 'passed',
            'full_saved_components_strict_CPU_load': {'status': 'passed', 'CUDA_initialized': False}}
        write(self.local / 'LOCAL_RUN_READY.json', self.ready)
        write(self.local / 'motion/joint_motion.ckpt', {'fixture': True})
        write(self.local / 'provenance/runtime_metadata_validation.json', {'fixture': True})
        write(self.local / 'dino_training_contract.json', {'motion_image_policy': 'vlm_color_clean_motion'})

    def validate(self):
        return p.validate_inputs(self.source, self.local, 4000, self.digest)

    def test_actual_checkpoint_specific_receipts_pass(self):
        self.assertEqual(self.validate()[0]['step'], 4000)

    def test_renamed_five_thousand_receipt_rejected(self):
        self.ready['task_step'] = self.ready['step'] = 5000
        write(self.local / 'LOCAL_RUN_READY.json', self.ready)
        with self.assertRaises(ValueError): self.validate()

    def test_failed_bs64_run_rejected(self):
        self.backup['run_identity']['run_directory'] = p.SOURCE_RUN.replace('bs4_restart_001/', '')
        write(self.source / 'backup_receipt.json', self.backup)
        with self.assertRaises(ValueError): self.validate()

    def test_incomplete_checkpoint_save_rejected(self):
        self.backup['completed_retention_marker']['all_saved_model_steps'].remove(4000)
        write(self.source / 'backup_receipt.json', self.backup)
        with self.assertRaises(ValueError): self.validate()

    def test_inconsistent_sha_rejected(self):
        self.backup['model']['source_sha256_after'] = '0' * 64
        write(self.source / 'backup_receipt.json', self.backup)
        with self.assertRaises(ValueError): self.validate()

    def test_same_bytes_copied_model_rejected(self):
        path = self.local / 'checkpoints/ckpt_4000/model.safetensors'
        value = path.read_bytes(); path.unlink(); path.write_bytes(value)
        with self.assertRaises(ValueError): self.validate()

    def test_cpu_acceptance_with_forward_rejected(self):
        self.ready['policy_forward_performed'] = True
        write(self.local / 'LOCAL_RUN_READY.json', self.ready)
        with self.assertRaises(ValueError): self.validate()

    def test_missing_actual_motion_rejected(self):
        (self.local / 'motion/joint_motion.ckpt').unlink()
        with self.assertRaises(ValueError): self.validate()

    def test_authorized_steps_exact(self):
        for step in (3000, 5000, 11000, True):
            with self.assertRaises(ValueError): p.names(step, 'campaign001')

    def test_unsafe_tag_rejected(self):
        for tag in ('../old', '', 'a/b', 'a b'):
            with self.assertRaises(ValueError): p.names(4000, tag)


@unittest.skipUnless(p.TEMPLATE.is_dir(), 'Accepted local R2 source unavailable')
class ActualSourceTests(unittest.TestCase):
    def changed(self, relative, step=4000):
        before = (p.TEMPLATE / relative).read_text()
        after = p.adapt(relative, before, step=step, tag='campaign001', source=Path('/backup') / f'ckpt_{step}')
        return before, after

    def test_all_six_step_template_outputs_parse(self):
        import ast
        for step in p.STEPS:
            files = list(p.SCRIPTS)
            files += [str(file.relative_to(p.TEMPLATE)) for sub in ('runtime_template', 'helper_templates')
                      for file in (p.TEMPLATE / sub).glob('*.py')]
            for relative in files:
                before, after = self.changed(relative, step)
                if relative.endswith('.py'): ast.parse(after, filename=relative)
                p.verify_runtime_bytes(relative, before, after, step)

    def test_numerical_templates_match_actual_accepted_manifest(self):
        mirror = Path(p.read(p.TEMPLATE / 'host_target.json')['mirror_home']).resolve()
        result = p.verify_template_runtime(p.TEMPLATE, mirror)
        self.assertEqual(result['status'], 'passed')
        self.assertEqual(len(result['runtime_and_helper_bindings']), 12)

    def test_sampler_and_controller_sources_byte_identical(self):
        for directory in ('runtime_template', 'helper_templates'):
            for file in (p.TEMPLATE / directory).glob('*.py'):
                relative = str(file.relative_to(p.TEMPLATE))
                before, after = self.changed(relative)
                if file.name not in ('protocol_core.py', 'rtc_acceptance.py'):
                    self.assertEqual(before, after, relative)

    def test_reference_five_thousand_not_relabelled(self):
        _, after = self.changed('prepare_checkpoint.py')
        self.assertIn('reference_model_step=5000', after)
        self.assertIn('STEPS=(4000,)', after)
        self.assertIn('handover149_experiment_{step}_official30_4090_trt10133_20261026_campaign001', after)

    def test_cleanup_hook_and_manifest_source_binding(self):
        _, evaluator = self.changed('evaluation_4090.py')
        _, preparer = self.changed('prepare_checkpoint.py')
        self.assertIn("run,base,strict=modules();__import__('owned_process_cleanup').install(run.ev);protocol,psha=", evaluator)
        self.assertIn("HERE/'runtime_audit.py',HERE/'owned_process_cleanup.py',HERE/'prepare_checkpoint.py'", preparer)
        before = (FILE.with_name('owned_process_cleanup.py')).read_text()
        self.assertEqual(p.adapt('owned_process_cleanup.py', before, step=4000,
            tag='campaign001', source=Path('/backup/ckpt_4000')), before)

    def test_old_real_audits_remain_original_fixtures(self):
        _, after = self.changed('test_dino_source_binding_CPU.py')
        self.assertIn("OLD=H.parent/'" + p.OLD_RUN.removesuffix('_r2') + "'", after)
        self.assertIn('handover149_experiment_4000_official30_4090_trt10133_20261026_campaign001', after)

    def test_physical_gpu_lock_and_namespace_gates_retained(self):
        before, after = self.changed('namespace_launch.sh')
        for line in before.splitlines():
            if any(token in line for token in ('flock -n 9', 'exec 9>', 'exec unshare', 'mount -t tmpfs')):
                self.assertIn(line, after)
        self.assertIn(p.names(4000, 'campaign001')[2], after)

    def test_runtime_guard_changes_cannot_hide_numeric_edits(self):
        before, after = self.changed('runtime_template/rtc_acceptance.py')
        with self.assertRaises(ValueError):
            p.verify_runtime_bytes('runtime_template/rtc_acceptance.py', before, after + '\n# unexpected\n', 4000)


if __name__ == '__main__':
    unittest.main(verbosity=2)
