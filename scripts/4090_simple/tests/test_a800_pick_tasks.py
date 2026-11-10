"""CPU checks of task isolation, real BS4 parsing/factory and strict migration."""
import copy
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1]
REPO = SCRIPT.parents[1]
for path in (SCRIPT, REPO/'VLMExpert/src', REPO, REPO/'VLMExpert'):
    sys.path.insert(0, str(path))
import a800_pick_tasks as run

REFERENCE = Path('/home/zxc/CVPR_Evaluation_runtime/pick299_training_20261110/reference')


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True)+'\n')


class PickTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary, cls.prior, cls.reference, cls.files = run.load_reference(REFERENCE)
        cls.legacy = run.module('pick_test_real_legacy_helpers', cls.files['a100_handover_merged114.py'])
        cls.up = run.module('pick_test_real_bs4_adapter', cls.files['quality100_bs4_shape_adapter.py'])

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        run.activate_profile('bend_pick')

    def plan(self, profile='bend_pick'):
        run.activate_profile(profile)
        return {'contract': run.CONTRACT, 'profile': profile, 'task_id': run.TASK_ID,
            'reference_launcher_sha256': run.REFERENCE_LAUNCHER_SHA256,
            'timestamp': 'test', 'release': str(run.RELEASE), 'fresh_run': True, 'initial_global_step': 0,
            'optimizer_scheduler_rng_restored': False, 'gpus': list(range(8)), 'gpu_count': 8,
            'effective_global_batch': 32, 'peak_learning_rates': run.PEAK_LRS,
            'max_optimizer_steps': 10000, 'warmup_steps': 1000, 'per_gpu_batch': 4,
            'gradient_accumulation_steps': 1, 'train_episodes': run.EXPECTED_COUNTS[0],
            'total_frames': run.EXPECTED_COUNTS[1], 'train_windows': run.EXPECTED_COUNTS[2],
            'validation_role': run.VALIDATION_ROLE, 'source_step': 38000,
            'source_checkpoint_sha256': run.SOURCE_SHA, 'source_checkpoint': self.prior['source_checkpoint'],
            'kind': 'experiment', 'migration_helper': str(self.files['state_dropout_migration.py'])}

    def fixture_release(self, profile='bend_pick'):
        self.plan(profile)
        root = self.root/profile
        episodes, frames, windows = run.EXPECTED_COUNTS
        entries = []
        for index in range(episodes):
            entries.append({'episode_index': index, 'length': frames if index == 0 else 0,
                'eligible_current_rows': list(range(windows)) if index == 0 else [],
                'task_id': run.TASK_ID, 'split': 'train', 'split_restriction': 'train_only'})
        metadata = root/'packed/train/meta/episodes.jsonl'
        metadata.parent.mkdir(parents=True)
        metadata.write_text(''.join(json.dumps(row)+'\n' for row in entries))
        write(root/'packed/train/meta/info.json', {})
        write(root/'packed/source_dataset_statistics.json', {})
        write(root/'dino/manifest.json', {})
        counts = dict(train_episodes=episodes, total_frames=frames, train_windows=windows)
        write(root/'packed/PACK_READY.json', {'contract': run.PACK_CONTRACT, 'task_id': run.TASK_ID,
            'selection_policy': 'native_only', **counts})
        payload = {str(path.relative_to(root)): {'sha256': run.sha(path), 'size_bytes': path.stat().st_size,
            'mtime_ns': path.stat().st_mtime_ns} for path in root.rglob('*') if path.is_file()}
        ready = {'contract': run.READY_CONTRACT, 'task_id': run.TASK_ID, 'selection_policy': 'native_only',
            'status': 'passed', 'remote_production_ready': True, 'production_ready_for_bound_pack_and_cache': True,
            'DINO_relocation_pending': False, 'training_started': False, 'CUDA_initialized': False, 'GPU_used': False,
            'normalization_refitted': False, 'split_restriction': 'train_only', **counts,
            'release_root': str(root), 'pack_root': str(root/'packed'), 'dino_root': str(root/'dino'),
            'payload_files': payload, 'pack_ready_sha256': run.sha(root/'packed/PACK_READY.json'),
            'cache_manifest_sha256': run.sha(root/'dino/manifest.json')}
        write(root/run.READY_NAME, ready)
        return root, ready

    def test_reference_is_actual_completed_bs4_and_tamper_rejected(self):
        self.assertEqual(self.reference['train']['train_batch_size'], 4)
        self.assertEqual(self.reference['model']['state_drop_prob'], 0)
        self.assertTrue(self.reference['model']['state_null_token'])
        self.assertEqual(self.reference['model']['action_architecture'], 'dual_source_v1')
        self.assertEqual(self.prior['source_checkpoint_sha256'], run.SOURCE_SHA)
        copied = self.root/'reference'
        shutil.copytree(REFERENCE, copied)
        target = copied/'quality100_bs4_shape_adapter.py'
        target.write_bytes(target.read_bytes()+b'\n# changed\n')
        with self.assertRaisesRegex(ValueError, 'BS4 reference changed'):
            run.load_reference(copied)

    def test_profile_counts_roots_and_fresh_config_cannot_cross_tasks(self):
        self.assertEqual(len({row['task_id'] for row in run.PROFILES.values()}), 3)
        for profile in run.PROFILES:
            plan = self.plan(profile)
            config = run.expected_config(self.reference, plan, self.root)
            run.validate_config(config, plan, self.reference, self.root)
            altered = dict(plan, task_id='cvpr/OtherTask-v0')
            with self.assertRaisesRegex(ValueError, 'fresh Pick'):
                run.validate_config(config, altered, self.reference, self.root)
            for key, value in [('state_drop_prob', .5), ('pretrained_joint_checkpoint', '/other/model.safetensors')]:
                changed = copy.deepcopy(config)
                changed['model'][key] = value
                with self.assertRaisesRegex(ValueError, 'Configuration drift'):
                    run.validate_config(changed, plan, self.reference, self.root)
        for invalid in (None, 'handover', '../bend_pick'):
            with self.assertRaises(ValueError):
                run.activate_profile(invalid)

    def test_real_training_argv_parser_and_external_bs4_factory_accept_new_schedule(self):
        from psi.trainers.trainer import should_save_training_checkpoint
        for profile in run.PROFILES:
            plan = self.plan(profile)
            # The real parser loads inherited normalization from disk. Bind
            # the byte-identical local accepted release instead of an absent
            # server path; no bounds are synthesized or patched into config.
            local_release = Path('/home/zxc/CVPR_Evaluation_runtime/pico_latest300_20261109/processing_001/releases')/profile
            plan['release'] = str(local_release)
            argv = self.up.shared.rewrite_argv(self.files['argv.txt'].read_text().split(),
                run.config_overrides(self.root, local_release, plan['timestamp']))
            path = self.root/'argv.txt'
            path.write_text('\n'.join(argv)+'\n')
            config = self.up.shared.parse_training_argv(path)
            run.validate_config(config, plan, self.reference, self.root)
            # The real archived adapter's old load/prepare guards are never used.
            with patch.object(self.up, 'load', side_effect=AssertionError('old hardcoded schedule must not run')):
                cls = run.trainer_class(plan, self.legacy, self.up)
            self.assertIn('Quality100Trainer', [base.__name__ for base in cls.__mro__])
            actual = [step for step in range(1, 10001) if should_save_training_checkpoint(step,
                config.train.checkpointing_steps, config.train.max_training_steps, config.train.early_checkpoint_steps)]
            self.assertEqual(actual, list(run.SAVE_STEPS))

    def test_real_packed_admission_rows_map_all_tasks_without_handover_filter(self):
        import numpy as np
        for profile in run.PROFILES:
            self.plan(profile)
            local = Path('/home/zxc/CVPR_Evaluation_runtime/pico_latest300_20261109/processing_001/releases')/profile
            episodes = [json.loads(line) for line in (local/'packed/train/meta/episodes.jsonl').read_text().splitlines()]
            ranges, start = {}, 0
            for episode in episodes:
                ranges[episode['episode_index']] = (start, start+episode['length'])
                start += episode['length']
            history = SimpleNamespace(pack_info=[None], source_episode_ranges=[ranges],
                original_indices=np.arange(start))
            selected, counts = self.legacy.map_accepted_rows(history, episodes, run.EXPECTED_COUNTS[2])
            self.assertEqual(len(selected), run.EXPECTED_COUNTS[2])
            self.assertEqual(len(counts), run.EXPECTED_COUNTS[0])
            for episode in episodes:
                first, last = ranges[episode['episode_index']]
                local_rows = {row-first for row in selected if first <= row < last}
                self.assertEqual(local_rows, set(episode['eligible_current_rows']))
                self.assertTrue(all(row >= 30 and row+29 < episode['length'] for row in local_rows))

    def test_ready_task_identity_counts_sha_and_full_byte_drift(self):
        for profile in run.PROFILES:
            root, ready = self.fixture_release(profile)
            with patch.object(run, 'RELEASE', root):
                digest = run.sha(root/run.READY_NAME)
                self.assertEqual(run.validate_ready(root, digest), ready)
                for field, value in [('task_id', 'cvpr/Other-v0'), ('train_windows', 1),
                        ('remote_production_ready', False), ('DINO_relocation_pending', True)]:
                    changed = dict(ready, **{field: value})
                    write(root/run.READY_NAME, changed)
                    with self.assertRaisesRegex(ValueError, 'readiness is incomplete'):
                        run.validate_ready(root, run.sha(root/run.READY_NAME))
                write(root/run.READY_NAME, ready)
                with self.assertRaisesRegex(ValueError, 'READY SHA differs'):
                    run.validate_ready(root, '0'*64)
                source = root/'packed/source_dataset_statistics.json'
                stat = source.stat()
                source.write_bytes(b'x'*stat.st_size)
                os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))
                with self.assertRaisesRegex(ValueError, 'payload changed'):
                    run.validate_ready(root, run.sha(root/run.READY_NAME))

    def test_launch_cpu_context_is_full_nice_zero_and_no_process_is_changed(self):
        cpus = list(range(64))
        plan = {'training_cpus': cpus, 'training_nice': 0}
        with patch.object(run.os, 'sched_getaffinity', return_value=set(cpus)), patch.object(run.os, 'getpriority', return_value=0):
            self.assertFalse(run.require_launch_runtime(plan)['other_processes_modified'])
        for actual, nice in [({63}, 19), (set(cpus), 19), ({0}, 0)]:
            with patch.object(run.os, 'sched_getaffinity', return_value=actual), patch.object(run.os, 'getpriority', return_value=nice):
                with self.assertRaisesRegex(ValueError, 'ordinary SSH'):
                    run.require_launch_runtime(plan)

    def test_thirteen_models_permanent_only_two_new_recovery_states_retained(self):
        import torch
        plan = self.plan()
        checkpoint_root = self.root/'training/checkpoints'
        class FakeBase:
            def save_checkpoint(trainer, step):
                path = checkpoint_root/f'ckpt_{step}'
                path.mkdir(parents=True)
                for name in ('model.safetensors', 'optimizer.bin', 'scheduler.bin', 'random_states_0.pkl'):
                    (path/name).write_bytes(b'owned new test run')
                return str(path)
        up = SimpleNamespace(trainer_class=lambda plan: FakeBase)
        cls = run.trainer_class(plan, self.legacy, up)
        trainer = cls()
        trainer.project_dir = self.root/'training'
        trainer.accelerator = SimpleNamespace(is_main_process=True)
        outside = self.root/'unrelated/optimizer.bin'
        outside.parent.mkdir(); outside.write_bytes(b'keep')
        with patch.object(torch.distributed, 'broadcast_object_list'):
            with self.assertRaises(ValueError):
                trainer.save_checkpoint(500)
            for step in run.SAVE_STEPS:
                trainer.save_checkpoint(step)
        self.assertEqual([int(path.name[5:]) for path in sorted(checkpoint_root.iterdir(), key=lambda p:int(p.name[5:]))],
            list(run.SAVE_STEPS))
        for step in run.SAVE_STEPS:
            self.assertTrue((checkpoint_root/f'ckpt_{step}/model.safetensors').is_file())
            self.assertEqual((checkpoint_root/f'ckpt_{step}/optimizer.bin').exists(), step in run.SAVE_STEPS[-2:])
        self.assertEqual(outside.read_bytes(), b'keep')

    def test_initializer_strictly_loads_all_cpu_source_weights_before_semantic_migration(self):
        import torch
        from safetensors.torch import save_file, load_file
        helper = run.module('pick_test_actual_migration', self.files['state_dropout_migration.py'])
        values = dict(action_dim=80, action_pred_horizon=30, action_hidden_dim=8, action_num_blocks=6,
            action_nheads=2, view_feature_dim=8, odim=45, time_dim=8, state_null_token=True,
            qk_norm='rms_norm', state_drop_prob=.5, dropout=0.)
        class TinyHead(torch.nn.Module):
            architecture_id = helper.ARCHITECTURE_ID
            def __init__(self, **kwargs):
                super().__init__()
                self.config = SimpleNamespace(**kwargs)
                self.state_drop_prob = kwargs['state_drop_prob']
                self.weight = torch.nn.Parameter(torch.empty(2, 2))
                self.register_buffer('_architecture_signature', torch.tensor(
                    helper.expected_signature(self, self.state_drop_prob), dtype=torch.int64))
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.action_header = TinyHead(**values)
                with torch.no_grad(): self.action_header.weight.copy_(torch.arange(4).reshape(2, 2))
        source_model = TinyModel()
        source = self.root/'source.safetensors'
        save_file(source_model.state_dict(), str(source))
        class StrictBase:
            def init_models(trainer):
                self.assertEqual(trainer.model_cfg.state_drop_prob, .5)
                trainer.model = TinyModel()
                trainer.model.load_state_dict(load_file(str(source)), strict=True)
        plan = self.plan()
        cls = run.trainer_class(plan, self.legacy, SimpleNamespace(trainer_class=lambda plan: StrictBase))
        trainer = cls()
        trainer.project_dir = self.root/'fresh'
        trainer.model_cfg = SimpleNamespace(state_drop_prob=0., pretrained_joint_checkpoint=str(source))
        trainer.train_cfg = SimpleNamespace(resume_from_checkpoint=None, skip_resumed_steps=False)
        trainer.optimizer = trainer.lr_scheduler = None
        trainer.init_models()
        receipt = run.read(trainer.project_dir/'state_policy_migration_rank_0.json')
        self.assertEqual(receipt['status'], 'passed')
        self.assertTrue(receipt['source']['all_serialized_tensors_exactly_equal_before_migration'])
        self.assertEqual(receipt['changed_state_tensors'], ['action_header._architecture_signature'])
        self.assertTrue(torch.equal(source_model.action_header.weight, trainer.model.action_header.weight))
        self.assertEqual(trainer.model_cfg.state_drop_prob, 0.)
        self.assertEqual(trainer.model.action_header.state_drop_prob, 0.)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main()
