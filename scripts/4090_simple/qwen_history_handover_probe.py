"""One-GPU, batch-four admission probe; never produces reusable model weights.

Uses the production accepted-window selector, trainer strict initialization and
loss. Two forward/backward passes exercise the full model, not an optimizer or
DDP throughput benchmark. Formal eight-rank training reloads joint38000 afresh.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import qwen_history_handover_train as training


CONTRACT = 'cvpr-handover149-qwen-history6-gpu-probe-v1'
GROUPS = ('lang_backbone', 'vision_tower', 'mm_projector', 'motion', 'action')


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def probe_configuration(config, plan, output):
    """Change only the disposable namespace and parallel-size bookkeeping."""
    original_config = config.model_dump(mode='json')
    original_plan = deepcopy(plan)
    require(plan['gpu_count'] == 8 and plan['per_gpu_batch'] == 4
            and plan['effective_global_batch'] == 32,
            'Probe must originate from the unchanged 8x4 production plan')
    require(config.data.transform.model.vlm_history_frames == 6
            and config.train.train_batch_size == 4
            and config.model.state_drop_prob == 0
            and config.train.resume_from_checkpoint is None,
            'Require six-image fresh state-keep production configuration')
    probe_config, probe_plan = deepcopy(config), deepcopy(plan)
    probe_config.train.output_dir = str(output / 'disposable_trainer')
    probe_plan.update(gpu_count=1, gpus=[0], effective_global_batch=4)
    require(config.model_dump(mode='json') == original_config and plan == original_plan,
            'Formal configuration mutated while making probe copies')
    return probe_config, probe_plan


def validate_batch(samples, batch, subset, indices):
    """Bind every Qwen image to the actual Motion cache selector before CUDA."""
    import numpy as np
    import torch
    require(len(samples) == 4 and len(indices) == 4, 'Actual batch must contain four samples')
    require(tuple(batch['actions'].shape) == (4, 30, 80)
            and tuple(batch['states'].shape) == (4, 1, 45)
            and tuple(batch['motion_img_embs'].shape) == (4, 6, 5, 1024)
            and tuple(batch['image_grid_thw'].shape) == (24, 3),
            'Production action/state/Motion shapes or the 24 real Qwen grids changed')
    require(torch.all(batch['actions_mask'][:, :, :78])
            and not torch.any(batch['actions_mask'][:, :, 78:]), 'Original 78/80 action mask changed')
    require(tuple(batch['observations'].shape[:2]) == (4, 6)
            and torch.equal(batch['vlm_history_frames'], torch.full((4,), 6)),
            'Six real RGB images must reach each collated sample')
    index = subset.dataset.raw_dataset.motion_history
    receipts = []
    for sample, selected in zip(samples, indices):
        actual_row = int(subset.indices[selected])
        rows = index.selected_cache_rows(actual_row)
        metadata = sample['vlm_history_metadata']
        require(metadata['contract'] == 'psi-qwen-motion-aligned-rgb-history-v1'
                and metadata['pack_id'] == int(index.pack_ids[actual_row])
                and metadata['episode_index'] == int(index.episode_ids[actual_row])
                and metadata['selected_cache_rows'] == rows.tolist()
                and metadata['current_is_last'] is True,
                'Qwen RGB metadata differs from this sample\'s actual Motion selector')
        episode = index._episode(metadata['pack_id'], metadata['episode_index'])
        np.testing.assert_array_equal(metadata['video_timestamps_s'], episode.video_timestamp[rows])
        np.testing.assert_array_equal(metadata['source_frame_indices'], episode.source_frame_index[rows])
        np.testing.assert_array_equal(metadata['frame_indices'], episode.frame_index[rows])
        np.testing.assert_array_equal(sample['motion_img_embs'], index.get_history(actual_row))
        require(np.all(np.abs(np.asarray(metadata['video_timestamps_s'])
                            - metadata['video_timestamps_s'][-1] - np.linspace(-1., 0., 6)) <= .25 / 30),
                'Real six-frame video history is not the original 5Hz Motion clock')
        receipts.append(dict(subset_index=selected, filtered_dataset_index=actual_row,
                             **metadata))
    require(torch.equal(batch['vlm_history_cache_rows'],
                        torch.tensor([r['selected_cache_rows'] for r in receipts]))
            and torch.equal(batch['vlm_history_video_timestamps_s'],
                            torch.tensor([r['video_timestamps_s'] for r in receipts], dtype=torch.float64)),
            'Collator dropped or changed the selected history identities')
    return {'actual_batch_size': 4, 'qwen_grids': 24,
            'input_ids_shape': list(batch['input_ids'].shape),
            'pixel_values_shape': list(batch['pixel_values'].shape), 'samples': receipts}


def gradient_summary(model, vlm_group_of):
    """Inspect every materialized gradient, including the complete VLM/Motion."""
    import torch
    grouped = {name: {'trainable_parameters': 0, 'gradient_parameters': 0,
                      'missing_gradient_names': [], 'finite': [], 'squares': []} for name in GROUPS}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith('vlm_model.'):
            group = vlm_group_of(name.removeprefix('vlm_model.'))
        elif name.startswith('motion_conditioner.'):
            group = 'motion'
        elif name.startswith('action_header.'):
            group = 'action'
        else:
            raise ValueError('Unexpected trainable component: ' + name)
        record = grouped[group]
        record['trainable_parameters'] += parameter.numel()
        if parameter.grad is None:
            record['missing_gradient_names'].append(name)
            continue
        grad = parameter.grad.detach()
        require(not grad.is_sparse, 'Unexpected sparse full-model gradient')
        record['gradient_parameters'] += parameter.numel()
        record['finite'].append(torch.isfinite(grad).all())
        record['squares'].append(torch.linalg.vector_norm(grad.float()).double().square())
    for group, record in grouped.items():
        finite = bool(torch.stack(record.pop('finite')).all().item()) if record['gradient_parameters'] else False
        norm = float(torch.stack(record.pop('squares')).sum().sqrt().item()) if record['gradient_parameters'] else 0.
        record.update(all_existing_gradients_finite=finite, gradient_l2_norm=norm)
        # Routed experts may legitimately be inactive; report their exact names.
        require(record['trainable_parameters'] > 0 and finite and math.isfinite(norm) and norm > 0,
                'Missing, nonfinite or zero gradient component: ' + group)
    return grouped


def run(prepared, output):
    import torch
    require(os.environ.get('WORLD_SIZE', '1') == '1' and os.environ.get('RANK', '0') == '0',
            'Probe is a standalone single process, never a formal training rank')
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
            'Expose exactly one CUDA GPU for this disposable probe')
    prepared = prepared.resolve(strict=True)
    require(not output.exists() and not output.is_symlink(), 'Probe output must be fresh; preserve earlier failures')
    output.mkdir(parents=True)
    state = {'contract': CONTRACT, 'status': 'running', 'prepared': str(prepared),
             'started_utc': training.now(), 'probe_code_sha256': training.sha(__file__),
             'probe_weights_reused_for_formal_training': False, 'optimizer_updates': 0,
             'optimizer_or_weights_saved': False, 'forward_backward_completed_steps': 0}
    training.write(output / 'PROBE_RESULT.json', state)
    try:
        runtime, formal_config, formal_plan, upstream = training.load(prepared)
        config_identity = digest(formal_config.model_dump(mode='json'))
        plan_identity = digest(formal_plan)
        state.update(config_sha256=formal_plan['config_sha256'],
                     plan_sha256=training.sha(prepared / 'preparation.json'),
                     code_acceptance_sha256=training.sha(formal_plan['code_acceptance']),
                     source_checkpoint_sha256=formal_plan['source_checkpoint_sha256'])
        config, plan = probe_configuration(formal_config, formal_plan, output)
        from psi.utils import seed_everything, move_to_device
        from psi.trainers.qwen3vl_mixin import PaddedCollatorForTogether
        from accelerate import Accelerator
        seed_everything(config.seed)
        trainer = runtime.trainer_class(plan, upstream)(config, torch.device('cuda:0'))
        require(Path(trainer.project_dir).resolve().is_relative_to(output.resolve()),
                'Disposable initialization receipts escaped the probe namespace')
        # The inherited helper strictly loads all joint38000 tensors on CPU,
        # then proves the original .5 -> 0 state-dropout migration.
        trainer.init_models()
        state['strict_initialization_receipt'] = str(Path(trainer.project_dir) / 'state_policy_migration_rank_0.json')
        migration = training.read(state['strict_initialization_receipt'])
        require(migration['status'] == 'passed', 'Inherited strict state migration failed')
        train, val, counts = runtime.datasets(config, trainer.vlm_processor, plan, upstream.shared)
        require(len(train) == 59468 and len(val) == 40, 'Original admitted windows changed')
        state['dataset_counts'] = counts
        # These are eight actual admitted training windows, not synthetic inputs.
        collator = PaddedCollatorForTogether(trainer.tokenizer.model_max_length,
                                            trainer.tokenizer.pad_token_id, padding_side='right')
        trainer.accelerator = Accelerator(mixed_precision=config.train.mixed_precision)
        require(trainer.accelerator.num_processes == 1, 'Unexpected distributed probe')
        trainer.model.to(trainer.device).train()
        state['trainable_parameters'] = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        state['total_parameters'] = sum(p.numel() for p in trainer.model.parameters())
        state['gpu'] = {'name': torch.cuda.get_device_name(0),
                        'total_memory_bytes': torch.cuda.get_device_properties(0).total_memory}
        state['steps'] = []
        for step in range(2):
            selected = list(range(step * 4, step * 4 + 4))
            samples = [train[i] for i in selected]
            batch = collator(samples)
            data_receipt = validate_batch(samples, batch, train, selected)
            trainer.model.zero_grad(set_to_none=True)
            batch = move_to_device(batch, trainer.device)
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            with trainer.forward_autocast():
                metrics = trainer.forward_and_loss(trainer.model, batch)
            loss = metrics['loss']
            require(loss.numel() == 1 and bool(torch.isfinite(loss).all()), 'Nonfinite full-model loss')
            loss.backward()
            groups = gradient_summary(trainer.model, trainer._vlm_group_of)
            require(trainer.model.action_header.last_state_drop_frac == 0,
                    'Actual forward unexpectedly dropped measured states')
            torch.cuda.synchronize()
            state['steps'].append({'completed_forward_backward': step + 1,
                'loss': float(loss.detach()), 'wall_seconds': time.monotonic() - started,
                'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                'batch': data_receipt, 'gradient_groups': groups, 'state_drop_fraction': 0.})
            state['forward_backward_completed_steps'] = step + 1
            training.write(output / 'PROBE_RESULT.json', state)
            del batch, samples, metrics, loss
        require(digest(formal_config.model_dump(mode='json')) == config_identity
                and digest(formal_plan) == plan_identity, 'Formal configuration mutated by probe')
        # Recheck frozen source/config acceptance after real model/data execution.
        training.load(prepared)
        state.update(status='passed', completed_utc=training.now(), actual_batch_size=4,
            qwen_images_per_sample=6, motion_history_frames=6,
            full_vlm_motion_gradients_finite=True, all_five_component_gradients_nonzero=True,
            formal_config_and_plan_unchanged=True,
            scope='Single GPU two full forward/backward passes; no optimizer/DDP/throughput claim; formal run reloads joint38000')
    except BaseException as error:
        state.update(status='failed', completed_utc=training.now(),
                     error=f'{type(error).__name__}: {error}')
        training.write(output / 'PROBE_RESULT.json', state)
        raise
    training.write(output / 'PROBE_RESULT.json', state)
    print(json.dumps({'status': state['status'], 'receipt': str(output / 'PROBE_RESULT.json')}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.prepared, args.output.absolute())


if __name__ == '__main__':
    main()
