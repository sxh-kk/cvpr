"""CPU admission-gate tests; never construct a production model or CUDA context."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch

import qwen_history_handover_probe as probe


class ProbeConfigurationTests(unittest.TestCase):
    def test_formal_plan_and_config_are_unchanged(self):
        class Config(SimpleNamespace):
            def model_dump(self, **kwargs):
                return {'frames': self.data.transform.model.vlm_history_frames,
                        'output': self.train.output_dir, 'exp': self.exp}
        config = Config(data=SimpleNamespace(transform=SimpleNamespace(model=SimpleNamespace(vlm_history_frames=6))),
                        train=SimpleNamespace(train_batch_size=4, output_dir='/formal', resume_from_checkpoint=None),
                        model=SimpleNamespace(state_drop_prob=0), exp='formal')
        plan = dict(gpu_count=8, per_gpu_batch=4, effective_global_batch=32, source_step=38000)
        original, old_plan = deepcopy(config.model_dump()), deepcopy(plan)
        new_config, new_plan = probe.probe_configuration(config, plan, Path('/probe'))
        self.assertEqual(config.model_dump(), original); self.assertEqual(plan, old_plan)
        self.assertEqual(new_plan['gpu_count'], 1); self.assertEqual(new_plan['effective_global_batch'], 4)
        self.assertEqual(new_config.train.output_dir, '/probe/disposable_trainer')
        plan['per_gpu_batch'] = 8
        with self.assertRaises(ValueError): probe.probe_configuration(config, plan, Path('/probe'))


def batch_fixture():
    episode = SimpleNamespace(video_timestamp=np.arange(60, dtype=float) / 30,
                              source_frame_index=np.arange(60), frame_index=np.arange(60))
    class History:
        pack_ids = np.zeros(4, dtype=int)
        episode_ids = np.zeros(4, dtype=int)
        def selected_cache_rows(self, row): return np.arange(row, row + 31, 6)
        def _episode(self, pack, episode_id): return episode
        def get_history(self, row): return np.full((6, 5, 1024), row, np.float32)
    index = History()
    subset = SimpleNamespace(indices=list(range(4)), dataset=SimpleNamespace(raw_dataset=SimpleNamespace(motion_history=index)))
    samples = []
    for row in range(4):
        rows = index.selected_cache_rows(row)
        metadata = dict(contract='psi-qwen-motion-aligned-rgb-history-v1', pack_id=0, episode_index=0,
                        selected_cache_rows=rows.tolist(), current_is_last=True,
                        video_timestamps_s=episode.video_timestamp[rows].tolist(),
                        source_frame_indices=rows.tolist(), frame_indices=rows.tolist())
        samples.append(dict(vlm_history_metadata=metadata, motion_img_embs=index.get_history(row)))
    mask = torch.ones(4, 30, 80, dtype=torch.bool); mask[:, :, 78:] = False
    batch = dict(actions=torch.zeros(4, 30, 80), states=torch.zeros(4, 1, 45),
                 motion_img_embs=torch.zeros(4, 6, 5, 1024), image_grid_thw=torch.ones(24, 3, dtype=torch.long),
                 observations=torch.zeros(4, 6, 8, 8, 3), vlm_history_frames=torch.full((4,), 6),
                 actions_mask=mask, input_ids=torch.ones(4, 100, dtype=torch.long), pixel_values=torch.zeros(24, 8),
                 vlm_history_cache_rows=torch.tensor([s['vlm_history_metadata']['selected_cache_rows'] for s in samples]),
                 vlm_history_video_timestamps_s=torch.tensor([s['vlm_history_metadata']['video_timestamps_s'] for s in samples], dtype=torch.float64))
    return samples, batch, subset


class ActualBatchGateTests(unittest.TestCase):
    def test_four_real_six_frame_selectors_and_24_grids(self):
        samples, batch, subset = batch_fixture()
        result = probe.validate_batch(samples, batch, subset, list(range(4)))
        self.assertEqual(result['actual_batch_size'], 4); self.assertEqual(result['qwen_grids'], 24)
        self.assertEqual(result['samples'][3]['selected_cache_rows'], [3, 9, 15, 21, 27, 33])

    def test_wrong_episode_cache_rows_motion_clock_grid_or_action_mask_rejected(self):
        mutations = [lambda s, b: s[0]['vlm_history_metadata'].update(episode_index=1),
                     lambda s, b: s[0]['vlm_history_metadata'].update(selected_cache_rows=[0] * 6),
                     lambda s, b: s[0]['vlm_history_metadata']['video_timestamps_s'].__setitem__(0, .02),
                     lambda s, b: s[0].update(motion_img_embs=np.ones((6, 5, 1024))),
                     lambda s, b: b.update(image_grid_thw=torch.ones(4, 3)),
                     lambda s, b: b['actions_mask'].__setitem__((0, 0, 79), True),
                     lambda s, b: b['vlm_history_cache_rows'].__setitem__((0, 0), 1)]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                samples, batch, subset = batch_fixture(); mutate(samples, batch)
                with self.assertRaises((ValueError, AssertionError)):
                    probe.validate_batch(samples, batch, subset, list(range(4)))


def tiny_full_model():
    model = torch.nn.Module()
    model.vlm_model = torch.nn.Module()
    model.vlm_model.language = torch.nn.Linear(2, 2)
    model.vlm_model.visual = torch.nn.Linear(2, 2)
    model.vlm_model.merger = torch.nn.Linear(2, 2)
    model.motion_conditioner = torch.nn.Linear(2, 2)
    model.action_header = torch.nn.Linear(2, 2)
    for parameter in model.parameters(): parameter.grad = torch.ones_like(parameter)
    return model


class FullGradientGateTests(unittest.TestCase):
    @staticmethod
    def group(name):
        return 'mm_projector' if 'merger' in name else 'vision_tower' if 'visual' in name else 'lang_backbone'

    def test_all_five_trainable_groups_and_inactive_expert_names_reported(self):
        model = tiny_full_model()
        model.motion_conditioner.register_parameter('inactive_expert', torch.nn.Parameter(torch.zeros(2)))
        result = probe.gradient_summary(model, self.group)
        self.assertEqual(set(result), set(probe.GROUPS))
        self.assertEqual(result['motion']['missing_gradient_names'], ['motion_conditioner.inactive_expert'])
        self.assertTrue(all(row['all_existing_gradients_finite'] and row['gradient_l2_norm'] > 0 for row in result.values()))

    def test_nonfinite_zero_missing_or_unrecognized_components_rejected(self):
        for corruption in ('nan', 'inf', 'zero', 'missing', 'unrecognized'):
            with self.subTest(corruption=corruption):
                model = tiny_full_model()
                if corruption in ('nan', 'inf'):
                    model.vlm_model.visual.weight.grad[0, 0] = float(corruption)
                elif corruption == 'zero':
                    for parameter in model.motion_conditioner.parameters(): parameter.grad.zero_()
                elif corruption == 'missing':
                    for parameter in model.vlm_model.merger.parameters(): parameter.grad = None
                else:
                    model.extra = torch.nn.Linear(2, 2)
                with self.assertRaises(ValueError): probe.gradient_summary(model, self.group)


if __name__ == '__main__': unittest.main()
