"""CPU-only saved-scene derivation and exact native Bend budget checks."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

import gymnasium as gym
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_closed_loop as loop
import pico_bend_pick_task as adapter
from test_pico_bend_pick_task import fixture as training_fixture
from test_mobile_eval_protocol import CounterEnv


def fixture():
    state = training_fixture()
    state['metadata'].update(max_episode_steps=400, control_hz=50, split='train')
    actors = state['layout']['actors']
    existing = actors.pop('distractor_6')
    actors['distractor_0'] = existing
    for index, uid in ((1, '7'), (2, '8')):
        actor = deepcopy(existing)
        actor['uid'] = actor['asset']['uid'] = uid
        actor['pose']['position'][0] += index*.1
        actors['distractor_'+str(index)] = actor
        state['dr_state_dict']['spatial'][uid] = deepcopy(actor['pose'])
    state['dr_state_dict']['distractors'] = ['6', '7', '8']
    state['dr_state_dict']['spatial']['99'] = deepcopy(existing['pose'])
    return state


class BendProtocolTests(unittest.TestCase):
    def test_unused_spatial_only_derivation_is_lossless_and_adapter_stays_strict(self):
        original = fixture(); before = deepcopy(original)
        derived, receipt = loop.derive_bend_eval_reset(original)
        self.assertEqual(original, before)
        self.assertEqual(receipt['removed_unused_spatial_uids'], ['99'])
        restored = deepcopy(derived)
        restored['dr_state_dict']['spatial'].update(receipt['removed_unused_spatial_entries'])
        self.assertEqual(restored, original)
        state, frame = adapter.adapt_official_config(derived)
        adapter.validate_adapted_config(state)
        self.assertEqual(frame['world_translation_m'], [0., 0., .405])
        self.assertEqual(derived['dr_state_dict']['material'], original['dr_state_dict']['material'])
        self.assertEqual(derived['metadata'], original['metadata'])
        changed = deepcopy(derived)
        changed['layout']['actors']['target']['pose']['position'][2] += .01
        with self.assertRaises(ValueError): adapter.adapt_official_config(changed)

    def test_missing_selected_actor_or_ambiguous_robot_is_never_dropped(self):
        for mutate in (lambda s: s['dr_state_dict']['spatial'].pop('7'),
            lambda s: s['layout']['actors'].pop('distractor_2'),
            lambda s: s['dr_state_dict']['spatial'].update(g1_other={}),
            lambda s: s['dr_state_dict'].update(distractors=['6', '7', '10'])):
            state = fixture(); mutate(state)
            with self.assertRaises(ValueError): loop.derive_bend_eval_reset(state)

    def test_native_budget_and_reset_do_not_resample_saved_dr(self):
        state = fixture(); protocol = loop.bend_official_protocol(state)
        self.assertEqual((protocol['max_episode_steps'], protocol['maximum_simulated_seconds']), (400, 8.))
        self.assertEqual(protocol['effective_success_criteria'], .9)
        self.assertEqual(protocol['source_metadata_split'], 'train')
        self.assertFalse(protocol['formal_evaluation_eligible'])
        self.assertIsNone(loop.episode_reset_options(state, 2, recorded_bend=True)['dr_level'])
        env = gym.wrappers.TimeLimit(CounterEnv(), max_episode_steps=400); env.reset()
        for index in range(400): self.assertEqual(env.step(0)[3], index == 399)
        self.assertEqual(env.unwrapped.steps, 400)  # Includes the first50 warmup ticks.
        episode = {'task_success_observed': False, 'fell': True, 'policy_frames': 0}
        self.assertEqual(loop.episode_termination(episode, True, False, .1), 'task_terminated')
        for field, value in [('max_episode_steps', 1200), ('physics_dt', .005), ('render_hz', 30)]:
            bad = fixture(); bad['metadata'][field] = value
            with self.assertRaises(ValueError): loop.bend_official_protocol(bad)

    def test_private_factory_uses_frozen_task_and_installs_frame_hook_before_reset(self):
        calls = []; native = ModuleType('simple.envs.sonic_loco_manip')
        def constructor(**kwargs):
            calls.append(('construct', kwargs))
            return CounterEnv()
        native.SonicLocoManipEnv = constructor
        with patch.dict(sys.modules, {'simple.envs.sonic_loco_manip': native}), \
            patch.object(adapter, 'make_bend_pick_task', side_effect=lambda config, **kw: ('private', kw)) as factory, \
            patch.object(adapter, 'install_frame_adapter', side_effect=lambda env: calls.append(('hook', env)) or {'installed': True}):
            env, receipt = loop.make_bend_environment({'dt': .005}, fixture(), sim_mode='mujoco_isaac',
                sync_observations=True, isaac_observation_only=True)
        self.assertEqual([row[0] for row in calls], ['construct', 'hook'])
        self.assertEqual(env._max_episode_steps, 400)
        self.assertEqual(calls[0][1]['task'][0], 'private')
        self.assertEqual(factory.call_args.kwargs['adapted_config']['robot_cfg']['uid'], 'g1_sonic')
        self.assertTrue(receipt['frame_binding']['installed'])
        env.close()

    def test_cli_forbids_mixed_tasks_and_foreign_initialization(self):
        with tempfile.TemporaryDirectory() as temp:
            base = ['run_closed_loop.py', '--data-root', temp, '--sonic-resources', temp, '--output', temp+'/out',
                '--tasks', loop.BEND_PICK_TASK, '--seconds', '8', '--bend-official-protocol',
                '--sync-observations', '--isaac-observation-only', '--sim-mode', 'mujoco_isaac']
            with patch.object(sys, 'argv', base): self.assertTrue(loop.parse_args().bend_official_protocol)
            invalid = [base+['--stop-on-fall'], [x for x in base if x != '--bend-official-protocol'],
                base+['--worker-task', loop.MOBILE_PP_TASK], base+['--mobile-official-protocol']]
            mixed = base.copy(); mixed.remove('--bend-official-protocol'); mixed.insert(mixed.index('--seconds'), loop.MOBILE_PP_TASK)
            invalid.append(mixed)
            for argv in invalid:
                with self.subTest(argv=argv), patch.object(sys, 'argv', argv), patch('sys.stderr'), self.assertRaises(SystemExit):
                    loop.parse_args()

    def test_actual_official30_metadata_derives_without_changing_material_or_real_actors(self):
        root = Path('/var/tmp/cvpr-bend-pick-ckpt5000-20261111/official_eval/data_root')
        if not root.is_dir(): self.skipTest('Local official metadata optional integration fixture absent')
        counts = []
        for level in range(3):
            for line in (root/'eval'/loop.BEND_PICK_SOURCE_TASK/f'dr-level-{level}/meta/episodes.jsonl').read_text().splitlines():
                source = json.loads(json.loads(line)['environment_config']); before = deepcopy(source)
                derived, receipt = loop.derive_bend_eval_reset(source)
                adapted, _ = adapter.adapt_official_config(derived); adapter.validate_adapted_config(adapted)
                self.assertEqual(source, before)
                self.assertEqual(derived['layout'], source['layout'])
                self.assertEqual(derived['dr_state_dict']['material'], source['dr_state_dict']['material'])
                counts.append((level, bool(receipt['removed_unused_spatial_uids'])))
        self.assertEqual(counts, [(level, level < 2) for level in range(3) for _ in range(10)])


if __name__ == '__main__': unittest.main()
