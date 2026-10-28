"""CPU checks for preserving fixed Mobile cases and the actual CLI budget."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import gymnasium as gym
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_closed_loop import (MOBILE_PP_TASK, episode_reset_options, episode_termination,
                             mobile_official_protocol, mobile_time_limit, parse_args)


class CounterEnv(gym.Env):
    observation_space = gym.spaces.Box(0, 2000, (1,), dtype=np.int32)
    action_space = gym.spaces.Discrete(1)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return np.array([0], dtype=np.int32), {}

    def step(self, action):
        self.steps += 1
        return np.array([self.steps], dtype=np.int32), 0., False, False, {}


class MobileProtocolTests(unittest.TestCase):
    def config(self):
        return {'uid': 'g1_wholebody_locomotion_pick_between_tables_teleop',
                'metadata': {'max_episode_steps': 1200, 'render_hz': 50,
                             'physics_dt': .005, 'success_criteria': .4, 'split': 'train'},
                'dr_state_dict': {'materials': {'fixed': 1}}}

    def test_fixed_mobile_reset_does_not_resample_dr0_or_rewrite_archive(self):
        config = self.config(); before = deepcopy(config)
        options = episode_reset_options(config, 0, recorded_mobile=True)
        self.assertIs(options['state_dict'], config)
        self.assertIsNone(options['dr_level'])
        protocol = mobile_official_protocol(config)
        self.assertEqual(protocol['source_metadata_success_criteria'], .4)
        self.assertEqual(protocol['effective_success_criteria'], .9)
        self.assertEqual(protocol['source_metadata_split'], 'train')
        self.assertEqual(config, before)

    def test_prior_dr_and_recorded_handover_semantics_are_retained(self):
        config = self.config()
        self.assertEqual(episode_reset_options(config, 1)['dr_level'], 1)
        self.assertIsNone(episode_reset_options(config, 0, recorded_handover=True)['dr_level'])

    def test_full_budget_includes_warmup_and_truncates_at_exact1200_steps(self):
        env = mobile_time_limit(CounterEnv()); env.reset()
        for step in range(1, 1201):
            _, _, terminated, truncated, _ = env.step(0)
            self.assertFalse(terminated)
            self.assertEqual(truncated, step == 1200)
        self.assertEqual(env.unwrapped.steps, 1200)  # The first50 warmup steps are not a free extra budget.
        env.reset()
        self.assertFalse(env.step(0)[3])

    def test_native_success_terminates_without_additional_geometry_or_release_gate(self):
        episode = {'task_success_observed': False, 'fell': True, 'policy_frames': 0}
        self.assertEqual(episode_termination(episode, True, False, 2.3), 'task_terminated')
        self.assertTrue(episode['task_success_observed'])
        self.assertEqual(episode['first_task_success_sim_time_s'], 2.3)

    def test_wrong_task_or_time_contract_is_rejected(self):
        for field, value in [('max_episode_steps', 15000), ('render_hz', 30), ('physics_dt', .002)]:
            config = self.config(); config['metadata'][field] = value
            with self.assertRaises(ValueError): mobile_official_protocol(config)
        config = self.config(); config['uid'] = 'g1_wholebody_handover_teleop'
        with self.assertRaises(ValueError): mobile_official_protocol(config)

    def test_official_flag_refuses_extra_fall_gate_or_nonmobile_worker(self):
        with tempfile.TemporaryDirectory() as temp:
            base = ['run_closed_loop.py', '--data-root', temp, '--sonic-resources', temp,
                    '--output', temp + '/out', '--tasks', MOBILE_PP_TASK,
                    '--seconds', '24', '--mobile-official-protocol']
            with patch.object(sys, 'argv', base):
                self.assertTrue(parse_args().mobile_official_protocol)
            for extra in (['--stop-on-fall'], ['--worker-task', 'G1WholebodyHandoverTeleop-v0']):
                with patch.object(sys, 'argv', base + extra), patch('sys.stderr'):
                    with self.assertRaises(SystemExit): parse_args()


if __name__ == '__main__':
    unittest.main()
