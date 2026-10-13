"""Ensure incompatible optimizer/EMA payloads fail before training resumes."""
import copy
import unittest
import torch
from MotionExpert.ema import EMAOptimizer


class DenseEMAResumeTests(unittest.TestCase):
    def make(self):
        parameter = torch.nn.Parameter(torch.ones(3))
        optimizer = EMAOptimizer(torch.optim.AdamW(
            [{'params': [parameter], 'phase': 'unfrozen'}], lr=.01),
            device=torch.device('cpu'), decay=.9)
        parameter.square().sum().backward()
        optimizer.step()
        optimizer.join()
        return parameter, optimizer

    def test_resume_retains_correct_ema_and_optimizer(self):
        _, source = self.make()
        _, destination = self.make()
        payload = source.state_dict()
        destination.load_state_dict(payload)
        torch.testing.assert_close(destination.ema_params[0], source.ema_params[0])
        self.assertEqual(destination.current_step, source.current_step)

    def test_wrong_tensor_count_shape_or_phase_is_rejected(self):
        _, source = self.make()
        payload = source.state_dict()
        bad_payloads = []
        missing = copy.deepcopy(payload)
        missing['ema'] = ()
        bad_payloads.append(missing)
        wrong_ema = copy.deepcopy(payload)
        wrong_ema['ema'] = (torch.ones(4),)
        bad_payloads.append(wrong_ema)
        wrong_phase = copy.deepcopy(payload)
        wrong_phase['opt']['param_groups'][0]['phase'] = 'text'
        bad_payloads.append(wrong_phase)
        wrong_momentum = copy.deepcopy(payload)
        next(iter(wrong_momentum['opt']['state'].values()))['exp_avg'] = torch.ones(4)
        bad_payloads.append(wrong_momentum)
        for payload in bad_payloads:
            with self.subTest(payload=payload):
                _, destination = self.make()
                before = destination.ema_params[0].clone()
                with self.assertRaises(ValueError):
                    destination.load_state_dict(payload)
                torch.testing.assert_close(destination.ema_params[0], before)


if __name__ == '__main__':
    unittest.main()
