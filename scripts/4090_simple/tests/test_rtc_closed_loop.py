"""CPU contracts for explicit synchronous test-time RTC, not a GPU result."""
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_closed_loop import requested_rtc_contract,validate_server_info


def arguments(**updates):
    values=dict(execute_steps=15,rtc_mode="test_time",rtc_inference_delay=0,
        min_exec_horizon=15,pig_mask_schedule="exponential",pig_guidance_alpha=.9,
        expected_checkpoint_step=6000)
    return SimpleNamespace(**(values|updates))


def info(args):
    return dict(requested_rtc_contract(args),ckpt_step=6000,
        action=dict(action_dim=80,action_chunk_size=30,action_exec_horizon=args.execute_steps),
        online_history=dict(frames=6))


class RTCContractTests(unittest.TestCase):
    def test_explicit_test_time_runtime_fields_must_all_match(self):
        args=arguments();actual=info(args)
        receipt=validate_server_info(actual,args)
        self.assertEqual(receipt["status"],"passed")
        self.assertEqual(receipt["requested"],receipt["actual"])
        self.assertEqual(receipt["action_exec_horizon"],15)
        for key,value in [("rtc_mode","off"),("rtc_enabled",False),("rtc_exec_shift",30),
            ("rtc_inference_delay",1),("rtc_min_exec_horizon",16),("pig_mask_schedule","linear"),
            ("pig_guidance_alpha",.8),("num_inference_steps",20)]:
            changed=deepcopy(actual);changed[key]=value
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,"RTC configuration"):
                validate_server_info(changed,args)

    def test_checkpoint_horizon_and_causal_history_are_not_relaxed(self):
        args=arguments();actual=info(args)
        for key,value in [("action_dim",78),("action_chunk_size",15),("action_exec_horizon",30)]:
            changed=deepcopy(actual);changed["action"][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate_server_info(changed,args)
        for key,value in [("ckpt_step",10000),("online_history",dict(frames=1))]:
            changed=deepcopy(actual);changed[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate_server_info(changed,args)

    def test_original_off_mode_retains_30_frame_contract(self):
        args=arguments(rtc_mode="off",execute_steps=30,min_exec_horizon=None)
        receipt=validate_server_info(info(args),args)
        self.assertFalse(receipt["actual"]["rtc_enabled"])
        self.assertIsNone(receipt["actual"]["rtc_exec_shift"])
        self.assertEqual(receipt["action_exec_horizon"],30)

    def test_sync_rtc_rejects_misalignment_and_fictitious_wall_clock_delay(self):
        for update in [dict(execute_steps=10),dict(execute_steps=30),dict(rtc_inference_delay=1),
            dict(min_exec_horizon=14),dict(min_exec_horizon=31),dict(pig_guidance_alpha=float("nan"))]:
            with self.subTest(update=update),self.assertRaises(ValueError):requested_rtc_contract(arguments(**update))


if __name__=="__main__":unittest.main()
