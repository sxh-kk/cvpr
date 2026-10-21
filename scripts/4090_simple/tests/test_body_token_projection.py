"""CPU contracts for optional projection of denormalized learned body64."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[3]/"VLMExpert/third_party/SIMPLE/src"))
from simple.baselines.dual_source_psi0 import (BODY_JOINT_NAMES,SIMPLE_HAND_JOINT_NAMES,
    TRAINING_HAND_JOINT_NAMES,DualSourcePsi0Agent,V11_ENCODER_SHA256,project_body_token)
from simple.core.action import ActionCmd


class Controller:
    identity={"body_token_version":"body_token_v1_1","decoder_sha256":"a"*64,"provenance":"CPU fixture"}
    def __init__(self):self.calls=[]
    def reset(self):self.calls.clear()
    def step(self,token,hands,observation):
        self.calls.append((token.copy(),dict(hands)))
        return ActionCmd("fixture")


class Client:
    def __init__(self,action):self.action=action;self.frames=[]
    def reset(self,*args):pass
    def observe(self,images,*,history):self.frames.append(history)
    def query_action(self,*args,**kwargs):return np.tile(self.action,(30,1))


def observation(t):
    return {"joint_names":BODY_JOINT_NAMES+SIMPLE_HAND_JOINT_NAMES,"joint_qpos":np.zeros(43,np.float32),
        "head_stereo_left":np.zeros((2,3,3),np.uint8),"sim_time_s":t}


class BodyTokenProjectionTests(unittest.TestCase):
    def test_exact_32_level_endpoints_and_ties_to_even(self):
        raw=np.zeros(64,np.float32)
        raw[:12]=[-2,-1,-.999,-.09375,-.03125,.03125,.09375,.625,.75,.9375,.96,2]
        out=project_body_token(raw,"fsq_v1_1")
        np.testing.assert_array_equal(out[:12],[-1,-1,-1,-.125,0,0,.125,.625,.75,.9375,.9375,.9375])
        np.testing.assert_array_equal(out*16,np.rint(out*16))
        grid=np.linspace(-1,15/16,32,dtype=np.float32)
        np.testing.assert_array_equal(project_body_token(np.tile(grid,2),"fsq_v1_1"),np.tile(grid,2))
        self.assertEqual(out.dtype,np.float32)

    def test_nonfinite_complex_wrong_size_and_unknown_mode_rejected(self):
        for value in (np.full(64,np.nan),np.full(64,np.inf),np.zeros(63),np.ones(64,dtype=complex),np.full(64,1e300)):
            for mode in ("none","fsq_v1_1"):
                with self.subTest(shape=value.shape,mode=mode),self.assertRaises(ValueError):
                    project_body_token(value,mode)
        with self.assertRaises(ValueError):project_body_token(np.zeros(64),"legacy")

    def test_actual_controller_gets_projected_body_hands_and_chunk_remain_unchanged(self):
        action=np.arange(80,dtype=float)*.013-0.4
        action[:4]=[-1.5,.03125,.09375,.75]
        action[78:]=[99,-99]
        original=action.copy();controller=Controller();client=Client(action)
        agent=DualSourcePsi0Agent(None,client=client,controller=controller,body_token_projection="fsq_v1_1")
        agent.reset(episode_id="one");agent.get_action(observation(0.))
        token,hands=controller.calls[0]
        np.testing.assert_array_equal(token,project_body_token(action[:64],"fsq_v1_1"))
        self.assertEqual(hands,dict(zip(TRAINING_HAND_JOINT_NAMES,action[64:78])))
        np.testing.assert_array_equal(action,original)
        np.testing.assert_array_equal(agent._chunk[0],original)
        receipt=agent.projection_receipt
        self.assertEqual(receipt["mode"],"fsq_v1_1")
        self.assertEqual(receipt["encoder_sha256"],V11_ENCODER_SHA256)
        self.assertEqual(receipt["range"],[-1,15/16])
        self.assertEqual(receipt["step"],1/16)
        self.assertEqual(receipt["command_samples"],1)
        self.assertEqual(receipt["elements_compared"],64)
        self.assertEqual(receipt["range_clipped_elements"],1)
        self.assertGreater(receipt["changed_elements"],0)
        self.assertAlmostEqual(receipt["mean_absolute_change"],np.abs(token.astype(float)-action[:64].astype(np.float32)).mean())
        agent.reset(episode_id="two")
        self.assertEqual(agent.projection_receipt["command_samples"],0)
        self.assertEqual(agent.projection_receipt["absolute_change_sum"],0.)
        self.assertEqual(controller.calls,[])

    def test_default_none_keeps_previous_float32_body_exactly(self):
        action=np.linspace(-1.2,1.2,80,dtype=np.float32)
        controller=Controller();agent=DualSourcePsi0Agent(None,client=Client(action),controller=controller)
        agent.get_action(observation(0.))
        np.testing.assert_array_equal(controller.calls[0][0],action[:64])
        self.assertEqual(agent.projection_receipt["mode"],"none")
        self.assertEqual(agent.projection_receipt["changed_elements"],0)
        self.assertEqual(agent.projection_receipt["maximum_absolute_change"],0.)


if __name__=="__main__":unittest.main()
