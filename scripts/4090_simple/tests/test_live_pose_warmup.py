"""CPU contracts for the optional live-reference deployment warmup."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[3]/"VLMExpert/third_party/SIMPLE/src"))
from simple.baselines.live_pose_sonic_warmup import LivePoseSonicWarmup, reference_input
from simple.baselines.sonic_v1_1 import POLICY_TO_BODY
from simple.baselines.dual_source_psi0 import (BODY_JOINT_NAMES,SIMPLE_HAND_JOINT_NAMES,
    DualSourcePsi0Agent,HistoryWarmingUp)
from simple.core.action import ActionCmd


class Session:
    def __init__(self): self.inputs=[]
    def run(self,names,feed):
        self.inputs.append(next(iter(feed.values())).copy())
        return [np.full((1,64),.125,np.float32)]


class Controller:
    identity={"body_token_version":"body_token_v1_1","decoder_sha256":"a"*64,"provenance":"CPU fixture"}
    def __init__(self): self.reset()
    def reset(self): self._history=[]; self.steps=[]
    def hold(self,obs): return ActionCmd("pd")
    def step(self,token,hands,obs):
        self.steps.append((token.copy(),dict(hands)))
        return ActionCmd("latent")


def warmup_fixture(hand_policy="measured"):
    warmup=object.__new__(LivePoseSonicWarmup)
    warmup.hand_policy=hand_policy
    warmup.session=Session();warmup.input_name="obs_dict";warmup.output_name="tokens"
    warmup.reset()
    return warmup


def observation(index,offset=0):
    return {"sim_time_s":.005+.02*index,"joint_names":BODY_JOINT_NAMES+SIMPLE_HAND_JOINT_NAMES,
        "joint_qpos":np.arange(43,dtype=np.float32)*.001+offset,
        "head_stereo_left":np.full((2,3,3),index,dtype=np.uint8)}


class LivePoseWarmupTests(unittest.TestCase):
    def test_official_mode0_layout_absolute_q_zero_velocity_and_live_heading(self):
        q=np.arange(29)*.01
        yaw=.4;actual=np.array([np.cos(yaw/2),0,0,np.sin(yaw/2)])
        value=reference_input(q,[1,0,0,0],actual)
        np.testing.assert_allclose(value[0,4:294].reshape(10,29),np.tile(q[POLICY_TO_BODY],(10,1)))
        expected=np.array([np.cos(yaw),np.sin(yaw),-np.sin(yaw),np.cos(yaw),0,0])
        np.testing.assert_allclose(value[0,584:644].reshape(10,6),np.tile(expected,(10,1)),atol=1e-7)
        np.testing.assert_array_equal(value[0,:4],0)
        np.testing.assert_array_equal(value[0,294:584],0)
        np.testing.assert_array_equal(value[0,644:],0)
        with self.assertRaises(ValueError):reference_input(q,[0,0,0,0],actual)

    def test_waits_for_real_proprio_freezes_first_reference_and_reset_clears_it(self):
        warmup,controller=warmup_fixture(),Controller()
        robot=SimpleNamespace(mjData=SimpleNamespace(body=lambda name:SimpleNamespace(xquat=np.array([1.,0,0,0]))))
        first=observation(0)
        self.assertEqual(warmup.command(robot,controller,first).type,"pd")
        self.assertFalse(warmup.session.inputs)
        controller._history=list(range(10))
        self.assertEqual(warmup.command(robot,controller,observation(10,offset=.2)).type,"latent")
        np.testing.assert_array_equal(warmup.body_q,first["joint_qpos"][:29])
        self.assertEqual(warmup.receipt["initialization_count"],1)
        self.assertEqual(len(warmup.receipt["initial_live_reference_sha256"]),64)
        self.assertEqual(len(warmup.receipt["helper_source_sha256"]),64)
        self.assertEqual(warmup.phase,"warmup_live_pose_sonic_v1_1")
        warmup.reset()
        self.assertIsNone(warmup.body_q)
        self.assertEqual(warmup.receipt["events"],[])

    def test_agent_observes_every_real_frame_and_keeps_decoder_history_at_handoff(self):
        class Client:
            def __init__(self):self.frames=[]
            def reset(self,*args):pass
            def observe(self,images,*,history):self.frames.append(history["sim_time_s"])
            def query_action(self,images,instruction,state,*,history):
                if history["sim_time_s"]<1.005-1e-8:raise HistoryWarmingUp({"status":"warming_up"})
                return np.zeros((30,80),np.float32)
        warmup,controller,client=warmup_fixture(),Controller(),Client()
        robot=SimpleNamespace(mjData=SimpleNamespace(body=lambda name:SimpleNamespace(xquat=np.array([1.,0,0,0]))))
        agent=DualSourcePsi0Agent(robot,client=client,controller=controller,warmup_control=warmup)
        agent.reset(episode_id="one")
        for index in range(51):
            command=agent.get_action(observation(index))
            self.assertEqual(command.type,"pd" if index<10 else "latent")
            if index<50:controller._history.append(observation(index)["sim_time_s"])
        self.assertEqual(len(client.frames),51)
        self.assertEqual(warmup.receipt["encoder_calls"],40)
        self.assertEqual(len(controller.steps),41)
        self.assertEqual(len(controller._history),50)
        self.assertFalse(agent.warming_up)
        agent.reset(episode_id="two")
        self.assertEqual(controller._history,[])
        self.assertIsNone(warmup.body_q)


if __name__=="__main__":unittest.main()
