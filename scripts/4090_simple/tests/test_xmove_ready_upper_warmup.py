"""Meaningful candidate invariants with actual private MuJoCo FK, no GPU."""
import os,sys
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parents[1]
sys.path[:0]=[str(ROOT),str(REPO/'scripts/4090_simple'),str(REPO/'VLMExpert/third_party/SIMPLE/src'),str(REPO/'VLMExpert/third_party/GR00T-WholeBodyControl/gear_sonic_deploy')]
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np,mujoco
import xmove_ready_upper_warmup as c


class FixedReadyTests(unittest.TestCase):
 def test_fixed_universal_targets_and_real_tick_ramp(self):
  self.assertEqual(c.READY_JOINTS,{'left_elbow_joint':.6,'right_elbow_joint':.6,'left_shoulder_roll_joint':.3,'right_shoulder_roll_joint':-.3})
  value=[c.blend(t)for t in range(50)]
  self.assertEqual(value[:11],[0.]*11);self.assertEqual(value[-1],1.)
  self.assertTrue(np.all(np.diff(value)>=0));self.assertTrue(all(0<=x<=1 for x in value))
  for bad in [-1,50,True,.1]:
   with self.assertRaises(ValueError):c.blend(bad)

 def test_actual_private_fk_never_writes_live_qpos_qvel_ctrl_or_time_and_finishes_fixed_angles(self):
  names=(*c.BODY_JOINT_NAMES,*c.SIMPLE_HAND_JOINT_NAMES)
  joints=''.join('<body name="jbody_'+str(i)+'"><joint name="'+n+'" type="hinge" axis="0 1 0"/><geom type="sphere" size=".01" mass=".01" contype="0" conaffinity="0"/></body>'for i,n in enumerate(names))
  model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body name="pelvis"><freejoint/><geom type="sphere" size=".02" mass="1"/>'+joints+'<body name="left_wrist_yaw_link" pos=".1 .2 0"/><body name="right_wrist_yaw_link" pos=".1 -.2 0"/><body name="torso_link" pos="0 0 .2"/></body></worldbody></mujoco>')
  data=mujoco.MjData(model);data.qpos[:3]=[-.8,0,.79];data.qvel[:]=.0123;mujoco.mj_forward(model,data)
  warm=object.__new__(c.ReadyUpperSonicWarmup);warm.private=None;warm.initial_named=None;warm.tick=0;warm.receipt={}
  robot=SimpleNamespace(mjModel=model,mjData=data)
  obs=dict(joint_names=names,joint_qpos=np.zeros(43,np.float32),sim_time_s=.005)
  initial={n:getattr(data,n).copy()for n in ['qpos','qvel','ctrl','xpos','xquat']};time=data.time
  def parent(owner,*args):owner.tick+=1;return 'normal_reference_command'
  with patch.object(c._NeutralMode1Warmup,'command',parent):
   for tick in range(50):
    obs['sim_time_s']=.005+.02*tick
    self.assertEqual(warm.command(robot,object(),obs),'normal_reference_command')
  for n,v in initial.items():np.testing.assert_array_equal(getattr(data,n),v)
  self.assertEqual(data.time,time);self.assertIsNot(warm.private,data)
  for n,v in c.READY_JOINTS.items():self.assertAlmostEqual(warm.private.joint(n).qpos[0],v)
  self.assertEqual(len(warm.receipt['ready_reference_events']),50)
  self.assertTrue(all(x['live_qpos_writes']is False for x in warm.receipt['ready_reference_events']))
  self.assertEqual(warm.receipt['ready_reference_events'][-1]['blend'],1.)

if __name__=='__main__':unittest.main()
