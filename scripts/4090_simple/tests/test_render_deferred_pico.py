"""CPU checks of deferred renderer's state and frame bridges, without RTX."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from render_deferred_pico import (assert_recorded_state_unchanged, canonical_event_sha256,
    restore_recorded_state, write_frame_bindings)


class DeferredRendererTests(unittest.TestCase):
    def test_full_state_roundtrip_never_forwards_or_steps_and_detects_non_qpos_mutation(self):
        model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body name="pelvis"><freejoint/>'
            '<geom type="sphere" size=".1"/><camera name="head_stereo_left"/></body></worldbody></mujoco>')
        data=mujoco.MjData(model)
        data.time=1.25
        data.qvel[:]=np.arange(model.nv)*.1
        data.qacc_warmstart[:]=np.arange(model.nv)+10.
        data.qfrc_applied[:]=np.arange(model.nv)+20.
        spec=mujoco.mjtState.mjSTATE_INTEGRATION
        state=np.empty(mujoco.mj_stateSize(model,spec))
        mujoco.mj_getState(model,data,state,spec)
        destination=mujoco.MjData(model)
        with mock.patch('mujoco.mj_forward',side_effect=AssertionError('No forward/solver')), \
                mock.patch('mujoco.mj_step',side_effect=AssertionError('No integration')):
            before=restore_recorded_state(model,destination,state,spec)
            assert_recorded_state_unchanged(model,destination,before,spec)
        self.assertTrue(np.array_equal(before,state))
        destination.qacc_warmstart[0]+=.01
        with self.assertRaisesRegex(ValueError,'full integration'):
            assert_recorded_state_unchanged(model,destination,before,spec)

    def event(self):
        return {'event_index':0,'capture_index30':0,'physical_tick50':0,'sim_time_s':1.,
            'nominal_elapsed_s':0.,'actual_elapsed_s':0.,'phase':'warmup','action_valid':False,
            'source_command_index':None,'joint_names':['joint'],'state43':[.1],
            'physical_joint_q43_float64':[.1],'physical_joint_qvel43':[.2],
            'root_pose7':[0.,0.,.8,1.,0.,0.,0.],'root_qvel6':[0.]*6,
            'objects':{'target':{'pose7':[0.,0.,.3,1.,0.,0.,0.]},
                       'container':{'pose7':[.2,0.,.3,1.,0.,0.,0.]}},'action':None,
            'integration_state_sha256':'a'*64,'source_fields':{'human':'原始示范'}}

    def test_canonical_hash_and_each_frame_state_rgb_bindings(self):
        source=self.event()
        expected=hashlib.sha256(json.dumps(source,ensure_ascii=False,allow_nan=False,
            sort_keys=True,separators=(',',':')).encode('utf-8')).hexdigest()
        self.assertEqual(canonical_event_sha256(source),expected)
        self.assertEqual(canonical_event_sha256(dict(reversed(list(source.items())))),expected)
        with tempfile.TemporaryDirectory() as directory:
            event=copy.deepcopy(source)
            event.pop('integration_state_sha256'); event.pop('source_fields')
            event['rgb']={'pixel_sha256':'b'*64,'file_sha256':'c'*64}
            path=Path(directory)/'events.jsonl'
            path.write_text(json.dumps(event)+'\n')
            bridge=Path(directory)/'rtx_frame_bindings.jsonl'
            receipt=write_frame_bindings(bridge,[source],path)
            self.assertTrue(receipt['full_integration_state_unchanged'])
            self.assertEqual(receipt['frame_bindings_count'],1)
            self.assertEqual(receipt['frame_bindings_sha256'],hashlib.sha256(bridge.read_bytes()).hexdigest())
            row=json.loads(bridge.read_text())
            self.assertEqual(row['source_event_sha256'],expected)
            self.assertEqual(row['rtx_event_sha256'],canonical_event_sha256(event))
            self.assertEqual(row['integration_state_sha256'],'a'*64)
            self.assertEqual(row['rgb_pixel_sha256'],'b'*64)
            self.assertEqual(row['rgb_file_sha256'],'c'*64)

    def test_bridge_rejects_changed_original_action_state_or_count(self):
        for field in ('action','root_pose7','physical_tick50','missing'):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                source=self.event(); event=copy.deepcopy(source)
                if field=='action':
                    event['action']={'body_token64':[.123456789012345]}
                elif field=='root_pose7':
                    event[field][0]+=.001
                elif field=='physical_tick50':
                    event[field]=2
                path=Path(directory)/'events.jsonl'
                path.write_text('' if field=='missing' else json.dumps(event)+'\n')
                with self.assertRaisesRegex(ValueError,'differ|changed'):
                    write_frame_bindings(Path(directory)/'bridge.jsonl',[source],path)


if __name__=='__main__':
    unittest.main()
