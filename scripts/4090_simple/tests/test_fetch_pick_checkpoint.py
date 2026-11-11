"""Exercise the real remote gate/byte stream without SSH, CUDA or large files."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import fetch_pick_checkpoint as fetch


class RemoteTransferTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.profile = 'xmove_bend_pick'
        self.prepared = self.root/'runs'/self.profile
        self.run = self.prepared/'training/posttrain'/(
            self.profile+'_joint38000_statekeep_bs4_8gpu.us.flow1000.cosine.lr1.0e-05.b32.gpus8.261111155500')
        self.release = self.root/'release'
        self.prepared.mkdir(parents=True)
        self.run.mkdir(parents=True)
        self.state = dict(state='complete',returncode=0,task_id='cvpr/G1SonicXMoveBendPickPico-v0',
            max_optimizer_steps=10000,run_directory=str(self.run))
        self.put(self.prepared/'training_state.json', self.state)
        self.put(self.root/'QUEUE_PLAN.json', {'authorized':True})
        self.retention = dict(all_models_preserved=True,saved_update=10000,all_saved_model_steps=list(fetch.SAVE_STEPS),task_id=self.state['task_id'])
        self.put(self.run/'checkpoint_retention.json', self.retention)
        for name in ('run_config.json','argv.txt','dataset_statistics.json','task_initialization.json',
                'task_window_selection.json','model_config_state_keep.json','gpu_check_update10000.json',
                *(f'state_policy_migration_rank_{i}.json'for i in range(8))):
            self.put(self.run/name, {'fixture':name})
        for name in ('run_config.json','argv.txt','preparation.json'):
            self.put(self.prepared/name, {'fixture':name})
        for name in ('packed/PACK_READY.json','dino/manifest.json','A100_PICO_DATASET_READY.json','dino/REMOTE_DINO_RELOCATION.json'):
            self.put(self.release/name, {'fixture':name})
        self.model = self.run/'checkpoints/ckpt_5000/model.safetensors'
        self.model.parent.mkdir(parents=True)
        self.payload = bytes(range(256))*257
        self.model.write_bytes(self.payload)
        self.selected = dict(profile=self.profile,step=5000,task_id=self.state['task_id'],prepared=str(self.prepared),
            release=str(self.release),queue_root=str(self.root),queue_plan_sha256=fetch.digest(self.root/'QUEUE_PLAN.json'),
            expected_project=None,bytes=len(self.payload),checkpoint_steps=list(fetch.SAVE_STEPS))

    @staticmethod
    def put(path, value):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value))

    def snapshot(self, selected=None):
        return subprocess.run([sys.executable,'-B','-c',fetch.SNAPSHOT],input=json.dumps(selected or self.selected),
            capture_output=True,text=True,timeout=15)

    def test_completed_nonfinal_save_has_exact24_metadata_and_stable_identity(self):
        first, second = self.snapshot(), self.snapshot()
        self.assertEqual(first.returncode,0,first.stderr)
        self.assertEqual(second.returncode,0,second.stderr)
        a,b = json.loads(first.stdout),json.loads(second.stdout)
        self.assertEqual(a,b)
        self.assertEqual(len(a['metadata']),24)
        self.assertEqual(a['model_stat'],fetch.identity(self.model))
        self.assertEqual(a['model_path'],str(self.model))
        self.assertTrue(a['checkpoint_save_completed'])

    def test_incomplete_or_wrong_task_and_unretained_save_rejected_before_stream(self):
        for changes in ({'state':'running'},{'returncode':1},{'task_id':'cvpr/G1SonicXMovePickPico-v0'}):
            self.put(self.prepared/'training_state.json',dict(self.state,**changes))
            result=self.snapshot();self.assertNotEqual(result.returncode,0);self.assertEqual(result.stdout,'')
        self.put(self.prepared/'training_state.json',self.state)
        self.put(self.run/'checkpoint_retention.json',dict(self.retention,all_saved_model_steps=[5000]))
        self.assertNotEqual(self.snapshot().returncode,0)

    def test_foreign_run_queue_or_model_size_rejected(self):
        for key,value in [('expected_project','different_run'),('queue_plan_sha256','0'*64),('bytes',1),('profile','xmove_pick')]:
            result=self.snapshot(dict(self.selected,**{key:value}))
            self.assertNotEqual(result.returncode,0,key)
            self.assertEqual(result.stdout,'')

    def test_real_stream_hashes_exact_bytes_and_refuses_changed_source(self):
        selected=dict(model=str(self.model),expected_stat=fetch.identity(self.model),bytes=len(self.payload))
        result=subprocess.run([sys.executable,'-B','-c',fetch.STREAM,json.dumps(selected)],capture_output=True,timeout=15)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(result.stdout,self.payload)
        receipt=json.loads(result.stderr.decode().removeprefix('CVPR_MODEL_RECEIPT '))
        self.assertEqual(receipt['sha256'],hashlib.sha256(self.payload).hexdigest())
        self.assertEqual(receipt['stat_before'],receipt['stat_after'])
        self.model.write_bytes(b'changed')
        rejected=subprocess.run([sys.executable,'-B','-c',fetch.STREAM,json.dumps(selected)],capture_output=True,timeout=15)
        self.assertNotEqual(rejected.returncode,0)
        self.assertEqual(rejected.stdout,b'')

    def test_cli_selection_rejects_unsafe_profile_or_unselected_step(self):
        for profile,step in [('../other',5000),('xmove_pick',True),('xmove_pick',5001)]:
            with self.assertRaises(ValueError):fetch.selection(profile,step)


if __name__ == '__main__':
    unittest.main()
