"""No network: temporary byte-transfer scripts and CPU-ready gate regressions."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import upload_pico_handover as upload
import verify_pico_handover as verify


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def fixture(root, *, task_id=None):
    for name in ('packed','dino'):(root/name).mkdir(parents=True)
    profile = upload.TASK_PROFILES[task_id or upload.HANDOVER_TASK_ID]
    pack = {'contract':profile['pack_contract'],'final_ready':False}
    if task_id is not None:pack['task_id'] = task_id
    write(root/'packed/PACK_READY.json',pack)
    (root/'dino/features.npy').write_bytes(b'unit test bytes, not real DINO')
    files = {name+'/'+relative:value for name in ('packed','dino')
        for relative,value in verify.tree_inventory(root/name).items()}
    ready = {'contract':profile['ready_contract'],'status':'passed','production_ready_for_bound_pack_and_cache':True,
        'source_kind':'live_pico_human','split_restriction':'train_only','normalization_refitted':False,
        'GPU_used':False,'CUDA_initialized':False,'training_started':False,'release_root':str(root),
        'verifier_sha256':verify.sha(verify.__file__),
        'payload_files':files,'pack_ready_sha256':verify.sha(root/'packed/PACK_READY.json'),
        'train_episodes':3,'train_windows':100}
    if task_id is not None:ready['task_id'] = task_id
    write(root/verify.READY_NAME,ready)
    return ready


class UploadPicoTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)/'release';self.root.mkdir()
        self.remote = '/gaozt-test1/datasets/SIMPLE_sonic_v1_1/pico_handover_unit_test'

    def test_requires_real_cpu_ready_marker_and_exact_bound_input_bytes(self):
        fixture(self.root)
        value = upload.inventory(self.root,self.remote)
        self.assertFalse(value['remote_production_ready'])
        self.assertTrue(value['DINO_relocation_pending'])
        self.assertIn(verify.READY_NAME,value['files'])
        feature = self.root/'dino/features.npy';before = feature.stat()
        feature.write_bytes(b'x'*before.st_size)
        os.utime(feature,ns=(before.st_atime_ns,before.st_mtime_ns))
        with self.assertRaisesRegex(ValueError,'identities changed'):upload.inventory(self.root,self.remote)
        self.assertFalse((self.root/'A100_PICO_DATASET_READY.json').exists())

    def test_missing_or_unpassed_ready_cannot_initiate_any_network(self):
        with patch.object(upload.subprocess,'run') as run:
            with self.assertRaises(ValueError):upload.transfer(self.root,self.remote)
            run.assert_not_called()
        fixture(self.root)
        path = self.root/verify.READY_NAME;ready = verify.read(path);ready['status'] = 'pending';write(path,ready)
        with patch.object(upload.subprocess,'run') as run:
            with self.assertRaises(ValueError):upload.transfer(self.root,self.remote)
            run.assert_not_called()

    def test_fresh_target_namespace_and_aliases_are_rejected(self):
        fixture(self.root)
        for path in ('relative','/gaozt-test1/datasets/SIMPLE_sonic_v1_1/handover_merged114_20261025',
                     '/tmp/pico_handover_new','/gaozt-test1/datasets/SIMPLE_sonic_v1_1/pico_handover_../other'):
            with self.assertRaises(ValueError):upload.remote_target(path)
        (self.root/'dino/aliased').symlink_to(self.root/'packed',target_is_directory=True)
        with self.assertRaisesRegex(ValueError,'aliases'):upload.inventory(self.root,self.remote)

    def execute(self,code,value,check=True):
        return subprocess.run([sys.executable,'-B','-c',code],input=json.dumps(value),
            text=True,capture_output=True,check=check)

    def test_actual_remote_cpu_scripts_preserve_sha_and_restore_nanosecond_mtime(self):
        fixture(self.root)
        value = upload.inventory(self.root,self.remote)
        target = Path(self.tmp.name)/'remote';value['remote_root'] = str(target)
        self.execute(upload.REMOTE_PREPARE,value)
        for relative,wanted in value['files'].items():
            path = target/relative;path.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(self.root/relative,path)
            os.utime(path,ns=(1_700_000_000_000_000_000,1_700_000_000_123_456_789))
        result = self.execute(upload.REMOTE_VERIFY,value)
        report = json.loads(result.stdout)
        self.assertEqual(report['status'],'all_copied_bytes_and_identities_verified')
        self.assertFalse(report['remote_production_ready'])
        self.assertTrue(report['DINO_relocation_pending'])
        for relative,wanted in value['files'].items():
            self.assertEqual(verify.identity(target/relative),wanted)
        self.execute(upload.REMOTE_PREPARE,value)
        self.execute(upload.REMOTE_VERIFY,value)  # Exact pending transfer may resume idempotently.
        self.assertFalse((target/'A100_PICO_DATASET_READY.json').exists())
        before = verify.sha(target/'A100_BYTE_VERIFICATION.json')
        (target/'dino/features.npy').write_bytes(b'corrupt')
        self.assertNotEqual(self.execute(upload.REMOTE_VERIFY,value,check=False).returncode,0)
        self.assertEqual(verify.sha(target/'A100_BYTE_VERIFICATION.json'),before)

    def test_remote_prepare_does_not_reuse_other_directory_or_completed_release(self):
        fixture(self.root)
        value = upload.inventory(self.root,self.remote)
        target = Path(self.tmp.name)/'remote';target.mkdir();(target/'existing-training.txt').write_text('untouched')
        value['remote_root'] = str(target)
        self.assertNotEqual(self.execute(upload.REMOTE_PREPARE,value,check=False).returncode,0)
        self.assertEqual((target/'existing-training.txt').read_text(),'untouched')
        (target/'existing-training.txt').unlink();target.rmdir()
        self.execute(upload.REMOTE_PREPARE,value)
        write(target/'A100_PICO_DATASET_READY.json',{'status':'passed'})
        self.assertNotEqual(self.execute(upload.REMOTE_PREPARE,value,check=False).returncode,0)

    def test_mobile_requires_exact_task_and_its_own_pack_ready_contract(self):
        fixture(self.root,task_id=upload.MOBILE_TASK_ID)
        remote = '/gaozt-test1/datasets/SIMPLE_sonic_v1_1/pico_mobile_pp_20261027_human100_01'
        value = upload.inventory(self.root,remote)
        self.assertEqual(value['task_id'],upload.MOBILE_TASK_ID)
        self.assertEqual(value['contract'],upload.MOBILE_CONTRACT)
        self.assertFalse(value['remote_production_ready'])
        self.assertTrue(value['DINO_relocation_pending'])
        with self.assertRaisesRegex(ValueError,'task-matched'):
            upload.inventory(self.root,self.remote)
        with self.assertRaises(ValueError):upload.remote_target(remote,task_id=upload.HANDOVER_TASK_ID)
        path = self.root/verify.READY_NAME
        ready = verify.read(path);del ready['task_id'];write(path,ready)
        with self.assertRaisesRegex(ValueError,'Explicit supported task'):upload.inventory(self.root,remote)
        ready['task_id'] = upload.HANDOVER_TASK_ID;write(path,ready)
        with self.assertRaisesRegex(ValueError,'contract differ'):upload.inventory(self.root,remote)
        ready['task_id'] = upload.MOBILE_TASK_ID;write(path,ready)
        pack_path = self.root/'packed/PACK_READY.json'
        for pack in ({'contract':upload.TASK_PROFILES[upload.HANDOVER_TASK_ID]['pack_contract'],'final_ready':False},
                     {'contract':upload.TASK_PROFILES[upload.MOBILE_TASK_ID]['pack_contract'],'final_ready':False}):
            write(pack_path,pack)
            ready['payload_files']['packed/PACK_READY.json'] = verify.identity(pack_path)
            ready['pack_ready_sha256'] = verify.sha(pack_path);write(path,ready)
            with self.assertRaisesRegex(ValueError,'pack contract'):upload.inventory(self.root,remote)

    def test_handover_legacy_marker_stays_in_handover_namespace(self):
        fixture(self.root)
        value = upload.inventory(self.root,self.remote)
        self.assertEqual(value['contract'],upload.CONTRACT)
        self.assertEqual(value['task_id'],upload.HANDOVER_TASK_ID)
        with self.assertRaises(ValueError):
            upload.inventory(self.root,'/gaozt-test1/datasets/SIMPLE_sonic_v1_1/pico_mobile_pp_invalid_reuse')
        path = self.root/verify.READY_NAME;ready = verify.read(path)
        ready['task_id'] = upload.MOBILE_TASK_ID;write(path,ready)
        with self.assertRaisesRegex(ValueError,'contract differ'):upload.inventory(self.root,self.remote)

    def test_mobile_remote_scripts_verify_actual_bytes_and_do_not_promote_readiness(self):
        fixture(self.root,task_id=upload.MOBILE_TASK_ID)
        value = upload.inventory(self.root,'/gaozt-test1/datasets/SIMPLE_sonic_v1_1/pico_mobile_pp_unit_test')
        target = Path(self.tmp.name)/'mobile_remote';value['remote_root'] = str(target)
        self.execute(upload.REMOTE_PREPARE,value)
        for relative in value['files']:
            path = target/relative;path.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(self.root/relative,path)
        report = json.loads(self.execute(upload.REMOTE_VERIFY,value).stdout)
        self.assertEqual(report['contract'],upload.MOBILE_BYTE_CONTRACT)
        self.assertEqual(report['task_id'],upload.MOBILE_TASK_ID)
        self.assertFalse(report['remote_production_ready'])
        for relative,wanted in value['files'].items():self.assertEqual(verify.identity(target/relative),wanted)
        self.assertFalse((target/'A100_PICO_DATASET_READY.json').exists())
        previous = (target/'A100_BYTE_VERIFICATION.json').read_bytes()
        (target/'dino/features.npy').write_bytes(b'changed after copy')
        self.assertNotEqual(self.execute(upload.REMOTE_VERIFY,value,check=False).returncode,0)
        self.assertEqual((target/'A100_BYTE_VERIFICATION.json').read_bytes(),previous)
        invalid = {**value,'task_id':upload.HANDOVER_TASK_ID}
        self.assertNotEqual(self.execute(upload.REMOTE_PREPARE,invalid,check=False).returncode,0)

    def test_remote_launcher_limits_only_its_process_and_preserves_stdin_json(self):
        value = {'exact':'stdin payload'}
        with patch.object(upload.subprocess,'run') as run:
            upload.remote_python('115.191.40.142','/absolute/python','print("not executed")',value)
        args,kwargs = run.call_args
        self.assertEqual(json.loads(kwargs['input']),value)
        command = args[0][-1]
        for token in ('CUDA_VISIBLE_DEVICES=', 'nice -n 19', 'os.sched_setaffinity', '<pico-cpu-remote>'):
            self.assertIn(token,command)
        parent_affinity = os.sched_getaffinity(0)
        parent_nice = os.getpriority(os.PRIO_PROCESS,0)
        result = subprocess.run(['env','CUDA_VISIBLE_DEVICES=','nice','-n','19',sys.executable,'-B','-c',
            upload.REMOTE_CPU_LAUNCHER,
            'import os,json; print(json.dumps({"cpus":sorted(os.sched_getaffinity(0)),"nice":os.getpriority(os.PRIO_PROCESS,0),"gpu":os.environ.get("CUDA_VISIBLE_DEVICES")}))'],
            text=True,capture_output=True,check=True)
        runtime = json.loads(result.stdout)
        self.assertEqual(len(runtime['cpus']),1)
        self.assertEqual(runtime['nice'],19)
        self.assertEqual(runtime['gpu'],'')
        self.assertEqual(os.sched_getaffinity(0),parent_affinity)
        self.assertEqual(os.getpriority(os.PRIO_PROCESS,0),parent_nice)


if __name__=='__main__':unittest.main()
