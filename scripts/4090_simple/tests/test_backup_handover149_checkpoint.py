"""CPU-only backup regressions with independent SSH/rsync doubles; no real downloads."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

SCRIPTS=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SCRIPTS))
spec=importlib.util.spec_from_file_location('backup_handover149_checkpoint',SCRIPTS/'backup_handover149_checkpoint.py')
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
REAL_QUERY_SOURCE=b.query_source


class BackupTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.reference=self.root/'reference';self.reference.mkdir()
        self.output=self.root/'outputs'
        self.model=b'real-fixture-source-model-step-4000'
        self.authority={'contract':b.AUTHORITY_CONTRACT,'step':4000,'run_directory':b.EXPECTED_RUN,
            'checked_utc':'before','completed_retention_marker':{'all_models_preserved':True,
                'saved_update':10000,'all_saved_model_steps':list(range(1000,10001,1000))},
            'files':{},'read_policy':{'CUDA_VISIBLE_DEVICES':'','nice':19,'cpu_affinity':[0],
                'stdlib_only':True,'optimizer_or_rng_read':False}}
        for name,remote in b.SMALL_FILES.items():
            source=self.reference/name;source.parent.mkdir(parents=True,exist_ok=True)
            source.write_text('independent-source-metadata:'+name)
            self.authority['files'][name]={'source_path':remote,'bytes':source.stat().st_size,
                'sha256':b.sha256(source),'mtime_ns':1,'inode':2,'device':3}
        model=self.root/'source_model';model.write_bytes(self.model)
        self.authority['model']={'source_path':b.EXPECTED_RUN+'/checkpoints/ckpt_4000/model.safetensors',
            'bytes':len(self.model),'sha256':b.sha256(model),'mtime_ns':1,'inode':2,'device':3}
        self.authority['retention_marker']={'source_path':b.EXPECTED_RUN+'/checkpoint_retention.json',
            'bytes':1,'sha256':'a'*64}
        self.query=patch.object(b,'query_source',side_effect=lambda *_:deepcopy(self.authority)).start()
        self.addCleanup(patch.stopall)
        self.transfer=patch.object(b,'transfer_model',side_effect=self.transfer_ok).start()
        self.validate=patch.object(b,'validate_source',side_effect=self.validate_fixture).start()

    def transfer_ok(self, host, remote, local):
        self.assertEqual(remote,self.authority['model']['source_path'])
        Path(local).write_bytes(self.model)

    def validate_fixture(self, source, release, *, expected_step, expected_model_sha256):
        # The real production validator has its own exhaustive CPU contract suite.
        # Assert its actual inputs here and read actual staged bytes/receipt.
        self.assertEqual(Path(source).name,'ckpt_4000')
        self.assertEqual(expected_step,4000)
        self.assertEqual(b.sha256(Path(source)/'model.safetensors'),expected_model_sha256)
        receipt=b.read(Path(source)/'backup_receipt.json')
        self.assertEqual(receipt['step'],4000)
        self.assertFalse(receipt['optimizer_copied'])
        self.assertEqual(receipt['model']['source_sha256_before'],expected_model_sha256)
        self.assertEqual(receipt['model']['source_sha256_after'],expected_model_sha256)
        self.assertEqual(receipt['model']['local_sha256'],expected_model_sha256)
        self.assertEqual(Path(release),Path(source)/'release_metadata')

    def run_backup(self, **kwargs):
        return b.backup(4000,self.output,reference=self.reference,minimum_free_gib=0,**kwargs)

    @property
    def partial(self):
        return self.output/'.ckpt_4000.partial'/'ckpt_4000'

    def test_real_bytes_and_twice_independent_authority_are_committed_atomically(self):
        result=self.run_backup()
        final=self.output/'ckpt_4000'
        self.assertEqual(result,{'source':str(final),'expected_model_sha256':self.authority['model']['sha256'],
            'authority_path':str(final/'source_authority_before.json')})
        self.assertEqual((final/'model.safetensors').read_bytes(),self.model)
        self.assertEqual(self.query.call_count,2)
        self.assertEqual(self.validate.call_count,1)
        self.assertEqual(self.validate.call_args.args[0],self.partial)
        self.assertFalse(self.partial.exists())
        self.assertTrue((final/'source_authority_after.json').is_file())
        self.assertEqual(set(b.read(final/'backup_receipt.json')['metadata']),set(b.METADATA))
        self.assertFalse(any('optimizer' in p.name or 'random_states' in p.name for p in final.rglob('*')))

    def test_completed_owned_directory_reuse_requires_full_validation(self):
        result=self.run_backup();self.query.reset_mock();self.transfer.reset_mock();self.validate.reset_mock()
        self.assertEqual(self.run_backup(),result)
        self.query.assert_not_called();self.transfer.assert_not_called()
        self.validate.assert_called_once()
        self.assertEqual(self.validate.call_args.args[0],self.output/'ckpt_4000')

    def test_reuse_refuses_failed_full_validation(self):
        self.run_backup()
        self.validate.side_effect=ValueError('corrupted complete model')
        with self.assertRaisesRegex(ValueError,'corrupted complete'):
            self.run_backup()
        self.assertTrue((self.output/'ckpt_4000'/'model.safetensors').exists())

    def test_incomplete_remote_save_is_rejected_before_transfer(self):
        self.authority['completed_retention_marker']['all_saved_model_steps']=[1000,2000,3000]
        with self.assertRaisesRegex(ValueError,'completed checkpoint save'):
            self.run_backup()
        self.transfer.assert_not_called()
        self.assertFalse((self.output/'ckpt_4000').exists())

    def test_bad_local_sha_preserves_owned_partial_and_failure_evidence(self):
        self.transfer.side_effect=lambda _,__,local:Path(local).write_bytes(b'x'*len(self.model))
        with self.assertRaisesRegex(ValueError,'full-model SHA differs'):
            self.run_backup()
        self.assertTrue((self.partial/'model.safetensors').is_file())
        self.assertTrue(list((self.partial/'attempts').glob('*/failure.json')))
        self.assertFalse((self.output/'ckpt_4000').exists())
        self.assertEqual(self.query.call_count,1)

    def test_remote_after_sha_mismatch_preserves_model_and_stops_commit(self):
        after=deepcopy(self.authority);after['model']['sha256']='b'*64
        self.query.side_effect=[self.authority,after]
        with self.assertRaisesRegex(ValueError,'Remote source changed'):
            self.run_backup()
        self.assertEqual((self.partial/'model.safetensors').read_bytes(),self.model)
        self.validate.assert_not_called()
        self.assertFalse((self.output/'ckpt_4000').exists())

    def test_rsync_failure_can_resume_its_owned_partial(self):
        def interrupted(_,__,local):
            Path(local).write_bytes(self.model[:7]);raise RuntimeError('rsync interrupted')
        self.transfer.side_effect=interrupted
        with self.assertRaisesRegex(RuntimeError,'rsync interrupted'):
            self.run_backup()
        self.assertEqual((self.partial/'model.safetensors').read_bytes(),self.model[:7])
        self.transfer.side_effect=self.transfer_ok
        result=self.run_backup()
        self.assertEqual(Path(result['source']).name,'ckpt_4000')
        self.assertEqual(len(list((Path(result['source'])/'attempts').glob('*/source_authority_before.json'))),2)
        self.assertEqual(len(list((Path(result['source'])/'attempts').glob('*/failure.json'))),1)

    def test_full_source_validation_failure_happens_before_atomic_commit(self):
        self.validate.side_effect=ValueError('actual run identity mismatch')
        with self.assertRaisesRegex(ValueError,'run identity mismatch'):
            self.run_backup()
        self.assertTrue((self.partial/'model.safetensors').is_file())
        self.assertFalse((self.output/'ckpt_4000').exists())

    def test_foreign_nonempty_partial_is_not_adopted_or_deleted(self):
        staging=self.partial.parent;staging.mkdir(parents=True)
        foreign=staging/'someone_else_model';foreign.write_bytes(b'keep')
        with self.assertRaisesRegex(ValueError,'foreign nonempty'):
            self.run_backup()
        self.assertEqual(foreign.read_bytes(),b'keep')
        self.query.assert_not_called()

    def test_wrong_reference_metadata_sha_is_not_copied(self):
        (self.reference/'run_config.json').write_text('changed')
        with self.assertRaisesRegex(ValueError,'Reference differs from independent A100'):
            self.run_backup()
        self.transfer.assert_not_called()

    def test_source_cpu_and_gpu_policy_must_be_explicit(self):
        self.authority['read_policy']['CUDA_VISIBLE_DEVICES']='0'
        with self.assertRaisesRegex(ValueError,'hide GPU'):
            self.run_backup()
        self.transfer.assert_not_called()

    def test_ssh_query_uses_independent_program_and_fixed_run_file_mapping(self):
        completed=Mock(stdout=json.dumps(self.authority))
        with patch.object(b.subprocess,'run',return_value=completed)as run:
            authority=REAL_QUERY_SOURCE('115.191.40.142',4000)
        self.assertEqual(authority['model']['sha256'],self.authority['model']['sha256'])
        command=run.call_args.args[0]
        self.assertEqual(command[:3],['ssh','-o','BatchMode=yes'])
        self.assertIn('CUDA_VISIBLE_DEVICES',command[-1])
        self.assertIn(b.EXPECTED_RUN,command[-1])
        self.assertIn(str(b.PREPARED/'preparation.json'),command[-1])
        self.assertIn('sched_setaffinity',command[-1])
        self.assertTrue(run.call_args.kwargs['check'])
        self.assertTrue(run.call_args.kwargs['capture_output'])

    def test_unsupported_step_and_unsafe_ssh_host_are_rejected(self):
        for step in (5000,1000,True):
            with self.subTest(step=step),self.assertRaisesRegex(ValueError,'Only requested149'):
                b.backup(step,self.output,reference=self.reference)
        for host in ('-oProxyCommand=bad','host;echo bad','host/path'):
            with self.subTest(host=host),self.assertRaisesRegex(ValueError,'plain SSH host'):
                self.run_backup(ssh_host=host)
        self.query.assert_not_called()

    def test_space_requirement_includes_remaining_model_bytes_and_reserve(self):
        with patch.object(b.shutil,'disk_usage',return_value=Mock(free=12*1024**3+10)):
            b.check_space(self.root,12,10)
            with self.assertRaisesRegex(ValueError,'Insufficient free space'):
                b.check_space(self.root,12,11)
        for value in (-1,float('inf'),float('nan')):
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,'finite and nonnegative'):
                b.check_space(self.root,value)


class ProcessTests(unittest.TestCase):
    def test_rsync_uses_append_verification_and_only_explicit_model_path(self):
        with patch.object(b.subprocess,'run')as run:
            b.transfer_model('115.191.40.142','/saved/ckpt_4000/model.safetensors',Path('/owned/model.safetensors'))
        command=run.call_args.args[0]
        self.assertIn('--partial',command);self.assertIn('--append-verify',command)
        self.assertIn('115.191.40.142:/saved/ckpt_4000/model.safetensors',command)
        self.assertIn("CUDA_VISIBLE_DEVICES",next(x for x in command if x.startswith('--rsync-path=')))
        self.assertIn('sched_setaffinity',next(x for x in command if x.startswith('--rsync-path=')))

    def test_ssh_program_runs_stdlib_hashing_with_nice_affinity_and_completed_save(self):
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory);model=root/'checkpoints/ckpt_4000/model.safetensors'
            model.parent.mkdir(parents=True);model.write_bytes(b'actual-saved-source')
            metadata=root/'argv.txt';metadata.write_text('original argument tokens')
            marker={'saved_update':10000,'all_models_preserved':True,'all_saved_model_steps':[4000,10000]}
            (root/'checkpoint_retention.json').write_text(json.dumps(marker))
            request={'contract':b.AUTHORITY_CONTRACT,'step':4000,'run_directory':str(root),
                'small_files':{'argv.txt':str(metadata)}}
            result=subprocess.run([sys.executable,'-c',b.REMOTE_PROGRAM,json.dumps(request)],
                check=True,capture_output=True,text=True)
            authority=json.loads(result.stdout)
            self.assertEqual(authority['model']['sha256'],b.sha256(model))
            self.assertEqual(authority['files']['argv.txt']['sha256'],b.sha256(metadata))
            self.assertEqual(authority['read_policy']['CUDA_VISIBLE_DEVICES'],'')
            self.assertEqual(len(authority['read_policy']['cpu_affinity']),1)
            self.assertGreaterEqual(authority['read_policy']['nice'],19)

    def test_cli_returns_compact_last_line_contract(self):
        expected={'source':'/complete/ckpt_4000','expected_model_sha256':'a'*64,'authority_path':'/authority.json'}
        with patch.object(sys,'argv',['backup.py','--step','4000','--output-root','/new/root']),\
             patch.object(b,'backup',return_value=expected)as backup,patch('builtins.print')as printed:
            b.main()
        self.assertEqual(json.loads(printed.call_args.args[0]),expected)
        self.assertEqual(backup.call_args.kwargs['minimum_free_gib'],12)
        self.assertEqual(backup.call_args.kwargs['ssh_host'],'115.191.40.142')


if __name__ == '__main__':
    unittest.main()
