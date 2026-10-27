"""Processing safety/regression checks without a GPU, SDK or remote host."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import subprocess
import tarfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import process_pico_dataset as workflow


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_only_actual_python_script_argv_blocks_processing(self):
        records=[(1,['bash','-lc','python pico_simple_teleop.py --output anything']),
            (2,['python3','/repo/pico_simple_teleop.py','--output','/data/live']),
            (3,['python','/repo/finish_pico_session.py','/data/live']),
            (4,['python','/repo/unrelated_cpu.py']),
            (5,['python','-c','print("render_deferred_pico.py")'])]
        self.assertEqual([r['pid'] for r in workflow.process_activity(records)],[2,3])

    def test_live_recording_blocks_before_gpu_query_or_any_stage_launch(self):
        with self.assertRaisesRegex(ValueError,'waits for collectors'):
            workflow.require_idle(probe=lambda:[{'pid':2,'script':'pico_simple_teleop.py'}],
                run=lambda *a,**kw:self.fail('No GPU probe while collecting'))

    def test_unrelated_gpu_jobs_block_without_changing_them(self):
        calls=[]
        def run(argv,**kwargs):
            calls.append(argv)
            return SimpleNamespace(stdout='1002, unrelated-training\n')
        with self.assertRaisesRegex(ValueError,'Another GPU compute process'):
            workflow.require_idle(probe=lambda:[],run=run)
        self.assertEqual(len(calls),1)
        self.assertEqual(calls[0][0],'nvidia-smi')

    def fixture(self):
        identity='5ea9892a-84dc-4c46-99e0-777482a2bf81'
        directory=self.root/'kept/001';directory.mkdir(parents=True)
        metadata={'human_demo_uuid':identity,'source_kind':'live_pico_human','replay_source_sha256':None}
        episode={'source_kind':'live_pico_human','state':{'human_demo_uuid':identity,'status':'pending_audit'}}
        workflow.write(directory/'metadata.json',metadata);workflow.write(directory/'episode.json',episode)
        digest=workflow.sha(directory/'episode.json')
        workflow.write(directory/'collection_record.json',{'status':'committed','disposition':'kept','episode_json_sha256':digest})
        entry={'number':1,'status':'committed','disposition':'kept','actual_relative_path':'kept/001',
            'source_relative_path':'live_human_10/episodes/'+identity,'episode_json_sha256':digest}
        history={'entries':{identity:entry}}
        workflow.write(self.root/'pico_episode_history.json',history)
        return identity,directory,history

    def inspect(self):
        with patch('pico_episode_review.discard_requested',return_value=False):
            return workflow.inventory(self.root)

    def test_inventory_reads_sealed_moved_directory_without_rewriting_it(self):
        identity,directory,_=self.fixture()
        before={str(p):p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        result=self.inspect()
        self.assertEqual(result['physical_preaccepted_candidates'],1)
        self.assertEqual(result['episodes'][0]['episode_dir'],str(directory))
        self.assertEqual(result['episodes'][0]['human_demo_uuid'],identity)
        self.assertFalse(result['episodes'][0]['rtx_completion_present'])
        self.assertEqual(before,{str(p):p.read_bytes() for p in self.root.rglob('*') if p.is_file()})

    def test_inventory_never_revives_discarded_replay_pending_or_changed_raw_records(self):
        changes=('discarded','replay','pending','sha','escape','intent')
        for change in changes:
            with self.subTest(change=change):
                with tempfile.TemporaryDirectory() as temporary:
                    old_root=self.root;self.root=Path(temporary)
                    identity,directory,history=self.fixture()
                    if change=='discarded':history['entries'][identity]['disposition']='discarded'
                    if change=='escape':history['entries'][identity]['actual_relative_path']='../escape'
                    if change=='replay':
                        value=workflow.read(directory/'metadata.json');value['source_kind']='recorded_pico_diagnostic';workflow.write(directory/'metadata.json',value)
                    if change=='pending':
                        value=workflow.read(directory/'episode.json');value['state']['status']='warmup';workflow.write(directory/'episode.json',value)
                    if change=='sha':(directory/'episode.json').write_text('{}')
                    workflow.write(self.root/'pico_episode_history.json',history)
                    with patch('pico_episode_review.discard_requested',return_value=change=='intent'):
                        result=workflow.inventory(self.root)
                    self.assertEqual(result['episodes'],[])
                    self.assertTrue(result['excluded'][0]['reasons'])
                    self.root=old_root

    def test_existing_admission_is_reused_only_when_its_bound_inputs_and_checker_match(self):
        directory=self.root/'episode';directory.mkdir()
        source=directory/'raw';source.write_bytes(b'original')
        checker=self.root/'checker.py';checker.write_bytes(b'accepted checker')
        report={'episode_directory':str(directory),'eligible_for_packing':True,
            'bindings':{str(source):workflow.sha(source)},'auditor_source_sha256':{str(checker):workflow.sha(checker)}}
        workflow.write(directory/'admission.json',report)
        with patch.object(workflow,'__file__',str(self.root/'process.py')):
            self.assertEqual(workflow.reusable_admission(directory),report)
        source.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'Prior admission inputs'):
            workflow.reusable_admission(directory)

    def test_cpu_child_receives_no_cuda_and_reject_exit_is_preserved_for_audit(self):
        log=self.root/'audit.log'
        with patch.object(workflow.subprocess,'run',return_value=SimpleNamespace(returncode=1)) as runner:
            result=workflow.run_child(['python','audit.py'],log,allow_rejection=True)
        self.assertEqual(result,1)
        environment=runner.call_args.kwargs['env']
        self.assertEqual(environment['CUDA_VISIBLE_DEVICES'],'')
        self.assertEqual(environment['OPENBLAS_NUM_THREADS'],'1')

    def test_gpu_child_cannot_start_when_live_guard_rejects(self):
        with patch.object(workflow,'require_idle',side_effect=ValueError('recording')):
            with patch.object(workflow.subprocess,'run',side_effect=AssertionError('must never run')):
                with self.assertRaisesRegex(ValueError,'recording'):
                    workflow.run_child(['python','render.py'],self.root/'gpu.log',gpu=True)
        self.assertFalse((self.root/'gpu.log').exists())

    def test_unknown_gpu_availability_is_a_deferral_not_a_data_rejection(self):
        def unavailable(*a,**kw):
            raise subprocess.CalledProcessError(1,a[0])
        with self.assertRaises(workflow.ProcessingDeferred):
            workflow.require_idle(probe=lambda:[],run=unavailable)

    def frozen_plan(self):
        source=self.root/'frozen.json'
        value={'contract':'cvpr-pico-collection-admission-inventory-v1','collection_root':'/original/collection',
            'physical_preaccepted_candidates':90,'episodes':[{'human_demo_uuid':str(i)} for i in range(90)]}
        workflow.write(source,value)
        plan={'contract':workflow.CONTRACT,'collection_root':'/original/collection',
            'frozen_inventory_path':str(source),'frozen_inventory_sha256':workflow.sha(source),
            'candidate_count':90,'release':str(self.root/'release')}
        return source,value,plan

    def test_processing_uses_frozen90_and_defers_without_creating_any_release(self):
        source,value,plan=self.frozen_plan()
        p=self.root/'plan.json';workflow.write(p,plan)
        with patch.object(workflow,'require_idle',side_effect=workflow.ProcessingDeferred('live')):
            with patch.object(workflow,'inventory',side_effect=AssertionError('never rescan current cohort')):
                with self.assertRaises(workflow.ProcessingDeferred):workflow.process(p)
        self.assertFalse((self.root/'release').exists())
        self.assertEqual(workflow.frozen_inventory(plan),(source,value))

    def test_changed_or_duplicate_or_expanded_cohort_is_rejected(self):
        for change in ('bytes','duplicate','expand'):
            source,value,plan=self.frozen_plan()
            if change=='bytes':source.write_text('{}')
            else:
                if change=='duplicate':value['episodes'][-1]=value['episodes'][0]
                else:value['episodes'].append({'human_demo_uuid':'extra'})
                workflow.write(source,value);plan['frozen_inventory_sha256']=workflow.sha(source)
            with self.assertRaises(ValueError):workflow.frozen_inventory(plan)

    def test_mobile_frozen100_has_its_own_task_and_selection_contract(self):
        source, value, plan = self.frozen_plan()
        value.update(contract=workflow.MOBILE_INVENTORY,task_id=workflow.MOBILE_TASK,
            selection_policy='native_only',physical_preaccepted_candidates=100,
            episodes=[{'human_demo_uuid':str(i)} for i in range(100)])
        workflow.write(source,value)
        plan.update(contract=workflow.MOBILE_CONTRACT,task_id=workflow.MOBILE_TASK,
            candidate_count=100,frozen_inventory_sha256=workflow.sha(source))
        self.assertEqual(workflow.frozen_inventory(plan),(source,value))
        self.assertEqual(workflow.plan_profile(plan)['selection_contract'],workflow.MOBILE_SELECTION)
        for field, changed in [('task_id','handover'),('selection_policy','extra_geometry'),
                               ('physical_preaccepted_candidates',99)]:
            with self.subTest(field=field):
                workflow.write(source,{**value,field:changed})
                altered={**plan,'frozen_inventory_sha256':workflow.sha(source)}
                with self.assertRaises(ValueError):workflow.frozen_inventory(altered)
        with self.assertRaises(ValueError):
            workflow.plan_profile({**plan,'task_id':'handover'})

    def test_remote_cpu_wrapper_never_enables_cuda_or_changes_training_arguments(self):
        argv=workflow.remote_cpu_environment(['/env/python','verify.py','--a100'])
        self.assertIn('CUDA_VISIBLE_DEVICES=',argv)
        self.assertIn('OPENBLAS_NUM_THREADS=1',argv)
        self.assertEqual(argv[-2:],['verify.py','--a100'])
        self.assertIn('/env/python',argv)
        self.assertEqual(argv[argv.index('nice'):argv.index('nice')+3],['nice','-n','19'])

    def test_cached_mobile_selection_cannot_add_or_replace_a_frozen_uuid(self):
        source,value,plan=self.frozen_plan()
        value.update(contract=workflow.MOBILE_INVENTORY,task_id=workflow.MOBILE_TASK,
            selection_policy='native_only',physical_preaccepted_candidates=100,
            episodes=[{'human_demo_uuid':str(i)} for i in range(100)])
        workflow.write(source,value)
        plan.update(contract=workflow.MOBILE_CONTRACT,task_id=workflow.MOBILE_TASK,
            candidate_count=100,frozen_inventory_sha256=workflow.sha(source))
        release=Path(plan['release']);release.mkdir()
        selection={'contract':workflow.MOBILE_SELECTION,'task_id':workflow.MOBILE_TASK,
            'selection_policy':'native_only','candidate_cohort':{'candidate_count':100,
                'frozen_inventory_sha256':plan['frozen_inventory_sha256']},
            'episodes':[{'human_demo_uuid':'later-recording'}]}
        workflow.write(release/'ADMISSIONS_SELECTION.json',selection)
        workflow.write(release/'PROCESSING_STATUS.json',{'episodes':{
            'later-recording':{'eligible_for_packing':True}}})
        plan_path=self.root/'plan.json';workflow.write(plan_path,plan)
        with patch.object(workflow,'require_idle'),patch.object(workflow,'run_child') as child:
            with self.assertRaisesRegex(ValueError,'repeats, adds or replaces'):
                workflow.process(plan_path)
            child.assert_not_called()

    def test_cached_selection_checks_contract_cohort_and_does_not_rewrite_policy(self):
        plan={'contract':workflow.MOBILE_CONTRACT,'task_id':workflow.MOBILE_TASK,
            'candidate_count':100,'frozen_inventory_sha256':'a'*64}
        selection={'contract':workflow.MOBILE_SELECTION,'task_id':workflow.MOBILE_TASK,
            'selection_policy':'native_only','candidate_cohort':{'candidate_count':100,
                'frozen_inventory_sha256':'a'*64},'episodes':[]}
        for change in ({'contract':workflow.SELECTION},{'task_id':'handover'},
                {'selection_policy':'extra-geometry'},
                {'candidate_cohort':{'candidate_count':100,'frozen_inventory_sha256':'b'*64}}):
            altered={**selection,**change};before=json.dumps(altered,sort_keys=True)
            with self.assertRaises(ValueError):
                workflow.validate_frozen_selection(altered,{'episodes':[]},plan,{'episodes':{}})
            self.assertEqual(json.dumps(altered,sort_keys=True),before)

    def test_remote_mobile_ready_requires_the_exact_mobile_contract_and_task(self):
        root=self.root/'remote';root.mkdir()
        expected={'remote_root':str(root),'verifier_sha256':'c'*64,'pack_ready_sha256':'d'*64,
            'train_episodes':100,'train_windows':55906,
            'verification_contract':workflow.MOBILE_VERIFICATION,'task_id':workflow.MOBILE_TASK}
        report={**expected,'contract':workflow.MOBILE_VERIFICATION,'status':'passed',
            'release_root':str(root),'pack_root':str(root/'packed'),'dino_root':str(root/'dino'),
            'remote_production_ready':True,'CUDA_initialized':False,'GPU_used':False,'training_started':False,
            'source_kind':'live_pico_human','split_restriction':'train_only',
            'payload_files':{},'external_reference_files':{}}
        def check():return subprocess.run([sys.executable,'-c',workflow.REMOTE_CHECK_READY],
            input=json.dumps(expected),text=True,capture_output=True)
        workflow.write(root/'A100_PICO_DATASET_READY.json',report)
        self.assertEqual(check().returncode,0)
        for change in ({'task_id':'handover'},{'contract':'cvpr-pico-handover-production-verification-v1'}):
            workflow.write(root/'A100_PICO_DATASET_READY.json',{**report,**change})
            self.assertNotEqual(check().returncode,0)

    def test_remote_code_extracts_only_owned_regular_committed_source(self):
        root=self.root/'remote';tools=root/'tools';tools.mkdir(parents=True)
        archive=tools/'source.tar'
        with tarfile.open(archive,'w') as tar:
            payload=b'CPU tools only'
            member=tarfile.TarInfo('scripts/verify.py');member.size=len(payload)
            tar.addfile(member,io.BytesIO(payload))
        value={'remote_root':str(root),'archive_name':archive.name,
            'archive_sha256':workflow.sha(archive),'code_revision':'a'*40}
        result=subprocess.run([sys.executable,'-c',workflow.REMOTE_CODE_SETUP],input=json.dumps(value),text=True,capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)
        extracted=tools/('code_'+'a'*40)
        self.assertEqual((extracted/'scripts/verify.py').read_bytes(),payload)
        # A retry recognizes the immutable receipt, rather than replacing code.
        retry=subprocess.run([sys.executable,'-c',workflow.REMOTE_CODE_SETUP],input=json.dumps(value),text=True,capture_output=True)
        self.assertEqual(retry.returncode,0,retry.stderr)
        self.assertEqual(workflow.read(extracted/'SOURCE_RECEIPT.json'),value)
        (extracted/'scripts/verify.py').write_bytes(b'modified')
        changed=subprocess.run([sys.executable,'-c',workflow.REMOTE_CODE_SETUP],input=json.dumps(value),text=True,capture_output=True)
        self.assertNotEqual(changed.returncode,0)

    def test_remote_code_rejects_archive_escape_before_extraction(self):
        root=self.root/'remote';tools=root/'tools';tools.mkdir(parents=True)
        archive=tools/'source.tar'
        with tarfile.open(archive,'w') as tar:
            member=tarfile.TarInfo('../escaped');member.size=1
            tar.addfile(member,io.BytesIO(b'x'))
        value={'remote_root':str(root),'archive_name':archive.name,
            'archive_sha256':workflow.sha(archive),'code_revision':'b'*40}
        result=subprocess.run([sys.executable,'-c',workflow.REMOTE_CODE_SETUP],input=json.dumps(value),text=True,capture_output=True)
        self.assertNotEqual(result.returncode,0)
        self.assertFalse((root/'escaped').exists())

    def test_tools_alias_is_rejected_before_any_remote_copy(self):
        root=self.root/'remote';root.mkdir()
        outside=self.root/'outside';outside.mkdir()
        (root/'tools').symlink_to(outside,target_is_directory=True)
        result=subprocess.run([sys.executable,'-c',workflow.REMOTE_TOOLS_DIRECTORY,str(root)],capture_output=True)
        self.assertNotEqual(result.returncode,0)
        self.assertEqual(list(outside.iterdir()),[])

    def test_cached_a100_ready_requires_current_bytes_mtimes_and_checker(self):
        root=self.root/'remote';(root/'packed').mkdir(parents=True);(root/'dino').mkdir()
        file=root/'dino/features.npy';file.write_bytes(b'verified features')
        st=file.stat();file_identity={'size_bytes':st.st_size,'mtime_ns':st.st_mtime_ns,'sha256':workflow.sha(file)}
        expected={'remote_root':str(root),'verifier_sha256':'c'*64,'pack_ready_sha256':'d'*64,'train_episodes':90,'train_windows':36091}
        report={**{k:v for k,v in expected.items() if k!='remote_root'},
            'contract':'cvpr-pico-handover-production-verification-v1','status':'passed',
            'release_root':str(root),'pack_root':str(root/'packed'),'dino_root':str(root/'dino'),
            'remote_production_ready':True,'CUDA_initialized':False,'GPU_used':False,'training_started':False,
            'source_kind':'live_pico_human','split_restriction':'train_only',
            'payload_files':{'dino/features.npy':file_identity},'external_reference_files':{}}
        workflow.write(root/'A100_PICO_DATASET_READY.json',report)
        def check():return subprocess.run([sys.executable,'-c',workflow.REMOTE_CHECK_READY],input=json.dumps(expected),text=True,capture_output=True)
        self.assertEqual(check().returncode,0)
        file.write_bytes(b'changed features!')
        self.assertNotEqual(check().returncode,0)

    def test_changed_checker_requires_new_admission_without_altering_old_receipt(self):
        directory=self.root/'episode';directory.mkdir()
        raw=directory/'raw';raw.write_bytes(b'unchanged')
        checker=self.root/'checker.py';checker.write_bytes(b'new implementation')
        report={'episode_directory':str(directory),'eligible_for_packing':True,
            'bindings':{str(raw):workflow.sha(raw)},'auditor_source_sha256':{str(checker):'0'*64}}
        workflow.write(directory/'admission.json',report)
        before=(directory/'admission.json').read_bytes()
        with patch.object(workflow,'__file__',str(self.root/'process.py')):
            self.assertIsNone(workflow.reusable_admission(directory))
        self.assertEqual(before,(directory/'admission.json').read_bytes())

    def test_remote_reader_affinity_is_process_local_and_inherited(self):
        import os
        before=set(os.sched_getaffinity(0))
        script=self.root/'affinity.py'
        script.write_text('import os,json;print(json.dumps({"cpus":list(os.sched_getaffinity(0)),"cuda":os.environ["CUDA_VISIBLE_DEVICES"],"nice":os.getpriority(os.PRIO_PROCESS,0)}))')
        result=subprocess.run(workflow.remote_cpu_environment([sys.executable,script]),capture_output=True,text=True,check=True)
        received=json.loads(result.stdout)
        self.assertEqual(received['cpus'],[max(before)])
        self.assertEqual(received['cuda'],'')
        self.assertEqual(received['nice'],19)
        self.assertEqual(set(os.sched_getaffinity(0)),before)


if __name__=='__main__':
    unittest.main()
