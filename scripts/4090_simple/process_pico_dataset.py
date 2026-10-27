"""After collection exits, admit real PICO captures, pack, extract and transfer.

Inventory is CPU-only and safe during recording. GPU stages fail closed while
any collector, automatic finisher or unrelated compute process is present.
This program never stops another process or starts/changes A100 training.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile

CONTRACT = 'cvpr-pico-admission-processing-plan-v1'
SELECTION = 'cvpr-pico-handover-admissions-selection-v1'
MOBILE_CONTRACT = 'cvpr-pico-mobile-pp-admission-processing-plan-v1'
MOBILE_TASK = 'simple/G1WholebodyLocomotionPickBetweenTablesTeleop-v0'
MOBILE_SELECTION = 'cvpr-pico-mobile-pp-admissions-selection-v1'
MOBILE_INVENTORY = 'cvpr-pico-mobile-pp-collection-admission-inventory-v1'
MOBILE_VERIFICATION = 'cvpr-pico-mobile-pp-production-verification-v1'
BLOCKING_SCRIPTS = frozenset(('pico_simple_teleop.py','finish_pico_session.py',
    'render_deferred_pico.py','extract_unifolm_dino.py'))


class ProcessingDeferred(ValueError):
    """Resource ownership changed; wait rather than reject valid recordings."""


class AdmissionRejected(ValueError):
    """A recorded example failed admission, rather than a processing tool."""


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4 << 20),b''):
            value.update(block)
    return value.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile('w',dir=path.parent,prefix='.'+path.name,delete=False) as stream:
        pending = Path(stream.name)
        stream.write(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def plan_profile(plan):
    if plan.get('contract') == CONTRACT:
        return {'candidate_count':90,'inventory_contract':'cvpr-pico-collection-admission-inventory-v1',
            'selection_contract':SELECTION,'verification_contract':'cvpr-pico-handover-production-verification-v1',
            'mobile':False}
    require(plan.get('contract') == MOBILE_CONTRACT and plan.get('task_id') == MOBILE_TASK,
        'Unknown processing plan or Mobile task identity')
    return {'candidate_count':100,'inventory_contract':MOBILE_INVENTORY,
        'selection_contract':MOBILE_SELECTION,'verification_contract':MOBILE_VERIFICATION,'mobile':True}


def process_activity(records=None):
    """Match actual Python argv tokens, never text inside an agent shell command."""
    if records is None:
        records = []
        for entry in Path('/proc').iterdir():
            if not entry.name.isdigit() or int(entry.name)==os.getpid():
                continue
            try:
                argv = [v.decode(errors='replace') for v in (entry/'cmdline').read_bytes().split(b'\0') if v]
                records.append((int(entry.name),argv))
            except (FileNotFoundError,PermissionError,ProcessLookupError):
                continue
    found = []
    for pid,argv in records:
        if not argv or not Path(argv[0]).name.startswith(('python','pypy')):
            continue
        script = next((Path(a).name for a in argv[1:] if Path(a).name in BLOCKING_SCRIPTS),None)
        if script:
            found.append({'pid':pid,'script':script,'argv':argv})
    return sorted(found,key=lambda r:r['pid'])


def require_idle(*, probe=process_activity, run=subprocess.run):
    active = probe()
    if active:
        raise ProcessingDeferred('GPU processing waits for collectors/finishers: '+json.dumps(active,ensure_ascii=False))
    try:
        result = run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader,nounits'],
            text=True,capture_output=True,check=True)
    except (OSError,subprocess.CalledProcessError) as error:
        raise ProcessingDeferred('GPU availability could not be confirmed: '+str(error)) from error
    if result.stdout.strip():
        raise ProcessingDeferred('Another GPU compute process is active: '+result.stdout.strip())


def owned(root, relative):
    name = Path(relative)
    require(not name.is_absolute() and '..' not in name.parts,'Collection path escapes its root')
    path = (root/name).resolve(strict=True)
    require(path.is_relative_to(root),'Resolved collection path escapes its root')
    return path


def inventory(collection_root):
    root = Path(collection_root).resolve(strict=True)
    history_path = root/'pico_episode_history.json'
    encoded = history_path.read_bytes()
    history = json.loads(encoded)
    require(isinstance(history.get('entries'),dict),'Invalid collection history')
    from pico_episode_review import discard_requested
    candidates, excluded = [], []
    counts = Counter()
    for identity,entry in sorted(history['entries'].items(),key=lambda pair:pair[1]['number']):
        counts[entry.get('disposition','unknown')] += 1
        if entry.get('status')!='committed' or entry.get('disposition')!='kept':
            excluded.append({'human_demo_uuid':identity,'number':entry['number'],
                'reasons':entry.get('reasons',[]) or ['not a committed retained recording']})
            continue
        try:
            directory = owned(root,entry['actual_relative_path'])
            metadata,episode,collection = (read(directory/name) for name in
                ('metadata.json','episode.json','collection_record.json'))
            require(metadata.get('human_demo_uuid')==identity==episode['state']['human_demo_uuid'],
                'Human demonstration UUID differs')
            require(metadata.get('source_kind')==episode.get('source_kind')=='live_pico_human'
                and metadata.get('replay_source_sha256') is None,'Not a new live human recording')
            require(collection.get('status')=='committed' and collection.get('disposition')=='kept'
                and not (directory/'operator_discard.json').exists() and not discard_requested(directory),
                'Discarded, pending-discard or unfinished collection record')
            require(sha(directory/'episode.json')==entry['episode_json_sha256']
                ==collection['episode_json_sha256'],'Sealed recording SHA changed')
            require(episode.get('state',{}).get('status')=='pending_audit','Recording did not finish normally')
            bindings = {name:sha(directory/name) for name in
                ('metadata.json','episode.json','collection_record.json')}
            candidates.append({'human_demo_uuid':identity,'number':entry['number'],
                'episode_dir':str(directory),'source_session':Path(entry['source_relative_path']).parts[0],
                'snapshot_sha256':bindings,'rtx_completion_present':(directory/'rtx_completion.json').is_file(),
                'full_admission_present':(directory/'admission.json').is_file()})
        except (OSError,ValueError,KeyError,TypeError) as error:
            excluded.append({'human_demo_uuid':identity,'number':entry['number'],'reasons':[str(error)]})
    return {'contract':'cvpr-pico-collection-admission-inventory-v1','created_utc':now(),
        'collection_root':str(root),'history_sha256':hashlib.sha256(encoded).hexdigest(),
        'history_entries':len(history['entries']),'dispositions':dict(counts),
        'physical_preaccepted_candidates':len(candidates),'episodes':candidates,'excluded':excluded,
        'scope':'Sealed committed retained human recordings only; no RTX, DINO or full admission claimed'}


def reusable_admission(directory):
    path = directory/'admission.json'
    if not path.is_file():
        return None
    report = read(path)
    require(report.get('episode_directory')==str(directory),'Cached admission belongs to another directory')
    bindings = report.get('bindings',{})
    require(bool(bindings) and all(Path(p).is_file() and sha(p)==digest for p,digest in bindings.items()),
        'Prior admission inputs or checker changed; retain its receipt for investigation')
    if report.get('eligible_for_packing') is not True:
        return None
    authorities = report.get('auditor_source_sha256',{})
    require(bool(authorities),'Prior admission has no auditor authority')
    for original,digest in authorities.items():
        current = Path(__file__).with_name(Path(original).name)
        if not current.is_file() or sha(current)!=digest:
            return None  # A changed checker needs fresh admission, with old receipt retained.
    return report


def frozen_inventory(plan):
    profile = plan_profile(plan)
    count = profile['candidate_count']
    source = Path(plan['frozen_inventory_path']).resolve(strict=True)
    require(sha(source)==plan['frozen_inventory_sha256'],'Frozen inventory bytes changed')
    value = read(source)
    require(value.get('contract')==profile['inventory_contract']
        and value.get('collection_root')==plan['collection_root']
        and value.get('physical_preaccepted_candidates')==plan['candidate_count']==count
        and len(value.get('episodes',[]))==count,'Use the explicitly frozen task cohort only')
    if profile['mobile']:
        require(value.get('task_id')==MOBILE_TASK and value.get('selection_policy')=='native_only',
            'Frozen Mobile task or official selection policy differs')
    identities = [e['human_demo_uuid'] for e in value['episodes']]
    require(len(set(identities))==count,'Frozen cohort repeats a human demonstration')
    return source,value


def validate_frozen_selection(selection, snapshot, plan, progress):
    """A cached admission list cannot replace or expand the frozen cohort."""
    profile = plan_profile(plan)
    require(selection.get('contract')==profile['selection_contract'], 'Cached selection contract differs')
    cohort=selection.get('candidate_cohort',{})
    require(cohort.get('candidate_count')==plan['candidate_count']
        and cohort.get('frozen_inventory_sha256')==plan['frozen_inventory_sha256'],
        'Cached selection belongs to another frozen cohort')
    if profile['mobile']:
        require(selection.get('task_id')==MOBILE_TASK and selection.get('selection_policy')=='native_only',
            'Cached Mobile task or native-only policy differs')
    frozen={item['human_demo_uuid']:item for item in snapshot['episodes']}
    entries=selection.get('episodes',[])
    identities=[item['human_demo_uuid'] for item in entries]
    require(len(set(identities))==len(identities) and set(identities)<=set(frozen),
        'Cached selection repeats, adds or replaces a frozen UUID')
    passed={key for key,value in progress['episodes'].items() if value.get('eligible_for_packing') is True}
    require(set(identities)==passed,'Cached selection differs from completed admission decisions')
    if profile['mobile']:
        require(set(identities)==set(frozen),'Frozen100 Mobile cohort is not fully admitted; no replacement or partial upload')
    from pico_episode_review import discard_requested
    for item in entries:
        original=frozen[item['human_demo_uuid']]
        require(all(item.get(key)==original.get(key) for key in
            ('episode_dir','number','source_session')),'Cached selection changed a frozen source identity')
        directory=Path(original['episode_dir'])
        require(directory.is_dir() and not directory.is_symlink(),'Frozen source directory is missing or aliased')
        require(all(sha(directory/name)==digest for name,digest in original['snapshot_sha256'].items()),
            'Frozen source bytes changed after admission')
        require(not discard_requested(directory)
            and read(directory/'collection_record.json')['disposition']=='kept',
            'A later operator discard cannot be revived by cached admission')
        admission=directory/'admission.json'
        require(item.get('admission_path')==str(admission) and sha(admission)==item.get('admission_sha256'),
            'Cached selection admission receipt binding differs')
        require(reusable_admission(directory) is not None,'Cached selection checker or admitted inputs changed')


def run_child(argv, log, *, gpu=False, allow_rejection=False):
    if gpu:
        require_idle()
    environment = dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        PYTHONNOUSERSITE='1',OMNI_KIT_ACCEPT_EULA='YES')
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS',
                'BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
        environment[key]='1'
    if not gpu:
        environment['CUDA_VISIBLE_DEVICES']=''
    with Path(log).open('xb') as stream:
        result = subprocess.run(list(map(str,argv)),env=environment,stdout=stream,stderr=subprocess.STDOUT)
    require(result.returncode==0 or (allow_rejection and result.returncode==1),'Stage failed, see '+str(log))
    return result.returncode


REMOTE_CODE_SETUP = r'''
import hashlib,json,pathlib,tarfile,sys
v=json.load(sys.stdin);root=pathlib.Path(v['remote_root']);tools=root/'tools'
assert root.is_dir() and not root.is_symlink() and not tools.is_symlink()
assert all(not p.is_symlink() for p in (root,*root.parents,tools))
archive=tools/v['archive_name'];assert archive.is_file() and not archive.is_symlink()
assert hashlib.sha256(archive.read_bytes()).hexdigest()==v['archive_sha256']
target=tools/('code_'+v['code_revision']);receipt=target/'SOURCE_RECEIPT.json'
assert not target.is_symlink() and not receipt.is_symlink()
with tarfile.open(archive) as source:
 members=source.getmembers()
 for member in members:
  name=pathlib.PurePosixPath(member.name)
  assert not name.is_absolute() and '..' not in name.parts and (member.isfile() or member.isdir())
if target.exists():
 assert target.is_dir() and receipt.is_file() and json.loads(receipt.read_text())==v
else:
 staging=tools/('.code_'+v['code_revision']+'.extracting');assert not staging.exists() and not staging.is_symlink();staging.mkdir()
 with tarfile.open(archive) as source:
  source.extractall(staging,members=members)
 with (staging/'SOURCE_RECEIPT.json').open('x') as stream:stream.write(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
 staging.rename(target)
with tarfile.open(archive) as source:
 for member in members:
  path=target/member.name;assert not path.is_symlink()
  assert all(not p.is_symlink() for p in path.parents if p.is_relative_to(target))
  if member.isdir():assert path.is_dir()
  else:
   assert path.is_file() and hashlib.sha256(path.read_bytes()).digest()==hashlib.sha256(source.extractfile(member).read()).digest()
print(json.dumps({'code_root':str(target),'training_checkout_modified':False,'GPU_used':False}))
'''


REMOTE_TOOLS_DIRECTORY = r'''
import pathlib,sys
p=pathlib.Path(sys.argv[1]);tools=p/'tools'
assert p.is_dir() and not any(q.is_symlink() for q in (p,*p.parents,tools))
tools.mkdir(exist_ok=True);assert tools.is_dir() and not tools.is_symlink()
'''


REMOTE_CHECK_READY = r'''
import hashlib,json,pathlib,sys
v=json.load(sys.stdin);root=pathlib.Path(v['remote_root']);p=root/'A100_PICO_DATASET_READY.json'
assert p.is_file() and not p.is_symlink();r=json.loads(p.read_text())
assert r['contract']==v.get('verification_contract','cvpr-pico-handover-production-verification-v1') and r['status']=='passed'
if v.get('task_id') is not None:assert r['task_id']==v['task_id']
assert r['release_root']==str(root) and r['pack_root']==str(root/'packed') and r['dino_root']==str(root/'dino')
assert r['remote_production_ready'] is True and r['CUDA_initialized'] is False and r['GPU_used'] is False and r['training_started'] is False
assert r['source_kind']=='live_pico_human' and r['split_restriction']=='train_only' and r['verifier_sha256']==v['verifier_sha256']
assert r['pack_ready_sha256']==v['pack_ready_sha256'] and r['train_episodes']==v['train_episodes'] and r['train_windows']==v['train_windows']
def check(p,wanted):
 assert p.is_file() and not p.is_symlink()
 assert all(not q.is_symlink() for q in p.parents)
 st=p.stat();h=hashlib.sha256()
 with p.open('rb') as stream:
  for b in iter(lambda:stream.read(4<<20),b''):h.update(b)
 assert st.st_size==wanted['size_bytes'] and st.st_mtime_ns==wanted['mtime_ns'] and h.hexdigest()==wanted['sha256']
for name,wanted in r['payload_files'].items():
 rel=pathlib.PurePosixPath(name);assert not rel.is_absolute() and '..' not in rel.parts
 check(root/name,wanted)
for name,wanted in r['external_reference_files'].items():check(pathlib.Path(name),wanted)
print(json.dumps({'status':'cached_remote_reader_bytes_still_verified','GPU_used':False}))
'''


def remote_cpu_environment(argv):
    # Restrict only this new low-priority reader. Decoder/native worker threads
    # inherit the affinity even when a library chooses an automatic pool size.
    argv = list(map(str,argv))
    launcher = ('import os,sys; allowed=os.sched_getaffinity(0); '
        'os.sched_setaffinity(0,{max(allowed)}); '
        'os.execv(sys.executable,[sys.executable,"-B",*sys.argv[1:]])')
    return ['env','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=1','MKL_NUM_THREADS=1',
        'OPENBLAS_NUM_THREADS=1','NUMEXPR_NUM_THREADS=1','HF_HUB_OFFLINE=1',
        'TRANSFORMERS_OFFLINE=1','PYTHONNOUSERSITE=1','nice','-n','19',
        argv[0],'-B','-c',launcher,*argv[1:]]


def remote_reader(plan, release):
    """Deploy only committed CPU tools to the new dataset, then verify its paths."""
    from upload_pico_handover import SSH,REMOTE_PYTHON,remote_python,remote_target
    profile = plan_profile(plan)
    host = plan['remote_host'];remote = remote_target(plan['remote_root'],task_id=(MOBILE_TASK
        if profile['mobile'] else 'simple/G1WholebodyHandoverTeleop-v0'))
    repo = Path(__file__).resolve().parents[2]
    revision = plan['code_revision']
    require(len(revision)==40 and all(c in '0123456789abcdef' for c in revision),'Pinned code revision required')
    tools = release/'tools';tools.mkdir(exist_ok=True)
    archive = tools/('source_'+revision+'.tar')
    if not archive.exists():
        subprocess.run(['git','archive','--format=tar','--output',str(archive),revision,
            'scripts/4090_simple','VLMExpert/src','VLMExpert/scripts/data','MotionExpert','pipeline','ImageProcess'],cwd=repo,check=True)
    payload = {'remote_root':str(remote),'code_revision':revision,'archive_name':archive.name,
        'archive_sha256':sha(archive),'scope':'Independent CPU dataset tools only; no training checkout updates'}
    # Paths are shell-quoted as individual arguments; JSON stays on stdin.
    command = shlex.join(remote_cpu_environment([REMOTE_PYTHON,'-c',
        REMOTE_TOOLS_DIRECTORY,str(remote)]))
    subprocess.run([*SSH,host,command],check=True)
    subprocess.run(['rsync','-a','--checksum','--protect-args','-e',shlex.join(SSH),
        str(archive),host+':'+str(remote/'tools')+'/'],check=True)
    remote_python(host,REMOTE_PYTHON,REMOTE_CODE_SETUP,payload)
    code = remote/'tools'/('code_'+revision)/'scripts/4090_simple/verify_pico_handover.py'
    environment = remote_cpu_environment([REMOTE_PYTHON,code])
    marker = remote/'A100_PICO_DATASET_READY.json'
    test = shlex.join(remote_cpu_environment([REMOTE_PYTHON,'-c','from pathlib import Path;import sys;sys.exit(0 if Path(sys.argv[1]).is_file() else 1)',str(marker)]))
    already = subprocess.run([*SSH,host,test],check=False).returncode==0
    if not already:
        relocate = shlex.join([*environment,'relocate','--pack-root',str(remote/'packed'),
            '--dino-root',str(remote/'dino')])
        # Relocation is idempotently recognized by its immutable receipt.
        relocated_marker = remote/'dino/REMOTE_DINO_RELOCATION.json'
        test_relocation = shlex.join(remote_cpu_environment([REMOTE_PYTHON,'-c',
            'from pathlib import Path;import sys;sys.exit(0 if Path(sys.argv[1]).is_file() else 1)',str(relocated_marker)]))
        if subprocess.run([*SSH,host,test_relocation],check=False).returncode!=0:
            subprocess.run([*SSH,host,relocate],check=True)
        verify = shlex.join([*environment,'--pack-root',str(remote/'packed'),'--dino-root',str(remote/'dino'),
            '--reference-dino-root',plan['remote_reference_dino'],'--run-config',plan['remote_run_config'],
            '--output',str(marker),'--a100',*(['--expected-episodes',
                str(read(release/'PICO_DATASET_READY.json')['train_episodes'])] if profile['mobile'] else [])])
        subprocess.run([*SSH,host,verify],check=True)
    fetch = shlex.join(remote_cpu_environment([REMOTE_PYTHON,'-c',
        'import pathlib,sys;sys.stdout.write(pathlib.Path(sys.argv[1]).read_text())',str(marker)]))
    local = read(release/'PICO_DATASET_READY.json')
    remote_python(host,REMOTE_PYTHON,REMOTE_CHECK_READY,{'remote_root':str(remote),
        'verifier_sha256':sha(Path(__file__).with_name('verify_pico_handover.py')),
        'pack_ready_sha256':local['pack_ready_sha256'],'train_episodes':local['train_episodes'],
        'train_windows':local['train_windows'],'verification_contract':profile['verification_contract'],
        **({'task_id':MOBILE_TASK} if profile['mobile'] else {})})
    result = subprocess.run([*SSH,host,fetch],check=True,text=True,capture_output=True)
    report = json.loads(result.stdout)
    require(report.get('status')=='passed' and report.get('remote_production_ready') is True
        and report.get('train_episodes')==local['train_episodes']
        and report.get('pack_ready_sha256')==local['pack_ready_sha256']
        and report.get('GPU_used') is False and report.get('training_started') is False,
        'Actual A100 CPU reader did not confirm the same admitted package')
    write(release/'A100_PICO_DATASET_READY.json',report)
    return report


def process(plan_path):
    plan_path = Path(plan_path).resolve(strict=True)
    plan = read(plan_path)
    profile = plan_profile(plan)
    frozen_path,frozen = frozen_inventory(plan)
    require_idle()
    release = Path(plan['release']).absolute()
    require(not release.is_symlink(),'Release must not alias another directory')
    release.mkdir(parents=True,exist_ok=True)
    with (release/'.processing.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        binding = release/'PROCESSING_PLAN.json'
        if binding.exists():
            require(read(binding)==plan,'Existing release belongs to another plan')
        else:
            write(binding,plan)
        script = Path(__file__).resolve().parent
        logs = release/'logs';logs.mkdir(exist_ok=True)
        snapshot_path = release/'INVENTORY.json'
        if not snapshot_path.exists():
            shutil.copyfile(frozen_path,snapshot_path)
        require(sha(snapshot_path)==plan['frozen_inventory_sha256'],'Release inventory differs from frozen cohort')
        snapshot = read(snapshot_path)
        progress_path = release/'PROCESSING_STATUS.json'
        progress = read(progress_path) if progress_path.exists() else {'created_utc':now(),'episodes':{}}
        progress.update(status='admitting',updated_utc=now(),training_started=False)
        write(progress_path,progress)
        selection_path = release/'ADMISSIONS_SELECTION.json'
        frozen_selection = selection_path.is_file()
        selection = read(selection_path) if frozen_selection else {'contract':profile['selection_contract'],'created_utc':now(),
            'candidate_cohort':{'candidate_count':plan['candidate_count'],'frozen_inventory_sha256':sha(snapshot_path)},'episodes':[]}
        if profile['mobile'] and not frozen_selection:
            selection.update(task_id=MOBILE_TASK,selection_policy='native_only')
        for item in ([] if frozen_selection else snapshot['episodes']):
            identity = item['human_demo_uuid'];directory = Path(item['episode_dir'])
            progress.update(active_uuid=identity,active_number=item['number'],updated_utc=now())
            write(progress_path,progress)
            if identity in progress['episodes']:
                previous = progress['episodes'][identity]
                if previous.get('eligible_for_packing') is not True and not previous.get('technical_failure'):
                    continue
            try:
                require_idle()
                if not directory.is_dir():
                    raise ValueError('Frozen episode is missing; retain evidence and diagnose before admission')
                if not all(sha(directory/name)==digest for name,digest in item['snapshot_sha256'].items()):
                    raise ValueError('Frozen collection bytes changed; preserve receipts and diagnose identity')
                from pico_episode_review import discard_requested
                if discard_requested(directory) or read(directory/'collection_record.json')['disposition']!='kept':
                    raise AdmissionRejected('Operator discarded the recording after selection')
                completion = directory/'rtx_completion.json'
                if not completion.exists():
                    require(not (directory/'capture').exists(),'Partial RTX exists; preserve it for diagnosis')
                    render_argv=[plan['simulation_python'],script/'render_deferred_pico.py',directory]
                    if profile['mobile']:
                        render_argv.extend(['--data-root',plan['render_data_root']])
                    run_child(render_argv,
                        logs/(identity+'.rtx.'+str(os.getpid())+'.log'),gpu=True)
                require(read(completion).get('status')=='deferred_rtx_completed_pending_independent_audit',
                    'RTX did not finish without changing archived state')
                admission = reusable_admission(directory)
                if admission is None:
                    old_admission = directory/'admission.json'
                    if old_admission.exists():
                        backup = logs/(identity+'.prior-admission.'+sha(old_admission)+'.json')
                        if not backup.exists():
                            shutil.copyfile(old_admission,backup)
                    # The checker performs its first complete independent
                    # admission; no policy/SONIC/dynamics are replayed.
                    audit_argv=[plan['simulation_python'],script/'audit_pico_teleop_episode.py',directory]
                    if profile['mobile']:
                        audit_argv.extend(['--historical-sources',plan['historical_sources']])
                    run_child(audit_argv,
                        logs/(identity+'.audit.'+str(os.getpid())+'.log'),allow_rejection=True)
                    fresh = read(directory/'admission.json')
                    if fresh.get('eligible_for_packing') is not True:
                        explicit_native_failure=(fresh.get('failure_kind')=='official_task_not_successful'
                            and fresh.get('technical_blocked') is False)
                        error_class=(ValueError if profile['mobile'] and not explicit_native_failure else AdmissionRejected)
                        raise error_class('; '.join(fresh.get('errors',[])) or 'Independent admission rejected the recording')
                    admission = reusable_admission(directory)
                require(admission.get('eligible_for_packing') is True,'; '.join(admission.get('errors',[])))
                selection['episodes'].append({key:item[key] for key in
                    ('human_demo_uuid','episode_dir','source_session','number')})
                selection['episodes'][-1].update(admission_path=str(directory/'admission.json'),
                    admission_sha256=sha(directory/'admission.json'))
                progress['episodes'][identity]={'eligible_for_packing':True,'number':item['number']}
            except ProcessingDeferred:
                raise
            except Exception as error:
                progress['episodes'][identity]={'eligible_for_packing':False,'number':item['number'],
                    'reasons':[str(error)],'technical_failure':not isinstance(error,AdmissionRejected),
                    'raw_evidence_deleted':False}
                if profile['mobile'] and not isinstance(error,AdmissionRejected):
                    progress.update(status='technical_failure_needs_repair',updated_utc=now())
                    write(progress_path,progress)
                    raise ValueError('Mobile processing stopped for technical diagnosis at '+identity+': '+str(error)) from error
            progress['updated_utc']=now();write(progress_path,progress)
        technical = [identity for identity,item in progress['episodes'].items() if item.get('technical_failure')]
        require(not technical,'Processing failures need diagnosis, not data rejection: '+','.join(technical))
        validate_frozen_selection(selection,snapshot,plan,progress)
        if not frozen_selection:
            write(selection_path,selection)
        require(selection['episodes'],'No independently admitted recordings; no package or upload created')
        require_idle()
        if not (release/'packed/PACK_READY.json').exists():
            run_child([plan['model_python'],script/'pack_pico_handover.py','--selection',
                release/'ADMISSIONS_SELECTION.json','--template',plan['template'],'--output',release/'packed'],
                logs/('pack.'+str(os.getpid())+'.log'))
        require(read(release/'packed/PACK_READY.json').get('selection',{}).get('sha256')==sha(selection_path),
            'Existing pack belongs to a different admitted selection')
        if not (release/'dino/manifest.json').is_file() or read(release/'dino/manifest.json').get('state')!='complete':
            run_child([plan['model_python'],script.parents[1]/'VLMExpert/scripts/data/extract_unifolm_dino.py',
                '--dataset-root',release/'packed/train','--output',release/'dino',
                '--model-path',plan['dino_model'],'--shared-image-profile','256x320','--device','cuda:0',
                '--batch-size','32','--workers','1','--decoder-threads','1','--cpu-threads','2',
                '--storage-dtype','float32','--inference-precision','bf16'],logs/('dino.'+str(os.getpid())+'.log'),gpu=True)
        if not (release/'PICO_DATASET_READY.json').exists():
            verify_argv=[plan['model_python'],script/'verify_pico_handover.py','--pack-root',release/'packed',
                '--dino-root',release/'dino','--reference-dino-root',plan['reference_dino'],
                '--run-config',plan['run_config'],'--output',release/'PICO_DATASET_READY.json']
            if profile['mobile']:
                verify_argv.extend(['--expected-episodes',str(len(selection['episodes']))])
            run_child(verify_argv,
                logs/('verify.'+str(os.getpid())+'.log'))
        progress.update(status='local_verified_uploading',updated_utc=now());write(progress_path,progress)
        require_idle()
        from upload_pico_handover import inventory as transfer_inventory
        transfer_inventory(release,plan['remote_root'])
        if not (release/'A100_BYTE_VERIFICATION.json').exists():
            run_child([plan['model_python'],script/'upload_pico_handover.py','--local-root',release,
                '--remote-root',plan['remote_root'],'--host',plan['remote_host']],logs/('upload.'+str(os.getpid())+'.log'))
        progress.update(status='uploaded_bytes_remote_reader_pending',updated_utc=now(),
            accepted_episodes=len(selection['episodes']),remote_root=plan['remote_root'])
        write(progress_path,progress)
        report = remote_reader(plan,release)
        progress.update(status='A100_ready_for_training_not_started',updated_utc=now(),
            remote_ready_sha256=sha(release/'A100_PICO_DATASET_READY.json'),train_windows=report['train_windows'])
        write(progress_path,progress)
        return progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command',required=True)
    p = commands.add_parser('inventory');p.add_argument('--collection-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p = commands.add_parser('run');p.add_argument('--plan',type=Path,required=True)
    p = commands.add_parser('activity')
    args = parser.parse_args()
    if args.command=='inventory':
        result = inventory(args.collection_root);write(args.output,result)
        print(json.dumps({key:result[key] for key in ('created_utc','history_entries','dispositions','physical_preaccepted_candidates')},ensure_ascii=False))
    elif args.command=='activity':
        print(json.dumps({'blocking':process_activity()},ensure_ascii=False))
    else:
        try:
            result = process(args.plan)
        except ProcessingDeferred as error:
            print(json.dumps({'status':'waiting_for_collection_or_gpu','reason':str(error),
                'data_rejected':False,'GPU_started':False},ensure_ascii=False))
            raise SystemExit(75)
        print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    main()
