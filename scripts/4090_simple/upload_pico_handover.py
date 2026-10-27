"""Transfer a CPU-verified PICO release to a fresh A100 directory.

Only copied bytes and nanosecond identities are accepted here. DINO host-path
relocation and a second real CPU reader verification remain explicit stages;
this uploader never writes A100_PICO_DATASET_READY or starts training.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess

import verify_pico_handover as verifier

CONTRACT = 'cvpr-pico-handover-a100-byte-transfer-v1'
BYTE_CONTRACT = 'cvpr-pico-handover-a100-byte-verification-v1'
MOBILE_CONTRACT = 'cvpr-pico-mobile-pp-a100-byte-transfer-v1'
MOBILE_BYTE_CONTRACT = 'cvpr-pico-mobile-pp-a100-byte-verification-v1'
HANDOVER_TASK_ID = 'simple/G1WholebodyHandoverTeleop-v0'
MOBILE_TASK_ID = 'simple/G1WholebodyLocomotionPickBetweenTablesTeleop-v0'
TASK_PROFILES = {
    HANDOVER_TASK_ID: {
        'ready_contract': 'cvpr-pico-handover-production-verification-v1',
        'pack_contract': 'cvpr-pico-handover-lerobot-pack-v1',
        'transfer_contract': CONTRACT, 'byte_contract': BYTE_CONTRACT,
        'remote_prefix': 'pico_handover_',
    },
    MOBILE_TASK_ID: {
        'ready_contract': 'cvpr-pico-mobile-pp-production-verification-v1',
        'pack_contract': 'cvpr-pico-mobile-pp-lerobot-pack-v1',
        'transfer_contract': MOBILE_CONTRACT, 'byte_contract': MOBILE_BYTE_CONTRACT,
        'remote_prefix': 'pico_mobile_pp_',
    },
}
MANIFEST_NAME = 'A100_TRANSFER_MANIFEST.json'
BYTE_NAME = 'A100_BYTE_VERIFICATION.json'
REMOTE_BASE = PurePosixPath('/gaozt-test1/datasets/SIMPLE_sonic_v1_1')
REMOTE_PYTHON = '/gaozt-test1/datasets/SIMPLE_sonic_v1_1/.pipeline/envs/task-finetune-py311-cu126/bin/python'
SSH = ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12']
require,read,identity = verifier.require,verifier.read,verifier.identity


def remote_target(value, *, task_id=None):
    path = PurePosixPath(str(value))
    require(task_id is None or (isinstance(task_id,str) and task_id in TASK_PROFILES),
            'Unsupported task identity for remote target')
    profiles = list(TASK_PROFILES.values()) if task_id is None else [TASK_PROFILES.get(task_id)]
    require(path.is_absolute() and '..' not in path.parts and path.parent==REMOTE_BASE
            and any(profile and re.fullmatch(re.escape(profile['remote_prefix'])+
                r'[A-Za-z0-9][A-Za-z0-9_.-]{0,100}',path.name) for profile in profiles),
            'Use a fresh task-matched pico_handover_<release> or pico_mobile_pp_<release> target under '+str(REMOTE_BASE))
    return path


def task_profile(ready):
    task_id = ready.get('task_id')
    # Historical Handover readiness receipts predate an explicit task_id.
    if task_id is None and ready.get('contract') == TASK_PROFILES[HANDOVER_TASK_ID]['ready_contract']:
        task_id = HANDOVER_TASK_ID
    require(isinstance(task_id,str) and task_id in TASK_PROFILES,
            'Explicit supported task identity is required for this CPU readiness contract')
    profile = TASK_PROFILES[task_id]
    require(ready.get('contract') == profile['ready_contract'], 'Task identity and CPU readiness contract differ')
    return task_id, profile


def inventory(root,remote):
    root = Path(root)
    require(not root.is_symlink(),'Release root cannot be a symlink')
    root = root.resolve(strict=True)
    ready_path = verifier.owned(root,verifier.READY_NAME)
    ready = read(ready_path)
    task_id, profile = task_profile(ready)
    remote = remote_target(remote,task_id=task_id)
    require(ready.get('status')=='passed'
            and ready.get('production_ready_for_bound_pack_and_cache') is True
            and ready.get('source_kind')=='live_pico_human' and ready.get('split_restriction')=='train_only'
            and ready.get('normalization_refitted') is False and ready.get('GPU_used') is False
            and ready.get('CUDA_initialized') is False and ready.get('training_started') is False
            and ready.get('verifier_sha256')==verifier.sha(verifier.__file__)
            and ready.get('release_root')==str(root),'Passed local CPU production readiness is required')
    files = {}
    for directory in ('packed','dino'):
        files.update({directory+'/'+name:value for name,value in verifier.tree_inventory(root/directory).items()})
    require(files==ready.get('payload_files'),'Previously verified pack/DINO bytes or identities changed')
    pack = read(verifier.owned(root,'packed/PACK_READY.json'))
    pack_task = pack.get('task_id',HANDOVER_TASK_ID)
    require(pack.get('contract')==profile['pack_contract'] and pack_task==task_id and pack.get('final_ready') is False
            and ready.get('pack_ready_sha256')==files['packed/PACK_READY.json']['sha256'],
            'Require the unmodified PICO pack contract')
    files[verifier.READY_NAME] = identity(ready_path)
    return {'contract':profile['transfer_contract'],'task_id':task_id,'local_root':str(root),'remote_root':str(remote),
        'created_utc':datetime.now(timezone.utc).isoformat(),'files':files,
        'data_directories':['packed','dino'],'root_files':[verifier.READY_NAME],
        'total_bytes':sum(value['size_bytes'] for value in files.values()),
        'train_episodes':ready['train_episodes'],'train_windows':ready['train_windows'],
        'local_ready_sha256':files[verifier.READY_NAME]['sha256'],
        'remote_production_ready':False,'DINO_relocation_pending':True,
        'GPU_used':False,'training_started':False,'uploader_sha256':verifier.sha(__file__)}


REMOTE_COMMON = r'''
import hashlib,json,os,pathlib,sys
value=json.load(sys.stdin);root=pathlib.Path(value['remote_root'])
def require(v,m):
 if not v:raise ValueError(m)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as s:
  for b in iter(lambda:s.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def regular(p):
 require(p.is_file() and not p.is_symlink(),'Expected regular copied file: '+str(p))
def owned(relative):
 name=pathlib.Path(relative);require(not name.is_absolute() and '..' not in name.parts and name.parts,'Escaping path')
 p=root
 for part in name.parts:
  p=p/part;require(not p.is_symlink(),'No destination symlinks')
 require(p.resolve().is_relative_to(root.resolve()),'Resolved path escapes target')
 return p
require(root.is_absolute() and '..' not in root.parts and not root.is_symlink(),'Independent absolute target required')
for p in root.parents:require(not p.is_symlink(),'Target ancestor is a symlink')
profiles={
 'cvpr-pico-handover-a100-byte-transfer-v1':('simple/G1WholebodyHandoverTeleop-v0','pico_handover_','cvpr-pico-handover-a100-byte-verification-v1'),
 'cvpr-pico-mobile-pp-a100-byte-transfer-v1':('simple/G1WholebodyLocomotionPickBetweenTablesTeleop-v0','pico_mobile_pp_','cvpr-pico-mobile-pp-a100-byte-verification-v1')}
require(value.get('contract') in profiles and value['remote_production_ready'] is False,'Invalid transfer contract')
task_id,prefix,byte_contract=profiles[value['contract']]
require(value.get('task_id',task_id if task_id=='simple/G1WholebodyHandoverTeleop-v0' else None)==task_id,'Task identity differs from transfer contract')
if root.parent==pathlib.Path('/gaozt-test1/datasets/SIMPLE_sonic_v1_1'):
 require(root.name.startswith(prefix),'Remote task namespace differs')
'''

REMOTE_PREPARE = REMOTE_COMMON + r'''
marker=root/'A100_TRANSFER_MANIFEST.json'
if root.exists():
 regular(marker);require(json.loads(marker.read_text())==value,'Existing target is not this exact pending transfer')
 require(not (root/'A100_PICO_DATASET_READY.json').exists() and not (root/'dino.before-remote-relocation').exists(),'Preserve completed/relocated release')
else:
 root.mkdir(parents=True)
 with marker.open('x') as s:json.dump(value,s,ensure_ascii=False,indent=2);s.write('\n')
for name in value['data_directories']:owned(name).mkdir(exist_ok=True)
print(json.dumps({'stage':'fresh_remote_pending_target','remote_root':str(root),'production_ready':False}),flush=True)
'''

REMOTE_VERIFY = REMOTE_COMMON + r'''
marker=root/'A100_TRANSFER_MANIFEST.json';regular(marker)
require(json.loads(marker.read_text())==value,'Transfer manifest changed')
actual=set()
for name in value['data_directories']:
 for p in (root/name).rglob('*'):
  require(not p.is_symlink(),'Copied directory contains aliases')
  if p.is_file():actual.add(str(p.relative_to(root)))
expected={r for r in value['files'] if pathlib.Path(r).parts[0] in value['data_directories']}
require(actual==expected,'Missing/extra package/cache files')
for relative,wanted in value['files'].items():
 p=owned(relative);regular(p);st=p.stat()
 require(st.st_size==wanted['size_bytes'] and sha(p)==wanted['sha256'],'Copied bytes differ: '+relative)
 if st.st_mtime_ns!=wanted['mtime_ns']:os.utime(p,ns=(st.st_atime_ns,wanted['mtime_ns']))
 require(p.stat().st_mtime_ns==wanted['mtime_ns'],'Nanosecond identity restoration failed')
report={'contract':byte_contract,'task_id':task_id,'status':'all_copied_bytes_and_identities_verified',
 'remote_root':str(root),'inventory_sha256':sha(marker),'local_ready_sha256':value['local_ready_sha256'],
 'verified_files':len(value['files']),'verified_bytes':value['total_bytes'],
 'DINO_relocation_pending':True,'remote_production_ready':False,'GPU_used':False,'training_started':False,
 'cpu_runtime':{'cpus':sorted(os.sched_getaffinity(0)),'nice':os.getpriority(os.PRIO_PROCESS,0),
                'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES')}}
target=root/'A100_BYTE_VERIFICATION.json'
if target.exists():regular(target);require(json.loads(target.read_text())==report,'Preserve previous byte-verification report')
else:
 with target.open('x') as s:json.dump(report,s,ensure_ascii=False,indent=2);s.write('\n')
print(json.dumps(report),flush=True)
'''


REMOTE_CPU_LAUNCHER = ('import os,sys; allowed=os.sched_getaffinity(0); '
    'os.sched_setaffinity(0,{max(allowed)}); '
    'exec(compile(sys.argv[1],"<pico-cpu-remote>","exec"))')


def remote_python(host,python,code,value):
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.@-]{0,200}',host),'Invalid SSH destination')
    command = shlex.join(['env','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=1',
        'MKL_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','NUMEXPR_NUM_THREADS=1',
        'PYTHONNOUSERSITE=1','nice','-n','19',python,'-B','-c',REMOTE_CPU_LAUNCHER,code])
    subprocess.run([*SSH,host,command],input=json.dumps(value,allow_nan=False),text=True,check=True)


def transfer(root,remote,*,host='115.191.40.142',remote_python_path=REMOTE_PYTHON):
    root = Path(root); remote = remote_target(remote)
    proposed = inventory(root,remote)
    root = Path(proposed['local_root'])
    manifest_path = root/MANIFEST_NAME
    require(not manifest_path.is_symlink(),'Transfer manifest cannot be an alias')
    if manifest_path.exists():
        value = read(manifest_path)
        comparable = lambda v:{key:item for key,item in v.items() if key!='created_utc'}
        require(comparable(value)==comparable(proposed),'Pending transfer target/source identity changed')
    else:
        value = proposed
        with manifest_path.open('x') as stream:json.dump(value,stream,ensure_ascii=False,indent=2);stream.write('\n')
    remote_python(host,remote_python_path,REMOTE_PREPARE,value)
    transport = shlex.join(SSH)
    common = ['rsync','-a','--checksum','--protect-args','--info=stats2','-e',transport]
    for directory in value['data_directories']:
        subprocess.run([*common,str(root/directory)+'/',f'{host}:{remote/directory}/'],check=True)
    for name in value['root_files']:
        subprocess.run([*common,str(root/name),f'{host}:{remote}/'],check=True)
    remote_python(host,remote_python_path,REMOTE_VERIFY,value)
    target = root/BYTE_NAME
    require(not target.is_symlink(),'Local byte receipt must not be an alias')
    subprocess.run([*common,f'{host}:{remote/BYTE_NAME}',str(target)],check=True)
    report = read(target)
    _, profile = task_profile(read(root/verifier.READY_NAME))
    cpu = report.get('cpu_runtime',{})
    require(report.get('contract')==profile['byte_contract'] and report.get('task_id')==value['task_id']
            and report.get('inventory_sha256')==verifier.sha(manifest_path)
            and report.get('status')=='all_copied_bytes_and_identities_verified'
            and report.get('remote_root')==str(remote) and report.get('remote_production_ready') is False
            and cpu.get('CUDA_VISIBLE_DEVICES')=='' and cpu.get('nice',-1)>=19
            and isinstance(cpu.get('cpus'),list) and len(cpu['cpus'])==1,
            'Returned byte receipt differs; remote production readiness is still pending')
    require(all(identity(verifier.owned(root,relative))==wanted for relative,wanted in value['files'].items()),
            'Local read-only transfer source changed')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local-root',type=Path,required=True)
    parser.add_argument('--remote-root',type=PurePosixPath,required=True)
    parser.add_argument('--host',default='115.191.40.142')
    parser.add_argument('--remote-python',default=REMOTE_PYTHON)
    args = parser.parse_args()
    print(json.dumps(transfer(args.local_root,args.remote_root,host=args.host,remote_python_path=args.remote_python),indent=2))


if __name__=='__main__':
    main()
