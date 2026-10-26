"""Back up only real requested149 BS4 model milestones, with independent SSH authority.

No training, optimizer/RNG reads, model imports or GPU initialization are used.
Owned partial directories survive failures and support rsync append verification.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import uuid

from prepare_handover149_eval import (BACKUP_CONTRACT, EXPECTED_RUN, METADATA,
    PREPARED_METADATA, RELEASE_METADATA, require, sha256, validate_source)

STEPS = (4000, 6000, 7000, 8000, 9000, 10000)
CONTRACT = 'cvpr-handover149-model-only-backup-owner-v1'
AUTHORITY_CONTRACT = 'cvpr-handover149-ssh-checkpoint-authority-v1'
DEFAULT_REFERENCE = Path('/home/zxc/CVPR_Evaluation_runtime/checkpoints/handover149_bs4_20261026/ckpt_5000')
PREPARED = Path(EXPECTED_RUN).parents[2]
RELEASE = Path('/gaozt-test1/datasets/SIMPLE_sonic_v1_1/handover_train149_20261026')
SMALL_FILES = {name:str((PREPARED if name == 'preparation.json' else Path(EXPECTED_RUN))/name)
    for name in METADATA}
SMALL_FILES.update({name:str(PREPARED/original) for name,original in
    zip(PREPARED_METADATA,('run_config.json','argv.txt'))})
SMALL_FILES.update({'release_metadata/'+name:str(RELEASE/name) for name in RELEASE_METADATA})

REMOTE_PROGRAM = r'''
import hashlib, json, os, pathlib, sys
from datetime import datetime, timezone
os.environ['CUDA_VISIBLE_DEVICES']=''
os.nice(19)
allowed=os.sched_getaffinity(0)
cpu=min(allowed)
os.sched_setaffinity(0,{cpu})
request=json.loads(sys.argv[1])
run=pathlib.Path(request['run_directory'])
step=request['step']
marker_path=run/'checkpoint_retention.json'
def record(path):
    path=pathlib.Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('Source must be an ordinary saved file: '+str(path))
    before=path.stat()
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    after=path.stat()
    identity=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    if identity(before)!=identity(after):raise ValueError('Source changed during hashing: '+str(path))
    return {'source_path':str(path),'bytes':after.st_size,'mtime_ns':after.st_mtime_ns,
        'inode':after.st_ino,'device':after.st_dev,'sha256':digest.hexdigest()}
marker_record=record(marker_path)
marker=json.loads(marker_path.read_text())
if not (marker.get('all_models_preserved') is True and type(marker.get('saved_update')) is int
    and marker['saved_update']>=step and step in marker.get('all_saved_model_steps',[])):
    raise ValueError('Checkpoint save completion authority does not include requested step')
files={name:record(path) for name,path in request['small_files'].items()}
model=record(run/'checkpoints'/('ckpt_'+str(step))/'model.safetensors')
if record(marker_path)['sha256']!=marker_record['sha256']:
    raise ValueError('Completion authority changed during hashing')
print(json.dumps({'contract':request['contract'],'step':step,'run_directory':str(run),
    'checked_utc':datetime.now(timezone.utc).isoformat(),'model':model,'files':files,
    'completed_retention_marker':marker,'retention_marker':marker_record,
    'read_policy':{'CUDA_VISIBLE_DEVICES':'','nice':os.nice(0),'cpu_affinity':[cpu],
        'stdlib_only':True,'optimizer_or_rng_read':False}}))
'''
RSYNC_WRAPPER = "import os,sys;os.environ['CUDA_VISIBLE_DEVICES']='';os.nice(19);os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});os.execvp('rsync',['rsync',*sys.argv[1:]])"


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def create_json(path, value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def check_space(path, minimum_free_gib, additional_bytes=0):
    require(isinstance(minimum_free_gib,(int,float)) and math.isfinite(minimum_free_gib)
        and minimum_free_gib >= 0, 'minimum-free-gib must be finite and nonnegative')
    required=math.ceil(minimum_free_gib*1024**3)+additional_bytes
    require(shutil.disk_usage(path).free >= required,
        f'Insufficient free space: require {required} bytes including safety reserve')


def validate_host(host):
    require(isinstance(host,str) and re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.@-]*',host),
        'Use a plain SSH host name or user@host')


def query_source(host, step):
    """Hash the actual remote model and every fixed metadata source independently."""
    validate_host(host)
    request={'contract':AUTHORITY_CONTRACT,'step':step,'run_directory':EXPECTED_RUN,'small_files':SMALL_FILES}
    command=shlex.join(['python3','-c',REMOTE_PROGRAM,json.dumps(request,separators=(',',':'))])
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=30',host,
        "CUDA_VISIBLE_DEVICES='' "+command],check=True,text=True,capture_output=True)
    authority=json.loads(result.stdout)
    validate_authority(authority,step)
    return authority


def validate_authority(authority, step):
    require(authority.get('contract') == AUTHORITY_CONTRACT and authority.get('step') == step
        and authority.get('run_directory') == EXPECTED_RUN, 'SSH authority run/step differs')
    marker=authority['completed_retention_marker']
    require(marker.get('all_models_preserved') is True and type(marker.get('saved_update')) is int
        and marker['saved_update'] >= step and step in marker.get('all_saved_model_steps',[]),
        'SSH authority lacks completed checkpoint save')
    retention=authority.get('retention_marker',{})
    require(retention.get('source_path') == EXPECTED_RUN+'/checkpoint_retention.json'
        and isinstance(retention.get('sha256'),str) and re.fullmatch('[0-9a-f]{64}',retention['sha256']),
        'SSH completed-save marker source identity differs')
    require(set(authority['files']) == set(SMALL_FILES), 'SSH authority fixed metadata set differs')
    records={**authority['files'],'model.safetensors':authority['model']}
    expected={**SMALL_FILES,'model.safetensors':f'{EXPECTED_RUN}/checkpoints/ckpt_{step}/model.safetensors'}
    for name,record in records.items():
        require(record.get('source_path') == expected[name] and type(record.get('bytes')) is int
            and record['bytes'] > 0 and isinstance(record.get('sha256'),str)
            and re.fullmatch('[0-9a-f]{64}',record['sha256']), 'SSH source file identity differs: '+name)
    policy=authority.get('read_policy',{})
    require(policy.get('CUDA_VISIBLE_DEVICES') == '' and policy.get('nice',0) >= 19
        and isinstance(policy.get('cpu_affinity'),list) and len(policy['cpu_affinity']) == 1
        and policy.get('stdlib_only') is True and policy.get('optimizer_or_rng_read') is False,
        'SSH read policy must hide GPU and use nice19 on one allowed CPU')


def same_sources(before, after):
    for key in ('contract','step','run_directory','model','files','completed_retention_marker','retention_marker'):
        require(before[key] == after[key], 'Remote source changed between independent checks: '+key)


def transfer_model(host, remote_path, local_path):
    """Only the model file is transferred; remote rsync also runs with low CPU priority."""
    validate_host(host)
    wrapper=shlex.join(['python3','-c',RSYNC_WRAPPER])
    subprocess.run(['rsync','--partial','--append-verify','--protect-args',
        '-e','ssh -o BatchMode=yes -o ConnectTimeout=30','--rsync-path='+wrapper,
        '--',host+':'+remote_path,str(local_path)],check=True,stdout=sys.stderr)


def validate_tree(root):
    """An owned directory cannot be used to follow foreign links or carry extra payloads."""
    allowed=set(SMALL_FILES)|{'model.safetensors','BACKUP_OWNER.json','backup_receipt.json',
        'source_authority_before.json','source_authority_after.json'}
    for path in root.rglob('*'):
        require(not path.is_symlink(), 'Refuse symlinks in backup directory')
        if path.is_file():
            name=path.relative_to(root).as_posix()
            require(name in allowed or re.fullmatch(r'attempts/[0-9a-f]{32}/(?:source_authority_before|source_authority_after|failure)\.json',name),
                'Refuse unrelated file in owned partial directory: '+name)
        else:
            require(path.is_dir(), 'Refuse special file in backup directory')


@contextmanager
def step_lock(root, step):
    descriptor=os.open(root/f'.ckpt_{step}.backup.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
    try:
        fcntl.flock(descriptor,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield
    finally:
        os.close(descriptor)


def validate_owned(path, owner):
    require(path.is_dir() and not path.is_symlink(), 'Backup directory must be an ordinary directory')
    require((path/'BACKUP_OWNER.json').is_file() and read(path/'BACKUP_OWNER.json') == owner,
        'Refuse foreign nonempty backup/partial directory')
    validate_tree(path)


def copy_metadata(reference, partial, authority, minimum_free_gib):
    for name,record in authority['files'].items():
        source,target=reference/name,partial/name
        require(source.is_file() and not source.is_symlink() and source.stat().st_size == record['bytes']
            and sha256(source) == record['sha256'], 'Reference differs from independent A100 metadata SHA: '+name)
        check_space(partial,minimum_free_gib,record['bytes'])
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            require(target.is_file() and not target.is_symlink() and target.stat().st_size == record['bytes']
                and sha256(target) == record['sha256'], 'Owned partial metadata changed: '+name)
            continue
        try:
            os.link(source,target)
        except OSError as error:
            if error.errno != 18:raise
            shutil.copy2(source,target)
        require(sha256(target) == record['sha256'], 'Copied metadata SHA differs: '+name)


def result(source, authority):
    return {'source':str(source),'expected_model_sha256':authority['model']['sha256'],
        'authority_path':str(source/'source_authority_before.json')}


def backup(step, output_root, *, ssh_host='115.191.40.142', reference=DEFAULT_REFERENCE, minimum_free_gib=12):
    require(type(step) is int and step in STEPS, 'Only requested149 task checkpoints may be backed up')
    validate_host(ssh_host)
    root=Path(output_root).absolute()
    require(not root.is_symlink(), 'Output root must not be a symlink')
    root.mkdir(parents=True,exist_ok=True)
    root=root.resolve(strict=True)
    check_space(root,minimum_free_gib)
    reference=Path(reference).resolve(strict=True)
    final,staging=root/f'ckpt_{step}',root/f'.ckpt_{step}.partial'
    owner={'contract':CONTRACT,'step':step,'run_directory':EXPECTED_RUN,
        'ssh_host':ssh_host,'output_root':str(root)}
    with step_lock(root,step):
        if final.exists() or final.is_symlink():
            validate_owned(final,owner)
            before,after=(read(final/name) for name in ('source_authority_before.json','source_authority_after.json'))
            validate_authority(before,step);validate_authority(after,step);same_sources(before,after)
            validate_source(final,final/'release_metadata',expected_step=step,
                expected_model_sha256=before['model']['sha256'])
            return result(final,before)
        if staging.exists() or staging.is_symlink():
            require(staging.is_dir() and not staging.is_symlink(), 'Partial staging must be an ordinary directory')
            if not any(staging.iterdir()):
                create_json(staging/'BACKUP_OWNER.json',owner)
            require((staging/'BACKUP_OWNER.json').is_file()
                and not (staging/'BACKUP_OWNER.json').is_symlink()
                and read(staging/'BACKUP_OWNER.json') == owner,
                'Refuse foreign nonempty partial staging directory')
            require(all(p.name in ('BACKUP_OWNER.json',f'ckpt_{step}') for p in staging.iterdir()),
                'Refuse unrelated staging payload')
        else:
            staging.mkdir()
            create_json(staging/'BACKUP_OWNER.json',owner)
        partial=staging/f'ckpt_{step}'
        if partial.exists() or partial.is_symlink():
            validate_owned(partial,owner)
        else:
            partial.mkdir()
            create_json(partial/'BACKUP_OWNER.json',owner)
        attempt=partial/'attempts'/uuid.uuid4().hex
        attempt.mkdir(parents=True)
        try:
            check_space(root,minimum_free_gib)
            before=query_source(ssh_host,step)
            validate_authority(before,step)
            create_json(attempt/'source_authority_before.json',before)
            if (partial/'source_authority_before.json').exists():
                same_sources(read(partial/'source_authority_before.json'),before)
            else:
                create_json(partial/'source_authority_before.json',before)
            weights=partial/'model.safetensors'
            require(not weights.is_symlink() and (not weights.exists() or weights.is_file()),
                'Partial model must be an ordinary file')
            present=weights.stat().st_size if weights.exists() else 0
            require(present <= before['model']['bytes'], 'Partial model exceeds actual source size')
            check_space(root,minimum_free_gib,before['model']['bytes']-present+sum(r['bytes'] for r in before['files'].values()))
            copy_metadata(reference,partial,before,minimum_free_gib)
            check_space(root,minimum_free_gib,before['model']['bytes']-present)
            transfer_model(ssh_host,before['model']['source_path'],weights)
            check_space(root,minimum_free_gib)
            require(weights.is_file() and not weights.is_symlink()
                and weights.stat().st_size == before['model']['bytes'], 'Transferred model is incomplete')
            local_sha=sha256(weights)
            require(local_sha == before['model']['sha256'], 'Actual local full-model SHA differs from A100 authority')
            after=query_source(ssh_host,step)
            validate_authority(after,step)
            create_json(attempt/'source_authority_after.json',after)
            same_sources(before,after)
            if (partial/'source_authority_after.json').exists():
                same_sources(read(partial/'source_authority_after.json'),after)
            else:
                create_json(partial/'source_authority_after.json',after)
            receipt={'contract':BACKUP_CONTRACT,'step':step,'optimizer_copied':False,'created_utc':now(),
                'run_identity':{'run_directory':EXPECTED_RUN,'run_config_sha256':before['files']['run_config.json']['sha256'],
                    'preparation_sha256':before['files']['preparation.json']['sha256']},
                'model':{**before['model'],'source_sha256_before':before['model']['sha256'],
                    'source_sha256_after':after['model']['sha256'],'local_sha256':local_sha},
                'metadata':{name:before['files'][name]['sha256'] for name in METADATA},
                'release_metadata':{name:before['files']['release_metadata/'+name] for name in RELEASE_METADATA},
                'completed_retention_marker':after['completed_retention_marker'],'training_modified':False,
                'source_read_policy':before['read_policy'],'independent_ssh_authority':True}
            check_space(root,minimum_free_gib)
            if not (partial/'backup_receipt.json').exists():
                create_json(partial/'backup_receipt.json',receipt)
            require(read(partial/'backup_receipt.json')['model']['local_sha256'] == local_sha,
                'Existing owned backup receipt differs from actual source')
            # The staging payload already has its real ckpt_<step> basename, so
            # every production source check passes before the atomic commit.
            validate_source(partial,partial/'release_metadata',expected_step=step,
                expected_model_sha256=before['model']['sha256'])
            check_space(root,minimum_free_gib)
            require(not final.exists() and not final.is_symlink(), 'Final backup appeared while downloading')
            partial.rename(final)
            return result(final,before)
        except BaseException as error:
            failure_root=attempt if partial.exists() else final/'attempts'/attempt.name
            create_json(failure_root/'failure.json',{'status':'failed','checked_utc':now(),'error':repr(error),
                'partial_retained':partial.exists(),'model_deleted':False})
            raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step',type=int,choices=STEPS,required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--ssh-host',default='115.191.40.142')
    parser.add_argument('--reference',type=Path,default=DEFAULT_REFERENCE)
    parser.add_argument('--minimum-free-gib',type=float,default=12)
    args=parser.parse_args()
    print(json.dumps(backup(args.step,args.output_root,ssh_host=args.ssh_host,
        reference=args.reference,minimum_free_gib=args.minimum_free_gib),separators=(',',':')))


if __name__ == '__main__':
    main()
