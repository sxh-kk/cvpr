"""Read-only, verified transfer of one completed Pick training checkpoint.

One source transmission SHA256, a persisted local SHA256 and stable metadata
bind the selected save. Training and optimizer files are never changed/copied.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import traceback

from prepare_pick_eval import (PROFILES, SAVE_STEPS, QUEUE_ROOT, SERVER_BASE,
    QUEUE_PLAN_SHA, QUEUE_PLAN, PickDeployment)

MODEL_BYTES = 12066487572
REMOTE_PYTHON = '/gaozt-test1/datasets/SIMPLE_sonic_v1_1/.pipeline/envs/task-finetune-py311-cu126/bin/python'
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12', '115.191.40.142']
LOG = Path('/home/zxc/CVPR2027/log.md')
LAUNCHING = False
PENDING_SIGNAL = None


class StopRequested(BaseException):
    pass


def interrupted(signum, frame):
    global PENDING_SIGNAL
    if LAUNCHING:
        PENDING_SIGNAL = signum
        return
    raise StopRequested('Checkpoint transfer received signal '+str(signum))


def stream_transfer(command, partial, transfer_log):
    """Own only this SSH child and reap it before publishing any failure."""
    global LAUNCHING, PENDING_SIGNAL
    process = None
    try:
        with partial.open('xb') as output, transfer_log.open('xb') as log:
            LAUNCHING = True
            try:
                process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=output,
                    stderr=log, start_new_session=True)
            finally:
                LAUNCHING = False
                if PENDING_SIGNAL is not None:
                    signum, PENDING_SIGNAL = PENDING_SIGNAL, None
                    raise StopRequested('Checkpoint transfer received signal '+str(signum))
            status = process.wait()
            output.flush()
            os.fsync(output.fileno())
            return status
    finally:
        if process is not None and process.poll() is None:
            # Popen owns this unreaped child identity; no name-based/global kill.
            previous = {sig:signal.signal(sig, signal.SIG_IGN) for sig in (signal.SIGINT,signal.SIGTERM)}
            try:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
            finally:
                for sig, handler in previous.items():
                    signal.signal(sig, handler)


def require(value, message):
    if not value:
        raise ValueError(message)


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def record(paths, description):
    names = '、'.join('`'+os.path.relpath(p, LOG.parent)+'`' for p in paths)
    with LOG.open('a') as f:
        f.write('- '+datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S +08:00')+
                ' | '+names+'：'+description+'。\n')


def save(path, value):
    path = Path(path)
    temp = path.with_name('.'+path.name+'.partial')
    with temp.open('x') as f:
        f.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def identity(path):
    s = Path(path).stat()
    return {key: getattr(s, key) for key in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns')}


SNAPSHOT = r'''
from pathlib import Path
import json,sys,os,hashlib,base64,re
v=json.load(sys.stdin)
os.environ['CUDA_VISIBLE_DEVICES']='';os.nice(19)
os.sched_setaffinity(0,{max(os.sched_getaffinity(0))})
def require(x,m):
 if not x:raise ValueError(m)
def regular(p):
 require(p.is_file() and not p.is_symlink(),'Missing regular source: '+str(p));return p
root=Path(v['prepared']);release=Path(v['release'])
state=json.loads(regular(root/'training_state.json').read_text())
require(state.get('state')=='complete' and state.get('returncode')==0 and state.get('task_id')==v['task_id'] and state.get('max_optimizer_steps')==10000,'Selected task training is not complete')
run_name=state.get('run_directory')
prefix=str(root)+'/training/posttrain/'+v['profile']+'_joint38000_statekeep_bs4_8gpu.us.flow1000.cosine.lr1.0e-05.b32.gpus8.'
require(isinstance(run_name,str) and re.fullmatch(re.escape(prefix)+r'[0-9]{12}',run_name) is not None,'Unexpected selected task run namespace')
run=Path(run_name)
if v['expected_project'] is not None:require(run.name==v['expected_project'],'Selected source run differs from known training')
queue_raw=regular(Path(v['queue_root'])/'QUEUE_PLAN.json').read_bytes()
require(hashlib.sha256(queue_raw).hexdigest()==v['queue_plan_sha256'],'Remote authorized training queue changed')
ret=json.loads(regular(run/'checkpoint_retention.json').read_text())
require(ret.get('all_models_preserved') is True and ret.get('saved_update')==10000 and ret.get('all_saved_model_steps')==v['checkpoint_steps'] and ret.get('task_id')==v['task_id'] and v['step'] in ret['all_saved_model_steps'],'Selected save is not completed and retained')
model=regular(run/'checkpoints'/('ckpt_'+str(v['step']))/'model.safetensors')
s=model.stat();stat={k:getattr(s,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns')}
require(stat['st_size']==v['bytes'],'Wrong selected model size')
paths={name:run/name for name in ('run_config.json','argv.txt','dataset_statistics.json','task_initialization.json','task_window_selection.json','checkpoint_retention.json','model_config_state_keep.json','gpu_check_update10000.json')}
paths.update({'prepared_run_config.json':root/'run_config.json','prepared_argv.txt':root/'argv.txt','preparation.json':root/'preparation.json','training_state.json':root/'training_state.json'})
paths.update({'state_policy_migration_rank_'+str(i)+'.json':run/('state_policy_migration_rank_'+str(i)+'.json') for i in range(8)})
paths.update({'release_metadata/'+name:release/name for name in ('packed/PACK_READY.json','dino/manifest.json','A100_PICO_DATASET_READY.json','dino/REMOTE_DINO_RELOCATION.json')})
metadata={}
for name,p in paths.items():
 raw=regular(p).read_bytes();require(len(raw)<20*1024**2,'Unexpected oversized metadata')
 metadata[name]={'source_path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'base64':base64.b64encode(raw).decode()}
print(json.dumps({'model_stat':stat,'model_path':str(model),'run_directory':str(run),'metadata':metadata,'training_completed':True,'checkpoint_save_completed':True}))
'''

STREAM = r'''
from pathlib import Path
import json,sys,os,hashlib
v=json.loads(sys.argv[1]);os.environ['CUDA_VISIBLE_DEVICES']='';os.nice(19)
os.sched_setaffinity(0,{max(os.sched_getaffinity(0))})
p=Path(v['model'])
def stat():
 s=p.stat();return {k:getattr(s,k)for k in ('st_dev','st_ino','st_size','st_mtime_ns')}
before=stat()
if before!=v['expected_stat'] or p.is_symlink():raise ValueError('Source changed before transfer')
h=hashlib.sha256();total=0
with p.open('rb')as f:
 for block in iter(lambda:f.read(8*1024**2),b''):
  h.update(block);sys.stdout.buffer.write(block);total+=len(block)
sys.stdout.buffer.flush();after=stat()
if before!=after or total!=v['bytes']:raise ValueError('Source changed during transfer')
print('CVPR_MODEL_RECEIPT '+json.dumps({'sha256':h.hexdigest(),'bytes':total,'stat_before':before,'stat_after':after}),file=sys.stderr,flush=True)
'''


def selection(profile, step, queue_plan=QUEUE_PLAN):
    require(profile in PROFILES and type(step) is int and step in SAVE_STEPS, 'Select an authorized profile/save')
    require(digest(queue_plan) == QUEUE_PLAN_SHA, 'Local authorized queue changed')
    item = PROFILES[profile]
    return dict(profile=profile, step=step, task_id=item['task_id'],
        prepared=QUEUE_ROOT+'/runs/'+profile, release=SERVER_BASE+'/'+item['release_name'],
        queue_root=QUEUE_ROOT, queue_plan_sha256=QUEUE_PLAN_SHA,
        expected_project=item['project_name'], bytes=MODEL_BYTES, checkpoint_steps=list(SAVE_STEPS))


def remote_snapshot(selected, output, label):
    result = subprocess.run([*SSH, shlex.join([REMOTE_PYTHON, '-B', '-c', SNAPSHOT])],
        input=json.dumps(selected), text=True, capture_output=True, timeout=120)
    path = output/f'remote_{label}.stderr.log'
    with path.open('x') as f:
        f.write(result.stderr)
    record([path], '只读检查所选已完成训练的来源，不修改服务器')
    require(result.returncode == 0, 'Remote completed-run check failed: '+result.stderr[-1500:])
    return json.loads(result.stdout)


def fetch(profile, step, output, queue_plan=QUEUE_PLAN):
    selected = selection(profile, step, queue_plan)
    output = Path(output).absolute()
    require(output.parent.is_dir() and not output.exists() and not output.is_symlink()
        and not any(p.is_symlink() for p in output.parents), 'Use a fresh independent transfer directory')
    require(not output.resolve().is_relative_to(Path(__file__).resolve().parents[2]), 'Store weights outside the repository')
    fs = os.statvfs(output.parent)
    require(fs.f_bavail*fs.f_frsize > MODEL_BYTES+4*1024**3, 'Insufficient space for one selected source and safe reserve')
    output.mkdir(mode=0o700)
    source = output/'source'
    source.mkdir(mode=0o700)
    record([output], '建立本次单任务单权重传输目录，只接收模型和24份来源metadata，不复制优化器')
    try:
        before = remote_snapshot(selected, output, 'before')
        require(before['training_completed'] is True and before['checkpoint_save_completed'] is True, 'Incomplete selected run/save')
        metadata = {}
        for name, item in before['metadata'].items():
            relative = Path(name)
            require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe metadata name')
            raw = base64.b64decode(item['base64'], validate=True)
            require(hashlib.sha256(raw).hexdigest() == item['sha256'] and len(raw) == item['bytes'], 'Bad metadata payload')
            target = source/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as f:
                f.write(raw)
            metadata[name] = {key:item[key] for key in ('source_path','sha256','bytes')}
        binding = PickDeployment(profile, step, source, queue_plan)
        require(before['run_directory'] == binding.EXPECTED_PROJECT and set(metadata) == set(binding.METADATA_SOURCES),
            'Transfer source differs from independently authorized deployment')
        require(all(item['source_path'] == binding.METADATA_SOURCES[name] for name,item in metadata.items()), 'Metadata source path differs')
        binding.validate_training(source, json.loads((source/'run_config.json').read_text()),
            json.loads((source/'preparation.json').read_text()))
        partial = source/'model.safetensors.partial'
        transfer_log = output/'MODEL_TRANSFER.log'
        save(output/'FETCH_STATE.json', dict(status='copying_selected_model', profile=profile, step=step, expected_bytes=MODEL_BYTES, started_beijing=now()))
        value = dict(model=before['model_path'], expected_stat=before['model_stat'], bytes=MODEL_BYTES)
        command = [*SSH, shlex.join([REMOTE_PYTHON, '-B', '-c', STREAM, json.dumps(value,separators=(',',':'))])]
        record([source, partial, transfer_log, output/'FETCH_STATE.json'], '开始单核低优先级只读SSH传输，源端对传输全字节计算一次SHA，失败保留partial')
        status = stream_transfer(command, partial, transfer_log)
        require(status == 0 and partial.stat().st_size == MODEL_BYTES, 'Transfer failed with actual exit '+str(status))
        receipts = [json.loads(line.removeprefix('CVPR_MODEL_RECEIPT ')) for line in transfer_log.read_text().splitlines()
                    if line.startswith('CVPR_MODEL_RECEIPT ')]
        require(len(receipts) == 1, 'Missing unique source transmission receipt')
        remote = receipts[0]
        require(remote['bytes'] == MODEL_BYTES and remote['stat_before'] == remote['stat_after'] == before['model_stat'], 'Source model identity changed')
        local_sha = digest(partial)
        require(local_sha == remote['sha256'], 'Persisted local SHA differs from source transmission')
        after = remote_snapshot(selected, output, 'after')
        require(before == after, 'Completed source metadata/model changed during transfer')
        os.replace(partial, source/'model.safetensors')
        model = source/'model.safetensors'
        receipt = dict(contract='cvpr-pick-checkpoint-backup-v1', status='passed', profile=profile,
            task_id=selected['task_id'], step=step, training_completed=True, checkpoint_save_completed=True,
            optimizer_copied=False, metadata=metadata,
            copy_method='single remote SHA256 over all transmitted bytes; persisted local full SHA256; stable source stat and24metadata before/after',
            model=dict(source_path=before['model_path'], bytes=MODEL_BYTES, source_sha256=remote['sha256'],
                local_sha256=local_sha, source_stat_before=before['model_stat'], source_stat_after=after['model_stat'], local_stat=identity(model)),
            run_identity=dict(run_directory=before['run_directory'], prepared_root=selected['prepared'],
                run_config_sha256=metadata['run_config.json']['sha256'], preparation_sha256=metadata['preparation.json']['sha256']),
            completed_beijing=now())
        save(source/'backup_receipt.json', receipt)
        save(output/'FETCH_STATE.json', dict(status='model_only_backup_complete', profile=profile, step=step,
            model_sha256=local_sha, model_bytes=MODEL_BYTES, server_training_modified=False, completed_beijing=now()))
        save(output/'FETCH_EXIT.json', dict(exit_code=0, finished_beijing=now()))
        record([source, output/'FETCH_STATE.json', output/'FETCH_EXIT.json'], '所选模型回传完成，来源与落盘完整SHA一致且24metadata前后不变；原子发布，不代表测评已完成')
        return receipt
    except BaseException as error:
        save(output/'FETCH_EXIT.json', dict(exit_code=1, error=repr(error), traceback=traceback.format_exc(), finished_beijing=now()))
        record([output, output/'FETCH_EXIT.json'], '权重回传实际失败并保留错误及partial，不自动重试、不改服务器训练')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=tuple(PROFILES), required=True)
    parser.add_argument('--step', type=int, choices=SAVE_STEPS, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--queue-plan', type=Path, default=QUEUE_PLAN)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.nice(19)
    os.sched_setaffinity(0, {max(os.sched_getaffinity(0))})
    previous = {sig:signal.signal(sig, interrupted) for sig in (signal.SIGINT,signal.SIGTERM)}
    try:
        receipt = fetch(args.profile, args.step, args.output, args.queue_plan)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    print(json.dumps(dict(status=receipt['status'], profile=args.profile, step=args.step,
        model_sha256=receipt['model']['local_sha256'])))


if __name__ == '__main__':
    main()
