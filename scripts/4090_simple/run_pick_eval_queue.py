"""Durable queue for the authorized XMovePick/XMoveBendPick checkpoint5000 pair.

Wait for existing training and GPU users; transfer, prepare and evaluate once.
Preserve every model, video, failure and real exit. Never restart training.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time
import traceback

from fetch_pick_checkpoint import SSH, REMOTE_PYTHON, record, save, digest
from prepare_pick_eval import PROFILES, QUEUE_ROOT, QUEUE_PLAN, QUEUE_PLAN_SHA

REPO = Path(__file__).resolve().parents[2]
RUNTIME = Path('/home/zxc/CVPR_Evaluation_runtime')
MODEL_PY = '/home/zxc/miniconda3/envs/cvpr-simple-model311/bin/python'
SIM_PY = '/home/zxc/miniconda3/envs/cvpr-simple-sim310/bin/python'
PRIOR_EXIT = Path('/var/tmp/cvpr-bendpick-init-contact-20261111/STABLE_SUPPORT_EXIT.json')
OWNED_HELPER = Path('/var/tmp/cvpr-bend-pick-ckpt5000-20261111/retry_campaign_003.py')
OWNED_SHA = 'dba9b03c404faf66785acfebb96fe64925982b614e4ead9c80f6d9816b3a07e2'
PROFILES_SELECTED = ('xmove_pick', 'xmove_bend_pick')
INITIALIZATION_PROFILE = 'native_saved_scene_observed_v1'
LAUNCHING = False
PENDING_SIGNAL = None


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def interrupted(signum, frame):
    global PENDING_SIGNAL
    if LAUNCHING:
        PENDING_SIGNAL = signum
        return
    raise KeyboardInterrupt('Pick evaluation queue received signal '+str(signum))


def load_owned_helpers():
    raw = OWNED_HELPER.read_bytes()
    require(not OWNED_HELPER.is_symlink() and hashlib.sha256(raw).hexdigest() == OWNED_SHA,
        'Frozen process ownership helper differs')
    spec = importlib.util.spec_from_file_location('pick_eval_frozen_owned', OWNED_HELPER)
    module = importlib.util.module_from_spec(spec)
    exec(compile(raw, str(OWNED_HELPER), 'exec'), module.__dict__)
    return module


def validate_acceptance(accepted):
    require(accepted.get('status') == 'passed' and accepted.get('user_authorized') is True and
        accepted.get('profiles') == list(PROFILES_SELECTED) and type(accepted.get('checkpoint_step')) is int and
        accepted['checkpoint_step'] == 5000 and accepted.get('rtc_mode') == 'test_time' and
        accepted.get('execute_steps') == 15 and accepted.get('cases_per_profile') == 30 and
        accepted.get('initialization_profile') == INITIALIZATION_PROFILE and accepted.get('repo') == str(REPO),
        'Queue scope differs from the authorized two5000 RTC evaluations')
    require(accepted.get('queue_plan_sha256') == digest(QUEUE_PLAN) == QUEUE_PLAN_SHA, 'Training queue identity differs')
    actual = subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()
    require(actual == accepted.get('git_revision'), 'Evaluation repository HEAD changed')
    sources = accepted.get('source_sha256', {})
    critical = {'scripts/4090_simple/'+name for name in ('run_pick_eval_queue.py','fetch_pick_checkpoint.py',
        'prepare_pick_eval.py','evaluate_pick30.py','pick_eval_protocol.py','run_closed_loop.py',
        'pico_xmove_pick_task.py','pico_xmove_bend_pick_task.py','bend_initialization_monitor.py')}
    require(isinstance(sources,dict) and critical <= set(sources), 'Missing full frozen production source inventory')
    for name, expected in sources.items():
        relative = Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts and isinstance(expected,str) and
            len(expected) == 64 and set(expected) <= set('0123456789abcdef'), 'Unsafe frozen source identity')
        p = REPO/relative
        require(p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(REPO) and
            p.stat().st_size < 32*1024**2 and digest(p) == expected, 'Frozen source changed: '+name)


def child_environment(cpu):
    return dict(os.environ, CUDA_VISIBLE_DEVICES='' if cpu else '0', TMPDIR='/dev/shm',
        PYTHONDONTWRITEBYTECODE='1', PYTHONOPTIMIZE='0', PYTHONNOUSERSITE='1',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
        OMP_NUM_THREADS='1' if cpu else '4', MKL_NUM_THREADS='1' if cpu else '4',
        OPENBLAS_NUM_THREADS='1', CVPR_SIMPLE_RUNTIME=str(RUNTIME),
        PYTHONPATH=os.pathsep.join(str(REPO/p) for p in ('scripts/4090_simple','.',
            'VLMExpert/src','VLMExpert','MotionExpert/src','pipeline/src','VLMExpert/third_party/SIMPLE/src')))


def run_owned(command, log_path, cpu, helpers, launched):
    """Reap only the child and descendants whose real identities were observed."""
    global LAUNCHING, PENDING_SIGNAL
    process = None
    owned = {}
    try:
        with log_path.open('xb') as log:
            LAUNCHING = True
            try:
                prefix = ['taskset','-c',str(max(os.sched_getaffinity(0)))] if cpu else []
                process = subprocess.Popen([*prefix,'nice','-n','19',*command], cwd=REPO, env=child_environment(cpu),
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                identity = helpers.process_identity(process.pid)
                require(identity is not None and identity['parent'] == os.getpid() and
                    identity['uid'] == os.getuid() and identity['group'] == identity['session'] == process.pid,
                    'Cannot establish child process ownership')
                owned[process.pid] = identity
            finally:
                LAUNCHING = False
                if PENDING_SIGNAL is not None:
                    signum, PENDING_SIGNAL = PENDING_SIGNAL, None
                    raise KeyboardInterrupt('Deferred queue stop '+str(signum))
            launched(process.pid, identity)
            while process.poll() is None:
                helpers.collect_owned(owned)
                time.sleep(.5)
            status = process.wait()
        returncode = status
    finally:
        previous = {sig:signal.signal(sig,signal.SIG_IGN) for sig in (signal.SIGINT,signal.SIGTERM)}
        try:
            cleanup = helpers.cleanup_campaign(process,owned)
            if process is not None:
                require(cleanup['status'] == 'verified_stopped', 'Stage process cleanup failed')
        finally:
            for sig, handler in previous.items():
                signal.signal(sig,handler)
    require(returncode == 0, 'Stage actual exit code '+str(returncode)+'; retained '+str(log_path))
    return cleanup


INSPECT = r'''
from pathlib import Path
import json,sys
r=Path(sys.argv[1]);profile=sys.argv[2]
def read(p):return json.loads(p.read_text())if p.is_file()else None
print(json.dumps({'state':read(r/'runs'/profile/'training_state.json'),'queue':read(r/'QUEUE_STATUS.json'),'exit':read(r/'PROCESS_EXIT.json')}))
'''


def training_state(profile):
    result = subprocess.run([*SSH,shlex.join([REMOTE_PYTHON,'-B','-c',INSPECT,QUEUE_ROOT,profile])],
        stdin=subprocess.DEVNULL,capture_output=True,text=True,timeout=35)
    require(result.returncode == 0, 'Read-only training status unavailable: '+result.stderr[-1000:])
    return json.loads(result.stdout)


def wait_training(profile, update):
    failures = 0
    while True:
        try:
            observed = training_state(profile)
        except Exception as error:
            failures += 1
            update(stage='waiting_training_connection', consecutive_read_failures=failures,error=repr(error))
            time.sleep(30)
            continue
        failures = 0
        state = observed.get('state') or {}
        if state.get('state') == 'complete':
            require(state.get('returncode') == 0 and state.get('task_id') == PROFILES[profile]['task_id'],
                'Existing training completed unsuccessfully or belongs to another task')
            return observed
        require(state.get('state') not in ('failed','error','stopped'), 'Existing selected training failed; no restart was requested')
        require(observed.get('exit') is None, 'Server training queue exited before selected task completed')
        update(stage='waiting_existing_training', server_active_profile=(observed.get('queue') or {}).get('active_profile'),
            training_state=state.get('state','not_started'))
        time.sleep(30)


def wait_gpu(update):
    while True:
        previous = read(PRIOR_EXIT) if PRIOR_EXIT.is_file() else None
        if previous is not None:
            require(previous.get('cleanup',{}).get('status') in ('verified_stopped','not_started'),
                'Previous Bend campaign ownership cleanup is unresolved')
            compute = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True,timeout=15)
            if not compute.strip():
                return
        update(stage='waiting_existing_local_evaluation', previous_exit_published=previous is not None)
        time.sleep(15)


def validate_final(final, profile, model_sha):
    require(final.get('contract') == 'cvpr-pick-fixed-official30-sonic-rtc-diagnostic-v1' and
        final.get('status') == 'complete' and final.get('profile') == profile and
        final.get('task_id') == PROFILES[profile]['task_id'] and type(final.get('checkpoint_step')) is int and
        final['checkpoint_step'] == 5000 and final.get('model_sha256') == model_sha and
        final.get('planned_cases') == final.get('technically_valid_cases') == 30 and final.get('technical_errors') == 0 and
        final.get('initialization_profile') == INITIALIZATION_PROFILE and final.get('original_official_initialization') is True,
        'Selected task evaluation scope or technical coverage differs')
    rows = final.get('cases',[])
    require(len(rows) == 30 and sorted((r['level'],r['case_index']) for r in rows) ==
        [(level,index) for level in range(3) for index in range(10)] and all(r['status'] == 'complete' and
        type(r.get('native_success')) is bool and r.get('initialization_profile') == INITIALIZATION_PROFILE and
        r.get('rtc_execution',{}).get('status') == 'passed' for r in rows), 'Incomplete real30-case records')
    require(type(final.get('native_successes')) is int and final['native_successes'] == sum(r['native_success'] for r in rows),
        'Native success count differs from actual cases')
    policy = final.get('evaluation_policy',{})
    require(policy.get('rtc_mode') == 'test_time' and policy.get('rtc_enabled') is True and
        policy.get('execute_steps') == 15 and policy.get('control_ticks_per_window') == 25, 'RTC protocol differs')


def execute_queue(output, accepted, helpers):
    state = dict(contract='cvpr-two-xmove5000-local-evaluation-queue-v1',status='running',
        started_beijing=now(),profiles=list(PROFILES_SELECTED),checkpoint_step=5000,rtc_mode='test_time',
        initialization_profile=INITIALIZATION_PROFILE,automatic_retries=0,server_training_modified=False,
        repo=str(REPO),git_revision=accepted['git_revision'],jobs=[])
    active = output/'ACTIVE.json'
    def publish():
        state['updated_beijing'] = now()
        save(active,state)
    publish()
    try:
        for profile in PROFILES_SELECTED:
            job = dict(profile=profile,task_id=PROFILES[profile]['task_id'],step=5000,status='pending')
            state['jobs'].append(job)
            directory = output/(profile+'_ckpt5000')
            def update(**fields):
                before = dict(job)
                job.update(fields)
                if job != before:
                    publish()
                    record([active,directory], '更新单权重流程：'+profile+'，'+job.get('stage',job['status']))
            observed = wait_training(profile,update)
            validate_acceptance(accepted)
            directory.mkdir(mode=0o700)
            save(directory/'COMPLETED_TRAINING_SNAPSHOT.json',observed)
            record([directory], '现有完整训练已完成，开始该任务5000步权重的独立回传、部署、30场景测评')
            fs = os.statvfs(output)
            require(fs.f_bavail*fs.f_frsize > 12066487572+1616029181+4*1024**3,
                'Insufficient space for retained source/Motion plus safe reserve; no old files were deleted')
            transfer = directory/'transfer'
            deployment = directory/'deployment'
            campaign = directory/'official30_rtc_001'
            def launched(pid, identity):
                update(stage_pid=pid,stage_identity=identity)
            update(status='running',stage='fetching_verified_model')
            run_owned([MODEL_PY,'-B',str(REPO/'scripts/4090_simple/fetch_pick_checkpoint.py'),
                '--profile',profile,'--step','5000','--output',str(transfer)],directory/'fetch.log',True,helpers,launched)
            receipt = read(transfer/'source/backup_receipt.json')
            model_sha = receipt['model']['local_sha256']
            require(receipt.get('status') == 'passed' and receipt.get('profile') == profile and receipt.get('step') == 5000,
                'Fetched save identity differs')
            update(stage='strict_CPU_preparation',model_sha256=model_sha)
            local_release = RUNTIME/'pico_latest300_20261109/processing_001/releases'/profile
            run_owned([MODEL_PY,'-B',str(REPO/'scripts/4090_simple/prepare_pick_eval.py'),
                '--profile',profile,'--step','5000','--source',str(transfer/'source'),'--resources',str(RUNTIME/'resources'),
                '--release',str(local_release),'--output',str(deployment),'--repo',str(REPO),
                '--expected-model-sha256',model_sha],directory/'prepare.log',True,helpers,launched)
            ready = read(deployment/'LOCAL_RUN_READY.json')
            require(ready.get('joint_sha256') == model_sha and ready.get('profile') == profile and
                ready.get('step') == 5000 and ready.get('training_completed') is True and
                ready.get('component_load_validation',{}).get('status') == 'passed', 'Strict CPU deployment not ready')
            wait_gpu(update)
            validate_acceptance(accepted)
            update(stage='evaluating_30_cases',campaign_directory=str(campaign),deployment=str(deployment))
            command = [SIM_PY,'-B',str(REPO/'scripts/4090_simple/evaluate_pick30.py'),
                '--profile',profile,'--checkpoint-step','5000','--deployment',str(deployment),
                '--data-root',str(RUNTIME/'resources/simple_local_usd_v1'),
                '--training-manifest',str(transfer/'source/release_metadata/dino/manifest.json'),
                '--output',str(campaign),'--runtime-root',str(RUNTIME),
                '--expected-model-sha256',model_sha,'--port','21125','--sampling-seed','20261105',
                '--rtc-mode','test_time','--execute-steps','15']
            save(directory/'CAMPAIGN_COMMAND.json',dict(command=command,initialization_profile=INITIALIZATION_PROFILE))
            cleanup = run_owned(command,directory/'campaign.log',False,helpers,launched)
            final_path = campaign/'FINAL_EVALUATION_ACCEPTANCE.json'
            final = read(final_path)
            validate_final(final,profile,model_sha)
            validate_acceptance(accepted)
            update(status='complete',stage='validated_complete',native_successes=final['native_successes'],
                technically_valid_cases=30,technical_errors=0,result=str(final_path),result_sha256=digest(final_path),
                initialization_warning_case_count=final.get('initialization_warning_case_count'),
                initialization_warning_count=final.get('initialization_warning_count'),
                initialization_warnings=final.get('initialization_warnings'),cleanup=cleanup,completed_beijing=now())
        state.update(status='complete',completed_beijing=now())
        publish()
        save(output/'RESULTS.json',state)
        save(output/'EXIT.json',dict(exit_code=0,status='complete',completed_beijing=now()))
        record([output], '两个任务5000权重均完成30例实际RTC测评，保留真实成功率、所有视频、源权重与完整验收；服务器训练保持原样')
        return 0
    except BaseException as error:
        state.update(status='failed_preserved',error=repr(error),finished_beijing=now())
        publish()
        save(output/'EXIT.json',dict(exit_code=1,status='failed_preserved',error=repr(error),
            traceback=traceback.format_exc(),finished_beijing=now()))
        record([output], '本次队列非零退出，保留当前模型、真实错误及全部证据，不自动重试或修改服务器训练')
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--acceptance',type=Path,required=True)
    args = parser.parse_args(argv)
    output = args.output.absolute()
    require(output.is_dir() and not output.is_symlink() and not any(p.is_symlink() for p in output.parents)
        and not output.resolve().is_relative_to(REPO), 'Use a prepared independent queue root')
    require(not any((output/name).exists() or (output/name).is_symlink() for name in
        ('ACTIVE.json','EXIT.json','RESULTS.json',*(p+'_ckpt5000' for p in PROFILES_SELECTED))), 'Preserve existing queue output; no retry')
    previous = {sig:signal.signal(sig,interrupted) for sig in (signal.SIGINT,signal.SIGTERM)}
    try:
        with (output/'queue.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            accepted = read(args.acceptance)
            validate_acceptance(accepted)
            helpers = load_owned_helpers()
            record([output], '启动已授权两任务5000权重持久测评队列；等待既有训练/本机评测，不抢GPU不重启训练')
            return execute_queue(output,accepted,helpers)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig,handler)


if __name__ == '__main__':
    raise SystemExit(main())
