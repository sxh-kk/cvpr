"""Fresh Handover149 Qwen six-image ablation, inheriting its audited trainer.

The only experimental configuration change is vlm_history_frames=6. Original
accepted rows, validation, state migration, fused optimizer, DDP and checkpoint
retention remain the original Handover149 implementation. No automatic restart.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import sys
import types


CONTRACT = 'cvpr-handover149-qwen-history6-fresh-v1'
CODE_CONTRACT = 'cvpr-handover-qwen-history6-code-v1'
BASE_MANIFEST_SHA = 'cbe9bc8f041f0f16bf6388a84816805aed926c111978d36cfc327121fd83b110'
BASE_RUNNER_SHA = '2423f9f8b59754ee1171612eaf7b68ce34a9f6f48bf3cbe8feccc207eb6c037e'
BASE_UPSTREAM_SHA = '17ae95022154638fc5771c19746f1ad65d89c1a3aab6e00f574f594b65840505'
BASE_MIGRATION_SHA = '95bbb9f4414c7a133e03f253c2219ed9683406e9569bd1084961a6c17f6d8e5a'
BASE_CONFIG_SHA = '0513f440cbeba2c3dfca3db272bfa2effe215d6c5dc50870dcd399bacbe04e09'
SOURCE_SHA = '12a71e046e221b83f0bdb0efb50f9ba82442045aca0efabef3ae337f185320b7'
ORIGINAL_CODE = '/gaozt-test1/datasets/SIMPLE_sonic_v1_1/.pipeline/task50_38k_preparation/fused_adamw_validation/code/632968d'
BASE_RELEASE = Path('/gaozt-test1/datasets/SIMPLE_sonic_v1_1/handover_train149_20261026')
BASE_VAL = Path('/gaozt-test1/datasets/SIMPLE_sonic_v1_1/handover_task50_30hz/Handover-sonic11-val')
BASE_VAL_DINO = Path('/gaozt-test1/datasets/SIMPLE_sonic_v1_1/dino/handover_task50_30hz/Handover-sonic11-val')
REPO = Path(__file__).resolve().parents[2]
SEMANTIC_CHANGE = 'data.transform.model.vlm_history_frames'
PATH_FIELDS = ('exp', 'timestamp', 'train.output_dir', 'train.deepspeed_config',
               'data.root_dir', 'data.dino_cache_root', 'wandb.project')


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'Missing regular input: '+str(path))
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    path = Path(path)
    sha(path)
    return json.loads(path.read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_name('.'+path.name+'.partial')
    require(not temporary.exists() and not temporary.is_symlink(), 'Preserve partial output')
    with temporary.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def now():
    return datetime.now(timezone.utc).isoformat()


def get(value, dotted):
    for key in dotted.split('.'):
        value = value[key]
    return value


def put(value, dotted, item):
    parts = dotted.split('.')
    for key in parts[:-1]:
        value = value[key]
    value[parts[-1]] = item


def validate_single_factor(original, actual, *, expected_paths=None):
    """Full recursive equality, with one semantic leaf and explicit namespace paths."""
    expected = deepcopy(original)
    require(get(expected, 'data.transform.model').get('vlm_history_frames',1) == 1,
            'Reference must be the original single-current-frame Handover')
    put(expected, SEMANTIC_CHANGE, 6)
    require(type(get(actual, SEMANTIC_CHANGE)) is int and get(actual, SEMANTIC_CHANGE) == 6,
            'Six real Qwen images must be explicitly configured')
    require(expected_paths is not None and set(expected_paths) == set(PATH_FIELDS),
            'All namespace/path changes must have explicit expected values')
    for name, value in expected_paths.items():
        put(expected,name,value)
    require(actual == expected, 'A setting beyond Qwen history or explicit namespace paths changed')
    require(get(actual,'seed') == 20261025 and get(actual,'model.img_chunk') == 1
            and get(actual,'model.n_cams') == 1 and get(actual,'model.observation_horizon') == 1
            and get(actual,'data.transform.repack.num_past_frames') == 0,
            'Preserve original model/action/state and dataset query semantics')
    return {'status':'passed', 'only_semantic_change':SEMANTIC_CHANGE,
            'baseline_value':1, 'experiment_value':6,
            'namespace_path_changes':expected_paths,'all_other_config_values_exact':True}


def baseline(path):
    path = Path(path).resolve(strict=True)
    require(sha(path/'SOURCE_MANIFEST.json') == BASE_MANIFEST_SHA, 'Original Handover149 evidence manifest changed')
    manifest = read(path/'SOURCE_MANIFEST.json')
    for name, identity in manifest['files'].items():
        relative = Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts,
                'Unsafe baseline metadata path')
        require(sha(path/relative) == identity['sha256'], 'Original baseline evidence changed: '+name)
    require(sha(path/'a100_handover149_bs4.py') == BASE_RUNNER_SHA
            and sha(path/'quality100_bs4_shape_adapter.py') == BASE_UPSTREAM_SHA
            and sha(path/'state_dropout_migration.py') == BASE_MIGRATION_SHA
            and sha(path/'prepared_config.json') == BASE_CONFIG_SHA,
            'Original runner/config/state migration identity differs')
    return path, manifest


def upstream_source(raw, code):
    require(hashlib.sha256(raw).hexdigest() == BASE_UPSTREAM_SHA, 'Original optimizer/trainer adapter changed')
    old = ('CODE=Path("'+ORIGINAL_CODE+'")').encode()
    new = ('CODE=Path('+json.dumps(str(code))+')').encode()
    require(raw.count(old) == 1, 'Expected one original frozen source-root literal')
    derived = raw.replace(old,new,1)
    require(derived.replace(new,old,1) == raw, 'Unexpected optimizer/trainer source modification')
    return derived


def external(plan):
    """Only the dependency root changes in the original, hash-bound adapter."""
    code = Path(plan['frozen_code']).resolve(strict=True)
    locations = [str(code/p) for p in ('scripts/4090_simple','VLMExpert/src','.', 'VLMExpert')]
    sys.path[:] = locations + [p for p in sys.path if p not in locations]
    source = Path(plan['upstream_runner'])
    require(sha(source) == BASE_UPSTREAM_SHA and sha(plan['migration_helper']) == BASE_MIGRATION_SHA,
            'Original audited external training helpers changed')
    raw = upstream_source(source.read_bytes(),code)
    require(hashlib.sha256(raw).hexdigest() == plan['upstream_source_root_derivation_sha256'],
            'Explicit source-root-only adapter derivation differs')
    result = types.ModuleType('qwen_history_handover_original_optimizer')
    result.__file__ = str(source)
    sys.modules[result.__name__] = result
    exec(compile(raw,str(source),'exec'),result.__dict__)
    require(Path(result.CODE).resolve() == code
            and Path(result.shared.__file__).resolve() == code/'scripts/4090_simple/finetune_carrybox.py',
            'Original trainer imported a different prepared data/config implementation')
    return result


def inherited(plan):
    """Execute frozen Handover code, routing its self-launch to this ablation entry."""
    raw = Path(plan['baseline'])/'a100_handover149_bs4.py'
    require(sha(raw) == BASE_RUNNER_SHA, 'Original Handover149 trainer bytes changed')
    result = types.ModuleType('qwen_history_handover_original_runner')
    # Its __file__ is used only for runner SHA and recursive name/train subprocess
    # dispatch. Route those to this reviewed wrapper; original bytes stay intact.
    result.__file__ = str(Path(__file__).resolve())
    sys.modules[result.__name__] = result
    exec(compile(raw.read_bytes(),str(raw),'exec'),result.__dict__)
    result.CONTRACT = CONTRACT
    result.external = external
    original_validate = result.validate_config
    def validate(config, current_plan):
        original_validate(config,current_plan)
        validate_single_factor(read(Path(current_plan['baseline'])/'prepared_config.json'),
            config.model_dump(mode='json'),expected_paths=current_plan['namespace_path_changes'])
    result.validate_config = validate
    return result


def code_acceptance(path, code):
    path = Path(path).resolve(strict=True)
    value = read(path)
    require(value.get('contract') == CODE_CONTRACT and value.get('status') == 'passed'
            and value.get('single_factor') == SEMANTIC_CHANGE
            and value.get('baseline_manifest_sha256') == BASE_MANIFEST_SHA,
            'Independent single-factor source review is required')
    sources = value.get('source_sha256',{})
    require(isinstance(sources,dict) and len(sources) >= 174
            and 'scripts/4090_simple/qwen_history_handover_train.py' in sources,
            'Full frozen source inventory, including launcher, is required')
    for name, expected in sources.items():
        relative = Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts
                and (code/relative).resolve().is_relative_to(code) and sha(code/relative) == expected,
                'Reviewed ablation source changed: '+name)
    return value


def prepare(args):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Prepare with CUDA_VISIBLE_DEVICES empty')
    code = args.code.resolve(strict=True)
    require(code == REPO, 'Execute the launcher from its actual reviewed code tree')
    base, manifest = baseline(args.baseline)
    accepted = code_acceptance(args.code_acceptance,code)
    original = read(base/'prepared_config.json')
    previous = read(base/'preparation.json')
    source = Path(previous['source_checkpoint'])
    require(previous['source_step'] == 38000 and previous['source_checkpoint_sha256'] == SOURCE_SHA
            and {'bytes':source.stat().st_size,'mtime_ns':source.stat().st_mtime_ns} == previous['source_identity'],
            'Original accepted joint38000 source stat changed; no finetuned checkpoint may initialize this run')
    output = args.output.absolute()
    require(not output.exists() and not output.is_symlink() and output.parent.is_dir()
            and not any(output.resolve().is_relative_to(p) or p.is_relative_to(output.resolve())
                for p in (code,base,BASE_RELEASE,BASE_VAL,BASE_VAL_DINO)), 'Use a fresh independent output directory')
    plan = deepcopy(previous)
    timestamp = datetime.now(timezone.utc).strftime('%y%m%d%H%M%S')
    changes = {'exp':'handover149_qwen_history6_joint38000_statekeep_bs4_8gpu', 'timestamp':timestamp,
        'train.output_dir':str(output/'training'),
        'train.deepspeed_config':str(code/'VLMExpert/scripts/deepspeed/zero3.json'),
        'data.root_dir':str(output/'data'),'data.dino_cache_root':str(output/'data/dino'),
        'wandb.project':'cvpr-handover149-qwen-history6'}
    plan.update(contract=CONTRACT,baseline=str(base),baseline_manifest_sha256=BASE_MANIFEST_SHA,
        frozen_code=str(code),upstream_runner=str(base/'quality100_bs4_shape_adapter.py'),
        upstream_runner_sha256=BASE_UPSTREAM_SHA,migration_helper=str(base/'state_dropout_migration.py'),
        migration_helper_sha256=BASE_MIGRATION_SHA,runner_sha256=sha(__file__),timestamp=timestamp,
        namespace_path_changes=changes,code_acceptance=str(args.code_acceptance.resolve()),
        code_acceptance_sha256=sha(args.code_acceptance),source_sha256_authority='existing accepted joint38000 SHA and stable stat; no redundant full rehash',
        only_semantic_change=SEMANTIC_CHANGE,vlm_history_frames=6,history_offsets_s=[-1.,-.8,-.6,-.4,-.2,0.],
        qwen_history_same_timestamps_as_motion=True,baseline_checkpoint_policy_unchanged=True,
        source_checkpoint_used_for_initialization='original_joint38000_not_handover_finetuned',
        fresh_run=True,initial_global_step=0,optimizer_scheduler_rng_restored=False)
    plan['upstream_source_root_derivation_sha256'] = hashlib.sha256(
        upstream_source((base/'quality100_bs4_shape_adapter.py').read_bytes(),code)).hexdigest()
    up = external(plan)
    inherited(plan).validate_remote_ready(BASE_RELEASE)
    output.mkdir(mode=0o700)
    data = output/'data'
    for name, target in {'train':BASE_RELEASE/'packed/train','val':BASE_VAL,
                         'dino/train':BASE_RELEASE/'dino','dino/val':BASE_VAL_DINO}.items():
        require(target.is_dir(), 'Original accepted data/cache missing')
        p=data/name;p.parent.mkdir(parents=True,exist_ok=True);p.symlink_to(target,target_is_directory=True)
    argv = up.shared.rewrite_argv(shlex.split((base/'prepared_argv.txt').read_text()),
        dict(changes,**{SEMANTIC_CHANGE:6}))
    (output/'argv.txt').write_text('\n'.join(argv)+'\n')
    config = up.shared.parse_training_argv(output/'argv.txt')
    difference = validate_single_factor(original,config.model_dump(mode='json'),expected_paths=changes)
    inherited(plan).validate_config(config,plan)
    write(output/'run_config.json',config.model_dump(mode='json'))
    write(output/'single_factor_config_validation.json',difference)
    paths = inherited(plan).metadata_paths(BASE_RELEASE,BASE_VAL,BASE_VAL_DINO)
    paths += [base/name for name in manifest['files']]
    paths += [base/'SOURCE_MANIFEST.json',args.code_acceptance.resolve()]
    plan['bindings_sha256'] = {str(p):sha(p) for p in paths}
    plan['code_sha256'] = {str(code/name):digest for name,digest in accepted['source_sha256'].items()}
    plan.update(config_sha256=sha(output/'run_config.json'),argv_sha256=sha(output/'argv.txt'),
        created_utc=now(),baseline_5000_native_successes=28,baseline_5000_native_denominator=30,
        historical_eval_raw_available=False,physical_task_success='not_measured')
    write(output/'preparation.json',plan)
    (output/'launch.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\nexec '+
        shlex.join([str(args.python),str(Path(__file__).resolve()),'run','--prepared',str(output)])+'\n')
    print(json.dumps(dict(status='prepared_pending_actual_six_image_data_and_gpu_check',prepared=str(output),
        only_semantic_change=SEMANTIC_CHANGE,train_windows=59468,validation_windows=40,formal_gpu_started=False)))


def load(root):
    root = Path(root).resolve(strict=True)
    plan = read(root/'preparation.json')
    baseline(plan['baseline'])
    code_acceptance(plan['code_acceptance'],Path(plan['frozen_code']))
    result = inherited(plan)
    config, checked, upstream = result.load(root)
    require(checked['only_semantic_change'] == SEMANTIC_CHANGE
            and checked['vlm_history_frames'] == 6 and checked['baseline_checkpoint_policy_unchanged'] is True,
            'Wrong explicit Qwen history ablation identity')
    return result,config,checked,upstream


def require_gpu_admission(root):
    root = Path(root)
    plan = read(root/'preparation.json')
    accepted = read(root/'ROOT_TRAINING_ADMISSION.json')
    require(accepted.get('status') == 'passed' and accepted.get('user_authorized') is True
            and accepted.get('prepared_config_sha256') == plan['config_sha256']
            and accepted.get('preparation_sha256') == sha(root/'preparation.json')
            and accepted.get('source_checkpoint_sha256') == SOURCE_SHA
            and accepted.get('fresh_formal_load_from_joint38000') is True,
            'Root formal-training admission missing or changed')
    probe_path = Path(accepted['single_gpu_probe_path'])
    require(sha(probe_path) == accepted['single_gpu_probe_sha256'], 'Independent real GPU probe receipt changed')
    probe = read(probe_path)
    require(probe.get('config_sha256') == plan['config_sha256']
            and probe.get('plan_sha256') == sha(root/'preparation.json')
            and probe.get('code_acceptance_sha256') == plan['code_acceptance_sha256']
            and probe.get('source_checkpoint_sha256') == SOURCE_SHA,
            'GPU probe belongs to another formal configuration/source/code acceptance')
    require(probe.get('status') == 'passed' and probe.get('actual_batch_size') == 4
            and probe.get('qwen_images_per_sample') == 6 and probe.get('motion_history_frames') == 6
            and probe.get('forward_backward_completed_steps',0) >= 2
            and probe.get('full_vlm_motion_gradients_finite') is True
            and probe.get('probe_weights_reused_for_formal_training') is False,
            'Require a real BS4 two-step full Qwen/Motion forward/backward check; never reuse probe updates')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode',required=True)
    p = sub.add_parser('prepare')
    for key in ('baseline','code','code-acceptance','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--python',type=Path,default=Path(sys.executable))
    for name in ('check','name','train','run'):
        sub.add_parser(name).add_argument('--prepared',type=Path,required=True)
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args);return 0
    runtime,_,plan,_ = load(args.prepared)
    if args.mode == 'run':
        require(os.getpriority(os.PRIO_PROCESS,0) == 0 and len(os.sched_getaffinity(0)) >= 128,
                'Formal training must retain full0-127 CPUs and nice0, not inspection limits')
        require_gpu_admission(args.prepared)
        return runtime.run(args.prepared)
    if args.mode == 'train':
        require_gpu_admission(args.prepared)
        return runtime.train(args.prepared)
    if args.mode == 'check':
        require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU check must hide GPUs')
        runtime.check(args.prepared)
        return 0
    runtime.name(args.prepared)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
