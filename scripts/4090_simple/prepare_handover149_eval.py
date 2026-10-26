"""Prepare a strict local deployment from a completed 149-task milestone backup.

This independent contract preserves state-dropout zero, all six dual-source
blocks, inherited 38k normalization and the complete saved VLM/action/Motion.
The old task50 exporter is deliberately unchanged. CPU preparation performs no
policy forward, simulation, feature extraction or GPU initialization.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import shlex
import struct
import sys

from export_bundle import canonical_visual_identity, export_motion, sha256, write_json
from prepare_task_snapshot import remap_config, validate_config

CONTRACT = 'cvpr-handover149-local-deployment-v1'
BACKUP_CONTRACT = 'cvpr-handover149-local-checkpoint-backup-v1'
PREPARATION_CONTRACT = 'cvpr-handover-train149-a100-bs4-fresh-v1'
PACK_CONTRACT = 'cvpr-handover-original45-random14-human90-training-view-v1'
PREPARED_METADATA = ('evidence/prepared_run_config.json','evidence/prepared_argv.txt')
EXPECTED_TRACKER = {'entity':'','group':'posttrain','id':'l5a01d3a',
    'name':'handover_train149_joint38000_statekeep_bs4_8gpu.us.flow1000.cosine.lr1.0e-05.b32.gpus8.261026054254'}
EXPECTED_STEP = 5000
SUPPORTED_STEPS = (4000, 5000, 6000, 7000, 8000, 9000, 10000)
EXPECTED_MODEL_SHA = '57f689768b0c0edf9b4407eb441f701d1b35be9807d76f2397dda4083fb02cfd'
EXPECTED_RUN = ('/gaozt-test1/datasets/SIMPLE_sonic_v1_1/.pipeline/handover149_training_20261026/'
    'bs4_restart_001/formal10000/training/posttrain/'
    'handover_train149_joint38000_statekeep_bs4_8gpu.us.flow1000.cosine.lr1.0e-05.b32.gpus8.261026054254')
RELEASE_METADATA = ('packed/PACK_READY.json','dino/manifest.json',
    'dino/MERGED149_REINDEX.json','TRAINING_DATASET_READY.json')
SOURCE_SHA = '12a71e046e221b83f0bdb0efb50f9ba82442045aca0efabef3ae337f185320b7'
STAT_SHA = '931f2b536b2fd47e8d0964ad966202ccbda852d05ea88074be321987200a6ac4'
METADATA = ('run_config.json', 'argv.txt', 'dataset_statistics.json', 'preparation.json')
REPO = Path(__file__).resolve().parents[2]
CRITICAL_PREFIXES = ('MotionExpert/', 'ImageProcess/', 'VLMExpert/src/psi/models/',
    'VLMExpert/src/psi/deploy/', 'VLMExpert/src/psi/config/', 'VLMExpert/src/psi/utils/')


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def validate_runtime_metadata(source, config, preparation):
    """Prove the exact prepared JSON/argv to persisted runtime serialization.

    Production train.py copies tracker metadata into cfg, dumps Path via repr,
    and groups argv list values onto one line. No policy/data/optimizer change
    is allowed through this bridge.
    """
    source = Path(source)
    paths = [source/name for name in PREPARED_METADATA]
    for path, key in zip(paths, ('config_sha256','argv_sha256')):
        require(path.is_file() and not path.is_symlink() and sha256(path) == preparation[key],
            'Original prepared config/argv evidence differs: '+path.name)
    prepared = read(paths[0])
    expected = deepcopy(prepared)
    require(expected['train'].get('hf_token') == '.hf_token'
        and all(expected['wandb'].get(key) is None for key in EXPECTED_TRACKER),
        'Prepared input is not the exact pre-tracker149 configuration')
    expected['train']['hf_token'] = "PosixPath('.hf_token')"
    expected['wandb'].update(EXPECTED_TRACKER)
    require(config == expected, 'Runtime configuration differs beyond exact Path/tracker serialization')
    original_tokens = shlex.split(paths[1].read_text())
    final_tokens = shlex.split((source/'argv.txt').read_text())
    require(original_tokens and original_tokens == final_tokens,
        'Runtime argv changed semantic tokens beyond list formatting')
    return {'status':'passed','prepared_config_sha256':sha256(paths[0]),
        'prepared_argv_sha256':sha256(paths[1]),'runtime_config_sha256':sha256(source/'run_config.json'),
        'runtime_argv_sha256':sha256(source/'argv.txt'),'all_other_config_values_exact':True,
        'argv_tokens_exact':True,'argv_token_count':len(original_tokens),
        'allowed_runtime_metadata':{'train.hf_token':{'prepared':'.hf_token','runtime':"PosixPath('.hf_token')"},
            **{'wandb.'+key:{'prepared':None,'runtime':value}for key,value in EXPECTED_TRACKER.items()}},
        'implementation_basis':{'VLMExpert/scripts/train.py':'tracker assignments154-157, argv formatting159-177, JSON228',
            'VLMExpert/src/psi/utils/utils.py':'cfg_json_default Path repr'}}


def milestone_model_sha(step, expected_model_sha256=None):
    """Require a separately obtained source SHA for each newly requested milestone."""
    require(type(step) is int and step in SUPPORTED_STEPS, 'Unsupported149 task milestone')
    if step == EXPECTED_STEP:
        require(expected_model_sha256 in (None, EXPECTED_MODEL_SHA),
            'The previously verified task5000 model SHA cannot be overridden')
        return EXPECTED_MODEL_SHA
    require(isinstance(expected_model_sha256,str) and len(expected_model_sha256) == 64
        and all(c in '0123456789abcdef' for c in expected_model_sha256),
        'A separate A100 source --expected-model-sha256 is required for this task milestone')
    require(expected_model_sha256 != EXPECTED_MODEL_SHA,
        'The task5000 model cannot be relabeled as another milestone')
    return expected_model_sha256


def validate_source(source, release, *, verify_weight=True, expected_step=EXPECTED_STEP,
                    expected_model_sha256=None):
    """Bind a downloaded milestone to its actual149/BS4 run and external model SHA.

    This accepts a completed checkpoint save while its A100 training continues;
    it does not require, or assert, completion of the10000-update training run.
    ``verify_weight=False`` is only for rechecking already SHA-verified metadata.
    The public preparation path always verifies the full model bytes.
    """
    model_sha = milestone_model_sha(expected_step,expected_model_sha256)
    source, release = Path(source).resolve(strict=True), Path(release).resolve(strict=True)
    receipt = read(source/'backup_receipt.json')
    step = receipt.get('step')
    require(receipt.get('contract') == BACKUP_CONTRACT and type(step) is int
        and step == expected_step and source.name == f'ckpt_{step}'
        and receipt.get('optimizer_copied') is False,
        f'Require the exact149 BS4 task{expected_step} model-only backup')
    marker = receipt['completed_retention_marker']
    require(marker.get('all_models_preserved') is True
        and type(marker.get('saved_update')) is int and marker['saved_update'] >= step
        and step in marker.get('all_saved_model_steps', []),
        f'Completed checkpoint retention authority must include task{step}')
    identity = receipt['run_identity']
    require(identity.get('run_directory') == EXPECTED_RUN,
        'Require the current149 BS4 restart run, not the failed BS8 or old114 run')
    expected_source = str(Path(EXPECTED_RUN)/'checkpoints'/f'ckpt_{step}'/'model.safetensors')
    hashes = [receipt['model'].get(k) for k in ('source_sha256_before','source_sha256_after','local_sha256')]
    require(hashes[0] == model_sha and len(set(hashes)) == 1
        and receipt['model'].get('source_path') == expected_source,
        'Model source/run identity or source-before/source-after/local SHA differs')
    require(set(receipt['metadata']) == set(METADATA), 'Require all four saved inference/provenance files')
    for name in METADATA:
        path = source/name
        require(path.is_file() and not path.is_symlink() and sha256(path) == receipt['metadata'][name],
            'Saved inference metadata changed: '+name)
    require(identity['run_config_sha256'] == receipt['metadata']['run_config.json']
        and identity['preparation_sha256'] == receipt['metadata']['preparation.json'], 'Backup run binding differs')
    weights = source/'model.safetensors'
    require(weights.is_file() and not weights.is_symlink() and weights.stat().st_size == receipt['model']['bytes'],
        'Local milestone model is incomplete')
    if verify_weight:
        require(sha256(weights) == hashes[0], 'Actual local milestone model SHA differs')
    config, preparation = read(source/'run_config.json'), read(source/'preparation.json')
    require(preparation.get('contract') == PREPARATION_CONTRACT
        and preparation.get('fresh_run') is True and preparation.get('initial_global_step') == 0
        and preparation.get('optimizer_scheduler_rng_restored') is False
        and preparation.get('source_step') == 38000 and preparation.get('source_checkpoint_sha256') == SOURCE_SHA
        and (preparation.get('gpu_count'), preparation.get('per_gpu_batch'), preparation.get('gradient_accumulation_steps'),
             preparation.get('effective_global_batch')) == (8, 4, 1, 32)
        and (preparation.get('train_episodes'), preparation.get('train_windows'), preparation.get('max_optimizer_steps'))
            == (149, 59468, 10000), 'Require the exact fresh38k/149-demo/eight-rank BS4 batch32 provenance')
    validate_runtime_metadata(source,config,preparation)
    model, train = config['model'], config['train']
    require((model.get('action_architecture'), model.get('num_blocks'), model.get('action_dim'), model.get('odim'),
        model.get('action_chunk_size'), model.get('state_drop_prob'), model.get('state_null_token'))
        == ('dual_source_v1', 6, 80, 45, 30, 0., True)
        and model.get('tune_motion') is True and model.get('tune_vlm') is True
        and model.get('rtc') is False and model.get('pretrained_joint_checkpoint') == preparation['source_checkpoint'],
        'Require complete six-block dual source with measured state retained')
    require((train.get('train_batch_size'), train.get('gradient_accumulation_steps'), train.get('max_training_steps'))
        == (4, 1, 10000) and train.get('resume_from_checkpoint') is None,
        'Require fresh149 BS4 optimization, not inherited task progress')
    repack, image, field = (config['data']['transform'][k] for k in ('repack','model','field'))
    require(repack.get('action_keys') == ['action.body_token_v1_1','action[:14]']
        and repack.get('state_keys') == ['observation.state']
        and (repack.get('pad_action_dim'), repack.get('pad_state_dim')) == (80,45)
        and repack.get('action_mask_key') == 'action_mask'
        and (image.get('shared_image_profile'), image.get('motion_image_policy'))
            == ('256x320','vlm_color_clean_motion'), 'Training body-token/state/clean-image contract changed')
    remote_release = Path(preparation['release'])
    bindings = preparation['bindings_sha256']
    for name in RELEASE_METADATA:
        path = release/name
        require(path.is_file() and not path.is_symlink()
            and sha256(path) == bindings[str(remote_release/name)],
            'Local149 training source metadata changed: '+name)
    pack_ready = read(release/'packed/PACK_READY.json')
    require(pack_ready.get('contract') == PACK_CONTRACT
        and (pack_ready.get('train_episodes'),pack_ready.get('total_frames'),pack_ready.get('train_windows'))
            == (149,68259,59468)
        and pack_ready.get('component_counts') == {'original45':45,'new_random14':14,'human90':90}
        and pack_ready.get('normalization_refitted') is False,
        'Require the exact original45+random14+human90 source package')
    checkpoint_contract = pack_ready['checkpoint_contract']
    require(checkpoint_contract.get('source_step') == 38000
        and checkpoint_contract.get('source_checkpoint_sha256') == SOURCE_SHA
        and checkpoint_contract.get('bounds_refitted') is False,
        'Source pack inherited checkpoint identity changed')
    require(sha256(source/'dataset_statistics.json') == STAT_SHA == bindings[field['stat_path']]
        == checkpoint_contract['statistics']['sha256'], 'Inherited38k normalization changed')
    require(set(checkpoint_contract['bounds_sha256']) == {'action_min','action_max','state_min','state_max'},
        'Require all four inherited action/state bound identities')
    for name, expected in checkpoint_contract['bounds_sha256'].items():
        require(digest(field[name]) == expected, 'Inline inherited bounds changed: '+name)
    ready, manifest, reindex = (read(release/name) for name in
        ('TRAINING_DATASET_READY.json','dino/manifest.json','dino/MERGED149_REINDEX.json'))
    require(ready.get('contract') == 'handover-train149-a100-ready-v1'
        and ready.get('status') == 'ready_for_training'
        and (ready.get('train_episodes'),ready.get('total_frames'),ready.get('complete_history_future_windows'))
            == (149,68259,59468)
        and ready.get('component_counts') == {'original_train':45,'new_random':14,'live_pico_human':90}
        and ready.get('normalization_refitted') is False and ready.get('validation_included') is False,
        'Require actual149 training readiness, not historical pack pending flags')
    require(manifest.get('state') == 'complete'
        and manifest.get('contract') == 'unifolm-dino-native-frames-v1'
        and manifest.get('episode_indices') == list(range(149))
        and manifest.get('source',{}).get('total_episodes') == 149
        and manifest.get('source',{}).get('total_frames') == 68259
        and manifest.get('source',{}).get('fps') == 30,
        'Require complete149 native30Hz DINO source identity')
    require(reindex.get('contract') == PACK_CONTRACT+'/DINO-reindex'
        and reindex.get('status') == 'passed' and reindex.get('train_episodes') == 149
        and reindex.get('pack_ready_sha256') == sha256(release/'packed/PACK_READY.json')
        and reindex.get('manifest_sha256') == sha256(release/'dino/manifest.json')
        and reindex.get('all_other_encoder_fields_exact') is True,
        '149 DINO reindex source linkage differs')
    return source, config, preparation, receipt


def validate_frozen_model_code(repo, preparation):
    records = {}
    for original, expected in preparation['code_sha256'].items():
        require('/632968d/' in original, 'Unexpected frozen training code identity')
        relative = original.split('/632968d/', 1)[1]
        if not any(relative.startswith(prefix) for prefix in CRITICAL_PREFIXES):
            continue
        path = repo/relative
        require(path.is_file() and sha256(path) == expected, 'Local inference model source differs from frozen632968d: '+relative)
        records[relative] = expected
    require(all(any(n.startswith(prefix) for n in records) for prefix in CRITICAL_PREFIXES),
        'Missing critical frozen inference package proof')
    return {'status':'passed','training_frozen_revision':'632968d','matched_files':len(records),
            'local_repo':str(repo),'sha256':records}


def recorded_encoder_transition(reference, actual):
    """Keep encoder bytes/semantics fixed while declaring recorded runtime versions.

    The historical export helper proves its accepted-reference visual leg. This
    independent deployment records the actual149 runtime leg without modifying
    that validator or claiming bitwise equality across library versions.
    """
    before = canonical_visual_identity({'encoder':deepcopy(reference)})['encoder']
    after = canonical_visual_identity({'encoder':deepcopy(actual)})['encoder']
    versions = {}
    for key in ('torch_version','transformers_version'):
        require(isinstance(before.get(key),str) and isinstance(after.get(key),str),
            'Both recorded DINO execution versions are required: '+key)
        versions[key] = {'accepted_reference':before.pop(key),'handover149_training':after.pop(key)}
    before.pop('checkpoint',None);after.pop('checkpoint',None)
    require(before == after, 'DINO transition changed weight/token/processor/precision/geometry identity')
    return {'status':'passed','scope':'explicit recorded torch/transformers runtime transition from the accepted reference',
        'versions':versions,'all_other_encoder_fields_equal':True,'DINO_file_bytes_equal':True,
        'numerical_bitwise_equivalence_claimed':False,
        'actual_online_contract':'dino_training_contract.json retains the real149 encoder versions'}


def bind_training_motion_image_policy(contract, config):
    """Publish the policy from this run, independently of the older reference."""
    policy = config['data']['transform']['model'].get('motion_image_policy')
    require(policy == 'vlm_color_clean_motion',
        'Require explicit vlm_color_clean_motion in the source run configuration')
    result = deepcopy(contract)
    result['motion_image_policy'] = policy
    return result


def strict_component_check(weights, config, qwen):
    """Strict-assign actual saved values into meta targets; no full CPU model allocation or forward."""
    import torch
    from safetensors import safe_open
    from transformers import AutoConfig, Qwen3VLForConditionalGeneration
    from psi.models.dual_source_action import DualSourceActionTransformer
    from psi.models.checkpoint_aliases import restore_shared_tensor_aliases
    torch.set_num_threads(2)
    m = config['model']
    with safe_open(weights, framework='pt', device='cpu') as saved:
        keys = set(saved.keys())
        require(all(n.startswith(('vlm_model.','action_header.','motion_conditioner.')) for n in keys),
            'Unexpected tensor namespace in the complete model')
        signature = [0x44534143,1,m['action_dim'],m['action_chunk_size'],m['hidden_dim'],m['num_blocks'],
            m['nhead'],m['view_feature_dim'],m['odim'],m['time_dim'],int(m['state_null_token']),
            {None:0,'rms_norm':1,'layer_norm':2}[m['qk_norm']],
            struct.unpack('q',struct.pack('d',float(m['state_drop_prob'])))[0],
            struct.unpack('q',struct.pack('d',float(m['dropout'])))[0],11,768]
        require(saved.get_tensor('action_header._architecture_signature').tolist() == signature,
            'Saved action architecture/state dropout signature differs from actual deployment config')
        qcfg = AutoConfig.from_pretrained(qwen,local_files_only=True)
        qcfg._attn_implementation = 'sdpa'
        with torch.device('meta'):
            vlm = Qwen3VLForConditionalGeneration(qcfg)
            head = DualSourceActionTransformer(action_dim=m['action_dim'],action_pred_horizon=m['action_chunk_size'],
                action_hidden_dim=m['hidden_dim'],action_num_blocks=m['num_blocks'],action_nheads=m['nhead'],
                view_feature_dim=m['view_feature_dim'],odim=m['odim'],time_dim=m['time_dim'],qk_norm=m['qk_norm'],
                state_drop_prob=m['state_drop_prob'],state_null_token=m['state_null_token'],dropout=m['dropout'])
        head._buffers['_architecture_signature'] = torch.tensor(signature,dtype=torch.int64)
        vlm_state = {n.removeprefix('vlm_model.'):saved.get_tensor(n) for n in keys if n.startswith('vlm_model.')}
        original_vlm_keys = set(vlm_state)
        vlm_state = restore_shared_tensor_aliases(vlm,vlm_state)
        vlm.load_state_dict(vlm_state,strict=True,assign=True)
        head_state = {n.removeprefix('action_header.'):saved.get_tensor(n) for n in keys if n.startswith('action_header.')}
        head.load_state_dict(head_state,strict=True,assign=True)
        require(head.state_drop_prob == 0. and len(head.transformer_blocks) == 6, 'Loaded state0/six-block semantics differ')
        counts = {prefix:sum(n.startswith(prefix+'.') for n in keys) for prefix in ('vlm_model','action_header','motion_conditioner')}
        require(all(counts.values()), 'Full VLM/action/Motion saved weights are required')
    require(not torch.cuda.is_initialized(), 'CPU structural check initialized CUDA')
    return {'status':'passed','strict_VLM_load':True,'strict_action_load':True,'strict':True,'assign':True,
        'target_construction':'meta','saved_values':'CPU mmap; no weight conversion','forward_performed':False,
        'source_tensor_counts':counts,'restored_proven_VLM_aliases':sorted(set(vlm_state)-original_vlm_keys),
        'action_signature':signature,'state_drop_prob':0.,'num_blocks':6,'CUDA_initialized':False}


def prepare(source, resources, output, repo, release, *, expected_step=EXPECTED_STEP,
            expected_model_sha256=None):
    os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
    model_sha = milestone_model_sha(expected_step,expected_model_sha256)
    source, config, preparation, receipt = validate_source(source,release,
        expected_step=expected_step,expected_model_sha256=model_sha)
    original_stat = (source/'model.safetensors').stat()
    resources, repo, release = (Path(p).resolve(strict=True) for p in (resources,repo,release))
    output = Path(output).absolute()
    require(not output.exists() and not output.is_symlink() and not output.resolve().is_relative_to(repo)
        and not output.resolve().is_relative_to(source), 'Use a fresh independent deployment outside the repository and source')
    code = validate_frozen_model_code(repo,preparation)
    sys.path[:0] = [str(repo/'VLMExpert/src'),str(repo),str(repo/'pipeline/src'),str(repo/'VLMExpert')]
    base = resources/'joint20k'
    for path in (base/'qwen',base/'motion/joint_motion.ckpt',resources/'dinov2-with-registers-large',resources/'simple_local_usd_v1'):
        require(path.exists(), 'Missing pinned local construction resource: '+str(path))
    manifest = read(release/'dino/manifest.json')
    reference_contract = read(base/'dino_training_contract.json')
    runtime_transition = recorded_encoder_transition(reference_contract['encoder'],manifest['encoder'])
    contract = bind_training_motion_image_policy(reference_contract,config)
    contract['encoder'] = canonical_visual_identity({'encoder':deepcopy(manifest['encoder'])})['encoder']
    contract['online_checkpoint'] = str(resources/'dinov2-with-registers-large')
    contract['source_manifest_sha256'] = sha256(release/'dino/manifest.json')
    contract['source_dataset'] = 'handover149 TRAIN native-frame cache; all features remain unchanged'
    step = receipt['step']
    replacements = {'model.model_name_or_path':str(base/'qwen'),
        'model.motion_checkpoint':str(output/'motion/joint_motion.ckpt'),'model.pretrained_joint_checkpoint':None,
        'data.root_dir':str(resources/'simple_local_usd_v1'),'data.dino_cache_root':None,
        'data.transform.field.stat_path':str(output/'dataset_statistics.json'),
        'train.output_dir':str(output/'unused_training'),'train.resume_from_checkpoint':None,
        'train.deepspeed_config':str(repo/'VLMExpert/scripts/deepspeed/zero3.json')}
    deployed, argv = remap_config(config,(source/'argv.txt').read_text(),replacements)
    output.mkdir(parents=True)
    try:
        provenance = output/'provenance';provenance.mkdir()
        for name in (*METADATA,'backup_receipt.json'):
            shutil.copy2(source/name,provenance/name)
        for name in PREPARED_METADATA:
            copied = provenance/name
            copied.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source/name,copied)
        write_json(provenance/'runtime_metadata_validation.json',
            validate_runtime_metadata(source,config,preparation))
        for name in RELEASE_METADATA:
            copied = provenance/'release_metadata'/name
            copied.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(release/name,copied)
        write_json(provenance/'model_code_validation.json',code)
        write_json(output/'run_config.json',deployed)
        (output/'argv.txt').write_text(argv)
        shutil.copy2(source/'dataset_statistics.json',output/'dataset_statistics.json')
        target = output/'checkpoints'/f'ckpt_{step}'/'model.safetensors';target.parent.mkdir(parents=True)
        os.link(source/'model.safetensors',target)
        write_json(output/'dino_training_contract.json',contract)
        from psi.deploy.dual_source_runtime import load_dino_contract
        load_dino_contract(output/'dino_training_contract.json')
        structure = strict_component_check(target,deployed,base/'qwen')
        write_json(output/'component_load_validation.json',structure)
        motion = export_motion(base/'motion/joint_motion.ckpt',target,output/'motion/joint_motion.ckpt',
            f'38000_then_handover149_task_{step}',reference_contract['encoder'])
        motion['export_helper_visual_check_scope'] = 'Accepted reference metadata bridge; source Motion metadata preserved'
        motion['handover149_actual_encoder_runtime_transition'] = runtime_transition
        motion['actual_encoder_sha256'] = digest(contract['encoder'])
        write_json(output/'motion/export_validation.json',motion)
        validate_config(deployed,output/'argv.txt',repo)
        require(deployed['model']['state_drop_prob'] == 0., 'Deployment must preserve state-retention semantics')
        validate_source(source,release,verify_weight=False,
            expected_step=expected_step,expected_model_sha256=model_sha)
        current_stat = (source/'model.safetensors').stat()
        require((original_stat.st_dev,original_stat.st_ino,original_stat.st_size,original_stat.st_mtime_ns)
            == (current_stat.st_dev,current_stat.st_ino,current_stat.st_size,current_stat.st_mtime_ns),
            'Source milestone model changed during CPU preparation')
        report = {'contract':CONTRACT,'status':'local_handover149_run_prepared','step':step,'task_step':step,
            'source_unifolm_step':38000,'initial_task_step':0,'step_semantics':'handover149_task_optimizer_updates',
            'train_episodes':149,'train_windows':59468,'source_global_batch':32,'state_drop_prob':0.,'num_blocks':6,
            'joint_sha256':receipt['model']['local_sha256'],'run_dir':str(output),'source_snapshot':str(source),
            'requested_task_step':expected_step,'expected_model_sha256':model_sha,
            'model_sha_authority':'previously verified task5000 source' if step == EXPECTED_STEP
                else 'independent A100 source SHA supplied explicitly to preparation',
            'source_backup_receipt_sha256':sha256(source/'backup_receipt.json'),
            'runtime_metadata_validation':validate_runtime_metadata(source,config,preparation),
            'full_saved_components_strict_CPU_load':structure,'strict_motion_load':motion['strict_meta_load'],
            'motion_weights':motion['weights'],'normalization_refitted':False,'statistics_sha256':STAT_SHA,
            'actual_DINO_runtime_transition':runtime_transition,
            'source_model_code_validation_sha256':sha256(provenance/'model_code_validation.json'),
            'deployment_path_changes':replacements,'optimizer_rng_restored':False,'GPU_used':False,
            'policy_forward_performed':False,'gpu_validation':'not_run','physical_task_success':'not_measured',
            'source_metadata_sha256':{p.name:sha256(p) for p in provenance.iterdir() if p.is_file()},
            'verifier_sha256':sha256(Path(__file__)),
            'shared_preparation_helpers_sha256':{name:sha256(Path(__file__).with_name(name))
                for name in ('export_bundle.py','prepare_task_snapshot.py')},
            'source_model_bytes_verified':True,'source_model_identity_unchanged_during_prepare':True,
            'deployment_scope':'Host CPU preparation; namespace aliases must be mapped by the independent runtime builder',
            'training_run_completion_required':False,'checkpoint_save_completed':True}
        write_json(output/'LOCAL_RUN_READY.json',report)
        return report
    except BaseException as error:
        write_json(output/'PREPARATION_FAILED.json',{'contract':CONTRACT,'status':'failed_incomplete','error':repr(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source','resources','output','release'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--repo',type=Path,default=REPO)
    parser.add_argument('--step',type=int,choices=SUPPORTED_STEPS,default=EXPECTED_STEP,
        help='Exact saved task optimizer step to deploy (default: previously verified5000)')
    parser.add_argument('--expected-model-sha256',
        help='Independent SHA256 of the exact A100 source model; required except for pinned5000')
    args = parser.parse_args()
    print(json.dumps(prepare(args.source,args.resources,args.output,args.repo,args.release,
        expected_step=args.step,expected_model_sha256=args.expected_model_sha256),ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
