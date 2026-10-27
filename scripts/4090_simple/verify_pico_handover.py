"""Read-only CPU admission of a UUID-lineage PICO pack and its real DINO cache.

This is a separate readiness receipt. PACK_READY, inherited statistics, sealed
human evidence and cache metadata are never promoted or rewritten here.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import uuid

import numpy as np

import verify_handover_supplement as shared

CONTRACT = 'cvpr-pico-handover-production-verification-v1'
PACK_CONTRACT = 'cvpr-pico-handover-lerobot-pack-v1'
HANDOVER_TASK_ID = 'simple/G1WholebodyHandoverTeleop-v0'
MOBILE_TASK_ID = 'simple/G1WholebodyLocomotionPickBetweenTablesTeleop-v0'
MOBILE_CONTRACT = 'cvpr-pico-mobile-pp-production-verification-v1'
MOBILE_PACK_CONTRACT = 'cvpr-pico-mobile-pp-lerobot-pack-v1'
READY_NAME = 'PICO_DATASET_READY.json'
SOURCE_FILES = ('source_initialization_receipt.json','source_run_config.json',
                'source_dataset_statistics.json','source_state_info.json')
require, sha, read, jsonl = shared.require, shared.sha, shared.read, shared.jsonl


def production_protocol(task_id):
    from pack_pico_handover import task_protocol
    protocol = task_protocol(task_id)
    kind = 'mobile-pp' if protocol['mobile'] else 'handover'
    return {**protocol,
        'production_contract':MOBILE_CONTRACT if protocol['mobile'] else CONTRACT,
        'relocation_contract':f'cvpr-pico-{kind}-dino-host-relocation-v1',
        'transfer_contract':f'cvpr-pico-{kind}-a100-byte-transfer-v1',
        'byte_contract':f'cvpr-pico-{kind}-a100-byte-verification-v1'}


def owned(root, relative):
    root = Path(root).resolve(strict=True)
    require(isinstance(relative,str),'Payload path must be a string')
    name = Path(relative)
    require(name.parts and not name.is_absolute()
            and '..' not in name.parts, 'Payload path must remain inside its root')
    path = root
    for part in name.parts:
        path /= part
        require(not path.is_symlink(), 'Payload symlinks are not accepted: '+relative)
    require(path.is_file() and path.resolve().is_relative_to(root), 'Missing regular payload: '+relative)
    return path


def identity(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'Regular payload required: '+str(path))
    before = path.stat()
    value = {'sha256':sha(path),'size_bytes':before.st_size,'mtime_ns':before.st_mtime_ns}
    after = path.stat()
    require((before.st_size,before.st_mtime_ns,before.st_ino)==
            (after.st_size,after.st_mtime_ns,after.st_ino),'Input changed while hashing: '+str(path))
    return value


def tree_inventory(root):
    root = Path(root)
    require(root.is_dir() and not root.is_symlink(),'Independent regular directory required')
    values = {}
    for path in sorted(root.rglob('*')):
        require(not path.is_symlink(),'Dataset/cache must not contain aliases: '+str(path))
        if path.is_file():
            values[str(path.relative_to(root))] = identity(path)
    require(values,'Dataset/cache is empty')
    return values


def shared_original_evidence(pack, payload, admission, archived, original, expected, role):
    from pack_pico_handover import valid_sha
    require(isinstance(archived,dict) and isinstance(original,str) and valid_sha(expected)
            and isinstance(archived.get('path'),str)
            and archived.get('original_path')==original
            and payload.get(archived['path'])==archived.get('sha256')==expected
            and admission.get('bindings',{}).get(original)==expected,
            'Mobile shared original '+role+' is not archived and SHA-bound')
    owned(pack,archived['path'])


def inspect_pack(pack, *, expected_episodes=None):
    pack = Path(pack).resolve(strict=True)
    ready = read(owned(pack,'PACK_READY.json'))
    from pack_pico_handover import artifact_protocol
    protocol = artifact_protocol(ready,'pack')
    if protocol['mobile']:
        require(ready.get('selection_policy')=='native_only','Mobile packed selection policy differs')
    require(ready.get('train_payload_packed') is True
            and ready.get('final_ready') is False and ready.get('normalization_refitted') is False,
            'Require an unpromoted independent PICO human pack with inherited statistics')
    payload = ready.get('payload_sha256')
    require(isinstance(payload,dict) and payload,'Missing packed SHA inventory')
    actual = tree_inventory(pack)
    require(set(actual)==set(payload)|{'PACK_READY.json'},'Packed payload coverage differs')
    for relative,digest in payload.items():
        require(identity(owned(pack,relative))['sha256']==digest,'Packed bytes changed: '+relative)
    require(all(name in payload for name in SOURCE_FILES),'Missing inherited checkpoint contract files')
    episodes = jsonl(owned(pack,'train/meta/episodes.jsonl'))
    entries = ready.get('episodes',[])
    require(episodes and len(episodes)==len(entries)==ready.get('train_episodes'), 'PICO episode count differs')
    require(expected_episodes is None or len(episodes)==expected_episodes,'Unexpected PICO episode count')
    require([e.get('episode_index') for e in episodes]==list(range(len(episodes)))
            and [e.get('episode_index') for e in entries]==list(range(len(entries))), 'Episode indices must be dense and unique')
    seen = set()
    for episode,entry in zip(episodes,entries):
        human_uuid = episode.get('human_demo_uuid')
        require(isinstance(human_uuid,str) and str(uuid.UUID(human_uuid))==human_uuid
                and human_uuid not in seen,'Missing/duplicate canonical human demonstration UUID')
        seen.add(human_uuid)
        source = episode.get('source_identity',{})
        if protocol['mobile']:
            require(episode.get('task_id')==entry.get('task_id')==MOBILE_TASK_ID,
                    'Mobile episode task identity differs from its pack')
        require(episode.get('source_kind')=='live_pico_human' and entry.get('source_kind')=='live_pico_human'
                and episode.get('split')=='train' and episode.get('split_restriction')=='train_only'
                and episode.get('lineage_complete') is True
                and source.get('source_kind')=='live_pico_human'
                and source.get('source_namespace')=='live_pico_human'
                and source.get('human_demo_uuid')==human_uuid
                and entry.get('source_identity')==source and entry.get('human_demo_uuid')==human_uuid
                and episode.get('candidate_id')==entry.get('candidate_id')==human_uuid,
                'Only complete independent train-only human UUID lineage is accepted')
        require(episode.get('eligible_current_rows')==entry.get('eligible_current_rows')
                and episode.get('length')==entry.get('frames',entry.get('length'))
                and episode['length']>=60,'Packed lengths/window identities differ')
        # This integer exists only for the old reader; it is not an official
        # demonstration ancestor. Identity remains the explicit human namespace.
        require(episode.get('source_episode_index')==1_000_000+episode['episode_index'],
                'Human compatibility integer differs from its UUID namespace')
        admission_path = entry.get('admission_path')
        admission = read(owned(pack,admission_path))
        require(payload.get(admission_path)==entry.get('admission_sha256')
                and admission.get('contract')==protocol['admission_contract']
                and admission.get('status')=='independent_live_human_episode_audit_passed'
                and admission.get('human_demo_uuid')==human_uuid
                and admission.get('eligible_for_packing') is True
                and admission.get('accepted_new_dataset') is False
                and admission.get('ckpt_feature_pack_ready') is False and admission.get('errors')==[]
                and admission.get('checks') and all(check.get('passed') is True for check in admission['checks'].values()),
                'Independent physical/RGB human admission is not passed and bound')
        evidence = entry.get('evidence_files',{})
        metadata_path = evidence.get('metadata.json')
        metadata = read(owned(pack,metadata_path))
        require(metadata_path in payload and metadata.get('source_kind')=='live_pico_human'
                and metadata.get('human_demo_uuid')==human_uuid
                and metadata.get('replay_source_sha256') is None,'Human source metadata differs')
        if protocol['mobile']:
            require(admission.get('task_id')==metadata.get('task_id')==MOBILE_TASK_ID,
                    'Mobile admission/source task identity differs')
            require(admission.get('selection_policy')=='native_only' and admission.get('technical_blocked') is False,
                    'Mobile admission is not a complete native-only selection')
            required_evidence = {'metadata.json','episode.json','commands.jsonl',
                'raw_quality_samples.jsonl','applied_force_samples.jsonl'}
            require(required_evidence<=set(evidence),'Mobile original evidence coverage differs')
            trace = admission.get('mobile_reference_audit',{}).get('control_trace',{})
            archived = entry.get('mobile_control_trace_file',{})
            require(isinstance(trace,dict),'Mobile original control trace metadata is missing')
            shared_original_evidence(pack,payload,admission,archived,trace.get('path'),trace.get('sha256'),'control trace')
            historical = admission.get('mobile_reference_audit',{}).get('historical_sources',{})
            archived_history = entry.get('mobile_historical_source_files',{})
            require(isinstance(historical,dict) and isinstance(archived_history,dict),
                    'Mobile historical source metadata is missing')
            shared_original_evidence(pack,payload,admission,archived_history.get('manifest'),
                historical.get('manifest_path'),historical.get('manifest_sha256'),'source manifest')
            expected_sources,archived_sources = historical.get('files'),archived_history.get('files')
            require(isinstance(expected_sources,list) and expected_sources
                    and isinstance(archived_sources,list) and len(expected_sources)==len(archived_sources),
                    'Mobile historical implementation coverage differs')
            for source,copy in zip(expected_sources,archived_sources):
                require(isinstance(source,dict) and isinstance(copy,dict)
                        and isinstance(source.get('filename'),str) and copy.get('filename')==source['filename'],
                        'Mobile historical implementation role differs')
                shared_original_evidence(pack,payload,admission,copy,source.get('path'),source.get('sha256'),'capture implementation')
        required = set(protocol['base_checks'])
        if metadata.get('render_mode')=='mujoco_deferred':required.add('original_full_state_to_deferred_rtx_bridge')
        if not protocol['mobile'] and metadata.get('quality_geometry_mode')=='deferred_geometry_with_original_solved_support':
            required.add('original_solved_support_and_deferred_geometry')
        if metadata.get('capture_start_protocol') is not None:required.add('independent_frozen_first_capture_boundary')
        require(required<=set(admission['checks']),'Human independent admission is missing mandatory checks')
        for relative in evidence.values():
            require(relative in payload,'Unbound original human evidence')
        source_directory = Path(entry['original_episode_directory'])
        for original_relative,copied_relative in evidence.items():
            if original_relative!='collection_record.json':
                require(admission.get('bindings',{}).get(str(source_directory/original_relative))==payload[copied_relative],
                        'Copied human evidence is not the independently admitted original: '+original_relative)
        authorities = entry.get('auditor_source_files',{})
        require({Path(name).name for name in authorities}==protocol['auditor_files'],'Archived auditor authority is incomplete')
        for original,value in authorities.items():
            require(payload.get(value['path'])==value['sha256']==admission.get('auditor_source_sha256',{}).get(original),
                    'Archived auditor implementation SHA differs')
        video = entry.get('video',{})
        require(entry.get('parquet_path') in payload and video.get('path') in payload
                and payload[entry['parquet_path']]==entry.get('parquet_sha256')
                and payload[video['path']]==video.get('sha256')
                and video.get('frames')==episode['length'] and video.get('fps')==30
                and video.get('pixels_exact') is True,
                'Parquet/video SHA binding differs')
    require(ready.get('total_frames')==sum(e['length'] for e in episodes)
            and ready.get('train_windows')==sum(len(e['eligible_current_rows']) for e in episodes),
            'Total PICO frame/window count differs')
    selection_binding = ready.get('selection',{})
    selection_path = selection_binding.get('path')
    selection = read(owned(pack,selection_path))
    require(payload.get(selection_path)==selection_binding.get('sha256')
            and selection.get('contract')==protocol['selection_contract']
            and [(row.get('human_demo_uuid'),row.get('admission_sha256')) for row in selection.get('episodes',[])]
                ==[(entry['human_demo_uuid'],entry['admission_sha256']) for entry in entries],
            'Frozen selected human UUID/admission identities differ from the packaged set')
    if protocol['mobile']:
        require(selection.get('task_id')==MOBILE_TASK_ID,'Mobile selected task identity differs')
    require(not (pack/'val').exists() and not (pack/'test').exists(),'Human release is train-only')
    return pack,ready,episodes,{e['episode_index']:e for e in entries}


def validate_labels(raw, length):
    valid = np.asarray(raw['action_valid'])
    hands,tokens,state,mask = (np.asarray(raw[key]) for key in
        ('action','action.body_token_v1_1','observation.state','action_mask'))
    require(valid.dtype==np.bool_ and valid.shape==(length,) and not valid[:30].any() and valid[30:].all(),
            'Warmup validity must be explicit and dense')
    require(hands.shape==(length,14) and tokens.shape==(length,64) and state.shape==(length,43)
            and mask.shape==(length,80),'Physical action/state widths differ')
    require(np.isnan(hands[:30]).all() and np.isnan(tokens[:30]).all() and not mask[:30].any()
            and np.isfinite(hands[30:]).all() and np.isfinite(tokens[30:]).all() and np.isfinite(state).all()
            and (mask[30:,:78]==1).all() and (mask[30:,78:]==0).all(),
            'Warmup NaNs, actual labels or 78+2 masks differ')
    return hands,tokens,state


def normalized(values, low, high, *, state=False):
    values = np.asarray(values,dtype=np.float32)
    low,high = np.asarray(low,dtype=np.float32),np.array(high,dtype=np.float32,copy=True)
    require(low.shape==high.shape==values.shape[-1:] and np.isfinite(low).all() and np.isfinite(high).all(),
            'Inherited normalization widths/values differ')
    constant = np.abs(high-low)<1e-4*(np.abs(high)+np.abs(low)+1e-8)
    high[constant] = 1.
    with np.errstate(divide='ignore',invalid='ignore'):
        result = np.where(constant,0 if state else values,(values-low)/(high-low)*2-1)
    return np.clip(result,-1,1).astype(np.float32)


def validate_numerical_sample(sample, raw, row, field):
    hands,tokens,states = (np.asarray(raw[key],dtype=np.float32) for key in
        ('action','action.body_token_v1_1','observation.state'))
    expected_actions = np.pad(np.concatenate((tokens[row:row+30],hands[row:row+30]),axis=-1),((0,0),(0,2)))
    expected_states = np.pad(states[row:row+1],((0,0),(0,2)))
    require(np.array_equal(np.asarray(sample['raw_actions']),expected_actions),'Actual raw action channels/time differ')
    require(np.array_equal(np.asarray(sample['actions']),normalized(expected_actions,field.action_min,field.action_max)),
            'Actual inherited action normalization differs')
    if field.normalize_state:
        expected_states = normalized(expected_states,field.state_min,field.state_max,state=True)
    require(np.array_equal(np.asarray(sample['states']),expected_states),'Actual no_aug state/time/normalization differs')


def verify(pack, dino_root, reference_dino_root, run_config, output, *, expected_episodes=None, a100=False):
    shared.cpu_environment(); shared.runtime_paths()
    import pyarrow.parquet as pq
    import torch
    from psi.data.lerobot.dino_cache import load_cache_manifest,load_episode_cache
    from psi.trainers.qwen3vl_mixin import PaddedCollatorForTogether
    from curate_handover_dino_cache import full_encoder_contract
    require(not torch.cuda.is_initialized(),'CPU verifier must not initialize CUDA')
    pack,ready,episodes,entries = inspect_pack(pack,expected_episodes=expected_episodes)
    protocol = production_protocol(ready.get('task_id',HANDOVER_TASK_ID))
    release = pack.parent
    dino_root,reference_dino_root = Path(dino_root).resolve(strict=True),Path(reference_dino_root).resolve(strict=True)
    require(pack.name=='packed' and dino_root.is_relative_to(release/'dino') and dino_root!=reference_dino_root,
            'Use independent release/packed and release/dino trees')
    output = Path(output).absolute()
    relocation_binding = None
    if a100:
        require(output.name=='A100_PICO_DATASET_READY.json','A100 CPU readiness uses an explicit separate marker')
        relocation_path = owned(dino_root,'REMOTE_DINO_RELOCATION.json')
        relocation = read(relocation_path)
        require(relocation.get('contract')==protocol['relocation_contract']
                and relocation.get('status')=='passed'
                and relocation.get('dataset_root')==str(pack/'train')
                and relocation.get('pack_ready_sha256')==sha(pack/'PACK_READY.json')
                and relocation.get('manifest_sha256')==sha(dino_root/'manifest.json')
                and relocation.get('data_and_features_byte_identical') is True,
                'A100 production readiness requires the completed explicit host relocation')
        if protocol['mobile']:
            require(relocation.get('task_id')==MOBILE_TASK_ID,'Mobile relocated task identity differs')
        incoming_transfer_binding(release,include_cache=False)
        relocation_binding = {'path':str(relocation_path),'sha256':sha(relocation_path),
            'byte_verification_sha256':sha(release/'A100_BYTE_VERIFICATION.json')}
    require(not output.exists() and not output.is_symlink() and not output.resolve().is_relative_to(shared.REPO)
            and not any(output.resolve().is_relative_to(p) for p in (pack,dino_root,reference_dino_root)),
            'Preserve source/cache/repository files; choose a fresh independent readiness receipt')
    inputs = {'packed/'+name:value for name,value in tree_inventory(pack).items()}
    inputs.update({str(path.relative_to(release)):identity(path) for path in (release/'dino').rglob('*') if path.is_file()})
    # Reject aliases anywhere in dino, including empty aliased directories.
    tree_inventory(release/'dino')
    external = {str(Path(run_config).resolve(strict=True)):identity(run_config),
                str(reference_dino_root/'manifest.json'):identity(reference_dino_root/'manifest.json')}
    manifest = load_cache_manifest(dino_root,pack/'train',expected_profile='256x320')
    reference = load_cache_manifest(reference_dino_root,expected_profile='256x320')
    require(manifest['storage_dtype']=='float32' and full_encoder_contract(manifest)==full_encoder_contract(reference),
            'Real DINO must inherit actual frozen encoder/clean256x320/precision contract')
    require(set(manifest['episode_indices'])==set(entries),'DINO episode coverage differs')
    raw_tables,cache_reports = {},[]
    for episode in episodes:
        index,length = episode['episode_index'],episode['length']
        cached = load_episode_cache(dino_root,index,manifest=manifest,dataset_root=pack/'train',verify_checksums=True)
        try:
            require(cached.features.shape==(length,5,1024) and cached.features.dtype==np.float32 and cached.valid.all()
                    and np.array_equal(cached.frame_index,np.arange(length))
                    and np.array_equal(cached.source_frame_index,np.arange(length))
                    and np.allclose(cached.video_timestamp,np.arange(length)/30,rtol=0,atol=1e-7)
                    and np.allclose(cached.timestamp,cached.video_timestamp,rtol=0,atol=.25/30),
                    'Actual native30Hz DINO frames/clock/dtype differ')
            for start in range(0,length,128):
                require(np.isfinite(cached.features[start:start+128]).all(),'Nonfinite actual DINO features')
        finally:
            cached.features._mmap.close()
        raw = pq.read_table(owned(pack,entries[index]['parquet_path'])).to_pydict()
        validate_labels(raw,length)
        require(np.array_equal(raw['frame_index'],np.arange(length))
                and np.array_equal(raw['source_frame_index'],(np.arange(length)*50+29)//30)
                and np.all(np.asarray(raw['source_episode_index'])==episode['source_episode_index']),
                'Native row versus physical50Hz source-frame identities differ')
        raw_tables[index] = raw
        cache_reports.append({'episode_index':index,'human_demo_uuid':episode['human_demo_uuid'],
            'feature_shape':[length,5,1024],'storage_dtype':'float32','every_feature_finite':True})
    subset,processor,cfg,norm = shared.make_production_subset(run_config,pack,dino_root,no_aug=True)
    require(len(subset)==ready['train_windows'],'Actual production complete-future window count differs')
    repack = cfg.data.transform.repack
    require(repack.action_keys==['action.body_token_v1_1','action[:14]']
            and repack.state_keys==['observation.state'] and repack.num_past_frames==0
            and repack.action_chunk_size==30 and repack.pad_action_dim==80 and repack.pad_state_dim==45,
            'Require inherited SONICv1.1 body64+hands14 action order/state43 layout')
    history = subset.dataset.raw_dataset.motion_history
    collator = PaddedCollatorForTogether(model_max_length=processor.tokenizer.model_max_length,
        pad_token_id=processor.tokenizer.pad_token_id,padding_side='right')
    queries = []
    for episode in episodes:
        local = [int(i) for i in subset.indices if int(history.episode_ids[i])==episode['episode_index']]
        require(local,'Every human episode must retain actual production windows')
        for index in sorted({local[0],local[len(local)//2],local[-1]}):
            sample = subset.dataset[index]
            shapes = shared.validate_sample(sample)
            row = int(history.history_indices[index][-1])
            validate_numerical_sample(sample,raw_tables[episode['episode_index']],row,cfg.data.transform.field)
            require(np.array_equal(np.asarray(sample['motion_img_embs']),history.get_history(index)),
                    'Actual clean six-frame DINO history changed')
            batch = collator([sample]); shared.validate_sample(batch,batched=True)
            require(torch.isfinite(batch['pixel_values']).all().item(),'Nonfinite actual processor RGB pixels')
            queries.append({'episode_index':episode['episode_index'],'filtered_index':index,'current_row':row,
                'history_rows':history.history_indices[index].tolist(),'shapes':shapes,'numerical_normalization_passed':True})
    for relative,wanted in inputs.items():
        require(identity(owned(release,relative))==wanted,'Read-only verification input changed: '+relative)
    for path,wanted in external.items():
        require(identity(path)==wanted,'Reference/config changed during CPU verification')
    require(not torch.cuda.is_initialized(),'CPU production reader unexpectedly initialized CUDA')
    report = {'contract':protocol['production_contract'],'task_id':protocol['task_id'],
        **({'selection_policy':'native_only'} if protocol['mobile'] else {}),
        'status':'passed','created_utc':datetime.now(timezone.utc).isoformat(),
        'release_root':str(release),'pack_root':str(pack),'dino_root':str(dino_root),
        'train_episodes':len(episodes),'total_frames':ready['total_frames'],'train_windows':len(subset),
        'pack_ready_sha256':sha(pack/'PACK_READY.json'),'cache_manifest_sha256':sha(dino_root/'manifest.json'),
        'payload_files':inputs,'external_reference_files':external,'normalization':norm,'normalization_refitted':False,
        'cache_episodes':cache_reports,'production_queries':queries,'source_kind':'live_pico_human','split_restriction':'train_only',
        'production_ready_for_bound_pack_and_cache':True,'remote_production_ready':bool(a100),
        'DINO_relocation_pending':False if a100 else True,'remote_relocation_binding':relocation_binding,
        'GPU_used':False,'CUDA_initialized':False,'policy_constructed':False,'training_started':False,
        'PACK_READY_modified':False,'final_ready_automatically_promoted':False,'verifier_sha256':sha(__file__)}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:
        stream.write(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    return report


def incoming_transfer_binding(release, *, include_cache=True):
    """Recheck immutable copied inputs; never trust a byte report in isolation."""
    release = Path(release).resolve(strict=True)
    manifest_path = owned(release,'A100_TRANSFER_MANIFEST.json')
    report = read(owned(release,'A100_BYTE_VERIFICATION.json'))
    transfer = read(manifest_path)
    from pack_pico_handover import artifact_protocol
    pack = read(owned(release,'packed/PACK_READY.json'))
    pack_protocol = artifact_protocol(pack,'pack')
    protocol = production_protocol(pack_protocol['task_id'])
    require(transfer.get('contract')==protocol['transfer_contract']
            and transfer.get('remote_root')==str(release)
            and report.get('contract')==protocol['byte_contract']
            and report.get('status')=='all_copied_bytes_and_identities_verified'
            and report.get('remote_root')==str(release)
            and report.get('inventory_sha256')==sha(manifest_path)
            and report.get('remote_production_ready') is False,
            'Explicit same-release A100 byte verification is required')
    local_ready = read(owned(release,READY_NAME))
    require(local_ready.get('contract')==protocol['production_contract'] and local_ready.get('status')=='passed'
            and local_ready.get('production_ready_for_bound_pack_and_cache') is True
            and local_ready.get('source_kind')=='live_pico_human'
            and local_ready.get('split_restriction')=='train_only'
            and transfer.get('local_ready_sha256')==sha(release/READY_NAME)
            and {relative:wanted for relative,wanted in transfer['files'].items()
                 if relative.startswith(('packed/','dino/'))}==local_ready.get('payload_files'),
            'Incoming byte verification must bind the original passed local production receipt')
    if protocol['mobile']:
        require(all(value.get('task_id')==MOBILE_TASK_ID for value in (transfer,report,local_ready)),
                'Mobile transfer and CPU receipt task identities differ')
    for relative,wanted in transfer['files'].items():
        if not include_cache and relative.startswith('dino/'):
            continue
        require(identity(owned(release,relative))==wanted,'Copied immutable input changed: '+relative)
    return transfer


def relocate(pack, dino_root):
    """Rebind only DINO metadata after a SHA/mtime-preserving byte transfer.

    The entire incoming cache remains at dino.before-remote-relocation. New
    metadata is staged separately; feature/sidecar files are ordinary copies
    with unchanged bytes and nanosecond identity. No model is constructed.
    A production-ready marker still requires verify(..., a100=True).
    """
    shared.cpu_environment();shared.runtime_paths()
    from curate_handover_dino_cache import extractor
    pack,ready,episodes,entries = inspect_pack(pack)
    protocol = production_protocol(ready.get('task_id',HANDOVER_TASK_ID))
    release = pack.parent;dino_root = Path(dino_root).resolve(strict=True)
    require(pack.name=='packed' and dino_root==release/'dino','Relocate only the independent release/dino tree')
    incoming_transfer_binding(release)
    ex = extractor();cache = ex.cache
    original_manifest = read(owned(dino_root,'manifest.json'))
    require(original_manifest.get('state')=='complete'
            and original_manifest.get('contract')==cache.CONTRACT
            and original_manifest.get('signature')==cache.manifest_signature(original_manifest),
            'Incoming cache manifest is not complete and signed')
    source,items = ex.plan_source(pack/'train',video_key=original_manifest['source']['video_key'])
    before,after = deepcopy(original_manifest['source']),deepcopy(source)
    before.pop('dataset_root',None);after.pop('dataset_root',None)
    require(before==after,'Only the host dataset root may change; restore all original byte/mtime identities first')
    originals = {}
    for item in items:
        directory = dino_root/'episodes'/f"{item['episode_index']:08d}"
        receipt = read(owned(directory,'receipt.json'))
        require(receipt.get('source')==item and receipt.get('manifest_signature')==original_manifest['signature']
                and receipt.get('signature')==cache.digest({'manifest_signature':original_manifest['signature'],'source':item}),
                'Incoming DINO episode source identity differs')
        for name,wanted in receipt['outputs'].items():
            require(identity(owned(directory,name))==wanted,'Incoming real feature/sidecar changed')
        originals[item['episode_index']] = receipt
    backup = dino_root.with_name(dino_root.name+'.before-remote-relocation')
    require(not backup.exists() and not backup.is_symlink(),'Preserve previous host relocation evidence')
    staging = dino_root.with_name('.'+dino_root.name+'.relocating-'+uuid.uuid4().hex)
    shutil.copytree(dino_root,staging,copy_function=shutil.copy2)
    try:
        archive = staging/'relocation_input_originals';archive.mkdir()
        shutil.copy2(dino_root/'manifest.json',archive/'manifest.json')
        manifest = deepcopy(original_manifest)
        manifest['source'] = source
        manifest['signature'] = cache.manifest_signature(manifest)
        cache.atomic_json(staging/'manifest.json',manifest)
        for item in items:
            index = item['episode_index'];directory = staging/'episodes'/f'{index:08d}'
            shutil.copy2(dino_root/'episodes'/f'{index:08d}'/'receipt.json',archive/f'episode_{index:08d}.json')
            receipt = deepcopy(originals[index])
            receipt.update(source=item,manifest_signature=manifest['signature'],
                signature=cache.digest({'manifest_signature':manifest['signature'],'source':item}))
            cache.atomic_json(directory/'receipt.json',receipt)
        # These are the same real CPU cache readers used by production.
        cache.load_cache_manifest(staging,pack/'train',expected_profile='256x320')
        for item in items:
            cached = cache.load_episode_cache(staging,item['episode_index'],manifest=manifest,
                dataset_root=pack/'train',verify_checksums=True)
            cached.features._mmap.close()
        report = {'contract':protocol['relocation_contract'],'task_id':protocol['task_id'],'status':'passed',
            'created_utc':datetime.now(timezone.utc).isoformat(),
            'old_dataset_root':original_manifest['source']['dataset_root'],'dataset_root':str(pack/'train'),
            'pack_ready_sha256':sha(pack/'PACK_READY.json'),'original_manifest_sha256':sha(dino_root/'manifest.json'),
            'manifest_sha256':sha(staging/'manifest.json'),'incoming_cache_backup':str(backup),
            'data_and_features_byte_identical':True,'PACK_READY_modified':False,
            'remote_production_ready':False,'CPU_reader_verification_pending':True,'GPU_used':False,'training_started':False}
        cache.atomic_json(staging/'REMOTE_DINO_RELOCATION.json',report)
        dino_root.rename(backup)
        try:staging.rename(dino_root)
        except BaseException:
            backup.rename(dino_root)
            raise
        return report
    except BaseException:
        # Keep a failed private stage for diagnosis; incoming cache is untouched.
        raise


def main():
    if len(sys.argv)>1 and sys.argv[1]=='relocate':
        parser = argparse.ArgumentParser(description='CPU-only explicit DINO host relocation; never training readiness')
        parser.add_argument('relocate')
        parser.add_argument('--pack-root',type=Path,required=True)
        parser.add_argument('--dino-root',type=Path,required=True)
        args = parser.parse_args()
        print(json.dumps(relocate(args.pack_root,args.dino_root),indent=2));return
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('pack-root','dino-root','reference-dino-root','run-config','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--expected-episodes',type=int)
    parser.add_argument('--a100',action='store_true',help='After explicit byte transfer/relocation, emit the separate A100 readiness receipt')
    args = parser.parse_args()
    report = verify(args.pack_root,args.dino_root,args.reference_dino_root,args.run_config,args.output,
        expected_episodes=args.expected_episodes,a100=args.a100)
    print(json.dumps({key:report[key] for key in ('status','train_episodes','train_windows','GPU_used')}))


if __name__=='__main__':
    main()
