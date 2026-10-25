"""Byte-preserving union of the admitted new14 and the complete original100.

This has an independent exact114 contract. It does not impose, relax or claim
the stricter footprint/diversity gates used by other collection/curation tools.
No simulation, feature inference or policy training is performed here.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import uuid

import numpy as np

import pack_handover_curated as packed
import curate_handover_dino_cache as dino
import verify_handover_supplement as shared
from verify_handover_curated import validate_labels

CONTRACT = 'cvpr-handover-original100-plus-new14-lerobot-pack-v1'
CACHE_CONTRACT = 'cvpr-handover-merged114-real-DINO-reindex-v1'
VERIFY_CONTRACT = 'cvpr-handover-merged114-production-reader-verification-v1'
require, sha, read, jsonl = shared.require, shared.sha, shared.read, shared.jsonl


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.partial')
    require(not temporary.exists(), 'Preserve an existing interrupted write')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def copied(source, target, expected):
    source, target = Path(source), Path(target)
    require(source.is_file() and not source.is_symlink(), 'Require actual source file')
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    require(sha(target) == expected, 'Copied source bytes differ: '+str(source))


def expected_mapping():
    return [('new14', i, i) for i in range(14)] + [('original100', i, i+14) for i in range(100)]


def validate_mapping(entries):
    require(len(entries) == 114 and [(e['origin']['component'], e['origin']['episode_index'], e['episode_index'])
        for e in entries] == expected_mapping(), 'Require all new14 followed by all original100 in original order')
    require(len({e['candidate_id'] for e in entries}) == 114, 'Repeated actual capture candidate')
    require([e['source_kind'] for e in entries[:14]] == ['new_random']*14
        and [e['source_kind'] for e in entries[14:]] == ['legacy_quality100']*100, 'Wrong component mapping')


def source_inputs(new_pack, old_pack):
    roots = {'new14': Path(new_pack).resolve(strict=True), 'original100': Path(old_pack).resolve(strict=True)}
    require(roots['new14'] != roots['original100'], 'Require two independent source packs')
    ready = {k:read(v/'PACK_READY.json') for k,v in roots.items()}
    require(ready['new14'].get('contract') == packed.CONTRACT and ready['new14'].get('new_episodes') == 14
        and ready['new14'].get('train_episodes') == 28, 'Require the already admitted new14 in the verified28 pack')
    require(ready['original100'].get('contract') == packed.legacy.CONTRACT
        and ready['original100'].get('train_episodes') == 100, 'Require the complete original100 pack')
    require(all(r.get('train_payload_packed') is True and r.get('normalization_refitted') is False for r in ready.values()),
        'Source packs must retain inherited normalization and completed CPU packing')
    metadata = {k:jsonl(v/'train/meta/episodes.jsonl') for k,v in roots.items()}
    for key in roots:
        require(len(metadata[key]) == ready[key]['train_episodes'] == len(ready[key]['episodes'])
            and [e['episode_index'] for e in metadata[key]] == list(range(len(metadata[key]))), 'Source coverage differs')
    require(ready['new14']['checkpoint_contract']['bounds_sha256'] == ready['original100']['checkpoint_contract']['bounds_sha256']
        and ready['new14']['checkpoint_contract']['source_checkpoint_sha256'] == ready['original100']['checkpoint_contract']['source_checkpoint_sha256'],
        'Source checkpoint/normalization differs')
    for name in packed.INHERITED_FILES:
        require(ready['new14']['payload_sha256'][name] == ready['original100']['payload_sha256'][name],
            'Byte-identical inherited configuration/statistics required: '+name)
    require(sha(roots['new14']/'train/meta/tasks.jsonl') == sha(roots['original100']/'train/meta/tasks.jsonl'),
        'Source instruction/task mappings differ')
    return roots, ready, metadata


def pack(new_pack, old_pack, output):
    import pyarrow.parquet as pq
    roots, original, metadata = source_inputs(new_pack, old_pack)
    output = Path(output).absolute()
    require(not output.exists() and not output.is_symlink()
        and not any(output.resolve().is_relative_to(r) or r.is_relative_to(output.resolve()) for r in roots.values()),
        'Use a fresh independent merged114 package')
    staging = output.with_name('.'+output.name+'.merged114-staging-'+uuid.uuid4().hex)
    staging.mkdir(parents=True)
    bindings = {k:{'root':str(r), 'PACK_READY_sha256':sha(r/'PACK_READY.json'),
        'PACK_READY_archived':'provenance/'+k+'_PACK_READY.json',
        'metadata_archived':'provenance/'+k+'_episodes.jsonl'} for k,r in roots.items()}
    for key, root in roots.items():
        copied(root/'PACK_READY.json', staging/bindings[key]['PACK_READY_archived'], bindings[key]['PACK_READY_sha256'])
        copied(root/'train/meta/episodes.jsonl', staging/bindings[key]['metadata_archived'], original[key]['payload_sha256']['train/meta/episodes.jsonl'])
    for name in (*packed.INHERITED_FILES, 'train/meta/tasks.jsonl'):
        copied(roots['original100']/name, staging/name, original['original100']['payload_sha256'][name])
    episodes, entries, offset = [], [], 0
    try:
        for component, source_index, index in expected_mapping():
            root, source_ready = roots[component], original[component]
            ep, entry = deepcopy(metadata[component][source_index]), source_ready['episodes'][source_index]
            require(ep['candidate_id'] == entry['candidate_id'] and ep['length'] == entry['frames'], 'Source candidate/length differs')
            require(ep.get('split') == 'train' and ep.get('split_restriction') == 'train_only'
                and ep.get('lineage_complete') is True and ep['source_episode_index'] not in shared.HELDOUT,
                'Require the originally protected TRAIN lineage')
            if component == 'new14':
                require(ep.get('source_kind') == 'new_random', 'Keep only actual new14, not old selected14')
            else:
                require(ep.get('physics_validation',{}).get('gate') == 'handover-physics-v2'
                    and ep['physics_validation'].get('verified_handover_success') is True,
                    'Legacy original physics validation failed')
            kind = 'new_random' if component == 'new14' else 'legacy_quality100'
            origin = {'component':component,'pack_root':str(root),'episode_index':source_index,
                'parquet_sha256':entry['parquet_sha256'],'video_sha256':entry['video']['sha256']}
            source_relative = f'train/data/chunk-000/episode_{source_index:06d}.parquet'
            source_table = pq.read_table(root/source_relative)
            require(sha(root/source_relative) == source_ready['payload_sha256'][source_relative] == entry['parquet_sha256'],
                'Source executed label bytes differ')
            table = packed.reindex_table(source_table,index,offset,0)
            target = staging/f'train/data/chunk-000/episode_{index:06d}.parquet'
            target.parent.mkdir(parents=True,exist_ok=True);pq.write_table(table,target,compression='zstd')
            packed.legacy.check_parquet_roundtrip(table,pq.read_table(target))
            validate_labels(table.to_pydict(), table.num_rows)
            old_video = f'train/videos/chunk-000/{packed.legacy.VIDEO_KEY}/episode_{source_index:06d}.mp4'
            new_video = f'train/videos/chunk-000/{packed.legacy.VIDEO_KEY}/episode_{index:06d}.mp4'
            copied(root/old_video,staging/new_video,source_ready['payload_sha256'][old_video])
            candidate = packed.safe_candidate(entry['candidate_id'])
            prefix = 'accepted_evidence/'+candidate+'/'
            evidence = {p:s for p,s in source_ready['payload_sha256'].items() if p.startswith(prefix)}
            require(evidence, 'Missing source admission evidence')
            for name, digest in evidence.items():copied(root/name,staging/name,digest)
            admission_relative = prefix+'admission.json'
            require(sha(staging/admission_relative) == entry['admission_sha256'], 'Portable admission changed')
            admission = read(staging/admission_relative)
            require(admission.get('eligible_dataset') is True, 'Original capture was not admitted')
            events_relative = prefix+('run/capture/events.jsonl' if component == 'new14' else 'events.jsonl')
            events = jsonl(staging/events_relative)
            eligible, histories = packed.legacy.eligible_currents(events)
            require(eligible == ep['eligible_current_rows'] == entry['eligible_current_rows']
                and histories == entry['history_rows'] and len(events) == table.num_rows == ep['length'],
                'Actual complete supervision/history windows changed')
            old_episode = deepcopy(ep)
            ep.update(episode_index=index,source_kind=kind,merge_origin=origin)
            entries.append({'episode_index':index,'candidate_id':candidate,'source_kind':kind,
                'source_episode_index':ep['source_episode_index'],'source_dataset_key':ep['source_dataset_key'],
                'source_identity':ep['source_identity'],'origin':origin,'frames':ep['length'],
                'eligible_current_rows':eligible,'history_rows':histories,'video':deepcopy(entry['video']),
                'parquet_sha256':sha(target),'admission_path':admission_relative,'admission_sha256':entry['admission_sha256'],
                'events_path':events_relative,'capture_content_sha256':entry['capture_content_sha256'],
                'unchanged_columns':sorted(set(table.column_names)-packed.REINDEX_COLUMNS),
                'original_metadata_sha256':dino.extractor().cache.digest(old_episode)})
            episodes.append(ep);offset += ep['length']
            print(json.dumps({'stage':'episode_copied_verified','episode_index':index,'component':component}),flush=True)
        validate_mapping(entries)
        meta = staging/'train/meta';info = read(roots['original100']/'train/meta/info.json')
        info.update(total_episodes=114,total_videos=114,total_frames=offset,splits={'train':'0:114'})
        write(meta/'info.json',info)
        (meta/'episodes.jsonl').write_text(''.join(json.dumps(e,ensure_ascii=False,allow_nan=False)+'\n' for e in episodes))
        report = {'contract':CONTRACT,'status':'original100_plus_new14_cpu_pack_complete','created_utc':now(),
            'train_episodes':114,'new_episodes':14,'legacy_episodes':100,'total_frames':offset,
            'train_windows':sum(len(e['eligible_current_rows']) for e in entries),'episodes':entries,
            'source_pack_bindings':bindings,'checkpoint_contract':original['original100']['checkpoint_contract'],
            'inherited_file_sha256':{name:original['original100']['payload_sha256'][name] for name in packed.INHERITED_FILES},
            'train_payload_packed':True,'normalization_refitted':False,'final_ready':False,
            'quality_scope':'All original100 retained under original handover-physics-v2; new14 retain their original stricter admission. No all114 stricter-footprint claim.',
            'original_packages_modified':False,'new_human_demonstration_claim':False,'cross_namespace_unique_ancestor_count_claimed':False,
            'physics_steps_performed':0,'GPU_used':False,'feature_extraction_performed':False,'training_started':False,
            'packer_sha256':sha(__file__)}
        report['payload_sha256'] = {str(p.relative_to(staging)):sha(p) for p in sorted(staging.rglob('*')) if p.is_file()}
        write(staging/'PACK_READY.json',report)
        inspect(staging, verify_payload=False)
        for key,root in roots.items():require(sha(root/'PACK_READY.json')==bindings[key]['PACK_READY_sha256'],'Source READY changed')
        staging.rename(output)
        return report
    except BaseException as error:
        write(staging/'FAILED.json',{'contract':CONTRACT,'state':'failed_incomplete','error':repr(error),'final_ready':False})
        raise


def inspect(pack_root, *, verify_payload=True):
    import pyarrow.parquet as pq
    root = Path(pack_root).resolve(strict=True);ready = read(root/'PACK_READY.json')
    require(ready.get('contract') == CONTRACT and ready.get('train_payload_packed') is True
        and ready.get('normalization_refitted') is False and ready.get('train_episodes') == 114
        and ready.get('new_episodes') == 14 and ready.get('legacy_episodes') == 100, 'Require exact114 independent merge contract')
    validate_mapping(ready['episodes'])
    if verify_payload:
        for name,digest in ready['payload_sha256'].items():require(sha(packed.legacy.owned(root,name))==digest,'Merged payload changed: '+name)
    episodes = jsonl(root/'train/meta/episodes.jsonl');require(len(episodes)==114,'Missing actual episode metadata')
    originals = {}
    for component,binding in ready['source_pack_bindings'].items():
        archived = packed.legacy.owned(root,binding['PACK_READY_archived'])
        require(sha(archived)==binding['PACK_READY_sha256'],'Original READY archival changed')
        source_ready = read(archived)
        metadata_path = packed.legacy.owned(root,binding['metadata_archived'])
        require(sha(metadata_path)==source_ready['payload_sha256']['train/meta/episodes.jsonl'],'Original metadata archival changed')
        originals[component] = (source_ready,jsonl(metadata_path))
    offset,windows = 0,0
    for index,(ep,entry) in enumerate(zip(episodes,ready['episodes'])):
        origin=entry['origin'];source_ready,source_metadata=originals[origin['component']]
        original=source_metadata[origin['episode_index']];source_entry=source_ready['episodes'][origin['episode_index']]
        expected=deepcopy(original);expected.update(episode_index=index,source_kind=entry['source_kind'],merge_origin=origin)
        require(ep==expected and ep['candidate_id']==entry['candidate_id'] and ep['source_identity']==entry['source_identity'],
            'Original episode metadata changed beyond merge fields')
        require(dino.extractor().cache.digest(original)==entry['original_metadata_sha256']
            and source_entry['parquet_sha256']==origin['parquet_sha256'] and source_entry['video']['sha256']==entry['video']['sha256'],
            'Archived source binding differs')
        admission=packed.legacy.owned(root,entry['admission_path'])
        require(sha(admission)==entry['admission_sha256']==source_entry['admission_sha256']
            and read(admission).get('eligible_dataset') is True,'Original admission differs')
        events=jsonl(packed.legacy.owned(root,entry['events_path']));eligible,histories=packed.legacy.eligible_currents(events)
        require(eligible==entry['eligible_current_rows']==ep['eligible_current_rows'] and histories==entry['history_rows'],
            'Merged supervision/history differs')
        table=pq.read_table(root/f'train/data/chunk-000/episode_{index:06d}.parquet')
        require(table.num_rows==ep['length']==entry['frames']==len(events)
            and table['episode_index'].to_pylist()==[index]*len(events)
            and table['index'].to_pylist()==list(range(offset,offset+len(events)))
            and set(table['task_index'].to_pylist())=={0}, 'Merged row/index identities differ')
        expected_table=packed.legacy.arrow_table(packed.legacy.episode_values(events,index,offset,
            ready['checkpoint_contract']['state_joint_names'],ready['checkpoint_contract']['hand_joint_names']))
        packed.legacy.check_parquet_roundtrip(expected_table,table.select(expected_table.column_names))
        validate_labels(table.to_pydict(),table.num_rows)
        offset+=table.num_rows;windows+=len(eligible)
    identity,_=packed.legacy.checkpoint_contract(*(root/name for name in packed.random100.CONTRACT_FILES))
    require(identity['bounds_sha256']==ready['checkpoint_contract']['bounds_sha256'],'Inherited normalization changed')
    require(offset==ready['total_frames'] and windows==ready['train_windows'],'Merged aggregate counts differ')
    return root,ready,episodes


def merge_cache(pack_root,new_cache,old_cache,output):
    root,ready,episodes=inspect(pack_root,verify_payload=False)
    ex=dino.extractor();cache=ex.cache
    source,items=ex.plan_source(root/'train')
    roots={'new14':Path(new_cache).resolve(strict=True),'original100':Path(old_cache).resolve(strict=True)}
    manifests={k:cache.load_cache_manifest(v,expected_profile='256x320') for k,v in roots.items()}
    for key,m in manifests.items():require(Path(m['source']['dataset_root']).resolve()==Path(ready['source_pack_bindings'][key]['root'])/'train',
        'Source cache belongs to another package')
    reference=manifests['original100'];require(dino.full_encoder_contract(reference)==dino.full_encoder_contract(manifests['new14']),
        'Full DINO model/precision/profile/storage contract differs')
    manifest=ex.build_manifest(source,items,reference['encoder'],shared_image_profile='256x320',storage_dtype=reference['storage_dtype'])
    output=Path(output).absolute();require(not output.exists(),'Use a fresh merged cache')
    staging=output.with_name('.'+output.name+'.merged114-staging-'+uuid.uuid4().hex)
    report={'contract':CACHE_CONTRACT,'state':'merging','created_utc':now(),'pack_ready_sha256':sha(root/'PACK_READY.json'),
        'source_cache_roots':{k:str(v) for k,v in roots.items()},'episodes':[],
        'complete_encoder_contract_sha256':cache.digest(dino.full_encoder_contract(reference)),
        'feature_extraction_performed':False,'original_caches_modified':False,'GPU_used':False}
    try:
        with ex.output_lock(staging):
            cache.atomic_json(staging/'manifest.json',manifest)
            for key,m in manifests.items():copied(roots[key]/'manifest.json',staging/f'relocation_originals/{key}/manifest.json',sha(roots[key]/'manifest.json'))
            for entry,item in zip(ready['episodes'],items):
                component=entry['origin']['component'];origin=entry['origin']['episode_index'];old_manifest=manifests[component]
                require(origin in old_manifest['episode_indices'],'Missing original real DINO episode')
                cached=cache.load_episode_cache(roots[component],origin,manifest=old_manifest,verify_checksums=True)
                original=cached.receipt
                dino.validate_frames(cached,old_manifest['source']['dataset_root'],original['source'],root/'train',item)
                old_dir=cache.episode_directory(roots[component],origin);new_dir=cache.episode_directory(staging,item['episode_index']);new_dir.mkdir(parents=True)
                archive=f'relocation_originals/{component}/episode_{origin:08d}.json'
                copied(old_dir/'receipt.json',staging/archive,sha(old_dir/'receipt.json'))
                for name in ('features.npy','frames.npz'):
                    copied(old_dir/name,new_dir/name,original['outputs'][name]['sha256'])
                    require(cache.file_identity(new_dir/name)==original['outputs'][name],'DINO byte identity changed')
                receipt=dino.reindexed_receipt(original,item,manifest);cache.atomic_json(new_dir/'receipt.json',receipt)
                report['episodes'].append({'episode_index':item['episode_index'],'component':component,'origin_episode_index':origin,
                    'candidate_id':entry['candidate_id'],'original_receipt_path':archive,'original_receipt_sha256':sha(staging/archive),
                    'new_receipt_sha256':sha(new_dir/'receipt.json'),'unchanged_output_sha256':{n:v['sha256'] for n,v in original['outputs'].items()},
                    'receipt_changes':dino.differences(original,receipt)})
                cached.features._mmap.close()
                print(json.dumps({'stage':'real_DINO_copied_verified','episode_index':item['episode_index']}),flush=True)
            manifest.update(state='complete',completed_at=now());cache.atomic_json(staging/'manifest.json',manifest)
            report.update(state='complete',completed_at=now(),manifest_sha256=sha(staging/'manifest.json'))
            cache.atomic_json(staging/'MERGED114_REINDEX.json',report)
            audit_cache(root,staging)
        require(not output.exists(),'Cache output appeared during final verification');staging.rename(output)
        return report
    except BaseException as error:
        if staging.exists():
            for filename in ('manifest.json','MERGED114_REINDEX.json'):
                path=staging/filename
                if path.is_file():value=read(path);value.update(state='failed_incomplete',failure=repr(error));cache.atomic_json(path,value)
        raise


def audit_cache(pack_root,cache_root):
    root,ready,_=inspect(pack_root,verify_payload=False);ex=dino.extractor();cache=ex.cache
    cache_root=Path(cache_root).resolve(strict=True);report=read(cache_root/'MERGED114_REINDEX.json')
    manifest=cache.load_cache_manifest(cache_root,root/'train','256x320');source,items=ex.plan_source(root/'train')
    require(report.get('contract')==CACHE_CONTRACT and report.get('state')=='complete' and len(report['episodes'])==114
        and report['pack_ready_sha256']==sha(root/'PACK_READY.json') and report['manifest_sha256']==sha(cache_root/'manifest.json')
        and manifest['source']==source and manifest['episode_indices']==list(range(114))
        and report['complete_encoder_contract_sha256']==cache.digest(dino.full_encoder_contract(manifest)), 'Merged DINO bindings differ')
    for entry,item,mapping in zip(ready['episodes'],items,report['episodes']):
        component=entry['origin']['component'];origin=entry['origin']['episode_index']
        require(mapping['episode_index']==item['episode_index'] and mapping['component']==component
            and mapping['origin_episode_index']==origin and mapping['candidate_id']==entry['candidate_id'],'DINO origin mapping differs')
        archived_manifest=read(cache_root/f'relocation_originals/{component}/manifest.json')
        require(cache.manifest_signature(archived_manifest)==archived_manifest['signature']
            and dino.full_encoder_contract(archived_manifest)==dino.full_encoder_contract(manifest),'Archived DINO contract differs')
        archived=cache.owned_file(cache_root,mapping['original_receipt_path']);old=read(archived)
        require(sha(archived)==mapping['original_receipt_sha256'] and old['episode_index']==origin
            and old['manifest_signature']==archived_manifest['signature']
            and old['signature']==cache.digest({'manifest_signature':archived_manifest['signature'],'source':old['source']})
            and old['source']['video']['identity']['sha256']==item['video']['identity']['sha256']
            and old['source']['record']['source_identity']==entry['source_identity'], 'Original real cache identity differs')
        cached=cache.load_episode_cache(cache_root,item['episode_index'],manifest=manifest,dataset_root=root/'train',verify_checksums=True)
        require(cached.receipt==dino.reindexed_receipt(old,item,manifest)
            and sha(cache.episode_directory(cache_root,item['episode_index'])/'receipt.json')==mapping['new_receipt_sha256']
            and mapping['unchanged_output_sha256']=={n:v['sha256'] for n,v in old['outputs'].items()}, 'Copied cache bytes/receipt differ')
        frames,clock=ex.read_episode_rows(root/'train',item)
        require(np.array_equal(cached.frame_index,frames) and np.array_equal(cached.timestamp,clock) and cached.valid.all(), 'DINO native clock differs')
        for start in range(0,len(frames),128):require(np.isfinite(cached.features[start:start+128]).all(),'Nonfinite DINO feature')
        cached.features._mmap.close()
    return report


def relocate(pack_root,cache_root):
    """Rebind a transferred cache on a new host without changing data bytes.

    Transfer must first restore source/array mtimes from the recorded file
    identities. Original metadata and the entire incoming cache are retained;
    the new tree is audited privately before replacing the requested cache.
    """
    root,ready,_=inspect(pack_root,verify_payload=False);ex=dino.extractor();cache=ex.cache
    cache_root=Path(cache_root).resolve(strict=True)
    source,items=ex.plan_source(root/'train')
    old_manifest=read(cache_root/'manifest.json');old_report=read(cache_root/'MERGED114_REINDEX.json')
    require(old_manifest.get('state')=='complete' and old_manifest.get('signature')==cache.manifest_signature(old_manifest)
        and old_report.get('contract')==CACHE_CONTRACT and old_report.get('state')=='complete'
        and old_report['manifest_sha256']==sha(cache_root/'manifest.json')
        and old_report['pack_ready_sha256']==sha(root/'PACK_READY.json'),'Transferred cache is not the bound complete merged114 cache')
    previous=deepcopy(old_manifest['source']);previous.pop('dataset_root',None)
    actual=deepcopy(source);actual.pop('dataset_root',None)
    require(previous==actual,'Transferred metadata byte identities/mtime differ; restore recorded mtimes before relocation')
    for item in items:
        directory=cache.episode_directory(cache_root,item['episode_index']);receipt=read(directory/'receipt.json')
        require(receipt['source']==item and receipt['manifest_signature']==old_manifest['signature']
            and receipt['signature']==cache.digest({'manifest_signature':old_manifest['signature'],'source':item}),
            'Transferred dataset/receipt identities changed beyond the source root')
        for name,identity in receipt['outputs'].items():cache.check_file_identity(directory/name,identity,checksum=True)
    backup=cache_root.with_name(cache_root.name+'.before-remote-relocation')
    require(not backup.exists() and not backup.is_symlink(),'Preserve previous remote relocation evidence')
    staging=cache_root.with_name('.'+cache_root.name+'.remote-relocation-'+uuid.uuid4().hex)
    shutil.copytree(cache_root,staging)
    try:
        archive=staging/'relocation_input_originals';archive.mkdir()
        for filename in ('manifest.json','MERGED114_REINDEX.json'):
            copied(cache_root/filename,archive/filename,sha(cache_root/filename))
        manifest=deepcopy(old_manifest);manifest.update(source=source,state='relocating')
        manifest['signature']=cache.manifest_signature(manifest);cache.atomic_json(staging/'manifest.json',manifest)
        report=deepcopy(old_report)
        for item,mapping in zip(items,report['episodes']):
            index=item['episode_index'];directory=cache.episode_directory(staging,index)
            original=read(cache_root/'episodes'/f'{index:08d}'/'receipt.json')
            copied(cache_root/'episodes'/f'{index:08d}'/'receipt.json',archive/f'episode_{index:08d}.json',sha(cache_root/'episodes'/f'{index:08d}'/'receipt.json'))
            receipt=dino.reindexed_receipt(original,item,manifest);cache.atomic_json(directory/'receipt.json',receipt)
            mapping['new_receipt_sha256']=sha(directory/'receipt.json')
            mapping['receipt_changes']=dino.differences(read(staging/mapping['original_receipt_path']),receipt)
        manifest.update(state='complete',relocated_at=now());cache.atomic_json(staging/'manifest.json',manifest)
        report.update(manifest_sha256=sha(staging/'manifest.json'),relocated_at=now())
        cache.atomic_json(staging/'MERGED114_REINDEX.json',report)
        audit_cache(root,staging)
        relocation={'contract':'cvpr-handover-merged114-host-relocation-v1','status':'passed','created_utc':now(),
            'old_dataset_root':old_manifest['source']['dataset_root'],'dataset_root':str(root/'train'),
            'pack_ready_sha256':sha(root/'PACK_READY.json'),'original_manifest_sha256':sha(archive/'manifest.json'),
            'manifest_sha256':sha(staging/'manifest.json'),'reindex_sha256':sha(staging/'MERGED114_REINDEX.json'),
            'incoming_cache_backup':str(backup),'data_and_features_byte_identical':True,'PACK_READY_modified':False}
        cache.atomic_json(staging/'REMOTE_DINO_RELOCATION.json',relocation)
        cache_root.rename(backup)
        try:staging.rename(cache_root)
        except BaseException:
            backup.rename(cache_root)
            raise
        return relocation
    except BaseException as error:
        if staging.exists():
            for filename in ('manifest.json','MERGED114_REINDEX.json'):
                path=staging/filename
                if path.is_file():value=read(path);value.update(state='failed_incomplete',failure=repr(error));cache.atomic_json(path,value)
        raise


def verify(pack_root,cache_root,run_config,output):
    shared.cpu_environment();shared.runtime_paths()
    import torch
    from psi.trainers.qwen3vl_mixin import PaddedCollatorForTogether
    root,ready,episodes=inspect(pack_root);audit_cache(root,cache_root)
    subset,processor,cfg,norm=shared.make_production_subset(run_config,root,cache_root,no_aug=True)
    require(len(subset)==ready['train_windows'],'Production complete-window count differs')
    history=subset.dataset.raw_dataset.motion_history
    collator=PaddedCollatorForTogether(model_max_length=processor.tokenizer.model_max_length,
        pad_token_id=processor.tokenizer.pad_token_id,padding_side='right')
    queries=[]
    for ep in episodes:
        local=[int(i) for i in subset.indices if int(history.episode_ids[i])==ep['episode_index']]
        require(local,'Every merged114 episode must enter production training')
        for index in sorted({local[0],local[len(local)//2],local[-1]}):
            sample=subset.dataset[index];shapes=shared.validate_sample(sample)
            require(np.array_equal(np.asarray(sample['motion_img_embs']),history.get_history(index)),'DINO history changed')
            batch=collator([sample]);shared.validate_sample(batch,batched=True)
            require(torch.isfinite(batch['pixel_values']).all().item(),'Nonfinite production RGB')
            queries.append({'episode_index':ep['episode_index'],'filtered_dataset_index':index,
                'history_rows':history.history_indices[index].tolist(),'sample_shapes':shapes,
                'actual_current_RGB_and_production_collator_passed':True})
    require(not torch.cuda.is_initialized(),'CPU verification initialized CUDA')
    report={'contract':VERIFY_CONTRACT,'status':'passed','verification':'real_merged114_DINO_and_production_reader',
        'created_utc':now(),
        'pack_root':str(root),'dino_root':str(Path(cache_root).resolve()),'train_episodes':114,
        'train_windows':len(subset),'total_frames':ready['total_frames'],'new_episodes':14,'original_episodes':100,
        'pack_ready_sha256':sha(root/'PACK_READY.json'),'cache_manifest_sha256':sha(Path(cache_root)/'manifest.json'),
        'cache_reindex_sha256':sha(Path(cache_root)/'MERGED114_REINDEX.json'),'run_config_sha256':sha(run_config),
        'actual_production_queries':queries,'normalization':norm,'normalization_refitted':False,
        'production_ready_for_bound_pack_and_cache':True,'GPU_used':False,'CUDA_initialized':False,
        'policy_constructed':False,'training_started':False,'original_packages_modified':False,'verifier_sha256':sha(__file__)}
    output=Path(output).absolute();require(not output.exists(),'Preserve any previous verification report');write(output,report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    p=commands.add_parser('pack')
    for name in ('new-pack','old-pack','output'):p.add_argument('--'+name,type=Path,required=True)
    p=commands.add_parser('cache')
    for name in ('pack-root','new-cache','old-cache','output'):p.add_argument('--'+name,type=Path,required=True)
    p=commands.add_parser('verify')
    for name in ('pack-root','cache-root','run-config','output'):p.add_argument('--'+name,type=Path,required=True)
    p=commands.add_parser('relocate')
    for name in ('pack-root','cache-root'):p.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.command=='pack':r=pack(args.new_pack,args.old_pack,args.output)
    elif args.command=='cache':r=merge_cache(args.pack_root,args.new_cache,args.old_cache,args.output)
    elif args.command=='relocate':r=relocate(args.pack_root,args.cache_root)
    else:r=verify(args.pack_root,args.cache_root,args.run_config,args.output)
    print(json.dumps({k:r[k] for k in ('status','state','train_episodes','total_frames','train_windows') if k in r}),flush=True)


if __name__=='__main__':main()
