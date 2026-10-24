"""Reindex verified real native-frame DINO arrays without re-encoding them.

The new14 cache must already have been extracted by the real writer. This CPU
tool copies existing feature/frame bytes and binds them to the actual curated
videos, Parquet clocks and metadata. Original caches remain read-only.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from functools import lru_cache
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import uuid

import numpy as np

REPO=Path(__file__).resolve().parents[2]
PACK_CONTRACT='cvpr-handover-curated-lerobot-pack-v1'
CONTRACT='cvpr-handover-curated-real-DINO-reindex-v1'
RECEIPT_FIELDS=frozenset(('episode_index','manifest_signature','source','signature'))


def require(value,message):
    if not value:raise ValueError(message)


@lru_cache(maxsize=1)
def extractor():
    path=REPO/'VLMExpert/scripts/data/extract_unifolm_dino.py'
    spec=importlib.util.spec_from_file_location('_curated_extract_unifolm_dino',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module


def read(path):return json.loads(Path(path).read_text())
def sha(path):return extractor().cache.file_sha256(path)
def now():return datetime.now(timezone.utc).isoformat()


def full_encoder_contract(manifest):
    return {key:deepcopy(manifest[key]) for key in ('contract','image_process','encoder','extraction','storage_dtype','shape')}


def differences(before,after,prefix=''):
    if isinstance(before,dict) and isinstance(after,dict):
        result=[]
        for key in sorted(set(before)|set(after)):
            path=prefix+'/'+key
            if key not in before:result.append({'path':path,'added':after[key]})
            elif key not in after:result.append({'path':path,'removed':before[key]})
            else:result.extend(differences(before[key],after[key],path))
        return result
    return [] if before==after else [{'path':prefix,'before':before,'after':after}]


def target_plan(pack):
    pack=Path(pack).resolve(strict=True);ready=read(pack/'PACK_READY.json')
    require(ready.get('contract')==PACK_CONTRACT and ready.get('train_payload_packed') is True,
        'Require an actual curated pack, not a relaxed exact100 pack')
    selection_binding=ready['curation_selection'];frozen=selection_binding['frozen_path']
    selection_path=extractor().cache.owned_file(pack,frozen)
    require(sha(selection_path)==selection_binding['sha256']==ready['payload_sha256'].get(frozen),
        'Frozen curation selection is unbound or changed')
    selection=read(selection_path)
    require(selection.get('contract')=='cvpr-handover-curated-selection-v1'
        and len(selection['new14'])==14,'Require the independently frozen new14/legacy curation selection')
    entries=ready['episodes'];count=ready['train_episodes']
    require(type(count) is int and count>=14 and len(entries)==count
        and {e['episode_index'] for e in entries}==set(range(count)), 'Curated episode mapping is incomplete')
    entries=sorted(entries,key=lambda e:e['episode_index'])
    require([e['episode_index'] for e in entries if e['source_kind']=='new_random']==list(range(14))
        and all(e['source_kind']=='legacy_quality100' for e in entries[14:]), 'Keep all new14 first, then selected legacy episodes')
    require([e['candidate_id'] for e in entries[:14]]==[e['candidate_id'] for e in selection['new14']]
        and [e['origin']['episode_index'] for e in entries[14:]]==selection['old_episode_indices']
        and count==14+len(selection['old_episode_indices']), 'Ready mapping differs from frozen ordered selection')
    require(len({e['candidate_id'] for e in entries})==count,'Duplicate candidate mapping')
    source,items=extractor().plan_source(pack/'train')
    require(len(items)==count and [i['episode_index'] for i in items]==list(range(count)),'Actual curated dataset differs')
    for entry,item in zip(entries,items):
        require(item['record']['candidate_id']==entry['candidate_id'] and item['record']['length']==entry['frames'],
            'Curated candidate/frame identity differs from ready receipt')
    return pack,ready,entries,source,items


def validate_frames(cached,source_root,source_item,target_root,target_item):
    ex=extractor();source_frames,source_clock=ex.read_episode_rows(source_root,source_item)
    target_frames,target_clock=ex.read_episode_rows(target_root,target_item)
    require(np.array_equal(source_frames,target_frames) and np.array_equal(source_clock,target_clock),
        'Reindexing must not change native frame order or timestamp clock')
    require(source_item['record']['candidate_id']==target_item['record']['candidate_id']
        and source_item['record']['source_identity']==target_item['record']['source_identity']
        and source_item['record']['length']==target_item['record']['length'], 'Source lineage/candidate/length differs')
    require(source_item['video']['identity']['sha256']==target_item['video']['identity']['sha256']
        and source_item['video']['identity']['size_bytes']==target_item['video']['identity']['size_bytes'],
        'DINO reuse requires byte-identical actual video')
    require(np.array_equal(cached.frame_index,target_frames) and np.array_equal(cached.timestamp,target_clock)
        and np.array_equal(cached.source_frame_index,np.arange(len(target_frames))) and cached.valid.all()
        and np.allclose(cached.video_timestamp,np.arange(len(target_frames))/30,rtol=0,atol=1e-7)
        and np.allclose(target_clock,cached.video_timestamp,rtol=0,atol=.25/30), 'Real DINO frame identities differ')
    for start in range(0,len(target_frames),128):
        require(np.isfinite(cached.features[start:start+128]).all(),'Nonfinite existing real DINO features')


def reindexed_receipt(original,item,manifest):
    cache=extractor().cache;receipt=deepcopy(original)
    receipt.update(episode_index=item['episode_index'],manifest_signature=manifest['signature'],source=item,
        signature=cache.digest({'manifest_signature':manifest['signature'],'source':item}))
    require({key for key in set(receipt)|set(original) if receipt.get(key)!=original.get(key)}<=RECEIPT_FIELDS,
        'Only explicit episode/source/signature reindex fields may change')
    return receipt


def _merge_into(pack,new_cache,old_cache,output):
    ex=extractor();cache=ex.cache
    pack,ready,entries,source,items=target_plan(pack)
    roots={'new_random':Path(new_cache).resolve(strict=True),'legacy_quality100':Path(old_cache).resolve(strict=True)}
    manifests={kind:cache.load_cache_manifest(root,expected_profile='256x320') for kind,root in roots.items()}
    selection=read(pack/ready['curation_selection']['frozen_path'])
    require(Path(manifests['legacy_quality100']['source']['dataset_root']).resolve()==(Path(selection['old_pack'])/'train').resolve()
        and Path(manifests['new_random']['source']['dataset_root']).resolve()==(pack/'train').resolve()
        and manifests['new_random']['episode_indices']==list(range(14)),
        'Use the actual selected legacy cache and real new14 extraction from this curated dataset')
    require(full_encoder_contract(manifests['new_random'])==full_encoder_contract(manifests['legacy_quality100']),
        'New and legacy caches must have the same complete actual encoder/precision/profile/storage contract')
    reference=manifests['legacy_quality100']
    manifest=ex.build_manifest(source,items,reference['encoder'],shared_image_profile='256x320',storage_dtype=reference['storage_dtype'])
    require(full_encoder_contract(manifest)==full_encoder_contract(reference),'Derived manifest changed the actual encoder contract')
    output=Path(output).resolve()
    inputs=[pack,*roots.values(),*(Path(m['source']['dataset_root']).resolve(strict=True) for m in manifests.values())]
    require(not output.exists() and not any(output.is_relative_to(root) or root.is_relative_to(output) for root in inputs),
        'Use a fresh independent cache output outside all source datasets/caches')
    bindings={str(pack/'PACK_READY.json'):sha(pack/'PACK_READY.json')}
    for root in roots.values():bindings[str(root/'manifest.json')]=sha(root/'manifest.json')
    report={'contract':CONTRACT,'state':'merging','created_utc':now(),'pack_root':str(pack),
        'pack_ready_sha256':bindings[str(pack/'PACK_READY.json')],'source_cache_roots':{k:str(v) for k,v in roots.items()},
        'feature_extraction_performed':False,'old_features_reextracted':False,'new14_required_real_extraction':True,
        'GPU_used':False,'original_caches_modified':False,'episodes':[],
        'receipt_changed_top_level_fields_allowed':sorted(RECEIPT_FIELDS),
        'complete_encoder_contract_sha256':cache.digest(full_encoder_contract(reference))}
    with ex.output_lock(output):
        cache.atomic_json(output/'manifest.json',manifest)
        for kind,root in roots.items():
            destination=output/'relocation_originals'/kind/'manifest.json';destination.parent.mkdir(parents=True)
            shutil.copy2(root/'manifest.json',destination)
            require(sha(destination)==bindings[str(root/'manifest.json')],'Original manifest archival changed bytes')
        seen=set()
        for entry,item in zip(entries,items):
            kind=entry['source_kind'];origin=item['episode_index'] if kind=='new_random' else entry['origin']['episode_index']
            require(type(origin) is int and (kind,origin) not in seen,'Invalid/repeated source cache episode')
            seen.add((kind,origin));root=roots[kind];old_manifest=manifests[kind]
            require(origin in old_manifest['episode_indices'],'Actual source cache is missing a selected episode')
            cached=cache.load_episode_cache(root,origin,manifest=old_manifest,verify_checksums=True)
            original=cached.receipt;source_root=Path(old_manifest['source']['dataset_root']).resolve(strict=True)
            validate_frames(cached,source_root,original['source'],pack/'train',item)
            original_dir=cache.episode_directory(root,origin);new_dir=cache.episode_directory(output,item['episode_index'])
            new_dir.mkdir(parents=True)
            archive=output/'relocation_originals'/kind/f'episode_{origin:08d}.json'
            shutil.copy2(original_dir/'receipt.json',archive)
            for name in ('features.npy','frames.npz'):
                require(not (original_dir/name).is_symlink(),'Source feature/frame arrays must be actual files')
                shutil.copy2(original_dir/name,new_dir/name)
                require(cache.file_identity(new_dir/name)==original['outputs'][name],'Copied DINO/frame bytes or mtime changed')
                bindings[str(original_dir/name)]=original['outputs'][name]['sha256']
            bindings[str(original_dir/'receipt.json')]=sha(original_dir/'receipt.json')
            receipt=reindexed_receipt(original,item,manifest)
            cache.atomic_json(new_dir/'receipt.json',receipt)
            report['episodes'].append({'episode_index':item['episode_index'],'source_kind':kind,'origin_episode_index':origin,
                'candidate_id':entry['candidate_id'],'original_receipt_path':str(archive.relative_to(output)),
                'original_receipt_sha256':sha(archive),'new_receipt_sha256':sha(new_dir/'receipt.json'),
                'source_manifest_path':f'relocation_originals/{kind}/manifest.json',
                'original_video_sha256':original['source']['video']['identity']['sha256'],
                'target_video_sha256':item['video']['identity']['sha256'],
                'unchanged_output_sha256':{n:v['sha256'] for n,v in original['outputs'].items()},
                'receipt_changes':differences(original,receipt),'all_frames_finite_and_valid':True})
            cached.features._mmap.close()
            ex._check_inputs(source_root,original['source']);ex._check_inputs(pack/'train',item)
        require(all(sha(path)==expected for path,expected in bindings.items()),'Read-only source changed during merge')
        manifest.update(state='complete',completed_at=now());cache.atomic_json(output/'manifest.json',manifest)
        report.update(state='complete',completed_at=now(),source_sha256=bindings,manifest_sha256=sha(output/'manifest.json'))
        cache.atomic_json(output/'CURATED_REINDEX.json',report)
        audit_merge(pack,output)
    return report


def merge(pack,new_cache,old_cache,output):
    """Validate a fresh private staging tree, then atomically publish it."""
    cache=extractor().cache
    output=Path(output).absolute()
    require(not output.exists() and not output.is_symlink(),'Use a fresh final cache path')
    output=output.resolve()
    inputs=[Path(value).resolve(strict=True) for value in (pack,new_cache,old_cache)]
    require(not any(output.is_relative_to(root) or root.is_relative_to(output) for root in inputs),
        'Keep the final cache independent of all inputs')
    output.parent.mkdir(parents=True,exist_ok=True)
    staging=output.with_name('.'+output.name+'.curated-staging-'+uuid.uuid4().hex)
    try:
        report=_merge_into(pack,new_cache,old_cache,staging)
        require(not output.exists() and not output.is_symlink(),'Final cache appeared during staging verification')
        staging.rename(output)
        return report
    except BaseException as error:
        # Preserve failed bytes as evidence and make every reader fail closed.
        # The requested final path is never created before full audit succeeds.
        if staging.exists():
            for name in ('manifest.json','CURATED_REINDEX.json'):
                path=staging/name
                if path.is_file():
                    value=read(path);value.update(state='failed_incomplete',failure=repr(error),failed_at=now())
                    cache.atomic_json(path,value)
            cache.atomic_json(staging/'FAILED.json',{'contract':CONTRACT,'state':'failed_incomplete',
                'failed_at':now(),'error':repr(error),'requested_final_cache':str(output),
                'original_inputs_modified':False})
        raise


def audit_merge(pack,cache_root):
    """Verify portable original receipts and actual derived target bindings."""
    ex=extractor();cache=ex.cache
    pack,ready,entries,source,items=target_plan(pack);root=Path(cache_root).resolve(strict=True)
    report=read(root/'CURATED_REINDEX.json');manifest=cache.load_cache_manifest(root,pack/'train','256x320')
    require(report.get('contract')==CONTRACT and report.get('state')=='complete'
        and report['pack_ready_sha256']==sha(pack/'PACK_READY.json') and report['manifest_sha256']==sha(root/'manifest.json')
        and report.get('feature_extraction_performed') is False and report.get('old_features_reextracted') is False
        and report.get('new14_required_real_extraction') is True and report.get('GPU_used') is False
        and report.get('original_caches_modified') is False,
        'Require completed explicit byte-preserving reindex evidence')
    require(manifest['source']==source and manifest['episode_indices']==list(range(len(entries)))
        and len(report['episodes'])==len(entries)
        and report['complete_encoder_contract_sha256']==cache.digest(full_encoder_contract(manifest)),
        'Derived cache episode/source/encoder plan differs')
    for entry,item,mapping in zip(entries,items,report['episodes']):
        require(mapping['episode_index']==item['episode_index'] and mapping['candidate_id']==entry['candidate_id']
            and mapping['source_kind']==entry['source_kind']
            and mapping['origin_episode_index']==(item['episode_index'] if entry['source_kind']=='new_random' else entry['origin']['episode_index']),
            'Reindex candidate/origin identity differs')
        kind=entry['source_kind'];original_root=Path(report['source_cache_roots'][kind])
        require(mapping['source_manifest_path']==f'relocation_originals/{kind}/manifest.json'
            and mapping['original_receipt_path']==f"relocation_originals/{kind}/episode_{mapping['origin_episode_index']:08d}.json",
            'Original manifest/receipt archive path differs')
        archived_manifest=cache.owned_file(root,mapping['source_manifest_path']);old_manifest=cache.read_json(archived_manifest)
        require(sha(archived_manifest)==report['source_sha256'][str(original_root/'manifest.json')]
            and cache.manifest_signature(old_manifest)==old_manifest['signature']
            and full_encoder_contract(old_manifest)==full_encoder_contract(manifest),'Archived source encoder contract/signature differs')
        archived=cache.owned_file(root,mapping['original_receipt_path']);old=read(archived)
        require(sha(archived)==mapping['original_receipt_sha256']
            ==report['source_sha256'][str(cache.episode_directory(original_root,mapping['origin_episode_index'])/'receipt.json')]
            and old['manifest_signature']==old_manifest['signature']
            and old['signature']==cache.digest({'manifest_signature':old_manifest['signature'],'source':old['source']}),
            'Original source receipt is unbound or modified')
        require(old['episode_index']==old['source']['episode_index']==mapping['origin_episode_index']
            and old['episode_index'] in old_manifest['episode_indices'] and old['source']['record']['candidate_id']==entry['candidate_id']
            and old['source']['record']['source_identity']==item['record']['source_identity']
            and old['source']['video']['identity']['sha256']==item['video']['identity']['sha256'], 'Original video/lineage mapping differs')
        require(mapping['original_video_sha256']==mapping['target_video_sha256']==item['video']['identity']['sha256']
            and mapping['unchanged_output_sha256']=={name:value['sha256'] for name,value in old['outputs'].items()},
            'Declared byte-preserving video/feature evidence differs')
        cached=cache.load_episode_cache(root,item['episode_index'],manifest=manifest,dataset_root=pack/'train',verify_checksums=True)
        require(cached.receipt==reindexed_receipt(old,item,manifest)
            and sha(cache.episode_directory(root,item['episode_index'])/'receipt.json')==mapping['new_receipt_sha256']
            and differences(old,cached.receipt)==mapping['receipt_changes'],'Reindex changes exceed archived receipt whitelist')
        target_frames,target_clock=ex.read_episode_rows(pack/'train',item)
        require(np.array_equal(cached.frame_index,target_frames) and np.array_equal(cached.timestamp,target_clock)
            and cached.valid.all(),'Derived frames/clock changed')
        for start in range(0,len(target_frames),128):require(np.isfinite(cached.features[start:start+128]).all(),'Nonfinite real DINO')
        cached.features._mmap.close()
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('pack-root','new-cache','old-cache','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();report=merge(args.pack_root,args.new_cache,args.old_cache,args.output)
    print(json.dumps({'state':report['state'],'episodes':len(report['episodes']),'GPU_used':False}))


if __name__=='__main__':main()
