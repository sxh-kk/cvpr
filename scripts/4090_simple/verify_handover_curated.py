"""CPU-only acceptance of a variable-size curated pack and real DINO cache.

This is its own curation contract. It does not weaken an exact100 or diversity
contract, refit normalization, construct a policy, or promote PACK_READY.
"""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
from pathlib import Path
import json

import verify_handover_supplement as shared
import curate_handover_dino_cache as relocated

REPO=Path(__file__).resolve().parents[2]
CONTRACT='cvpr-handover-curated-real-DINO-production-reader-verification-v1'
require,sha,read,jsonl=shared.require,shared.sha,shared.read,shared.jsonl


def validate_lineage(episode,entry):
    source=episode.get('source_identity')
    require(isinstance(source,dict) and source==entry['source_identity']
        and episode['source_dataset_key']==source['source_dataset_key']
        and episode['source_episode_index']==source['source_episode_index'], 'Full receipt/source namespace identity differs')
    # Legacy metadata records a minimal ancestry key while its source_identity
    # additionally binds category and original Parquet/video bytes. Keep that
    # inherited representation; all full identities remain checked above and
    # by the packer's byte-for-byte old metadata/selection audit.
    group=([{'source_dataset_key':source['source_dataset_key'],'source_episode_index':source['source_episode_index']}]
        if episode['source_kind']=='legacy_quality100' else [source])
    require(episode.get('lineage_group_sources')==group,'Protected lineage namespace/id differs')


def inspect_pack(pack):
    """Require the dedicated packer's complete portable quality/label audit."""
    import pack_handover_curated as packer
    pack=Path(pack).resolve(strict=True)
    inspected,_=packer.inspect(pack,verify_payload=True)
    actual=read(pack/'PACK_READY.json')
    require(actual==inspected and actual.get('contract')==relocated.PACK_CONTRACT
        and actual.get('train_payload_packed') is True and actual.get('normalization_refitted') is False
        and actual.get('final_ready') is False,'Require immutable curated quality acceptance, not a relaxed new100 pack')
    _,_,entries,_,_=relocated.target_plan(pack)
    episodes=jsonl(pack/'train/meta/episodes.jsonl')
    require(len(episodes)==len(entries)==actual['train_episodes']
        and {e['episode_index'] for e in episodes}==set(range(len(episodes))), 'Actual curated episode metadata differs')
    receipts={str(pack/'PACK_READY.json'):sha(pack/'PACK_READY.json')}
    for relative,wanted in actual['payload_sha256'].items():
        path=relocated.extractor().cache.owned_file(pack,relative)
        require(sha(path)==wanted,'Changed curated payload: '+relative);receipts[str(path)]=wanted
    by_index={e['episode_index']:e for e in entries}
    for episode in episodes:
        entry=by_index[episode['episode_index']]
        require(episode['candidate_id']==entry['candidate_id'] and episode['source_kind']==entry['source_kind']
            and episode['length']==entry['frames'] and episode['eligible_current_rows']==entry['eligible_current_rows']
            and episode.get('split')=='train' and episode.get('split_restriction')=='train_only'
            and episode.get('lineage_complete') is True, 'Protected single-source TRAIN identity differs')
        validate_lineage(episode,entry)
    require(sum(e['length'] for e in episodes)==actual['total_frames']
        and sum(len(e['eligible_current_rows']) for e in episodes)==actual['train_windows'], 'Curated frame/window counts differ')
    return pack,actual,episodes,by_index,receipts


def validate_labels(raw,length):
    import numpy as np
    valid=np.asarray(raw['action_valid'],dtype=bool)
    hands,tokens,mask=(np.asarray(raw[key]) for key in ('action','action.body_token_v1_1','action_mask'))
    require(valid.shape==(length,) and hands.shape==(length,14) and tokens.shape==(length,64)
        and mask.shape==(length,80) and not valid[:30].any() and valid[30:].all()
        and np.isnan(hands[:30]).all() and np.isnan(tokens[:30]).all() and not mask[:30].any()
        and np.isfinite(hands[30:]).all() and np.isfinite(tokens[30:]).all()
        and mask[30:,:78].all() and not mask[30:,78:].any(), 'Actual warmup/78+2 masks/labels differ')


def make_production_subset(run_config,pack,dino_root,*,processor=None,no_aug=True):
    inspect_pack(pack)
    return shared.make_production_subset(run_config,pack,dino_root,processor=processor,no_aug=no_aug)


def verify(pack,dino_root,reference_dino_root,run_config,output):
    shared.cpu_environment();shared.runtime_paths()
    import numpy as np
    import pyarrow.parquet as pq
    import torch
    from psi.data.lerobot.dino_cache import load_cache_manifest,load_episode_cache
    from psi.trainers.qwen3vl_mixin import PaddedCollatorForTogether
    require(not torch.cuda.is_initialized(),'CPU verification must not initialize CUDA')
    pack,ready,episodes,entries,receipts=inspect_pack(pack)
    dino_root=Path(dino_root).resolve(strict=True);reference_dino_root=Path(reference_dino_root).resolve(strict=True)
    output=Path(output).resolve();run_config=Path(run_config).resolve(strict=True)
    require(not output.exists() and not any(output.is_relative_to(root) for root in (REPO,pack,dino_root,reference_dino_root)),
        'Use a fresh report outside source/package/cache trees')
    merge=relocated.audit_merge(pack,dino_root)
    manifest=load_cache_manifest(dino_root,pack/'train',expected_profile='256x320')
    reference=load_cache_manifest(reference_dino_root,expected_profile='256x320')
    require(relocated.full_encoder_contract(manifest)==relocated.full_encoder_contract(reference),
        'Actual complete encoder/compute precision/profile/storage contract changed')
    receipts[str(run_config)]=sha(run_config)
    for path in (dino_root/'manifest.json',dino_root/'CURATED_REINDEX.json',reference_dino_root/'manifest.json'):
        receipts[str(path)]=sha(path)
    reports=[]
    for episode in episodes:
        index,length=episode['episode_index'],episode['length']
        cached=load_episode_cache(dino_root,index,manifest=manifest,dataset_root=pack/'train',verify_checksums=True)
        require(cached.features.shape==(length,5,1024) and cached.features.dtype==np.float32 and cached.valid.all()
            and np.array_equal(cached.frame_index,np.arange(length))
            and np.array_equal(cached.source_frame_index,np.arange(length))
            and np.allclose(cached.video_timestamp,np.arange(length)/30,rtol=0,atol=1e-7)
            and np.allclose(cached.timestamp,cached.video_timestamp,rtol=0,atol=.25/30), 'Actual full-frame DINO identity differs')
        for start in range(0,length,128):require(np.isfinite(cached.features[start:start+128]).all(),'Nonfinite real DINO')
        directory=dino_root/'episodes'/f'{index:08d}'
        for filename in ('receipt.json','features.npy','frames.npz'):receipts[str(directory/filename)]=sha(directory/filename)
        raw=pq.read_table(pack/'train/data/chunk-000'/f'episode_{index:06d}.parquet',
            columns=['action_valid','action','action.body_token_v1_1','action_mask']).to_pydict()
        validate_labels(raw,length)
        reports.append({'episode_index':index,'candidate_id':episode['candidate_id'],'source_kind':episode['source_kind'],
            'actual_shape':list(cached.features.shape),'storage_dtype':str(cached.features.dtype),'all_real_frames_finite_and_valid':True})
        cached.features._mmap.close()
    subset,processor,cfg,normalization=make_production_subset(run_config,pack,dino_root,no_aug=True)
    require(len(subset)==ready['train_windows'],'Actual production admitted window count differs')
    history=subset.dataset.raw_dataset.motion_history
    collator=PaddedCollatorForTogether(model_max_length=processor.tokenizer.model_max_length,
        pad_token_id=processor.tokenizer.pad_token_id,padding_side='right')
    queries=[]
    for episode in episodes:
        local=[int(index) for index in subset.indices if int(history.episode_ids[index])==episode['episode_index']]
        require(local,'Every curated episode needs actual eligible production supervision')
        chosen=sorted({local[0],local[len(local)//2],local[-1]})
        for index in chosen:
            sample=subset.dataset[index];shapes=shared.validate_sample(sample)
            require(np.array_equal(np.asarray(sample['motion_img_embs']),history.get_history(index)),
                'Production sample changed native clean DINO history')
            batch=collator([sample]);shared.validate_sample(batch,batched=True)
            require(torch.isfinite(batch['pixel_values']).all().item(),'Actual production RGB/collator has nonfinite pixels')
            queries.append({'episode_index':episode['episode_index'],'filtered_dataset_index':index,
                'current_row':int(history.history_indices[index][-1]),'history_rows':history.history_indices[index].tolist(),
                'sample_shapes':shapes,'actual_current_RGB_and_production_collator_passed':True})
    for path in (dino_root/'relocation_originals').rglob('*'):
        if path.is_file():receipts[str(path)]=sha(path)
    require(all(sha(path)==wanted for path,wanted in receipts.items()),'Read-only verification input changed')
    require(not torch.cuda.is_initialized(),'CPU verifier initialized CUDA')
    report={'contract':CONTRACT,'status':'real_curated_DINO_and_production_reader_passed',
        'created_utc':datetime.now(timezone.utc).isoformat(),'pack_root':str(pack),'dino_root':str(dino_root),
        'train_episodes':len(episodes),'train_windows':len(subset),'new14_retained':True,
        'frozen_selection_sha256':ready['curation_selection']['sha256'],'actual_cache_episodes':reports,
        'actual_production_queries':queries,'normalization':normalization,'normalization_refitted':False,
        'dataset_type':type(subset.dataset).__module__+'.'+type(subset.dataset).__name__,
        'raw_dataset_type':type(subset.dataset.raw_dataset).__module__+'.'+type(subset.dataset.raw_dataset).__name__,
        'source_sha256':receipts,'verifier_sha256':sha(__file__),'cache_tool_sha256':sha(relocated.__file__),
        'old_feature_arrays_recomputed':False,'GPU_used':False,'CUDA_initialized':False,'policy_constructed':False,
        'training_started':False,'A100_uploaded':False,'original_packages_modified':False,'VAL_or_TEST_loaded':False,
        'PACK_READY_modified':False,'final_ready_automatically_promoted':False,
        'production_ready_for_bound_pack_and_cache':True,'formal100_quality_or_diversity_claimed':False,
        'scope':'Variable-size curated TRAIN package; all new14 plus frozen selected legacy quality episodes, inherited38k normalization, true DINO frame bytes and actual CPU Dataset/MotionHistoryIndex/RGB/collator verified.'}
    output.parent.mkdir(parents=True,exist_ok=True)
    partial=output.with_name(output.name+'.partial')
    require(not output.exists() and not output.is_symlink() and not partial.exists() and not partial.is_symlink(),
        'Preserve any prior verification/failure output; choose a fresh report path')
    relocated.extractor().cache.atomic_json(output,report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ('pack-root','dino-root','reference-dino-root','run-config','output'):parser.add_argument('--'+flag,type=Path,required=True)
    args=parser.parse_args();report=verify(args.pack_root,args.dino_root,args.reference_dino_root,args.run_config,args.output)
    print(json.dumps({key:report[key] for key in ('status','train_episodes','train_windows','GPU_used')}))


if __name__=='__main__':main()
