"""Small explicit cache fixtures; no model inference, video decode or GPU."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import curate_handover_dino_cache as tool
import verify_handover_curated as verifier


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n')


def dataset(root,records):
    info={'fps':30,'total_episodes':len(records),'total_frames':3*len(records),'chunks_size':1000,
        'features':{'observation.images.egocentric':{'dtype':'video','video_info':{'video.fps':30}}},
        'data_path':'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet',
        'video_path':'videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4'}
    put(root/'meta/info.json',info)
    (root/'meta/episodes.jsonl').write_text(''.join(json.dumps(record)+'\n' for record in records))
    for record in records:
        index=record['episode_index'];parquet=root/f'data/chunk-000/episode_{index:06d}.parquet'
        parquet.parent.mkdir(parents=True,exist_ok=True)
        pq.write_table(pa.table({'episode_index':np.full(3,index,np.int64),'frame_index':np.arange(3,dtype=np.int64),
            'timestamp':np.arange(3,dtype=np.float64)/30}),parquet)
        video=root/f'videos/chunk-000/observation.images.egocentric/episode_{index:06d}.mp4'
        video.parent.mkdir(parents=True,exist_ok=True);video.write_bytes(('fixture-video-'+record['candidate_id']).encode())


def cache(root,source,indices):
    ex=tool.extractor();c=ex.cache
    from ImageProcess import SharedImageProcessor
    image=SharedImageProcessor('256x320').metadata
    encoder={'revision':'local','local_file_sha256':{'model.safetensors':'1'*64},'image_process':image,
        'optimization':{'inference_precision':'bf16','compute_precision':'bf16'},'inference_dtype':'torch.float32',
        'preprocessing':{'do_resize':False,'do_center_crop':False,'do_rescale':True,'do_normalize':True,
            'image_mean':[.485,.456,.406],'image_std':[.229,.224,.225],'rescale_factor':1/255}}
    planned,items=ex.plan_source(source,episodes=indices)
    manifest=ex.build_manifest(planned,items,encoder,storage_dtype='float32')
    manifest.update(state='complete',completed_at='fixture-only-not-extraction')
    root.mkdir();c.atomic_json(root/'manifest.json',manifest)
    for item in items:
        directory=c.episode_directory(root,item['episode_index']);directory.mkdir(parents=True)
        np.save(directory/'features.npy',np.full((3,5,1024),item['episode_index']+.25,np.float32))
        frames,clock=ex.read_episode_rows(source,item)
        np.savez(directory/'frames.npz',frame_index=frames,timestamp=clock,source_frame_index=np.arange(3,dtype=np.int64),
            video_timestamp=np.arange(3,dtype=np.float64)/30,valid=np.ones(3,dtype=bool))
        receipt={'contract':c.CONTRACT,'state':'complete','episode_index':item['episode_index'],
            'manifest_signature':manifest['signature'],'source':item,
            'signature':c.digest({'manifest_signature':manifest['signature'],'source':item}),
            'frames':3,'valid_frames':3,'completed_at':'fixture-only-not-extraction',
            'outputs':{name:c.file_identity(directory/name) for name in ('features.npy','frames.npz')}}
        c.atomic_json(directory/'receipt.json',receipt)


class ReindexTests(unittest.TestCase):
    def fixture(self,root):
        old_pack=root/'old-pack';pack=root/'curated';old=old_pack/'train';new=pack/'train'
        record=lambda i,kind:{'episode_index':i,'length':3,'candidate_id':f'{kind}-{i}',
            'source_identity':{'source_dataset_key':kind,'source_episode_index':i},'source_kind':kind}
        old_records=[record(i,'legacy_quality100') for i in range(8)];dataset(old,old_records)
        records=[record(i,'new_random') for i in range(14)]
        retained=deepcopy(old_records[7]);retained['episode_index']=14;records.append(retained)
        dataset(new,records)
        shutil.copy2(old/'videos/chunk-000/observation.images.egocentric/episode_000007.mp4',
            new/'videos/chunk-000/observation.images.egocentric/episode_000014.mp4')
        selection={'contract':'cvpr-handover-curated-selection-v1','new14':[{'candidate_id':r['candidate_id']} for r in records[:14]],
            'old_pack':str(old_pack),'old_episode_indices':[7]}
        put(pack/'selection.json',selection)
        ready={'contract':tool.PACK_CONTRACT,'train_payload_packed':True,'train_episodes':15,
            'curation_selection':{'frozen_path':'selection.json','sha256':tool.sha(pack/'selection.json')},
            'payload_sha256':{'selection.json':tool.sha(pack/'selection.json')},
            'episodes':[{'episode_index':r['episode_index'],'candidate_id':r['candidate_id'],'source_kind':r['source_kind'],
                'frames':3,'origin':{'episode_index':i if i<14 else 7}} for i,r in enumerate(records)]}
        put(pack/'PACK_READY.json',ready)
        new_cache=root/'new-real-fixture-cache';old_cache=root/'old-real-fixture-cache'
        cache(new_cache,new,list(range(14)));cache(old_cache,old,[7])
        return pack,new_cache,old_cache

    def test_all_new14_and_reindexed_legacy_keep_array_and_original_receipt_bytes(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);pack,new,old=self.fixture(root);before=tool.sha(old/'manifest.json')
            report=tool.merge(pack,new,old,root/'merged')
            self.assertEqual(len(report['episodes']),15);self.assertEqual(tool.sha(old/'manifest.json'),before)
            for name in ('features.npy','frames.npz'):
                self.assertEqual(tool.sha(old/'episodes/00000007'/name),tool.sha(root/'merged/episodes/00000014'/name))
            receipt=tool.read(root/'merged/episodes/00000014/receipt.json')
            self.assertEqual(receipt['episode_index'],14);self.assertEqual(receipt['source']['record']['candidate_id'],'legacy_quality100-7')
            self.assertEqual(tool.audit_merge(pack,root/'merged')['state'],'complete')

    def test_video_byte_mismatch_is_rejected(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);pack,new,old=self.fixture(root)
            (pack/'train/videos/chunk-000/observation.images.egocentric/episode_000014.mp4').write_bytes(b'changed-video')
            with self.assertRaisesRegex(ValueError,'byte-identical'):tool.merge(pack,new,old,root/'merged')

    def test_clock_change_is_rejected_even_when_video_bytes_match(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);pack,new,old=self.fixture(root)
            path=pack/'train/data/chunk-000/episode_000014.parquet';table=pq.read_table(path).to_pydict()
            table['timestamp']=[0.,.04,.08];pq.write_table(pa.table(table),path)
            with self.assertRaisesRegex(ValueError,'timestamp clock'):tool.merge(pack,new,old,root/'merged')

    def test_complete_precision_contract_cannot_be_silently_changed(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);pack,new,old=self.fixture(root);path=old/'manifest.json';manifest=tool.read(path)
            manifest['encoder']['optimization']['compute_precision']='fp32'
            manifest['signature']=tool.extractor().cache.manifest_signature(manifest);put(path,manifest)
            with self.assertRaisesRegex(ValueError,'complete actual encoder'):tool.merge(pack,new,old,root/'merged')

    def test_mapping_receipt_cannot_claim_another_legacy_origin(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);pack,new,old=self.fixture(root);tool.merge(pack,new,old,root/'merged')
            path=root/'merged/CURATED_REINDEX.json';report=tool.read(path);report['episodes'][-1]['origin_episode_index']=0;put(path,report)
            with self.assertRaisesRegex(ValueError,'origin identity'):tool.audit_merge(pack,root/'merged')

    def test_failed_final_audit_never_publishes_complete_cache(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);pack,new,old=self.fixture(root);source_SHA=tool.sha(old/'manifest.json')
            with patch.object(tool,'audit_merge',side_effect=ValueError('fixture final audit failed')):
                with self.assertRaisesRegex(ValueError,'fixture final audit failed'):tool.merge(pack,new,old,root/'merged')
            self.assertFalse((root/'merged').exists())
            staging=list(root.glob('.merged.curated-staging-*'));self.assertEqual(len(staging),1)
            self.assertEqual(tool.read(staging[0]/'manifest.json')['state'],'failed_incomplete')
            self.assertEqual(tool.read(staging[0]/'CURATED_REINDEX.json')['state'],'failed_incomplete')
            with self.assertRaisesRegex(ValueError,'incomplete'):tool.extractor().cache.load_cache_manifest(staging[0])
            self.assertEqual(tool.sha(old/'manifest.json'),source_SHA)


class LabelTests(unittest.TestCase):
    def labels(self):
        action=np.ones((64,14));token=np.ones((64,64));action[:30]=np.nan;token[:30]=np.nan
        mask=np.zeros((64,80),bool);mask[30:,:78]=True
        return {'action_valid':[False]*30+[True]*34,'action':action,'action.body_token_v1_1':token,'action_mask':mask}
    def test_real_warmup_and78_plus2_mask(self):verifier.validate_labels(self.labels(),64)
    def test_padded_action_dimension_cannot_be_marked_valid(self):
        values=self.labels();values['action_mask'][40,79]=True
        with self.assertRaises(ValueError):verifier.validate_labels(values,64)
    def test_warmup_cannot_be_given_finite_action_labels(self):
        values=self.labels();values['action'][0]=0
        with self.assertRaises(ValueError):verifier.validate_labels(values,64)


class LineageTests(unittest.TestCase):
    def episode(self):
        source={'source_dataset_key':'legacy_train','source_episode_index':7,'category':'handover','video_sha256':'a'*64}
        return {'source_kind':'legacy_quality100','source_identity':source,'source_dataset_key':'legacy_train',
            'source_episode_index':7,'lineage_group_sources':[{'source_dataset_key':'legacy_train','source_episode_index':7}]}
    def test_legacy_minimal_ancestry_keeps_full_receipt_identity(self):
        episode=self.episode();verifier.validate_lineage(episode,{'source_identity':deepcopy(episode['source_identity'])})
    def test_legacy_group_cannot_change_namespace(self):
        episode=self.episode();episode['lineage_group_sources'][0]['source_dataset_key']='reserved'
        with self.assertRaises(ValueError):verifier.validate_lineage(episode,{'source_identity':episode['source_identity']})
    def test_legacy_extra_identity_cannot_differ_from_receipt(self):
        episode=self.episode();entry={'source_identity':deepcopy(episode['source_identity'])};entry['source_identity']['video_sha256']='b'*64
        with self.assertRaises(ValueError):verifier.validate_lineage(episode,entry)


if __name__=='__main__':unittest.main()
