"""Critical merged114 completeness and atomic host relocation regressions."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

SCRIPTS=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SCRIPTS))
import merge_handover_114 as merged


class MappingTests(unittest.TestCase):
    def entries(self):
        return [{'episode_index':i,'origin':{'component':c,'episode_index':o},
            'candidate_id':f'capture-{i}','source_kind':'new_random' if c=='new14' else 'legacy_quality100'}
            for c,o,i in merged.expected_mapping()]

    def test_rejects_previous_curated28_instead_of_requested114(self):
        with self.assertRaises(ValueError):merged.validate_mapping(self.entries()[:28])

    def test_rejects_missing_old_episode_disguised_with_duplicate(self):
        entries=self.entries();entries[-1]=copy.deepcopy(entries[-2]);entries[-1]['episode_index']=113
        with self.assertRaises(ValueError):merged.validate_mapping(entries)

    def test_rejects_changed_component_order_and_repeated_capture(self):
        entries=self.entries();entries[14],entries[15]=entries[15],entries[14]
        with self.assertRaises(ValueError):merged.validate_mapping(entries)
        entries=self.entries();entries[-1]['candidate_id']=entries[0]['candidate_id']
        with self.assertRaises(ValueError):merged.validate_mapping(entries)


class RelocationTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name);self.pack=self.root/'packed';self.pack.mkdir()
        merged.write(self.pack/'PACK_READY.json',{'bound':'immutable-test-fixture'})
        self.cache_root=self.root/'dino';self.cache_root.mkdir()
        self.cache=merged.dino.extractor().cache
        self.source={'dataset_root':str(self.pack/'train'),'fps':30,'video_key':'test','meta_files':{'test':'identity'}}
        self.old_source=copy.deepcopy(self.source);self.old_source['dataset_root']='/original-host/packed/train'
        self.items=[{'episode_index':i,'record':{'candidate_id':f'fixture-{i}'}} for i in range(2)]
        self.manifest={'contract':'test-only-real-file-identity-fixture','state':'complete','source':self.old_source,
            'image_process':{},'encoder':{},'extraction':{},'storage_dtype':'float32','shape':[5,1024],
            'episode_indices':[0,1]}
        self.manifest['signature']=self.cache.manifest_signature(self.manifest)
        self.cache.atomic_json(self.cache_root/'manifest.json',self.manifest)
        mappings=[]
        for item in self.items:
            i=item['episode_index'];directory=self.cache.episode_directory(self.cache_root,i);directory.mkdir(parents=True)
            np.save(directory/'features.npy',np.ones((3,5,1024),np.float32)*i)
            np.savez(directory/'frames.npz',frame_index=np.arange(3))
            receipt={'episode_index':i,'source':item,'manifest_signature':self.manifest['signature'],
                'signature':self.cache.digest({'manifest_signature':self.manifest['signature'],'source':item}),
                'outputs':{n:self.cache.file_identity(directory/n) for n in ('features.npy','frames.npz')}}
            self.cache.atomic_json(directory/'receipt.json',receipt)
            archive=f'relocation_originals/fixture-{i}.json'
            (self.cache_root/archive).parent.mkdir(exist_ok=True)
            self.cache.atomic_json(self.cache_root/archive,receipt)
            mappings.append({'episode_index':i,'original_receipt_path':archive,'new_receipt_sha256':merged.sha(directory/'receipt.json')})
        report={'contract':merged.CACHE_CONTRACT,'state':'complete','manifest_sha256':merged.sha(self.cache_root/'manifest.json'),
            'pack_ready_sha256':merged.sha(self.pack/'PACK_READY.json'),'episodes':mappings}
        self.cache.atomic_json(self.cache_root/'MERGED114_REINDEX.json',report)
        self.original_manifest_bytes=(self.cache_root/'manifest.json').read_bytes()
        self.original_feature_sha=merged.sha(self.cache_root/'episodes/00000000/features.npy')
        fake=SimpleNamespace(cache=self.cache,plan_source=lambda _: (self.source,self.items))
        self.patches=[patch.object(merged,'inspect',return_value=(self.pack,{},[])),patch.object(merged.dino,'extractor',return_value=fake)]
        for p in self.patches:p.start();self.addCleanup(p.stop)

    def test_private_audit_then_publish_preserves_features_and_original(self):
        def audit(pack,cache_root):
            self.assertEqual((self.cache_root/'manifest.json').read_bytes(),self.original_manifest_bytes)
            self.assertNotEqual(Path(cache_root),self.cache_root)
            self.assertEqual(merged.read(Path(cache_root)/'manifest.json')['source'],self.source)
        with patch.object(merged,'audit_cache',side_effect=audit):report=merged.relocate(self.pack,self.cache_root)
        self.assertEqual(report['status'],'passed')
        self.assertEqual(merged.read(self.cache_root/'manifest.json')['source'],self.source)
        self.assertEqual(merged.sha(self.cache_root/'episodes/00000000/features.npy'),self.original_feature_sha)
        backup=self.root/'dino.before-remote-relocation'
        self.assertEqual((backup/'manifest.json').read_bytes(),self.original_manifest_bytes)
        self.assertTrue((self.cache_root/'relocation_input_originals/episode_00000000.json').is_file())

    def test_modified_transfer_mtime_fails_before_any_publication(self):
        feature=self.cache_root/'episodes/00000000/features.npy';s=feature.stat()
        os.utime(feature,ns=(s.st_atime_ns,s.st_mtime_ns+1000000000))
        with patch.object(merged,'audit_cache') as audit:
            with self.assertRaises(ValueError):merged.relocate(self.pack,self.cache_root)
            audit.assert_not_called()
        self.assertEqual((self.cache_root/'manifest.json').read_bytes(),self.original_manifest_bytes)
        self.assertFalse((self.root/'dino.before-remote-relocation').exists())

    def test_failed_private_audit_does_not_publish_complete_cache(self):
        with patch.object(merged,'audit_cache',side_effect=ValueError('injected audit failure')):
            with self.assertRaisesRegex(ValueError,'injected audit failure'):merged.relocate(self.pack,self.cache_root)
        self.assertEqual((self.cache_root/'manifest.json').read_bytes(),self.original_manifest_bytes)
        self.assertFalse((self.root/'dino.before-remote-relocation').exists())
        partials=list(self.root.glob('.dino.remote-relocation-*'));self.assertEqual(len(partials),1)
        self.assertEqual(merged.read(partials[0]/'manifest.json')['state'],'failed_incomplete')


if __name__=='__main__':unittest.main()
