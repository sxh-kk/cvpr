"""CPU fixtures for independent human packing/normalization/relocation gates."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import verify_pico_handover as verify


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def pack_fixture(root):
    from pack_pico_handover import BASE_CHECKS,AUDITOR_FILES
    pack = root/'packed';pack.mkdir()
    human_uuid = str(uuid.uuid4());original = '/local/source/kept/001'
    source = {'source_kind':'live_pico_human','source_namespace':'live_pico_human',
        'human_demo_uuid':human_uuid,'source_session':'human_test'}
    episode = {'episode_index':0,'length':61,'frames':61,'human_demo_uuid':human_uuid,'candidate_id':human_uuid,
        'source_kind':'live_pico_human','source_episode_index':1_000_000,'source_identity':source,
        'lineage_complete':True,'split':'train','split_restriction':'train_only','eligible_current_rows':[30,31]}
    prefix = f'accepted_evidence/{human_uuid}'
    meta_relative = prefix+'/run/metadata.json'
    write(pack/meta_relative,{'source_kind':'live_pico_human','human_demo_uuid':human_uuid,'replay_source_sha256':None})
    checks = {name:{'passed':True} for name in BASE_CHECKS}
    authorities = {};archived = {}
    for name in AUDITOR_FILES:
        relative = prefix+'/auditor_source/'+name
        (pack/relative).parent.mkdir(parents=True,exist_ok=True)
        (pack/relative).write_bytes(b'unit fixture auditor source authority')
        authorities['/source/'+name] = verify.sha(pack/relative)
        archived['/source/'+name] = {'path':relative,'sha256':verify.sha(pack/relative)}
    admission = {'contract':'cvpr-live-pico-handover-independent-admission-v1',
        'status':'independent_live_human_episode_audit_passed','human_demo_uuid':human_uuid,
        'eligible_for_packing':True,'accepted_new_dataset':False,'ckpt_feature_pack_ready':False,
        'errors':[],'checks':checks,'bindings':{original+'/metadata.json':verify.sha(pack/meta_relative)},
        'auditor_source_sha256':authorities}
    admission_relative = prefix+'/admission.json';write(pack/admission_relative,admission)
    parquet = 'train/data/chunk-000/episode_000000.parquet'
    video = 'train/videos/chunk-000/observation.images.egocentric/episode_000000.mp4'
    for relative in (parquet,video):
        path = pack/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'not real data; envelope unit fixture only')
    meta = pack/'train/meta/episodes.jsonl';meta.parent.mkdir(parents=True,exist_ok=True)
    meta.write_text(json.dumps(episode)+'\n')
    for name in verify.SOURCE_FILES:write(pack/name,{'frozen':'source bytes'})
    entry = {**episode,'original_episode_directory':original,'admission_path':admission_relative,
        'admission_sha256':verify.sha(pack/admission_relative),'evidence_files':{'metadata.json':meta_relative},
        'auditor_source_files':archived,'parquet_path':parquet,'parquet_sha256':verify.sha(pack/parquet),
        'video':{'path':video,'sha256':verify.sha(pack/video),'frames':61,'fps':30,'pixels_exact':True}}
    ready = {'contract':verify.PACK_CONTRACT,'train_payload_packed':True,'final_ready':False,
        'normalization_refitted':False,'train_episodes':1,'total_frames':61,'train_windows':2,'episodes':[entry]}
    write(pack/'selection.json',{'contract':'cvpr-pico-handover-admissions-selection-v1',
        'episodes':[{'human_demo_uuid':human_uuid,'admission_sha256':entry['admission_sha256']}]})
    ready['selection'] = {'path':'selection.json','sha256':verify.sha(pack/'selection.json')}
    freeze(pack,ready)
    return pack,ready


def freeze(pack,ready):
    ready['payload_sha256'] = {str(p.relative_to(pack)):verify.sha(p) for p in pack.rglob('*')
        if p.is_file() and p.name!='PACK_READY.json'}
    write(pack/'PACK_READY.json',ready)


def cache_and_transfer_fixture(root,pack,ready):
    verify.shared.runtime_paths()
    from curate_handover_dino_cache import extractor
    ex = extractor();cache = ex.cache
    video_key = 'observation.images.egocentric'
    length = ready['episodes'][0]['length']
    protocol = verify.production_protocol(ready.get('task_id',verify.HANDOVER_TASK_ID))
    write(pack/'train/meta/info.json',{'fps':30,'total_episodes':1,'total_frames':length,
        'features':{video_key:{'dtype':'video','video_info':{'video.fps':30}}},
        'data_path':'data/chunk-000/episode_{episode_index:06d}.parquet',
        'video_path':'videos/chunk-000/{video_key}/episode_{episode_index:06d}.mp4'})
    freeze(pack,ready)
    source,items = ex.plan_source(pack/'train')
    source['dataset_root'] = '/different-original-host/release/packed/train'
    from ImageProcess import SharedImageProcessor
    profile = SharedImageProcessor('256x320').metadata
    encoder = {'revision':'local','local_file_sha256':{'model.safetensors':'a'*64},'image_process':profile,
        'preprocessing':{'do_resize':False,'do_center_crop':False,'do_rescale':True,'do_normalize':True,
            'image_mean':[.485,.456,.406],'image_std':[.229,.224,.225],'rescale_factor':1/255}}
    manifest = ex.build_manifest(source,items,encoder,storage_dtype='float32');manifest['state'] = 'complete'
    dino = root/'dino';directory = dino/'episodes/00000000';directory.mkdir(parents=True)
    np.save(directory/'features.npy',np.full((length,5,1024),.125,np.float32))
    np.savez(directory/'frames.npz',frame_index=np.arange(length,dtype=np.int64),
        timestamp=np.arange(length,dtype=np.float64)/30,source_frame_index=np.arange(length,dtype=np.int64),
        video_timestamp=np.arange(length,dtype=np.float64)/30,valid=np.ones(length,bool))
    receipt = {'contract':cache.CONTRACT,'state':'complete','episode_index':0,'frames':length,
        'source':items[0],'manifest_signature':manifest['signature'],
        'signature':cache.digest({'manifest_signature':manifest['signature'],'source':items[0]}),
        'outputs':{name:verify.identity(directory/name) for name in ('features.npy','frames.npz')}}
    write(directory/'receipt.json',receipt);write(dino/'manifest.json',manifest)
    payload = {name+'/'+relative:identity for name in ('packed','dino')
        for relative,identity in verify.tree_inventory(root/name).items()}
    local = {'contract':protocol['production_contract'],'task_id':protocol['task_id'],
        'status':'passed','production_ready_for_bound_pack_and_cache':True,
        'source_kind':'live_pico_human','split_restriction':'train_only','payload_files':payload}
    write(root/verify.READY_NAME,local)
    transfer = {'contract':protocol['transfer_contract'],'task_id':protocol['task_id'],'remote_root':str(root),
        'local_ready_sha256':verify.sha(root/verify.READY_NAME),
        'files':{**payload,verify.READY_NAME:verify.identity(root/verify.READY_NAME)}}
    write(root/'A100_TRANSFER_MANIFEST.json',transfer)
    write(root/'A100_BYTE_VERIFICATION.json',{'contract':protocol['byte_contract'],'task_id':protocol['task_id'],
        'status':'all_copied_bytes_and_identities_verified','remote_root':str(root),
        'inventory_sha256':verify.sha(root/'A100_TRANSFER_MANIFEST.json'),'remote_production_ready':False})
    return dino,manifest,receipt


class PicoVerificationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_human_envelope_uses_uuid_train_only_without_legacy_count_or_ancestors(self):
        pack,ready = pack_fixture(self.root)
        _,_,episodes,_ = verify.inspect_pack(pack,expected_episodes=1)
        self.assertEqual(episodes[0]['source_episode_index'],1_000_000)
        with self.assertRaises(ValueError):verify.inspect_pack(pack,expected_episodes=100)
        self.assertEqual(verify.read(pack/'PACK_READY.json'),ready)

    def test_rejects_synthetic_or_foreign_namespace_even_after_fresh_sha(self):
        pack,ready = pack_fixture(self.root)
        path = pack/'train/meta/episodes.jsonl'
        episode = json.loads(path.read_text())
        episode['source_identity']['source_namespace'] = 'official_teacher'
        path.write_text(json.dumps(episode)+'\n');freeze(pack,ready)
        with self.assertRaisesRegex(ValueError,'human UUID lineage'):verify.inspect_pack(pack)

    def test_rejects_missing_admission_check_and_changed_evidence(self):
        pack,ready = pack_fixture(self.root)
        entry = ready['episodes'][0];path = pack/entry['admission_path']
        value = verify.read(path);value['checks'] = {'only_one_check':{'passed':True}}
        write(path,value);entry['admission_sha256'] = verify.sha(path);freeze(pack,ready)
        with self.assertRaisesRegex(ValueError,'mandatory checks'):verify.inspect_pack(pack)

    def test_frozen_selection_cannot_be_replaced_by_another_valid_human_set(self):
        pack,ready = pack_fixture(self.root)
        selected = verify.read(pack/'selection.json')
        selected['episodes'][0]['human_demo_uuid'] = str(uuid.uuid4())
        write(pack/'selection.json',selected)
        ready['selection']['sha256'] = verify.sha(pack/'selection.json');freeze(pack,ready)
        with self.assertRaisesRegex(ValueError,'Frozen selected human'):verify.inspect_pack(pack)

    def test_sha_coverage_and_symlink_escape_fail_closed(self):
        pack,ready = pack_fixture(self.root)
        (pack/'unbound.txt').write_text('extra')
        with self.assertRaisesRegex(ValueError,'coverage'):verify.inspect_pack(pack)
        (pack/'unbound.txt').unlink()
        (pack/'escape').symlink_to(self.root,target_is_directory=True)
        with self.assertRaisesRegex(ValueError,'aliases'):verify.inspect_pack(pack)
        with self.assertRaises(ValueError):verify.owned(pack,'../outside')
        with self.assertRaises(ValueError):verify.owned(pack,None)

    def test_real_label_layout_keeps_warmup_unsupervised_and_tail_valid(self):
        length = 70
        raw = {'action_valid':np.r_[np.zeros(30,bool),np.ones(length-30,bool)],
            'action':np.r_[np.full((30,14),np.nan),np.zeros((length-30,14))],
            'action.body_token_v1_1':np.r_[np.full((30,64),np.nan),np.zeros((length-30,64))],
            'observation.state':np.zeros((length,43)),
            'action_mask':np.r_[np.zeros((30,80)),np.c_[np.ones((length-30,78)),np.zeros((length-30,2))]]}
        verify.validate_labels(raw,length)
        for key,position,value in (('action',(0,0),0),('action_mask',(35,79),1),('action_valid',35,False)):
            changed = deepcopy(raw);changed[key][position] = value
            with self.assertRaises(ValueError):verify.validate_labels(changed,length)

    def test_numerical_sample_detects_channel_order_future_shift_and_state_noise(self):
        rng = np.random.default_rng(22)
        raw = {'action':rng.normal(size=(100,14)).astype(np.float32),
            'action.body_token_v1_1':rng.normal(size=(100,64)).astype(np.float32),
            'observation.state':rng.normal(size=(100,43)).astype(np.float32)}
        field = SimpleNamespace(action_min=np.full(80,-2),action_max=np.full(80,2),
            state_min=np.full(45,-3),state_max=np.full(45,3),normalize_state=True)
        actions = np.pad(np.c_[raw['action.body_token_v1_1'][30:60],raw['action'][30:60]],((0,0),(0,2)))
        states = np.pad(raw['observation.state'][30:31],((0,0),(0,2)))
        sample = {'raw_actions':actions,'actions':verify.normalized(actions,field.action_min,field.action_max),
            'states':verify.normalized(states,field.state_min,field.state_max,state=True)}
        verify.validate_numerical_sample(sample,raw,30,field)
        for key in ('raw_actions','actions','states'):
            changed = deepcopy(sample);changed[key].flat[0] += .001
            with self.assertRaises(ValueError):verify.validate_numerical_sample(changed,raw,30,field)
        with self.assertRaises(ValueError):verify.validate_numerical_sample(sample,raw,31,field)

    def test_constant_padded_bounds_match_production_conventions(self):
        values = np.array([[.2,.5,5]],np.float32)
        low = [0,1,2];high = [1,1,2]
        np.testing.assert_array_equal(verify.normalized(values,low,high),np.array([[-.6,.5,1]],np.float32))
        np.testing.assert_array_equal(verify.normalized(values,low,high,state=True),np.array([[-.6,0,0]],np.float32))

    def test_relocation_refuses_absent_byte_receipt_without_loading_gpu_or_modifying_cache(self):
        pack,_ = pack_fixture(self.root)
        dino = self.root/'dino';dino.mkdir();(dino/'manifest.json').write_text('{}')
        with self.assertRaises(ValueError):verify.relocate(pack,dino)
        self.assertEqual((dino/'manifest.json').read_text(),'{}')
        self.assertFalse((self.root/'dino.before-remote-relocation').exists())

    def test_real_cpu_cache_relocation_keeps_incoming_tree_and_feature_bytes(self):
        pack,ready = pack_fixture(self.root)
        dino,manifest,receipt = cache_and_transfer_fixture(self.root,pack,ready)
        original = verify.tree_inventory(dino)
        original_pack = verify.tree_inventory(pack)
        report = verify.relocate(pack,dino)
        self.assertTrue(report['data_and_features_byte_identical'])
        self.assertFalse(report['remote_production_ready'])
        self.assertEqual(verify.tree_inventory(self.root/'dino.before-remote-relocation'),original)
        self.assertEqual(verify.tree_inventory(pack),original_pack)
        changed = verify.read(dino/'manifest.json')
        self.assertEqual(changed['source']['dataset_root'],str(pack/'train'))
        self.assertNotEqual(changed['signature'],manifest['signature'])
        from curate_handover_dino_cache import extractor
        cache = extractor().cache
        cache.load_cache_manifest(dino,pack/'train')
        loaded = cache.load_episode_cache(dino,0,dataset_root=pack/'train',verify_checksums=True)
        np.testing.assert_array_equal(loaded.features,.125)
        loaded.features._mmap.close()
        for name in ('features.npy','frames.npz'):
            self.assertEqual(verify.identity(dino/'episodes/00000000'/name),receipt['outputs'][name])
        self.assertFalse((self.root/'A100_PICO_DATASET_READY.json').exists())
        with self.assertRaises(ValueError):verify.relocate(pack,dino)

    def test_relocation_refuses_changed_feature_even_if_byte_report_was_passed(self):
        pack,ready = pack_fixture(self.root)
        dino,_,_ = cache_and_transfer_fixture(self.root,pack,ready)
        path = dino/'episodes/00000000/features.npy'
        path.write_bytes(b'changed after byte verification')
        with self.assertRaisesRegex(ValueError,'Copied immutable input changed'):verify.relocate(pack,dino)
        self.assertFalse((self.root/'dino.before-remote-relocation').exists())


    def mobile_pack(self, root=None):
        from test_pack_pico_handover import make_mobile_case
        import pack_pico_handover as packer
        root = root or self.root
        case = make_mobile_case(root)
        ready = packer.pack(case['template'],case['selection'],case['output'])
        return case['output'],ready

    def test_mobile_pack_requires_explicit_task_and_its_own_admission_authority(self):
        pack,ready = self.mobile_pack()
        _,_,episodes,_ = verify.inspect_pack(pack,expected_episodes=1)
        self.assertEqual(episodes[0]['task_id'],verify.MOBILE_TASK_ID)
        changed = deepcopy(ready);changed.pop('task_id');freeze(pack,changed)
        with self.assertRaisesRegex(ValueError,'Task identity'):
            verify.inspect_pack(pack)
        freeze(pack,ready)
        admission_path=pack/ready['episodes'][0]['admission_path']
        admission=verify.read(admission_path)
        admission['contract']='cvpr-live-pico-handover-independent-admission-v1'
        write(admission_path,admission)
        ready['episodes'][0]['admission_sha256']=verify.sha(admission_path)
        freeze(pack,ready)
        with self.assertRaisesRegex(ValueError,'admission'):
            verify.inspect_pack(pack)

    def test_mobile_shared_original_trace_and_capture_sources_cannot_be_rebound_after_packing(self):
        for role in ('trace','historical_source'):
            with self.subTest(role=role):
                root=self.root/role;root.mkdir()
                pack,ready=self.mobile_pack(root)
                entry=ready['episodes'][0]
                archived=(entry['mobile_control_trace_file'] if role=='trace' else
                          entry['mobile_historical_source_files']['files'][0])
                (pack/archived['path']).write_bytes(b'changed original evidence after audit')
                freeze(pack,ready)
                with self.assertRaisesRegex(ValueError,'shared original'):
                    verify.inspect_pack(pack)

    def test_mobile_cpu_cache_relocation_preserves_portable_evidence_and_real_array_bytes(self):
        pack,ready=self.mobile_pack()
        dino,manifest,receipt=cache_and_transfer_fixture(self.root,pack,ready)
        original_cache,original_pack=verify.tree_inventory(dino),verify.tree_inventory(pack)
        report=verify.relocate(pack,dino)
        self.assertEqual(report['contract'],'cvpr-pico-mobile-pp-dino-host-relocation-v1')
        self.assertEqual(report['task_id'],verify.MOBILE_TASK_ID)
        self.assertTrue(report['data_and_features_byte_identical'])
        self.assertFalse(report['remote_production_ready'])
        self.assertEqual(verify.tree_inventory(pack),original_pack)
        self.assertEqual(verify.tree_inventory(self.root/'dino.before-remote-relocation'),original_cache)
        for name in ('features.npy','frames.npz'):
            self.assertEqual(verify.identity(dino/'episodes/00000000'/name),receipt['outputs'][name])
        from curate_handover_dino_cache import extractor
        cache=extractor().cache
        loaded=cache.load_episode_cache(dino,0,dataset_root=pack/'train',verify_checksums=True)
        self.assertEqual(loaded.features.shape,(65,5,1024))
        np.testing.assert_array_equal(loaded.features,.125)
        loaded.features._mmap.close()
        self.assertFalse((self.root/'A100_PICO_DATASET_READY.json').exists())

    def test_mobile_incoming_byte_receipts_cannot_use_handover_contract_or_missing_task(self):
        pack,ready=self.mobile_pack()
        cache_and_transfer_fixture(self.root,pack,ready)
        verify.incoming_transfer_binding(self.root)
        path=self.root/'A100_BYTE_VERIFICATION.json'
        original=verify.read(path)
        changed=deepcopy(original);changed['contract']='cvpr-pico-handover-a100-byte-verification-v1'
        write(path,changed)
        with self.assertRaisesRegex(ValueError,'byte verification'):
            verify.incoming_transfer_binding(self.root)
        changed=deepcopy(original);changed.pop('task_id');write(path,changed)
        with self.assertRaisesRegex(ValueError,'task identities'):
            verify.incoming_transfer_binding(self.root)


if __name__=='__main__':unittest.main()
