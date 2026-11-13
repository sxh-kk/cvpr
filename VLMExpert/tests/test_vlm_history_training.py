"""Real cache/video/processor input checks: no pretrained model or GPU required."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import AutoProcessor

from ImageProcess import SharedImageProcessor
from psi.config.transform import Psi0ModelTransform, DataTransform, FieldTransform
from psi.config.data_lerobot import LerobotDataConfig
from psi.config.transform_psi0_sonic import SonicRepackTransform
from psi.trainers.qwen3vl_mixin import PaddedCollatorForTogether
from test_unifolm_dino_history import history_modules, actual_wrapper, _cache_pack, _index, _Pack, IMAGE_KEY


@pytest.fixture(scope="module")
def processor():
    path=Path('/home/zxc/CVPR_Evaluation_runtime/resources/joint20k/qwen')
    if not path.is_dir():
        pytest.skip('Local CPU Qwen processor/tokenizer unavailable; do not download weights')
    return AutoProcessor.from_pretrained(path,local_files_only=True)


def color_model(frames=6,aug=False):
    return Psi0ModelTransform(vlm_history_frames=frames,motion_image_policy='vlm_color_clean_motion',
        shared_image_profile='256x320',img_aug=aug)


def model_data(frames=6):
    rng=np.random.default_rng(83)
    return dict(observations=[Image.fromarray(rng.integers(0,256,(48,64,3),dtype=np.uint8))for _ in range(frames)],
        instruction='pick up the object',motion_img_embs=np.arange(6*5*1024,dtype=np.float32).reshape(6,5,1024),
        actions=np.zeros((30,80),np.float32),raw_actions=np.zeros((30,80),np.float32),states=np.zeros((1,45),np.float32))


def test_real_config_strict_integer_history_only_and_requires_existing_cache_color_policy():
    assert Psi0ModelTransform().vlm_history_frames==1
    for value in (1,6):assert Psi0ModelTransform.model_validate({'vlm_history_frames':value}).vlm_history_frames==value
    for value in (True,False,6.,'6',1.5,0,2,5,7,None):
        with pytest.raises(ValueError):Psi0ModelTransform.model_validate({'vlm_history_frames':value})
    transform=DataTransform(repack=SonicRepackTransform(),field=FieldTransform(),model=color_model())
    with pytest.raises(ValueError,match='existing clean DINO cache'):
        LerobotDataConfig(root_dir='/not-read',train_repo_ids=['train'],transform=transform)
    config=LerobotDataConfig(root_dir='/not-read',train_repo_ids=['train'],dino_cache_root='/existing-cache',transform=transform)
    assert config.transform.repack.num_past_frames==0
    assert config.transform.model.vlm_history_frames==6
    assert config.transform.repack.delta_timestamps(30)['observation.state']==[0.]


def _wrapper(actual_wrapper,fixture,frames,*,decode=None):
    root,cache,_,original=fixture
    class Pack(_Pack):
        def __getitem__(self,index):
            self.item_calls.append(index)
            episode=int(self.hf_dataset['episode_index'][index]);frame=int(self.hf_dataset['frame_index'][index])
            if decode is None:image=torch.full((3,48,64),(frame+episode*80)/255,dtype=torch.float32)
            else:image=self._query_videos({IMAGE_KEY:[frame/30]},episode)[IMAGE_KEY]
            return {IMAGE_KEY:image,'native_index':index,'observation.state':torch.full((1,45),float(index)),
                'action':torch.arange(30*14,dtype=torch.float32).reshape(30,14),
                'action.body_token_v1_1':torch.arange(30*64,dtype=torch.float32).reshape(30,64),'task':'Pick up the object'}
        def _query_videos(self,queries,episode):
            self.video_calls.append((deepcopy(queries),episode))
            if decode is not None:return decode(self,queries,episode)
            return {IMAGE_KEY:torch.stack([torch.full((3,48,64),(round(t*30)+episode*80)/255,dtype=torch.float32)
                for t in queries[IMAGE_KEY]])}
    pack=Pack(**original.__dict__,root=root,item_calls=[],video_calls=[],video_backend='pyav',tolerance_s=.25/30)
    pack.meta.get_video_file_path=lambda episode,key:Path(f'videos/episode_{episode:06d}.mp4')
    cls,namespace=actual_wrapper
    namespace.update(LeRobotDatasetMetadata=lambda *args:pack.meta,LerobotRepackTransform=SonicRepackTransform,
        LeRobotDataset=lambda *args,**kwargs:pack,MultiLeRobotDataset=lambda *args,**kwargs:pack)
    repack=SonicRepackTransform(image_keys=[IMAGE_KEY],action_keys=['action.body_token_v1_1','action[:14]'],pad_action_dim=80)
    cfg=SimpleNamespace(root_dir=str(root.parent),train_repo_ids=[root.name],val_repo_ids=[root.name],dino_cache_root=str(cache),
        shared_image_profile='256x320',dino_max_open_episodes=2,transform=SimpleNamespace(repack=repack,
            model=SimpleNamespace(motion_image_policy='vlm_color_clean_motion',vlm_history_frames=frames)))
    return cls(cfg),pack,repack


def test_actual_wrapper_exact_motion_rows_and_same_episode_without_querying_historical_labels(history_modules,actual_wrapper,tmp_path):
    fixture=_cache_pack(tmp_path,history_modules.cache,lengths=(35,37))
    one,_,_=_wrapper(actual_wrapper,fixture,1);six,pack,_=_wrapper(actual_wrapper,fixture,6)
    assert len(one)==len(six)==12
    np.testing.assert_array_equal(one.motion_history.original_indices,six.motion_history.original_indices)
    for filtered,episode,current in ((4,0,34),(5,1,30)):
        before=list(pack.item_calls);sample=six[filtered]
        assert pack.item_calls==before+[int(six.motion_history.original_indices[filtered])]
        rows=six.motion_history.selected_cache_rows(filtered)
        np.testing.assert_array_equal(rows,np.arange(current-30,current+1,6))
        queries,actual_episode=pack.video_calls[-1]
        assert actual_episode==episode
        np.testing.assert_allclose(queries[IMAGE_KEY],rows/30,atol=1e-7)
        assert sample['vlm_history_metadata']['selected_cache_rows']==rows.tolist()
        assert sample['vlm_history_metadata']['episode_index']==episode
        assert sample['vlm_history_rgb'].shape==(6,3,48,64)
        np.testing.assert_array_equal(torch.round(sample['vlm_history_rgb'][:,0,0,0]*255).numpy(),rows+episode*80)
        np.testing.assert_array_equal(sample['motion_img_embs'][:,0,0],rows+episode*1000)
        assert sample['observation.state'].shape==(1,45)
    assert 'vlm_history_rgb'not in one[4]


def test_decoder_refuses_padding_missing_frames_and_wrong_current_identity(history_modules,actual_wrapper,tmp_path):
    fixture=_cache_pack(tmp_path,history_modules.cache)
    six,pack,_=_wrapper(actual_wrapper,fixture,6)
    original=pack._query_videos
    pack._query_videos=lambda q,e:{IMAGE_KEY:original(q,e)[IMAGE_KEY][:-1]}
    with pytest.raises(ValueError,match='six real floating RGB'):six[0]
    pack._query_videos=lambda q,e:{IMAGE_KEY:torch.zeros(6,3,48,64)}
    with pytest.raises(ValueError,match='current frame differs'):six[0]


def test_multi_pack_RGB_decode_follows_same_Motion_pack_identity(history_modules,tmp_path):
    a=_cache_pack(tmp_path/'a',history_modules.cache,lengths=(35,))
    b=_cache_pack(tmp_path/'b',history_modules.cache,lengths=(37,))
    root=tmp_path/'multi';root.mkdir();(root/'a').symlink_to(a[1],target_is_directory=True);(root/'b').symlink_to(b[1],target_is_directory=True)
    calls=[]
    class Pack:
        def __init__(self,source,tag):self.hf_dataset=source.hf_dataset;self.meta=source.meta;self.tag=tag
        def __len__(self):return len(self.hf_dataset)
        def _query_videos(self,q,ep):
            calls.append((self.tag,ep,q));return {IMAGE_KEY:torch.full((6,3,8,8),self.tag/255)}
    packs=[Pack(a[3],7),Pack(b[3],17)]
    index=history_modules.history.MotionHistoryIndex(root,packs,['a','b'],[a[0],b[0]],IMAGE_KEY)
    rgb,receipt=index.get_rgb_history(5,packs)
    assert calls[-1][0]==17 and receipt['pack_id']==1 and receipt['episode_index']==0
    assert torch.round(rgb[-1,0,0,0]*255)==17
    np.testing.assert_array_equal(index.get_history(5)[:,0,0],np.arange(0,31,6))


def test_actual_pyav_video_batch_uses_the_real_six_source_frames(history_modules,actual_wrapper,tmp_path):
    import av
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    root,cache,manifest,pack=_cache_pack(tmp_path,history_modules.cache,lengths=(35,))
    video=root/'videos/episode_000000.mp4'
    with av.open(str(video),'w')as output:
        stream=output.add_stream('mpeg4',rate=30);stream.width=64;stream.height=48;stream.pix_fmt='yuv420p'
        for row in range(35):
            rgb=np.full((48,64,3),row*5,dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(rgb,format='rgb24')):output.mux(packet)
        for packet in stream.encode():output.mux(packet)
    directory=history_modules.cache.episode_directory(cache,0)
    receipt=json.loads((directory/'receipt.json').read_text())
    receipt['source']['video']['identity']=history_modules.cache.file_identity(video)
    receipt['signature']=history_modules.cache.digest({'manifest_signature':manifest['signature'],'source':receipt['source']})
    history_modules.cache.atomic_json(directory/'receipt.json',receipt)
    before=(directory/'features.npy').read_bytes()
    six,actual_pack,_=_wrapper(actual_wrapper,(root,cache,manifest,pack),6,decode=LeRobotDataset._query_videos)
    sample=six[0]
    with av.open(str(video))as source:decoded=[frame.to_ndarray(format='rgb24')for frame in source.decode(video=0)]
    actual=torch.round(sample['vlm_history_rgb']*255).to(torch.uint8).permute(0,2,3,1).numpy()
    np.testing.assert_array_equal(actual,np.stack([decoded[i]for i in(0,6,12,18,24,30)]))
    np.testing.assert_array_equal(actual[-1],torch.round(sample[IMAGE_KEY]*255).to(torch.uint8).permute(1,2,0).numpy())
    assert (directory/'features.npy').read_bytes()==before
    assert len(actual_pack.item_calls)==1 and [len(q[IMAGE_KEY])for q,e in actual_pack.video_calls]==[1,6]


class KeepField(FieldTransform):
    def __call__(self,data,**kwargs):return {**data,'raw_actions':data['actions'].copy()}


def test_real_repack_field_transform_changes_only_RGB_not_state_action_or_Motion(history_modules,actual_wrapper,tmp_path,processor):
    six,_,repack=_wrapper(actual_wrapper,_cache_pack(tmp_path,history_modules.cache),6)
    raw=six[0];before={key:raw[key].clone()for key in('observation.state','action','action.body_token_v1_1')}
    history=raw['motion_img_embs'].copy();transform=DataTransform(repack=repack,field=KeepField(),model=color_model())
    output=transform(raw,vlm_processor=processor,no_aug=True)
    assert len(output['observations'])==6 and output['states'].shape==(1,45) and output['actions'].shape==(30,80)
    np.testing.assert_array_equal(output['states'],raw['observation.state'].numpy())
    np.testing.assert_array_equal(output['actions'][:,:64],raw['action.body_token_v1_1'].numpy())
    np.testing.assert_array_equal(output['actions'][:,64:78],raw['action'].numpy())
    np.testing.assert_array_equal(output['motion_img_embs'],history)
    assert output['vlm_history_metadata']['selected_cache_rows']==[0,6,12,18,24,30]
    assert output['image_grid_thw'].shape==(6,3)
    for key,value in before.items():torch.testing.assert_close(raw[key],value,rtol=0,atol=0)
    del raw['vlm_history_rgb']
    with pytest.raises(ValueError,match='six real RGB'):transform(raw,vlm_processor=processor,no_aug=True)


def test_color_is_one_shared_window_draw_and_Motion_cache_is_clean(monkeypatch,processor):
    data=model_data();history=data['motion_img_embs'].copy();model=color_model(aug=True)
    calls=[];real=SharedImageProcessor.process_window
    def spy(self,images,**kw):calls.append(kw);assert len(images)==6;return real(self,images,**kw)
    monkeypatch.setattr(SharedImageProcessor,'process_window',spy)
    outputs=[]
    for seed in (43,43,47):
        torch.manual_seed(seed);sample=model(data,vlm_processor=processor)
        expected=real(SharedImageProcessor('256x320'),data['observations'],**calls[-1])
        assert len(calls[-1])==3 and calls[-1]['training'] is True
        for a,b in zip(sample['observations'],expected.images):np.testing.assert_array_equal(np.asarray(a),b)
        np.testing.assert_array_equal(sample['motion_img_embs'],history);outputs.append(np.stack([np.asarray(v)for v in sample['observations']]))
    assert len(calls)==3 and calls[0]['seed']==calls[1]['seed']!=calls[2]['seed']
    np.testing.assert_array_equal(outputs[0],outputs[1]);assert not np.array_equal(outputs[0],outputs[2])
    torch.manual_seed(7);before=torch.random.get_rng_state().clone();model(data,vlm_processor=processor,no_aug=True)
    assert torch.equal(before,torch.random.get_rng_state())


def test_actual_Qwen_processor_six_ordered_images_and_collator_never_truncates_them(processor):
    data=model_data();sample=color_model()(data,vlm_processor=processor,no_aug=True)
    assert tuple(sample['image_grid_thw'].shape)==(6,3) and sample['vlm_image_token_count']==480
    assert sample['pixel_values'].shape[0]==1920
    maximum=sample['input_ids'].numel()
    collator=PaddedCollatorForTogether(model_max_length=maximum,pad_token_id=processor.tokenizer.pad_token_id)
    batch=collator([sample,sample]);assert batch['image_grid_thw'].shape==(12,3)
    assert batch['motion_img_embs'].shape==(2,6,5,1024) and batch['states'].shape==(2,1,45)
    with pytest.raises(ValueError,match='exceeds model_max_length'):
        PaddedCollatorForTogether(model_max_length=maximum-1,pad_token_id=processor.tokenizer.pad_token_id)([sample])
    for mutate in (lambda x:x['input_ids'].__setitem__((0,1),x['vlm_image_token_id']),
        lambda x:x.update(image_grid_thw=x['image_grid_thw'][:-1]),lambda x:x.update(pixel_values=x['pixel_values'][:-1]),
        lambda x:x.pop('motion_img_embs'),lambda x:x.update(vlm_history_frames=1)):
        bad=deepcopy(sample);mutate(bad)
        with pytest.raises(ValueError):collator([bad])
    one=color_model(1)(model_data(1),vlm_processor=processor,no_aug=True)
    with pytest.raises(ValueError,match='mix six-frame'):collator([sample,one])


def test_single_current_default_pixels_tokens_and_legacy_truncation_are_unchanged(processor):
    data=model_data(1)
    a=Psi0ModelTransform(motion_image_policy='vlm_color_clean_motion',shared_image_profile='256x320')(data,vlm_processor=processor,no_aug=True)
    b=color_model(1)(data,vlm_processor=processor,no_aug=True)
    assert set(a)==set(b) and 'vlm_history_frames'not in a
    for key in('input_ids','pixel_values','image_grid_thw'):torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
    expected=SharedImageProcessor('256x320').process_image(data['observations'][0]);np.testing.assert_array_equal(np.asarray(a['observations'][0]),expected)
    cap=a['input_ids'].numel()-1
    batch=PaddedCollatorForTogether(model_max_length=cap,pad_token_id=processor.tokenizer.pad_token_id)([a])
    torch.testing.assert_close(batch['input_ids'],a['input_ids'][:,:cap],rtol=0,atol=0)
    assert 'vlm_history_frames'not in batch
