import json
from concurrent.futures import Future
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from egomotion_pipeline.smplh_features import plan_extraction, run_extraction
from egomotion_pipeline.video import indexed_video_frames
from current_fixtures import Encoder, make_source, write_video


@pytest.mark.parametrize('kind',['egosuite','egoexo4d_hm'])
def test_native30hz_cache_uses_original_video_and_preserves_source(tmp_path,kind):
    root,record,motion = make_source(tmp_path,kind)
    # Retired original embeddings are not needed: provenance + motion joins suffice.
    assert not (root/record['vision_path']).exists()
    original = {p:p.read_bytes() for p in root.rglob('*') if p.is_file()}
    output = tmp_path/'profile'
    encoder = Encoder()
    result = run_extraction(root,output,{'image_profile':'256x480','batch_size':2},encoder=encoder)
    selected=np.arange(13)
    validity=np.ones(13,dtype=bool); validity[6]=False
    assert result['state']=='complete' and result['completed_frames']==len(selected)
    directory=output/'episodes/00000000'
    feature=np.load(directory/'features.npy',mmap_mode='r')
    assert feature.shape==(len(selected),5,1024) and feature.dtype==np.float32
    with np.load(directory/'join.npz') as join:
        np.testing.assert_array_equal(join['frame_index'],selected)
        np.testing.assert_array_equal(join['observation.valid.dino'],validity)
        np.testing.assert_array_equal(join['source.video_frame_index'],selected if kind=='egosuite' else selected+2)
        assert abs(join['observation.video_timestamp'][0]-(0 if kind=='egosuite' else 2/30))<1e-6
    assert np.all(feature[6]==0)
    assert result['validity_equal_source'] is True
    assert all(im.shape==(256,480,3) for im in encoder.seen)
    assert all(p.read_bytes()==value for p,value in original.items())
    encoder.seen.clear()
    assert run_extraction(root,output,{'image_profile':'256x480','batch_size':2},encoder=encoder)['state']=='complete'
    assert not encoder.seen
    with pytest.raises(ValueError,match='different source/configuration'):
        run_extraction(root,output,{'image_profile':'256x320'},encoder=Encoder('256x320'))


def test_dry_run_does_not_load_encoder_or_create_output(tmp_path):
    root,_,_=make_source(tmp_path,'egoexo4d_hm')
    output=tmp_path/'profile'
    result=run_extraction(root,output,{'image_profile':'256x480'},dry_run=True)
    assert result['state']=='ready' and result['total_frames']==13 and result['fps']==30
    assert not output.exists()
    with pytest.raises(ValueError,match='source archive'):
        run_extraction(root,root,{},dry_run=True)
    with pytest.raises(ValueError,match='native30Hz'):
        run_extraction(root,output,{},fps=5,dry_run=True)


def test_default_native30hz_cache_keeps_every_frame_and_source_mask(tmp_path):
    root,_,_=make_source(tmp_path,'egoexo4d_hm')
    output=tmp_path/'native30'
    result=run_extraction(root,output,{},encoder=Encoder())
    assert result['fps']==30 and result['sampling_step']==1
    assert result['completed_frames']==13
    values=np.load(output/'episodes/00000000/features.npy',mmap_mode='r')
    assert values.shape==(13,5,1024)
    with np.load(output/'episodes/00000000/join.npz') as join:
        np.testing.assert_array_equal(join['frame_index'],np.arange(13))
        np.testing.assert_array_equal(join['source.video_frame_index'],np.arange(13)+2)
        assert join['observation.valid.dino'].sum()==12 and not join['observation.valid.dino'][6]


def test_changed_committed_cache_is_not_silently_reused(tmp_path):
    root,_,_=make_source(tmp_path,'egoexo4d_hm')
    output=tmp_path/'profile'
    run_extraction(root,output,{},encoder=Encoder())
    path=output/'episodes/00000000/features.npy'
    with path.open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError,match='file changed'):
        run_extraction(root,output,{},encoder=Encoder())


def test_wrong_encoder_resume_preserves_complete_manifest(tmp_path):
    root,_,_=make_source(tmp_path,'egoexo4d_hm')
    output=tmp_path/'profile'
    run_extraction(root,output,{},encoder=Encoder())
    original=(output/'manifest.json').read_bytes()
    with pytest.raises(ValueError,match='different source/configuration'):
        run_extraction(root,output,{'revision':'different'},encoder=Encoder())
    assert (output/'manifest.json').read_bytes()==original


def test_replaced_video_does_not_reuse_certified_mapping(tmp_path):
    root,record,_=make_source(tmp_path,'egoexo4d_hm')
    path=Path(record['dino']['source_extraction']['video']['video_path'])
    with path.open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError,match='Original video identity'):
        plan_extraction(root,{})


def test_changed_motion_or_unpublished_source_is_rejected(tmp_path):
    root, _, motion = make_source(tmp_path)
    state = root / ".pipeline/dataset-state.json"
    state.write_text(json.dumps({"state": "accepted"}))
    with pytest.raises(ValueError, match="accepted SMPL-H"):
        plan_extraction(root, {})
    state.write_text(json.dumps({"state": "accepted", "independent_acceptance_pending": False}))
    with motion.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="motion identity"):
        plan_extraction(root, {})


def test_decoder_missing_valid_frame_never_commits_training_cache(tmp_path, monkeypatch):
    import egomotion_pipeline.smplh_features as module
    root, _, _ = make_source(tmp_path)
    original = module.indexed_video_frames
    def missing(*args, **kwargs):
        for index, (image, pts) in enumerate(original(*args, **kwargs)):
            yield (None, pts) if index == 0 else (image, pts)
    monkeypatch.setattr(module, "indexed_video_frames", missing)
    output = tmp_path / "profile"
    with pytest.raises(ValueError, match="validity differs from source"):
        run_extraction(root, output, {}, encoder=Encoder())
    assert json.loads((output / "manifest.json").read_text())["state"] == "failed"
    assert not (output / "episodes/00000000/receipt.json").exists()
    assert not list(output.rglob("*.tmp"))
    monkeypatch.setattr(module, "indexed_video_frames", original)
    assert run_extraction(root, output, {}, encoder=Encoder())["state"] == "complete"


def test_exact_frame_decoder_preserves_nonzero_frame_mapping(tmp_path):
    path=write_video(tmp_path/'source.mp4',range(15))
    result=list(indexed_video_frames(path,np.array([2,8,14],np.int64)))
    np.testing.assert_allclose([pts for _,pts in result],np.array([2,8,14])/30)


def test_disk_space_failure_with_only_lock_file_can_be_retried(tmp_path,monkeypatch):
    import egomotion_pipeline.smplh_features as module
    from types import SimpleNamespace
    root,_,_=make_source(tmp_path,'egoexo4d_hm')
    output=tmp_path/'profile'
    original=module.shutil.disk_usage
    monkeypatch.setattr(module.shutil,'disk_usage',lambda _:SimpleNamespace(free=0))
    with pytest.raises(OSError,match='disk space'):
        run_extraction(root,output,{},encoder=Encoder())
    assert (output/'.extraction.lock').exists()
    monkeypatch.setattr(module.shutil,'disk_usage',original)
    assert run_extraction(root,output,{},encoder=Encoder())['state']=='complete'


def test_encoding_failure_cleans_temporary_arrays_and_can_resume(tmp_path):
    root,_,motion=make_source(tmp_path)
    output=tmp_path/'profile'
    original=motion.read_bytes()
    class Broken(Encoder):
        calls=0
        def encode_prepared(self,images):
            self.calls+=1
            if self.calls==2:
                raise RuntimeError('injected encoding failure')
            return super().encode_prepared(images)
    config={'batch_size':2,'prefetch_batches':2}
    with pytest.raises(RuntimeError,match='injected encoding failure'):
        run_extraction(root,output,config,encoder=Broken())
    directory=output/'episodes/00000000'
    assert not (directory/'receipt.json').exists()
    assert not (directory/'features.npy.tmp').exists()
    assert not (directory/'join.npz.tmp').exists()
    assert json.loads((output/'manifest.json').read_text())['state']=='failed'
    resumed=run_extraction(root,output,config,encoder=Encoder())
    assert resumed['state']=='complete' and resumed['completed_frames']==13
    assert motion.read_bytes()==original


def test_worker_failure_terminates_own_pool_and_records_failure(tmp_path,monkeypatch):
    import egomotion_pipeline.smplh_features as module
    root,_,_=make_source(tmp_path,'egoexo4d_hm')
    output=tmp_path/'profile'
    events=[]
    class Process:
        alive=True
        def is_alive(self): return self.alive
        def terminate(self): events.append('terminate'); self.alive=False
        def join(self,timeout): events.append('join')
    class Pool:
        def __init__(self,**kwargs): self._processes={1:Process()}
        def submit(self,*args):
            future=Future(); future.set_exception(RuntimeError('injected worker failure')); return future
        def shutdown(self,**kwargs): events.append('shutdown')
    monkeypatch.setattr(module,'ProcessPoolExecutor',Pool)
    with pytest.raises(RuntimeError,match='injected worker failure'):
        run_extraction(root,output,{},gpu_ids=[0])
    assert events==['terminate','join','shutdown']
    assert json.loads((output/'manifest.json').read_text())['state']=='failed'
