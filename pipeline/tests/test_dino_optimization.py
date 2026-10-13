import threading
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import pytest

from egomotion_pipeline.dino_optimization import (
    ParallelPILPreprocessor, optimization_options, prepared_batches)
from egomotion_pipeline.features import extract_features
from egomotion_pipeline.storage import read_arrays, write_arrays
from egomotion_pipeline.video import nearest_mcap_video_frames, nearest_video_frames
from conftest import write_video
from mcap_fixtures import make_mcap
from test_archive_and_video import FakeEncoder
from test_mcap import read
from test_parallel_features import make_archive, ParallelFakeEncoder
from egomotion_pipeline.parallel_features import extract_features_parallel


@pytest.mark.parametrize('shape', [(23,38), (38,23), (336,336), (1456,1920), (900,1200)])
def test_parallel_pil_pixels_exactly_match_hf_reference(shape):
    transformers = pytest.importorskip('transformers')
    from egomotion_pipeline.features import DINO_PREPROCESSING
    processor = transformers.BitImageProcessor(**DINO_PREPROCESSING)
    images = [np.random.default_rng(seed).integers(0,256,(*shape,3),dtype=np.uint8)
              for seed in [7,8]]
    expected = processor(images=images,return_tensors='np')['pixel_values']
    parallel = ParallelPILPreprocessor(2)
    try:
        np.testing.assert_array_equal(parallel(images), expected)
        rotated = [np.rot90(image) for image in images]
        np.testing.assert_array_equal(parallel(rotated),
            processor(images=rotated,return_tensors='np')['pixel_values'])
    finally:
        parallel.close()


@pytest.mark.parametrize('image', [np.zeros((3,4,3),np.float32),
                                   np.zeros((3,4),np.uint8),np.zeros((3,4,4),np.uint8)])
def test_parallel_pil_rejects_non_rgb_uint8(image):
    processor = ParallelPILPreprocessor(1)
    try:
        with pytest.raises(ValueError,match='RGB uint8'):
            processor([image])
    finally:
        processor.close()


def test_prefetch_is_ordered_bounded_and_cleans_up_on_early_close():
    closed = threading.Event()
    full = threading.Event()
    produced = []
    def source():
        try:
            for index in range(1000):
                yield [index], [index], [index/30]
        finally:
            closed.set()
    def prepare(images):
        produced.append(images[0])
        if len(produced) == 4:
            full.set()
        return np.asarray(images)
    batches = prepared_batches(source(),prepare,2)
    first = next(batches)
    assert first[1] == [0]
    assert full.wait(timeout=2)
    assert len(produced) <= 4
    batches.close()
    assert closed.is_set()
    assert not any(t.name == 'dino-prefetch' for t in threading.enumerate())


@pytest.mark.parametrize('depth', [0,2])
def test_prefetch_propagates_source_and_prepare_errors_and_closes(depth):
    for origin in ['source','prepare']:
        closed = []
        def source():
            try:
                for index in range(4):
                    if origin == 'source' and index == 2:
                        raise ValueError('injected source failure')
                    yield [index], [index], [index/30]
            finally:
                closed.append(True)
        def prepare(images):
            if origin == 'prepare' and images[0] == 2:
                raise ValueError('injected prepare failure')
            return images
        with pytest.raises(ValueError,match='injected '+origin):
            list(prepared_batches(source(),prepare,depth))
        assert closed == [True]
        assert not any(t.name == 'dino-prefetch' for t in threading.enumerate())


@pytest.mark.parametrize('threads',[1,2,4])
def test_threaded_decoders_preserve_pixels_pts_and_bad_frame_masks(tmp_path,threads):
    path = make_mcap(tmp_path/'video.mcap',video_indices=[0,1,3,4],bad={'image':[1]})
    episode = read(path)
    mp4 = write_video(tmp_path/'gap.mp4',[0,1,3,4])
    for video, function in [(episode.video,nearest_mcap_video_frames),(mp4,nearest_video_frames)]:
        expected = list(function(video,np.arange(5)/30,1/60+.001))
        actual = list(function(video,np.arange(5)/30,1/60+.001,decoder_threads=threads))
        for (image,pts),(ref,ref_pts) in zip(actual,expected):
            assert pts == ref_pts
            if ref is None:
                assert image is None
            else:
                np.testing.assert_array_equal(image,ref)


def test_optimized_parquet_is_lossless_and_uses_float_byte_stream_split(tmp_path):
    archive,_ = make_archive(tmp_path,episodes=1)
    arrays = read_arrays(archive.data_path(0),archive.specs)
    arrays['observation.features.dino'][:] = np.random.default_rng(7).standard_normal(
        arrays['observation.features.dino'].shape).astype(np.float32)
    write_arrays(archive.data_path(0),arrays,archive.specs,optimized_dino=True)
    actual = read_arrays(archive.data_path(0),archive.specs)
    for key in arrays:
        np.testing.assert_array_equal(actual[key],arrays[key])
    parquet = pq.ParquetFile(archive.data_path(0))
    columns = [parquet.metadata.row_group(0).column(i) for i in range(parquet.metadata.num_columns)]
    column = next(c for c in columns if c.path_in_schema == 'observation.features.dino.list.element')
    assert 'BYTE_STREAM_SPLIT' in column.encodings
    assert 'RLE_DICTIONARY' not in column.encodings


def test_early_mcap_close_drains_threaded_decoder(tmp_path,monkeypatch):
    import egomotion_pipeline.video as video_module
    native_av = video_module.av
    drained = []
    class Decoder:
        def __init__(self,*args):
            object.__setattr__(self,'codec',native_av.CodecContext.create(*args))
        def __setattr__(self,key,value):
            setattr(self.codec,key,value)
        def decode(self,packet):
            if packet is None:
                drained.append(True)
            return self.codec.decode(packet)
    path = make_mcap(tmp_path/'partial.mcap')
    episode = read(path)
    monkeypatch.setattr(video_module,'av',SimpleNamespace(
        CodecContext=SimpleNamespace(create=Decoder),Packet=native_av.Packet,FFmpegError=native_av.FFmpegError))
    frames = nearest_mcap_video_frames(episode.video,[0.],.0176666667,decoder_threads=4)
    assert next(frames)[1] == 0
    frames.close()
    assert drained == [True]


def test_prefetch_parallel_extraction_matches_serial_and_rejects_precision_changes(tmp_path):
    archive,_ = make_archive(tmp_path,episodes=3)
    before = {i:read_arrays(archive.data_path(i),archive.specs) for i in range(3)}
    config = {'batch_size':2,'prefetch_batches':2,'optimized_storage':True,'inference_precision':'bf16'}
    with archive.lock():
        extract_features_parallel(archive,config,[0,1],encoder_factory=ParallelFakeEncoder)
    # Rebuild pending files from the saved original arrays for serial comparison.
    expected,_ = make_archive(tmp_path/'serial',episodes=3)
    extract_features(expected,config,FakeEncoder())
    for i in range(3):
        result = read_arrays(archive.data_path(i),archive.specs)
        reference = read_arrays(expected.data_path(i),expected.specs)
        for key in result:
            np.testing.assert_array_equal(result[key],reference[key])
            if key not in {'observation.features.dino','observation.valid.dino',
                           'observation.video_timestamp','observation.video_time_error'}:
                np.testing.assert_array_equal(result[key],before[i][key])
    with archive.lock(),pytest.raises(ValueError,match='different model'):
        extract_features_parallel(archive,{**config,'inference_precision':'tf32'},[0],
                                  encoder_factory=ParallelFakeEncoder)


@pytest.mark.parametrize('config',[{'inference_precision':'fp16'},{'prefetch_batches':-1},
    {'preprocess_workers':False},{'pin_memory':1},{'decoder_threads':1.5},
    {'preprocessor':'gpu_resize'},{'attention_implementation':'unknown'}])
def test_invalid_optimization_options_fail_early(config):
    with pytest.raises(ValueError):
        optimization_options(config)
