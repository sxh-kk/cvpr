import numpy as np
import pytest

from ImageProcess import SharedImageProcessor, load_profile
from current_fixtures import make_encoder


@pytest.mark.parametrize('profile', ['256x480', '256x320'])
def test_hf_and_parallel_use_shared_geometry_and_identical_normalization(tmp_path, monkeypatch, profile):
    hf = make_encoder(tmp_path, monkeypatch, image_profile=profile)
    parallel = make_encoder(tmp_path, monkeypatch, image_profile=profile, preprocessor='pil_parallel')
    images = [np.random.default_rng(12).integers(0,256,(73,96,3),dtype=np.uint8),
              np.random.default_rng(13).integers(0,256,(81,81,3),dtype=np.uint8)]
    shared = SharedImageProcessor(load_profile(profile))
    rgb = [shared.process_image(image) for image in images]
    expected = ((np.asarray(rgb).astype(np.float64)/255).astype(np.float32)-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
    try:
        left, right = hf.prepare(images)['pixel_values'], parallel.prepare(images)['pixel_values']
        assert tuple(left.shape) == (2,3,*load_profile(profile).output_hw)
        np.testing.assert_array_equal(left.numpy(), right.numpy())
        np.testing.assert_array_equal(left.numpy(), expected.transpose(0,3,1,2))
        assert not hf.processor.do_resize and not hf.processor.do_center_crop
        assert hf.metadata['image_process'] == shared.metadata
        assert hf(images).shape == (2,5,1024)
    finally:
        hf.close(); parallel.close()


@pytest.mark.parametrize('stage,profile', [('posttrain','256x320'),('robust','256x480')])
def test_window_rgb_is_shared_replayable_and_never_resized_twice(tmp_path, monkeypatch, stage, profile):
    encoder = make_encoder(tmp_path, monkeypatch, image_profile=profile, preprocessor='pil_parallel')
    image = np.random.default_rng(15).integers(0,256,(64,96,3),dtype=np.uint8)
    captured = []
    original = encoder.encode_prepared
    def capture(inputs):
        captured.append(inputs['pixel_values'].numpy().copy())
        return original(inputs)
    encoder.encode_prepared = capture
    try:
        out = encoder.encode_window([image]*6,stage=stage,seed=41)
        repeat = encoder.encode_window([image]*6,stage=stage,augmentation=out['augmentation'])
        assert out['img_embs'].shape == (6,5,1024)
        np.testing.assert_array_equal(out['rgb'],repeat['rgb'])
        for rgb in out['rgb'][1:]:
            np.testing.assert_array_equal(rgb,out['rgb'][0])
        expected = encoder.prepare(out['rgb'],processed=True)['pixel_values'].numpy()
        np.testing.assert_array_equal(captured[0],expected)
        with pytest.raises(ValueError,match='dimensions'):
            encoder.prepare([image],processed=True)
    finally:
        encoder.close()


def test_offline_and_inference_disable_random_augmentation(tmp_path, monkeypatch):
    encoder = make_encoder(tmp_path,monkeypatch,image_profile='256x480')
    image = np.random.default_rng(18).integers(0,256,(32,48,3),dtype=np.uint8)
    try:
        original = encoder.shared_processor.process_image(image)
        result = encoder.encode_window([image],stage='robust',training=False,seed=2)
        np.testing.assert_array_equal(result['rgb'][0],original)
        assert result['img_embs'].dtype == np.float32
    finally:
        encoder.close()


@pytest.mark.parametrize('initial_profile', ['256x480', '256x320'])
def test_one_frozen_dino_switches_stage_geometry_without_reloading_weights(tmp_path, monkeypatch, initial_profile):
    encoder=make_encoder(tmp_path,monkeypatch,image_profile=initial_profile,preprocessor='pil_parallel')
    model=encoder.model
    image=np.random.default_rng(22).integers(0,256,(48,64,3),dtype=np.uint8)
    try:
        post=encoder.encode_window([image]*6,stage='posttrain',training=False)
        robust=encoder.encode_window([image]*6,stage='robust',training=False)
        pretrain=encoder.encode_window([image]*6,stage='pretrain')
        assert post['rgb'][0].shape==(256,320,3)
        assert robust['rgb'][0].shape==(256,480,3)
        assert pretrain['rgb'][0].shape==(256,480,3)
        assert post['image_process']['profile_id']=='psi0_256x320_v1'
        assert robust['image_process']['profile_id']=='psi0_256x480_v1'
        assert encoder.model is model
        assert not any(p.requires_grad for p in encoder.model.parameters())
        assert post['img_embs'].shape==robust['img_embs'].shape==(6,5,1024)
    finally:
        encoder.close()
