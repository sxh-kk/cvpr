from types import SimpleNamespace
import numpy as np
import pytest
from PIL import Image
from egomotion_pipeline.features import DinoEncoder, extract_features
from egomotion_pipeline.schema import DINO_CHECKPOINT, DINO_SHAPE, dino_format
from egomotion_pipeline.storage import Archive, read_arrays
from egomotion_pipeline.cli import convert
from test_archive_and_video import FakeEncoder, source_config
from conftest import make_egosuite


def make_encoder(tmp_path, monkeypatch, **model_overrides):
    torch = pytest.importorskip('torch')
    transformers = pytest.importorskip('transformers')
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))
            self.config = SimpleNamespace(model_type='dinov2_with_registers', hidden_size=1024,
                                          num_register_tokens=4, patch_size=14, num_hidden_layers=24)
            for key, value in model_overrides.items():
                setattr(self.config, key, value)
        def forward(self, pixel_values):
            # Distinct CLS/register/patch values expose wrong token selection.
            hidden = torch.arange(7, device=pixel_values.device).view(1, 7, 1)
            return SimpleNamespace(last_hidden_state=hidden.expand(len(pixel_values), 7, 1024))
    processor = transformers.BitImageProcessor(size={'shortest_edge':256}, crop_size={'height':224,'width':224})
    monkeypatch.setattr(transformers.AutoImageProcessor, 'from_pretrained', lambda *a, **k: processor)
    monkeypatch.setattr(transformers.AutoModel, 'from_pretrained', lambda *a, **k: Model())
    (tmp_path/'config.json').write_text('{}')
    return DinoEncoder({'checkpoint':str(tmp_path), 'revision':'local', 'device':'cpu'})


@pytest.mark.parametrize('shape', [(23,38), (38,23)])
def test_encoder_register_order_and_uniegomotion_preprocessing(tmp_path, monkeypatch, shape):
    encoder = make_encoder(tmp_path, monkeypatch)
    image = np.random.default_rng(42).integers(0, 256, (*shape, 3), dtype=np.uint8)
    h, w = shape
    size = (int(w*336/min(h,w)), int(h*336/min(h,w)))
    resized = np.asarray(Image.fromarray(image).resize(size, Image.Resampling.BICUBIC)).astype(np.float32)/255
    expected = (resized-np.array([.485,.456,.406]))/np.array([.229,.224,.225])
    pixels = encoder.processor(images=[image], return_tensors='pt')['pixel_values'].numpy()[0]
    assert min(pixels.shape[1:]) == 336 and max(pixels.shape[1:]) > 336
    np.testing.assert_allclose(pixels, expected.transpose(2,0,1), atol=1e-6)
    output = encoder([image])
    assert output.shape == (1, *DINO_SHAPE) and output.dtype == np.float32
    np.testing.assert_array_equal(output[0,:,0], np.arange(5))
    for key in ['shape','flat_shape','token_order','flatten_order','pooling']:
        assert encoder.metadata[key] == dino_format()[key]
    assert not encoder.metadata['processor']['do_center_crop']
    assert encoder.metadata['local_file_sha256']['config.json']


@pytest.mark.parametrize('override', [{'model_type':'dinov2'}, {'num_register_tokens':0},
                                      {'hidden_size':768}, {'num_hidden_layers':12}])
def test_encoder_rejects_non_large_or_missing_registers(tmp_path, monkeypatch, override):
    with pytest.raises(ValueError, match=r'CLS \+ 4 registers'):
        make_encoder(tmp_path, monkeypatch, **override)


def test_real_hf_register_model_prefix_cpu():
    torch = pytest.importorskip('torch')
    transformers = pytest.importorskip('transformers')
    config = transformers.Dinov2WithRegistersConfig(hidden_size=32, num_hidden_layers=1,
        num_attention_heads=4, intermediate_size=64, image_size=28, patch_size=14, num_register_tokens=4)
    model = transformers.Dinov2WithRegistersModel(config).eval()
    encoder = DinoEncoder.__new__(DinoEncoder)
    encoder.model, encoder.torch, encoder.device = model, torch, 'cpu'
    pixels = torch.randn(2,3,28,28)
    encoder.processor = lambda **kwargs: {'pixel_values':pixels}
    with torch.inference_mode():
        all_tokens = model(pixel_values=pixels).last_hidden_state.numpy()
    assert all_tokens.shape == (2, 9, 32)
    np.testing.assert_allclose(encoder([None, None]), all_tokens[:, :5], atol=0, rtol=0)


@pytest.mark.parametrize('kind', ['cls_only', 'nonfinite'])
def test_extraction_rejects_invalid_encoder_output_without_committing(tmp_path, kind):
    source = make_egosuite(tmp_path/'source', episodes=1)
    convert(source_config(source), tmp_path/'archive')
    archive = Archive(tmp_path/'archive')
    class Broken(FakeEncoder):
        def __call__(self, images):
            return (np.zeros((len(images), 1024), np.float32) if kind == 'cls_only'
                    else np.full((len(images), 5, 1024), np.nan, np.float32))
    with pytest.raises(ValueError, match='batch, 5, 1024'):
        extract_features(archive, {}, Broken())
    assert Archive(archive.root).records()[0]['dino']['state'] == 'pending'
    assert not read_arrays(archive.data_path(0), archive.specs)['observation.valid.dino'].any()


def test_default_register_checkpoint():
    assert DINO_CHECKPOINT == 'facebook/dinov2-with-registers-large'
