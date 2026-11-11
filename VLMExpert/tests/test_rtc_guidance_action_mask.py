"""CPU math and real serving routes; no pretrained model or GPU acceptance claim."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from psi.models.dual_source_action import rtc_guidance_error
from psi.deploy.serve_psi0_simple import (
    SimpleServerConfig, validated_repack_guidance, _PICK_MASK_PACKER_SHA256,
)
from test_dual_source_serving import make_server, observe_episode, wire
from test_vlm_features import build_psi0_harness, _VLM, _Processor, _LegacyHeader, _Scheduler


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, sort_keys=True) + "\n").encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest(), len(raw)


def _proof_fixture(root, *, profile="xmove_pick"):
    task = {"xmove_pick": "cvpr/G1SonicXMovePickPico-v0",
            "bend_pick": "cvpr/G1SonicBendPickPico-v0",
            "xmove_bend_pick": "cvpr/G1SonicXMoveBendPickPico-v0"}[profile]
    config = dict(model=dict(action_dim=80, action_architecture="dual_source_v1"),
        data=dict(transform=dict(repack=dict(action_keys=["action.body_token_v1_1", "action[:14]"],
            pad_action_dim=80, action_mask_key="action_mask"))))
    config_sha, size = _write(root / "run_config.json", config)
    pack = dict(contract="cvpr-pico-pick-lerobot-pack-v1", task_id=task,
        packer_sha256=_PICK_MASK_PACKER_SHA256, checkpoint_contract=dict(hand_joint_names=[
            side + "_hand_" + name + "_joint" for side in ("left", "right")
            for name in ("thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1")]))
    name = "release_metadata/packed/PACK_READY.json"
    pack_sha, _ = _write(root / "provenance" / name, pack)
    ready = dict(contract="cvpr-pick-local-deployment-v1", status="local_pick_checkpoint_prepared",
        profile=profile, task_id=task, training_completed=True, checkpoint_save_completed=True,
        component_load_validation=dict(status="passed"),
        deployment_artifacts={"run_config.json": dict(sha256=config_sha, bytes=size)},
        source_metadata_sha256={name: pack_sha})
    _write(root / "LOCAL_RUN_READY.json", ready)
    return config, pack, ready


def _vjp(previous, sample, matrix, mask):
    predicted = sample @ matrix
    error = rtc_guidance_error(previous, predicted, torch.ones(1, 3, 1), mask)
    correction = torch.autograd.grad(predicted, sample, grad_outputs=error)[0]
    return error, correction


def test_padding_only_residual_cannot_guide_valid_outputs_through_cross_dimension_jacobian():
    sample = torch.randn(1, 3, 80, requires_grad=True)
    matrix = torch.randn(80, 80)
    previous = (sample @ matrix).detach().clone()
    previous[..., 78:] += 1000
    mask = torch.tensor([True] * 78 + [False] * 2)
    error, correction = _vjp(previous, sample, matrix, mask)
    assert error.count_nonzero() == correction.count_nonzero() == 0
    legacy_error, legacy_correction = _vjp(previous, sample, matrix, None)
    assert legacy_error.norm() > 0 and legacy_correction[..., :78].norm() > 0


def test_masked_padding_target_has_no_effect_on_error_norm_or_full_input_vjp():
    sample = torch.randn(1, 3, 80, requires_grad=True)
    matrix = torch.randn(80, 80)
    previous = torch.randn(1, 3, 80)
    mask = torch.tensor([True] * 78 + [False] * 2)
    expected = _vjp(previous, sample, matrix, mask)
    previous[..., 78:] = float("nan")
    actual = _vjp(previous, sample, matrix, mask)
    for a, b in zip(expected, actual):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert actual[1][..., 78:].abs().sum() > 0  # input channels remain part of the Jacobian


def test_default_and_all_true_mask_match_original_residual_exactly():
    previous, predicted = torch.randn(1, 3, 80), torch.randn(1, 3, 80)
    temporal = torch.rand(1, 3, 1)
    expected = (previous - predicted.detach()) * temporal
    for mask in (None, torch.ones(80, dtype=torch.bool)):
        torch.testing.assert_close(rtc_guidance_error(previous, predicted, temporal, mask), expected, rtol=0, atol=0)


@pytest.mark.parametrize("mask", [torch.ones(80), torch.zeros(80, dtype=torch.bool),
    torch.ones(1, 80, dtype=torch.bool), torch.ones(78, dtype=torch.bool), [True] * 80])
def test_bad_guidance_masks_are_rejected(mask):
    with pytest.raises(ValueError, match="boolean"):
        rtc_guidance_error(torch.zeros(1, 3, 80), torch.zeros(1, 3, 80), torch.ones(1, 3, 1), mask)


@pytest.mark.filterwarnings("ignore:User provided device_type of 'cuda'.*:UserWarning")
def test_actual_sampler_preserves_full_gaussian_padding_and_80d_shape():
    model = build_psi0_harness()()
    model.device, model.action_horizon, model.action_dim = torch.device("cpu"), 4, 80
    model.vlm_model, model.vlm_processor = _VLM(), _Processor()
    model.action_header, model.noise_scheduler = _LegacyHeader(), _Scheduler()
    model.vlm_layer_indices, model.action_architecture, model.uses_dual_source = None, "psi0_joint", False
    args = dict(observations=[[object()]], states=torch.zeros(1, 1, 3), instructions=["1 2 3"],
        num_inference_steps=3, traj2ds=None, prev_actions=torch.full((1, 4, 80), .3),
        inference_delay=0, execution_horizon=2)
    torch.manual_seed(91)
    expected_noise = torch.randn(1, 4, 80)
    torch.manual_seed(91)
    legacy = model.predict_action_with_rtc_flow(**args)
    legacy_noise = model.action_header.calls[0]["joint_attention_kwargs"]["action_hidden_embeds"].detach().clone()
    torch.manual_seed(91)
    all_true = model.predict_action_with_rtc_flow(**args, guidance_action_mask=torch.ones(80, dtype=torch.bool))
    torch.testing.assert_close(all_true, legacy, rtol=0, atol=0)
    model.action_header.calls.clear()
    torch.manual_seed(91)
    result = model.predict_action_with_rtc_flow(**args, guidance_action_mask=torch.tensor([True]*78+[False]*2))
    actual_noise = model.action_header.calls[0]["joint_attention_kwargs"]["action_hidden_embeds"].detach()
    torch.testing.assert_close(actual_noise, expected_noise, rtol=0, atol=0)
    torch.testing.assert_close(legacy_noise, actual_noise, rtol=0, atol=0)
    assert actual_noise[..., 78:].count_nonzero() == 8
    assert result.shape == (1, 4, 80) and torch.isfinite(result).all()


@pytest.mark.parametrize("profile", ["xmove_pick", "bend_pick", "xmove_bend_pick"])
def test_only_authenticated_pack_repack_contract_enables_78d_guidance(tmp_path, profile):
    _proof_fixture(tmp_path, profile=profile)
    mask, proof = validated_repack_guidance(tmp_path, 80)
    assert mask.tolist() == [True]*78+[False]*2
    assert proof["guided_action_dimensions"] == 78 and proof["excluded_action_indices"] == [78, 79]
    assert SimpleServerConfig(run_dir=str(tmp_path)).rtc_guidance_action_scope == "all"


@pytest.mark.parametrize("bad", ["config_sha", "pack_sha", "packer", "task", "repack", "hand_order", "acceptance"])
def test_unknown_or_changed_pack_cannot_self_authorize_a_guessed_78_mask(tmp_path, bad):
    config, pack, ready = _proof_fixture(tmp_path)
    if bad in ("config_sha", "repack"):
        config["data"]["transform"]["repack"]["action_keys"] = ["action"]
        sha, size = _write(tmp_path / "run_config.json", config)
        if bad == "repack":
            ready["deployment_artifacts"]["run_config.json"] = dict(sha256=sha, bytes=size)
    elif bad in ("pack_sha", "packer", "task", "hand_order"):
        if bad == "task": pack["task_id"] = "cvpr/Other-v0"
        elif bad == "hand_order": pack["checkpoint_contract"]["hand_joint_names"].reverse()
        else: pack["packer_sha256"] = "0"*64
        sha, _ = _write(tmp_path / "provenance/release_metadata/packed/PACK_READY.json", pack)
        if bad != "pack_sha": ready["source_metadata_sha256"]["release_metadata/packed/PACK_READY.json"] = sha
    else:
        ready["component_load_validation"]["status"] = "failed"
    _write(tmp_path / "LOCAL_RUN_READY.json", ready)
    with pytest.raises(ValueError, match="RTC validated_repack"):
        validated_repack_guidance(tmp_path, 80)


def _audit_module():
    path = Path(__file__).parents[2] / "scripts/4090_simple/serve_evaluation.py"
    spec = importlib.util.spec_from_file_location("rtc_audit_wrapper", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_real_http_handler_passes_mask_and_audit_records_actual_inputs_outputs(tmp_path):
    module = _audit_module()
    server = make_server()
    server.Ta, server.exec_shift, server.min_exec_horizon = 15, 15, 15
    server.rtc_guidance_action_scope = "validated_repack"
    server.rtc_guidance_action_mask = torch.tensor([True]*78+[False]*2)
    server.rtc_guidance_action_proof = dict(scope="validated_repack", guidance_action_mask=[True]*78+[False]*2)
    server.model.parameters = lambda: ()
    server.online_dino.encoder = type("DinoReceipt", (), {"metadata": {"cpu_double": True}})()
    module.audited_server(lambda: server, 91, torch.manual_seed, tmp_path, "tiny")
    with TestClient(server.create_app()) as client:
        observe_episode(client)
        assert client.post("/act", json=wire(1.)).status_code == 200
        for i in range(1, 16):
            assert client.post("/observe", json=wire(1. + i/30)).status_code == 200
        assert client.post("/act", json=wire(1.5)).status_code == 200
        info = client.get("/info").json()
    assert server.model.calls[-1][1]["guidance_action_mask"].tolist() == [True]*78+[False]*2
    startup = json.loads((tmp_path / "startup.json").read_text())
    assert info["rtc_guidance_action_proof"] == startup["rtc_guidance_action_proof"]
    rows = [json.loads(line) for line in (tmp_path / "requests.jsonl").read_text().splitlines()]
    assert rows[-1]["actual_rtc_guidance_action_mask"] == [True]*78+[False]*2
    assert rows[-1]["normalized_shifted_previous"][0][:15] == [[1.]*80]*15
    assert rows[-1]["normalized_shifted_previous"][0][15:] == [[0.]*80]*15
    assert rows[-1]["normalized_prediction"] == [[[2.]*80]*30]
    assert rows[-1]["denormalized_returned_action"] == [[12.]*80]*15
    assert rows[-1]["sampling_path"] == "test_time"


def test_wrapper_rejects_mask_mismatch_before_sampling(tmp_path):
    module = _audit_module()
    server = make_server()
    server.rtc_guidance_action_proof = dict(scope="validated_repack", guidance_action_mask=[True]*78+[False]*2)
    server.model.parameters = lambda: ()
    server.online_dino.encoder = type("DinoReceipt", (), {"metadata": {}})()
    module.audited_server(lambda: server, 1, torch.manual_seed, tmp_path, "tiny")
    with pytest.raises(RuntimeError, match="differs from accepted"):
        server.model.predict_action_with_rtc_flow(guidance_action_mask=None)
    assert server.model.calls == []
