"""Evaluate held-out, preprocessed VLM samples and export features for linear probes.

Each JSONL record has id, images (paths relative to the manifest), instruction,
and normalized_actions (the exact normalized FAST input, shape [horizon, dim]).
Images must have the same deterministic resize/crop as the pretraining transform.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from psi.models.global_vlm import (
    build_vlm_inputs, select_global_features, validate_vlm_tokens,
)
from psi.tokenizer.fast_action_tokenizer import FastActionTokenizer
from psi.trainers.pretrain_checkpoint import validate_checkpoint_metadata


def probe_features(train, validation, ridge=1.0):
    if ridge <= 0:
        raise ValueError("Probe regularization must be positive")
    if set(train["ids"]) & set(validation["ids"]):
        raise ValueError("Probe train and validation sample IDs overlap")
    scores = {}
    targets = train["targets"].double()
    for name in ("global", "mean", "last"):
        if name not in train or name not in validation:
            continue
        x = train[name].double()
        mean, scale = x.mean(0), x.std(0, unbiased=False).clamp_min(1e-6)

        def design(values):
            values = (values.double() - mean) / scale
            return torch.cat([values, torch.ones(len(values), 1)], dim=1)

        design_train = design(x)
        penalty = torch.eye(design_train.shape[1], dtype=torch.float64) * ridge
        penalty[-1, -1] = 0
        weight = torch.linalg.solve(design_train.T @ design_train + penalty, design_train.T @ targets)
        expected = validation["targets"].double()

        def error(values):
            return (design(values) @ weight - expected).abs().mean().item()

        values = validation[name]
        scores[f"{name}_normalized_action_mae"] = error(values)
        if name == "global":
            scores["global_zeroed_mae"] = error(torch.zeros_like(values))
            if len(values) > 1:
                scores["global_swapped_mae"] = error(values.roll(1, dims=0))
    return scores


@torch.inference_mode()
def evaluate_samples(model, processor, action_tokenizer, records, *, max_new_tokens=256):
    model.eval()
    global_id = validate_vlm_tokens(model, processor)
    fast_ids = set(model.config.fast_token_ids)
    nll = supervised = correct = action_count = 0
    tf_errors, generated_errors, causal_errors = [], [], []
    invalid_tf = invalid_generation = decode_failures = 0
    features = {"ids": [], "mean": [], "last": [], "targets": []}
    if global_id is not None:
        features["global"] = []
    device = next(model.parameters()).device

    def decode_error(ids, target):
        nonlocal decode_failures
        try:
            prediction = np.asarray(action_tokenizer.decode_token_ids_to_actions([ids]))[0]
            if prediction.shape != target.shape or not np.isfinite(prediction).all():
                raise ValueError("Invalid decoded action shape or values")
            return float(np.abs(prediction - target).mean())
        except (ValueError, IndexError, RuntimeError):
            decode_failures += 1
            return None

    for record in records:
        target = np.asarray(record["normalized_actions"], dtype=np.float32)
        if target.ndim != 2 or not np.isfinite(target).all():
            raise ValueError(f"Invalid normalized_actions for sample {record['id']}")
        with_images = [record["images"]]
        instructions = [record["instruction"]]
        answer = action_tokenizer(target)
        full = build_vlm_inputs(processor, with_images, instructions, global_token_id=global_id,
                                action_texts=[answer],
                                max_instruction_tokens=model.config.max_instruction_tokens).to(device)
        output = model(**full, output_hidden_states=True, use_cache=False)
        logits, labels = output.logits[:, :-1].float(), full["labels"][:, 1:]
        nll += F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1),
                               ignore_index=-100, reduction="sum").item()
        supervised += labels.ne(-100).sum().item()
        action_mask = (labels >= model.config.fast_token_ids[0]) & (labels <= model.config.fast_token_ids[-1])
        predicted = logits.argmax(-1)
        correct += (predicted[action_mask] == labels[action_mask]).sum().item()
        action_count += action_mask.sum().item()
        predicted_ids = predicted[action_mask].tolist()
        if any(token not in fast_ids for token in predicted_ids):
            invalid_tf += 1
        else:
            error = decode_error(predicted_ids, target)
            if error is not None:
                tf_errors.append(error)

        # Reuse the observation boundary, excluding all teacher-forced answer tokens.
        first_answer = full["labels"][0].ne(-100).nonzero()[0].item()
        prefix = {key: value for key, value in full.items() if key not in ("labels", "input_ids", "attention_mask")}
        prefix["input_ids"] = full["input_ids"][:, :first_answer]
        prefix["attention_mask"] = full["attention_mask"][:, :first_answer]
        observation = model(**prefix, output_hidden_states=True, logits_to_keep=1,
                            use_cache=False, return_dict=True).hidden_states[-1]
        ordinary = observation[0]
        if global_id is not None:
            selected = select_global_features(observation, prefix["input_ids"], prefix["attention_mask"], global_id)
            trained = select_global_features(output.hidden_states[-1], full["input_ids"], full["attention_mask"], global_id)
            causal_errors.append((selected.global_hidden - trained.global_hidden).abs().max().item())
            features["global"].append(selected.global_hidden[0].float().cpu())
            ordinary = observation[0, selected.ordinary_mask[0]]
        features["mean"].append(ordinary.mean(0).float().cpu())
        features["last"].append(ordinary[-1].float().cpu())
        features["targets"].append(torch.from_numpy(target.reshape(-1).copy()))
        features["ids"].append(record["id"])
        del output, logits, observation

        if max_new_tokens:
            generated = model.generate(**prefix, max_new_tokens=max_new_tokens, do_sample=False,
                                       pad_token_id=processor.tokenizer.pad_token_id,
                                       eos_token_id=processor.tokenizer.eos_token_id, use_cache=True)
            generated_ids = generated[0, first_answer:].tolist()
            ended = bool(generated_ids and generated_ids[-1] == processor.tokenizer.eos_token_id)
            if ended:
                generated_ids.pop()
            if not ended or not generated_ids or any(token not in fast_ids for token in generated_ids):
                invalid_generation += 1
            else:
                error = decode_error(generated_ids, target)
                if error is not None:
                    generated_errors.append(error)

    if not features["ids"]:
        raise ValueError("Evaluation manifest is empty")
    if len(set(features["ids"])) != len(features["ids"]):
        raise ValueError("Evaluation sample IDs must be unique")
    features = {key: value if key == "ids" else torch.stack(value) for key, value in features.items()}
    metrics = {
        "samples": len(records),
        "teacher_forcing_cross_entropy": nll / max(supervised, 1),
        "teacher_forcing_action_accuracy": correct / max(action_count, 1),
        "teacher_forcing_normalized_action_mae": float(np.mean(tf_errors)) if tf_errors else None,
        "teacher_forcing_valid_decodes": len(tf_errors),
        "teacher_forcing_invalid_samples": invalid_tf,
        "generation_enabled": bool(max_new_tokens),
        "generation_normalized_action_mae": float(np.mean(generated_errors)) if generated_errors else None,
        "generation_valid_decodes": len(generated_errors),
        "generation_invalid_samples": invalid_generation,
        "decode_failures": decode_failures,
        "global_prefix_max_abs_error": max(causal_errors) if causal_errors else None,
    }
    return metrics, features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("samples", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--features-output", type=Path)
    parser.add_argument("--probe-train", type=Path, help="Features from a disjoint training split for linear probes")
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--fast-checkpoint", help="Override the FAST processor path recorded in training metadata")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=256, help="0 disables generation evaluation")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    args = parser.parse_args()
    if args.max_samples <= 0 or args.max_new_tokens < 0:
        parser.error("max-samples must be positive and max-new-tokens must be nonnegative")
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(args.checkpoint, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.checkpoint, local_files_only=True, dtype=getattr(torch, args.dtype), attn_implementation="sdpa",
    ).to(args.device).eval()
    metadata = validate_checkpoint_metadata(args.checkpoint, model, processor)
    validate_vlm_tokens(model, processor)
    fast_checkpoint = args.fast_checkpoint or metadata["run_config"]["model"]["action_tokenizer"]["pretrained_checkpoint"]
    records = []
    with args.samples.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            images = []
            for path in record["images"]:
                with Image.open(args.samples.parent / path) as image:
                    images.append(image.convert("RGB"))
            record["images"] = images
            records.append(record)
            if len(records) >= args.max_samples:
                break
    if not records:
        parser.error("samples manifest is empty")
    horizon, action_dim = np.asarray(records[0]["normalized_actions"]).shape
    action_tokenizer = FastActionTokenizer(processor.tokenizer, horizon, action_dim,
                                            bins=len(model.config.fast_token_ids), pretrained_checkpoint=fast_checkpoint)
    validate_vlm_tokens(model, processor)
    metrics, features = evaluate_samples(model, processor, action_tokenizer, records,
                                         max_new_tokens=args.max_new_tokens)
    if args.probe_train:
        metrics["linear_probes"] = probe_features(
            torch.load(args.probe_train, map_location="cpu", weights_only=True), features, args.ridge,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, allow_nan=False)
    if args.features_output:
        args.features_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(features, args.features_output)
    print(json.dumps(metrics, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
