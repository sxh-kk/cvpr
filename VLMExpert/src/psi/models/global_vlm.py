"""Shared vocabulary, observation inputs and feature extraction for PSI0 VLMs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoProcessor, BatchFeature, Qwen3VLForConditionalGeneration

from psi.tokenizer.fast_action_tokenizer import FastActionTokenizer


def validate_vlm_tokens(model, processor, *, require_global=False):
    tokenizer = processor.tokenizer
    config = model.config
    rows = model.get_input_embeddings().weight.shape[0]
    if (len(tokenizer) > rows or model.get_output_embeddings().weight.shape[0] != rows
            or config.text_config.vocab_size != rows):
        raise ValueError("Tokenizer and VLM embedding sizes do not match")
    fast_ids = getattr(config, "fast_token_ids", None)
    if fast_ids is not None:
        actual = tokenizer.convert_tokens_to_ids([f"<|a_{i}|>" for i in range(len(fast_ids))])
        if actual != fast_ids or fast_ids != list(range(fast_ids[0], fast_ids[0] + len(fast_ids))):
            raise ValueError("Saved FAST token mapping does not match the tokenizer")
    enabled = getattr(config, "use_global_token", False)
    if require_global and not enabled:
        raise ValueError("This stage requires a checkpoint already trained with GLOBAL")
    if enabled:
        token = config.global_token
        token_id = config.global_token_id
        if (token not in tokenizer.get_vocab()
                or tokenizer.encode(token, add_special_tokens=False) != [token_id]
                or token_id not in tokenizer.all_special_ids
                or token_id in (tokenizer.pad_token_id, tokenizer.eos_token_id)
                or token_id in (fast_ids or [])
                or token_id in [getattr(config, name, None) for name in (
                    "image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id")]):
            raise ValueError("GLOBAL token metadata does not match the tokenizer")
    return getattr(config, "global_token_id", None) if enabled else None


def initialize_vlm_tokens(model, processor, *, bins, use_global_token=False,
                          global_token="<GLOBAL>", global_token_id=None,
                          model_max_length=8192, max_instruction_tokens=512,
                          require_global_checkpoint=False):
    tokenizer = processor.tokenizer
    if model_max_length <= 0:
        raise ValueError("model_max_length must be positive")
    if max_instruction_tokens < 0:
        raise ValueError("max_instruction_tokens must be nonnegative")
    saved_global = getattr(model.config, "use_global_token", False)
    if saved_global or require_global_checkpoint:
        validate_vlm_tokens(model, processor, require_global=True)
        if not use_global_token or global_token != model.config.global_token:
            raise ValueError("GLOBAL configuration differs from the loaded checkpoint")
        if len(model.config.fast_token_ids) != bins:
            raise ValueError("FAST vocabulary size differs from the loaded checkpoint")
        if global_token_id is not None and global_token_id != model.config.global_token_id:
            raise ValueError("GLOBAL token ID differs from the loaded checkpoint")
    elif use_global_token and global_token in tokenizer.get_vocab():
        raise ValueError("GLOBAL is already registered but its model metadata is missing")

    original_vocab_size = len(tokenizer)
    original_special_ids = [i for i in tokenizer.all_special_ids if i < original_vocab_size]
    if original_vocab_size == len(original_special_ids):
        raise ValueError("At least one ordinary token is needed to initialize GLOBAL")
    fast_ids = FastActionTokenizer.register_tokens(tokenizer, bins)
    if use_global_token:
        tokenizer.add_special_tokens(
            {"additional_special_tokens": [global_token]},
            replace_additional_special_tokens=False,
        )
        actual_id = tokenizer.convert_tokens_to_ids(global_token)
        if global_token_id is not None and actual_id != global_token_id:
            raise ValueError("Requested GLOBAL token ID does not match registration order")
    else:
        actual_id = None

    rows = model.get_input_embeddings().weight.shape[0]
    target_rows = ((max(rows, len(tokenizer)) + 191) // 192) * 192
    if target_rows != rows:
        model.resize_token_embeddings(target_rows, mean_resizing=True)
    if use_global_token and not saved_global:
        with torch.no_grad():
            weight = model.get_input_embeddings().weight
            total = weight[:original_vocab_size].sum(dim=0, dtype=torch.float32)
            total -= weight[original_special_ids].sum(dim=0, dtype=torch.float32)
            mean = total / (original_vocab_size - len(original_special_ids))
            weight[actual_id].copy_(mean.to(weight.dtype))
            output_weight = model.get_output_embeddings().weight
            if output_weight is not weight:
                output_weight[actual_id].copy_(mean.to(output_weight.dtype))
    model.tie_weights()
    model.config.use_global_token = use_global_token
    model.config.global_token = global_token
    model.config.global_token_id = actual_id
    model.config.fast_token_ids = fast_ids
    model.config.model_max_length = model_max_length
    model.config.max_instruction_tokens = max_instruction_tokens
    tokenizer.model_max_length = model_max_length
    tokenizer.padding_side = "right"
    validate_vlm_tokens(model, processor)
    return actual_id


def collate_vlm_inputs(instances, pad_token_id, model_max_length):
    """Concatenate image patches/grids in sample order, including multiple views."""
    if not instances:
        raise ValueError("Cannot collate an empty batch")
    ids = [item["input_ids"].reshape(-1) for item in instances]
    lengths = torch.tensor([len(item) for item in ids], device=ids[0].device)
    if lengths.max().item() > model_max_length:
        raise ValueError("Input exceeds model_max_length; refusing to truncate images or action labels")
    padded = pad_sequence(ids, batch_first=True, padding_value=pad_token_id)
    mask = torch.arange(padded.shape[1], device=padded.device)[None, :] < lengths[:, None]
    result = BatchFeature({"input_ids": padded, "attention_mask": mask.long()})
    for key in ("pixel_values", "image_grid_thw"):
        result[key] = torch.cat([item[key] for item in instances], dim=0)
    if "labels" in instances[0]:
        labels = [item["labels"].reshape(-1) for item in instances]
        if any(label.shape != sample.shape for label, sample in zip(labels, ids)):
            raise ValueError("Labels and input_ids must have the same length")
        result["labels"] = pad_sequence(labels, batch_first=True, padding_value=-100)
    return result


def build_vlm_inputs(processor, images, instructions, *, global_token_id=None,
                     action_texts=None, model_max_length=None, max_instruction_tokens=512):
    """Build identical observation prefixes for teacher forcing and extraction."""
    from qwen_vl_utils import process_vision_info

    if len(images) != len(instructions) or not images:
        raise ValueError("Expected one nonempty image list per instruction")
    if action_texts is not None and len(action_texts) != len(images):
        raise ValueError("Expected one action answer per observation")
    if max_instruction_tokens < 0:
        raise ValueError("max_instruction_tokens must be nonnegative")
    tokenizer = processor.tokenizer
    limit = tokenizer.model_max_length if model_max_length is None else model_max_length
    if limit <= 0:
        raise ValueError("model_max_length must be positive")
    instances = []
    for index, (views, instruction) in enumerate(zip(images, instructions)):
        if not views:
            raise ValueError("At least one image is required")
        instruction_ids = tokenizer.encode(instruction, add_special_tokens=False)
        if global_token_id is not None and global_token_id in instruction_ids:
            raise ValueError("GLOBAL must not occur in instructions or action answers")
        # An answer-independent cap preserves the same prefix at training and extraction time.
        if len(instruction_ids) > max_instruction_tokens:
            instruction = tokenizer.decode(instruction_ids[:max_instruction_tokens], skip_special_tokens=False)
        user = {"role": "user", "content": [
            *[{"type": "image", "image": view} for view in views],
            {"type": "text", "text": instruction},
        ]}
        prefix_text = processor.apply_chat_template(
            [user], tokenize=False, add_generation_prompt=True,
        )
        suffix_ids = []
        if action_texts is not None:
            assistant = {"role": "assistant", "content": [
                {"type": "text", "text": action_texts[index]},
            ]}
            full_text = processor.apply_chat_template(
                [user, assistant], tokenize=False, add_generation_prompt=False,
            )
            prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
            full_ids = tokenizer.encode(full_text, add_special_tokens=False)
            if full_ids[:len(prefix_ids)] != prefix_ids:
                raise ValueError("Chat template answer boundary does not match generation prefix")
            suffix_ids = full_ids[len(prefix_ids):]
            if not suffix_ids:
                raise ValueError("Action answer must not be empty")

        image_inputs, _ = process_vision_info([user], image_patch_size=16)
        encoded = processor(text=[prefix_text], images=image_inputs,
                            padding=False, return_tensors="pt")
        prefix = encoded["input_ids"][0]
        suffix = prefix.new_tensor(suffix_ids)
        if global_token_id is not None:
            if (prefix == global_token_id).any() or (suffix == global_token_id).any():
                raise ValueError("GLOBAL must not occur in instructions or action answers")
            prefix = torch.cat([prefix, prefix.new_tensor([global_token_id])])
        ids = torch.cat([prefix, suffix])
        if len(ids) > limit:
            raise ValueError(
                f"Sample {index} exceeds model_max_length={limit}; reduce the instruction/image "
                "budget or increase the limit. Images/answers were not truncated."
            )

        item = {"input_ids": ids, "pixel_values": encoded["pixel_values"],
                "image_grid_thw": encoded["image_grid_thw"]}
        if action_texts is not None:
            labels = ids.clone()
            labels[:len(prefix)] = -100
            item["labels"] = labels
        instances.append(item)
    return collate_vlm_inputs(instances, tokenizer.pad_token_id, limit)


@dataclass
class GlobalVLMFeatures:
    vlm_hidden_states: torch.Tensor
    attention_mask: torch.Tensor
    global_index: torch.Tensor
    global_hidden: torch.Tensor
    ordinary_mask: torch.Tensor


def select_global_features(hidden_states, input_ids, attention_mask, global_token_id):
    if hidden_states.shape[:2] != input_ids.shape or attention_mask.shape != input_ids.shape:
        raise ValueError("Hidden states, IDs and attention mask must have aligned sequence lengths")
    global_mask = input_ids.eq(global_token_id) & attention_mask.bool()
    if not torch.all(global_mask.sum(dim=1) == 1):
        raise ValueError("Each sample must contain exactly one unmasked GLOBAL token")
    indices = global_mask.long().argmax(dim=1)
    positions = torch.arange(input_ids.shape[1], device=input_ids.device)[None, :]
    return GlobalVLMFeatures(
        hidden_states, attention_mask, indices,
        hidden_states[torch.arange(len(input_ids), device=hidden_states.device), indices],
        attention_mask.bool() & (positions < indices[:, None]),
    )


@torch.inference_mode()
def extract_global_features(model, processor, images, instructions):
    token_id = validate_vlm_tokens(model, processor, require_global=True)
    model.eval()
    inputs = build_vlm_inputs(processor, images, instructions, global_token_id=token_id,
                              model_max_length=model.config.model_max_length,
                              max_instruction_tokens=model.config.max_instruction_tokens)
    inputs = inputs.to(next(model.parameters()).device)
    # Use PSI0's conditional-generation API: its captured states differ from model.model's final RMSNorm output.
    output = model(**inputs, output_hidden_states=True, logits_to_keep=1,
                   use_cache=False, return_dict=True)
    return select_global_features(output.hidden_states[-1], inputs["input_ids"],
                                  inputs["attention_mask"], token_id)


def load_global_backbone(path, *, device="cpu", dtype=torch.float32, attn_implementation="sdpa"):
    from psi.trainers.pretrain_checkpoint import validate_checkpoint_metadata

    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        path, local_files_only=True, dtype=dtype, attn_implementation=attn_implementation,
    )
    validate_vlm_tokens(model, processor, require_global=True)
    validate_checkpoint_metadata(path, model, processor)
    return model.to(device).eval(), processor
