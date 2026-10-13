"""Export a pretraining checkpoint with its actual tokenizer and model config."""

import argparse
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from psi.models.global_vlm import validate_vlm_tokens
from psi.trainers.pretrain_checkpoint import (
    resolve_pretrain_checkpoint, save_checkpoint_metadata, validate_checkpoint_metadata,
)


def export_pretrained(checkpoint, output, *, dtype="auto", max_shard_size="5GB"):
    checkpoint = Path(checkpoint)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory is not empty: {output}")
    processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        checkpoint, local_files_only=True, dtype=dtype, attn_implementation="sdpa",
    )
    validate_vlm_tokens(model, processor)
    metadata = validate_checkpoint_metadata(checkpoint, model, processor)
    model.save_pretrained(output, safe_serialization=True, max_shard_size=max_shard_size)
    save_checkpoint_metadata(output, model, processor, metadata)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Checkpoint directory or run directory (latest checkpoint)")
    parser.add_argument("--step", type=int, help="Select a checkpoint within a run directory")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dtype", choices=["auto", "float32", "bfloat16"], default="auto")
    args = parser.parse_args()
    source = args.checkpoint / "checkpoints" / f"ckpt_{args.step}" if args.step else args.checkpoint
    source = resolve_pretrain_checkpoint(source)
    dtype = "auto" if args.dtype == "auto" else getattr(torch, args.dtype)
    print(export_pretrained(source, args.output, dtype=dtype))


if __name__ == "__main__":
    main()
