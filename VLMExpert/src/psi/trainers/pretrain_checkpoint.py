"""Checkpoint metadata shared by pretraining, export and feature extraction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


METADATA_FILE = "pretraining_metadata.json"


def vocabulary_digest(tokenizer):
    data = json.dumps(tokenizer.get_vocab(), sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def resolve_pretrain_checkpoint(path):
    path = Path(path)
    if path.name.startswith("ckpt_") and path.is_dir():
        return path
    directory = path / "checkpoints"
    candidates = [p for p in directory.glob("ckpt_*")
                  if p.is_dir() and p.name[5:].isdigit()]
    if not candidates:
        raise FileNotFoundError(f"No pretraining checkpoint found in {path}")
    return max(candidates, key=lambda p: int(p.name[5:]))


def save_checkpoint_metadata(path, model, processor, metadata):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.config.save_pretrained(path)
    processor.save_pretrained(path)
    metadata = {
        **metadata,
        "format_version": 1,
        "tokenizer_sha256": vocabulary_digest(processor.tokenizer),
        "tokenizer_size": len(processor.tokenizer),
        "embedding_rows": model.config.text_config.vocab_size,
        "global_token_id": getattr(model.config, "global_token_id", None),
        "model_max_length": processor.tokenizer.model_max_length,
        "max_instruction_tokens": model.config.max_instruction_tokens,
    }
    with (path / METADATA_FILE).open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=True)


def validate_checkpoint_metadata(path, model, processor, *, expected_stage=None):
    metadata_path = Path(path) / METADATA_FILE
    if not metadata_path.is_file():
        raise ValueError(f"Missing pretraining metadata: {metadata_path}")
    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    expected = {
        "format_version": 1,
        "tokenizer_sha256": vocabulary_digest(processor.tokenizer),
        "tokenizer_size": len(processor.tokenizer),
        "embedding_rows": model.config.text_config.vocab_size,
        "global_token_id": getattr(model.config, "global_token_id", None),
        "model_max_length": processor.tokenizer.model_max_length,
        "max_instruction_tokens": model.config.max_instruction_tokens,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Checkpoint metadata mismatch for {key}: {metadata.get(key)} != {value}")
    if model.config.model_max_length != processor.tokenizer.model_max_length:
        raise ValueError("Model and tokenizer input length limits differ")
    if expected_stage is not None and metadata.get("stage") != expected_stage:
        raise ValueError("Resume must use the same pretraining stage; load weights for a new stage")
    return metadata
