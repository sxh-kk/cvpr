#!/usr/bin/env python3
"""Precompute frozen CLIP text features for EE4D/Nymeria motion windows.

The resulting sidecar is intentionally independent of the large motion packs:
the dataset only needs ``(sequence_name, window_start)`` to retrieve a pooled
sentence embedding and a short token sequence.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import torch
from tqdm.auto import tqdm
from .text_encoder import CLIPTextEncoder

from .dataset.text_conditions import (
    TEXT_FEATURE_FORMAT,
    compose_window_prompt,
    load_take_metadata,
)


def _load_payload(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


def _collect_prompts(
    pack_path: Path,
    *,
    source: str,
    ee_text_path: Path | None,
    takes_path: Path | None,
    window: int,
    limit: int,
):
    print(f"Loading motion pack: {pack_path}", flush=True)
    pack = _load_payload(pack_path)
    if not isinstance(pack, dict):
        raise ValueError(f"motion pack must be a dictionary: {pack_path}")
    text_sidecar = _load_payload(ee_text_path) if ee_text_path is not None else None
    text_sequences = (text_sidecar or {}).get("sequences", {})
    take_metadata = load_take_metadata(takes_path)

    index: dict[str, dict[str, str]] = {}
    prompts: dict[str, str] = {}
    sequence_names = list(pack)
    if limit > 0:
        sequence_names = sequence_names[:limit]
    for sequence_name in tqdm(sequence_names, desc=f"prompts:{source}"):
        record = pack[sequence_name]
        text_record = text_sequences.get(sequence_name, record)
        num_frames = int(record.get("num_frames", 0))
        starts = range(0, max(0, num_frames - 20), 20)
        sequence_index: dict[str, str] = {}
        for start in starts:
            prompt = compose_window_prompt(
                text_record,
                start,
                source=source,
                window=window,
                take_metadata=take_metadata,
            )
            key = f"{sequence_name}::{int(start)}"
            prompts[key] = prompt
            sequence_index[str(int(start))] = key
        index[sequence_name] = sequence_index

    del pack, text_sidecar
    gc.collect()
    return index, prompts


def _encode_prompts(
    prompts: list[str],
    *,
    model_name: str,
    token_count: int,
    batch_size: int,
    device: torch.device,
):
    if token_count != 16:
        raise ValueError("E7 K12 text conditioning requires exactly 16 CLIP tokens")
    if not prompts:
        return torch.zeros(0, 768), torch.zeros(0, 16, 768), torch.zeros(0, 16, dtype=torch.bool)
    encoder = CLIPTextEncoder(model_name, device)
    chunks = []
    for offset in tqdm(range(0, len(prompts), batch_size), desc="CLIP text"):
        features = encoder(prompts[offset:offset + batch_size])
        chunks.append({key: value.cpu() for key, value in features.items()})
    return tuple(torch.cat([chunk[key] for chunk in chunks]) for key in ("text_pooled", "text_tokens", "text_mask"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ee-pack", type=Path, required=True)
    parser.add_argument("--nymeria-pack", type=Path, required=True)
    parser.add_argument("--ee-text", type=Path)
    parser.add_argument("--takes", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window", type=int, default=80)
    parser.add_argument("--token-count", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--model", default="openai/clip-vit-large-patch14")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ee-limit", type=int, default=0)
    parser.add_argument("--nymeria-limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.token_count != 16 or args.batch_size < 1 or args.window < 1:
        raise ValueError("E7 requires 16 tokens, a positive batch size and a positive window")
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested {device}, but CUDA is unavailable")

    ee_index, ee_prompts = _collect_prompts(
        args.ee_pack,
        source="ee4d",
        ee_text_path=args.ee_text,
        takes_path=args.takes,
        window=args.window,
        limit=args.ee_limit,
    )
    ny_index, ny_prompts = _collect_prompts(
        args.nymeria_pack,
        source="nymeria",
        ee_text_path=None,
        takes_path=None,
        window=args.window,
        limit=args.nymeria_limit,
    )

    if set(ee_index) & set(ny_index):
        raise ValueError("EE4D and Nymeria packs contain duplicate sequence names")
    all_prompt_keys = list(ee_prompts) + list(ny_prompts)
    prompt_to_id: dict[str, int] = {}
    unique_prompts: list[str] = []
    for key in all_prompt_keys:
        prompt = ee_prompts.get(key, ny_prompts.get(key, ""))
        if prompt and prompt not in prompt_to_id:
            prompt_to_id[prompt] = len(unique_prompts) + 1  # zero is the empty row
            unique_prompts.append(prompt)
    print(f"Unique non-empty prompts: {len(unique_prompts)}", flush=True)
    pooled, tokens, token_mask = _encode_prompts(
        unique_prompts,
        model_name=args.model,
        token_count=args.token_count,
        batch_size=args.batch_size,
        device=device,
    )

    feature_dim = int(pooled.shape[-1])
    empty_pooled = torch.zeros(1, feature_dim, dtype=torch.float16)
    empty_tokens = torch.zeros(1, args.token_count, feature_dim, dtype=torch.float16)
    empty_mask = torch.zeros(1, args.token_count, dtype=torch.bool)
    pooled = torch.cat((empty_pooled, pooled.to(torch.float16)), dim=0)
    tokens = torch.cat((empty_tokens, tokens.to(torch.float16)), dim=0)
    token_mask = torch.cat((empty_mask, token_mask), dim=0)
    valid = torch.zeros(pooled.shape[0], dtype=torch.bool)
    valid[1:] = True

    def resolve_index(index: dict[str, dict[str, str]], prompt_map: dict[str, str]):
        resolved = {}
        for sequence_name, starts in index.items():
            resolved[sequence_name] = {}
            for start, prompt_key in starts.items():
                prompt = prompt_map.get(prompt_key, "")
                resolved[sequence_name][start] = int(prompt_to_id.get(prompt, 0))
        return resolved

    output = {
        "format": TEXT_FEATURE_FORMAT,
        "encoder": args.model,
        "token_count": int(args.token_count),
        "feature_dim": feature_dim,
        "empty_id": 0,
        "pooled": pooled,
        "tokens": tokens,
        "token_mask": token_mask,
        "valid": valid,
        "index": {
            **resolve_index(ee_index, ee_prompts),
            **resolve_index(ny_index, ny_prompts),
        },
        "metadata": {
            "window": int(args.window),
            "sources": ["ee4d", "nymeria"],
            "non_empty_prompts": len(unique_prompts),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print(f"Wrote {args.output} ({args.output.stat().st_size / 2**30:.2f} GiB)", flush=True)


if __name__ == "__main__":
    main()
