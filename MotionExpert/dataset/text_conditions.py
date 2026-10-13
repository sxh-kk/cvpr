"""Text-condition helpers shared by preprocessing and the motion dataset.

The training path consumes frozen CLIP features produced by
``MotionExpert.precompute_text_features``. Keeping prompt construction here makes
the offline cache and online lookup use exactly the same window convention.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Mapping

import torch


TEXT_FEATURE_FORMAT = "uniegomotion_clip_text_features_v1"


def _clean_text(value: object, max_chars: int = 260) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_chars:
        return text
    # Keep a complete prefix where possible.  CLIP tokenization is performed
    # after this truncation, so long expert commentary cannot crowd out the
    # short action phrase that follows it.
    clipped = text[:max_chars].rsplit(" ", 1)[0].strip()
    return clipped or text[:max_chars].strip()


def load_take_metadata(path: str | os.PathLike | None) -> dict[str, dict]:
    """Return EE4D take metadata indexed by both name and UID."""

    if path is None:
        return {}
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Take metadata not found: {path}")
    with open(path, encoding="utf-8") as handle:
        rows = json.load(handle)
    if isinstance(rows, Mapping):
        rows = rows.values()
    result: dict[str, dict] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        for key in ("take_name", "take_uid", "capture_uid"):
            value = row.get(key)
            if value not in (None, ""):
                result[str(value)] = dict(row)
    return result


def _annotation_map(record: Mapping) -> dict[str, dict]:
    annotations = record.get("text_annotations") or record.get("narration") or []
    return {
        str(item.get("text_id")): dict(item)
        for item in annotations
        if isinstance(item, Mapping) and item.get("text_id") is not None
    }


def _window_annotations(record: Mapping, start: int, window: int) -> list[dict]:
    """Resolve annotation ids for one 10 FPS motion window."""

    end = int(start) + int(window) - 1
    by_id = _annotation_map(record)
    ids = (record.get("text_window_ids") or {}).get(str(int(start)), [])
    selected = [by_id[text_id] for text_id in ids if text_id in by_id]
    if selected:
        return selected

    # Older packs may only contain annotations, not the explicit index.  The
    # frame overlap fallback keeps those packs usable without changing their
    # serialized representation.
    selected = []
    for item in by_id.values():
        item_start = item.get("frame_start")
        item_end = item.get("frame_end", item_start)
        if item_start is None:
            continue
        if int(item_start) <= end and int(item_end) >= int(start):
            selected.append(item)
    return selected


def _take_for_record(record: Mapping, take_metadata: Mapping[str, dict]) -> Mapping:
    for key in ("take_name", "source_take_uid", "take_uid"):
        value = record.get(key)
        if value is not None and str(value) in take_metadata:
            return take_metadata[str(value)]
    return {}


def compose_window_prompt(
    record: Mapping,
    start: int,
    *,
    source: str,
    window: int = 80,
    take_metadata: Mapping[str, dict] | None = None,
    max_chars: int = 700,
) -> str:
    """Create a compact intent-plus-local-action prompt for one window.

    EE4D uses task/keystep/atomic annotations.  Nymeria uses activity
    summaries plus motion narration/atomic annotations.  Expert commentary is
    deliberately a fallback only; its long prose is not a stable motion label.
    """

    annotations = _window_annotations(record, start, window)
    grouped: dict[str, list[str]] = {}
    for item in annotations:
        text = _clean_text(item.get("text"), max_chars=260)
        if text:
            grouped.setdefault(str(item.get("text_type", "unknown")), []).append(text)

    parts: list[str] = []
    if source == "ee4d":
        take = _take_for_record(record, take_metadata or {})
        task = _clean_text(take.get("parent_task_name") or take.get("task_name"), 180)
        if task:
            parts.append(f"Task: {task}.")
        for text in grouped.get("keystep", [])[:2]:
            parts.append(f"Step: {text}.")
        for text in grouped.get("atomic_action", [])[:3]:
            parts.append(f"Action: {text}.")
        if len(parts) == 0:
            for text in grouped.get("expert_commentary", [])[:1]:
                parts.append(f"Context: {text}.")
    else:
        for text in grouped.get("activity_summarization", [])[:1]:
            parts.append(f"Activity: {text}.")
        for text in grouped.get("motion_narration", [])[:2]:
            parts.append(f"Motion: {text}.")
        for text in grouped.get("atomic_action", [])[:2]:
            parts.append(f"Action: {text}.")

    prompt = _clean_text(" ".join(parts), max_chars=max_chars)
    return prompt


class TextFeatureStore:
    """Read-only lookup for precomputed per-window CLIP features."""

    def __init__(self, path: str | os.PathLike, *, token_count: int = 16, feature_dim: int = 768):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"text feature sidecar not found: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        if payload.get("format") != TEXT_FEATURE_FORMAT:
            raise ValueError(
                f"unsupported text feature format in {path}: {payload.get('format')!r}"
            )
        self.path = str(path)
        self.index = payload["index"]
        self.pooled = torch.as_tensor(payload["pooled"])
        self.tokens = torch.as_tensor(payload["tokens"])
        self.token_mask = torch.as_tensor(payload["token_mask"], dtype=torch.bool)
        self.valid = torch.as_tensor(payload["valid"], dtype=torch.bool)
        self.token_count = int(payload.get("token_count", self.tokens.shape[1]))
        self.feature_dim = int(payload.get("feature_dim", self.pooled.shape[-1]))
        if self.feature_dim != int(feature_dim):
            raise ValueError(
                f"text feature dim mismatch: sidecar={self.feature_dim}, config={feature_dim}"
            )
        if self.token_count != int(token_count):
            raise ValueError(
                f"text sidecar has {self.token_count} tokens, config requests {token_count}"
            )
        if self.pooled.ndim != 2 or self.tokens.ndim != 3:
            raise ValueError("text sidecar pooled/tokens tensors have invalid dimensions")
        rows = self.pooled.shape[0]
        if (self.pooled.shape != (rows, self.feature_dim)
                or self.tokens.shape != (rows, self.token_count, self.feature_dim)
                or self.token_mask.shape != (rows, self.token_count)
                or self.valid.shape != (rows,)):
            raise ValueError("Text sidecar tensor shapes do not match metadata")
        self.empty_id = int(payload.get("empty_id", 0))
        if not 0 <= self.empty_id < rows or self.valid[self.empty_id]:
            raise ValueError("Text sidecar must contain an invalid empty row")

    def get(self, sequence_name: str, start: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sequence_index = self.index.get(sequence_name, {})
        key = str(int(start))
        if key not in sequence_index:
            raise KeyError(f"Text sidecar has no window {sequence_name!r} at {start}; regenerate its index")
        feature_id = int(sequence_index[key])
        if not 0 <= feature_id < self.pooled.shape[0]:
            raise ValueError(f"Text feature index out of range: {sequence_name}:{start} -> {feature_id}")
        pooled = self.pooled[feature_id]
        tokens = self.tokens[feature_id, : self.token_count]
        token_mask = self.token_mask[feature_id, : self.token_count]
        valid = self.valid[feature_id]
        if not torch.isfinite(pooled).all() or not torch.isfinite(tokens).all():
            raise ValueError(f"Non-finite CLIP features: {sequence_name}:{start}")
        return pooled, tokens, token_mask, valid


__all__ = [
    "TEXT_FEATURE_FORMAT",
    "TextFeatureStore",
    "compose_window_prompt",
    "load_take_metadata",
]
