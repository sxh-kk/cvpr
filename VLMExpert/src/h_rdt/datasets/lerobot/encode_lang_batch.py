import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.lerobot.lerobot_dataset import LeRobotDataset
from models.encoder.t5_encoder import T5Embedder


def instruction_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=os.environ.get("LEROBOT_DATA_ROOT", ""))
    parser.add_argument("--config_path", default=os.environ.get("HRDT_CONFIG_PATH", "configs/hrdt_finetune_lerobot.yaml"))
    parser.add_argument("--model_path", default=os.environ.get("T5_MODEL_PATH", ""))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--offload_dir", default="")
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser()
    if not data_root.exists():
        raise FileNotFoundError("LeRobot data root not found.")

    meta_dir = data_root / "meta"
    episodes_path = meta_dir / "episodes.jsonl"
    if not episodes_path.exists():
        raise FileNotFoundError(f"Missing {episodes_path}.")

    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    offload_dir = args.offload_dir.strip() or None
    if offload_dir is not None:
        Path(offload_dir).mkdir(parents=True, exist_ok=True)

    text_embedder = T5Embedder(
        from_pretrained=args.model_path,
        model_max_length=config["dataset"]["tokenizer_max_length"],
        device=torch.device(args.device),
        use_offload_folder=offload_dir,
    )
    tokenizer = text_embedder.tokenizer
    text_encoder = text_embedder.model

    instructions = {}
    with episodes_path.open("r") as f:
        for line in f:
            row = json.loads(line)
            instruction = str(row.get("instruction", "") or "").strip()
            if not instruction:
                instruction = LeRobotDataset.DEFAULT_INSTRUCTION
            instructions.setdefault(instruction, instruction_hash(instruction))

    output_dir = meta_dir / "lang_embeddings"
    output_dir.mkdir(parents=True, exist_ok=True)
    lang_map = {}

    for instruction, hashed in instructions.items():
        tokens = tokenizer(
            instruction,
            return_tensors="pt",
            padding="longest",
            truncation=True,
        )["input_ids"].to(args.device)
        with torch.no_grad():
            embeddings = text_encoder(tokens).last_hidden_state.detach().cpu()

        embed_name = f"{hashed}.pt"
        torch.save({"instruction": instruction, "embeddings": embeddings}, output_dir / embed_name)
        lang_map[instruction] = embed_name
        print(f"Wrote {output_dir / embed_name}")

    with (meta_dir / "lang_map.json").open("w") as f:
        json.dump(lang_map, f, indent=2)
    print(f"Wrote {meta_dir / 'lang_map.json'}")


if __name__ == "__main__":
    main()
