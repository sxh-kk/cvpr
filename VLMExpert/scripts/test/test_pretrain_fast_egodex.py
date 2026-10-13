import dotenv
dotenv.load_dotenv()

import torch
import numpy as np
from pathlib import Path
from psi.utils import parse_args_to_tyro_config, seed_everything 
from psi.config.config import LaunchConfig
from psi.config.tokenizer import FastActionTokenizerConfig

TRAINING_SCRIPT = "scripts/train/psi0/pretrain-egodex-psi0-fast.sh"
PRETRAIN_CHECKPOINT = "/mnt/beegfs/hfm/cache/checkpoints/psi0/pre.fast.egodex.2512241941.ckpt200k"
EGODEX_DIR = "/mnt/beegfs/datasets/egodex"
DEVICE = "cuda:0"
NUM_EVALS = 50

from psi.utils import parse_args_to_tyro_config, seed_everything
from psi.config.config import LaunchConfig

config:LaunchConfig = parse_args_to_tyro_config(TRAINING_SCRIPT)
seed_everything(config.seed or 42)

from psi.config.data_lerobot import LerobotDataConfig
data_cfg: LerobotDataConfig = config.data # type: ignore
data_cfg.root_dir = EGODEX_DIR

from psi.config.model_qwen3vl import Qwen3VLModelConfig
model_cfg: Qwen3VLModelConfig = config.model # type: ignore

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from psi.tokenizer import FastActionTokenizer
from qwen_vl_utils import process_vision_info

Da = 48
vlm_processor = AutoProcessor.from_pretrained(PRETRAIN_CHECKPOINT)
tokenizer = vlm_processor.tokenizer

print(f"Using device: {DEVICE}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")

model = Qwen3VLForConditionalGeneration.from_pretrained(
    PRETRAIN_CHECKPOINT,
    attn_implementation="flash_attention_2",
    dtype=torch.bfloat16,
    device_map={"": DEVICE},  # Load entire model to GPU 7
)

if isinstance(model_cfg.action_tokenizer, FastActionTokenizerConfig):
    model.resize_token_embeddings(
        len(tokenizer) + model_cfg.action_tokenizer.bins, 
        pad_to_multiple_of = 192,
        mean_resizing = True
    )
    print(f"Resized model token embeddings to {model.lm_head.weight.shape[0]}")

action_tokenizer = FastActionTokenizer(
    tokenizer, data_cfg.chunk_size, Da,
    pretrained_checkpoint=model_cfg.action_tokenizer.pretrained_checkpoint,
    bins=model_cfg.action_tokenizer.bins
)

transform_kwargs=dict(
    action_tokenizer=action_tokenizer,
    vlm_processor=vlm_processor,
)

train_dataset = data_cfg(split="train", transform_kwargs=transform_kwargs)
# val_dataset = data_cfg(split="val", transform_kwargs=transform_kwargs)

print(f"Train dataset size: {len(train_dataset)}")
# print(f"Validation dataset size: {len(val_dataset)}")

from PIL import Image
import numpy as np
np.set_printoptions(precision=4, suppress=True)

raw_dataset = train_dataset
# print(type(raw_dataset))
if "Mixed" in data_cfg.__class__.__name__:
    raw_dataset = raw_dataset.datasets[1]
    print(raw_dataset)
# raw_dataset = val_dataset
maxmin = data_cfg.transform.field

l1_xyz = []
for i in range(0, len(raw_dataset), max(1, len(raw_dataset)//NUM_EVALS)):
    frame = raw_dataset[i]
    # print(list(frame.keys()));print(frame["dataset_name"]);exit(0)
    image = Image.fromarray(np.array(frame["raw_images"][0], dtype=np.uint8))
    # image.save(f"input_image_{i}.png")
    gt_action = frame["raw_actions"][:1]
    input_ids_0 = frame["input_ids"][0]

    vision_end_idx = None
    for idx in range(len(input_ids_0) - 1):
        if input_ids_0[idx] == 151653:
            vision_end_idx = idx
            break

    idx_151645_198_pair_second_last = None
    for idx in range(len(input_ids_0) - 3, -1, -1):
        if input_ids_0[idx] == 151645 and input_ids_0[idx +1] == 198:
            idx_151645_198_pair_second_last = idx
            break

    prompt = tokenizer.decode(frame["input_ids"][0][vision_end_idx+1:idx_151645_198_pair_second_last], skip_special_tokens=True)
    print("Prompt:", prompt)

    with torch.inference_mode():
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image }, 
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages, image_patch_size=16)
        inputs = vlm_processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
        )

        # Decode output (trim input tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = vlm_processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        # print(output_text)
    
    assert tokenizer.decode(generated_ids_trimmed[0][-1]) == "<|im_end|>"

    action_pred = action_tokenizer.decode_token_ids_to_actions([generated_ids_trimmed[0][:-1].cpu().numpy().tolist()])
    denorm_action_pred = maxmin.denormalize(action_pred)
    # print("Predicted action:", denorm_action_pred)

    denorm_gt_action = maxmin.denormalize(gt_action)
    # print("Ground-truth action:", gt_action)

    errs = np.abs(denorm_action_pred - denorm_gt_action).mean()
    print("L1 error:", errs)
    l1_xyz.append(errs) 

print(f"L1 errors: {l1_xyz}")
print(f"Mean L1 error over {NUM_EVALS} samples: {np.mean(l1_xyz)}")
