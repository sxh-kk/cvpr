"""Frozen CLIP encoder shared by text preprocessing and raw-text inference."""

import torch


class CLIPTextEncoder:
    def __init__(self, model_name="openai/clip-vit-large-patch14", device="cpu", local_files_only=False):
        from transformers import CLIPTextModelWithProjection, CLIPTokenizer

        self.device = torch.device(device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
        self.model = CLIPTextModelWithProjection.from_pretrained(
            model_name, local_files_only=local_files_only
        ).to(self.device).eval().requires_grad_(False)
        if self.model.config.hidden_size != 768 or self.model.config.projection_dim != 768:
            raise ValueError("E7 text conditioning requires CLIP hidden/projection dimensions of 768")

    @torch.inference_mode()
    def __call__(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        if not texts or not all(isinstance(text, str) for text in texts):
            raise ValueError("Text must be a string or a non-empty list of strings")
        encoded = self.tokenizer(texts, padding="max_length", truncation=True, max_length=16, return_tensors="pt")
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        output = self.model(**encoded, return_dict=True)
        valid = torch.tensor([bool(text.strip()) for text in texts], device=self.device)
        return {
            "text_pooled": output.text_embeds.float().masked_fill(~valid[:, None], 0),
            "text_tokens": output.last_hidden_state.float().masked_fill(~valid[:, None, None], 0),
            "text_mask": encoded["attention_mask"].bool() & valid[:, None],
            "text_valid": valid,
        }
