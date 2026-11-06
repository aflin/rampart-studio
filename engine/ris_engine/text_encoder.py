# Qwen3-VL-4B as the text encoder: hidden states of 8 layers, stacked per token.
#
# Portions derived from black-forest-labs/flux-3-image-klein (src/flux3/text_encoder.py),
# Copyright Black Forest Labs, licensed under the Apache License 2.0; see NOTICE.

import json
import math
import os

import torch
from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration

from .stream import STREAM_SLOTS, WeightStream, spread, tensor_bytes

OUTPUT_LAYERS = (4, 8, 12, 16, 20, 24, 28, 32)
MAX_LENGTH = 4096
PAD_MULTIPLE = 80


class TextEncoder:
    """Loaded on the CPU; place() then moves it to the GPU, keeping as many of its decoder layers
    there as asked and streaming the others from CPU memory each encode (stream.py).  The token
    embedding table (0.8 GB) stays on the CPU, where looking up a prompt's rows costs nothing."""

    def __init__(self, path):
        model = Qwen3VLForConditionalGeneration.from_pretrained(path, dtype=torch.bfloat16).eval()
        self.text = model.model.language_model
        self.text.layers = self.text.layers[:max(OUTPUT_LAYERS) + 1]  # the later layers' outputs are never used
        self.model = model.model  # without the LM head; the vision tower stays unused on the CPU
        self.stream = None
        # the tokenizer alone, with the processor's chat template: the full processor needs torchvision
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.tokenizer.padding_side = "right"
        with open(os.path.join(path, "chat_template.json")) as f:
            self.chat_template = json.load(f)["chat_template"]

    def gpu_bytes(self, layers):
        """GPU memory for place(), with `layers` decoder layers resident."""
        text = self.text
        fixed = tensor_bytes(text.rotary_emb) + tensor_bytes(text.norm)
        layer = tensor_bytes(text.layers[0])
        streamed = len(text.layers) - layers
        return fixed + layers * layer + (STREAM_SLOTS * layer if streamed else 0)

    def place(self, device, layers=None):
        """Move to `device` with `layers` decoder layers (default all) resident."""
        text = self.text
        n = len(text.layers) if layers is None else min(layers, len(text.layers))
        for module in (text.rotary_emb, text.norm):
            module.to(device)
        streamed = spread(list(text.layers), len(text.layers) - n)
        for layer in text.layers:
            if layer not in streamed:
                layer.to(device)
        self.stream = WeightStream(streamed, device, STREAM_SLOTS) if streamed else None
        self.device = device
        return self

    @torch.inference_mode()
    def encode(self, prompt):
        """prompt -> tokens (L, 20480) bf16 and ids (L, 4) = (0, 0, 0, position); L padded to a multiple of 80."""
        tokenizer = self.tokenizer
        text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], chat_template=self.chat_template,
                                             tokenize=False, add_generation_prompt=True)
        real = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)["input_ids"].shape[1]
        length = min(math.ceil(real / PAD_MULTIPLE) * PAD_MULTIPLE, MAX_LENGTH)
        inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=length)
        mask = inputs["attention_mask"].to(self.device)
        positions = (mask.long().cumsum(-1) - 1).masked_fill(mask == 0, 1)
        embeds = self.text.embed_tokens(inputs["input_ids"]).to(self.device)
        if self.stream is not None:
            self.stream.start(list(self.text.layers))
        output = self.model(
            inputs_embeds=embeds,
            attention_mask=mask,
            position_ids=positions[None].expand(4, -1, -1),
            output_hidden_states=True,
            use_cache=False,
        )
        stacked = torch.stack([output.hidden_states[k] for k in OUTPUT_LAYERS], dim=1)  # (1, 8, L, 2560)
        tokens = stacked.permute(0, 2, 1, 3).reshape(length, -1).to(torch.bfloat16)
        positions = torch.arange(length, device=self.device)
        ids = torch.stack([torch.zeros_like(positions)] * 3 + [positions], dim=-1)
        return tokens, ids
