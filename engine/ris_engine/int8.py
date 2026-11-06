"""INT8 DiT blocks on comfy_kitchen (Apache-2.0) kernels.

Weights: BFL's INT8 "convrot" checkpoint (black-forest-labs/flux-3-image-klein-comfy,
diffusion_models/flux3-image-klein-int8-convrot.safetensors): per-row scaled INT8 weights,
rotated in groups of 256 so the activations quantize well.

Per block the kernels fuse what would otherwise be separate passes over memory: activation
quantization into the GEMMs, the SwiGLU activation into mlp_out's input, and the attention
output's sum with the MLP into mlp_out's epilogue.  (Stacking qkv and mlp_in into one GEMM
saves a quantization but makes the MLP half a strided slice, which costs a bigger copy.)
Attention runs in INT8 for long sequences, where it is faster.
"""

import json

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn

from .dit import HEAD_DIM, HIDDEN, Block, DiT, gated_add, modulate
from .stream import WeightStream, spread

INT8_ATTENTION_MIN = 1024  # shorter sequences are faster in bf16 SDPA

_ck = None


def modulate_bf16(x, seg, shift, scale):
    return modulate(x, seg, shift, scale).to(torch.bfloat16)


_modulate = modulate_bf16
_gated_add = gated_add


def kitchen():
    global _ck
    if _ck is None:
        import comfy_kitchen
        _ck = comfy_kitchen
    return _ck


def compile_helpers(**kwargs):
    """Compile the elementwise steps around the kernels (layer norm + modulation, gated residual)."""
    global _modulate, _gated_add
    _modulate, _gated_add = torch.compile(modulate_bf16, **kwargs), torch.compile(gated_add, **kwargs)


class Int8Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.register_buffer("weight", torch.empty(out_features, in_features, dtype=torch.int8, device=device))
        self.register_buffer("weight_scale", torch.empty(out_features, 1, dtype=torch.float32, device=device))

    def forward(self, x, **kwargs):
        return kitchen().int8_linear(x, self.weight, self.weight_scale, out_dtype=torch.bfloat16,
                                     convrot=True, convrot_groupsize=256, **kwargs)


def attention(q, k, v, bounds):
    """q, k, v (1, L, H, D); samples (bounds) attend only to themselves -> (L, H*D)."""
    from comfy_kitchen.sage_attention import int8_attention
    length = q.shape[1]
    out = q.new_empty(length, HIDDEN)
    for a, b in bounds:
        if a == b:
            continue
        qs, ks, vs = (t[:, a:b].transpose(1, 2) for t in (q, k, v))  # (1, H, n, D) views
        o = int8_attention(qs, ks, vs) if b - a >= INT8_ATTENTION_MIN else F.scaled_dot_product_attention(qs, ks, vs)
        out[a:b] = o[0].transpose(0, 1).reshape(b - a, HIDDEN)
    return out


class Int8Block(nn.Module):
    """The INT8 forward of one Block (whose norm scales it reuses)."""

    def __init__(self, block, device=None):
        super().__init__()
        self.heads = block.heads
        self.__dict__["norm"] = block.norm  # shared, not registered: its scales stay under the Block's keys
        mlp_hidden = block.mlp_out.in_features
        self.qkv = Int8Linear(HIDDEN, 3 * HIDDEN, device)
        self.mlp_in = Int8Linear(HIDDEN, 2 * mlp_hidden, device)
        self.attn_out = Int8Linear(HIDDEN, HIDDEN, device)
        self.mlp_out = Int8Linear(mlp_hidden, HIDDEN, device)
        self.register_buffer("ones", torch.ones(HIDDEN, dtype=torch.bfloat16, device=device), persistent=False)

    def forward(self, x, freqs, bounds, seg, mod):
        shift, scale, gate = mod
        length = x.shape[0]
        x_mod = _modulate(x, seg, shift, scale)
        qkv = self.qkv(x_mod).view(1, length, 3, self.heads, HEAD_DIM)
        mlp = self.mlp_in(x_mod)
        del x_mod
        q, k = kitchen().rms_rope(qkv[:, :, 0], qkv[:, :, 1], freqs[None, :, None], self.norm.query_norm.scale,
                                  self.norm.key_norm.scale, 1e-6)
        attn = self.attn_out(attention(q, k, qkv[:, :, 2], bounds))
        out = self.mlp_out(mlp, input_act="swiglu", residual=attn, residual_scale=self.ones)
        return _gated_add(x, seg, gate, out)


def dit_blocks(dit):
    """The Int8Blocks, the image-reference ones (which run only for edits and upscales) first."""
    cond = [b.int8 for b in dit.content_mode_blocks["image_cond"]]
    other = [b.int8 for d in range(len(dit.txt_mode_blocks))
             for b in (dit.content_mode_blocks["image"][d], dit.txt_mode_blocks[d])]
    return cond, other + [b.int8 for b in dit.single_blocks]


def stream_blocks(dit, resident, device):
    """Move dit (loaded on the CPU) to the GPU with only `resident` blocks' weights kept there;
    the others stream from CPU memory.  The image-reference blocks are streamed first, the rest
    spread evenly through the pass so their copies overlap with resident blocks' computation."""
    cond, other = dit_blocks(dit)
    n = max(0, len(cond) + len(other) - resident)
    streamed = cond[:n] + spread(other, n - len(cond))
    dit.block_stream = WeightStream(streamed, device) if streamed else None  # takes their weights off the modules
    return dit.to(device)


def is_int8_checkpoint(keys):
    return any(k.endswith(".comfy_quant") for k in keys)


def load_dit(path, device):
    """A DiT from the INT8 convrot checkpoint, every block converted to an Int8Block."""
    state = load_file(path, device=str(device))
    with torch.device("meta"):
        dit = DiT(flow_map="ref_time_in.in_layer.weight" in state)
    blocks = [(n, m) for n, m in dit.named_modules() if isinstance(m, Block)]
    for name, block in blocks:
        p = name + "."
        for layer in ("qkv", "mlp_in", "attn_out", "mlp_out"):
            fmt = json.loads(bytes(state.pop(p + layer + ".comfy_quant").tolist()))
            if fmt != {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}:
                raise ValueError(f"{p}{layer}: unsupported quantization {fmt}")
        state.pop(p + "comfy_attention.config", None)
        fast = Int8Block(block, device="meta")
        for layer in ("qkv", "mlp_in", "attn_out", "mlp_out"):
            for suffix in ("weight", "weight_scale"):
                state[f"{p}int8.{layer}.{suffix}"] = state.pop(f"{p}{layer}.{suffix}")
        for layer in ("q_proj", "k_proj", "v_proj", "mlp_in", "attn_out", "mlp_out"):
            delattr(block, layer)
        block.int8 = fast
    state = {k: v.to(torch.bfloat16) if v.is_floating_point() and not k.endswith("weight_scale") else v
             for k, v in state.items()}
    dit.load_state_dict(state, strict=True, assign=True)
    for _, block in blocks:  # buffers that are not in the checkpoint
        block.int8.ones = torch.ones(HIDDEN, dtype=torch.bfloat16, device=device)
    return dit.to(device).eval()
