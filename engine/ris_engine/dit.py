# FLUX 3 Image Klein transformer (DiT).
#
# Portions derived from black-forest-labs/flux-3-image-klein (src/flux3/dit.py),
# Copyright Black Forest Labs, licensed under the Apache License 2.0; see NOTICE.
# Changes: per-sample timestep embeddings and modulation tables, a packed batch built from
# independent jobs that may be at different sampling steps, and an INT8 block path on
# comfy_kitchen kernels (see int8.py).

import math

import torch
import torch.nn.functional as F
from torch import nn

# token groups, in the order they take inside one sample of the joint stream
MODES = ("image", "image_cond", "text")
IN_CHANNELS = {"image": 128, "image_cond": 128, "text": 20480}
HIDDEN = 3072
HEADS = 24
HEAD_DIM = HIDDEN // HEADS
DEPTH = 5
DEPTH_SINGLE = 28
AXES_DIM = (32, 32, 32, 32)
MLP_RATIO = 3.0
THETA = 10000


def timestep_embedding(t, dim=256, max_period=10000):
    t = 1000.0 * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1).to(t)


def _rope_axis(pos, dim):
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    out = torch.einsum("...n,d->...nd", pos.double(), 1.0 / (THETA ** scale))
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    return out.reshape(*out.shape[:-1], 2, 2).float()


def rope_table(ids):
    """ids (L, 4) -> rotation matrices (L, head_dim // 2, 2, 2)."""
    return torch.cat([_rope_axis(ids[:, i], AXES_DIM[i]) for i in range(len(AXES_DIM))], dim=-3)


def rmsnorm_rope(x, scale, freqs, eps=1e-6):
    """x (1, H, L, D), freqs (1, 1, L, D/2, 2, 2)."""
    with torch.autocast(device_type=x.device.type, enabled=False):
        b, h, l, d = x.shape
        x_f = x.float()
        x_norm = x_f * x_f.pow(2).mean(-1, keepdim=True).add(eps).rsqrt() * scale.float()
        x_r = x_norm.reshape(b, h, l, d // 2, 1, 2)
        rotated = freqs[..., 0] * x_r[..., 0] + freqs[..., 1] * x_r[..., 1]
        return rotated.reshape(b, h, l, d).to(x.dtype)


def modulate(x, seg, shift, scale):
    """Layer norm, then each row's (1 + scale) and shift: row r uses row seg[r] of the tables."""
    return (1.0 + scale[seg]) * F.layer_norm(x, (x.shape[-1],), eps=1e-6) + shift[seg]


def gated_add(x, seg, gate, y):
    return x + gate[seg] * y


def packed_attention(q, k, v, bounds):
    """Attention over (1, H, L, D) where each (start, end) range is one sample; samples never see each other."""
    _, heads, length, dim = q.shape
    out = q.new_empty(length, heads * dim)
    for start, end in bounds:
        if start == end:
            continue
        chunk = F.scaled_dot_product_attention(q[:, :, start:end], k[:, :, start:end], v[:, :, start:end])
        out[start:end] = chunk[0].transpose(0, 1).reshape(end - start, heads * dim)
    return out


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))


class QKNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q, k, freqs):
        return rmsnorm_rope(q, self.query_norm.scale, freqs), rmsnorm_rope(k, self.key_norm.scale, freqs)


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=False)
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.out_layer(F.silu(self.in_layer(x)))


class ModeModulation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lin = nn.Linear(dim, 3 * dim, bias=False)

    def forward(self, vec):
        """(S, dim) -> shift, scale, gate tables, one row per sample."""
        return tuple(c.contiguous() for c in self.lin(F.silu(vec)).chunk(3, dim=-1))


class LastLayer(nn.Module):
    def __init__(self, hidden, out_channels):
        super().__init__()
        self.linear = nn.Linear(hidden, out_channels, bias=False)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 2 * hidden, bias=False))

    def forward(self, x, seg, vec):
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=-1)
        return self.linear(modulate(x, seg, shift.contiguous(), scale.contiguous()))


class Block(nn.Module):
    """Parallel attention and SwiGLU MLP under one modulation and one gate.

    Rows of x belong to segments (a sample, or a sample's mode in the joint stream); seg gives
    each row's segment and the modulation tables have one row per segment."""

    def __init__(self, hidden, heads, mlp_ratio):
        super().__init__()
        self.heads = heads
        mlp_hidden = int(hidden * mlp_ratio)
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.mlp_in = nn.Linear(hidden, mlp_hidden * 2, bias=False)
        self.attn_out = nn.Linear(hidden, hidden, bias=False)
        self.mlp_out = nn.Linear(mlp_hidden, hidden, bias=False)
        self.norm = QKNorm(hidden // heads)
        self.int8 = None  # int8.Int8Block once converted by int8.load_dit

        # the halves before and after attention; compile() replaces these with compiled versions
        self.pre, self.post = self._pre, self._post

    def compile(self, **kwargs):
        """Compile the two halves separately: the attention between them depends on how many
        samples are packed, and would otherwise make every new batch mix recompile."""
        self.pre, self.post = torch.compile(self._pre, **kwargs), torch.compile(self._post, **kwargs)

    def _pre(self, x, freqs, seg, shift, scale):
        x_mod = modulate(x, seg, shift, scale)
        split = lambda t: t.reshape(1, t.shape[0], self.heads, -1).transpose(1, 2)  # L (H D) -> 1 H L D
        q, k = self.norm(split(self.q_proj(x_mod)), split(self.k_proj(x_mod)), freqs[None, None])
        v = split(self.v_proj(x_mod))
        return x_mod, q, k, v.to(q.dtype)

    def _post(self, x, x_mod, attn, seg, gate):
        gate_in, up = self.mlp_in(x_mod).chunk(2, dim=-1)
        return gated_add(x, seg, gate, self.attn_out(attn) + self.mlp_out(F.silu(gate_in) * up))

    def forward(self, x, freqs, bounds, seg, mod):
        """freqs: (L, D/2, 2, 2) rope table of the rows; mod: (shift, scale, gate) tables."""
        if self.int8 is not None:
            return self.int8(x, freqs, bounds, seg, mod)
        shift, scale, gate = mod
        x_mod, q, k, v = self.pre(x, freqs, seg, shift, scale)
        return self.post(x, x_mod, packed_attention(q, k, v, bounds), seg, gate)


class Sample:
    """One image being generated, as the DiT sees it for one step.

    tokens/rope: per mode, (L, C) tokens and their (L, D/2, 2, 2) rope table, or None.
    t: the sample's current time; t_next: the time this step jumps to (flow map)."""

    __slots__ = ("tokens", "rope", "t", "t_next")

    def __init__(self, tokens, rope, t, t_next):
        self.tokens, self.rope, self.t, self.t_next = tokens, rope, t, t_next


def _ranges(lengths):
    starts = [0]
    for n in lengths:
        starts.append(starts[-1] + n)
    return [(starts[i], starts[i + 1]) for i in range(len(lengths))]


def _segments(lengths, device):
    """Row -> segment index for consecutive segments of the given lengths."""
    return torch.repeat_interleave(torch.arange(len(lengths), device=device), torch.tensor(lengths, device=device))


class DiT(nn.Module):
    def __init__(self, flow_map=True):
        super().__init__()
        self.flow_map = flow_map
        self.emb_in = nn.ModuleDict({m: nn.Linear(IN_CHANNELS[m], HIDDEN, bias=False) for m in ("image", "image_cond")})
        self.txt_in = nn.Linear(IN_CHANNELS["text"], HIDDEN, bias=False)
        self.time_in = MLPEmbedder(256, HIDDEN)
        if flow_map:
            self.ref_time_in = MLPEmbedder(256, HIDDEN)
        self.early_stream_modulations = nn.ModuleDict({k: ModeModulation(HIDDEN) for k in ("image", "image_cond", "txt")})
        self.single_stream_modulations = nn.ModuleDict({k: ModeModulation(HIDDEN) for k in ("image", "image_cond", "txt")})
        self.content_mode_blocks = nn.ModuleDict({
            m: nn.ModuleList([Block(HIDDEN, HEADS, MLP_RATIO) for _ in range(DEPTH)]) for m in ("image", "image_cond")
        })
        self.txt_mode_blocks = nn.ModuleList([Block(HIDDEN, HEADS, MLP_RATIO) for _ in range(DEPTH)])
        self.single_blocks = nn.ModuleList([Block(HIDDEN, HEADS, MLP_RATIO) for _ in range(DEPTH_SINGLE)])
        self.final_layer = nn.ModuleDict({"image": LastLayer(HIDDEN, IN_CHANNELS["image"])})

    def forward(self, samples):
        """Velocity for each sample's image tokens: a list of (L_image, 128) tensors, in sample order.

        Every sample carries its own time, so samples at different steps share one pass."""
        device = samples[0].tokens["image"].device
        t = torch.tensor([s.t for s in samples], device=device, dtype=torch.bfloat16)
        vec_image = self.time_in(timestep_embedding(t))  # (S, HIDDEN); text and references sit at time 0
        vec_zero = self.time_in(timestep_embedding(torch.zeros(1, device=device, dtype=torch.bfloat16)))
        if self.flow_map:
            t_next = torch.tensor([s.t_next for s in samples], device=device, dtype=torch.bfloat16)
            ref = self.ref_time_in(timestep_embedding(t_next))
            vec_image = vec_image + ref
            vec_other = vec_zero + ref
        else:
            vec_other = vec_zero.expand(len(samples), -1)

        mod_key = lambda m: "txt" if m == "text" else m
        x, freqs, bounds, seg, lengths, early, single = {}, {}, {}, {}, {}, {}, {}
        for m in MODES:
            lengths[m] = [0 if s.tokens[m] is None else s.tokens[m].shape[0] for s in samples]
            if not any(lengths[m]):
                continue
            proj = self.txt_in if m == "text" else self.emb_in[m]
            x[m] = proj(torch.cat([s.tokens[m] for s in samples if s.tokens[m] is not None]))
            freqs[m] = torch.cat([s.rope[m] for s in samples if s.tokens[m] is not None])
            bounds[m] = _ranges(lengths[m])
            seg[m] = _segments(lengths[m], device)
            vec = vec_image if m == "image" else vec_other
            early[m] = self.early_stream_modulations[mod_key(m)](vec)
            single[m] = self.single_stream_modulations[mod_key(m)](vec)

        stream = getattr(self, "block_stream", None)  # int8.BlockStream: weights copied in per block
        if stream is not None:
            mode_blocks = lambda m: self.txt_mode_blocks if m == "text" else self.content_mode_blocks[m]
            stream.start([mode_blocks(m)[d].int8 for d in range(DEPTH) for m in x]
                         + [block.int8 for block in self.single_blocks])

        for d in range(DEPTH):
            for m in x:
                blocks = self.txt_mode_blocks if m == "text" else self.content_mode_blocks[m]
                x[m] = blocks[d](x[m], freqs[m], bounds[m], seg[m], early[m])

        # joint stream: sample-major, modes in MODES order inside each sample; one modulation
        # table row per (sample, mode) segment
        order, joint_lengths = [], []
        for i in range(len(samples)):
            n = 0
            for m in MODES:
                if lengths[m][i]:
                    order.append((m, i, *bounds[m][i]))
                    n += lengths[m][i]
            joint_lengths.append(n)
        h = torch.cat([x[m][a:b] for m, _, a, b in order])
        freqs_joint = torch.cat([freqs[m][a:b] for m, _, a, b in order])
        mods = tuple(torch.stack([single[m][j][i] for m, i, _, _ in order]) for j in range(3))
        seg_joint = _segments([b - a for _, _, a, b in order], device)
        joint_bounds = _ranges(joint_lengths)
        for block in self.single_blocks:
            h = block(h, freqs_joint, joint_bounds, seg_joint, mods)

        image_rows, offset = [], 0
        for m, _, a, b in order:
            if m == "image":
                image_rows.append(h[offset:offset + b - a])
            offset += b - a
        out = self.final_layer["image"](torch.cat(image_rows), seg["image"], vec_image)
        return list(out.split(lengths["image"]))
