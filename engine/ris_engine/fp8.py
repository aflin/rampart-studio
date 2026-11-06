# Rowwise FP8 linears: E4M3 weights with a per-row fp32 scale, activations quantized per row.
#
# Portions derived from black-forest-labs/flux-3-image-klein (src/flux3/fp8.py),
# Copyright Black Forest Labs, licensed under the Apache License 2.0; see NOTICE.

import math

import torch
import torch.nn.functional as F
from torch import nn

FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0
SCALE_SUFFIX = ".weight_scale"
GEMM_ALIGNMENT = 256  # N and K of the GEMM padded to this
ROW_ALIGNMENT = 128  # activation rows padded to this
MAX_ROWS = 5120  # on the GB10 the rowwise kernel drops from ~170 to ~50 TFLOPS above ~5000 rows


def _ceil(value, alignment):
    return (value + alignment - 1) // alignment * alignment


def quantize_rows(value):
    rows = value.reshape(-1, value.shape[-1]).float()
    scale = (rows.abs().amax(dim=1) / FP8_MAX).clamp(min=1e-12)
    quantized = (rows / scale[:, None]).clamp(-FP8_MAX, FP8_MAX).to(FP8)
    return quantized.reshape_as(value), scale


class FP8Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.register_buffer("weight", torch.empty(out_features, in_features, dtype=FP8, device=device))
        self.register_buffer("weight_scale", torch.empty(out_features, dtype=torch.float32, device=device))

    def forward(self, x):
        leading = tuple(x.shape[:-1])
        rows = math.prod(leading)
        flat = x.to(torch.bfloat16).reshape(rows, self.in_features)
        if rows == 0:
            return x.new_zeros(*leading, self.out_features, dtype=torch.bfloat16)
        n, k, m = _ceil(self.out_features, GEMM_ALIGNMENT), _ceil(self.in_features, GEMM_ALIGNMENT), _ceil(rows, ROW_ALIGNMENT)
        weight, scale = self.weight, self.weight_scale
        if (n, k) != (self.out_features, self.in_features):
            weight = F.pad(weight.view(torch.uint8), (0, k - self.in_features, 0, n - self.out_features)).view(FP8)
            scale = F.pad(scale, (0, n - self.out_features), value=1.0)
        flat = F.pad(flat, (0, k - self.in_features, 0, m - rows))  # unconditional: one compiled variant for all row counts
        q, q_scale = quantize_rows(flat.contiguous())
        out = scaled_mm_rows(q, q_scale, weight, scale)
        return out[:rows, : self.out_features].reshape(*leading, self.out_features)


@torch.library.custom_op("ris_engine::scaled_mm_rows", mutates_args=())
def scaled_mm_rows(q: torch.Tensor, q_scale: torch.Tensor, weight: torch.Tensor, w_scale: torch.Tensor) -> torch.Tensor:
    """(m, k) FP8 rows with (m,) scales times the (n, k) FP8 weight with (n,) scales -> (m, n) bf16, in even
    chunks of at most MAX_ROWS rows.  An opaque op to torch.compile, so the number of chunks never
    causes a recompile."""
    m = q.shape[0]
    chunk = _ceil(-(-m // -(-m // MAX_ROWS)), ROW_ALIGNMENT)
    w_t, w_scale = weight.T, w_scale[None, :].contiguous()
    mm = lambda qc, sc: torch._scaled_mm(qc, w_t, sc[:, None].contiguous(), w_scale, out_dtype=torch.bfloat16,
                                         use_fast_accum=True)
    if chunk >= m:
        return mm(q, q_scale)
    return torch.cat([mm(qc, sc) for qc, sc in zip(q.split(chunk), q_scale.split(chunk))])


@scaled_mm_rows.register_fake
def _(q, q_scale, weight, w_scale):
    return q.new_empty(q.shape[0], weight.shape[0], dtype=torch.bfloat16)


def swap_linears(module, keys):
    """Every nn.Linear whose weight has a scale in ``keys`` becomes an FP8Linear, in place; returns how many."""
    names = [n for n, m in module.named_modules() if isinstance(m, nn.Linear) and n + SCALE_SUFFIX in keys]
    for name in names:
        linear = module.get_submodule(name)
        parent, _, child = name.rpartition(".")
        setattr(module.get_submodule(parent) if parent else module, child,
                FP8Linear(linear.in_features, linear.out_features, linear.weight.device))
    return len(names)
