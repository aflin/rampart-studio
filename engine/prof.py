"""Profile one DiT step on synthetic tokens:  python3r prof.py [dit.safetensors] [image_tokens] [text_tokens] [samples]"""

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "pylib")]

import torch  # noqa: E402

from ris_engine.dit import Sample, rope_table  # noqa: E402
from ris_engine.engine import load_dit  # noqa: E402

path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "weights/flux-3-image-klein-comfy/flux3-image-klein-int8-convrot.safetensors")
n_img = int(sys.argv[2]) if len(sys.argv) > 2 else 4056
n_txt = int(sys.argv[3]) if len(sys.argv) > 3 else 480
n_samples = int(sys.argv[4]) if len(sys.argv) > 4 else 1

dit = load_dit(path, "cuda")
ids = lambda n: torch.stack([torch.zeros(n), torch.arange(n) % 64 * 0.5, torch.arange(n) // 64 * 0.5, torch.zeros(n)], -1).cuda()
samples = [Sample({"image": torch.randn(n_img, 128, device="cuda"), "image_cond": None,
                   "text": torch.randn(n_txt, 20480, device="cuda", dtype=torch.bfloat16)},
                  {"image": rope_table(ids(n_img)), "image_cond": None, "text": rope_table(ids(n_txt))}, 0.5, 0.4)
           for _ in range(n_samples)]


if "--compile" in sys.argv:
    torch._dynamo.config.recompile_limit = 64
    from ris_engine import int8
    int8.compile_helpers(dynamic=True)
    for blocks in [dit.txt_mode_blocks, dit.single_blocks, *dit.content_mode_blocks.values()]:
        for b in blocks:
            b.compile(dynamic=True)


def step():
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        return dit(samples)


for _ in range(3):
    step()
torch.cuda.synchronize()
t0 = time.time()
for _ in range(5):
    step()
torch.cuda.synchronize()
print(f"{(time.time() - t0) / 5 * 1000:.0f} ms per step  ({n_samples} x {n_img} image + {n_txt} text tokens)")

if "--trace" in sys.argv:
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        step()
        torch.cuda.synchronize()
    print(p.key_averages().table(sort_by="cuda_time_total", row_limit=25))
