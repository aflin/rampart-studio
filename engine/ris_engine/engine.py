"""The CUDA backend: the models in PyTorch, sized to the GPU's free memory (weights that do not fit
stream from CPU memory, stream.py), and jobs of different sizes packed into one DiT forward pass,
with attention kept per job.  The queue, the loop and the HTTP-facing calls are in base.py."""

import math
import os
import time

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file

from . import fp8, int8
from .autoencoder import AutoEncoder, latent_ids
from .base import IMAGE_ROPE_SCALE, EngineBase, noise, reference_image, start_tokens
from .dit import DiT, Sample, rope_table
from .stream import STREAM_SLOTS, GiB, tensor_bytes
from .text_encoder import TextEncoder

FLOW_MAP_KEY = "ref_time_in.in_layer.weight"
MEMORY_MARGIN = 0.5 * GiB  # left free for the CUDA libraries' workspaces and other processes


def load_dit(path, device):
    """bf16 or rowwise-FP8 checkpoints run on PyTorch alone; the INT8 convrot checkpoint on
    comfy_kitchen kernels (int8.py)."""
    with safe_open(path, "pt") as f:
        keys = set(f.keys())
    if int8.is_int8_checkpoint(keys):
        return int8.load_dit(path, device)
    with torch.device("meta"):
        dit = DiT(flow_map=FLOW_MAP_KEY in keys)
    state = load_file(path, device=str(device))
    fp8.swap_linears(dit, keys)
    keep = lambda k, v: k.endswith(fp8.SCALE_SUFFIX) or v.dtype == fp8.FP8 or not v.is_floating_point()
    state = {k: v if keep(k, v) else v.to(torch.bfloat16) for k, v in state.items()}
    dit.load_state_dict(state, strict=True, assign=True)
    return dit.to(device).eval()


def load_autoencoder(path, device):
    """bf16 convolutions (about twice as fast as fp32, within 0.2/255 per pixel); the latent
    statistics stay fp32."""
    ae = AutoEncoder()
    ae.load_state_dict(load_file(path), strict=True)
    ae = ae.to(device, torch.bfloat16).eval()
    ae.bn.float()
    return ae


def to_tensor(image):
    return torch.from_numpy(np.asarray(image)).permute(2, 0, 1).float() / 127.5 - 1.0


class Engine(EngineBase):
    """gpu_memory: GB this engine may use on its GPU (default: what is free when it starts, less
    a margin), of which working_memory GB is kept for activations (enough for 4 MP).  What does
    not fit in the rest streams from CPU memory: the image model's blocks are placed on the GPU
    first, since every step needs them, then the text encoder's layers, needed once per prompt."""

    def __init__(self, dit_path, text_encoder_path, ae_path, out_dir, ref_dirs=(), device="cuda",
                 max_active=16, pass_tokens=3584, compile=True, log=print, gpu_memory=None, working_memory=2.5):
        super().__init__(out_dir, ref_dirs, max_active, pass_tokens, log)
        self.device = device
        t0 = time.time()
        free, total = torch.cuda.mem_get_info(device)
        if gpu_memory:  # a hard limit, so the engine leaves the rest of the GPU alone
            torch.cuda.set_per_process_memory_fraction(min(1.0, gpu_memory * GiB / total), torch.device(device).index or 0)
        budget = (gpu_memory * GiB if gpu_memory else free - MEMORY_MARGIN) - working_memory * GiB
        ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        if torch.cuda.get_device_properties(device).is_integrated or total > 0.8 * ram:
            budget = math.inf  # memory shared with the CPU (DGX Spark): streaming would only copy RAM to RAM
        self.ae = load_autoencoder(ae_path, device)
        dit = load_dit(dit_path, "cpu")
        te = self.text_encoder = TextEncoder(text_encoder_path)
        left = budget - tensor_bytes(self.ae) - te.gpu_bytes(0)
        if dit.single_blocks[0].int8 is not None:  # INT8 blocks stream; FP8/bf16 ones stay on the GPU
            cond, other = int8.dit_blocks(dit)
            n_blocks, block = len(cond) + len(other), tensor_bytes(other[0])
            left -= tensor_bytes(dit) - n_blocks * block
            if left >= n_blocks * block:
                blocks = n_blocks
                left -= n_blocks * block
            else:
                blocks = max(0, int(left / block) - STREAM_SLOTS)
                left -= (blocks + STREAM_SLOTS) * block
            self.dit = int8.stream_blocks(dit, blocks, device)
        else:
            n_blocks = blocks = len(dit.single_blocks) + len(dit.txt_mode_blocks) * 3
            left -= tensor_bytes(dit)
            self.dit = dit.to(device)
        left += te.gpu_bytes(0)
        n_layers = len(te.text.layers)
        layers = max([k for k in range(n_layers + 1) if te.gpu_bytes(k) <= left], default=0)
        left -= te.gpu_bytes(layers)
        te.place(device, layers)
        if left < 0:
            log(f"warning: {(budget / GiB) + working_memory:.1f} GB of GPU memory is not enough; expect out-of-memory errors")
        streamed = lambda s: 0 if s is None else len(s.cpu) * s.unit_bytes / GiB
        log(f"GPU memory: {free / GiB:.1f} GB free, {working_memory:.1f} GB kept for activations; on the GPU: "
            f"image model {blocks}/{n_blocks} blocks ({streamed(getattr(self.dit, 'block_stream', None)):.1f} GB streamed), "
            f"text encoder {layers}/{n_layers} layers ({streamed(self.text_encoder.stream):.1f} GB streamed)")
        self.log(f"models loaded in {time.time() - t0:.1f}s")
        if compile:
            t0 = time.time()
            self._compile()
            self.log(f"DiT compiled in {time.time() - t0:.1f}s")

    def _compile(self):
        """Compile the DiT blocks (fusing norms, rope and FP8 quantization) and warm them up on
        throwaway batches, so the first real request does not pay for compilation."""
        # text blocks keep bf16 linears and the rest are FP8, so Block.forward needs a few variants
        torch._dynamo.config.recompile_limit = 64
        dit = self.dit
        int8.compile_helpers(dynamic=True)
        for blocks in (dit.txt_mode_blocks, dit.single_blocks, *dit.content_mode_blocks.values()):
            for block in blocks:
                block.compile(dynamic=True)  # the FP8/bf16 halves; INT8 blocks use the helpers

        def sample(n_image, n_cond):
            ids = lambda n: torch.zeros(n, 4, device=self.device)
            tokens = {"image": torch.randn(n_image, 128, device=self.device),
                      "image_cond": torch.randn(n_cond, 128, device=self.device, dtype=torch.bfloat16) if n_cond else None,
                      "text": torch.randn(160, 20480, device=self.device, dtype=torch.bfloat16)}
            rope = {m: None if t is None else rope_table(ids(t.shape[0])) for m, t in tokens.items()}
            return Sample(tokens, rope, 1.0, 0.9)

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for batch in ([sample(256, 0)], [sample(1024, 256), sample(256, 0)], [sample(576, 0)] * 3):
                self.dit(batch)
        self.ae.decoder.compile(dynamic=True)
        for h, w in ((16, 16), (24, 40)):
            self.ae.decode(torch.zeros(h * w, 128, device=self.device), h, w)
        torch.cuda.synchronize()

    def _sync(self):
        torch.cuda.synchronize()

    def _error_message(self, error):
        if isinstance(error, torch.OutOfMemoryError):
            return "Not enough GPU memory for this image size here; try a smaller size."
        return super()._error_message(error)

    def _encode_image(self, image):
        return self.ae.encode(to_tensor(image).to(self.device))[0]

    @torch.inference_mode()
    def _prepare_job(self, job):
        request = job.request
        if request.refs and request.cond is None:
            self._encode_references(request)
        if job.caption not in request.text:
            tokens, ids = self.text_encoder.encode(job.caption)
            request.text[job.caption] = (tokens, rope_table(ids.float()))
        text_tokens, text_rope = request.text[job.caption]

        h, w = request.h, request.w
        job.x = noise(job.seed, h * w).to(self.device)  # bf16 values, fp32 state across the steps
        if request.box is not None and request.box.pin:
            if request.pin is None:
                weights = torch.tensor(request.box.weights(w, h), dtype=torch.float32, device=self.device)
                request.pin = (self._encode_image(request.box.source_crop()).float(), weights.reshape(-1, 1))
            job.noise = job.x
        start = start_tokens(job, self._encode_image)
        if start is not None:
            t = request.schedule[0]
            job.x = (1.0 - t) * start + t * job.x
        ids = latent_ids(h, w, self.device).float()
        ids[:, 1:3] *= IMAGE_ROPE_SCALE
        cond_tokens, cond_rope = request.cond if request.cond else (None, None)
        job.sample = Sample(
            tokens={"image": None, "image_cond": cond_tokens, "text": text_tokens},
            rope={"image": rope_table(ids), "image_cond": cond_rope, "text": text_rope},
            t=0.0, t_next=0.0,
        )

    def _encode_references(self, request):
        tokens, ropes = [], []
        for index, path in enumerate(request.refs):
            t, h, w = self.ae.encode(to_tensor(reference_image(request, path)).to(self.device))
            ids = latent_ids(h, w, self.device).float()
            ids[:, 1:3] *= IMAGE_ROPE_SCALE
            ids[:, 3] = -(1 + index)  # the l axis tells references apart
            tokens.append(t.to(torch.bfloat16))
            ropes.append(rope_table(ids))
        request.cond = (torch.cat(tokens), torch.cat(ropes))
        request.cond_tokens = request.cond[0].shape[0]

    @torch.inference_mode()
    def _step(self, batch):
        for job in batch:
            s = job.request.schedule
            job.sample.tokens["image"] = job.x
            job.sample.t, job.sample.t_next = s[job.step], s[job.step + 1]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            velocity = self.dit([job.sample for job in batch])
        for job, v in zip(batch, velocity):
            s = job.request.schedule
            job.x = job.x + (s[job.step + 1] - s[job.step]) * v
            if job.request.pin is not None:  # outside the box, back to the source noised to the new level
                source, weights = job.request.pin
                t = s[job.step + 1]
                job.x = weights * job.x + (1.0 - weights) * ((1.0 - t) * source + t * job.noise)
            job.step += 1

    @torch.inference_mode()
    def _decode(self, job):
        pixels = self.ae.decode(job.x, job.request.h, job.request.w)
        return ((pixels.float().clamp(-1, 1) + 1) * 127.5).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
