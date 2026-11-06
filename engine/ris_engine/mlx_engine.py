"""The Apple silicon backend: the models in MLX, through mflux's FLUX 3 Image Klein implementation
(the diffusers-format weights in weights/flux-3-image-klein, quantized as it loads).  The queue,
the loop and the HTTP-facing calls are in base.py, as for the CUDA backend.

Each job is its own transformer call: mflux's transformer has no per-sample attention for packed
batches, and batching same-size previews gained only about 5% on an M1 Ultra.  Compute is
float16, which the GPU runs natively (bfloat16 was slower there).  The noise and the schedule come
from base.py, so a seed starts from the same noise as on the CUDA backend.
"""

import time

import mlx.core as mx
import numpy as np
from mflux.models.flux3.latent_creator.flux3_latent_creator import Flux3LatentCreator
from mflux.models.flux3.latent_creator.flux3_reference_images import Flux3ReferenceImages
from mflux.models.flux3.model.flux3_text_encoder.prompt_encoder import Flux3PromptEncoder
from mflux.models.flux3.variants import Flux3Klein

from .base import EngineBase, noise, reference_image, start_tokens

GiB = 1 << 30


def to_array(image):
    """PIL RGB -> (1, 3, H, W) float32 in [-1, 1]"""
    return mx.array(Flux3ReferenceImages.to_array(image))


class MLXEngine(EngineBase):
    """quantize: bits for the transformer and text encoder weights (8: same speed as 16-bit on an
    M1 Ultra, half the memory).  cache_gb: how much freed GPU memory MLX keeps for reuse; it grows to
    the working set otherwise (about 30 GB at 4 MP), crowding other programs on a shared Mac."""

    def __init__(self, model_path, out_dir, ref_dirs=(), max_active=16, pass_tokens=3584, log=print,
                 quantize=8, cache_gb=8.0, warm_up=True):
        super().__init__(out_dir, ref_dirs, max_active, pass_tokens, log)
        mx.set_cache_limit(int(cache_gb * GiB))
        t0 = time.time()
        self.model = Flux3Klein(quantize=quantize, model_path=model_path, compute_precision=mx.float16)
        self.transformer, self.vae = self.model.transformer, self.model.vae
        self.log(f"models loaded in {time.time() - t0:.1f}s ({'q' + str(quantize) if quantize else '16-bit'}, "
                 f"MLX cache limit {cache_gb:g} GB)")
        if warm_up:  # Metal compiles its kernels on first use; do that before the first request
            t0 = time.time()
            text = self._encode_text("warm up")
            for size in (256, 1024):
                h = w = size // 16
                x = self.transformer(hidden_states=mx.zeros((1, h * w, 128)), img_ids=Flux3LatentCreator.prepare_ids(h, w),
                                     encoder_hidden_states=text[0], txt_ids=text[1],
                                     timestep=mx.array([1.0]), target_timestep=mx.array([0.9]))
                mx.eval(Flux3LatentCreator.decode_tokens(self.vae, x, h, w))
            self.log(f"warmed up in {time.time() - t0:.1f}s")

    def _sync(self):
        mx.synchronize()

    def _error_message(self, error):
        if "malloc" in str(error) or "memory" in str(error).lower():
            return "Not enough GPU memory for this image size here; try a smaller size."
        return super()._error_message(error)

    def _encode_text(self, caption):
        tokens, ids = Flux3PromptEncoder.encode_prompt(prompt=caption, tokenizer=self.model.tokenizers["qwen3_vl"],
                                                       text_encoder=self.model.text_encoder)
        mx.eval(tokens)
        return tokens, ids  # (1, L, 20480) bfloat16, (L, 4)

    def _encode_image(self, image):
        tokens = Flux3LatentCreator.encode_tokens(self.vae, to_array(image)).astype(mx.float32)
        mx.eval(tokens)
        return tokens  # (1, h*w, 128)

    def _prepare_job(self, job):
        request = job.request
        if request.refs and request.cond is None:
            tokens, ids = [], []
            for index, path in enumerate(request.refs):
                image = reference_image(request, path)
                tokens.append(self._encode_image(image).astype(mx.bfloat16))
                ids.append(Flux3LatentCreator.prepare_ids(image.height // 16, image.width // 16, layer=-(1 + index)))
            request.cond = (mx.concatenate(tokens, axis=1), mx.concatenate(ids, axis=0))
            request.cond_tokens = request.cond[0].shape[1]
        if job.caption not in request.text:
            request.text[job.caption] = self._encode_text(job.caption)

        h, w = request.h, request.w
        job.x = mx.array(noise(job.seed, h * w).numpy())[None]  # bf16 values, fp32 state across the steps
        if request.box is not None and request.box.pin:
            if request.pin is None:
                weights = mx.array(request.box.weights(w, h), dtype=mx.float32).reshape(1, -1, 1)
                request.pin = (self._encode_image(request.box.source_crop()), weights)
            job.noise = job.x
        start = start_tokens(job, self._encode_image)
        if start is not None:
            t = request.schedule[0]
            job.x = (1.0 - t) * start + t * job.x
        mx.eval(job.x)
        job.sample = Flux3LatentCreator.prepare_ids(h, w)

    def _step(self, batch):
        for job in batch:
            request = job.request
            s = request.schedule
            text_tokens, text_ids = request.text[job.caption]
            cond_tokens, cond_ids = request.cond if request.cond else (None, None)
            velocity = self.transformer(
                hidden_states=job.x, img_ids=job.sample,
                encoder_hidden_states=text_tokens, txt_ids=text_ids,
                timestep=mx.array([s[job.step]], dtype=mx.float32),
                target_timestep=mx.array([s[job.step + 1]], dtype=mx.float32),
                cond_hidden_states=cond_tokens, cond_ids=cond_ids,
            )
            job.x = job.x + (s[job.step + 1] - s[job.step]) * velocity.astype(mx.float32)
            if request.pin is not None:  # outside the box, back to the source noised to the new level
                source, weights = request.pin
                t = s[job.step + 1]
                job.x = weights * job.x + (1.0 - weights) * ((1.0 - t) * source + t * job.noise)
            mx.eval(job.x)
            job.step += 1

    def _decode(self, job):
        pixels = Flux3LatentCreator.decode_tokens(self.vae, job.x, job.request.h, job.request.w)[0]
        pixels = ((mx.clip(pixels.astype(mx.float32), -1, 1) + 1) * 127.5).astype(mx.uint8)
        return np.array(pixels.transpose(1, 2, 0))
