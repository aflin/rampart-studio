"""What every backend of the engine shares: requests and jobs, the queue, the loop that steps active
jobs, the HTTP-facing calls (submit, status, cancel, info, import_upload) and saving the images.

A backend (engine.Engine on CUDA, mlx_engine.MLXEngine on Apple silicon) loads the models and
implements four calls on one job or one pass:
  _prepare_job(job)  references, text and starting latents (job.state is already "running")
  _step(batch)       advance every job of the batch by one sampling step (job.step += 1)
  _decode(job)       the finished latents as an (H, W, 3) uint8 array
  _sync()            wait for queued GPU work (for timing)
and may override _error_message(error) to explain out-of-memory errors.

Each pass of the loop advances a set of jobs by one sampling step.  Which jobs go into a pass:
whole requests, least remaining work first (credited for time spent waiting, so nothing starves),
up to a token budget of about what keeps the GPU busy -- beyond that, batching adds no throughput
(on the GB10 a 1600-token pass is already as fast per token as any).  Small jobs (previews)
therefore batch together and overtake a large upscale, while large jobs run one after another,
first come first served: for equal jobs that finishes each sooner on average than sharing the
GPU between them.
"""

import collections
import concurrent.futures
import math
import os
import re
import threading
import time
import traceback
import uuid

import numpy as np
import torch  # CPU only here: the schedule and the noise, so every backend draws the same numbers
from PIL import Image, ImageOps
from PIL.PngImagePlugin import PngInfo

DOWNSAMPLING_RATIO = 16
IMAGE_ROPE_SCALE = 0.5
MIN_SIDE, MAX_SIDE = 64, 4096
MAX_IMAGE_PIXELS = 4 * 1024 * 1024
MAX_REFERENCES = 10
MULTI_REFERENCE_MAX_PIXELS = 1024 * 1024
KEEP_FINISHED = 3600  # seconds a finished request stays queryable
TOKEN_STEPS_PER_SECOND = 8500  # rough DiT throughput, to turn remaining work into seconds
WAIT_CREDIT = 0.5  # seconds of priority gained per second waited
TRACE = os.environ.get("RIS_ENGINE_TRACE") == "1"  # log the time of every pass, prepare and decode

# marks every generated image as AI-generated: PNG text, and the IPTC digital source type in XMP
# (what Google, Meta and Adobe read to label AI images)
AI_PNG_INFO = PngInfo()
AI_PNG_INFO.add_text("Software", "Rampart Image Studio")
AI_PNG_INFO.add_text("Source", "FLUX 3 Image Klein")
AI_PNG_INFO.add_text("Comment", "AI-generated image")
AI_PNG_INFO.add_itxt("XML:com.adobe.xmp",
    '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
    '<rdf:Description rdf:about="" xmlns:Iptc4xmpExt="http://iptc.org/std/Iptc4xmpExt/2008-02-29/">'
    '<Iptc4xmpExt:DigitalSourceType>http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia'
    '</Iptc4xmpExt:DigitalSourceType></rdf:Description></rdf:RDF></x:xmpmeta>')


def schedule(steps, alpha):
    t = torch.linspace(1.0, 0.0, steps + 1)
    return (alpha * t / (1.0 + (alpha - 1.0) * t)).tolist()


def noise(seed, tokens):
    """The starting noise of a job: (tokens, 128) float32 holding bf16 values."""
    return torch.randn((tokens, 128), generator=torch.Generator().manual_seed(seed)).to(torch.bfloat16).float()


def fit_reference(image, max_pixels):
    """RGB, scaled down to at most max_pixels (never up), centre-cropped to the 16-px grid."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    scale = min(1.0, math.sqrt(max_pixels / (width * height)), MAX_SIDE / max(width, height))
    if scale < 1.0:
        width, height = max(MIN_SIDE, int(width * scale)), max(MIN_SIDE, int(height * scale))
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    w, h = width // DOWNSAMPLING_RATIO * DOWNSAMPLING_RATIO, height // DOWNSAMPLING_RATIO * DOWNSAMPLING_RATIO
    left, top = (width - w) // 2, (height - h) // 2
    return image.crop((left, top, left + w, top + h))


def resize_reference(image, pixels):
    """RGB, resized up or down to about ``pixels`` with sides on the 16-px grid (Rampart Image Studio's upscale)."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    scale = math.sqrt(pixels / (image.width * image.height))
    snap = lambda v: max(MIN_SIDE, round(v * scale / DOWNSAMPLING_RATIO) * DOWNSAMPLING_RATIO)
    return image.resize((snap(image.width), snap(image.height)), Image.Resampling.LANCZOS)


class BoxEdit:
    """An edit confined to a box of the source image: the box plus a margin of surroundings is cropped,
    edited at about ``mp`` megapixels (more detail for a small box), scaled back and pasted into the
    source, fading from the new pixels at the box to the source's at the edge of the margin.  pin (the
    default): also hold the margin's latents to the source's, noised to each step's level, so the model
    cannot redraw what the fade blends across (unpinned, edges there come out doubled)."""

    def __init__(self, path, box, margin=None, mp=1.0, pin=True):
        self.path, self.pin = path, bool(pin)
        with Image.open(path) as image:
            self.source_size = ImageOps.exif_transpose(image).size
        width, height = self.source_size
        x, y, w, h = (int(round(float(v))) for v in box)
        x, y = max(0, x), max(0, y)
        w, h = min(w, width - x), min(h, height - y)
        if w < 16 or h < 16:
            raise ValueError("box must lie inside the image and be at least 16 px a side")
        self.box = (x, y, x + w, y + h)
        m = max(16, round(max(w, h) / 8)) if margin is None else max(0, int(margin))
        self.crop = (max(0, x - m), max(0, y - m), min(width, x + w + m), min(height, y + h + m))
        cw, ch = self.crop[2] - self.crop[0], self.crop[3] - self.crop[1]
        # edited at about mp megapixels, but never more than 4x the crop's size a side
        pixels = min(float(mp) * 1024 * 1024, cw * ch * 16, MAX_IMAGE_PIXELS)
        scale = math.sqrt(pixels / (cw * ch))
        snap = lambda v: min(MAX_SIDE, max(MIN_SIDE, round(v * scale / DOWNSAMPLING_RATIO) * DOWNSAMPLING_RATIO))
        self.size = (snap(cw), snap(ch))

    def source_crop(self):
        """The crop of the source at the size it is edited at (the reference, and pin's latents)."""
        with Image.open(self.path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            return image.crop(self.crop).resize(self.size, Image.Resampling.LANCZOS)

    def weights(self, width, height):
        """(height, width) float array over the crop at width x height: 1 inside the box, falling
        linearly to 0 across the margin (a margin cut short by the image's edge has no fall)."""
        cx0, cy0, cx1, cy1 = self.crop
        bx0, by0, bx1, by1 = self.box
        sx, sy = width / (cx1 - cx0), height / (cy1 - cy0)

        def ramp(n, scale, c0, b0, b1, c1):
            v = (np.arange(n) + 0.5) / scale + c0  # source coordinate of each pixel's centre
            up = np.clip((v - c0) / (b0 - c0), 0, 1) if b0 > c0 else np.ones(n)
            down = np.clip((c1 - v) / (c1 - b1), 0, 1) if c1 > b1 else np.ones(n)
            return np.minimum(up, down)

        return np.outer(ramp(height, sy, cy0, by0, by1, cy1), ramp(width, sx, cx0, bx0, bx1, cx1))

    def paste(self, array):
        """The edited crop (H, W, 3) uint8 pasted into the source: the full-size result."""
        with Image.open(self.path) as image:
            source = ImageOps.exif_transpose(image).convert("RGB")
        cw, ch = self.crop[2] - self.crop[0], self.crop[3] - self.crop[1]
        edited = np.asarray(Image.fromarray(array).resize((cw, ch), Image.Resampling.LANCZOS), dtype=np.float32)
        region = np.asarray(source.crop(self.crop), dtype=np.float32)
        a = self.weights(cw, ch)[..., None]
        source.paste(Image.fromarray((a * edited + (1 - a) * region).round().astype(np.uint8)), self.crop[:2])
        return np.asarray(source)


def reference_image(request, path):
    """A reference image as the request asks for it: resized to ref_mp, or fitted to the limits."""
    if request.box is not None and path == request.box.path:
        return request.box.source_crop()
    with Image.open(path) as image:
        if request.ref_mp:
            return resize_reference(image, float(request.ref_mp) * 1024 * 1024)
        if len(request.refs) > 1:
            return fit_reference(image, MULTI_REFERENCE_MAX_PIXELS)
        return fit_reference(image, MAX_IMAGE_PIXELS)


def start_tokens(job, encode):
    """The latent tokens a job starts from at its request's current size, or None to start from
    noise: its own result at the previous size of a staged request, or the request's init image
    (encoded once for all its jobs).  encode: the backend's PIL image -> tokens."""
    request = job.request
    if job.init_image is not None:
        image, job.init_image = job.init_image, None
        return encode(image.resize((request.width, request.height), Image.Resampling.LANCZOS))
    if not request.init:
        return None
    if request.init_tokens is None:
        with Image.open(request.init) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image = image.resize((request.width, request.height), Image.Resampling.LANCZOS)
        request.init_tokens = encode(image)
    return request.init_tokens


class Job:
    """One image."""

    def __init__(self, request, index, caption, seed):
        self.request, self.index, self.caption, self.seed = request, index, caption, seed
        self.state = "queued"
        self.step = 0
        self.file = None
        self.error = None
        self.started = self.finished = None
        self.sample = None  # the backend's inputs for this job, once prepared
        self.x = None  # fp32 image tokens being denoised
        self.init_image = None  # this job's result at the previous size of a staged request
        self.noise = None  # the starting noise, kept for a pinned box edit

    @property
    def tokens(self):
        return self.request.h * self.request.w + self.request.extra_tokens(self)

    def priority(self, now):
        """Lower runs first: estimated seconds of work left, minus credit for time waited."""
        left = self.tokens * (self.request.steps - self.step) / TOKEN_STEPS_PER_SECOND
        return left - WAIT_CREDIT * (now - self.request.created)


class Request:
    """What a client submitted: one or more images of one size, sharing references."""

    def __init__(self, spec, rid):
        self.id = rid
        self.created = time.time()
        # via: sizes rendered first, each result the start of the next size, noised to via_t (for big
        # upscales: one starting from a small image at the final size comes out blurred)
        self.sizes = [(int(v["width"]), int(v["height"])) for v in spec.get("via") or []]
        self.sizes.append((int(spec["width"]), int(spec["height"])))
        for width, height in self.sizes:
            if width % DOWNSAMPLING_RATIO or height % DOWNSAMPLING_RATIO:
                raise ValueError(f"width and height must be multiples of {DOWNSAMPLING_RATIO}")
            if min(width, height) < MIN_SIDE or max(width, height) > MAX_SIDE or width * height > MAX_IMAGE_PIXELS:
                raise ValueError(f"size must be {MIN_SIDE}-{MAX_SIDE} px a side and at most 4 MP")
        self.full_steps = int(spec.get("steps", 16))
        self.alpha = float(spec.get("alpha", 16.0))
        if not 1 <= self.full_steps <= 100:
            raise ValueError("steps must be 1-100")
        # init: an image to start from instead of pure noise, noised to init_t; only the steps below init_t run
        self.init = spec.get("init")
        starts = [float(spec.get("init_t", 0.9)) if self.init else None]
        starts += [float(spec.get("via_t", 0.85))] * (len(self.sizes) - 1)
        if any(t is not None and not 0.0 < t < 1.0 for t in starts):
            raise ValueError("init_t and via_t must be between 0 and 1")
        self.schedules = []
        for t0 in starts:
            full = schedule(self.full_steps, self.alpha)
            self.schedules.append(full if t0 is None else [t0] + [t for t in full if t < t0])
        self.stage = -1
        self.next_stage()
        self.refs = list(spec.get("refs") or [])
        if len(self.refs) > MAX_REFERENCES:
            raise ValueError(f"at most {MAX_REFERENCES} references")
        self.ref_mp = spec.get("ref_mp")
        images = spec.get("images") or []
        if not images:
            raise ValueError("images: give at least one {prompt, seed}")
        self.jobs = [Job(self, i, str(item["prompt"]), int(item.get("seed", 0)) % 2**64)
                     for i, item in enumerate(images)]
        self.cond = None  # the backend's encoded references, shared by all jobs
        self.text = {}  # caption -> the backend's encoded text, (tokens, ...) with tokens (..., L, D)
        self.cond_tokens = 0
        self.init_tokens = None  # the init image's latent tokens
        self.box = None  # a BoxEdit, for an edit confined to a box of refs[0]
        self.pin = None  # the backend's (source latents, weights) holding a box edit's margin

    def next_stage(self):
        self.stage += 1
        self.width, self.height = self.sizes[self.stage]
        self.h, self.w = self.height // DOWNSAMPLING_RATIO, self.width // DOWNSAMPLING_RATIO
        self.schedule = self.schedules[self.stage]
        self.steps = len(self.schedule) - 1

    @property
    def last_stage(self):
        return self.stage == len(self.sizes) - 1

    @property
    def steps_done_before(self):
        """steps of the stages before the current one"""
        return sum(len(s) - 1 for s in self.schedules[:self.stage])

    def extra_tokens(self, job):
        text = self.text.get(job.caption)
        return self.cond_tokens + (text[0].shape[-2] if text else 0)

    @property
    def state(self):
        states = {j.state for j in self.jobs}
        if states & {"saving", "between"}:
            states.add("running")
        for s in ("error", "running", "queued"):
            if s in states:
                return s
        return "canceled" if "canceled" in states else "done"


class EngineBase:
    def __init__(self, out_dir, ref_dirs=(), max_active=16, pass_tokens=3584, log=print):
        self.out_dir, self.log = out_dir, log
        self.ref_dirs = [os.path.realpath(d) for d in (out_dir, *ref_dirs)]
        self.max_active, self.pass_tokens = max_active, pass_tokens
        os.makedirs(out_dir, exist_ok=True)
        # image numbers are never reused (clients may cache an image by its name): the last one
        # handed out is kept in a file, since the highest file left can be lower after deletions
        self.number_file = os.path.join(out_dir, ".last_image_number")
        numbers = [int(m.group(1)) for f in os.listdir(out_dir) if (m := re.match(r"img_(\d+)_\.png$", f))]
        try:
            with open(self.number_file) as f:
                numbers.append(int(f.read()))
        except (OSError, ValueError):
            pass
        self.last_number = max(numbers, default=0)
        self.number_lock = threading.Lock()

        self.cond = threading.Condition()
        self.queue = collections.deque()  # jobs not prepared yet
        self.active = []  # prepared jobs (text encoded, noise drawn); each pass steps some of them
        self.requests = {}
        self.passes = 0
        self.saver = concurrent.futures.ThreadPoolExecutor(2, thread_name_prefix="png")  # PNG encoding is CPU work

    def run(self):
        """The GPU loop; never returns.  Call it from the main thread: compiled code is only
        reused reliably there (from another thread, new batch shapes ran uncompiled, 3x slower)."""
        self._loop()

    # --- the backend ---------------------------------------------------------------

    def _prepare_job(self, job):
        raise NotImplementedError

    def _step(self, batch):
        raise NotImplementedError

    def _decode(self, job):
        raise NotImplementedError

    def _sync(self):
        pass

    def _error_message(self, error):
        return str(error) or error.__class__.__name__

    # --- client side (any thread) -------------------------------------------------

    def submit(self, spec):
        rid = str(spec.get("id") or uuid.uuid4())
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", rid):
            raise ValueError("bad id")
        box = None
        if spec.get("box") is not None:
            if not spec.get("refs"):
                raise ValueError("a box edit needs the image to edit as refs[0]")
            box = BoxEdit(self._ref_path(spec["refs"][0]), spec["box"], spec.get("margin"),
                          spec.get("box_mp", 1.0), spec.get("pin", True))
            spec = dict(spec, width=box.size[0], height=box.size[1], via=None, init=None)
        request = Request(spec, rid)
        request.box = box
        request.refs = [self._ref_path(r) for r in request.refs]
        if request.init:
            request.init = self._ref_path(request.init)
        with self.cond:
            if rid in self.requests:
                raise ValueError("id already used")
            self.requests[rid] = request
            self.queue.extend(request.jobs)
            self.cond.notify()
        return rid

    def cancel(self, rid):
        with self.cond:
            request = self.requests.get(rid)
            if request is None:
                return False
            for job in request.jobs:
                if job.state in ("queued", "running", "between"):
                    job.state = "canceled"
            self.queue = collections.deque(j for j in self.queue if j.state == "queued")
            return True

    def status(self, rid):
        with self.cond:
            request = self.requests.get(rid)
            if request is None:
                return None
            # a prepared job that has not had a step yet is still waiting, as far as clients go
            waiting = lambda j: request.stage == 0 and (j.state == "queued" or (j.state == "running" and j.step == 0))
            done = request.steps_done_before  # steps count across all of a staged request's sizes
            state_of = lambda j: "queued" if waiting(j) else "running" if j.state == "between" else j.state
            jobs = [{"index": j.index, "state": state_of(j), "step": done + j.step, "file": j.file,
                     "error": j.error,
                     "seconds": round(j.finished - j.started, 2) if j.finished and j.started else None}
                    for j in request.jobs]
            state = request.state
            if state == "running" and all(waiting(j) for j in request.jobs if j.state != "done"):
                state = "queued" if not any(j.state in ("saving", "done") for j in request.jobs) else state
            # requests with work in progress, or that came in earlier
            ahead = len({j.request.id for j in (*self.active, *self.queue)
                         if j.request is not request and (j.step > 0 or j.request.created < request.created)})
            width, height = request.sizes[-1]
            out = {"id": rid, "state": state, "width": width, "height": height,
                   "steps": sum(len(s) - 1 for s in request.schedules), "jobs": jobs,
                   "files": [j["file"] for j in jobs] if state == "done" else []}
            if state == "queued":
                out["position"] = ahead
            errors = [j["error"] for j in jobs if j["error"]]
            if errors:
                out["error"] = errors[0]
            return out

    def info(self):
        with self.cond:
            return {"active": len(self.active), "queued": len(self.queue), "passes": self.passes,
                    "active_tokens": sum(j.tokens for j in self.active),
                    "max_active": self.max_active, "pass_tokens": self.pass_tokens}

    def _new_image_name(self):
        with self.number_lock:
            self.last_number += 1
            while os.path.exists(os.path.join(self.out_dir, f"img_{self.last_number:05d}_.png")):
                self.last_number += 1  # put there by something else since the engine started
            temp = self.number_file + ".tmp"
            with open(temp, "w") as f:
                f.write(str(self.last_number))
            os.replace(temp, self.number_file)
            return f"img_{self.last_number:05d}_.png"

    def import_upload(self, name):
        """Turn an uploaded file in the output directory (upload_<uuid>, written by Rampart Image Studio)
        into a gallery image: EXIF rotation applied, RGB, at most 4 MP, saved as the next
        img_NNNNN_.png.  The upload is removed either way.  -> {file, width, height}"""
        if not re.fullmatch(r"upload_[0-9a-f-]{36}", name):
            raise ValueError("bad upload name")
        path = os.path.join(self.out_dir, name)
        try:
            try:
                with Image.open(path) as image:
                    image = fit_reference(image, MAX_IMAGE_PIXELS)
            except (OSError, Image.DecompressionBombError):
                raise ValueError("That file could not be read as an image.") from None
            if min(image.size) < MIN_SIDE:
                raise ValueError(f"the image is too small (at least {MIN_SIDE} pixels a side)")
            file = self._new_image_name()
            image.save(os.path.join(self.out_dir, file), compress_level=4)
            return {"file": file, "width": image.width, "height": image.height}
        finally:
            if os.path.exists(path):
                os.remove(path)

    def _ref_path(self, name):
        """A reference is a file name inside the output directory or one of the reference directories."""
        for root in self.ref_dirs:
            path = os.path.realpath(os.path.join(root, name))
            if path.startswith(root + os.sep) and os.path.isfile(path):
                return path
        raise ValueError(f"reference not found: {name}")

    # --- the GPU loop --------------------------------------------------------------

    def _admit(self):
        """Jobs to prepare now, first come first served, up to max_active prepared jobs."""
        admitted = []
        while self.queue and len(self.active) + len(admitted) < self.max_active:
            admitted.append(self.queue.popleft())
        return admitted

    def _select(self):
        """The jobs for the next pass: whole requests by priority, within the token budget (always at
        least one).  A request's images go together so that none of them lags behind the others."""
        now = time.time()
        groups = {}
        for job in self.active:
            groups.setdefault(job.request.id, []).append(job)
        batch, tokens = [], 0
        for jobs in sorted(groups.values(), key=lambda js: min(j.priority(now) for j in js)):
            need = sum(j.tokens for j in jobs)
            if batch and tokens + need > self.pass_tokens:
                continue  # a smaller request further down may still fit
            batch.extend(jobs)
            tokens += need
        return batch

    def _loop(self):
        while True:
            with self.cond:
                while not self.queue and not self.active:
                    self._expire()
                    self.cond.wait(60)
                admitted = self._admit()
            for job in admitted:
                self._run_guarded(job, self._prepare)
            with self.cond:
                self.active.extend(j for j in admitted if j.state == "running")
                self._prune()
                batch = self._select()
            if batch:
                try:
                    t0 = time.time()
                    self._step(batch)
                    self.passes += 1
                    if TRACE:
                        self._sync()
                        self.log(f"trace pass {len(batch)} jobs, {sum(j.tokens for j in batch)} tokens: "
                                 f"{(time.time() - t0) * 1000:.0f} ms")
                except Exception as error:
                    self.log(traceback.format_exc())
                    for job in batch:
                        self._fail(job, error)
            for job in batch:
                if job.state == "running" and job.step == job.request.steps:
                    self._run_guarded(job, self._finish)
            with self.cond:
                self._prune()

    def _prune(self):
        """Drop jobs no longer running from the active ones (self.cond held), freeing what a canceled
        job holds: its latents, and its request's encodings once nothing of the request is left to run."""
        for job in self.active:
            if job.state == "canceled":
                job.sample = job.x = job.noise = job.init_image = None
                request = job.request
                if not any(j.state in ("queued", "running", "between") for j in request.jobs):
                    request.text.clear()
                    request.cond = request.init_tokens = request.pin = None
        self.active = [j for j in self.active if j.state == "running"]

    def _run_guarded(self, job, fn):
        try:
            t0 = time.time()
            fn(job)
            if TRACE:
                self._sync()
                self.log(f"trace {fn.__name__} {job.request.width}x{job.request.height}: {(time.time() - t0) * 1000:.0f} ms")
        except Exception as error:
            self.log(traceback.format_exc())
            self._fail(job, error)

    def _fail(self, job, error):
        if job.state in ("queued", "running"):
            job.state, job.error = "error", self._error_message(error)
            job.sample = job.x = None

    def _expire(self):
        now = time.time()
        for rid, request in list(self.requests.items()):
            if request.state not in ("queued", "running") and now - request.created > KEEP_FINISHED:
                del self.requests[rid]

    def _prepare(self, job):
        if job.state != "queued":
            return
        job.state, job.started = "running", job.started or time.time()
        self._prepare_job(job)

    def _finish(self, job):
        request = job.request
        array = self._decode(job)
        job.sample = job.x = job.noise = None
        if not request.last_stage:  # the next size starts from this image once every job is here
            job.init_image, job.state = Image.fromarray(array), "between"
            with self.cond:
                if all(j.state in ("between", "error", "canceled") for j in request.jobs):
                    request.next_stage()
                    for j in request.jobs:
                        if j.state == "between":
                            j.state, j.step = "queued", 0
                            self.queue.append(j)
            return
        job.state = "saving"
        if all(j.state not in ("queued", "running") for j in request.jobs):
            request.text.clear()
            request.cond = request.init_tokens = request.pin = None
        self.saver.submit(self._save, job, array)

    def _save(self, job, array):
        try:
            if job.request.box is not None:
                array = job.request.box.paste(array)
            name = self._new_image_name()
            Image.fromarray(array).save(os.path.join(self.out_dir, name), compress_level=4, pnginfo=AI_PNG_INFO)
            job.file, job.state, job.finished = name, "done", time.time()
        except Exception as error:
            self.log(traceback.format_exc())
            job.state, job.error = "error", str(error)
