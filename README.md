# Rampart Image Studio

A web front end and image server for Black Forest Labs' FLUX 3 Image Klein, running on
[Rampart](https://rampart.dev):

- **Rampart Image Studio** (`web_server/`): a Rampart web app. Prompts are expanded by BFL's
  prompt upsampler LLM through `rampart-llamacpp`, previews come four at a time, and any
  image can be upscaled or edited with a prompt. Images can also be uploaded (button or
  drag and drop) to upscale or edit them. Edit prompts go to the model as written, since the
  prompt enhancer cannot see images in this setup.
- **RIS Engine** (`engine/`), Rampart Image Studio's engine: an MIT-licensed image server
  written in Python and run with Rampart's bundled Python (`python3r`). It loads the models
  once and runs one shared GPU loop that steps several users' images together, so previews
  overtake a large upscale and small jobs batch into one pass.

Everything lives in this directory: the code is in git, while `setup.sh` installs the
Python packages into `engine/pylib/` and the weights into `weights/`, and uses an installed
Rampart (or, once Rampart's installer supports it, installs one into `rampart/`).

## Requirements

- Rampart 0.7.3 or newer. `setup.sh` adds the modules Rampart Image Studio needs (`rampart-python`, and
  `rampart-langtools-cu13` on Linux or `rampart-langtools` on a Mac) when they are missing.
- **Linux** with an NVIDIA GPU and a driver from the 580 series or newer, 8 GB or more,
  or two GPUs. The prompt enhancer needs about 6.3 GB and loads first; the engine then uses
  whatever GPU memory is left. Weights that do not fit (the image model's blocks first get the
  GPU, then the text encoder's layers) stay in CPU memory and stream to the GPU as they are
  needed, which costs speed but gives identical images. Everything fits on the GPU from about
  20 GB; one 12 GB card runs everything with image steps up to about 2x slower. See
  `studio.conf.example` for 8 GB cards and for two-GPU setups.
  Tested on a DGX Spark (GB10), an RTX 4070 Ti alone, and an RTX 4070 Ti + RTX 3060.
  No CUDA toolkit is needed: the engine and the prompt enhancer both use the CUDA 13 and cuDNN
  libraries that come with PyTorch's packages in `engine/pylib/`, ahead of any toolkit installed
  on the system (`engine/engine.sh` and `studio.sh` put them first on `LD_LIBRARY_PATH`).
  A C compiler (gcc or clang) lets the engine compile its GPU kernels (`torch.compile`) the
  first time they run; without one it runs them uncompiled, about 10% slower.
- RAM: about 16 GB free for the weights the engine keeps in CPU memory.
- **macOS** on Apple silicon, with about 30 GB of memory free: the engine runs on MLX (through
  mflux), with the image model and text encoder quantized to 8 bits as they load (`--quantize`),
  and the prompt enhancer on Metal. Tested on an M1 Ultra (64 GB), where images take about 6-7x
  as long as on a DGX Spark. Studio and the engine API are the same as on Linux.
- Disk: about 30 GB, for packages and weights (about 26 GB on Linux, 25 GB on a Mac), plus
  room for the images you keep (1.5-2 MB per 1 MP PNG).

## Install

    git clone https://github.com/aflin/rampart-studio && cd rampart-studio
    ./setup.sh                  # packages + weights for this machine; safe to re-run
    ./studio.sh start           # then open http://127.0.0.1:8088/studio/

`setup.sh` first asks for the web address: the port (default 8088) and the address to listen
on (default 127.0.0.1, this machine only; 0.0.0.0 for every interface, so other machines can
connect). Run again, it offers the current settings and changes only what you change.
`--port` and `--host` set them without asking, as when it runs without a terminal.
`setup.sh --help` lists its options. `git pull` followed by `./setup.sh` picks up new
package pins or weights.

`setup.sh` checks for the GPU driver first, then looks for Rampart 0.7.3 or newer: the one it
chose before (`RAMPART=` in `studio.conf`), `./rampart/`, the `PATH`, `~/.rampart/` and
`/usr/local/rampart/`, in that order (or set `RAMPART=/path/to/bin/rampart`). It installs missing
modules into that Rampart, and if you cannot write to it, prints the `sudo` command instead.
With no Rampart at all, it installs one into `./rampart/` without changing your `PATH` or shell
files, once Rampart's installer has a `--prefix` option; until then it asks you to install
Rampart with `curl -fsSL https://get.rampart.dev/ | sh` and run it again.
`./setup.sh --own-rampart` installs into `./rampart/` even when another Rampart is found, so
Rampart Image Studio uses no Rampart or CUDA libraries from outside its directory.

Weights are downloaded from public Hugging Face mirrors of BFL's release (unsloth,
ml-resources, SD-Next, DeepBeepMeep), so no Hugging Face login is needed. Every file is
checked against the SHA-256 that BFL published, and the hashes and sources are listed in
`weights.manifest`. If every mirror of a file disappears, set `HF_TOKEN` to a token whose
account accepted the license on the `black-forest-labs` model pages, and `setup.sh` will
fetch that file from BFL directly.

## Run

    ./studio.sh start | stop | restart | status

Settings are in `studio.conf`, which `setup.sh` creates from `studio.conf.example`:
the web and engine ports, the address to listen on (`HOST`), and which GPU each part uses
(for example the enhancer on one card and the engine on another). The two parts can also be
run separately: `engine/engine.sh` (see `engine/engine.sh help`) and
`rampart web_server/web_server_conf.js`.

Images and job records are kept in `web_server/data/ris/`, and logs in
`web_server/logs/` and `engine/engine.log`.

## AI-generated marking

Every generated PNG is marked as AI-generated, with text chunks and the IPTC digital source
type `trainedAlgorithmicMedia` in its XMP metadata. Uploaded images are left as they are.

## Layout

    setup.sh                 packages and weights
    studio.sh                start/stop both parts
    studio.conf.example      per-machine settings (copied to studio.conf)
    weights.manifest         every weight file: SHA-256, size, mirrors
    engine/                  RIS Engine (MIT)
      server.py              HTTP API on 127.0.0.1:8189
      engine.sh              start/stop/status for the engine alone
      ris_engine/            base.py: requests, queue and the shared GPU loop;
                             engine.py: the CUDA backend (model code alongside);
                             mlx_engine.py: the Apple silicon backend (mflux)
      bench.py, prof.py      load test and per-step profiler
    web_server/              Rampart web server
      web_server_conf.js     server settings, loads the prompt enhancer
      apps/ris.js            the Rampart Image Studio API (/apps/ris/...)
      html/studio/index.html the page
    weights/                 downloaded by setup.sh (not in git)
    rampart/                 a Rampart of its own, when setup.sh installs one (not in git)

## Engine API

    POST /generate  {"id"?, "width", "height", "steps"?: 16, "alpha"?: 16,
                     "refs"?: [file], "ref_mp"?: megapixels,
                     "init"?: file, "init_t"?: 0.9,
                     "via"?: [{"width", "height"}, ...], "via_t"?: 0.85,
                     "images": [{"prompt", "seed"}, ...]}            -> {"id"}
    GET  /status?id=   -> state (queued|running|done|error), position, per-image steps, files
    POST /import       {"name": "upload_<uuid>"}  -> {"file", "width", "height"}
    POST /cancel       {"id"}
    GET  /info         -> loop load
    GET  /image?name=  -> PNG

`refs` are reference images for editing. `init` starts from an existing image noised to
`init_t` instead of from pure noise, and runs only the steps below that level. `via` lists
sizes to render first: each result is scaled up and noised to `via_t` to start the next, which
keeps large upscales sharp (Studio upscales a 256 px preview to 1 MP at `init_t` 0.941, then
to 2 or 4 MP at `via_t` 0.85). `steps` and per-image `step` in the status count all sizes. File names
refer to the engine's output directory.

## Licenses

The code is MIT licensed (`LICENSE`). Parts of the engine's model code are derived from
BFL's Apache-2.0 reference implementation; `NOTICE` lists them.

FLUX is a trademark of Black Forest Labs. Rampart Image Studio is not made or endorsed by
Black Forest Labs; it names the model only to say what it runs.

The model weights are not part of this repository. FLUX 3 Image Klein is released by Black
Forest Labs under the **FLUX Kommunity License**, which allows non-commercial use, plus
commercial use of outputs by qualifying users (under US$5M annual revenue) subject to its
terms, including content filtering or review and disclosure that images are AI-generated.
Downloading the weights from a mirror does not change those terms. Read the license at
https://huggingface.co/black-forest-labs/flux-3-image-klein before you use them. The Qwen3-VL
text encoder is Apache-2.0.
