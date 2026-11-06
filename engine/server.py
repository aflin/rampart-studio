"""RIS Engine HTTP server.  Run with rampart's python:  python3r server.py [options]

  POST /generate   {"id"?, "width", "height", "steps"?, "alpha"?, "refs"?, "ref_mp"?,
                    "init"?, "init_t"?, "via"?: [{"width", "height"}], "via_t"?,
                    "box"?: [x, y, w, h], "margin"?, "box_mp"?, "pin"?,
                    "images": [{"prompt", "seed"}, ...]}                            -> {"id"}
                   box: edit only that part of refs[0] (its size then sets width and height;
                   the saved image is refs[0] with the edited box pasted in)
  GET  /status?id=ID                                                -> request state, per-image progress, files
  POST /import     {"name": "upload_<uuid>"}  (a file in --out)  -> {"file", "width", "height"}
  POST /cancel     {"id"}
  GET  /info                                                        -> loop load
  GET  /image?name=img_00001_.png                                   -> the PNG

Files are written to --out as img_NNNNN_.png.  References and init images are file names in --out or --refs.
"""

import argparse
import json
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "pylib")]  # packages are installed with pip3r --target pylib

MAC = sys.platform == "darwin"  # Apple silicon: the MLX backend; elsewhere CUDA
ROOT = os.path.dirname(HERE)  # the Rampart Image Studio checkout
WEIGHTS = os.path.join(ROOT, "weights", "flux-3-image-klein")
INT8_DIT = os.path.join(ROOT, "weights", "flux-3-image-klein-comfy", "flux3-image-klein-int8-convrot.safetensors")
FP8_DIT = os.path.join(WEIGHTS, "variants", "fp8r", "model.safetensors")


def default_dit():
    """INT8 convrot on comfy_kitchen when both are present (fastest, closest to bf16), else rowwise FP8."""
    try:
        import comfy_kitchen  # noqa: F401
    except ImportError:
        return FP8_DIT
    return INT8_DIT if os.path.exists(INT8_DIT) else FP8_DIT


def log(msg):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)


class Handler(BaseHTTPRequestHandler):
    engine = None
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def _send(self, code, body, ctype="application/json"):
        if not isinstance(body, bytes):
            body = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json_body(self):
        length = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(length) or b"{}")

    def do_GET(self):
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        if url.path == "/status":
            status = self.engine.status(q.get("id", ""))
            return self._send(200, status) if status else self._send(404, {"error": "unknown id"})
        if url.path == "/info":
            return self._send(200, self.engine.info())
        if url.path == "/image":
            name = q.get("name", "")
            if not re.fullmatch(r"img_\d+_\.png", name):
                return self._send(400, {"error": "bad name"})
            try:
                with open(os.path.join(self.engine.out_dir, name), "rb") as f:
                    return self._send(200, f.read(), "image/png")
            except OSError:
                return self._send(404, {"error": "not found"})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        url = urlparse(self.path)
        try:
            body = self._json_body()
            if url.path == "/generate":
                return self._send(200, {"id": self.engine.submit(body)})
            if url.path == "/import":
                return self._send(200, self.engine.import_upload(str(body.get("name", ""))))
            if url.path == "/cancel":
                return self._send(200, {"canceled": self.engine.cancel(str(body.get("id", "")))})
        except (ValueError, KeyError, TypeError) as error:
            return self._send(400, {"error": str(error)})
        self._send(404, {"error": "not found"})


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--listen", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8189)
    p.add_argument("--dit", default=None,
                   help="DiT weights: INT8 convrot (default, needs comfy_kitchen), rowwise FP8 "
                        "(variants/fp8r/model.safetensors, the fallback) or bf16 (model.safetensors)")
    p.add_argument("--text-encoder", default=os.path.join(WEIGHTS, "text_encoder"))
    p.add_argument("--ae", default=os.path.join(WEIGHTS, "ae.safetensors"))
    p.add_argument("--out", default=os.path.join(ROOT, "web_server", "data", "ris", "images"))
    p.add_argument("--refs", action="append", default=[], help="extra directory references may come from")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-active", type=int, default=16, help="images in progress at once (the rest wait in line)")
    p.add_argument("--pass-tokens", type=int, default=3584,
                   help="token budget of one pass, about what keeps the GPU busy: one preview set is ~3300 "
                        "tokens, a 1 MP image ~4700 (a request bigger than the budget runs alone)")
    p.add_argument("--no-compile", action="store_true", help="skip torch.compile (faster start, slower steps)")
    p.add_argument("--gpu-memory", type=float, default=None, metavar="GB",
                   help="GPU memory the engine may use, a hard limit (default: what is free when it starts, less 0.5 GB); "
                        "model weights that do not fit stream from CPU memory")
    p.add_argument("--working-memory", type=float, default=2.5, metavar="GB",
                   help="part of it kept for activations (default 2.5, enough for 4 MP images)")
    p.add_argument("--quantize", default="8", choices=["4", "8", "none"],
                   help="Mac: bits for the image model and text encoder (default 8; none: 16-bit)")
    p.add_argument("--cache-gb", type=float, default=8.0,
                   help="Mac: freed GPU memory MLX keeps for reuse (default 8)")
    args = p.parse_args()

    if MAC:
        from ris_engine.mlx_engine import MLXEngine
        log(f"MLX backend, weights: {WEIGHTS}")
        engine = Handler.engine = MLXEngine(
            WEIGHTS, args.out, args.refs, args.max_active, args.pass_tokens, log=log,
            quantize=None if args.quantize == "none" else int(args.quantize), cache_gb=args.cache_gb)
    else:
        from ris_engine.engine import Engine
        log(f"DiT weights: {args.dit or default_dit()}")
        engine = Handler.engine = Engine(
            args.dit or default_dit(), args.text_encoder, args.ae, args.out, args.refs, args.device,
            args.max_active, args.pass_tokens, not args.no_compile, log=log,
            gpu_memory=args.gpu_memory, working_memory=args.working_memory)
    server = ThreadingHTTPServer((args.listen, args.port), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, name="http", daemon=True).start()
    log(f"RIS Engine listening on http://{args.listen}:{args.port}")
    engine.run()


if __name__ == "__main__":
    main()
