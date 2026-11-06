"""Submit jobs to a running RIS Engine and time them.

  python3r bench.py [--url http://127.0.0.1:8189] [--users N] [--images K] [--size WxH] [--steps S] prompt
"""

import argparse
import json
import threading
import time
import urllib.request


def call(url, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def run_user(args, user, results):
    w, h = map(int, args.size.split("x"))
    t0 = time.time()
    rid = call(args.url, "/generate", {"width": w, "height": h, "steps": args.steps,
                                       "images": [{"prompt": args.prompt, "seed": args.seed + user * 100 + i}
                                                  for i in range(args.images)]})["id"]
    first_step = None
    while True:
        s = call(args.url, f"/status?id={rid}")
        if first_step is None and any(j["step"] > 0 for j in s["jobs"]):
            first_step = time.time() - t0
        if s["state"] not in ("queued", "running"):
            break
        time.sleep(0.1)
    results[user] = (time.time() - t0, first_step, s)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("prompt")
    p.add_argument("--url", default="http://127.0.0.1:8189")
    p.add_argument("--users", type=int, default=1)
    p.add_argument("--images", type=int, default=1)
    p.add_argument("--size", default="1024x1024")
    p.add_argument("--steps", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    results = {}
    t0 = time.time()
    threads = [threading.Thread(target=run_user, args=(args, u, results)) for u in range(args.users)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for user in sorted(results):
        total, first, s = results[user]
        print(f"user {user}: {s['state']:6} {total:6.1f}s  (first step after {first or 0:.1f}s)  "
              f"{' '.join(f or '-' for f in s['files'])}  {s.get('error', '')}")
    print(f"{args.users} user(s) x {args.images} image(s) at {args.size}: all done in {time.time() - t0:.1f}s")
    print(call(args.url, "/info"))


if __name__ == "__main__":
    main()
