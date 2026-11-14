"""Phase F: time and memory of one attention layer and of a full BERT-base forward/backward pass."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from gam.attention import GAMConfig, GravitationalSelfAttention, VARIANTS
from gam.utils import environment, write_json


def timeit(fn, warmup, iters, cuda):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        if cuda:
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record(); fn(); e.record(); torch.cuda.synchronize()
            times.append(s.elapsed_time(e))
        else:
            import time
            t = time.perf_counter(); fn(); times.append((time.perf_counter() - t) * 1000)
    return float(np.median(times)), float(np.percentile(times, 25)), float(np.percentile(times, 75))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--lengths", default="128,512,1024,2048")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=768)
    ap.add_argument("--heads", type=int, default=12)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--amp", action="store_true")
    a = ap.parse_args()
    cuda = torch.cuda.is_available()
    dev = torch.device("cuda" if cuda else "cpu")
    dtype = (torch.bfloat16 if cuda and torch.cuda.is_bf16_supported() else torch.float16) if a.amp and cuda else torch.float32
    rows = []
    for n in [int(x) for x in a.lengths.split(",")]:
        x = torch.randn(a.batch, n, a.hidden, device=dev, dtype=dtype, requires_grad=True)
        mask = torch.zeros(a.batch, 1, 1, n, device=dev, dtype=torch.float32)
        for variant in ("standard", "hybrid", "pure"):
            mod = GravitationalSelfAttention(a.hidden, a.heads, 0.0, VARIANTS[variant]).to(dev, dtype)
            for direction in ("forward", "forward_backward"):
                def fn():
                    y = mod(x, attention_mask=mask)[0]
                    if direction == "forward_backward":
                        y.float().sum().backward()
                if cuda:
                    torch.cuda.reset_peak_memory_stats()
                ctx = torch.no_grad() if direction == "forward" else torch.enable_grad()
                with ctx:
                    med, q1, q3 = timeit(fn, a.warmup, a.iters, cuda)
                mem = torch.cuda.max_memory_allocated() / 2 ** 20 if cuda else None
                rows.append({"n": n, "variant": variant, "direction": direction, "median_ms": med,
                             "iqr_ms": [q1, q3], "peak_mem_mb": mem})
                print(rows[-1], flush=True)
    write_json(os.path.join(a.out, "results.json"), {"phase": "benchmark", "rows": rows, "args": vars(a),
                                                     "dtype": str(dtype), "env": environment()})


if __name__ == "__main__":
    main()
