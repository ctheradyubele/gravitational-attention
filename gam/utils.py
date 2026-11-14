"""Reproducibility helpers: seeding, environment capture, result files."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def git_commit() -> str:
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True)
        return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "") if out.returncode == 0 else "no-git"
    except Exception:
        return "no-git"


def environment() -> dict:
    import transformers
    env = {
        "python": sys.version.split()[0], "platform": platform.platform(), "torch": torch.__version__,
        "transformers": transformers.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "n_gpu": torch.cuda.device_count(), "git_commit": git_commit(),
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "argv": sys.argv,
    }
    try:
        import datasets
        env["datasets"] = datasets.__version__
    except ImportError:
        pass
    return env


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: str, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)  # atomic: a results file is either complete or absent


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def autocast(enabled: bool):
    if enabled and torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.autocast("cuda", dtype=dtype)
    return torch.autocast("cpu", enabled=False)
