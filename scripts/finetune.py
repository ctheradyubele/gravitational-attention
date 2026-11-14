"""Phases B-D: fine-tune one variant on one task with one learning rate and one seed.

Model selection (best checkpoint) uses a fixed 10% held-out split of the training data.
The official GLUE dev set (or e-SNLI test) is evaluated exactly once, at the selected checkpoint.
"""
import argparse
import copy
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from gam.attention import gam_modules, lambda_table, param_groups, set_record
from gam.data import GLUE_TASKS, load_esnli, load_glue, synthetic_classification, synthetic_esnli, tiny_tokenizer
from gam.modeling import batches, build_model, encode, predict, save_checkpoint, task_metrics
from gam.utils import autocast, device, environment, set_seed, write_json

SUBSAMPLE_SEED = 4321  # the same subsample for every variant and seed


def load_task(a, tok):
    if a.task == "esnli":
        if a.synthetic:
            exs = synthetic_esnli(300, seed=1)
            return {"train": exs[:200], "heldout": exs[200:250], "test": exs[250:]}, 3, "acc"
        d = load_esnli(a.esnli_dir, "train")
        d["test"] = load_esnli(a.esnli_dir, "test")
        return d, 3, "acc"
    _, nl, _, metric = GLUE_TASKS[a.task]
    if a.synthetic:
        pair = a.task not in ("cola", "sst2")
        exs = synthetic_classification(300, nl, pair, seed=1)
        return {"train": exs[:200], "heldout": exs[200:250], "test": exs[250:]}, nl, metric
    return load_glue(a.task), nl, metric


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="GLUE task name or 'esnli'")
    ap.add_argument("--variant", required=True)
    ap.add_argument("--init", default=None, help="adapted MLM checkpoint dir (Phase A)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--gam_lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--train_subsample", type=int, default=0)
    ap.add_argument("--esnli_dir", default="data/esnli")
    ap.add_argument("--record_energy", action="store_true")
    ap.add_argument("--save_best", action="store_true")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()

    res_path = os.path.join(a.out, "results.json")
    if os.path.exists(res_path):
        print("exists, skipping", a.out)
        return
    set_seed(a.seed)
    dev = device()
    tok = tiny_tokenizer(os.path.join(a.out, "tok")) if a.synthetic else AutoTokenizer.from_pretrained("bert-base-uncased")
    data, num_labels, metric = load_task(a, tok)
    if a.train_subsample:
        rng = np.random.RandomState(SUBSAMPLE_SEED)
        idx = rng.choice(len(data["train"]), size=min(a.train_subsample, len(data["train"])), replace=False)
        data["train"] = [data["train"][i] for i in sorted(idx)]
    regression = num_labels == 1
    model = build_model("cls", a.variant, num_labels=num_labels, init_ckpt=a.init, tokenizer=tok,
                        synthetic=a.synthetic).to(dev)
    opt = torch.optim.AdamW(param_groups(model, a.lr, a.gam_lr))
    steps_per_epoch = (len(data["train"]) + a.batch_size - 1) // a.batch_size
    total = steps_per_epoch * a.epochs
    sched = get_linear_schedule_with_warmup(opt, int(a.warmup_ratio * total), total)
    use_scaler = a.amp and torch.cuda.is_available() and not torch.cuda.is_bf16_supported()
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    loss_fn = torch.nn.MSELoss() if regression else torch.nn.CrossEntropyLoss()
    heldout_labels = np.array([e.label for e in data["heldout"]])
    energy_batch = data["heldout"][:64]

    def evaluate_heldout():
        logits = predict(model, tok, data["heldout"], a.max_len, 128, dev, regression)
        return task_metrics(a.task, logits, heldout_labels)

    def energy_snapshot():
        set_record(model, True)
        enc, _ = encode(tok, energy_batch, a.max_len)
        with torch.no_grad():
            model.eval()
            model(**{k: v.to(dev) for k, v in enc.items()})
        snap = [m.last for m in gam_modules(model) if m.is_gam]
        set_record(model, False)
        keys = ("energy_raw", "energy_norm", "mean_mass", "max_score")
        return {k: float(np.mean([s[k] for s in snap if k in s])) for k in keys if snap and k in snap[0]}

    curve, best, best_state, best_step, step, t0 = [], -np.inf, None, 0, 0, time.time()
    for epoch in range(a.epochs):
        model.train()
        for chunk in batches(data["train"], a.batch_size, True, a.seed * 1000 + epoch):
            enc, labels = encode(tok, chunk, a.max_len)
            enc = {k: v.to(dev) for k, v in enc.items()}
            labels = labels.to(dev)
            with autocast(a.amp):
                logits = model(**enc).logits
            logits = logits.float()
            loss = loss_fn(logits.squeeze(-1), labels.float()) if regression else loss_fn(logits, labels.long())
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}")
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1
            if step % a.eval_every == 0 or step == steps_per_epoch * (epoch + 1):
                m = evaluate_heldout()
                entry = {"step": step, "epoch": epoch, "train_loss": float(loss.item()), "heldout": m,
                         "elapsed_s": time.time() - t0}
                if a.record_energy:
                    entry["energy"] = energy_snapshot()
                curve.append(entry)
                print(entry, flush=True)
                val = m[metric] if np.isfinite(m[metric]) else -1e9
                if best_state is None or val > best:
                    best = val
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    best_step = step
                model.train()
    model.load_state_dict(best_state)
    test_labels = np.array([e.label for e in data["test"]])
    test_logits = predict(model, tok, data["test"], a.max_len, 128, dev, regression)
    test = task_metrics(a.task, test_logits, test_labels)
    if a.save_best:
        save_checkpoint(model, os.path.join(a.out, "best"),
                        {"kind": "cls", "variant": a.variant, "task": a.task, "num_labels": num_labels,
                         "args": vars(a)})
    write_json(res_path, {
        "phase": "finetune", "task": a.task, "variant": a.variant, "seed": a.seed, "lr": a.lr,
        "train_size": len(data["train"]), "primary_metric": metric, "best_step": best_step,
        "best_heldout": best, "test": test, "test_primary": test[metric], "curve": curve,
        "lambda_G": lambda_table(model), "args": vars(a), "env": environment(),
        "train_time_s": time.time() - t0})
    # Per-example test predictions, so paired bootstrap tests can be run later.
    np.save(os.path.join(a.out, "test_logits.npy"), test_logits.numpy())
    np.save(os.path.join(a.out, "test_labels.npy"), test_labels)


if __name__ == "__main__":
    main()
