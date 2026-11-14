"""Phase A: equal-budget continued MLM pre-training ("adaptation") for every variant.

All variants, including the BERT control, start from bert-base-uncased and receive exactly the
same data, steps, batch size and schedule, so that differences cannot come from extra training.
"""
import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, get_linear_schedule_with_warmup

from gam.attention import calibrate_g, lambda_table, param_groups
from gam.data import SYN_VOCAB, load_wikitext, tiny_tokenizer
from gam.modeling import build_model, save_checkpoint
from gam.utils import autocast, device, environment, set_seed, write_json


def chunk_corpus(tokenizer, texts, seq_len):
    parts = []
    for i in range(0, len(texts), 10000):
        enc = tokenizer(texts[i:i + 10000], add_special_tokens=False)["input_ids"]
        parts.append(np.fromiter((t for e in enc for t in e), dtype=np.int32))
    ids = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int32)
    body = seq_len - 2
    n = len(ids) // body
    arr = ids[: n * body].astype(np.int64).reshape(n, body)
    cls, sep = tokenizer.cls_token_id, tokenizer.sep_token_id
    return np.concatenate([np.full((n, 1), cls), arr, np.full((n, 1), sep)], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--batch_size", type=int, default=256, help="effective sequences per step")
    ap.add_argument("--micro_batch", type=int, default=64)
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--gam_lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()

    if os.path.exists(os.path.join(a.out, "results.json")):
        print("exists, skipping", a.out)
        return
    set_seed(a.seed)
    dev = device()
    if a.synthetic:
        tok = tiny_tokenizer(os.path.join(a.out, "tok"))
        rng = np.random.RandomState(0)
        texts = [" ".join(rng.choice(SYN_VOCAB, 30)) for _ in range(400)]
        corpus = {"train": texts, "validation": texts[:50]}
    else:
        tok = AutoTokenizer.from_pretrained("bert-base-uncased")
        corpus = load_wikitext()
    train = chunk_corpus(tok, corpus["train"], a.seq_len)
    val = chunk_corpus(tok, corpus["validation"], a.seq_len)[:2000]
    model = build_model("mlm", a.variant, tokenizer=tok, synthetic=a.synthetic).to(dev)
    calibrated_g = None
    if a.variant != "standard":  # pre-specified data-dependent initialisation of G (same batch for all)
        ids = torch.tensor(val[:32]).to(dev)
        calibrated_g = calibrate_g(model, {"input_ids": ids, "attention_mask": torch.ones_like(ids)})
        print("calibrated G per layer (head mean):", [round(float(np.mean(g)), 3) for g in calibrated_g])
    collator = DataCollatorForLanguageModeling(tok, mlm_probability=0.15)
    opt = torch.optim.AdamW(param_groups(model, a.lr, a.gam_lr))
    sched = get_linear_schedule_with_warmup(opt, a.warmup, a.steps)
    use_scaler = a.amp and torch.cuda.is_available() and not torch.cuda.is_bf16_supported()
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    accum = max(1, a.batch_size // a.micro_batch)
    rng = np.random.RandomState(a.seed)
    order, pos = rng.permutation(len(train)), 0

    def val_loss():
        model.eval()
        torch.manual_seed(1234)  # identical masking for every variant and every evaluation
        losses = []
        with torch.no_grad():
            for i in range(0, len(val), a.micro_batch):
                b = collator([{"input_ids": torch.tensor(r)} for r in val[i:i + a.micro_batch]])
                with autocast(a.amp):
                    losses.append(model(**{k: v.to(dev) for k, v in b.items()}).loss.float().item())
        model.train()
        return float(np.mean(losses))

    log, t0 = [{"step": 0, "val_loss": val_loss()}], time.time()
    model.train()
    for step in range(1, a.steps + 1):
        for _ in range(accum):
            if pos + a.micro_batch > len(order):
                order, pos = rng.permutation(len(train)), 0
            rows = train[order[pos:pos + a.micro_batch]]
            pos += a.micro_batch
            b = collator([{"input_ids": torch.tensor(r)} for r in rows])
            with autocast(a.amp):
                loss = model(**{k: v.to(dev) for k, v in b.items()}).loss / accum
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}")
            scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        sched.step()
        if step % a.eval_every == 0 or step == a.steps:
            log.append({"step": step, "val_loss": val_loss(), "elapsed_s": time.time() - t0})
            print(log[-1], flush=True)
    save_checkpoint(model, a.out, {"kind": "mlm", "variant": a.variant, "args": vars(a)})
    write_json(os.path.join(a.out, "results.json"), {
        "phase": "adapt", "variant": a.variant, "args": vars(a), "env": environment(), "log": log,
        "final_val_loss": log[-1]["val_loss"], "final_val_ppl": math.exp(log[-1]["val_loss"]),
        "lambda_G": lambda_table(model), "calibrated_G_init": calibrated_g,
        "train_sequences": int(len(train))})


if __name__ == "__main__":
    main()
