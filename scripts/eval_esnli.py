"""Phase C evaluation: plausibility, faithfulness and mass interventions on the e-SNLI test set.

Run once per fine-tuned e-SNLI checkpoint (variant x seed). Every importance source is evaluated on
the model it comes from. All per-example values are saved so the analysis can be re-run offline.
"""
import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from transformers import AutoTokenizer

from gam.attention import gam_modules
from gam.data import esnli_stats, load_esnli, synthetic_esnli, tiny_tokenizer
from gam.importance import integrated_gradients, predict_proba, to_words, token_scores
from gam.metrics import faithfulness, plausibility
from gam.modeling import build_model
from gam.utils import device, environment, set_seed, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="finetune.py output dir containing best/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--esnli_dir", default="data/esnli")
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--ig", action="store_true", help="also compute Integrated Gradients (slow)")
    ap.add_argument("--ig_steps", type=int, default=50)
    ap.add_argument("--agg", default="mean", choices=["mean", "max"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    if os.path.exists(os.path.join(a.out, "results.json")):
        print("exists, skipping", a.out)
        return
    set_seed(0)
    dev = device()
    import json
    meta = json.load(open(os.path.join(a.ckpt, "best", "meta.json")))
    variant = meta["variant"]
    if a.synthetic:
        tok = tiny_tokenizer(os.path.join(a.out, "tok"))
        test = synthetic_esnli(300, seed=1)[250:]
    else:
        tok = AutoTokenizer.from_pretrained("bert-base-uncased")
        test = load_esnli(a.esnli_dir, "test")
    if a.limit:
        test = test[: a.limit]
    model = build_model("cls", variant, num_labels=3, init_ckpt=os.path.join(a.ckpt, "best"), tokenizer=tok,
                        synthetic=a.synthetic).to(dev)
    model.eval()
    is_gam = any(m.is_gam for m in gam_modules(model))
    pairs = [(e.meta["words_a"], e.meta["words_b"]) for e in test]
    soft = [np.array(e.meta["soft_a"] + e.meta["soft_b"], float) for e in test]
    labels = np.array([e.label for e in test], int)
    nwords = [len(p[0]) + len(p[1]) for p in pairs]
    sources = (["mass", "mass_last"] if is_gam else []) + ["attention", "rollout", "random"]

    word_scores = {s: [] for s in sources}
    preds = []
    for i in range(0, len(pairs), a.bs):
        chunk = pairs[i:i + a.bs]
        tsc, maps, pr = token_scores(model, tok, chunk, a.max_len, dev,
                                     ["mass", "attention", "rollout", "random"], seed=i)
        preds.append(pr)
        for s in sources:
            for j, m in enumerate(maps):
                word_scores[s].append(to_words(tsc[s][j], m, nwords[i + j], a.agg))
    preds = np.concatenate(preds)
    if a.ig:
        word_scores["ig"] = []
        for i in range(0, len(pairs), 16):
            sc, maps = integrated_gradients(model, tok, pairs[i:i + 16], a.max_len, dev, preds[i:i + 16],
                                            n_steps=a.ig_steps)
            for j, m in enumerate(maps):
                word_scores["ig"].append(to_words(sc[j], m, nwords[i + j], a.agg))

    def pfn(ps):
        return predict_proba(model, tok, ps, a.max_len, dev, bs=a.bs)

    results = {"variant": variant, "ckpt": a.ckpt, "n": len(test), "accuracy": float((preds == labels).mean() * 100),
               "data_stats": esnli_stats(test), "plausibility": {}, "faithfulness": {}, "env": environment(),
               "args": vars(a)}
    per_example = {"labels": labels, "preds": preds, "soft": soft, "word_scores": word_scores}
    for s, ws in word_scores.items():
        pl = plausibility(ws, soft)
        per_example[f"plaus_{s}"] = {"per_example": pl.pop("per_example"), "z": pl.pop("pooled_z"),
                                     "y": pl.pop("pooled_y")}
        results["plausibility"][s] = pl
        fa = faithfulness(pfn, pairs, ws, preds)
        per_example[f"faith_{s}"] = {"comp": fa.pop("comp_per_example"), "suff": fa.pop("suff_per_example")}
        results["faithfulness"][s] = fa
        print(s, results["plausibility"][s], results["faithfulness"][s], flush=True)

    if is_gam:
        p_full = pfn(pairs)
        conf = p_full[np.arange(len(pairs)), preds]
        g = torch.Generator().manual_seed(0)

        def permute(mass, attention_mask):
            out = mass.clone()
            valid = attention_mask[:, 0, 0, :] > -1.0
            for b in range(mass.shape[0]):
                idx = valid[b].nonzero().squeeze(-1)
                out[b, idx] = mass[b, idx[torch.randperm(len(idx), generator=g).to(idx.device)]]
            return out

        def uniform(mass, attention_mask):
            valid = (attention_mask[:, 0, 0, :] > -1.0).float()
            mean = (mass * valid).sum(-1, keepdim=True) / valid.sum(-1, keepdim=True)
            return torch.where(valid > 0, mean.expand_as(mass), mass)

        results["mass_interventions"] = {}
        for name, fn in (("permute", permute), ("uniform", uniform)):
            for m in gam_modules(model):
                m.mass_override = fn
            p = pfn(pairs)
            for m in gam_modules(model):
                m.mass_override = None
            pred_i = p.argmax(-1)
            results["mass_interventions"][name] = {
                "accuracy": float((pred_i == labels).mean() * 100),
                "accuracy_drop": results["accuracy"] - float((pred_i == labels).mean() * 100),
                "mean_conf_drop": float((conf - p[np.arange(len(pairs)), preds]).mean()),
                "flip_rate": float((pred_i != preds).mean())}
            per_example[f"interv_{name}"] = conf - p[np.arange(len(pairs)), preds]
        print(results["mass_interventions"], flush=True)

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "per_example.pkl"), "wb") as f:
        pickle.dump(per_example, f)
    write_json(os.path.join(a.out, "results.json"), results)


if __name__ == "__main__":
    main()
