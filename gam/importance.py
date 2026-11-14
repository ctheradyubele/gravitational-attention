"""Word-level importance scores from GAM masses, attention, attention rollout, Integrated Gradients and chance."""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from .attention import gam_modules, set_record

Pair = Tuple[List[str], List[str]]


def encode_words(tokenizer, pairs: Sequence[Pair], max_len: int):
    enc = tokenizer([p[0] for p in pairs], [p[1] for p in pairs], is_split_into_words=True,
                    truncation=True, max_length=max_len, padding=True, return_tensors="pt")
    maps = []
    for i, (wa, _) in enumerate(pairs):
        wid, sid = enc.word_ids(i), enc.sequence_ids(i)
        maps.append([None if w is None else (w if s == 0 else len(wa) + w) for w, s in zip(wid, sid)])
    return enc, maps


def to_words(token_scores: np.ndarray, tok_map: List[Optional[int]], n_words: int, how: str = "mean"):
    """Aggregate sub-word scores to words. Words lost to truncation get NaN."""
    out = np.full(n_words, np.nan)
    buckets = {}
    for t, w in enumerate(tok_map):
        if w is not None and t < len(token_scores):
            buckets.setdefault(w, []).append(float(token_scores[t]))
    for w, v in buckets.items():
        out[w] = np.mean(v) if how == "mean" else np.max(v)
    return out


@torch.no_grad()
def predict_proba(model, tokenizer, pairs: Sequence[Pair], max_len: int, device, bs: int = 128):
    model.eval()
    out = []
    for i in range(0, len(pairs), bs):
        enc, _ = encode_words(tokenizer, pairs[i:i + bs], max_len)
        logits = model(**{k: v.to(device) for k, v in enc.items()}).logits.float()
        out.append(torch.softmax(logits, -1).cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def token_scores(model, tokenizer, pairs, max_len, device, sources: Sequence[str], seed: int = 0):
    """Returns {source: [np.ndarray per example (token level)]} plus the token->word maps."""
    model.eval()
    enc, maps = encode_words(tokenizer, pairs, max_len)
    enc = {k: v.to(device) for k, v in enc.items()}
    want_mass = "mass" in sources and len([m for m in gam_modules(model) if m.is_gam]) > 0
    if want_mass:
        set_record(model, True)
    out = model(**enc, output_attentions=True)
    lengths = enc["attention_mask"].sum(-1).tolist()
    res = {}
    if want_mass:
        masses = torch.stack([m.last["mass"] for m in gam_modules(model) if m.is_gam])  # [L,B,T]
        set_record(model, False)
        res["mass"] = masses.mean(0).cpu().numpy()
        res["mass_last"] = masses[-1].cpu().numpy()
    att = [a.float() for a in out.attentions]                                         # L x [B,H,T,T]
    if "attention" in sources:
        res["attention"] = att[-1][:, :, 0, :].mean(1).cpu().numpy()
    if "rollout" in sources:
        roll = None
        for a in att:
            a = a.mean(1)
            a = 0.5 * a + 0.5 * torch.eye(a.shape[-1], device=a.device)
            a = a / a.sum(-1, keepdim=True)
            roll = a if roll is None else a @ roll
        res["rollout"] = roll[:, 0, :].cpu().numpy()
    if "random" in sources:
        rng = np.random.RandomState(seed)
        res["random"] = rng.rand(*enc["input_ids"].shape)
    split = {k: [v[i][: lengths[i]] for i in range(len(pairs))] for k, v in res.items()}
    return split, maps, out.logits.float().argmax(-1).cpu().numpy()


def integrated_gradients(model, tokenizer, pairs, max_len, device, targets, n_steps: int = 50,
                         internal_batch_size: int = 64):
    """|sum over embedding dim| of IG attributions w.r.t. the embedding layer output, baseline = [PAD] tokens."""
    from captum.attr import LayerIntegratedGradients
    model.eval()
    enc, maps = encode_words(tokenizer, pairs, max_len)
    ids = enc["input_ids"].to(device)
    am = enc["attention_mask"].to(device)
    tt = enc["token_type_ids"].to(device)
    special = (ids == tokenizer.cls_token_id) | (ids == tokenizer.sep_token_id)
    base = torch.where(special, ids, torch.full_like(ids, tokenizer.pad_token_id))

    def fwd(input_ids, attention_mask, token_type_ids):
        return model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).logits

    lig = LayerIntegratedGradients(fwd, model.base_model.embeddings)
    attr = lig.attribute(ids, baselines=base, additional_forward_args=(am, tt),
                         target=torch.as_tensor(targets, device=device), n_steps=n_steps,
                         internal_batch_size=internal_batch_size)
    scores = attr.sum(-1).abs().detach().cpu().numpy()
    lengths = am.sum(-1).tolist()
    return [scores[i][: lengths[i]] for i in range(len(pairs))], maps
