"""Plausibility (agreement with human rationales) and faithfulness (ERASER-style) metrics."""
from __future__ import annotations

import math
from typing import Dict, List, Sequence

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score

ERASER_BINS = (0.01, 0.05, 0.10, 0.20, 0.50)


def plausibility_per_example(scores: np.ndarray, soft: np.ndarray) -> Dict[str, float]:
    """scores, soft: word-level arrays for one example. Binary label = majority of annotators (>= 0.5)."""
    ok = ~np.isnan(scores)
    s, y = scores[ok], soft[ok]
    yb = (y >= 0.5).astype(int)
    out = {"auprc": np.nan, "token_f1": np.nan}
    if 0 < yb.sum() < len(yb):
        out["auprc"] = float(average_precision_score(yb, s))
    if yb.sum() > 0:
        k = int(yb.sum())
        top = np.zeros_like(yb)
        top[np.argsort(-s, kind="stable")[:k]] = 1
        tp = int((top & yb).sum())
        out["token_f1"] = 2 * tp / (top.sum() + yb.sum())
    return out


def plausibility(all_scores: Sequence[np.ndarray], all_soft: Sequence[np.ndarray]) -> Dict:
    per = [plausibility_per_example(np.asarray(s), np.asarray(y)) for s, y in zip(all_scores, all_soft)]
    # Pooled correlations use within-example z-scored importance, so scale differences between
    # examples do not drive the correlation.
    zs, ys = [], []
    for s, y in zip(all_scores, all_soft):
        s, y = np.asarray(s, float), np.asarray(y, float)
        ok = ~np.isnan(s)
        s, y = s[ok], y[ok]
        if len(s) > 1 and s.std() > 0:
            zs.append((s - s.mean()) / s.std())
            ys.append(y)
    z, y = np.concatenate(zs), np.concatenate(ys)
    return {
        "auprc": float(np.nanmean([p["auprc"] for p in per])),
        "token_f1": float(np.nanmean([p["token_f1"] for p in per])),
        "pearson_r": float(pearsonr(z, y)[0]),
        "spearman_rho": float(spearmanr(z, y)[0]),
        "n_examples_auprc": int(np.sum(~np.isnan([p["auprc"] for p in per]))),
        "per_example": per,
        "pooled_z": z, "pooled_y": y,
    }


def topk_indices(scores: np.ndarray, frac: float) -> np.ndarray:
    s = np.where(np.isnan(scores), -np.inf, scores)
    k = max(1, int(math.ceil(frac * len(s))))
    return np.argsort(-s, kind="stable")[:k]


def remove_words(words_a: List[str], words_b: List[str], idx, keep: bool):
    """keep=False: delete words at global indices idx (comprehensiveness); keep=True: keep only them."""
    idx = set(int(i) for i in idx)
    na = len(words_a)
    sel = (lambda g: g in idx) if keep else (lambda g: g not in idx)
    a = [w for i, w in enumerate(words_a) if sel(i)]
    b = [w for i, w in enumerate(words_b) if sel(na + i)]
    return a, b


def faithfulness(predict_fn, pairs, word_scores, pred_labels, bins=ERASER_BINS) -> Dict:
    """predict_fn(list of (words_a, words_b)) -> probs [N, C]. Returns AOPC comprehensiveness/sufficiency."""
    base = predict_fn(pairs)
    p0 = base[np.arange(len(pairs)), pred_labels]
    comp, suff = [], []
    for b in bins:
        comp_pairs, suff_pairs = [], []
        for (wa, wb), s in zip(pairs, word_scores):
            idx = topk_indices(np.asarray(s, float), b)
            comp_pairs.append(remove_words(wa, wb, idx, keep=False))
            suff_pairs.append(remove_words(wa, wb, idx, keep=True))
        pc = predict_fn(comp_pairs)[np.arange(len(pairs)), pred_labels]
        ps = predict_fn(suff_pairs)[np.arange(len(pairs)), pred_labels]
        comp.append(p0 - pc)
        suff.append(p0 - ps)
    comp, suff = np.stack(comp, 1), np.stack(suff, 1)            # [N, bins]
    return {"comprehensiveness": float(comp.mean()), "sufficiency": float(suff.mean()),
            "comp_by_bin": comp.mean(0).tolist(), "suff_by_bin": suff.mean(0).tolist(),
            "comp_per_example": comp.mean(1), "suff_per_example": suff.mean(1)}
