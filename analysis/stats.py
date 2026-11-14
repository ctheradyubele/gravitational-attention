"""Pre-specified statistical tests (see PROTOCOL: statistical analysis plan)."""
from __future__ import annotations

import math
from typing import Dict, Sequence

import numpy as np
from scipy import stats


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 10000, seed: int = 0,
                     stat=np.nanmean) -> Dict:
    """Difference stat(a) - stat(b) over paired per-example values; percentile CI; two-sided p."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    rng = np.random.RandomState(seed)
    n = len(a)
    obs = stat(a) - stat(b)
    idx = rng.randint(0, n, size=(n_boot, n))
    diffs = np.array([stat(a[i]) - stat(b[i]) for i in idx])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    # p-value: proportion of the centred bootstrap distribution at least as extreme as the observed value
    p = float(min(1.0, (np.sum(np.abs(diffs - obs) >= abs(obs)) + 1) / (n_boot + 1)))
    # one-sided p for H1: diff > 0 (proportion of bootstrap differences at or below zero)
    p_greater = float((np.sum(diffs <= 0) + 1) / (n_boot + 1))
    return {"diff": float(obs), "ci95": [float(lo), float(hi)], "p": p,
            "p_one_sided_greater": p_greater, "n": int(n)}


def steiger_z(r12: float, r13: float, r23: float, n: int) -> Dict:
    """Steiger (1980) Z for two dependent correlations sharing variable 1 (eq. 14, pooled r).

    Variable 1 = human rationale, 2 and 3 = the two importance scores being compared.
    """
    z12, z13 = np.arctanh(r12), np.arctanh(r13)
    rbar = (r12 + r13) / 2.0
    psi = r23 * (1 - 2 * rbar ** 2) - 0.5 * rbar ** 2 * (1 - 2 * rbar ** 2 - r23 ** 2)
    c = psi / (1 - rbar ** 2) ** 2
    z = (z12 - z13) * math.sqrt(n - 3) / math.sqrt(2 - 2 * c)
    return {"z": float(z), "p": float(2 * stats.norm.sf(abs(z)))}


def holm(pvals: Dict[str, float]) -> Dict[str, float]:
    """Holm-Bonferroni adjusted p-values."""
    pvals = {k: (1.0 if p is None or np.isnan(p) else p) for k, p in pvals.items()}
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, adj, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adj[k] = running
    return adj


def seed_paired_t(a: Sequence[float], b: Sequence[float], alternative: str = "two-sided") -> Dict:
    """Paired t-test across seeds (same seed = same data order and head initialisation)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    n = len(d)
    se = d.std(ddof=1) / math.sqrt(n)
    df = n - 1
    if se == 0:  # identical differences across seeds (e.g. both models predict the same)
        t = 0.0 if d.mean() == 0 else float(np.inf * np.sign(d.mean()))
    else:
        t = d.mean() / se
    if alternative == "greater":
        p = stats.t.sf(t, df)
    elif alternative == "less":
        p = stats.t.cdf(t, df)
    else:
        p = 2 * stats.t.sf(abs(t), df)
    half = stats.t.ppf(0.975, df) * se
    return {"mean_diff": float(d.mean()), "ci95": [float(d.mean() - half), float(d.mean() + half)],
            "t": float(t), "df": df, "p": float(p), "n_seeds": n}


def noninferiority(a: Sequence[float], b: Sequence[float], margin: float) -> Dict:
    """H0: mean(a - b) <= -margin. Non-inferior if the one-sided 95% lower bound exceeds -margin."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    n = len(d)
    se = d.std(ddof=1) / math.sqrt(n)
    lower = d.mean() - stats.t.ppf(0.95, n - 1) * se
    t = (d.mean() + margin) / se if se > 0 else np.inf
    p = float(stats.t.sf(t, n - 1))
    return {"mean_diff": float(d.mean()), "lower95": float(lower), "margin": margin,
            "noninferior": bool(lower > -margin), "p": p, "n_seeds": n}


def mean_sd(x: Sequence[float]) -> Dict:
    x = np.asarray(x, float)
    return {"mean": float(x.mean()), "sd": float(x.std(ddof=1)) if len(x) > 1 else 0.0, "n": int(len(x))}
