"""Draw the paper's figures from the analysis outputs in results/ (no number is typed in by hand)."""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = "results"
OUT = "figures"
os.makedirs(OUT, exist_ok=True)

# Fixed categorical order (reference palette, light mode): colour follows the variant.
COL = {"standard": "#2a78d6", "hybrid": "#eb6834", "pure": "#1baf7a",
       "simplified": "#eda100", "l2": "#e87ba4", "standard_raw": "#6250d6"}
NAME = {"standard": "BERT (standard)", "hybrid": "Hybrid GAM", "pure": "Pure GAM",
        "simplified": "Simplified GAM", "l2": "L2 attention", "standard_raw": "BERT, no adaptation"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def rows(name):
    with open(os.path.join(R, name)) as f:
        return list(csv.DictReader(f))


H = json.load(open(os.path.join(R, "hypothesis_tests.json")))
TASKS = ["cola", "sst2", "mrpc", "stsb", "qqp", "mnli", "qnli", "rte"]
TLAB = {"cola": "CoLA", "sst2": "SST-2", "mrpc": "MRPC", "stsb": "STS-B", "qqp": "QQP",
        "mnli": "MNLI-m", "qnli": "QNLI", "rte": "RTE"}

# ---------------------------------------------------------------- Figure: GLUE differences
fig, ax = plt.subplots(figsize=(6.6, 2.9))
y = np.arange(len(TASKS))[::-1]
for off, v in ((0.17, "hybrid"), (-0.17, "pure")):
    res = H[f"secondary_{v}_vs_standard_per_task"]
    d = np.array([res[t]["mean_diff"] for t in TASKS])
    lo = np.array([res[t]["ci95"][0] for t in TASKS])
    hi = np.array([res[t]["ci95"][1] for t in TASKS])
    ax.hlines(y + off, lo, hi, color=COL[v], lw=2)
    ax.plot(d, y + off, "o", ms=6, color=COL[v], mec="white", mew=1.2, label=NAME[v])
ax.axvline(0, color=INK2, lw=1)
ax.axvspan(-1.0, 1.0, color="#f1f0ed", zorder=0, lw=0)
ax.set_yticks(y)
ax.set_yticklabels([TLAB[t] for t in TASKS])
ax.set_xlabel("Difference from BERT (GLUE points; mean over 5 paired seeds, 95% CI)")
ax.grid(axis="y", visible=False)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
ax.text(0.15, y[0] + 0.55, "±1-point band", color=INK2, fontsize=7.5)
fig.savefig(f"{OUT}/fig_glue_delta.pdf")
fig.savefig(f"{OUT}/fig_glue_delta.png")
plt.close(fig)

# ---------------------------------------------------------------- Figure: e-SNLI plausibility vs faithfulness
es = rows("esnli.csv")
val = {(r["variant"], r["source"], r["metric"]): float(r["mean"]) for r in es}
sd = {(r["variant"], r["source"], r["metric"]): float(r["sd"]) for r in es}
points = [  # (variant whose model, source, label, marker)
    ("standard", "attention", "BERT attention", "s"),
    ("standard", "ig", "BERT Integrated Gradients", "D"),
    ("standard", "rollout", "BERT attention rollout", "^"),
    ("hybrid", "mass", "Hybrid GAM mass", "o"),
    ("pure", "mass", "Pure GAM mass", "o"),
    ("standard", "random", "Random", "x"),
]
fig, ax = plt.subplots(figsize=(4.6, 3.3))
for v, s, lab, mk in points:
    x, yv = val[(v, s, "auprc")], val[(v, s, "comprehensiveness")]
    xe, ye = sd[(v, s, "auprc")], sd[(v, s, "comprehensiveness")]
    c = COL[v] if s != "random" else INK2
    ax.errorbar(x, yv, xerr=xe, yerr=ye, fmt="none", ecolor=c, elinewidth=1, alpha=0.6)
    ax.plot(x, yv, mk, ms=8, color=c, mec="white" if mk != "x" else c, mew=1.2 if mk != "x" else 2)
    dx, ha = (0.012, "left")
    dy = 0.0
    if s == "ig":
        dy = 0.022
    if s == "attention":
        dx, ha, dy = -0.012, "right", -0.022
    ax.annotate(lab, (x + dx, yv + dy), fontsize=7.5, color=INK, ha=ha, va="center")
ax.set_xlabel("Plausibility: AUPRC vs human rationales")
ax.set_ylabel("Faithfulness: comprehensiveness (AOPC)")
ax.set_xlim(0.25, 0.85)
ax.set_ylim(0.08, 0.45)
fig.savefig(f"{OUT}/fig_esnli.pdf")
fig.savefig(f"{OUT}/fig_esnli.png")
plt.close(fig)

# ---------------------------------------------------------------- Figure: lambda by layer
lam = rows("lambda_by_layer.csv")
fig, axs = plt.subplots(2, 4, figsize=(6.8, 3.0), sharex=True, sharey=True)
for ax, t in zip(axs.ravel(), TASKS):
    L = [int(r["layer"]) for r in lam if r["task"] == t]
    m = np.array([float(r["lambda_mean"]) for r in lam if r["task"] == t])
    s = np.array([float(r["lambda_sd_over_seeds"]) for r in lam if r["task"] == t])
    ax.fill_between(L, m - s, m + s, color=COL["hybrid"], alpha=0.18, lw=0)
    ax.plot(L, m, "-o", color=COL["hybrid"], lw=2, ms=3.5)
    ax.axhline(0.1, color=INK2, lw=0.8, ls="--")
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_title(TLAB[t], fontsize=8.5, color=INK)
for ax in axs[1]:
    ax.set_xlabel("Layer")
for ax in axs[:, 0]:
    ax.set_ylabel("Mean λ")
axs[0, 0].text(11, 0.104, "init 0.1", ha="right", va="bottom", fontsize=6.5, color=INK2)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_lambda.pdf")
fig.savefig(f"{OUT}/fig_lambda.png")
plt.close(fig)

# ---------------------------------------------------------------- Figure: cost
bm = rows("benchmark_amp.csv")
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.6))
for v in ("standard", "hybrid", "pure"):
    rr = [r for r in bm if r["variant"] == v and r["direction"] == "forward_backward"]
    n = [int(r["n"]) for r in rr]
    axs[0].plot(n, [float(r["median_ms"]) for r in rr], "-o", color=COL[v], lw=2, ms=5, label=NAME[v])
    axs[1].plot(n, [float(r["peak_mem_mb"]) / 1024 for r in rr], "-o", color=COL[v], lw=2, ms=5, label=NAME[v])
for ax, lab in zip(axs, ("Forward + backward time (ms)", "Peak memory (GB)")):
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([128, 512, 1024, 2048])
    ax.set_xticklabels(["128", "512", "1024", "2048"])
    ax.set_xlabel("Sequence length")
    ax.set_ylabel(lab)
axs[0].legend(loc="upper left")
fig.tight_layout()
fig.savefig(f"{OUT}/fig_cost.pdf")
fig.savefig(f"{OUT}/fig_cost.png")
plt.close(fig)
print("figures written to", OUT)
