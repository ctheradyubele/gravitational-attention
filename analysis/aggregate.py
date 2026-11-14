"""Turn run outputs into every table, figure input and hypothesis test reported in the paper.

    python analysis/aggregate.py --runs runs --out results

Every number in the revised manuscript must come from a file this script writes.
"""
import argparse
import glob
import json
import os
import pickle
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import yaml

from analysis.stats import holm, mean_sd, noninferiority, paired_bootstrap, seed_paired_t, steiger_z

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = yaml.safe_load(open(os.environ.get("GAM_PROTOCOL", os.path.join(HERE, "configs", "protocol.yaml"))))
TASK_LABEL = {"cola": "CoLA", "sst2": "SST-2", "mrpc": "MRPC", "stsb": "STS-B", "qqp": "QQP",
              "mnli": "MNLI-m", "qnli": "QNLI", "rte": "RTE"}
NONINF_MARGIN = 1.0  # GLUE points, pre-specified


def load(pattern):
    out = []
    for p in sorted(glob.glob(pattern, recursive=True)):
        r = json.load(open(p))
        r["_dir"] = os.path.dirname(p)
        out.append(r)
    return out


def fmt(ms, d=1):
    return f"{ms['mean']:.{d}f} $\\pm$ {ms['sd']:.{d}f}"


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def glue_section(runs_dir, out, report):
    rs = load(f"{runs_dir}/B2/*/*/*/results.json")
    by = defaultdict(dict)  # (task, variant) -> seed -> primary
    for r in rs:
        v = os.path.basename(os.path.dirname(r["_dir"]))
        by[(r["task"], v)][r["seed"]] = r["test_primary"]
    tasks = [t for t in P["glue"]["tasks"]]
    variants = [v for v in P["glue"]["main_variants"] + ["standard_raw"] if any((t, v) in by for t in tasks)]
    seeds = P["glue"]["final_seeds"]
    rows, csv = [], ["task,variant,mean,sd,n"]
    avg = {}
    for v in variants:
        per_seed = []
        for s in seeds:
            vals = [by[(t, v)].get(s) for t in tasks]
            per_seed.append(np.mean(vals) if all(x is not None for x in vals) else np.nan)
        avg[v] = np.array(per_seed)
    for t in tasks:
        cells = []
        for v in variants:
            vals = [by[(t, v)][s] for s in seeds if s in by[(t, v)]]
            ms = mean_sd(vals) if vals else {"mean": np.nan, "sd": np.nan, "n": 0}
            csv.append(f"{t},{v},{ms['mean']:.3f},{ms['sd']:.3f},{ms['n']}")
            cells.append(fmt(ms) if vals else "--")
        rows.append(f"{TASK_LABEL[t]} & " + " & ".join(cells) + r" \\")
    avg_cells = []
    for v in variants:
        x = avg[v][~np.isnan(avg[v])]
        ms = mean_sd(x) if len(x) else {"mean": np.nan, "sd": np.nan, "n": 0}
        csv.append(f"average,{v},{ms['mean']:.3f},{ms['sd']:.3f},{ms['n']}")
        avg_cells.append(fmt(ms, 2) if len(x) else "--")
    rows.append(r"\midrule" + "\nAverage & " + " & ".join(avg_cells) + r" \\")
    write(f"{out}/glue.csv", "\n".join(csv) + "\n")
    write(f"{out}/glue_rows.tex", "\n".join(rows) + "\n")

    # Figure 3: mean per-task difference vs the BERT control (seed-paired).
    fig = ["task,variant,mean_delta,sd_delta"]
    for v in ("pure", "hybrid"):
        for t in tasks:
            common = sorted(set(by[(t, v)]) & set(by[(t, "standard")]))
            if common:
                d = np.array([by[(t, v)][s] - by[(t, "standard")][s] for s in common])
                fig.append(f"{t},{v},{d.mean():.3f},{d.std(ddof=1) if len(d) > 1 else 0:.3f}")
    write(f"{out}/figure3_deltas.csv", "\n".join(fig) + "\n")

    # H3 (primary): Hybrid non-inferior to the BERT control on the GLUE average.
    if "hybrid" in avg and "standard" in avg:
        ok = ~(np.isnan(avg["hybrid"]) | np.isnan(avg["standard"]))
        if ok.sum() >= 2:
            report["H3_hybrid_noninferior_glue_avg"] = noninferiority(avg["hybrid"][ok], avg["standard"][ok], NONINF_MARGIN)
    # Secondary: per-task differences vs control, Holm-corrected within each comparison family.
    for v in ("pure", "hybrid"):
        tests = {}
        for t in tasks:
            common = sorted(set(by[(t, v)]) & set(by[(t, "standard")]))
            if len(common) >= 2:
                tests[t] = seed_paired_t([by[(t, v)][s] for s in common], [by[(t, "standard")][s] for s in common])
        adj = holm({t: r["p"] for t, r in tests.items()}) if tests else {}
        for t in tests:
            tests[t]["p_holm"] = adj[t]
        report[f"secondary_{v}_vs_standard_per_task"] = tests

    # Learned lambda per task (mean over seeds, heads), per layer.
    lam = defaultdict(list)
    for r in rs:
        v = os.path.basename(os.path.dirname(r["_dir"]))
        L = r.get("lambda_G", {}).get("lambda")
        if v == "hybrid" and L and L[0] is not None:
            lam[r["task"]].append(np.array(L))  # [layers, heads]
    lines = ["task,layer,lambda_mean,lambda_sd_over_seeds"]
    for t, arrs in lam.items():
        a = np.stack(arrs)  # [seeds, layers, heads]
        per_layer = a.mean(-1)
        for l in range(a.shape[1]):
            lines.append(f"{t},{l},{per_layer[:, l].mean():.4f},{per_layer[:, l].std(ddof=1) if len(arrs) > 1 else 0:.4f}")
    write(f"{out}/lambda_by_layer.csv", "\n".join(lines) + "\n")

    # Ablation (SST-2): every variant's SST-2 score.
    ab = ["variant,mean,sd,n"]
    for v in ("hybrid", "simplified", "hybrid_nodist", "pure", "hybrid_fixedlam", "l2", "standard"):
        vals = list(by[("sst2", v)].values())
        if vals:
            ms = mean_sd(vals)
            ab.append(f"{v},{ms['mean']:.3f},{ms['sd']:.3f},{ms['n']}")
    write(f"{out}/ablation_sst2.csv", "\n".join(ab) + "\n")

    # Energy and convergence on SST-2.
    en = ["variant,seed,step,energy_raw,energy_norm,mean_mass,max_score"]
    conv = defaultdict(list)
    for r in rs:
        if r["task"] != "sst2":
            continue
        v = os.path.basename(os.path.dirname(r["_dir"]))
        curve = r["curve"]
        best = max(c["heldout"][r["primary_metric"]] for c in curve)
        first = next(c["step"] for c in curve if c["heldout"][r["primary_metric"]] >= 0.99 * best)
        conv[v].append(first)
        for c in curve:
            e = c.get("energy")
            if e:
                en.append(f"{v},{r['seed']},{c['step']},{e.get('energy_raw', '')},{e.get('energy_norm', '')},"
                          f"{e.get('mean_mass', '')},{e.get('max_score', '')}")
    write(f"{out}/energy_sst2.csv", "\n".join(en) + "\n")
    report["steps_to_99pct_best_heldout_sst2"] = {v: mean_sd(x) for v, x in conv.items()}


def esnli_section(runs_dir, out, report):
    evs = load(f"{runs_dir}/C/esnli/*/*/eval/results.json")
    if not evs:
        return
    per = defaultdict(dict)  # variant -> seed -> per_example dict
    res = defaultdict(dict)
    for r in evs:
        v = os.path.basename(os.path.dirname(os.path.dirname(r["_dir"])))
        seed = int(re.search(r"_s(\d+)", r["_dir"]).group(1))
        res[v][seed] = r
        per[v][seed] = pickle.load(open(os.path.join(r["_dir"], "per_example.pkl"), "rb"))
    # Table: plausibility and faithfulness, mean +- SD over seeds.
    lines = ["variant,source,metric,mean,sd,n_seeds"]
    for v, seeds in res.items():
        acc = mean_sd([s["accuracy"] for s in seeds.values()])
        lines.append(f"{v},-,accuracy,{acc['mean']:.3f},{acc['sd']:.3f},{acc['n']}")
        sources = set.intersection(*[set(s["plausibility"]) for s in seeds.values()])
        for src in sorted(sources):
            for m in ("auprc", "token_f1", "pearson_r", "spearman_rho"):
                ms = mean_sd([s["plausibility"][src][m] for s in seeds.values()])
                lines.append(f"{v},{src},{m},{ms['mean']:.4f},{ms['sd']:.4f},{ms['n']}")
            for m in ("comprehensiveness", "sufficiency"):
                ms = mean_sd([s["faithfulness"][src][m] for s in seeds.values()])
                lines.append(f"{v},{src},{m},{ms['mean']:.4f},{ms['sd']:.4f},{ms['n']}")
        if all("mass_interventions" in s for s in seeds.values()):
            for name in ("permute", "uniform"):
                for m in ("accuracy_drop", "mean_conf_drop", "flip_rate"):
                    ms = mean_sd([s["mass_interventions"][name][m] for s in seeds.values()])
                    lines.append(f"{v},mass_{name},{m},{ms['mean']:.4f},{ms['sd']:.4f},{ms['n']}")
    write(f"{out}/esnli.csv", "\n".join(lines) + "\n")

    def seed_avg(v, key, field):
        """Per-example value averaged over the seeds available for (variant, source)."""
        arrs = []
        for s, pe in sorted(per[v].items()):
            if key in pe:
                if field == "auprc":
                    arrs.append(np.array([d["auprc"] for d in pe[key]["per_example"]], float))
                else:
                    arrs.append(np.asarray(pe[key][field], float))
        return np.nanmean(np.stack(arrs), 0) if arrs else None

    # H1 (primary): Hybrid mass vs BERT-control attention, per-example AUPRC.
    h1 = {}
    for gam_v in ("hybrid", "pure"):
        a, b = seed_avg(gam_v, "plaus_mass", "auprc"), seed_avg("standard", "plaus_attention", "auprc")
        if a is not None and b is not None:
            h1[f"{gam_v}_mass_vs_standard_attention"] = paired_bootstrap(a, b)
        c = seed_avg("standard", "plaus_ig", "auprc")
        if a is not None and c is not None:
            h1[f"{gam_v}_mass_vs_standard_ig"] = paired_bootstrap(a, c)
    # Steiger test on pooled correlations, per common seed (secondary).
    st = {}
    for gam_v in ("hybrid", "pure"):
        for s in sorted(set(per.get(gam_v, {})) & set(per.get("standard", {}))):
            pa, pb = per[gam_v][s].get("plaus_mass"), per["standard"][s].get("plaus_attention")
            if pa is None or pb is None or len(pa["z"]) != len(pb["z"]):
                continue  # word sets differ when some examples were skipped; then rely on the bootstrap
            y = pa["y"]
            r12, r13 = np.corrcoef(y, pa["z"])[0, 1], np.corrcoef(y, pb["z"])[0, 1]
            r23 = np.corrcoef(pa["z"], pb["z"])[0, 1]
            st[f"{gam_v}_s{s}"] = {"r_mass": r12, "r_attention": r13, **steiger_z(r12, r13, r23, len(y))}
    h1["steiger_pooled_r"] = st
    report["H1_plausibility"] = h1

    # H2 (primary): Hybrid mass is faithful: comprehensiveness above chance on the same model.
    h2 = {}
    for gam_v in ("hybrid", "pure"):
        a, r = seed_avg(gam_v, "faith_mass", "comp"), seed_avg(gam_v, "faith_random", "comp")
        if a is not None and r is not None:
            h2[f"{gam_v}_mass_vs_random_comprehensiveness"] = paired_bootstrap(a, r)
        b = seed_avg("standard", "faith_attention", "comp")
        if a is not None and b is not None:
            h2[f"{gam_v}_mass_vs_standard_attention_comprehensiveness"] = paired_bootstrap(a, b)
    report["H2_faithfulness"] = h2


def data_size_section(runs_dir, out, report):
    rs = load(f"{runs_dir}/D/*/*/*/results.json")
    if not rs:
        return
    by = defaultdict(dict)
    for r in rs:
        v, n = re.match(r"(.+)_n(\d+)$", os.path.basename(os.path.dirname(r["_dir"]))).groups()
        by[(v, int(n))][r["seed"]] = r["test_primary"]
    lines = ["variant,train_size,mean,sd,n"]
    for (v, n), d in sorted(by.items()):
        ms = mean_sd(list(d.values()))
        lines.append(f"{v},{n},{ms['mean']:.3f},{ms['sd']:.3f},{ms['n']}")
    write(f"{out}/data_size.csv", "\n".join(lines) + "\n")
    h4 = {}
    for n in P["data_size"]["sizes"]:
        a, b = by.get(("pure", n), {}), by.get(("standard", n), {})
        common = sorted(set(a) & set(b))
        if len(common) >= 2:
            h4[f"pure_minus_standard_n{n}"] = seed_paired_t([a[s] for s in common], [b[s] for s in common])
    report["H4_data_size"] = h4


def adapt_section(runs_dir, out, report):
    rs = load(f"{runs_dir}/A/*/results.json")
    lines = ["variant,final_val_loss,final_val_ppl"]
    for r in rs:
        lines.append(f"{r['variant']},{r['final_val_loss']:.4f},{r['final_val_ppl']:.2f}")
    if rs:
        write(f"{out}/adaptation.csv", "\n".join(lines) + "\n")


def lra_section(runs_dir, out, report):
    rs = load(f"{runs_dir}/G/*/*/*/results.json")
    if not rs:
        return
    by = defaultdict(dict)
    for r in rs:
        by[(r["task"], r["variant"])][r["seed"]] = r
    lines = ["task,variant,test_acc_mean,test_acc_sd,n,train_time_s_mean,peak_mem_gb_mean"]
    for (t, v), d in sorted(by.items()):
        acc = mean_sd([x["test_acc"] for x in d.values()])
        lines.append(f"{t},{v},{acc['mean']:.3f},{acc['sd']:.3f},{acc['n']},"
                     f"{np.mean([x['train_time_s'] for x in d.values()]):.1f},"
                     f"{np.mean([x['peak_mem_gb'] for x in d.values()]):.2f}")
    write(f"{out}/lra.csv", "\n".join(lines) + "\n")
    tests = {}
    for v in ("hybrid", "pure"):
        for t in sorted({t for t, _ in by}):
            a, b = by.get((t, v), {}), by.get((t, "standard"), {})
            common = sorted(set(a) & set(b))
            if len(common) >= 2:
                tests[f"{v}_{t}"] = seed_paired_t([a[s]["test_acc"] for s in common], [b[s]["test_acc"] for s in common])
    report["secondary_lra"] = tests


def benchmark_section(runs_dir, out):
    for r in load(f"{runs_dir}/F/*/results.json"):
        lines = ["n,variant,direction,median_ms,peak_mem_mb"]
        for row in r["rows"]:
            lines.append(f"{row['n']},{row['variant']},{row['direction']},{row['median_ms']:.3f},{row['peak_mem_mb']}")
        write(f"{out}/{os.path.basename(r['_dir'])}.csv", "\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=P["runs_dir"])
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    report = {}
    adapt_section(a.runs, a.out, report)
    glue_section(a.runs, a.out, report)
    esnli_section(a.runs, a.out, report)
    data_size_section(a.runs, a.out, report)
    lra_section(a.runs, a.out, report)
    benchmark_section(a.runs, a.out)
    # Holm correction across the three pre-registered primary hypotheses (Hybrid GAM).
    prim = {}
    if "hybrid_mass_vs_standard_attention" in report.get("H1_plausibility", {}):
        prim["H1"] = report["H1_plausibility"]["hybrid_mass_vs_standard_attention"]["p_one_sided_greater"]
    if "hybrid_mass_vs_random_comprehensiveness" in report.get("H2_faithfulness", {}):
        prim["H2"] = report["H2_faithfulness"]["hybrid_mass_vs_random_comprehensiveness"]["p_one_sided_greater"]
    if "H3_hybrid_noninferior_glue_avg" in report:
        prim["H3"] = report["H3_hybrid_noninferior_glue_avg"]["p"]
    if prim:
        report["primary_holm"] = {"raw_p": prim, "holm_p": holm(prim), "alpha": 0.05,
                                  "note": "H1, H2 one-sided (GAM mass > comparator); H3 one-sided non-inferiority (margin 1.0)"}

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.bool_):
            return bool(o)
        return o
    with open(os.path.join(a.out, "hypothesis_tests.json"), "w") as f:
        json.dump(clean(report), f, indent=2)
    print(json.dumps(clean(report), indent=2)[:4000])


if __name__ == "__main__":
    main()
