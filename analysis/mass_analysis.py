"""Section 6.1: distribution of learned masses and their relation to linguistic features (e-SNLI test).

    python analysis/mass_analysis.py --runs runs --esnli_dir data/esnli --out results

Pre-specified definitions:
  * word mass = mean over layers and sub-words (as in eval_esnli.py, source "mass")
  * function words = scikit-learn's ENGLISH_STOP_WORDS; content words = all other alphabetic words
  * IDF from e-SNLI training sentences (premises + hypotheses), lower-cased whitespace words
  * sentiment words = words with any non-zero SentiWordNet score (optional; needs nltk + corpus)
  * qualitative examples = the 5 test pairs with the highest and the 5 with the lowest mass AUPRC
    (Hybrid, seed 1), not hand-picked
"""
import argparse
import glob
import json
import math
import os
import pickle
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from gam.data import load_esnli, synthetic_esnli


def norm(w):
    return re.sub(r"[^a-z']", "", w.lower())


def sentiwordnet_words():
    try:
        from nltk.corpus import sentiwordnet as swn
        words = set()
        for s in swn.all_senti_synsets():
            if s.pos_score() > 0 or s.neg_score() > 0:
                for l in s.synset.lemma_names():
                    words.add(l.lower())
        return words
    except Exception as e:  # corpus not installed
        print("SentiWordNet unavailable (pip install nltk; nltk.download('sentiwordnet'); nltk.download('wordnet')):", e)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--esnli_dir", default="data/esnli")
    ap.add_argument("--out", default="results")
    ap.add_argument("--variants", default="hybrid,pure")
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    if a.synthetic:
        test = synthetic_esnli(300, seed=1)[250:]
        train_sents = [e.text_a for e in synthetic_esnli(300, seed=2)] + [e.text_b for e in synthetic_esnli(300, seed=2)]
    else:
        test = load_esnli(a.esnli_dir, "test")
        tr = load_esnli(a.esnli_dir, "train")["train"]
        train_sents = [e.text_a for e in tr] + [e.text_b for e in tr]
    df = Counter()
    for s in train_sents:
        df.update(set(norm(w) for w in s.split()))
    n_docs = len(train_sents)
    idf = {w: math.log(n_docs / (1 + c)) for w, c in df.items()}
    senti = sentiwordnet_words()
    words_per_ex = [[norm(w) for w in e.meta["words_a"] + e.meta["words_b"]] for e in test]

    report = {}
    for v in a.variants.split(","):
        files = sorted(glob.glob(f"{a.runs}/C/esnli/{v}/*/eval/per_example.pkl"))
        if not files:
            continue
        per_seed = []
        for f in files:
            pe = pickle.load(open(f, "rb"))
            per_seed.append(pe)
        stats = []
        for pe in per_seed:
            ws = pe["word_scores"]["mass"]
            masses, words = [], []
            for sc, wl in zip(ws, words_per_ex):
                for m, w in zip(sc, wl):
                    if not np.isnan(m) and w:
                        masses.append(float(m)); words.append(w)
            masses = np.array(masses)
            func = np.array([w in ENGLISH_STOP_WORDS for w in words])
            idf_v = np.array([idf.get(w, math.log(n_docs)) for w in words])
            q10, q90 = np.quantile(masses, [0.1, 0.9])
            top = Counter(w for w, m in zip(words, masses) if m >= q90).most_common(15)
            bot = Counter(w for w, m in zip(words, masses) if m <= q10).most_common(15)
            s = {"mean": float(masses.mean()), "sd": float(masses.std(ddof=1)), "min": float(masses.min()),
                 "max": float(masses.max()), "skew": float(((masses - masses.mean()) ** 3).mean() / masses.std() ** 3),
                 "mean_content": float(masses[~func].mean()), "mean_function": float(masses[func].mean()),
                 "spearman_mass_idf": float(spearmanr(masses, idf_v)[0]),
                 "top10pct_words": top, "bottom10pct_words": bot}
            if senti is not None:
                hi = [w for w, m in zip(words, masses) if m >= q90]
                s["frac_top10pct_in_sentiwordnet"] = float(np.mean([w in senti for w in hi]))
                s["frac_all_in_sentiwordnet"] = float(np.mean([w in senti for w in words]))
            stats.append(s)
        keys = [k for k in stats[0] if isinstance(stats[0][k], float)]
        report[v] = {"per_seed": stats,
                     "mean_over_seeds": {k: float(np.mean([s[k] for s in stats])) for k in keys},
                     "sd_over_seeds": {k: float(np.std([s[k] for s in stats], ddof=1)) if len(stats) > 1 else 0.0
                                       for k in keys}}
        # Pre-specified qualitative examples from seed 1 (first file).
        pe = per_seed[0]
        au = np.array([d["auprc"] for d in pe["plaus_mass"]["per_example"]], float)
        order = np.argsort(np.where(np.isnan(au), np.inf, au))
        valid = [i for i in order if not np.isnan(au[i])]
        def show(i):
            e = test[i]
            w = e.meta["words_a"] + e.meta["words_b"]
            return {"premise": e.text_a, "hypothesis": e.text_b, "auprc": float(au[i]),
                    "words": [(wd, round(float(m), 3), round(h, 2)) for wd, m, h in
                              zip(w, pe["word_scores"]["mass"][i], e.meta["soft_a"] + e.meta["soft_b"])]}
        report[v]["examples_best"] = [show(i) for i in valid[-5:][::-1]]
        report[v]["examples_worst"] = [show(i) for i in valid[:5]]
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "mass_analysis.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({v: r["mean_over_seeds"] for v, r in report.items()}, indent=2))


if __name__ == "__main__":
    main()
