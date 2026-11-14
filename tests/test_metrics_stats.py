import csv
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from analysis.stats import holm, noninferiority, paired_bootstrap, seed_paired_t, steiger_z
from gam.data import esnli_stats, heldout_split, load_esnli_csv, parse_marked
from gam.importance import to_words
from gam.metrics import faithfulness, plausibility, plausibility_per_example, remove_words, topk_indices


def test_plausibility_perfect_and_inverted():
    y = np.array([0, 1, 0, 1, 0.0])
    good = plausibility_per_example(np.array([0.1, 0.9, 0.2, 0.8, 0.0]), y)
    bad = plausibility_per_example(np.array([0.9, 0.1, 0.8, 0.2, 1.0]), y)
    assert good["auprc"] == 1.0 and good["token_f1"] == 1.0
    assert bad["auprc"] < 0.5 and bad["token_f1"] == 0.0


def test_plausibility_pooled_correlation_sign():
    rng = np.random.RandomState(0)
    ys = [rng.randint(0, 2, 12).astype(float) for _ in range(50)]
    good = plausibility([y + 0.3 * rng.randn(12) for y in ys], ys)
    rand = plausibility([rng.randn(12) for _ in ys], ys)
    assert good["pearson_r"] > 0.5 and abs(rand["pearson_r"]) < 0.15


def test_remove_and_keep_words():
    a, b = ["the", "dog", "runs"], ["a", "cat"]
    assert remove_words(a, b, [1, 4], keep=False) == (["the", "runs"], ["a"])
    assert remove_words(a, b, [1, 4], keep=True) == (["dog"], ["cat"])
    assert list(topk_indices(np.array([0.1, 0.9, np.nan, 0.5]), 0.5)) == [1, 3]


def test_faithfulness_rewards_the_true_rationale():
    # Toy model: probability of class 1 = fraction of words that are "key".
    def predict(pairs):
        out = []
        for wa, wb in pairs:
            w = wa + wb
            p = (sum(x == "key" for x in w) + 1) / (len(w) + 2)
            out.append([1 - p, p])
        return np.array(out)
    pairs = [(["key", "x", "x"], ["key", "x"]) for _ in range(20)]
    true = [np.array([1, 0, 0, 1, 0.0]) for _ in pairs]
    rand = [np.random.RandomState(i).rand(5) for i in range(20)]
    preds = np.ones(20, int)
    ft, fr = faithfulness(predict, pairs, true, preds), faithfulness(predict, pairs, rand, preds)
    assert ft["comprehensiveness"] > fr["comprehensiveness"]
    assert ft["sufficiency"] < fr["sufficiency"]


def test_to_words_mean_and_truncation():
    s = np.array([0.0, 1.0, 3.0, 2.0, 0.0])
    m = [None, 0, 0, 1, None]
    assert np.allclose(to_words(s, m, 3, "mean")[:2], [2.0, 2.0])
    assert to_words(s, m, 3, "max")[0] == 3.0
    assert np.isnan(to_words(s, m, 3)[2])


def test_steiger_null_calibration():
    """Under H0 (equal correlations with y) the test should reject about 5% of the time."""
    rng = np.random.RandomState(0)
    rej, n = 0, 300
    for _ in range(400):
        y = rng.randn(n)
        shared = rng.randn(n)
        a = 0.4 * y + 0.6 * shared + 0.7 * rng.randn(n)
        b = 0.4 * y + 0.6 * shared + 0.7 * rng.randn(n)
        r12, r13, r23 = np.corrcoef(y, a)[0, 1], np.corrcoef(y, b)[0, 1], np.corrcoef(a, b)[0, 1]
        rej += steiger_z(r12, r13, r23, n)["p"] < 0.05
    assert 0.02 < rej / 400 < 0.09


def test_steiger_detects_difference():
    rng = np.random.RandomState(1)
    n = 2000
    y = rng.randn(n)
    a, b = 0.6 * y + rng.randn(n), 0.3 * y + rng.randn(n)
    r = steiger_z(np.corrcoef(y, a)[0, 1], np.corrcoef(y, b)[0, 1], np.corrcoef(a, b)[0, 1], n)
    assert r["z"] > 3 and r["p"] < 0.001


def test_paired_bootstrap():
    rng = np.random.RandomState(0)
    a = rng.rand(500) + 0.1
    b = a - 0.1 + 0.01 * rng.randn(500)
    r = paired_bootstrap(a, b, n_boot=2000)
    assert r["ci95"][0] > 0.09 and r["p"] < 0.01
    same = paired_bootstrap(a, a + 0.01 * rng.randn(500), n_boot=2000)
    assert same["ci95"][0] < 0 < same["ci95"][1]


def test_holm_and_noninferiority_and_t():
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj["a"] == 0.03 and adj["c"] == 0.06 and adj["b"] == 0.06
    assert noninferiority([82.9, 83.1, 83.0, 82.8, 83.2], [83.0, 83.0, 83.1, 82.9, 83.0], 1.0)["noninferior"]
    assert not noninferiority([80.0, 80.5, 79.8, 80.2, 80.1], [83.0, 83.0, 83.1, 82.9, 83.0], 1.0)["noninferior"]
    assert seed_paired_t([1, 2, 3, 4, 5], [0, 1, 2, 3, 4.1])["p"] < 0.001


def test_heldout_split_is_fixed_and_disjoint():
    a1, b1 = heldout_split(1000)
    a2, b2 = heldout_split(1000)
    assert a1 == a2 and b1 == b2 and not set(a1) & set(b1) and len(b1) == 100


def test_esnli_parser():
    assert parse_marked("A *dog* runs *fast.*") == (["A", "dog", "runs", "fast."], [0, 1, 0, 1])
    assert parse_marked("*big dog* sleeps") == (["big", "dog", "sleeps"], [1, 1, 0])
    rows = [
        {"pairID": "1", "gold_label": "contradiction", "Sentence1": "A dog runs", "Sentence2": "A cat sleeps",
         "Sentence1_marked_1": "A *dog* runs", "Sentence2_marked_1": "A *cat* sleeps",
         "Sentence1_marked_2": "A *dog* *runs*", "Sentence2_marked_2": "A cat *sleeps*",
         "Sentence1_marked_3": "A dog runs", "Sentence2_marked_3": "A *cat* sleeps"},
        {"pairID": "2", "gold_label": "-", "Sentence1": "x", "Sentence2": "y",
         "Sentence1_marked_1": "x", "Sentence2_marked_1": "y"},
    ]
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in rows[0]})
    exs = load_esnli_csv([f.name], with_rationales=True)
    assert len(exs) == 1
    e = exs[0]
    assert e.meta["n_annotators"] == 3
    assert np.allclose(e.meta["soft_a"], [0, 2 / 3, 1 / 3]) and np.allclose(e.meta["soft_b"], [0, 2 / 3, 1 / 3])
    st = esnli_stats(exs)
    assert st["n_examples"] == 1 and st["frac_examples_with_majority_highlight"] == 1.0
