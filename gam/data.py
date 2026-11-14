"""Data loading: GLUE, e-SNLI (original CSVs), WikiText-103 and synthetic stand-ins for tests."""
from __future__ import annotations

import csv
import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

GLUE_TASKS = {
    # task: (text keys, num_labels, official eval split, primary metric)
    "cola": (("sentence", None), 2, "validation", "mcc"),
    "sst2": (("sentence", None), 2, "validation", "acc"),
    "mrpc": (("sentence1", "sentence2"), 2, "validation", "f1"),
    "stsb": (("sentence1", "sentence2"), 1, "validation", "pearson"),
    "qqp": (("question1", "question2"), 2, "validation", "f1"),
    "mnli": (("premise", "hypothesis"), 3, "validation_matched", "acc"),
    "qnli": (("question", "sentence"), 2, "validation", "acc"),
    "rte": (("sentence1", "sentence2"), 2, "validation", "acc"),
}
ESNLI_LABELS = {"entailment": 0, "neutral": 1, "contradiction": 2}
HELDOUT_FRACTION = 0.10
SPLIT_SEED = 1234  # fixed, independent of the run seed


@dataclass
class Example:
    text_a: str
    text_b: Optional[str]
    label: float
    meta: Dict = field(default_factory=dict)


def heldout_split(n: int, fraction: float = HELDOUT_FRACTION, seed: int = SPLIT_SEED, max_heldout: int = 5000):
    """Fixed train/held-out index split used for model selection (official dev is never used for it)."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    k = min(max(1, int(round(n * fraction))), max_heldout)
    return idx[k:].tolist(), idx[:k].tolist()


def load_glue(task: str, cache_dir: Optional[str] = None):
    from datasets import load_dataset
    (ka, kb), _, eval_split, _ = GLUE_TASKS[task]
    ds = load_dataset("nyu-mll/glue", task, cache_dir=cache_dir)

    def conv(split):
        return [Example(r[ka], r[kb] if kb else None, float(r["label"])) for r in ds[split]]

    train_all = conv("train")
    tr, ho = heldout_split(len(train_all))
    return {"train": [train_all[i] for i in tr], "heldout": [train_all[i] for i in ho], "test": conv(eval_split)}


# ----------------------------------------------------------------------
# e-SNLI from the original release (github.com/OanaMariaCamburu/e-SNLI, dataset/*.csv).
_STAR = re.compile(r"\*([^*]+)\*")


def parse_marked(marked: str):
    """'A *dog* runs .' -> (['A','dog','runs','.'], [0,1,0,0]). Words are whitespace tokens."""
    words, labels = [], []
    for tok in marked.split():
        hit = "*" in tok
        w = tok.replace("*", "")
        if w == "":
            continue
        words.append(w)
        labels.append(1 if hit else 0)
    return words, labels


def load_esnli_csv(paths: List[str], with_rationales: bool) -> List[Example]:
    out = []
    for p in paths:
        with open(p, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                lab = row.get("gold_label", "").strip()
                if lab not in ESNLI_LABELS:
                    continue
                ex = Example(row["Sentence1"], row["Sentence2"], float(ESNLI_LABELS[lab]),
                             {"pairID": row.get("pairID")})
                if with_rationales:
                    ann = []
                    for a in (1, 2, 3):
                        k1, k2 = f"Sentence1_marked_{a}", f"Sentence2_marked_{a}"
                        if k1 in row and row[k1] and row.get(k2):
                            w1, l1 = parse_marked(row[k1])
                            w2, l2 = parse_marked(row[k2])
                            ann.append((w1, l1, w2, l2))
                    if not ann:
                        continue
                    # Words are taken from annotator 1; others must align word-for-word.
                    w1, _, w2, _ = ann[0]
                    ann = [a for a in ann if a[0] == w1 and a[2] == w2]
                    lab1 = np.mean([a[1] for a in ann], axis=0).tolist()
                    lab2 = np.mean([a[3] for a in ann], axis=0).tolist()
                    ex = Example(" ".join(w1), " ".join(w2), ex.label, {
                        **ex.meta, "words_a": w1, "words_b": w2,
                        "soft_a": lab1, "soft_b": lab2, "n_annotators": len(ann)})
                out.append(ex)
    return out


def esnli_stats(examples: List[Example]) -> Dict:
    """Sanity statistics the authors should inspect after the first load."""
    soft = [s for e in examples for s in e.meta["soft_a"] + e.meta["soft_b"]]
    maj = [1 if s >= 0.5 else 0 for s in soft]
    return {
        "n_examples": len(examples),
        "mean_annotators": float(np.mean([e.meta["n_annotators"] for e in examples])),
        "frac_words_any_highlight": float(np.mean([s > 0 for s in soft])),
        "frac_words_majority_highlight": float(np.mean(maj)),
        "frac_examples_with_majority_highlight": float(np.mean(
            [any(s >= 0.5 for s in e.meta["soft_a"] + e.meta["soft_b"]) for e in examples])),
    }


def load_esnli(data_dir: str, split: str):
    import os
    if split == "train":
        exs = load_esnli_csv([os.path.join(data_dir, f) for f in ("esnli_train_1.csv", "esnli_train_2.csv")], False)
        tr, ho = heldout_split(len(exs))
        return {"train": [exs[i] for i in tr], "heldout": [exs[i] for i in ho]}
    if split == "test":
        return load_esnli_csv([os.path.join(data_dir, "esnli_test.csv")], True)
    raise ValueError(split)


# ----------------------------------------------------------------------
def load_wikitext(cache_dir: Optional[str] = None):
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", cache_dir=cache_dir)
    return {s: [t for t in ds[s]["text"] if t.strip()] for s in ("train", "validation")}


# ----------------------------------------------------------------------
# Synthetic stand-ins (tests and pipeline smoke runs only; never for reported results).
SYN_VOCAB = ["good", "bad", "not", "very", "the", "a", "dog", "cat", "runs", "sleeps", "is", "and", "man", "park"]


def synthetic_classification(n: int, num_labels: int, pair: bool, seed: int = 0) -> List[Example]:
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        a = [rng.choice(SYN_VOCAB) for _ in range(rng.randint(4, 10))]
        b = [rng.choice(SYN_VOCAB) for _ in range(rng.randint(4, 10))] if pair else None
        score = a.count("good") - a.count("bad") - (1 if "not" in a else 0)
        if num_labels == 1:
            label = float(score)
        elif num_labels == 2:
            label = float(score > 0)
        else:
            label = float(min(2, max(0, score + 1)))
        out.append(Example(" ".join(a), " ".join(b) if b else None, label))
    return out


def synthetic_esnli(n: int, seed: int = 0) -> List[Example]:
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        wa = [rng.choice(SYN_VOCAB) for _ in range(rng.randint(4, 9))]
        wb = [rng.choice(SYN_VOCAB) for _ in range(rng.randint(4, 9))]
        sa = [float(w in ("good", "bad", "not")) for w in wa]
        sb = [float(w in ("good", "bad", "not")) for w in wb]
        if sum(sa) + sum(sb) == 0:
            sa[0] = 1.0
        label = float(rng.randint(0, 2))
        out.append(Example(" ".join(wa), " ".join(wb), label,
                           {"words_a": wa, "words_b": wb, "soft_a": sa, "soft_b": sb, "n_annotators": 1}))
    return out


def tiny_tokenizer(path: str):
    """A real BertTokenizerFast over a tiny vocabulary (for offline tests)."""
    import os
    from transformers import BertTokenizerFast
    os.makedirs(path, exist_ok=True)
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", ".", ","] + SYN_VOCAB
    with open(os.path.join(path, "vocab.txt"), "w") as f:
        f.write("\n".join(vocab) + "\n")
    return BertTokenizerFast(os.path.join(path, "vocab.txt"), do_lower_case=True)
