"""Model construction, checkpoint I/O, encoding and evaluation helpers."""
from __future__ import annotations

import json
import os
from typing import List, Optional

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score, matthews_corrcoef

from .attention import GAMConfig, VARIANTS, convert_bert
from .data import Example

TINY_CONFIG = dict(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
                   max_position_embeddings=128)


def variant_config(name: str) -> GAMConfig:
    import copy
    return copy.deepcopy(VARIANTS[name])


def build_model(kind: str, variant: str, base: str = "bert-base-uncased", num_labels: int = 2,
                init_ckpt: Optional[str] = None, tokenizer=None, synthetic: bool = False):
    """kind: 'mlm' or 'cls'. init_ckpt: directory written by save_checkpoint (adapted MLM or fine-tuned)."""
    from transformers import BertConfig, BertForMaskedLM, BertForSequenceClassification
    cls_ = BertForMaskedLM if kind == "mlm" else BertForSequenceClassification
    extra = {} if kind == "mlm" else {"num_labels": num_labels}
    if synthetic:
        cfg = BertConfig(vocab_size=len(tokenizer), attn_implementation="eager", **TINY_CONFIG, **extra)
        torch.manual_seed(0)  # identical random "pre-trained" weights for every variant
        model = cls_(cfg)
    else:
        model = cls_.from_pretrained(base, attn_implementation="eager", **extra)
    gcfg = variant_config(variant)
    convert_bert(model, gcfg)
    if init_ckpt:
        load_encoder_weights(model, init_ckpt, kind)
    return model


def save_checkpoint(model, path: str, meta: dict):
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(path, "model.pt"))
    with open(os.path.join(path, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def load_encoder_weights(model, path: str, kind: str):
    """Load a checkpoint. From an MLM checkpoint into a classifier: encoder + embeddings only."""
    sd = torch.load(os.path.join(path, "model.pt"), map_location="cpu")
    meta = json.load(open(os.path.join(path, "meta.json")))
    if meta.get("kind") == kind:
        model.load_state_dict(sd, strict=True)
        return
    assert meta.get("kind") == "mlm" and kind == "cls", (meta.get("kind"), kind)
    enc = {k: v for k, v in sd.items() if k.startswith("bert.") and not k.startswith("bert.pooler")}
    missing, unexpected = model.load_state_dict(enc, strict=False)
    assert not unexpected, unexpected
    allowed = ("bert.pooler", "classifier")
    bad = [m for m in missing if not m.startswith(allowed)]
    assert not bad, f"weights missing from adapted checkpoint: {bad[:5]}"


# ----------------------------------------------------------------------
def encode(tokenizer, examples: List[Example], max_len: int):
    a = [e.text_a for e in examples]
    b = [e.text_b for e in examples] if examples and examples[0].text_b is not None else None
    enc = tokenizer(a, b, truncation=True, max_length=max_len, padding=True, return_tensors="pt")
    labels = torch.tensor([e.label for e in examples])
    return enc, labels


def batches(examples, bs, shuffle, seed):
    idx = list(range(len(examples)))
    if shuffle:
        rng = np.random.RandomState(seed)
        rng.shuffle(idx)
    for i in range(0, len(idx), bs):
        yield [examples[j] for j in idx[i:i + bs]]


@torch.no_grad()
def predict(model, tokenizer, examples, max_len, bs, device, regression=False):
    model.eval()
    preds = []
    for chunk in batches(examples, bs, False, 0):
        enc, _ = encode(tokenizer, chunk, max_len)
        logits = model(**{k: v.to(device) for k, v in enc.items()}).logits.float()
        preds.append(logits.squeeze(-1).cpu() if regression else logits.cpu())
    return torch.cat(preds)


def task_metrics(task: str, logits: torch.Tensor, labels: np.ndarray) -> dict:
    if task == "stsb":
        p = logits.numpy()
        return {"pearson": float(pearsonr(p, labels)[0]) * 100, "spearman": float(spearmanr(p, labels)[0]) * 100}
    pred = logits.argmax(-1).numpy()
    y = labels.astype(int)
    out = {"acc": float((pred == y).mean()) * 100}
    if task == "cola":
        out["mcc"] = float(matthews_corrcoef(y, pred)) * 100
    if task in ("mrpc", "qqp"):
        out["f1"] = float(f1_score(y, pred)) * 100
    return out
