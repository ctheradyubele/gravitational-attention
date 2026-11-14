"""Gravitational attention for BERT-style encoders.

One module implements every variant used in the study, so that the BERT
control and the GAM variants run through identical code:

    standard       alpha = q.k / sqrt(d)                       (BERT control)
    hybrid         alpha = q.k / sqrt(d) + lam * G * m_i m_j / d_ij
    pure           alpha = G * m_i m_j / d_ij
    hybrid_nodist  alpha = q.k / sqrt(d) + lam * G * m_i m_j   (ablation: no distance)
    l2             alpha = -||q_i - k_j||^2 / sqrt(d)          (distance baseline, no mass; Kim et al. 2021)

with d_ij = ||q_i - k_j||^2 + eps, masses m_i = softplus(w_m^T x_i + b_m)
("learned") or m_i = ||x_i||_2 ("norm", the Simplified variant), a learnable
per-head mixing coefficient lam and a learnable per-head gravitational
constant G (parameterised as exp(log_g)). The gravitational term is always
computed in float32 because 1/eps overflows float16.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

MODES = ("standard", "hybrid", "pure", "hybrid_nodist", "l2")
MASS_TYPES = ("learned", "norm")


@dataclass
class GAMConfig:
    mode: str = "hybrid"
    mass_type: str = "learned"
    lambda_init: float = 0.1
    learn_lambda: bool = True
    g_init: float = 1.0
    eps: float = 1e-6
    score_clamp: Optional[float] = None  # None in the primary analysis

    def __post_init__(self):
        assert self.mode in MODES, self.mode
        assert self.mass_type in MASS_TYPES, self.mass_type

    def to_dict(self):
        return asdict(self)


# Named variants used throughout the protocol.
VARIANTS = {
    "standard": GAMConfig(mode="standard"),
    "hybrid": GAMConfig(mode="hybrid"),
    "pure": GAMConfig(mode="pure"),
    "simplified": GAMConfig(mode="hybrid", mass_type="norm"),
    "hybrid_nodist": GAMConfig(mode="hybrid_nodist"),
    "hybrid_fixedlam": GAMConfig(mode="hybrid", lambda_init=0.5, learn_lambda=False),
    "l2": GAMConfig(mode="l2"),
}

MASS_BIAS_INIT = math.log(math.e - 1.0)  # softplus(b) = 1 at initialisation


def pairwise_sq_dist(q: torch.Tensor, k: torch.Tensor, qk: Optional[torch.Tensor] = None) -> torch.Tensor:
    """||q_i - k_j||^2 via ||q||^2 + ||k||^2 - 2 q.k, reusing q.k when given. Shapes [..., T, d]."""
    if qk is None:
        qk = q @ k.transpose(-1, -2)
    d = q.pow(2).sum(-1).unsqueeze(-1) + k.pow(2).sum(-1).unsqueeze(-2) - 2.0 * qk
    return d.clamp_min(0.0)


class GravitationalSelfAttention(nn.Module):
    """Drop-in replacement for transformers' BertSelfAttention (eager path, encoder only)."""

    def __init__(self, hidden_size: int, num_heads: int, attn_dropout: float, gcfg: GAMConfig):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_attention_heads = num_heads
        self.attention_head_size = hidden_size // num_heads
        self.all_head_size = hidden_size
        self.gcfg = gcfg
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(attn_dropout)
        self.is_gam = gcfg.mode not in ("standard", "l2")
        if self.is_gam:
            if gcfg.mass_type == "learned":
                self.mass = nn.Linear(hidden_size, 1)
                nn.init.normal_(self.mass.weight, std=0.02)
                nn.init.constant_(self.mass.bias, MASS_BIAS_INIT)
            if gcfg.mode != "pure":
                self.lam = nn.Parameter(torch.full((num_heads,), float(gcfg.lambda_init)),
                                        requires_grad=gcfg.learn_lambda)
            self.log_g = nn.Parameter(torch.full((num_heads,), math.log(gcfg.g_init)))
        # Instrumentation (off by default, no cost when off).
        self.record = False
        self.last: dict = {}
        self.mass_override: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None

    # ------------------------------------------------------------------
    @classmethod
    def from_bert(cls, bert_self_attn: nn.Module, gcfg: GAMConfig, attn_dropout: float):
        hidden = bert_self_attn.query.in_features
        mod = cls(hidden, bert_self_attn.num_attention_heads, attn_dropout, gcfg)
        for name in ("query", "key", "value"):
            getattr(mod, name).load_state_dict(getattr(bert_self_attn, name).state_dict())
        return mod.to(bert_self_attn.query.weight.device, bert_self_attn.query.weight.dtype)

    def compute_mass(self, x: torch.Tensor) -> torch.Tensor:
        if self.gcfg.mass_type == "learned":
            return F.softplus(self.mass(x).squeeze(-1).float())
        return x.float().norm(dim=-1)

    def _split(self, x):
        b, t, _ = x.shape
        return x.view(b, t, self.num_attention_heads, self.attention_head_size).transpose(1, 2)

    # ------------------------------------------------------------------
    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None,
                encoder_attention_mask=None, past_key_value=None, output_attentions=False, **kwargs):
        if encoder_hidden_states is not None or past_key_value is not None:
            raise NotImplementedError("GAM is implemented for encoder self-attention only")
        q = self._split(self.query(hidden_states))
        k = self._split(self.key(hidden_states))
        v = self._split(self.value(hidden_states))
        qk = q @ k.transpose(-1, -2)                                   # [B,H,T,T]
        mode = self.gcfg.mode
        scores = qk.float() / math.sqrt(self.attention_head_size)
        mass = dist = None
        if mode == "l2":
            scores = -pairwise_sq_dist(q.float(), k.float(), qk.float()) / math.sqrt(self.attention_head_size)
        if self.is_gam:
            mass = self.compute_mass(hidden_states)                    # [B,T]
            if self.mass_override is not None:
                mass = self.mass_override(mass, attention_mask)
            g = self.log_g.exp().view(1, -1, 1, 1)
            mm = mass[:, None, :, None] * mass[:, None, None, :]       # [B,1,T,T]
            if mode == "hybrid_nodist":
                grav = g * mm
            else:
                dist = pairwise_sq_dist(q.float(), k.float(), qk.float()) + self.gcfg.eps
                grav = g * mm / dist
            if self.gcfg.score_clamp is not None:
                grav = grav.clamp(max=self.gcfg.score_clamp)
            if mode == "pure":
                scores = grav
            else:
                scores = scores + self.lam.view(1, -1, 1, 1) * grav
        if attention_mask is not None:
            scores = scores + attention_mask.float()
        probs = torch.softmax(scores, dim=-1).to(v.dtype)
        probs_dropped = self.dropout(probs)
        if head_mask is not None:
            probs_dropped = probs_dropped * head_mask
        ctx = (probs_dropped @ v).transpose(1, 2).contiguous()
        ctx = ctx.view(ctx.shape[0], ctx.shape[1], self.all_head_size)
        if self.record:
            self._record(mass, dist, scores, attention_mask)
        return (ctx, probs) if output_attentions else (ctx,)

    @torch.no_grad()
    def _record(self, mass, dist, scores, attention_mask):
        rec = {}
        if attention_mask is not None:
            valid = (attention_mask[:, 0, 0, :] > -1.0).float()       # [B,T]
        else:
            valid = None
        if mass is not None:
            rec["mass"] = mass.detach()
            rec["lam"] = self.lam.detach().clone() if hasattr(self, "lam") else None
            rec["g"] = self.log_g.exp().detach().clone()
            if dist is not None:
                vm = valid if valid is not None else torch.ones_like(mass)
                pair = (vm[:, :, None] * vm[:, None, :])[:, None]          # [B,1,T,T]
                inv_r = dist.rsqrt()                                        # 1/r
                e_raw = -(mass[:, None, :, None] * mass[:, None, None, :] * inv_r * pair).sum((-1, -2))
                mean_m = (mass * vm).sum(-1) / vm.sum(-1).clamp_min(1.0)    # per sequence
                mn = mass / mean_m[:, None]
                e_norm = -(mn[:, None, :, None] * mn[:, None, None, :] * inv_r * pair).sum((-1, -2))
                rec["energy_raw"] = e_raw.mean().item()
                rec["energy_norm"] = e_norm.mean().item()
                rec["mean_mass"] = mean_m.mean().item()
        rec["max_score"] = scores.max().item()
        self.last = rec


# ----------------------------------------------------------------------
def encoder_layers(model):
    base = getattr(model, "base_model", model)
    return base.encoder.layer


def convert_bert(model: nn.Module, gcfg: GAMConfig) -> nn.Module:
    """Replace every BertSelfAttention in `model` with GravitationalSelfAttention (in place)."""
    p = model.config.attention_probs_dropout_prob
    for layer in encoder_layers(model):
        layer.attention.self = GravitationalSelfAttention.from_bert(layer.attention.self, gcfg, p)
    model.config.gam = gcfg.to_dict()
    return model


def gam_modules(model):
    return [layer.attention.self for layer in encoder_layers(model)
            if isinstance(layer.attention.self, GravitationalSelfAttention)]


def set_record(model, flag: bool):
    for m in gam_modules(model):
        m.record = flag
        if not flag:
            m.last = {}


def gam_param_names(model):
    names = []
    for n, p in model.named_parameters():
        if n.endswith(".lam") or n.endswith(".log_g") or ".mass." in n:
            names.append(n)
    return names


def param_groups(model, lr: float, gam_lr: float, weight_decay: float = 0.01):
    """Standard AdamW groups; GAM scalars/mass head get their own learning rate and no decay."""
    gam_names = set(gam_param_names(model))
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    groups = {"decay": [], "nodecay": [], "gam": []}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n in gam_names:
            groups["gam"].append(p)
        elif any(nd in n for nd in no_decay):
            groups["nodecay"].append(p)
        else:
            groups["decay"].append(p)
    out = [{"params": groups["decay"], "lr": lr, "weight_decay": weight_decay},
           {"params": groups["nodecay"], "lr": lr, "weight_decay": 0.0}]
    if groups["gam"]:
        out.append({"params": groups["gam"], "lr": gam_lr, "weight_decay": 0.0})
    return [g for g in out if g["params"]]


def lambda_table(model):
    """Per-layer, per-head lambda and G (lists), for logging."""
    lam, g = [], []
    for m in gam_modules(model):
        lam.append(m.lam.detach().float().cpu().tolist() if hasattr(m, "lam") else None)
        g.append(m.log_g.exp().detach().float().cpu().tolist() if m.is_gam else None)
    return {"lambda": lam, "G": g}


# ----------------------------------------------------------------------
# Generic module for non-BERT encoders (used by the LRA adapter).
class GravitationalAttention(nn.Module):
    """x: [B,T,D], key_padding_mask: [B,T] bool (True = keep). Returns [B,T,D]."""

    def __init__(self, dim: int, num_heads: int, gcfg: GAMConfig, dropout: float = 0.0):
        super().__init__()
        self.inner = GravitationalSelfAttention(dim, num_heads, dropout, gcfg)
        self.out = nn.Linear(dim, dim)

    def forward(self, x, key_padding_mask=None):
        ext = None
        if key_padding_mask is not None:
            ext = (1.0 - key_padding_mask[:, None, None, :].float()) * torch.finfo(torch.float32).min
        return self.out(self.inner(x, attention_mask=ext)[0])


# ----------------------------------------------------------------------
@torch.no_grad()
def calibrate_g(model, inputs: dict) -> list:
    """Data-dependent initialisation of the gravitational constant G (pre-specified in the protocol).

    For each GAM layer in turn (bottom to top, re-running the model after each layer so later layers
    see calibrated inputs), set G per head so that the median gravitational score over valid query-key
    pairs equals the standard deviation of the scaled dot-product scores of that head. Without this,
    m_i m_j / d_ij is about 100x smaller than q.k/sqrt(d) in BERT-base and the gravitational term is
    inert at initialisation. Returns the calibrated G values per layer.
    """
    was_training = model.training
    model.eval()
    mods = [m for m in gam_modules(model) if m.is_gam]
    out = []
    for mod in mods:
        cap = {}

        def hook(module, args, kwargs):
            cap["h"] = args[0] if args else kwargs["hidden_states"]
            cap["am"] = args[1] if len(args) > 1 else kwargs.get("attention_mask")
        handle = mod.register_forward_pre_hook(hook, with_kwargs=True)
        model(**inputs)
        handle.remove()
        h, am = cap["h"], cap["am"]
        q, k = mod._split(mod.query(h)).float(), mod._split(mod.key(h)).float()
        qk = q @ k.transpose(-1, -2)
        std_scores = qk / math.sqrt(mod.attention_head_size)
        m = mod.compute_mass(h)
        mm = (m[:, None, :, None] * m[:, None, None, :])
        if mod.gcfg.mode == "hybrid_nodist":
            raw = mm.expand(-1, q.shape[1], -1, -1)     # same mass product for every head
        else:
            raw = mm / (pairwise_sq_dist(q, k, qk) + mod.gcfg.eps)
        if am is not None:
            v = am[:, 0, 0, :] > -1.0
            pair = (v[:, :, None] & v[:, None, :])[:, None].expand_as(raw)
        else:
            pair = torch.ones_like(raw, dtype=torch.bool)
        g = []
        for hd in range(raw.shape[1]):
            r = raw[:, hd][pair[:, hd]]
            s = std_scores[:, hd][pair[:, hd]]
            g.append(float(s.std() / r.median().clamp_min(1e-12)))
        mod.log_g.data = torch.log(torch.tensor(g, device=mod.log_g.device, dtype=mod.log_g.dtype))
        out.append(g)
    model.train(was_training)
    return out
