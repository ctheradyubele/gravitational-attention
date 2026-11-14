import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
from transformers import BertConfig, BertForSequenceClassification

from gam.attention import (GAMConfig, GravitationalSelfAttention, VARIANTS, convert_bert, gam_modules,
                           gam_param_names, pairwise_sq_dist, param_groups, set_record)


def tiny_model(seed=0):
    torch.manual_seed(seed)
    cfg = BertConfig(vocab_size=50, hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
                     intermediate_size=64, max_position_embeddings=64, attn_implementation="eager",
                     hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0)
    return BertForSequenceClassification(cfg).eval()


def batch():
    torch.manual_seed(1)
    ids = torch.randint(5, 50, (3, 10))
    am = torch.ones_like(ids)
    am[1, 7:] = 0
    am[2, 4:] = 0
    return {"input_ids": ids, "attention_mask": am}


def test_standard_variant_reproduces_bert_exactly():
    ref = tiny_model()
    gam = convert_bert(copy.deepcopy(ref), VARIANTS["standard"])
    b = batch()
    with torch.no_grad():
        assert torch.allclose(ref(**b).logits, gam(**b).logits, atol=1e-6)


def test_hybrid_with_lambda_zero_equals_standard():
    ref = tiny_model()
    gam = convert_bert(copy.deepcopy(ref), GAMConfig(mode="hybrid", lambda_init=0.0))
    b = batch()
    with torch.no_grad():
        assert torch.allclose(ref(**b).logits, gam(**b).logits, atol=1e-6)


def test_hybrid_with_nonzero_lambda_differs():
    ref = tiny_model()
    gam = convert_bert(copy.deepcopy(ref), GAMConfig(mode="hybrid", lambda_init=0.5, g_init=10.0))
    b = batch()
    with torch.no_grad():
        assert not torch.allclose(ref(**b).logits, gam(**b).logits, atol=1e-4)


def test_pairwise_distance_identity_matches_cdist():
    torch.manual_seed(0)
    q, k = torch.randn(2, 3, 7, 8, dtype=torch.float64), torch.randn(2, 3, 7, 8, dtype=torch.float64)
    ref = torch.cdist(q, k) ** 2
    assert torch.allclose(pairwise_sq_dist(q, k), ref, atol=1e-9)


@pytest.mark.parametrize("variant", ["hybrid", "pure", "simplified", "hybrid_nodist", "hybrid_fixedlam", "l2"])
def test_attention_rows_are_distributions_and_padding_is_masked(variant):
    m = convert_bert(tiny_model(), VARIANTS[variant])
    b = batch()
    with torch.no_grad():
        att = m(**b, output_attentions=True).attentions
    for a in att:
        assert torch.allclose(a.sum(-1), torch.ones_like(a.sum(-1)), atol=1e-5)
        assert a[1, :, :, 7:].abs().max() < 1e-6 and a[2, :, :, 4:].abs().max() < 1e-6
        assert torch.isfinite(a).all()


def test_masses_positive_and_energy_recorded():
    m = convert_bert(tiny_model(), VARIANTS["hybrid"])
    set_record(m, True)
    with torch.no_grad():
        m(**batch())
    for mod in gam_modules(m):
        assert (mod.last["mass"] > 0).all()
        assert all(torch.isfinite(torch.tensor(mod.last[k])) for k in ("energy_raw", "energy_norm", "max_score"))
    set_record(m, False)
    assert all(mod.last == {} for mod in gam_modules(m))


def test_query_mass_only_rescales_row_temperature():
    """m_i multiplies every logit in row i; ranking within the row depends only on m_j / d_ij."""
    torch.manual_seed(0)
    mod = GravitationalSelfAttention(16, 2, 0.0, GAMConfig(mode="pure")).eval()
    x = torch.randn(1, 6, 16)
    with torch.no_grad():
        q = mod._split(mod.query(x)); k = mod._split(mod.key(x))
        m = mod.compute_mass(x)
        d = pairwise_sq_dist(q, k) + mod.gcfg.eps
        scores = m[:, None, :, None] * m[:, None, None, :] / d
        rank_full = scores.argsort(-1)
        rank_key_only = (m[:, None, None, :] / d).argsort(-1)
    assert torch.equal(rank_full, rank_key_only)


def test_gradients_reach_gam_parameters_and_groups_separate():
    m = convert_bert(tiny_model(), GAMConfig(mode="hybrid", lambda_init=0.1, g_init=5.0))
    m.train()
    m(**batch(), labels=torch.tensor([0, 1, 0])).loss.backward()
    names = gam_param_names(m)
    assert any(n.endswith(".lam") for n in names) and any(n.endswith(".log_g") for n in names)
    params = dict(m.named_parameters())
    for n in names:
        assert params[n].grad is not None and torch.isfinite(params[n].grad).all(), n
    groups = param_groups(m, 2e-5, 1e-3)
    assert sorted(g["lr"] for g in groups) == [2e-5, 2e-5, 1e-3]
    n_grouped = sum(len(g["params"]) for g in groups)
    assert n_grouped == sum(1 for p in m.parameters() if p.requires_grad)


def test_fixed_lambda_is_frozen():
    m = convert_bert(tiny_model(), VARIANTS["hybrid_fixedlam"])
    for mod in gam_modules(m):
        assert not mod.lam.requires_grad and torch.allclose(mod.lam, torch.full_like(mod.lam, 0.5))


def test_mass_override_changes_output():
    m = convert_bert(tiny_model(), GAMConfig(mode="pure", g_init=50.0))
    b = batch()
    with torch.no_grad():
        base = m(**b).logits
        for mod in gam_modules(m):
            mod.mass_override = lambda mass, am: torch.ones_like(mass) * 3.0
        changed = m(**b).logits
    assert not torch.allclose(base, changed)


def test_fp32_gravity_under_half_precision_inputs():
    mod = GravitationalSelfAttention(16, 2, 0.0, GAMConfig(mode="hybrid", lambda_init=1.0)).to(torch.bfloat16)
    x = torch.randn(1, 5, 16, dtype=torch.bfloat16)
    y = mod(x)[0]
    assert torch.isfinite(y.float()).all()


def test_calibrate_g_matches_scales_in_first_layer():
    from gam.attention import calibrate_g
    m = convert_bert(tiny_model(), VARIANTS["pure"])
    b = batch()
    gs = calibrate_g(m, b)
    assert len(gs) == 2 and all(g > 0 for layer in gs for g in layer)
    mod = gam_modules(m)[0]
    with torch.no_grad():
        h = m.bert.embeddings(input_ids=b["input_ids"])
        q, k = mod._split(mod.query(h)), mod._split(mod.key(h))
        qk = q @ k.transpose(-1, -2)
        mass = mod.compute_mass(h)
        grav = mod.log_g.exp().view(1, -1, 1, 1) * mass[:, None, :, None] * mass[:, None, None, :] / (pairwise_sq_dist(q, k, qk) + 1e-6)
        valid = b["attention_mask"].bool()
        pair = (valid[:, :, None] & valid[:, None, :])
        for hd in range(grav.shape[1]):
            med = grav[:, hd][pair].median()
            sd = (qk[:, hd] / (q.shape[-1] ** 0.5))[pair].std()
            assert torch.isclose(med, sd, rtol=1e-3)


def test_l2_variant_is_softmax_of_negative_distance():
    torch.manual_seed(0)
    mod = GravitationalSelfAttention(16, 2, 0.0, VARIANTS["l2"]).eval()
    x = torch.randn(1, 5, 16)
    with torch.no_grad():
        _, probs = mod(x, output_attentions=True)
        q, k = mod._split(mod.query(x)), mod._split(mod.key(x))
        ref = torch.softmax(-(torch.cdist(q, k) ** 2) / (8 ** 0.5), -1)
    assert torch.allclose(probs, ref, atol=1e-5)
    assert not mod.is_gam and not any(n.endswith((".lam", ".log_g")) for n, _ in mod.named_parameters())


@pytest.mark.parametrize("variant", ["hybrid", "pure", "simplified", "hybrid_nodist", "hybrid_fixedlam"])
def test_calibrate_then_save_load_roundtrip(variant, tmp_path):
    """Calibrated G must keep one value per head and survive an MLM -> classifier checkpoint load."""
    from gam.attention import calibrate_g
    m = convert_bert(tiny_model(), VARIANTS[variant])
    gs = calibrate_g(m, batch())
    assert all(len(g) == 4 for g in gs)
    for mod in gam_modules(m):
        assert mod.log_g.shape == (4,) and torch.isfinite(mod.log_g).all()
    sd = m.state_dict()
    m2 = convert_bert(tiny_model(seed=1), VARIANTS[variant])
    m2.load_state_dict(sd)
    with torch.no_grad():
        assert torch.allclose(m(**batch()).logits, m2(**batch()).logits, atol=1e-6)
