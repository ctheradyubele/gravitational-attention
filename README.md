# Gravitational Attention in Pretrained Transformers

Code, experimental protocol and results for the paper

> **Gravitational Attention in Pretrained Transformers: An Interpretability–Accuracy Trade-Off**
> [Authors]. *Applied Sciences* (MDPI), 2026 (under review).

Gravitational attention gives each token a learned **mass** and scores query–key pairs with an
inverse-square law of their distance:

$$g_{ij} = G\,\frac{m_i\,m_j}{\lVert q_i - k_j \rVert^2 + \epsilon}, \qquad m_i = \mathrm{softplus}(w_m^\top x_i + b_m)$$

*Hybrid GAM* adds this score to standard dot-product attention with a learned weight λ;
*Pure GAM* uses it alone. The masses are intended as a built-in measure of token importance.
The study tests, under a pre-specified protocol, whether these masses are plausible (agree with
human rationales), faithful (the model depends on them) and compatible with accuracy.

**Main findings.** Hybrid GAM matches BERT on GLUE (81.64 vs 81.59) but learns small mixing
weights and largely ignores its masses. Pure GAM relies on its masses, which were the most plausible
importance scores measured (AUPRC 0.625), but it loses 5.4 GLUE points. Gravitational attention
therefore exposes an interpretability–accuracy trade-off.

---

## Repository layout

```
gam/                 Model code: gravitational attention, BERT integration, data loading,
                     importance scores (attention, rollout, Integrated Gradients), metrics
scripts/             adapt_mlm.py (Phase A), finetune.py (B1, B2, D), eval_esnli.py (C),
                     benchmark.py (F), run_matrix.py (prints the command list for each phase)
configs/             protocol.yaml: every hyperparameter, variant, task and seed, fixed in advance
analysis/            aggregate.py (all tables and hypothesis tests), stats.py, mass_analysis.py
results/             Analysis outputs reported in the paper (CSV/JSON); see below
figures/             make_figures.py and the figures it draws from results/
notebooks/           Colab notebook used to run the study (Colab Pro+, NVIDIA A100)
tests/               34 unit tests and a synthetic end-to-end smoke test
lra/                 Notes on the optional Long Range Arena experiment (not run)
DEVIATIONS.md        Deviations from the protocol and infrastructure events
```

## Variants

| Variant | Attention score | Mass | Role |
| --- | --- | --- | --- |
| `standard` | dot product | – | baseline (BERT) |
| `standard_raw` | dot product, no continued pretraining | – | baseline |
| `hybrid` | dot product + λ · gravitational | learned | GAM |
| `pure` | gravitational only | learned | GAM |
| `simplified` | dot product + λ · gravitational | vector norm | GAM |
| `l2` | −‖q − k‖² / √d | – | control: distance, no mass |
| `hybrid_nodist` | dot product + λ · G · mᵢmⱼ | learned | control: mass, no distance |
| `hybrid_fixedlam` | dot product + 0.5 · gravitational | learned | control: λ fixed |

## Setup

```bash
git clone https://github.com/ctheradyubele/gravitational-attention.git
cd gravitational-attention
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q tests          # 34 tests, all must pass
```

The study was run with PyTorch 2.x and `transformers` 4.46.3 on NVIDIA A100 GPUs (Google Colab Pro+).

## Data

No dataset is redistributed in this repository. All datasets are public:

| Dataset | Use | How to obtain |
| --- | --- | --- |
| WikiText-103 | continued pretraining (Phase A) | downloaded automatically: [`Salesforce/wikitext`](https://huggingface.co/datasets/Salesforce/wikitext) |
| GLUE (8 tasks) | accuracy (B1, B2, D) | downloaded automatically: [`nyu-mll/glue`](https://huggingface.co/datasets/nyu-mll/glue) |
| e-SNLI | plausibility and faithfulness (C) | download `esnli_train_1.csv`, `esnli_train_2.csv`, `esnli_dev.csv`, `esnli_test.csv` from [github.com/OanaMariaCamburu/e-SNLI](https://github.com/OanaMariaCamburu/e-SNLI) (folder `dataset/`) into `data/esnli/` |

## Reproducing the study

The study runs in phases. `run_matrix.py` prints one shell command per run; each run writes
`results.json` and is skipped if that file already exists, so a phase can be resumed after an
interruption.

| Phase | What | Runs |
| --- | --- | --- |
| A | continued MLM pretraining of 7 variants on WikiText-103 (10k steps) | 7 |
| B1 | learning-rate selection on a held-out 10% split (seed 0) | 144 |
| B2 | GLUE fine-tuning, 8 tasks × 6 variants × 5 seeds, plus SST-2 ablations | 250 |
| C | e-SNLI fine-tuning and interpretability evaluation, 4 variants × 5 seeds | 40 |
| D | MNLI with 2,500 and 8,500 training examples, 3 variants × 5 seeds | 30 |
| F | time and memory benchmark of one attention layer | 2 |

```bash
for phase in A F B1 B2 C D; do
  python scripts/run_matrix.py $phase > cmds_$phase.txt
  while read -r c; do eval "$c" || echo "FAILED: $c" >> failed.txt; done < cmds_$phase.txt
done

python analysis/aggregate.py --runs runs --out results
python analysis/mass_analysis.py --runs runs --esnli_dir data/esnli --out results
python figures/make_figures.py
```

Model outputs are written to `runs/`; to write them elsewhere (e.g. Google Drive), copy
`configs/protocol.yaml`, change `runs_dir`, and point `GAM_PROTOCOL` at the copy.
The full study takes roughly 60–70 GPU-hours on one A100. The notebook in `notebooks/` shows the
exact sequence used on Colab (with outputs written to Google Drive).

## Results files

| File | Contents |
| --- | --- |
| `hypothesis_tests.json` | primary hypotheses H1–H3 (one-sided, Holm-adjusted) and secondary tests |
| `glue.csv`, `glue_rows.tex` | GLUE development-set results per task and variant (mean ± SD over 5 seeds) |
| `esnli.csv` | plausibility (AUPRC, token F1, correlations) and faithfulness (comprehensiveness, sufficiency) on e-SNLI |
| `data_size.csv` | MNLI accuracy with 2,500 and 8,500 training examples |
| `ablation_sst2.csv` | SST-2 accuracy of all variants including the ablations |
| `lambda_by_layer.csv` | learned mixing weight λ per layer and task |
| `adaptation.csv` | validation loss after continued pretraining |
| `benchmark_amp.csv`, `benchmark_fp32.csv` | time and memory of one attention layer |
| `mass_analysis.json`, `energy_sst2.csv`, `figure3_deltas.csv` | analyses of the learned masses and figure data |

Model checkpoints are not included (about 440 MB each).

## Citation

If you use this code, please cite the paper (see `CITATION.cff`).

## License

MIT (see `LICENSE`).
