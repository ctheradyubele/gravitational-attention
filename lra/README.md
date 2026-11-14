# Phase G: Long Range Arena (LRA)

LRA is trained **from scratch** with small models under the benchmark's fixed protocol, so it is run
separately from the BERT-based phases. Do not write a new LRA pipeline: data preprocessing and the
training protocol have many details that change the results. Use an established PyTorch LRA codebase
that already supports swapping the attention module (for example, the LRA code released with
Nyströmformer, Skyformer or similar work). Record its repository URL and commit in `DEVIATIONS.md`.

## Integration

`gam/attention.py` provides a framework-agnostic module:

```python
from gam.attention import GAMConfig, GravitationalAttention, VARIANTS
attn = GravitationalAttention(dim=d_model, num_heads=n_heads, gcfg=VARIANTS["hybrid"], dropout=p)
y = attn(x, key_padding_mask=mask)   # x: [B, T, D]; mask: [B, T] bool, True = real token
```

Replace the codebase's softmax-attention class with this module for the `standard`, `pure` and
`hybrid` variants. Run `standard` through this same module as well (not the codebase's own class),
so all three variants share identical code. Before training, call `calibrate_g` on one batch for the
GAM variants, exactly as in Phase A (wrap the encoder so `gam_modules` can find the layers, or apply
the same median/std rule per layer by hand).

## Validity checks before the main LRA runs

1. **Baseline reproduction.** With `standard`, train one seed on ListOps and Text. Accuracy must be
   within about 1 point of the codebase's own softmax-attention result with the same settings. If it
   is not, the integration is wrong; fix it before running GAM.
2. **Memory.** At 4K tokens, check that a full-matrix model fits; LRA Text and Retrieval need 4K. If
   not, reduce the batch size and increase gradient accumulation identically for all variants.

## Design

| Setting | Value |
| --- | --- |
| Tasks | ListOps (2K), Text (4K), Retrieval (4K per document), Image (1K), Pathfinder (1K) |
| Variants | standard, pure, hybrid (all through `GravitationalAttention`) |
| Seeds | 3 per task and variant (45 runs) |
| Hyperparameters | the codebase's per-task defaults, identical for all variants; `gam_lr` = 1e-3 for λ, G and the mass head |
| Model selection | the codebase's validation split; test evaluated once |
| Report | test accuracy mean ± SD over seeds; time and peak memory per variant |

Write each run's output as `runs/G/<task>/<variant>/s<seed>/results.json` containing at least
`{"task", "variant", "seed", "test_acc", "train_time_s", "peak_mem_gb", "env"}` so the analysis
script can include it.
