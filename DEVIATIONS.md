# Deviations from the protocol and infrastructure events

The protocol (`configs/protocol.yaml`) was fixed before the main experiments. The events below are
also reported in Appendix C of the paper. None changed a model, a hyperparameter or a data split.

| Item | Description and effect |
| --- | --- |
| Lost continued-pretraining outputs | The first Phase A runs were written to temporary storage and lost before any fine-tuning or analysis. All variants were re-run from the start with the protocol settings. |
| GPU model | B2 runs 144–167 (QQP L2 attention seeds 4–5 to MNLI Simplified GAM seed 2), e-SNLI Simplified GAM seeds 4–5 (training and evaluation) and all Phase D runs used an A100 40 GB; all other runs used an A100 80 GB. Same architecture, code and precision. All timing results come from the 80 GB A100. |
| Repeated timing run | The Phase F benchmark was run again in a later session on the same GPU model; the later run is reported. |
| Interrupted runs | Runs interrupted by the end of a cloud session were discarded and restarted from scratch; no partial run contributes to the results. |
| Direction of primary tests | `aggregate.py` first passed two-sided bootstrap p-values for H1 and H2 to the Holm procedure, which would have counted a significant difference in the wrong direction as support. It was corrected to the one-sided tests specified in the protocol (`stats.paired_bootstrap` now returns `p_one_sided_greater`) after the results were known. The correction changed H1 from apparently significant to not supported; H2 and H3 were unaffected. |
| Long Range Arena | The optional Long Range Arena experiment was not run. |
