# Stage 2 decisions (validation, applied 18 Oct 2026 ~04:00 Vietnam time, before any Stage 4 test run)

Rules: `experiments/PREREGISTRATION_rev2.md`.
- **Backbone, seed, controls:** Qwen3.5-0.8B, seed 0, matched control cells.
- **Paired 95% bootstrap over units.** Norman validation has 48 perturbations; PBMC validation has 19 cell types.
- **Sources:** `~/cad_out/auto_rev2/norman_val_vs_{ctrl,noctrl,additive}.txt`, `norman_val_desc_vs_opaque.txt`, `norman_val_desc_real_vs_other.txt`, `pbmc_val_rev2.json`.

| Id | Comparator | Primary metrics: difference [CI] | Decision |
|---|---|---|---|
| B1a (log b residual, no control text) | v2_q_ctrl | cell NLL −0.039 [−0.046, −0.032] | beats comparator → test, seeds 0–2 |
| B1b (control text, no residual) | v2_q_ctrl | cell NLL +0.029 [+0.015, +0.048] | descriptive; test seed 0 (ablation trio) |
| B2 (contrast control text) | v2_q_ctrl | cell NLL −0.031 [−0.038, −0.024]; MMD −0.000 [−0.003, +0.002]; r_Δ −0.010 [−0.023, +0.002] | promoted → test, seeds 0–2 |
| B2 (mass levels, top-128) | v2_q_ctrl | not evaluated (OOM three times on the shared GPU) | dropped; stated |
| B3 (NCBI summaries) | other-gene summaries; opaque ids | unseen singles NLL vs other −0.007 [−0.037, +0.011] | not promoted |
| B3 control (opaque ids vs names) | v2_q_noctrl | opaque worse: all +0.050 [+0.034, +0.070]; unseen singles +0.026 [+0.009, +0.045] | evaluated on test (Gemma, seeds 0–2) as the test of the gene-knowledge claim |
| B5 (UMI-weighted loss) | v2_q_noctrl / v2_q_cond | NLL −0.003 [−0.011, +0.004] / −0.0013 [−0.0047, +0.0023]; r_Δ, r_DE tie | not promoted |
| B6 (NB head), Norman, no control | v2_q_noctrl | NLL +0.005 [−0.001, +0.012]; r_Δ +0.025 [+0.013, +0.037]; r_DE −0.001 [−0.010, +0.009] | promoted → test, seeds 0–2 |
| B6 (NB head), PBMC | v2_q_cond | NLL +0.008 [+0.001, +0.018] (worse) | not promoted |
| B7 (low-rank head), Norman, no control | v2_q_noctrl | cell NLL −0.094 [−0.106, −0.084]; MMD (mult) +0.001 [−0.005, +0.007]; AUC (mult) −0.010 [−0.025, +0.005] | promoted → test, seeds 0–2 |
| B7 (low-rank head), PBMC | v2_q_cond | cell NLL −0.032 [−0.045, −0.021]; MMD +0.016 [−0.012, +0.047]; AUC +0.012 [−0.001, +0.031] | promoted (same configuration as above) |
| B7 (low-rank head), Norman, control residual | v2_q_ctrl | cell NLL −0.035; MMD (mult) +0.008 [+0.002, +0.015] (worse) | not promoted |

**Promoted new configurations: three** (B2, B6, B7). B1a goes to test as part of the residual ablation, with 3 seeds, because it beat its comparator.

**Recorded side effects, to be reported:**
- **B7, pooled NLL:** the low-rank head worsens the pooled NLL (the mean over draws) and r_Δ. This was expected for pooled scoring of a mixture.
- **B7 on PBMC:** it lowers type recovery (−0.088 [−0.188, −0.014]); type recovery is not a primary metric.

**Automatic rules fixed now for the remaining decisions:**
- **C2S-Gemma mixing weight ε:** the value in {0.01, 0.03, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9} with the lowest mean validation pooled NLL.
- **Zero-shot (Stage 3, gate G3):** test runs happen only if CAD with Cell Ontology definitions beats the nearest-ontology-type mean on validation pooled NLL with a CI that excludes zero (pooled over the 5 folds). Otherwise zero-shot is reported on validation only.
