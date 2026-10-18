# Pre-registered decision rules for the additional experiments

Written 17 Oct 2026, after the Stage 1 test-split diagnostics (Δ/DE metrics and strata on already-saved test outputs). This was before any Stage 2 run, validation or test.

## Rules for all experiments
- **Selection split:** Norman validation (48 perturbations; strata `val_s`, `val_pu`, `val_p` from `splits.json`) and PBMC validation (donor P6, 19 cell types).
- **Unit and statistic:** the unit is a perturbation or a cell type. Every test is a 95% paired bootstrap over units (2000 resamples, seed 0).
- **Promotion to test:** a configuration goes to the test split only if, on validation, the paired CI against its comparator **excludes zero in its favor on at least one primary metric**, and no primary metric is worse with a CI that excludes zero.
- **Test runs:** every promoted configuration is run on test once per seed (0, 1, 2). At most 3 new CAD configurations are promoted.
- **Control cells:** every new run uses the matched, unit-seeded draws (`cell_eval.eval_units`, crc32 of the unit name).
- **Δ metrics are pseudobulk** (`de_eval --source qbar`):
  - `pearson_delta_pert` is Pearson over all panel genes of the predicted against the real change, both relative to the training perturbed centroid;
  - `pearson_de20` is Pearson on the top-20 Welch-t DE genes, real against held-out controls.

## Experiments, comparators and primary metrics

| Id | Change | Comparator (same backbone, same seed) | Primary metrics |
|---|---|---|---|
| B1a | control residual log b, **no** control text | current CAD with control residual | cell NLL (descriptive ablation; not promoted unless it beats the comparator) |
| B1b | control text, base `none` | current CAD with control residual | cell NLL (descriptive) |
| B2 | contrast control text: top-48 over-represented genes, log2(p0/p̄ctrl) | current CAD with control residual (mass text) | cell NLL; MMD (DM); `pearson_delta_pert` |
| B3 | perturbation prompt + NCBI RefSeq summaries (MyGene.info) | the same model with opaque gene IDs, and with descriptions of other genes | NLL and `pearson_delta_pert` on the **unseen-singles** stratum |
| B4 | autoregressive C2S on C2S-Scale-Gemma-2-2B with LoRA r16, budget matched to CAD | CAD (Gemma), no control cell | NLL (log-rank reconstruction), MMD, `pearson_delta_pert`, wall-clock |
| B5 | UMI-weighted soft cross-entropy | CAD with equal-weight loss | NLL; `pearson_delta_pert`; `pearson_de20` |
| B6 | NB head: q(c) plus per-gene dispersion θ_g(c), NB likelihood loss | CAD (softmax) | NLL; NB NLL; `pearson_delta_pert`; `pearson_de20` |
| B7 | low-rank logistic-normal head with an amortized encoder | CAD (softmax) | cell NLL (mixture over draws); MMD (multinomial sampler); AUC |
| Z | zero-shot cell types, Cell Ontology text prompt | nearest-ontology-type mean; global mean | NLL; MMD; type recovery |

## Fixed in advance
- No metric, stratum or comparator is added after seeing validation results without being labeled post hoc in the paper.
- **Unseen singles:** additive equals the perturbed mean on this stratum, so CAD-versus-additive on unseen singles is not interpreted as gene knowledge. The control mean also scores high on `pearson_delta_pert` there (weak perturbations sit close to control).
