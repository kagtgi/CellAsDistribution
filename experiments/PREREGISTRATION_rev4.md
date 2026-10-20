# Rules for the rev4 analyses (written 21 Oct 2026, before any rev4 run)

These analyses answer a second simulated RECOMB review. They are **controls and diagnostics, not candidate improvements**:
- nothing is promoted;
- every analysis is run once (test seed 0 unless stated);
- every analysis is labeled "added after the test results" in the paper.

**Units and statistics.** Units are perturbations (Norman) or cell types (PBMC). Intervals are paired 95% bootstrap intervals over units (2000 resamples, seed 0).

| Id | Question | Configuration | Comparator | Metrics |
|---|---|---|---|---|
| R4-1 | Does a cell-pretrained backbone read unseen cell-type names better than Qwen? | Zero-shot cell types with C2S-Scale-Gemma-2-2B. 5 leave-types-out folds × {name, name + Cell Ontology definition, opaque id}. Same settings as the Qwen runs (`t2_zs_g_*`). | Name-overlap type mean; Qwen rows | pooled NLL, type accuracy |
| R4-2 | Do biological gene embeddings carry information to genes never perturbed in training? | No-LM decoder with a frozen gene embedding: GenePT ada-002 embeddings of NCBI summaries (Zenodo 10833191), or co-expression PCA loadings from training cells. Test seeds 0–2. | No-LM decoder with learned embeddings (unseen gene = zero); CAD | pooled NLL per stratum (primary: unseen singles); downstream r_DE |
| R4-3 | Target induction and unseen singles when every perturbed gene is in the panel | Add-on panel `norman_tp`: same splits (verified cell by cell), all 105 perturbed genes forced into the 2048 genes. On test, seed 0: additive, additive + target induction, no-LM (learned, GenePT, co-expression), CAD (Gemma, no control cell). | Additive | target error, target sign, downstream r_DE, NLL per stratum |
| R4-4 | Does an explicit target prior remove CAD's under-prediction of target induction? | Post hoc on the saved CAD q (main panel). Every target seen alone in training gets its own training-single frequency; unseen targets get the global factor (`run_ti_baseline --mode all`). | CAD; additive | target error; NLL; downstream r_DE |

**Reporting rules, fixed now:**
- R4-1 reports whatever Gemma does. If it does not beat name overlap with an interval excluding zero, the paper keeps "composes the words of names".
- R4-2 counts as transfer only if no-LM + GenePT beats the learned no-LM decoder on unseen singles with an interval excluding zero.
- R4-3 does not replace any main-panel number; it is reported as an additional analysis.
- R4-4 is a diagnostic of the shrinkage mechanism, not a new method.
