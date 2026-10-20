# Pre-registered rules for the rev3 experiments

Written 20 Oct 2026, before any E1 run on validation or test.

## Context
- **Answers:** the strict RECOMB review (19 Oct).
- **Core claim of the paper:** a decision language model generates a cell as a distribution over a fixed gene panel in one forward pass.
- **What rev3 adds:**
  - **(E1):** tests whether the language model is the source of the likelihood gain;
  - **(E2/E3/E5):** biology and representation re-analyses of saved outputs.

## E1: does the language model matter? (new training runs)
**Comparator:** CAD without a control cell (softmax head, LoRA r16), same backbone and seed.

| Id | Configuration | Tag (val / test) |
|---|---|---|
| E1a | Frozen backbone (`--lora-r 0`, `--base mean --no-ctrl`) with an MLP head (`--head-arch mlp`: LayerNorm, Linear 1024, GELU, Linear 2048, zero-init output). Otherwise identical (cells, prompt, epochs 1, batch 16, head lr 1e-3). | `v3_q_frozen_mlp` / `t3_g_frozen_mlp_s{0,1,2}` |
| E1a' | Frozen backbone with the linear head (`--head-arch linear`); validation only, descriptive. | `v3_q_frozen_lin` |
| E1b | No language model (`src/train_nolm.py`): sum of learned gene embeddings (128), MLP 512, softmax, with the same fixed training-mean log-prior as CAD (`--base mean`). Same 20,000 training cells, 5 epochs, batch 64, lr 1e-3, unseen gene = zero embedding. | `v3_nolm_s0` / `t3_nolm_s{0,1,2}` |

**Validation and test:**
- Validation: Qwen3.5-0.8B, seed 0, comparator `v2_q_noctrl`.
- Test: Gemma (C2S-Scale-Gemma-2-2B), seeds 0, 1, 2, comparator `t2_g_noctrl_s{seed}` where it exists, otherwise `t_g_noctrl_s{seed}`.
- E1 is a **control, not a promoted configuration**. E1a and E1b therefore go to test whatever the validation result; validation only checks that the runs work (sanity: better than the perturbed mean overall).

**Primary metrics.** Paired 95% bootstrap over perturbations (2000 resamples, seed 0):
1. pooled NLL, overall and per stratum (single, one unseen, both seen);
2. target-excluded r_DE (`de_eval --source qbar --exclude-targets`, `pearson_de20`).

**Secondary metrics:**
- MMD and AUC (Dirichlet-multinomial sampler);
- target-excluded sign accuracy on the top-20 downstream DE genes;
- trainable parameters;
- decode time per prompt.

**Decision rule, fixed now:**
- **CAD better on a primary metric** (CI excludes 0 in CAD's favor) on test seed 0: we report a measured benefit of the fine-tuned language model for that metric and stratum only.
- **E1b or E1a ties or beats CAD on both primary metrics:** the paper states that the likelihood gain does not require the language model. The language model's role is then the **text interface**:
  - gene and type names;
  - zero-shot cell types from names;
  - one prompt format for every state.
- **Either way:** no claim that language-model *knowledge* drives the gain.

## E2: target-excluded downstream biology (re-analysis, post hoc)
**Labeling.** Computed after the test results were seen, and labeled post hoc in the paper.

**Method.** `src/de_eval.py --source qbar --exclude-targets`:
- the perturbed gene(s) are removed from the DE ranking, from r_Δ and from r_DE;
- the targets are scored separately (`target_abs_err`, `target_sign`).

**Biology-aware baseline.** `src/run_ti_baseline.py` (`cb_additive_ti`): partial additive, with every target gene not seen alone in training multiplied by one scalar fitted on training singles.

**Sanity check.** Without `--exclude-targets`, the additive r_DE must reproduce `auto_rev2/test_all_de_qbar`.

## E3: learned rank-to-frequency decoders (re-analysis, post hoc)
**Decoders.** In `src/repr_ceiling.py`, fitted on training cells only:
- E[frequency | rank position];
- the same given the library-size quintile;
- E[frequency | gene, rank-position bin].

**Reporting.**
- Each decoder's share of the matched headroom is reported next to the C2S log-rank fit.
- The 38% stays attributed to the C2S reconstruction protocol.

## E5-lite: co-expression (re-analysis, post hoc, supplement)
- Off-diagonal gene–gene correlation error, real vs generated, on 100 training-selected high-variance genes.
- Eigenspectrum ratio.
- Main methods only.
