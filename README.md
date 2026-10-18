# CellAsDistribution

Code for *CellAsDistribution: Decision Language Models Decode Cells as Gene-Level Distributions* (RECOMB 2027 submission).

A cell over a fixed gene panel is treated as a probability distribution (its normalized expression vector). A decision language model decodes that distribution in one forward pass from a text state (a cell type, or a perturbation optionally with a control cell). The model is a language model with LoRA adapters and a linear softmax head over the $G=2048$ panel genes, plus an optional fixed log-prior on the logits (none, the training mean, or the control cell itself: the *control residual*). Cells are sampled from the decoded distribution with a shared sampler, and every method is scored by the same multinomial likelihood of held-out counts.

Use cases:
* **Conditional generation.** Human PBMC (Hao et al. 2021, CELLxGENE Census): donors P1–P5 train, P6 validation, P7–P8 test.
* **Zero-shot cell types.** Five leave-cell-types-out folds; the prompt holds the type name and its Cell Ontology definition.
* **Perturbation response.** Norman et al. K562 CRISPRa (scPerturb), with a leakage-controlled gene-level split. Results are reported per stratum: unseen single genes, pairs with one unseen gene, and pairs of two seen genes.

## Layout

| Path | Purpose |
|---|---|
| `src/gen_data.py` | PBMC from CELLxGENE Census and the donor split |
| `src/preprocess.py` | Norman count check, train-only 2048-gene panel (dispersion of log-normalized expression), gene-level leakage-safe split with assertions |
| `src/cell_dist.py` | target transformations, control-cell serializations (mass levels, rank sentence, contrast to the average control), shared multinomial / Dirichlet-multinomial sampler |
| `src/train_cell.py` | the decision language model. Options: `--model` (any causal LM; add `--attn eager --grad-ckpt` for Gemma-2), `--base none/mean/ctrl`, `--no-ctrl`, `--ctrl-text on/off`, `--ctrl-format mass/rank/contrast`, `--pert-desc`, `--weight cell/umi`, `--head softmax/nb/lowrank`, `--zs-fold`, `--type-text name/def/opaque`, `--transform`, `--lora-r`, `--seed`, `--eval-split` |
| `src/heads.py` | negative-binomial head and low-rank logistic-normal head (trained as a VAE) |
| `src/cell_eval.py` | `World` (data, panel, split) and `eval_units`: pooled and per-cell NLL, MMD, AUC, gene-mean correlation, type recovery; control cells matched across methods (seeded by the unit name) |
| `src/de_eval.py` | perturbation-specific metrics (`--source qbar` is the pseudobulk definition used in the paper): r_Δ against the training perturbed centroid; r_DE, sign and precision on the top-20 Welch-t DE genes |
| `src/strata_results.py`, `src/cell_results.py` | paired-bootstrap comparisons, overall and per stratum of the split |
| `src/run_cell_baselines.py` | frequency-space baselines: global/type mean, control/perturbed mean, partial additive, ridge, control mixtures |
| `src/run_residual_baseline.py` | the control-cell residual without a language model (the additive shift applied to each control cell's log-prior) |
| `src/run_scvi.py`, `src/run_scvi_native.py` | scVI/scANVI (type-mean decoder frequency, and sampling from their own negative binomial) and the scGen-style latent shift; separate scvi-tools environment |
| `src/run_gears.py` | GEARS on the same split (`--train-ctrl-only`), output mapped back to frequencies; separate cell-gears environment |
| `src/c2s_gemma.py` | autoregressive Cell2Sentence on the same backbone and budget as CAD (`train`, `gen`, `score`) |
| `src/c2s_baseline.py` | Cell2Sentence fine-tune following the C2S perturbation tutorial (Pythia-1B) |
| `src/repr_ceiling.py`, `src/rank_r2.py` | likelihood ceilings of rank sentence, value bins, mass levels and frequency (matched estimator); the C2S R² of rank-to-expression |
| `src/zs_baselines.py`, `src/zs_results.py` | zero-shot baselines (global mean, nearest Cell Ontology type, ontology parents, name overlap), and pooling over folds |
| `src/fetch_cell_ontology.js` | fetches Cell Ontology ids, formal definitions and ancestors from EBI OLS into `splits/pbmc_cell_ontology.json` |
| `src/speed_bench.py`, `src/bio_checks.py`, `src/nb_eval.py` | decoding-speed benchmark, panel and control-text checks, negative-binomial evaluation |
| `src/train_flat.py`, `src/kfold.py`, `src/train_native.py`, `src/results.py` | population-level decoder (CAD-P), gene-level 5-fold, hierarchical-head ablation |
| `figures/` | `fig1_method.py` (Figure 1), `fig3_representation.py` (Figure 2), `make_tables_rev2.js` (main-text tables from the result JSONs), `demo_outputs.py` (appendix worked examples), `umap_compare.py`, `make_figures.py` (k-fold forest plot) |
| `experiments/` | the run queues. Validation decisions: `queue_rev2_val.sh`, `queue_rev2_zs.sh`, `summarize_rev2_val.sh`. Test runs and summaries: `queue_stage4.sh`, `summarize_stage4.sh`. Baselines: `queue_rev2_baselines.sh`. C2S-Gemma: `queue_c2sg.sh`, `queue_c2sg_regen.sh`. Decision rules written before the test runs: `PREREGISTRATION_rev2.md`, `DECISIONS_rev2.md`. Earlier queues: `queue_val.sh`, `queue_test.sh`. |
| `splits/` | held-out perturbations and cell types, zero-shot folds (`pbmc_zs_folds.json`), Cell Ontology terms (`pbmc_cell_ontology.json`) |
| `paper/` | LaTeX source, bibliography, generated tables (`auto*/`, `auto_rev2/tab_*.tex`), result JSONs (`auto_rev2/`), figures |

## Environment

- **Hardware and system:** one NVIDIA A100 40 GB, Ubuntu 22.04, Python 3.10.
- **Main environment:** `torch==2.8.0`, `transformers==5.2.0`, `peft`, `unsloth` (import it before `transformers`), and `anndata scanpy scikit-learn scipy pandas pyarrow matplotlib umap-learn`.
- **scvi-tools environment:** for `run_scvi*.py`.
- **cell-gears environment:** `cell-gears==0.1.2`, for `run_gears.py`. Two small local fixes were needed on this data:
  - control-expression indexing in `gears/gears.py`;
  - in `gears/utils.py`, the loss filter falls back to all genes when a condition is missing.
- **Variables:** set `CAD_DATA` (input h5ad directory) and `CAD_OUT` (output directory).
- **Figures:** the schematic figures use a small matplotlib layout kit that is not part of this repository; their PDF outputs are in `paper/figs/`. Figures use Google Sans if `fonts/GoogleSans-Regular.ttf` is present; the font is not redistributed here.

## Data

* scPerturb Zenodo record `10.5281/zenodo.13350497`: `NormanWeissman2019_filtered.h5ad` (K562 CRISPRa), raw integer counts (checked by `preprocess.py`).
* PBMC: CELLxGENE Census release 2025-11-08, dataset of Hao et al. (dataset, donor and cell-type ids in `src/gen_data.py`).

## Reproduce

```bash
# data and splits
python -m src.preprocess --h5ad $CAD_DATA/NormanWeissman2019_filtered.h5ad --name norman --pert-col perturbation
python -m src.gen_data
node src/fetch_cell_ontology.js > splits/pbmc_cell_ontology.json

# one model, by hand (Gemma backbone; seed and split as in the paper)
G="--model vandijklab/C2S-Scale-Gemma-2-2B --attn eager --grad-ckpt"
python -m src.train_cell --task pert --name norman --tag t2_g_noctrl_s0      $G --base mean --no-ctrl          --seed 0 --epochs 1 --max-train-cells 20000 --eval-split test
python -m src.train_cell --task pert --name norman --tag t2_g_ctrl_notext_s0 $G --base ctrl --ctrl-text off --seed 0 --epochs 1 --max-train-cells 20000 --eval-split test
python -m src.train_cell --task cond --name pbmc   --tag t2_g_cond_s0        $G                                --seed 0 --epochs 1 --max-train-cells 20000 --eval-split test

# everything in the paper
bash experiments/queue_rev2_baselines.sh         # baselines, matched controls, scVI/scANVI own sampling, GEARS
bash experiments/queue_c2sg.sh                   # C2S-Gemma training and validation generation
bash experiments/queue_rev2_val.sh norman; bash experiments/queue_rev2_val.sh pbmc   # validation decisions
bash experiments/queue_rev2_zs.sh                # zero-shot cell types, validation
bash experiments/queue_stage4.sh                 # test runs (seeds 0-2), automatic rules, summaries, demo figure
python -m src.repr_ceiling --task pert --name norman --split test; python -m src.rank_r2 --name norman --split test
python -m src.speed_bench --out $CAD_OUT/auto_rev2/speed_bench.json
node figures/make_tables_rev2.js                 # Tables 1-4 from paper/auto_rev2/*.json
```

**Protocol.**
- Decisions use the validation split or the gene-level folds.
- New configurations were promoted to the test split by the rules in `experiments/PREREGISTRATION_rev2.md`, which were written before the test runs. The decisions taken are in `experiments/DECISIONS_rev2.md`.
- The test split is evaluated once per configuration and seed.

## Notes

* An early Norman split leaked held-out single genes through training pairs (7/10 validation and 14/21 test single genes occurred inside training pairs). It was found by audit, and every reported result uses the corrected split.
* Baselines are in frequency space. The additive baseline is *partial additive*: an unseen gene contributes nothing.
* GEARS runs with its default settings (15 epochs, one seed), and its log-normalized output is mapped back to frequencies. CPA was not run.
* Main-text tables are generated by `figures/make_tables_rev2.js` and are never edited by hand. The captions and row labels of a few older appendix tables were edited for wording only; their numbers are unchanged.
* License: MIT for the code (`LICENSE`). Datasets and model checkpoints keep their own licenses.
