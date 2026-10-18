# Held-out units

The units over which every paired bootstrap in the paper is computed.

| File | Units | Task |
|---|---|---|
| `norman_test_perturbations.txt` | 76 perturbations | Norman K562, test split |
| `norman_val_perturbations.txt` | 48 perturbations | Norman K562, validation split |
| `pbmc_test_cell_types.txt` | 24 cell types (at least 30 cells) | PBMC, test donors P7 and P8 |
| `pbmc_val_cell_types.txt` | 19 cell types (at least 30 cells) | PBMC, validation donor P6 |

The full cell-level split (gene-level Norman split with its leakage assertions, PBMC donor split P1-P5 / P6 / P7-P8) is regenerated deterministically by `src/preprocess.py` and `src/gen_data.py`, which write `splits.json` next to the processed data.
