# bulk2scDiff

Pseudobulk-conditioned latent diffusion for scRNA-seq generation. Given one sample's pseudobulk vector, generates a population of single-cell profiles whose aggregate expression matches it.

## how it works

![Model architecture](docs/model_workflow.png)

A VAE first compresses single cells into a shared latent space. A diffusion model, conditioned on each cell's sample-level pseudobulk, learns to sample that latent space per sample. At generation time: hand the model a pseudobulk vector — real or held-out — and it decodes a plausible population of cells for it.

## datasets

| dataset  | source          | vae sees                            | diffusion sees        | steps      |
|----------|-----------------|--------------------------------------|------------------------|------------|
| AML      | van Galen 2019  | 24 samples (`MUTZ3`, `OCI.AML3` excluded) | same 24, 80/20 split   | 1,000,000  |
| BRCA2021 | Sunny Wu 2021   | all 26 samples                       | 21 training samples    | 1,000,000  |

Held-out status is always enforced at the diffusion stage. BRCA's VAE additionally trains on the 5 test samples — a pilot ablation found this raises test-cell reconstruction fidelity (mean per-gene Pearson r: 0.166 → 0.253) versus excluding them.

## run

```
bash deploy_aml.sh
bash deploy_brca.sh
```

| env override | effect |
|---|---|
| `DATA_DIR=/path/to/data.h5ad` | override default data path |
| `FORCE_REGENERATE_SAMPLES=1` | overwrite existing `.npz` files |
| `NUM_SAMPLES_OVERRIDE=N` | generate N cells instead of matching real count |

Both drivers are idempotent — each skips any stage whose checkpoint or sample file already exists.

| output | path |
|---|---|
| checkpoints | `output/checkpoint/` |
| logs | `output/logs/` |
| generated samples | `output/simulated_samples/` |

## manual stages

| # | step | script | key flags |
|---|------|--------|-----------|
| 1 | split | `split_aml.py` / `split_brca.py` | `--test_fraction` · `--exclude_sample_ids_path` / fixed subtype-balanced |
| 2 | VAE | `VAE/VAE_train.py` | `--num_genes` · `--state_dict` · `--include_sample_ids_path` / `--exclude_sample_ids_path` |
| 3 | diffusion | `train.py` | `--vae_path` · `--lr_anneal_steps 1000000` · `--cond_pseudobulk True` · `--cond_embed_dim 128` · `--mmd_eval_interval` |
| 4 | generate | `sample.py` | `--model_path` · `--sample_id` · `--cond_pseudobulk True` · `--cond_embed_dim 128` |

`--cond_embed_dim` must match between steps 3 and 4 (128 in both drivers; the code default is 256). Each generated `.npz` stores: the generated latent cells (`cell_gen`), the source `SampleID`, the conditioning pseudobulk (`input_pseudobulk`), the real cell count, and the fixed `sum` reduction mode.

## evaluation notebooks

| notebook | covers |
|---|---|
| `{aml,brca}_train_progress.ipynb` | loss, gradients, train-vs-held-out MMD |
| `{aml,brca}_cond_audit.ipynb` | MMD + E-distance heatmap, real vs. generated (primary metric) |
| `{aml,brca}_multi_umap.ipynb` | pooled + per-sample UMAP, gene and latent space |
| `{aml,brca}_global_umap_{train,test}.ipynb` | immune marker-gene overlays, split-highlighted (run `*_multi_umap.ipynb` once first — it caches the embedding these load) |

AML notebooks ship with executed outputs; BRCA2021 notebooks are code-only — rerun them to see the figures.

## data format

- input: `.h5ad`, keyed by `SampleID`
- gene filter: detected in ≥3 cells · cell filter: expresses ≥10 genes
- per-cell normalization: fixed library size `1e4` (pre-filter total count) + `log1p`
- pseudobulk: raw counts summed per sample in the retained gene space, then a separate sample-level library-size normalization + `log1p`

Loader: [guided_diffusion/cell_datasets_loader.py](guided_diffusion/cell_datasets_loader.py). Adjust it if your data is preprocessed differently.

## environment

| package | version |
|---|---|
| torch | 1.13.0 |
| numpy | 1.23.4 |
| anndata | 0.8.0 |
| scanpy | 1.9.1 |
| scikit-learn | 1.2.2 |
| scipy | 1.13.1 |
| matplotlib | 3.6.0 |
| blobfile | 2.0.0 |
| pandas | 1.5.1 |
| mpi4py | 3.1.4 |

Full pins in [requirements.txt](requirements.txt). Install `torch` first with the CUDA build matching your system, then `pip install -r requirements.txt`.

## data & license

The AML and BRCA2021 datasets, and the resulting trained checkpoints, are not redistributed in this repository. See the manuscript for data and materials availability.

MIT — see [LICENSE](LICENSE). This project includes code adapted from [scDiffusion](https://github.com/EperLuo/scDiffusion) (MIT License, Copyright (c) 2023 Erpai Luo).
