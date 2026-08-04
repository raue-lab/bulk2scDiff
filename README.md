# bulk2scDiff: Pseudobulk-Conditioned Single-Cell Generation

bulk2scDiff is a pseudobulk-conditioned latent diffusion model for scRNA-seq generation. Given one sample-level pseudobulk vector, it generates a population of single-cell profiles whose aggregate expression is compatible with that condition.

![Model architecture](docs/model_workflow.png)

The pipeline:

- fine-tune a VAE (latent autoencoder) on single cells
- train a latent diffusion model conditioned on each cell's sample-level pseudobulk
- generate one synthetic population per real sample pseudobulk (train and held-out test)
- evaluate the outputs with the notebooks in [notebooks/](notebooks/)

This repository ships the single canonical, manuscript-reported configuration for each of the two datasets — **AML** (van Galen 2019) and **BRCA2021** (Sunny Wu 2021) — both trained for 1,000,000 diffusion steps:

- **AML**: the two immortalized cell-line samples (`MUTZ3`, `OCI.AML3`) are excluded from both VAE and diffusion training.
- **BRCA2021**: the VAE is fine-tuned on all 26 samples, **including the 5 held-out test samples**; only the diffusion backbone is restricted to the 21 training samples. A pilot ablation found this improves test-cell VAE reconstruction fidelity substantially (mean per-gene Pearson r: 0.166 → 0.253) versus excluding test samples from VAE training. In both datasets, held-out status is always enforced at the diffusion-training stage.

## Environment

```
torch                     1.13.0
numpy                     1.23.4
anndata                   0.8.0
scanpy                    1.9.1
scikit-learn              1.2.2
scipy                     1.13.1
matplotlib                3.6.0
blobfile                  2.0.0
pandas                    1.5.1
mpi4py                    3.1.4
```

See [requirements.txt](requirements.txt). Install `torch` first with the CUDA build matching your system, then `pip install -r requirements.txt`.

## Workflow

Training data is expected in `.h5ad` format, keyed by `SampleID`. The loader in [guided_diffusion/cell_datasets_loader.py](guided_diffusion/cell_datasets_loader.py) performs light preprocessing:

- filter genes detected in fewer than 3 cells
- filter cells expressing fewer than 10 genes
- normalize each retained cell to a fixed library size (`1e4`) using its pre-gene-filter total count
- apply `log1p`

For pseudobulk conditioning, the loader aggregates raw counts per sample in the retained gene space, then applies a separate sample-level library-size normalization plus `log1p`. Because that normalization uses pre-gene-filter library sizes, it is numerically equivalent on the retained genes to aggregating the full raw matrix first and only then subsetting to the retained genes. If your data is preprocessed differently, adjust the loader before training.

## One-command runs (recommended)

Each dataset has an idempotent end-to-end driver that creates splits, fine-tunes the VAE, trains the diffusion backbone on the training samples only (with periodic held-out MMD evaluation), and generates one `.npz` per sample for both splits:

```
# AML — 1M steps, cell lines excluded
bash deploy_aml.sh

# BRCA2021 — 1M steps, VAE trained on all samples (incl. held-out test)
bash deploy_brca.sh
```

Useful env overrides for the drivers: `FORCE_REGENERATE_SAMPLES=1`, `NUM_SAMPLES_OVERRIDE=N`.

Outputs land under `output/`: checkpoints in `output/checkpoint/`, logs in `output/logs/`, generated samples in `output/simulated_samples/`.

## Manual stages

The drivers wrap these steps, which can also be run directly:

1. **Create sample splits** — `split_aml.py` (random 80/20) or `split_brca.py` (fixed, subtype-balanced); both write `train_samples.txt`, `test_samples.txt`, and `sample_summary.tsv`.
2. **Train the VAE** — `python VAE/VAE_train.py --data_dir <h5ad> --num_genes <retained_genes> --save_dir <dir> --max_steps 200000 --state_dict /share/models/SCimilarity/annotation_model_v1 --sample_key SampleID`. Use `--include_sample_ids_path`/`--exclude_sample_ids_path` to restrict which samples the VAE sees (the AML driver excludes cell lines only; the BRCA2021 driver passes neither flag, so the VAE sees every sample).
3. **Train the diffusion model** — `python train.py --data_dir <h5ad> --vae_path <vae.pt> --model_name <name> --save_dir output/checkpoint/backbone --lr_anneal_steps 1000000 --sample_key SampleID --cond_pseudobulk True --cond_embed_dim 128 --include_sample_ids_path <train_samples.txt>`. Add `--mmd_eval_interval 10000 --mmd_eval_validation_sample_ids_path <test_samples.txt>` for periodic held-out MMD evaluation.
4. **Generate samples** — `python sample.py --data_dir <h5ad> --model_path <model.pt> --sample_dir <prefix> --sample_id <SampleID> --sample_key SampleID --cond_pseudobulk True --cond_embed_dim 128`.

> Note: `--cond_embed_dim` must match between training and sampling (128 in both drivers; the code default is 256).

Each generated `.npz` stores: the generated latent cells (`cell_gen`), the source `SampleID`, the conditioning pseudobulk vector (`input_pseudobulk`), the real cell count, and the fixed reduction mode (`sum`).

## Evaluation

The notebooks in [notebooks/](notebooks/) cover the full evaluation suite (see §10 of the [walkthrough](docs/pseudobulk_conditioned_scdiffusion.md) for details). AML notebooks carry their executed outputs inline; the BRCA2021 notebooks were executed headlessly and ship as code only — rerun them to see the figures:

- **Training diagnostics** — `{aml,brca}_train_progress.ipynb` (loss, gradients, and train-vs-held-out MMD over training).
- **Held-out conditioning audit (primary metric)** — `{aml,brca}_cond_audit_{mmd,edist}.ipynb` (per-sample latent-space MMD / E-distance vs same-sample and other-sample baselines).
- **Qualitative UMAPs** — `{aml,brca}_multi_umap.ipynb` (pooled + per-sample, in gene and latent space).
- **Marker-gene biology** — `{aml,brca}_marker_{train,test}.ipynb` and `{aml,brca}_global_umap_{train,test}.ipynb` (cell-type / subtype marker fidelity).

The held-out MMD/E-distance audit is the recommended quantitative evaluation; the UMAP and marker notebooks are supporting qualitative checks.

## Data

The AML (van Galen 2019) and BRCA2021 (Sunny Wu 2021) datasets used to train and evaluate these models, and the resulting trained model checkpoints, are not redistributed in this repository. See the manuscript for data and materials availability.

## License

MIT — see [LICENSE](LICENSE). This project includes code adapted from [scDiffusion](https://github.com/EperLuo/scDiffusion) (MIT License, Copyright (c) 2023 Erpai Luo).
