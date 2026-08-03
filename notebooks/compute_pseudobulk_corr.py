#!/usr/bin/env python
"""Recover the Suppl-7 per-sample pseudobulk correlation for the BRCA all-train run,
using the exact methodology of the multi-sample notebook (cell 7):

    proxy_pseudobulk = log1p(mean over cells of expm1(decoded_gene_expr))
    r = pearson( input_pseudobulk , generated_proxy_pseudobulk )   # per sample

Prints train/test mean and range so the manuscript numbers can be updated verbatim.
"""
import glob
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys; sys.path.insert(0, str(PROJECT_ROOT))
from guided_diffusion.cell_datasets_loader import decode_latents_with_vae

VAE = str(PROJECT_ROOT / "output/checkpoint/AE/brca2021_VAE_alltrain/model_seed=1234_step=199999.pt")
GLOB = str(PROJECT_ROOT / "output/simulated_samples/brca2021_pseudobulk_1M_alltrain_*.npz")


def scalar(v):
    v = np.asarray(v)
    return v.item() if v.shape == () else v.reshape(-1)[0].item()


def proxy_pseudobulk(cell_matrix):
    m = np.asarray(cell_matrix, dtype=np.float32)
    linear = np.expm1(m)
    np.maximum(linear, 0.0, out=linear)
    linear = np.nan_to_num(linear, nan=0.0, posinf=0.0, neginf=0.0)
    return np.log1p(linear.mean(axis=0, dtype=np.float64)).astype(np.float32)


rows = []
for p in sorted(glob.glob(GLOB)):
    stem = Path(p).stem
    split = "test" if "_test_" in stem else ("train" if "_train_" in stem else "unknown")
    with np.load(p, allow_pickle=True) as z:
        sid = str(scalar(z["source_sample_id"]))
        input_pb = np.asarray(z["input_pseudobulk"], dtype=np.float32)
        latents = np.asarray(z["cell_gen"], dtype=np.float32)
    decoded = decode_latents_with_vae(latents, vae_path=VAE, hidden_dim=128)
    gen_proxy = proxy_pseudobulk(decoded)
    r = float(np.corrcoef(input_pb, gen_proxy)[0, 1])
    rows.append((sid, split, r))
    print(f"{sid:10s} {split:5s} r={r:.4f}")

import statistics as st
for split in ["train", "test"]:
    rs = [r for _, s, r in rows if s == split]
    if rs:
        print(f"\n{split}: n={len(rs)} mean={sum(rs)/len(rs):.4f} "
              f"min={min(rs):.4f} max={max(rs):.4f}")
