#!/usr/bin/env python
"""
Build the two manuscript conditioning-swap figures from the cross-metric matrices
written by ``run_conditioning_swap.py``:

    Figure 6        : MMD swap heatmap,        BRCA (left) + AML (right)   [MAIN]
    Suppl. Fig. 8   : E-distance swap heatmap, BRCA (left) + AML (right)   [SUPPL]

Each panel is a real-sample x generated-population matrix (lower = closer). Rows and
columns share one ordering (train samples first, then held-out test, each alphabetised)
so the matched pairs lie on the diagonal, which is outlined. A dividing line separates
the train and test blocks.

Inputs (defaults assume you ran, from the project root):
    run_conditioning_swap.py --dataset brca --metric mmd   --vae <alltrain> --generated-glob '<alltrain>' --out output/conditioning_swap_brca_alltrain
    run_conditioning_swap.py --dataset brca --metric edist --vae <alltrain> --generated-glob '<alltrain>' --out output/conditioning_swap_brca_alltrain
    run_conditioning_swap.py --dataset aml  --metric edist                                                  --out output/conditioning_swap_aml
    # AML MMD reuses the pre-existing output/conditioning_swap_aml/cross_mmd_matrix.csv (unchanged)

Outputs (written to --out-dir, default: this folder):
    Figure6_mmd_swap_brca_aml.png / .pdf
    SupplFigure8_edist_swap_brca_aml.png / .pdf
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle


def read_ids(path):
    p = Path(path)
    if not p.exists():
        return []
    return [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]


def ordered_ids(matrix, train_ids, test_ids):
    """train (sorted) then test (sorted), restricted to ids present in the matrix."""
    present = list(matrix.index)
    train = sorted([s for s in present if s in set(train_ids)])
    test = sorted([s for s in present if s in set(test_ids)])
    rest = sorted([s for s in present if s not in set(train_ids) | set(test_ids)])
    order = train + test + rest
    n_train = len(train)
    return order, n_train


def load_matrix(csv_path):
    m = pd.read_csv(csv_path, index_col=0)
    m.index = m.index.astype(str)
    m.columns = m.columns.astype(str)
    return m


def draw_panel(ax, matrix, train_ids, test_ids, title, metric_label):
    order, n_train = ordered_ids(matrix, train_ids, test_ids)
    m = matrix.loc[order, order].astype(float)
    values = m.to_numpy()

    # Robust color scaling: clip at the 98th percentile so a few large mismatched
    # cells do not wash out the matched structure.
    vmax = np.nanpercentile(values, 98)
    im = ax.imshow(values, aspect="auto", cmap="viridis_r", vmin=0.0, vmax=vmax)

    n = len(order)
    # outline matched (diagonal) cells
    for i in range(n):
        ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                               edgecolor="white", linewidth=1.0))
    # train/test divider
    if 0 < n_train < n:
        ax.axhline(n_train - 0.5, color="red", lw=1.2, ls="--")
        ax.axvline(n_train - 0.5, color="red", lw=1.2, ls="--")

    ax.set_xticks(range(n)); ax.set_xticklabels(order, rotation=90, fontsize=5)
    ax.set_yticks(range(n)); ax.set_yticklabels(order, fontsize=5)
    ax.set_xlabel("generated population (this sample's pseudobulk)", fontsize=8)
    ax.set_ylabel("real sample", fontsize=8)
    ax.set_title(title, fontsize=10)
    cbar = plt.colorbar(im, ax=ax, shrink=0.75)
    cbar.set_label(f"latent-space {metric_label}", fontsize=8)
    cbar.ax.tick_params(labelsize=6)
    return n_train, n


def build_figure(brca_csv, aml_csv, metric_label, out_stem,
                 brca_train, brca_test, aml_train, aml_test, fig_tag):
    brca = load_matrix(brca_csv)
    aml = load_matrix(aml_csv)
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    draw_panel(axes[0], brca, brca_train, brca_test,
               "BRCA (Wu 2021)", metric_label)
    draw_panel(axes[1], aml, aml_train, aml_test,
               "AML (van Galen 2019)", metric_label)
    fig.suptitle(f"{fig_tag}  Conditioning specificity ({metric_label}): "
                 f"real sample vs generated population (matched pairs outlined; "
                 f"red dashes separate train / held-out test)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_stem}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_stem}.png / .pdf")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--brca-dir", default=str(root / "output/conditioning_swap_brca_alltrain"))
    ap.add_argument("--aml-dir", default=str(root / "output/conditioning_swap_aml"))
    ap.add_argument("--brca-splits", default=str(root / "output/sample_splits/brca2021_manual"))
    ap.add_argument("--aml-splits", default=str(root / "output/sample_splits/aml_nocl"))
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent))
    args = ap.parse_args()

    brca_dir, aml_dir = Path(args.brca_dir), Path(args.aml_dir)
    bt = read_ids(Path(args.brca_splits) / "train_samples.txt")
    bte = read_ids(Path(args.brca_splits) / "test_samples.txt")
    at = read_ids(Path(args.aml_splits) / "train_samples.txt")
    ate = read_ids(Path(args.aml_splits) / "test_samples.txt")
    out = Path(args.out_dir)

    # Figure 6 — MMD (AML MMD is the pre-existing, unchanged matrix)
    build_figure(brca_dir / "cross_mmd_matrix.csv",
                 aml_dir / "cross_mmd_matrix.csv",
                 "MMD", str(out / "Figure6_mmd_swap_brca_aml"),
                 bt, bte, at, ate, "Figure 6.")

    # Suppl Fig 8 — E-distance
    build_figure(brca_dir / "cross_mmd_matrix_edist.csv",
                 aml_dir / "cross_mmd_matrix_edist.csv",
                 "E-distance", str(out / "SupplFigure8_edist_swap_brca_aml"),
                 bt, bte, at, ate, "Supplementary Figure 8.")


if __name__ == "__main__":
    main()
