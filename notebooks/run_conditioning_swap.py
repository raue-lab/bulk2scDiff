#!/usr/bin/env python
"""
Conditioning-specificity audit (mismatched-pseudobulk control), command-line version.

Run this from the bulk2scDiff project root over SSH. It rebuilds everything itself:
loads the preprocessed AnnData, loads the autoencoder, reads the generated .npz
files, encodes the real cells into latent space, then compares every real sample
against every generated population with latent-space MMD.

    matched    : MMD(real_i, gen_i)          generated from the correct pseudobulk
    mismatched : MMD(real_i, gen_j), j != i   generated from a different pseudobulk

If the pseudobulk conditioning drives generation, the matched MMD should be the
lowest in each row, so each real sample's own generated population is its nearest
match. The headline number is the top-1 correct-match rate.

No notebook, no cell editing, no display required. Figures are written to files.

Quick start (run from the repo root or from this notebooks/ directory; the
project root is auto-detected either way):

    python notebooks/run_conditioning_swap.py --dataset aml
    python notebooks/run_conditioning_swap.py --dataset brca

Fast held-out-only pass:

    python run_conditioning_swap.py --dataset aml --rows test

Everything is overridable, for example:

    python run_conditioning_swap.py \
        --data "/share/data/.../AML_vanGalen_2019_NanoWell_AnnData.h5ad" \
        --vae  output/checkpoint/AE/my_VAE_nocl/model_seed=0_step=199999.pt \
        --generated-glob 'output/simulated_samples/aml_pseudobulk_1M_nocl_*.npz' \
        --train-ids output/sample_splits/aml_nocl/train_samples.txt \
        --test-ids  output/sample_splits/aml_nocl/test_samples.txt \
        --out output/conditioning_swap_aml

Outputs written to --out:
    cross_mmd_matrix.csv                real x generated MMD matrix
    cross_mmd_long.csv                  same, long form
    conditioning_specificity_summary.csv   per-sample summary (the table for the paper)
    summary.txt                         the printed headline numbers
    heatmap.png                         real vs generated MMD heatmap
    matched_vs_mismatched.png           per-sample matched vs mismatched plot
"""

import argparse
import glob
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: write figures to files, never open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Dataset presets. Non-absolute paths are resolved against --project-root.
PRESETS = {
    "aml": {
        "data": "/share/data/transcriptomics/single cell/curated_mini_umi/AML_vanGalen_2019_NanoWell_AnnData.h5ad",
        "vae": "output/checkpoint/AE/my_VAE_nocl/model_seed=0_step=199999.pt",
        "generated_glob": "output/simulated_samples/aml_pseudobulk_1M_nocl_*.npz",
        "train_ids": "output/sample_splits/aml_nocl/train_samples.txt",
        "test_ids": "output/sample_splits/aml_nocl/test_samples.txt",
    },
    "brca": {
        "data": "/share/data/transcriptomics/single cell/curated_mini_umi/BreastCancer_SunnyWu_2021_AnnData.h5ad",
        # VAE trained on all 26 samples (incl. held-out test); see
        # brca2021_pseudobulk_1M_alltrain_deployment.sh.
        "vae": "output/checkpoint/AE/brca2021_VAE_alltrain/model_seed=1234_step=199999.pt",
        "generated_glob": "output/simulated_samples/brca2021_pseudobulk_1M_alltrain_*.npz",
        "train_ids": "output/sample_splits/brca2021_manual/train_samples.txt",
        "test_ids": "output/sample_splits/brca2021_manual/test_samples.txt",
    },
}


def detect_project_root(explicit):
    """Find the directory that contains guided_diffusion, matching the notebook."""
    if explicit:
        root = Path(explicit).resolve()
    else:
        cwd = Path.cwd()
        if (cwd / "guided_diffusion").exists():
            root = cwd
        elif (cwd.parent / "guided_diffusion").exists():
            root = cwd.parent
        else:
            root = cwd
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def resolve(path_str, project_root):
    """Resolve a path against project_root unless it is already absolute."""
    if path_str is None:
        return None
    p = Path(path_str)
    return p if p.is_absolute() else (project_root / p)


def scalar_from_npz(value):
    value = np.asarray(value)
    if value.shape == ():
        return value.item()
    flat = value.reshape(-1)
    return flat[0].item() if hasattr(flat[0], "item") else flat[0]


def to_numpy_dense(matrix):
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def build_payload(args, project_root):
    """Load adata, autoencoder, generated npz files; return latents per sample."""
    from guided_diffusion.cell_datasets_loader import (
        encode_cells_with_vae,
        read_preprocessed_adata,
        read_sample_ids_file,
    )

    data_path = resolve(args.data, project_root)
    vae_path = resolve(args.vae, project_root)
    gen_glob = str(resolve(args.generated_glob, project_root))

    train_ids = _safe_read_ids(read_sample_ids_file, resolve(args.train_ids, project_root))
    test_ids = _safe_read_ids(read_sample_ids_file, resolve(args.test_ids, project_root))

    generated_paths = sorted(Path(p) for p in glob.glob(gen_glob))
    if not generated_paths:
        raise FileNotFoundError(f"No generated files matched: {gen_glob}")
    print(f"generated files found: {len(generated_paths)}")

    # One generated population per sample (last file wins if duplicated).
    gen_by_sample = {}
    for path in generated_paths:
        with np.load(path, allow_pickle=True) as npzfile:
            sid = str(scalar_from_npz(npzfile["source_sample_id"]))
            gen_by_sample[sid] = np.asarray(npzfile["cell_gen"], dtype=np.float32)

    print(f"loading preprocessed adata: {data_path}")
    adata = read_preprocessed_adata(str(data_path))
    sample_values = adata.obs[args.sample_key].astype(str).to_numpy()
    print(f"adata shape: {adata.shape}, real SampleIDs: {len(set(sample_values))}")

    payload = {}
    for i, (sid, gen_latent) in enumerate(sorted(gen_by_sample.items())):
        mask = sample_values == sid
        if not mask.any():
            print(f"  skip {sid}: no real cells after preprocessing")
            continue
        real_gene = to_numpy_dense(adata[mask].X)
        real_latent = encode_cells_with_vae(
            real_gene,
            vae_path=str(vae_path),
            hidden_dim=args.hidden_dim,
            encode_batch_size=args.encode_batch,
        )
        payload[sid] = {"real_latent": real_latent, "generated_latent": gen_latent}
        print(f"  [{i + 1}/{len(gen_by_sample)}] encoded {sid}: "
              f"real={real_latent.shape[0]}, gen={gen_latent.shape[0]}")

    def split_label(sid):
        if test_ids is not None and sid in test_ids:
            return "test"
        if train_ids is not None and sid in train_ids:
            return "train"
        return "unknown"

    return payload, split_label, train_ids, test_ids


def _safe_read_ids(reader, path):
    if path is None:
        return None
    try:
        if not Path(path).exists():
            print(f"  note: split file not found, ignoring: {path}")
            return None
        ids = reader(str(path))
        return set(ids) if ids is not None else None
    except Exception as exc:  # noqa: BLE001
        print(f"  note: could not read split file {path}: {exc}")
        return None


def resolve_metric(name):
    """Return (metric_fn, human_label, filename_suffix) for the requested metric.

    Both back-ends live in guided_diffusion.pseudobulk_mmd_eval and take the same
    (real_latent, generated_latent) call shape, so the audit is metric-agnostic.
    """
    from guided_diffusion.pseudobulk_mmd_eval import (
        compute_edistance_summary,
        compute_mmd_summary,
    )

    if name == "edist":
        return (lambda a, b: float(compute_edistance_summary(a, b)["latent_edistance"]),
                "E-distance", "_edist")
    return (lambda a, b: float(compute_mmd_summary(a, b)["latent_mmd_rbf"]),
            "MMD", "")


def run_audit(payload, split_label, args, metric_fn):
    from guided_diffusion.pseudobulk_mmd_eval import subsample_rows

    # Subsample each population once for consistent, cheaper comparisons.
    real_sub, gen_sub = {}, {}
    for sid, p in payload.items():
        r = np.asarray(p["real_latent"], dtype=np.float32)
        g = np.asarray(p["generated_latent"], dtype=np.float32)
        if r.shape[0] < 2 or g.shape[0] < 2:
            continue
        real_sub[sid] = subsample_rows(r, args.max_cells, seed=args.seed)
        gen_sub[sid] = subsample_rows(g, args.max_cells, seed=args.seed)

    all_ids = sorted(set(real_sub) & set(gen_sub))
    if len(all_ids) < 2:
        raise ValueError("Need at least 2 samples with real and generated latents.")

    if args.rows == "all":
        row_ids = list(all_ids)
    elif args.rows == "test":
        row_ids = [s for s in all_ids if split_label(s) == "test"]
        if not row_ids:
            raise ValueError("--rows test but no sample is labeled 'test'. "
                             "Check --test-ids.")
    else:
        raise ValueError("--rows must be 'all' or 'test'.")

    gen_ids = list(all_ids)
    print(f"\nbuilding {len(row_ids)} x {len(gen_ids)} MMD matrix "
          f"({len(row_ids) * len(gen_ids)} evaluations)...")

    matrix = pd.DataFrame(index=row_ids, columns=gen_ids, dtype=float)
    long_rows = []
    for i, rid in enumerate(row_ids):
        ra = real_sub[rid]
        for gid in gen_ids:
            mmd = metric_fn(ra, gen_sub[gid])
            matrix.loc[rid, gid] = mmd
            long_rows.append({"real_id": rid, "gen_id": gid, "mmd": mmd,
                              "is_matched": rid == gid, "real_split": split_label(rid)})
        print(f"  [{i + 1}/{len(row_ids)}] real={rid}")

    cross_mmd_df = pd.DataFrame(long_rows)

    spec_rows = []
    for rid in row_ids:
        row = matrix.loc[rid]
        matched = float(row[rid])
        mismatched = row.drop(labels=rid)
        order = row.sort_values(ascending=True)
        true_rank = int(list(order.index).index(rid)) + 1
        spec_rows.append({
            "real_id": rid,
            "real_split": split_label(rid),
            "matched_mmd": matched,
            "nearest_mismatched_mmd": float(mismatched.min()),
            "mean_mismatched_mmd": float(mismatched.mean()),
            "margin": float(mismatched.min()) - matched,
            "true_match_rank": true_rank,
            "is_top1": true_rank == 1,
            "nearest_wrong_gen_id": str(mismatched.idxmin()),
            "n_generated_compared": len(gen_ids),
        })
    specificity_df = pd.DataFrame(spec_rows).sort_values("matched_mmd").reset_index(drop=True)
    return cross_mmd_df, specificity_df, matrix, row_ids


def summarize(specificity_df, metric_label="MMD"):
    m = metric_label
    lines = ["=" * 60, f"CONDITIONING-SPECIFICITY SUMMARY ({m})", "=" * 60]

    def block(df, label):
        if len(df) == 0:
            return
        lines.append(f"\n{label} (n = {len(df)})")
        lines.append(f"  top-1 correct-match rate      : {df['is_top1'].mean():.1%}")
        lines.append(f"  mean matched {m}              : {df['matched_mmd'].mean():.4f}")
        lines.append(f"  mean nearest-mismatched {m}   : {df['nearest_mismatched_mmd'].mean():.4f}")
        lines.append(f"  mean over all mismatched {m}  : {df['mean_mismatched_mmd'].mean():.4f}")
        lines.append(f"  mean rank of true match       : {df['true_match_rank'].mean():.2f}  (1 = best)")

    block(specificity_df, "All evaluated samples")
    for split in ["train", "test"]:
        block(specificity_df[specificity_df["real_split"] == split], f"Split = {split}")
    text = "\n".join(lines)
    print("\n" + text)
    return text


def plot_heatmap(matrix, row_ids, out_path, metric_label="MMD"):
    values = matrix.astype(float).to_numpy()
    fig, ax = plt.subplots(figsize=(0.28 * matrix.shape[1] + 3, 0.28 * matrix.shape[0] + 3))
    im = ax.imshow(values, aspect="auto")
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns, rotation=90, fontsize=6)
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels(matrix.index, fontsize=6)
    ax.set_xlabel("generated population (conditioned on this sample's pseudobulk)")
    ax.set_ylabel("real sample")
    ax.set_title(f"Cross-{metric_label}: real vs generated (lower is closer)")
    col_pos = {cid: j for j, cid in enumerate(matrix.columns)}
    for i, rid in enumerate(row_ids):
        if rid in col_pos:
            ax.scatter(col_pos[rid], i, marker="s", s=18,
                       facecolors="none", edgecolors="white", linewidths=1.0)
    fig.colorbar(im, ax=ax, shrink=0.7, label=f"latent-space {metric_label}")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_matched_vs_mismatched(specificity_df, out_path, metric_label="MMD"):
    def panel(ax, df, title):
        if len(df) == 0:
            ax.set_title(title)
            ax.text(0.5, 0.5, "no samples", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            return
        d = df.sort_values("matched_mmd")
        y = np.arange(len(d))
        ax.barh(y, d["matched_mmd"], alpha=0.85, label="matched (correct pseudobulk)")
        ax.scatter(d["nearest_mismatched_mmd"], y, color="tab:red", zorder=3, label="nearest mismatched")
        ax.scatter(d["mean_mismatched_mmd"], y, color="tab:gray", marker="|", zorder=3, label="mean mismatched")
        ax.set_yticks(y)
        ax.set_yticklabels(d["real_id"], fontsize=6)
        ax.set_xlabel(f"latent-space {metric_label} (lower is better)")
        ax.set_title(title)

    splits = [s for s in ["train", "test"] if (specificity_df["real_split"] == s).any()]
    if len(splits) == 2:
        n = max((specificity_df["real_split"] == "train").sum(),
                (specificity_df["real_split"] == "test").sum())
        fig, axes = plt.subplots(1, 2, figsize=(14, 0.32 * n + 2))
        panel(axes[0], specificity_df[specificity_df["real_split"] == "train"], "Training samples")
        panel(axes[1], specificity_df[specificity_df["real_split"] == "test"], "Test samples")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3)
        fig.suptitle("Conditioning specificity: matched vs mismatched pseudobulk")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        fig, ax = plt.subplots(figsize=(8, 0.32 * len(specificity_df) + 2))
        panel(ax, specificity_df, "Conditioning specificity: matched vs mismatched pseudobulk")
        ax.legend()
        plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=sorted(PRESETS), help="preset paths for aml or brca")
    ap.add_argument("--project-root", default=None, help="dir containing guided_diffusion (default: auto)")
    ap.add_argument("--data", default=None, help="preprocessed .h5ad (overrides preset)")
    ap.add_argument("--vae", default=None, help="autoencoder checkpoint (overrides preset)")
    ap.add_argument("--generated-glob", default=None, help="glob for generated .npz (overrides preset)")
    ap.add_argument("--train-ids", default=None, help="train sample-id file (overrides preset)")
    ap.add_argument("--test-ids", default=None, help="test sample-id file (overrides preset)")
    ap.add_argument("--sample-key", default="SampleID")
    ap.add_argument("--max-cells", type=int, default=1200, help="cell cap per population for MMD")
    ap.add_argument("--encode-batch", type=int, default=1024)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--rows", choices=["all", "test"], default="all",
                    help="which real samples form matrix rows (test = fast held-out pass)")
    ap.add_argument("--metric", choices=["mmd", "edist"], default="mmd",
                    help="distributional metric: mmd (RBF-MMD, default) or edist (energy distance)")
    ap.add_argument("--out", default=None, help="output dir (default: output/conditioning_swap_<dataset>)")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    # Fill from preset, letting explicit flags win.
    if args.dataset:
        preset = PRESETS[args.dataset]
        for key in ["data", "vae", "generated_glob", "train_ids", "test_ids"]:
            if getattr(args, key) is None:
                setattr(args, key, preset[key])

    if args.data is None or args.vae is None or args.generated_glob is None:
        ap.error("provide --dataset, or all of --data, --vae and --generated-glob.")

    if args.out is None:
        tag = args.dataset if args.dataset else "run"
        args.out = f"output/conditioning_swap_{tag}"
    return args


def main():
    args = parse_args()
    project_root = detect_project_root(args.project_root)
    print(f"project root: {project_root}")

    try:
        payload, split_label, _, _ = build_payload(args, project_root)
    except ModuleNotFoundError as exc:
        print(f"\nERROR: could not import project code ({exc}).")
        print("Run this from the bulk2scDiff project root, or pass --project-root "
              "pointing at the directory that contains guided_diffusion.")
        sys.exit(1)

    if len(payload) < 2:
        print("ERROR: fewer than 2 samples had both real and generated cells. Nothing to compare.")
        sys.exit(1)

    metric_fn, metric_label, suffix = resolve_metric(args.metric)
    print(f"metric: {metric_label}")

    cross_mmd_df, specificity_df, matrix, row_ids = run_audit(payload, split_label, args, metric_fn)
    summary_text = summarize(specificity_df, metric_label)

    out_dir = resolve(args.out, project_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Metric-suffixed filenames so an edist pass never overwrites the mmd outputs
    # (and vice versa) when both are written to the same directory.
    matrix.to_csv(out_dir / f"cross_mmd_matrix{suffix}.csv")
    cross_mmd_df.to_csv(out_dir / f"cross_mmd_long{suffix}.csv", index=False)
    specificity_df.to_csv(out_dir / f"conditioning_specificity_summary{suffix}.csv", index=False)
    (out_dir / f"summary{suffix}.txt").write_text(summary_text + "\n")

    if not args.no_plots:
        plot_heatmap(matrix, row_ids, out_dir / f"heatmap{suffix}.png", metric_label)
        plot_matched_vs_mismatched(specificity_df, out_dir / f"matched_vs_mismatched{suffix}.png", metric_label)

    print(f"\nAll outputs written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
