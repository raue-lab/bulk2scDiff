import csv
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch as th

from . import dist_util, logger
from .cell_datasets_loader import (
    compute_pseudobulk,
    encode_cells_with_vae,
    read_preprocessed_adata,
    read_sample_ids_file,
)


@dataclass(frozen=True)
class EvalSample:
    sample_id: str
    split_name: str
    pseudobulk: np.ndarray
    real_latent_eval: np.ndarray
    real_num_cells: int


def to_numpy_dense(matrix):
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def subsample_rows(matrix, max_rows, seed=1234):
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape[0] <= max_rows:
        return matrix
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(matrix.shape[0], size=max_rows, replace=False))
    return matrix[idx]


def pairwise_sq_dists(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x_norm = np.sum(x * x, axis=1, keepdims=True)
    y_norm = np.sum(y * y, axis=1, keepdims=True).T
    d2 = x_norm + y_norm - 2.0 * x @ y.T
    return np.maximum(d2, 0.0)


def median_heuristic_sigma(x, y):
    pooled = np.concatenate([x, y], axis=0)
    if pooled.shape[0] < 2:
        return np.nan
    d2 = pairwise_sq_dists(pooled, pooled)
    tri = d2[np.triu_indices_from(d2, k=1)]
    tri = tri[tri > 0]
    if tri.size == 0:
        return np.nan
    return float(np.sqrt(np.median(tri)))


def rbf_kernel_from_sq_dists(d2, sigma):
    sigma = float(sigma)
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be positive")
    return np.exp(-d2 / (2.0 * sigma * sigma))


def unbiased_mmd2_rbf(x, y, sigma):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = x.shape[0]
    m = y.shape[0]
    if n < 2 or m < 2:
        return np.nan

    k_xx = rbf_kernel_from_sq_dists(pairwise_sq_dists(x, x), sigma)
    k_yy = rbf_kernel_from_sq_dists(pairwise_sq_dists(y, y), sigma)
    k_xy = rbf_kernel_from_sq_dists(pairwise_sq_dists(x, y), sigma)

    np.fill_diagonal(k_xx, 0.0)
    np.fill_diagonal(k_yy, 0.0)

    term_xx = k_xx.sum() / (n * (n - 1))
    term_yy = k_yy.sum() / (m * (m - 1))
    term_xy = k_xy.mean()
    return float(term_xx + term_yy - 2.0 * term_xy)


def compute_mmd_summary(real_latent, generated_latent):
    sigma = median_heuristic_sigma(real_latent, generated_latent)
    if not np.isfinite(sigma) or sigma <= 0:
        return {
            "latent_mmd_rbf": np.nan,
            "latent_mmd_sigma": np.nan,
            "latent_mmd_kernel_count": 0,
        }

    sigmas = [sigma * 0.5, sigma, sigma * 2.0]
    mmd_values = [unbiased_mmd2_rbf(real_latent, generated_latent, s) for s in sigmas]
    return {
        "latent_mmd_rbf": float(np.nanmean(mmd_values)),
        "latent_mmd_sigma": float(sigma),
        "latent_mmd_kernel_count": len(sigmas),
    }


def energy_distance_sq(x, y):
    """scperturb-style E-distance in squared-Euclidean space (Peidli et al. 2024).

    E = 2 * delta_xy - delta_xx - delta_yy, where each delta is the mean pairwise
    squared-Euclidean distance. The within-sample terms include the zero diagonal,
    matching scperturb's estimator. Computed in latent space (same space as the
    RBF-MMD above); lower means the two populations are more alike, and E is exactly
    0 when x and y are the same array and ~0 for two samples of the same distribution.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape[0] < 2 or y.shape[0] < 2:
        return np.nan
    delta_xy = float(pairwise_sq_dists(x, y).mean())
    delta_xx = float(pairwise_sq_dists(x, x).mean())
    delta_yy = float(pairwise_sq_dists(y, y).mean())
    return float(2.0 * delta_xy - delta_xx - delta_yy)


def compute_edistance_summary(real_latent, generated_latent):
    """E-distance analogue of compute_mmd_summary, with the same call shape.

    Returns a dict with a single ``latent_edistance`` key so notebooks can import
    and call it exactly like ``compute_mmd_summary``.
    """
    return {"latent_edistance": energy_distance_sq(real_latent, generated_latent)}


def _resolve_eval_split_ids(
    *,
    all_sample_ids,
    include_sample_ids_path="",
    exclude_sample_ids_path="",
    validation_sample_ids_path="",
):
    available_ids = list(map(str, all_sample_ids))
    available_set = set(available_ids)

    include_ids = read_sample_ids_file(include_sample_ids_path)
    exclude_ids = read_sample_ids_file(exclude_sample_ids_path)
    validation_ids = read_sample_ids_file(validation_sample_ids_path)

    if include_ids is not None:
        train_set = set(include_ids)
    elif exclude_ids is not None:
        train_set = available_set - set(exclude_ids)
    else:
        train_set = set(available_ids)

    if validation_ids is not None:
        validation_set = set(validation_ids)
    elif exclude_ids is not None:
        validation_set = set(exclude_ids)
    elif include_ids is not None:
        validation_set = available_set - train_set
    else:
        validation_set = set()

    missing_train = sorted(train_set - available_set)
    missing_validation = sorted(validation_set - available_set)
    if missing_train:
        logger.log(
            f"ignoring {len(missing_train)} training eval SampleIDs missing from the dataset"
        )
    if missing_validation:
        logger.log(
            f"ignoring {len(missing_validation)} validation eval SampleIDs missing from the dataset"
        )

    overlap = train_set & validation_set
    if overlap:
        logger.log(
            f"removing {len(overlap)} overlapping SampleIDs from validation eval split"
        )
        validation_set -= overlap

    train_ids = [sample_id for sample_id in available_ids if sample_id in train_set]
    validation_ids = [
        sample_id for sample_id in available_ids if sample_id in validation_set
    ]
    return train_ids, validation_ids


class PeriodicPseudobulkMMDEvaluator:
    def __init__(
        self,
        *,
        data_dir,
        vae_path,
        sample_key,
        eval_interval,
        include_sample_ids_path="",
        exclude_sample_ids_path="",
        validation_sample_ids_path="",
        max_cells_for_mmd=1200,
        sampling_batch_size=1000,
        encode_batch_size=1024,
        use_ddim=False,
        random_seed=1234,
        hidden_dim=128,
        output_dir="",
    ):
        self.data_dir = data_dir
        self.vae_path = vae_path
        self.sample_key = sample_key
        self.eval_interval = int(eval_interval)
        self.include_sample_ids_path = include_sample_ids_path
        self.exclude_sample_ids_path = exclude_sample_ids_path
        self.validation_sample_ids_path = validation_sample_ids_path
        self.max_cells_for_mmd = int(max_cells_for_mmd)
        self.sampling_batch_size = int(sampling_batch_size)
        self.encode_batch_size = int(encode_batch_size)
        self.use_ddim = bool(use_ddim)
        self.random_seed = int(random_seed)
        self.hidden_dim = int(hidden_dim)
        self.output_dir = Path(output_dir) if output_dir else None
        self.latent_dim = None

        self.split_to_samples = self._prepare_eval_samples()
        self.detail_csv_path = None
        if self.output_dir is not None:
            self.detail_csv_path = self.output_dir / "mmd_eval_history.csv"
            self.detail_csv_path.parent.mkdir(parents=True, exist_ok=True)
            self._ensure_detail_csv_header()

    def _prepare_eval_samples(self):
        logger.log("precomputing periodic pseudobulk MMD evaluation payload...")
        adata, raw_counts, cell_library_sums = read_preprocessed_adata(
            self.data_dir,
            return_raw_counts=True,
            return_library_sums=True,
        )
        if self.sample_key not in adata.obs:
            raise KeyError(f"'{self.sample_key}' not found in adata.obs")

        sample_values = adata.obs[self.sample_key].astype(str).to_numpy()
        pseudobulk_matrix, _, sample_ids, sample_counts = compute_pseudobulk(
            raw_counts,
            sample_values,
            sample_library_sums=cell_library_sums,
            normalize_and_log1p=True,
        )
        real_latent = encode_cells_with_vae(
            adata.X,
            vae_path=self.vae_path,
            hidden_dim=self.hidden_dim,
            encode_batch_size=self.encode_batch_size,
            device=dist_util.dev(),
        )
        self.latent_dim = int(real_latent.shape[1])

        train_ids, validation_ids = _resolve_eval_split_ids(
            all_sample_ids=sample_ids,
            include_sample_ids_path=self.include_sample_ids_path,
            exclude_sample_ids_path=self.exclude_sample_ids_path,
            validation_sample_ids_path=self.validation_sample_ids_path,
        )

        sample_to_index = {sample_id: idx for idx, sample_id in enumerate(sample_ids)}
        split_to_ids = {
            "train": train_ids,
            "validation": validation_ids,
        }
        split_to_samples = {}
        for split_name, split_ids in split_to_ids.items():
            eval_samples = []
            for sample_id in split_ids:
                sample_mask = sample_values == sample_id
                sample_latent = real_latent[sample_mask]
                sample_latent = subsample_rows(
                    sample_latent,
                    self.max_cells_for_mmd,
                    seed=self.random_seed,
                )
                if sample_latent.shape[0] < 2:
                    logger.log(
                        f"skipping {split_name} SampleID {sample_id!r}: fewer than 2 cells after subsampling"
                    )
                    continue

                sample_idx = sample_to_index[sample_id]
                eval_samples.append(
                    EvalSample(
                        sample_id=sample_id,
                        split_name=split_name,
                        pseudobulk=np.asarray(
                            pseudobulk_matrix[sample_idx],
                            dtype=np.float32,
                        ),
                        real_latent_eval=np.asarray(sample_latent, dtype=np.float32),
                        real_num_cells=int(sample_counts[sample_idx]),
                    )
                )
            split_to_samples[split_name] = eval_samples

        logger.log(
            "prepared periodic MMD evaluation splits: "
            f"{len(split_to_samples['train'])} train samples, "
            f"{len(split_to_samples['validation'])} validation samples"
        )
        return split_to_samples

    def _ensure_detail_csv_header(self):
        if self.detail_csv_path is None or self.detail_csv_path.exists():
            return
        with open(self.detail_csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "step",
                    "sample_id",
                    "split",
                    "latent_mmd_rbf",
                    "latent_mmd_sigma",
                    "real_num_cells",
                    "mmd_eval_real_cells",
                    "mmd_eval_generated_cells",
                ],
            )
            writer.writeheader()

    def _append_detail_rows(self, rows):
        if self.detail_csv_path is None or not rows:
            return
        self._ensure_detail_csv_header()
        with open(self.detail_csv_path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "step",
                    "sample_id",
                    "split",
                    "latent_mmd_rbf",
                    "latent_mmd_sigma",
                    "real_num_cells",
                    "mmd_eval_real_cells",
                    "mmd_eval_generated_cells",
                ],
            )
            writer.writerows(rows)

    def _capture_rng_state(self):
        state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": th.get_rng_state(),
        }
        if th.cuda.is_available():
            state["torch_cuda"] = th.cuda.get_rng_state_all()
        return state

    def _restore_rng_state(self, state):
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        th.set_rng_state(state["torch"])
        if th.cuda.is_available() and "torch_cuda" in state:
            th.cuda.set_rng_state_all(state["torch_cuda"])

    def _set_eval_seed(self, step):
        seed = self.random_seed + int(step)
        random.seed(seed)
        np.random.seed(seed)
        th.manual_seed(seed)
        if th.cuda.is_available():
            th.cuda.manual_seed_all(seed)

    def _sample_generated_latents(self, model, diffusion, pseudobulk, num_samples):
        sample_fn = diffusion.ddim_sample_loop if self.use_ddim else diffusion.p_sample_loop
        pseudobulk_batch = (
            th.from_numpy(np.asarray(pseudobulk, dtype=np.float32))
            .to(dist_util.dev())
            .unsqueeze(0)
        )

        generated = []
        total_created = 0
        while total_created < num_samples:
            current_batch_size = min(
                self.sampling_batch_size,
                num_samples - total_created,
            )
            model_kwargs = {
                "pseudobulk": pseudobulk_batch.repeat(current_batch_size, 1),
            }
            sample, _ = sample_fn(
                model,
                (current_batch_size, self.latent_dim),
                clip_denoised=False,
                model_kwargs=model_kwargs,
                start_time=diffusion.betas.shape[0],
            )
            generated.append(sample.detach().cpu().numpy())
            total_created += current_batch_size

        return np.concatenate(generated, axis=0).astype(np.float32)

    def _summarize_split(self, split_name, values, metrics):
        prefix = f"mmd_{split_name}"
        if values:
            arr = np.asarray(values, dtype=np.float64)
            metrics[f"{prefix}_mean"] = float(np.nanmean(arr))
            metrics[f"{prefix}_median"] = float(np.nanmedian(arr))
            metrics[f"{prefix}_min"] = float(np.nanmin(arr))
            metrics[f"{prefix}_max"] = float(np.nanmax(arr))
            metrics[f"{prefix}_num_samples"] = int(np.isfinite(arr).sum())
        else:
            metrics[f"{prefix}_mean"] = np.nan
            metrics[f"{prefix}_median"] = np.nan
            metrics[f"{prefix}_min"] = np.nan
            metrics[f"{prefix}_max"] = np.nan
            metrics[f"{prefix}_num_samples"] = 0

    def evaluate(self, model, diffusion, *, step):
        rng_state = self._capture_rng_state()
        model_was_training = model.training
        detail_rows = []
        metrics = {}

        try:
            self._set_eval_seed(step)
            model.eval()
            with th.no_grad():
                for split_name, eval_samples in self.split_to_samples.items():
                    split_mmds = []
                    for eval_sample in eval_samples:
                        generated_latent = self._sample_generated_latents(
                            model,
                            diffusion,
                            eval_sample.pseudobulk,
                            num_samples=eval_sample.real_latent_eval.shape[0],
                        )
                        mmd_summary = compute_mmd_summary(
                            eval_sample.real_latent_eval,
                            generated_latent,
                        )
                        split_mmds.append(mmd_summary["latent_mmd_rbf"])
                        detail_rows.append(
                            {
                                "step": int(step),
                                "sample_id": eval_sample.sample_id,
                                "split": split_name,
                                "latent_mmd_rbf": mmd_summary["latent_mmd_rbf"],
                                "latent_mmd_sigma": mmd_summary["latent_mmd_sigma"],
                                "real_num_cells": eval_sample.real_num_cells,
                                "mmd_eval_real_cells": int(
                                    eval_sample.real_latent_eval.shape[0]
                                ),
                                "mmd_eval_generated_cells": int(
                                    generated_latent.shape[0]
                                ),
                            }
                        )
                    self._summarize_split(split_name, split_mmds, metrics)

            train_mean = metrics.get("mmd_train_mean")
            validation_mean = metrics.get("mmd_validation_mean")
            if np.isfinite(train_mean) and np.isfinite(validation_mean):
                metrics["mmd_validation_minus_train_mean"] = float(
                    validation_mean - train_mean
                )
            else:
                metrics["mmd_validation_minus_train_mean"] = np.nan
            return metrics, detail_rows
        finally:
            if model_was_training:
                model.train()
            self._restore_rng_state(rng_state)

    def log_evaluation(self, model, diffusion, *, step):
        metrics, detail_rows = self.evaluate(model, diffusion, step=step)
        for key, value in metrics.items():
            logger.logkv(key, value)
        self._append_detail_rows(detail_rows)
        return metrics
