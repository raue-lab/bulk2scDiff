from pathlib import Path

import numpy as np
import torch
from anndata import read_h5ad
from scipy import sparse
from torch.utils.data import DataLoader, Dataset

from VAE.VAE_model import VAE


MIN_CELLS_PER_GENE = 3
MIN_GENES_PER_CELL = 10
NORM_TARGET_SUM = 1e4


def _device_or_default(device=None):
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _row_to_numpy(row):
    if sparse.issparse(row):
        row = row.toarray()
    return np.asarray(row, dtype=np.float32).reshape(-1)


def _matrix_slice_to_numpy(matrix, start=None, end=None):
    view = matrix if start is None and end is None else matrix[start:end]
    if sparse.issparse(view):
        view = view.toarray()
    return np.asarray(view, dtype=np.float32)


def _count_nonzero(matrix, axis):
    if sparse.issparse(matrix):
        return np.asarray(matrix.getnnz(axis=axis)).reshape(-1)
    return np.count_nonzero(np.asarray(matrix), axis=axis)


def _row_sums(matrix):
    if sparse.issparse(matrix):
        return np.asarray(matrix.sum(axis=1)).reshape(-1).astype(np.float32)
    return np.asarray(matrix, dtype=np.float32).sum(axis=1).astype(np.float32)


def _normalize_total_and_log1p(matrix, target_sum=NORM_TARGET_SUM, row_sums=None):
    if sparse.issparse(matrix):
        matrix = matrix.tocsr(copy=True).astype(np.float32)
        cell_sums = _row_sums(matrix) if row_sums is None else np.asarray(row_sums, dtype=np.float32)
        scale = np.divide(
            target_sum,
            np.maximum(cell_sums, 1e-12),
            out=np.zeros_like(cell_sums, dtype=np.float32),
            where=cell_sums > 0,
        )
        matrix = sparse.diags(scale) @ matrix
        matrix = matrix.tocsr()
        matrix.data = np.log1p(matrix.data)
        return matrix

    matrix = np.asarray(matrix, dtype=np.float32)
    cell_sums = matrix.sum(axis=1, keepdims=True).astype(np.float32)
    if row_sums is not None:
        cell_sums = np.asarray(row_sums, dtype=np.float32).reshape(-1, 1)
    cell_sums[cell_sums <= 0] = 1.0
    matrix = matrix * (target_sum / cell_sums)
    np.log1p(matrix, out=matrix)
    return matrix.astype(np.float32)


def encode_categories(values):
    categories, codes = np.unique(np.asarray(values).astype(str), return_inverse=True)
    return codes.astype(np.int64), categories.astype(str)


def read_sample_ids_file(path):
    if not path:
        return None

    sample_ids = []
    with open(Path(path), "r", encoding="utf-8") as handle:
        for line in handle:
            value = line.strip()
            if value and not value.startswith("#"):
                sample_ids.append(value)
    return set(sample_ids)


def filter_adata_by_sample_ids(
    adata,
    *,
    sample_key,
    include_sample_ids=None,
    exclude_sample_ids=None,
):
    keep_mask = build_sample_keep_mask(
        adata,
        sample_key=sample_key,
        include_sample_ids=include_sample_ids,
        exclude_sample_ids=exclude_sample_ids,
    )
    if keep_mask is None:
        return adata
    return adata[keep_mask].copy()


def build_sample_keep_mask(
    adata,
    *,
    sample_key,
    include_sample_ids=None,
    exclude_sample_ids=None,
):
    include_sample_ids = read_sample_ids_file(include_sample_ids)
    exclude_sample_ids = read_sample_ids_file(exclude_sample_ids)
    if include_sample_ids is None and exclude_sample_ids is None:
        return None

    if sample_key not in adata.obs:
        raise KeyError(f"'{sample_key}' not found in adata.obs")

    sample_values = adata.obs[sample_key].astype(str).to_numpy()
    keep_mask = np.ones(sample_values.shape[0], dtype=bool)
    if include_sample_ids is not None:
        keep_mask &= np.isin(sample_values, list(include_sample_ids))
    if exclude_sample_ids is not None:
        keep_mask &= ~np.isin(sample_values, list(exclude_sample_ids))
    if not keep_mask.any():
        raise ValueError("sample filtering removed every cell from the dataset")
    return keep_mask


def infer_num_genes_from_vae_checkpoint(vae_path):
    state_dict = torch.load(vae_path, map_location="cpu")
    return state_dict["encoder.network.0.1.weight"].shape[1]


def load_VAE(vae_path, num_gene, hidden_dim=128, device=None):
    device = _device_or_default(device)
    autoencoder = VAE(
        num_genes=num_gene,
        device=device,
        seed=0,
        hidden_dim=hidden_dim,
    )
    autoencoder.load_state_dict(torch.load(vae_path, map_location=device))
    autoencoder.eval()
    return autoencoder


def encode_cells_with_vae(
    expression_matrix,
    vae_path,
    hidden_dim=128,
    encode_batch_size=1024,
    device=None,
):
    autoencoder = load_VAE(
        vae_path=vae_path,
        num_gene=expression_matrix.shape[1],
        hidden_dim=hidden_dim,
        device=device,
    )
    device = _device_or_default(device)

    encoded_batches = []
    with torch.no_grad():
        for start in range(0, expression_matrix.shape[0], encode_batch_size):
            batch = _matrix_slice_to_numpy(expression_matrix, start, start + encode_batch_size)
            latent = autoencoder(torch.from_numpy(batch).to(device), return_latent=True)
            encoded_batches.append(latent.cpu())

    return torch.cat(encoded_batches, dim=0).numpy().astype(np.float32)


def decode_latents_with_vae(
    latent_data,
    vae_path,
    hidden_dim=128,
    decode_batch_size=1024,
    device=None,
):
    num_gene = infer_num_genes_from_vae_checkpoint(vae_path)
    autoencoder = load_VAE(
        vae_path=vae_path,
        num_gene=num_gene,
        hidden_dim=hidden_dim,
        device=device,
    )
    device = _device_or_default(device)

    decoded_batches = []
    latent_data = np.asarray(latent_data, dtype=np.float32)
    with torch.no_grad():
        for start in range(0, latent_data.shape[0], decode_batch_size):
            batch = latent_data[start : start + decode_batch_size]
            decoded = autoencoder(torch.from_numpy(batch).to(device), return_decoded=True)
            decoded_batches.append(decoded.cpu())

    return torch.cat(decoded_batches, dim=0).numpy().astype(np.float32)


def read_preprocessed_adata(
    data_dir,
    min_cells=MIN_CELLS_PER_GENE,
    min_genes=MIN_GENES_PER_CELL,
    target_sum=NORM_TARGET_SUM,
    return_raw_counts=False,
    return_library_sums=False,
):
    if not data_dir:
        raise ValueError("unspecified data directory")

    adata = read_h5ad(data_dir)
    adata.var_names_make_unique()

    raw_matrix = adata.X
    gene_mask = _count_nonzero(raw_matrix, axis=0) >= min_cells
    cell_mask = _count_nonzero(raw_matrix, axis=1) >= min_genes

    # Keep pre-gene-filter cell totals so normalization reflects the original library size.
    cell_library_sums = _row_sums(raw_matrix[cell_mask])
    adata = adata[cell_mask, gene_mask].copy()
    raw_counts = adata.X.copy()

    adata.X = _normalize_total_and_log1p(
        raw_counts,
        target_sum=target_sum,
        row_sums=cell_library_sums,
    )
    outputs = [adata]
    if return_raw_counts:
        outputs.append(raw_counts)
    if return_library_sums:
        outputs.append(cell_library_sums.astype(np.float32))
    if len(outputs) == 1:
        return outputs[0]
    return tuple(outputs)


def compute_pseudobulk(
    expression_matrix,
    sample_values,
    sample_library_sums=None,
    normalize_and_log1p=False,
    target_sum=NORM_TARGET_SUM,
):
    sample_codes, sample_ids = encode_categories(sample_values)
    sample_counts = np.bincount(sample_codes, minlength=len(sample_ids)).astype(np.int64)
    pseudobulk = np.zeros((len(sample_ids), expression_matrix.shape[1]), dtype=np.float32)
    aggregated_library_sums = None
    if sample_library_sums is not None:
        sample_library_sums = np.asarray(sample_library_sums, dtype=np.float32).reshape(-1)
        if sample_library_sums.shape[0] != expression_matrix.shape[0]:
            raise ValueError("sample_library_sums must align with the rows in expression_matrix")
        aggregated_library_sums = np.zeros(len(sample_ids), dtype=np.float32)

    for sample_idx in range(len(sample_ids)):
        sample_mask = sample_codes == sample_idx
        sample_matrix = expression_matrix[sample_mask]
        vector = np.asarray(sample_matrix.sum(axis=0)).reshape(-1)
        pseudobulk[sample_idx] = vector.astype(np.float32)
        if aggregated_library_sums is not None:
            sample_totals = sample_library_sums[sample_mask]
            aggregated_library_sums[sample_idx] = sample_totals.sum(dtype=np.float32)

    if normalize_and_log1p:
        pseudobulk = _normalize_total_and_log1p(
            pseudobulk,
            target_sum=target_sum,
            row_sums=aggregated_library_sums,
        )

    return pseudobulk, sample_codes.astype(np.int64), sample_ids, sample_counts


def load_data(
    *,
    data_dir,
    batch_size,
    vae_path=None,
    deterministic=False,
    train_vae=False,
    hidden_dim=128,
    sample_key="SampleID",
    include_pseudobulk=False,
    encode_batch_size=1024,
    include_sample_ids_path="",
    exclude_sample_ids_path="",
    device=None,
):
    adata, raw_counts, cell_library_sums = read_preprocessed_adata(
        data_dir,
        return_raw_counts=True,
        return_library_sums=True,
    )
    keep_mask = build_sample_keep_mask(
        adata,
        sample_key=sample_key,
        include_sample_ids=include_sample_ids_path,
        exclude_sample_ids=exclude_sample_ids_path,
    )
    if keep_mask is not None:
        adata = adata[keep_mask].copy()
        raw_counts = raw_counts[keep_mask]
        cell_library_sums = cell_library_sums[keep_mask]

    expression_matrix = adata.X
    if train_vae:
        cell_data = expression_matrix
    else:
        if vae_path is None:
            raise ValueError("vae_path must be provided when train_vae=False")
        cell_data = encode_cells_with_vae(
            expression_matrix=expression_matrix,
            vae_path=vae_path,
            hidden_dim=hidden_dim,
            encode_batch_size=encode_batch_size,
            device=device,
        )

    sample_codes = None
    pseudobulk = None
    if include_pseudobulk:
        if sample_key not in adata.obs:
            raise KeyError(f"'{sample_key}' not found in adata.obs")
        sample_values = adata.obs[sample_key].astype(str).to_numpy()
        # Build the condition from raw counts and normalize at the sample level.
        pseudobulk, sample_codes, _, _ = compute_pseudobulk(
            raw_counts,
            sample_values,
            sample_library_sums=cell_library_sums,
            normalize_and_log1p=True,
        )

    dataset = CellDataset(
        cell_data=cell_data,
        sample_index=sample_codes,
        pseudobulk=pseudobulk,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not deterministic,
        num_workers=0,
        drop_last=True,
    )

    while True:
        yield from loader


class CellDataset(Dataset):
    def __init__(self, cell_data, sample_index=None, pseudobulk=None):
        super().__init__()
        self.data = cell_data
        self.sample_index = sample_index
        self.pseudobulk = pseudobulk

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        out_dict = {}
        if self.pseudobulk is not None and self.sample_index is not None:
            out_dict["pseudobulk"] = np.asarray(
                self.pseudobulk[self.sample_index[idx]],
                dtype=np.float32,
            )
        return _row_to_numpy(self.data[idx]), out_dict
