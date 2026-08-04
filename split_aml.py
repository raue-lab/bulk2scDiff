import argparse
import csv
from pathlib import Path

import numpy as np

from guided_diffusion.cell_datasets_loader import (
    read_preprocessed_adata,
    read_sample_ids_file,
)


def assign_splits(sample_ids, test_fraction, seed):
    """Randomly split sample IDs into train and test sets.

    Args:
        sample_ids: Sequence of unique sample identifiers.
        test_fraction: Fraction of samples to reserve for the test set (e.g. 0.2).
        seed: Integer seed for the random number generator.

    Returns:
        A tuple (train_ids, test_ids), each sorted numpy array of sample IDs.

    Raises:
        ValueError: If test_fraction is not in (0, 1) or the split leaves no
                    training samples.
    """
    if not (0 < test_fraction < 1):
        raise ValueError("test_fraction must be strictly between 0 and 1")

    rng = np.random.default_rng(seed)
    order = np.arange(len(sample_ids))
    rng.shuffle(order)
    sample_ids = np.asarray(sample_ids)[order]

    n_total = len(sample_ids)
    n_test = int(round(n_total * test_fraction))
    n_train = n_total - n_test
    if n_train <= 0:
        raise ValueError("test_fraction leaves no training samples")

    train_ids = np.sort(sample_ids[:n_train])
    test_ids = np.sort(sample_ids[n_train:])
    return train_ids, test_ids


def inherit_splits(sample_ids, inherit_split_dir):
    """Assign each sample to the split it had in *inherit_split_dir*.

    Reads ``train_samples.txt`` and ``test_samples.txt`` from the directory and
    assigns every sample in *sample_ids* to whichever split previously contained
    it. This preserves an existing train/test partition exactly (e.g. when only a
    few samples are removed) instead of drawing a fresh random split.

    Args:
        sample_ids: Sequence of unique sample identifiers to assign.
        inherit_split_dir: Directory holding the reference train/test files.

    Returns:
        A tuple (train_ids, test_ids), each a sorted numpy array of sample IDs.

    Raises:
        ValueError: If any sample in *sample_ids* is absent from both reference
                    split files.
    """
    inherit_dir = Path(inherit_split_dir)
    prev_train = read_sample_ids_file(inherit_dir / "train_samples.txt")
    prev_test = read_sample_ids_file(inherit_dir / "test_samples.txt")

    train_ids, test_ids, unassigned = [], [], []
    for sample_id in sample_ids:
        if sample_id in prev_train:
            train_ids.append(sample_id)
        elif sample_id in prev_test:
            test_ids.append(sample_id)
        else:
            unassigned.append(sample_id)

    if unassigned:
        raise ValueError(
            f"{len(unassigned)} sample(s) absent from both split files in "
            f"{inherit_dir}: {sorted(unassigned)}"
        )

    return np.sort(np.asarray(train_ids)), np.sort(np.asarray(test_ids))


def write_sample_file(path, values):
    """Write one sample ID per line to *path*, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for value in values:
            handle.write(f"{value}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Create an 80/20 train/test split from an AnnData dataset."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/share/data/AML_vanGalen_2019_NanoWell_AnnData.h5ad",
    )
    parser.add_argument("--sample_key", type=str, default="SampleID")
    parser.add_argument("--test_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output_dir", type=str, default="output/sample_splits/aml")
    parser.add_argument(
        "--exclude_sample_ids_path",
        type=str,
        default="",
        help="Optional file of sample IDs to drop from the cohort before splitting.",
    )
    parser.add_argument(
        "--inherit_split_dir",
        type=str,
        default="",
        help=(
            "Optional directory with train_samples.txt/test_samples.txt. When set, "
            "each kept sample keeps the split it had there instead of drawing a "
            "fresh random split (--seed/--test_fraction are then unused)."
        ),
    )
    args = parser.parse_args()

    adata = read_preprocessed_adata(args.data_dir)
    if args.sample_key not in adata.obs:
        raise KeyError(f"'{args.sample_key}' not found in adata.obs")

    sample_values = adata.obs[args.sample_key].astype(str).to_numpy()
    sample_ids, sample_counts = np.unique(sample_values, return_counts=True)

    if args.exclude_sample_ids_path:
        excluded = read_sample_ids_file(args.exclude_sample_ids_path)
        missing = excluded - set(sample_ids.tolist())
        if missing:
            raise ValueError(
                f"--exclude_sample_ids_path lists IDs absent from the data: {sorted(missing)}"
            )
        keep_mask = ~np.isin(sample_ids, list(excluded))
        sample_ids = sample_ids[keep_mask]
        sample_counts = sample_counts[keep_mask]
        print(f"Excluded {len(excluded)} sample(s): {sorted(excluded)}")

    if args.inherit_split_dir:
        train_ids, test_ids = inherit_splits(sample_ids, args.inherit_split_dir)
    else:
        train_ids, test_ids = assign_splits(
            sample_ids,
            test_fraction=args.test_fraction,
            seed=args.seed,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_sample_file(output_dir / "train_samples.txt", train_ids)
    write_sample_file(output_dir / "test_samples.txt", test_ids)

    split_lookup = {}
    for split_name, values in (("train", train_ids), ("test", test_ids)):
        split_lookup.update({value: split_name for value in values})

    with open(output_dir / "sample_summary.tsv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "split", "num_cells"],
            delimiter="\t",
        )
        writer.writeheader()
        for sample_id, num_cells in sorted(zip(sample_ids, sample_counts), key=lambda item: item[0]):
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "split": split_lookup[sample_id],
                    "num_cells": int(num_cells),
                }
            )

    print(
        {
            "num_samples": int(len(sample_ids)),
            "train": int(len(train_ids)),
            "test": int(len(test_ids)),
            "output_dir": str(output_dir),
        }
    )


if __name__ == "__main__":
    main()
