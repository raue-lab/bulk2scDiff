"""Write the fixed BRCA2021 subtype-balanced sample split files.

Implements a strict 80/20 train/test split (21 train, 5 test out of 26 total)
with balanced cancer-subtype representation in both sets.  The assignment is
fixed (not random) to guarantee reproducibility and subtype balance.

Subtype composition:
    Train (21): ER+ × 9, HER2+ × 4, TNBC × 8
    Test  ( 5): ER+ × 2, HER2+ × 1, TNBC × 2
"""
import argparse
import csv
from pathlib import Path

import anndata as ad


DATA_DIR_DEFAULT = (
    "/share/data/"
    "BreastCancer_SunnyWu_2021_AnnData.h5ad"
)
SAMPLE_KEY_DEFAULT = "SampleID"
GROUP_KEY_DEFAULT = "subtype"
OUTPUT_DIR_DEFAULT = "output/sample_splits/brca2021_manual"

# Fixed assignment — former validation samples have been absorbed into train to
# achieve a strict 80/20 split while preserving subtype balance.
SPLITS = {
    "train": [
        # original train samples
        "CID3586",
        "CID3838",
        "CID3921",
        "CID3941",
        "CID3946",
        "CID4040",
        "CID4067",
        "CID4290A",
        "CID44041",
        "CID4461",
        "CID4465",
        "CID4471",
        "CID4495",
        "CID44991",
        "CID4515",
        "CID4530N",
        # former validation samples merged into train
        "CID3963",
        "CID4066",
        "CID4398",
        "CID4463",
        "CID4513",
    ],
    "test": [
        "CID3948",
        "CID44971",
        "CID45171",
        "CID4523",
        "CID4535",
    ],
}

# Expected subtype counts per split — used as a sanity check against the data.
EXPECTED_COUNTS = {
    "train": {"ER+": 9, "HER2+": 4, "TNBC": 8},
    "test": {"ER+": 2, "HER2+": 1, "TNBC": 2},
}


def write_sample_file(path, values):
    """Write one sample ID per line to *path*, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for value in values:
            handle.write(f"{value}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Write the fixed BRCA2021 subtype-balanced train/test split files."
    )
    parser.add_argument("--data_dir", type=str, default=DATA_DIR_DEFAULT)
    parser.add_argument("--sample_key", type=str, default=SAMPLE_KEY_DEFAULT)
    parser.add_argument("--group_key", type=str, default=GROUP_KEY_DEFAULT)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR_DEFAULT)
    args = parser.parse_args()

    adata = ad.read_h5ad(args.data_dir, backed="r")
    if args.sample_key not in adata.obs:
        raise KeyError(f"'{args.sample_key}' not found in adata.obs")
    if args.group_key not in adata.obs:
        raise KeyError(f"'{args.group_key}' not found in adata.obs")

    sample_values = adata.obs[args.sample_key].astype(str)
    group_values = adata.obs[args.group_key].astype(str)
    sample_ids = sorted(sample_values.unique())

    # Verify no sample appears in more than one split.
    assigned_ids = [sid for split_ids in SPLITS.values() for sid in split_ids]
    if len(assigned_ids) != len(set(assigned_ids)):
        raise ValueError("duplicate SampleID detected across splits")

    # Verify the fixed splits exactly cover the dataset.
    missing = sorted(set(sample_ids) - set(assigned_ids))
    extra = sorted(set(assigned_ids) - set(sample_ids))
    if missing or extra:
        raise ValueError(f"split coverage mismatch: missing={missing}, extra={extra}")

    # Map each sample to its unique subtype and cell count.
    sample_to_group = {}
    sample_to_count = {}
    for sample_id in sample_ids:
        sample_mask = sample_values == sample_id
        sample_groups = sorted(group_values[sample_mask].unique())
        if len(sample_groups) != 1:
            raise ValueError(
                f"SampleID {sample_id!r} maps to multiple groups: {sample_groups}"
            )
        sample_to_group[sample_id] = sample_groups[0]
        sample_to_count[sample_id] = int(sample_mask.sum())

    # Validate subtype composition matches the expected counts.
    for split_name, split_ids in SPLITS.items():
        observed = {}
        for sample_id in split_ids:
            group_name = sample_to_group[sample_id]
            observed[group_name] = observed.get(group_name, 0) + 1
        if observed != EXPECTED_COUNTS[split_name]:
            raise ValueError(
                f"{split_name} group counts mismatch: "
                f"expected {EXPECTED_COUNTS[split_name]}, observed {observed}"
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_ids in SPLITS.items():
        write_sample_file(output_dir / f"{split_name}_samples.txt", split_ids)

    split_lookup = {
        sample_id: split_name
        for split_name, split_ids in SPLITS.items()
        for sample_id in split_ids
    }
    with open(output_dir / "sample_summary.tsv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "split", "subtype", "num_cells"],
            delimiter="\t",
        )
        writer.writeheader()
        for sample_id in sample_ids:
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "split": split_lookup[sample_id],
                    "subtype": sample_to_group[sample_id],
                    "num_cells": sample_to_count[sample_id],
                }
            )

    print(
        {
            "output_dir": str(output_dir),
            "split_sizes": {split: len(ids) for split, ids in SPLITS.items()},
            "group_sizes": {split: EXPECTED_COUNTS[split] for split in ("train", "test")},
        }
    )


if __name__ == "__main__":
    main()
