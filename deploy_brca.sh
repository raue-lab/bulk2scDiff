#!/usr/bin/env bash
# BRCA2021 pseudobulk-conditioned full pipeline 
#
# Steps:
#   1. Write the fixed subtype-balanced train/test split files 
#   2. Fine-tune the VAE 
#   3. Train the diffusion backbone (train split only)
#   4. Generate one .npz per sample for the train and test splits
#
# Usage:
#   bash deploy_brca.sh
#
# Optional env-var overrides:
#   FORCE_REGENERATE_SAMPLES=1   — overwrite existing .npz files
#   NUM_SAMPLES_OVERRIDE=N       — generate N cells instead of matching real count

set -euo pipefail
shopt -s nullglob

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate scDiffusion

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR="${DATA_DIR:-/share/data/transcriptomics/single cell/curated_mini_umi/BreastCancer_SunnyWu_2021_AnnData.h5ad}"
SAMPLE_KEY="${SAMPLE_KEY:-SampleID}"
GROUP_KEY="${GROUP_KEY:-subtype}"
SEED="${SEED:-1234}"

SPLIT_DIR="${SPLIT_DIR:-${PROJECT_ROOT}/output/sample_splits/brca2021_manual}"
TRAIN_SAMPLE_IDS_PATH="${SPLIT_DIR}/train_samples.txt"
TEST_SAMPLE_IDS_PATH="${SPLIT_DIR}/test_samples.txt"

VAE_NUM_GENES="${VAE_NUM_GENES:-25209}"
VAE_SAVE_DIR="${VAE_SAVE_DIR:-${PROJECT_ROOT}/output/checkpoint/AE/brca2021_VAE_alltrain}"
VAE_LOG_DIR="${VAE_LOG_DIR:-${PROJECT_ROOT}/output/logs/brca2021_VAE_alltrain}"
VAE_MAX_STEPS="${VAE_MAX_STEPS:-200000}"
VAE_CHECKPOINT_FREQ="${VAE_CHECKPOINT_FREQ:-50000}"
VAE_BATCH_SIZE="${VAE_BATCH_SIZE:-128}"

MODEL_NAME="${MODEL_NAME:-brca2021_pseudobulk_1M_alltrain}"
BACKBONE_SAVE_DIR="${BACKBONE_SAVE_DIR:-${PROJECT_ROOT}/output/checkpoint/backbone}"
MODEL_DIR="${BACKBONE_SAVE_DIR}/${MODEL_NAME}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-1000000}"
DIFFUSION_SAVE_INTERVAL="${DIFFUSION_SAVE_INTERVAL:-200000}"
DIFFUSION_BATCH_SIZE="${DIFFUSION_BATCH_SIZE:-128}"
COND_EMBED_DIM="${COND_EMBED_DIM:-128}"
MMD_EVAL_INTERVAL="${MMD_EVAL_INTERVAL:-10000}"
MMD_EVAL_MAX_CELLS="${MMD_EVAL_MAX_CELLS:-1200}"

SAMPLING_BATCH_SIZE="${SAMPLING_BATCH_SIZE:-1000}"
TRAIN_SAMPLE_PREFIX="${TRAIN_SAMPLE_PREFIX:-${PROJECT_ROOT}/output/simulated_samples/brca2021_pseudobulk_1M_alltrain_train}"
TEST_SAMPLE_PREFIX="${TEST_SAMPLE_PREFIX:-${PROJECT_ROOT}/output/simulated_samples/brca2021_pseudobulk_1M_alltrain_test}"

NUM_SAMPLES_OVERRIDE="${NUM_SAMPLES_OVERRIDE:-0}"
FORCE_REGENERATE_SAMPLES="${FORCE_REGENERATE_SAMPLES:-0}"

# ── Helpers ────────────────────────────────────────────────────────────────────

# Resolve the checkpoint with the highest training step in a glob pattern.
resolve_highest_step_checkpoint() {
  python - "$1" <<'PY'
import glob, os, re, sys

paths = glob.glob(sys.argv[1])
if not paths:
    raise SystemExit(f"No checkpoint found matching: {sys.argv[1]}")

def extract_step(path):
    for regex in (r"step=(\d+)\.pt$", r"model(\d+)\.pt$", r"_(\d+)\.pt$"):
        m = re.search(regex, os.path.basename(path))
        if m:
            return int(m.group(1))
    raise ValueError(f"Cannot extract step from: {path}")

print(max(paths, key=lambda p: (extract_step(p), os.path.getmtime(p))))
PY
}

read_sample_ids() {
  mapfile -t SAMPLE_IDS < <(sed '/^#/d;/^$/d' "$1")
}

# Decide whether to generate, skip, or regenerate an existing .npz file.
# Prints one of: generate | skip | regenerate
sample_file_action() {
  python - "$1" "${NUM_SAMPLES_OVERRIDE}" "${FORCE_REGENERATE_SAMPLES}" <<'PY'
import pathlib, sys
import numpy as np

path     = pathlib.Path(sys.argv[1])
override = int(sys.argv[2])
force    = int(sys.argv[3])

if force or not path.exists():
    print("generate"); raise SystemExit(0)

with np.load(path, allow_pickle=False) as f:
    saved = int(np.asarray(f["cell_gen"]).shape[0])
    real  = int(np.asarray(f["real_num_cells"]).reshape(-1)[0])

expected = override if override > 0 else real
print("skip" if saved == expected else "regenerate")
PY
}

generate_for_split() {
  local sample_ids_path="$1"
  local sample_prefix="$2"
  local split_label="$3"

  read_sample_ids "${sample_ids_path}"
  for sample_id in "${SAMPLE_IDS[@]}"; do
    local safe_id="${sample_id//[^[:alnum:]_-]/_}"
    local sample_file="${sample_prefix}_${safe_id}.npz"
    local action
    action="$(sample_file_action "${sample_file}")"

    if [[ "${action}" == "skip" ]]; then
      echo "[${split_label}] Skipping ${sample_id}: file already has the correct cell count"
      continue
    fi
    if [[ "${action}" == "regenerate" ]]; then
      echo "[${split_label}] Regenerating ${sample_id}: cell count mismatch"
      rm -f "${sample_file}"
    fi

    echo "[${split_label}] Generating ${sample_id}"
    python sample.py \
      --data_dir   "${DATA_DIR}" \
      --model_path "${MODEL_PATH}" \
      --sample_dir "${sample_prefix}" \
      --sample_id  "${sample_id}" \
      --sample_key "${SAMPLE_KEY}" \
      --batch_size "${SAMPLING_BATCH_SIZE}" \
      --num_samples "${NUM_SAMPLES_OVERRIDE}" \
      --cond_pseudobulk True \
      --cond_embed_dim "${COND_EMBED_DIM}" \
      --seed "${SEED}"
  done
}

# ── Step 1: Write split files ──────────────────────────────────────────────────
echo "Writing BRCA2021 subtype-balanced train/test split files"
python split_brca.py \
  --data_dir  "${DATA_DIR}" \
  --sample_key "${SAMPLE_KEY}" \
  --group_key  "${GROUP_KEY}" \
  --output_dir "${SPLIT_DIR}"

# ── Step 2: Fine-tune the VAE  ──────────
if compgen -G "${VAE_SAVE_DIR}/model_seed=${SEED}_step=*.pt" > /dev/null; then
  echo "VAE checkpoint found, skipping VAE training"
else
  echo "Training VAE on all BRCA2021 samples (train + held-out test)"
  python VAE/VAE_train.py \
    --data_dir      "${DATA_DIR}" \
    --num_genes     "${VAE_NUM_GENES}" \
    --save_dir      "${VAE_SAVE_DIR}" \
    --log_dir       "${VAE_LOG_DIR}" \
    --max_steps     "${VAE_MAX_STEPS}" \
    --checkpoint_freq "${VAE_CHECKPOINT_FREQ}" \
    --batch_size    "${VAE_BATCH_SIZE}" \
    --sample_key    "${SAMPLE_KEY}" \
    --state_dict    '/share/models/SCimilarity/annotation_model_v1' \
    --seed          "${SEED}"
fi

VAE_PATH="$(resolve_highest_step_checkpoint "${VAE_SAVE_DIR}/model_seed=${SEED}_step=*.pt")"
echo "Using VAE: ${VAE_PATH}"

# ── Step 3: Train the pseudobulk-conditioned diffusion backbone (train split only) ─
if compgen -G "${MODEL_DIR}/model*.pt" > /dev/null; then
  echo "Diffusion checkpoint found, skipping diffusion training"
else
  echo "Training diffusion backbone on BRCA2021 training samples"
  python train.py \
    --data_dir              "${DATA_DIR}" \
    --vae_path              "${VAE_PATH}" \
    --model_name            "${MODEL_NAME}" \
    --save_dir              "${BACKBONE_SAVE_DIR}" \
    --lr_anneal_steps       "${DIFFUSION_STEPS}" \
    --save_interval         "${DIFFUSION_SAVE_INTERVAL}" \
    --batch_size            "${DIFFUSION_BATCH_SIZE}" \
    --sample_key            "${SAMPLE_KEY}" \
    --include_sample_ids_path "${TRAIN_SAMPLE_IDS_PATH}" \
    --mmd_eval_validation_sample_ids_path "${TEST_SAMPLE_IDS_PATH}" \
    --cond_pseudobulk       True \
    --cond_embed_dim        "${COND_EMBED_DIM}" \
    --mmd_eval_interval     "${MMD_EVAL_INTERVAL}" \
    --mmd_eval_max_cells    "${MMD_EVAL_MAX_CELLS}" \
    --seed                  "${SEED}"
fi

MODEL_PATH="$(resolve_highest_step_checkpoint "${MODEL_DIR}/model*.pt")"
echo "Using model: ${MODEL_PATH}"

# ── Step 4: Generate samples ───────────────────────────────────────────────────
echo "FORCE_REGENERATE_SAMPLES=${FORCE_REGENERATE_SAMPLES}, NUM_SAMPLES_OVERRIDE=${NUM_SAMPLES_OVERRIDE}"

echo "Generating train-split samples (21 samples)"
generate_for_split "${TRAIN_SAMPLE_IDS_PATH}" "${TRAIN_SAMPLE_PREFIX}" train

echo "Generating test-split samples (5 samples)"
generate_for_split "${TEST_SAMPLE_IDS_PATH}" "${TEST_SAMPLE_PREFIX}" test

echo "BRCA pseudobulk pipeline complete"
echo "Evaluate with: notebooks/brca_cond_audit_mmd.ipynb"
