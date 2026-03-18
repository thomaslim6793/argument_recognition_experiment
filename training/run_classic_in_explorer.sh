#!/bin/bash
#SBATCH --job-name=train_classic_re
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=logs/classic_re_%j.out
#SBATCH --error=logs/classic_re_%j.err

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
CONDA_ENV="${CONDA_ENV:-indra}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/drugprot_dual}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs}"
MODEL_NAME="${MODEL_NAME:-microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext}"
SEED="${SEED:-1337}"

# Optional OOD file (leave empty to skip OOD eval):
CLASSIC_OOD_FILE="${CLASSIC_OOD_FILE:-}"
PUSH_TO_HUB="${PUSH_TO_HUB:-0}"
CLASSIC_HF_REPO="${CLASSIC_HF_REPO:-thomaslim6793/classic_drugprot}"
HF_PRIVATE="${HF_PRIVATE:-0}"

mkdir -p "${PROJECT_ROOT}/logs" "${OUTPUT_ROOT}"

cd "${PROJECT_ROOT}"
echo "Running from: $(pwd)"

source ~/.bashrc
conda activate "${CONDA_ENV}"

echo "Environment:"
echo "  PROJECT_ROOT=${PROJECT_ROOT}"
echo "  DATA_DIR=${DATA_DIR}"
echo "  OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "  MODEL_NAME=${MODEL_NAME}"
echo "  SEED=${SEED}"
echo

COMMON_ARGS=(
  --data_dir "${DATA_DIR}"
  --model_name "${MODEL_NAME}"
  --seed "${SEED}"
)

CLASSIC_ARGS=(
  --output_dir "${OUTPUT_ROOT}/classic_re"
  --num_train_epochs 5
  --learning_rate 2e-5
  --per_device_train_batch_size 16
  --per_device_eval_batch_size 32
  --max_length 256
  --bf16
)

if [[ -n "${CLASSIC_OOD_FILE}" ]]; then
  CLASSIC_ARGS+=(--ood_file "${CLASSIC_OOD_FILE}")
fi
if [[ "${PUSH_TO_HUB}" == "1" ]]; then
  CLASSIC_ARGS+=(--push_to_hub --hub_model_id "${CLASSIC_HF_REPO}")
  if [[ "${HF_PRIVATE}" == "1" ]]; then
    CLASSIC_ARGS+=(--hub_private)
  fi
fi

echo "=== Training classic RE ==="
python training/train_classic_re.py "${COMMON_ARGS[@]}" "${CLASSIC_ARGS[@]}"

echo "Classic RE training finished."
