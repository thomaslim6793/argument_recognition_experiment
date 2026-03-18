#!/bin/bash
#SBATCH --job-name=train_span_re
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=logs/span_re_%j.out
#SBATCH --error=logs/span_re_%j.err

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
CONDA_ENV="${CONDA_ENV:-indra}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/drugprot_dual}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs}"
MODEL_NAME="${MODEL_NAME:-microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext}"
SEED="${SEED:-1337}"

# Optional OOD file (leave empty to skip OOD eval):
SPAN_OOD_FILE="${SPAN_OOD_FILE:-}"
PUSH_TO_HUB="${PUSH_TO_HUB:-0}"
SPAN_HF_REPO="${SPAN_HF_REPO:-thomaslim6793/span_drugprot}"
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

SPAN_ARGS=(
  --output_dir "${OUTPUT_ROOT}/span_re"
  --num_train_epochs 5
  --learning_rate 2e-5
  --per_device_train_batch_size 16
  --per_device_eval_batch_size 32
  --max_length 256
  --bf16
)

if [[ -n "${SPAN_OOD_FILE}" ]]; then
  SPAN_ARGS+=(--ood_file "${SPAN_OOD_FILE}")
fi
if [[ "${PUSH_TO_HUB}" == "1" ]]; then
  SPAN_ARGS+=(--push_to_hub --hub_model_id "${SPAN_HF_REPO}")
  if [[ "${HF_PRIVATE}" == "1" ]]; then
    SPAN_ARGS+=(--hub_private)
  fi
fi

echo "=== Training span RE ==="
python training/train_span_re.py "${COMMON_ARGS[@]}" "${SPAN_ARGS[@]}"

echo "Span RE training finished."
