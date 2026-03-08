#!/bin/bash
#SBATCH --job-name=bepi_finetune_search
#SBATCH --output=./log/slurm_finetune_%A_%a.out
#SBATCH --error=./log/slurm_finetune_%A_%a.err
#SBATCH --array=0-11           # 12 parameter combinations (0-indexed)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:4
#SBATCH --time=1440:00:00
#SBATCH --partition=8gpu        # modify to your partition name

# ══════════════════════════════════════════════════════════════════════════════
# FINE-TUNING HYPERPARAMETER SEARCH
# ══════════════════════════════════════════════════════════════════════════════
# Search over key fine-tuning hyperparameters:
#   - Learning rate: {1e-5, 5e-6, 1e-6}
#   - Freeze encoder: {True, False}
#   - Epochs: {30, 50}
#
# Total combinations: 3 * 2 * 2 = 12
# ══════════════════════════════════════════════════════════════════════════════

# ── Environment ──────────────────────────────────────────────────────────────
source ~/.bashrc
conda activate bepi             # modify to your conda env name

cd /work/home/maben/project/epitope_prediction/GraphBepi

mkdir -p ./log ./model

# ── Parameter grid ───────────────────────────────────────────────────────────
LRS=(1e-5 5e-6 1e-6)            # Fine-tuning learning rates
FREEZE_OPTIONS=(0 1)            # 0=train all, 1=freeze encoder
EPOCHS_OPTIONS=(30 50)          # Number of fine-tuning epochs

NUM_LRS=${#LRS[@]}
NUM_FREEZE=${#FREEZE_OPTIONS[@]}
NUM_EPOCHS=${#EPOCHS_OPTIONS[@]}

# Decode SLURM_ARRAY_TASK_ID -> (lr_idx, freeze_idx, epochs_idx)
TASK_ID=${SLURM_ARRAY_TASK_ID}
LR_IDX=$((TASK_ID / (NUM_FREEZE * NUM_EPOCHS)))
FREEZE_IDX=$(( (TASK_ID / NUM_EPOCHS) % NUM_FREEZE ))
EPOCHS_IDX=$((TASK_ID % NUM_EPOCHS))

LR=${LRS[$LR_IDX]}
FREEZE=${FREEZE_OPTIONS[$FREEZE_IDX]}
EPOCHS=${EPOCHS_OPTIONS[$EPOCHS_IDX]}

# Configuration
DATASET="BCE_633"
MODE="esm_gangxu"
PRETRAIN_DATASET="Dockground_5K"
PRETRAIN_TAG="GraphBepi_pretrain"
PRETRAIN_CKPT="./model/${PRETRAIN_DATASET}_${PRETRAIN_TAG}/model_-1.ckpt"

# Build tag based on parameters
if [ ${FREEZE} -eq 1 ]; then
    TAG="finetune_lr${LR}_frozen_ep${EPOCHS}"
    FREEZE_FLAG="--freeze_encoder"
else
    TAG="finetune_lr${LR}_full_ep${EPOCHS}"
    FREEZE_FLAG=""
fi

# Check if pre-trained checkpoint exists
if [ ! -f "${PRETRAIN_CKPT}" ]; then
    echo "ERROR: Pre-trained checkpoint not found: ${PRETRAIN_CKPT}"
    echo "Please run Stage 1 first: sbatch run_pretrain_stage.sh"
    exit 1
fi

echo "=============================================="
echo "TASK_ID       : ${TASK_ID}"
echo "Learning rate : ${LR}"
echo "Freeze encoder: ${FREEZE}"
echo "Epochs        : ${EPOCHS}"
echo "Tag           : ${TAG}"
echo "Pretrain ckpt : ${PRETRAIN_CKPT}"
echo "=============================================="

# ── Fine-tuning ──────────────────────────────────────────────────────────────
python train_pretrain_finetune.py \
    --stage         finetune \
    --mode          "${MODE}" \
    --dataset       "${DATASET}" \
    --tag           "${TAG}" \
    --pretrain_ckpt "${PRETRAIN_CKPT}" \
    ${FREEZE_FLAG} \
    --hidden        256 \
    --batch         4 \
    --epochs        ${EPOCHS} \
    --lr            ${LR} \
    --gpu           0 \
    --fold          -1 \
    --seed          2022 \
    --logger        ./log

echo "=============================================="
echo "Fine-tuning completed for TASK_ID ${TASK_ID}"
echo "=============================================="
