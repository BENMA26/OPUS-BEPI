#!/bin/bash
#SBATCH --job-name=bepi_finetune
#SBATCH --output=./log/slurm_finetune_%j.out
#SBATCH --error=./log/slurm_finetune_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:4
#SBATCH --time=1440:00:00
#SBATCH --partition=8gpu        # modify to your partition name

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2: FINE-TUNING ON B-CELL EPITOPES
# ══════════════════════════════════════════════════════════════════════════════
# This stage fine-tunes the pre-trained binding site model on epitope data.
# The model has already learned general binding patterns from protein-protein
# interfaces, and now specializes to antibody-antigen epitopes.
#
# Key hyperparameters to tune:
#   - Learning rate: typically lower than pre-training (1e-5 vs 1e-4)
#   - Freeze encoder: optionally freeze feature extraction layers
#   - Epochs: fewer epochs needed (50 vs 100)
# ══════════════════════════════════════════════════════════════════════════════

# ── Environment ──────────────────────────────────────────────────────────────
source ~/.bashrc
conda activate bepi             # modify to your conda env name

cd /work/home/maben/project/epitope_prediction/GraphBepi

mkdir -p ./log ./model

# ── Configuration ────────────────────────────────────────────────────────────
DATASET="BCE_633"               # epitope dataset
TAG="GraphBepi_finetune"
MODE="esm_gangxu"

# Path to pre-trained checkpoint from Stage 1
PRETRAIN_DATASET="Dockground_5K"
PRETRAIN_TAG="GraphBepi_pretrain"
PRETRAIN_CKPT="./model/${PRETRAIN_DATASET}_${PRETRAIN_TAG}/model_-1.ckpt"

# Check if pre-trained checkpoint exists
if [ ! -f "${PRETRAIN_CKPT}" ]; then
    echo "ERROR: Pre-trained checkpoint not found: ${PRETRAIN_CKPT}"
    echo "Please run Stage 1 first: sbatch run_pretrain_stage.sh"
    exit 1
fi

echo "=============================================="
echo "STAGE 2: FINE-TUNING ON EPITOPES"
echo "=============================================="
echo "Dataset       : ${DATASET}"
echo "Tag           : ${TAG}"
echo "Mode          : ${MODE}"
echo "Pretrain ckpt : ${PRETRAIN_CKPT}"
echo "Purpose       : Specialize to epitope prediction"
echo "=============================================="

# ── Fine-tuning ──────────────────────────────────────────────────────────────
# Option 1: Fine-tune all layers (recommended)
python train_pretrain_finetune.py \
    --stage         finetune \
    --mode          "${MODE}" \
    --dataset       "${DATASET}" \
    --tag           "${TAG}" \
    --pretrain_ckpt "${PRETRAIN_CKPT}" \
    --hidden        256 \
    --batch         4 \
    --epochs        50 \
    --lr            1e-5 \
    --gpu           0 \
    --fold          -1 \
    --seed          2022 \
    --logger        ./log

# Option 2: Freeze encoder, only train classifier head (uncomment to use)
# python train_pretrain_finetune.py \
#     --stage         finetune \
#     --mode          "${MODE}" \
#     --dataset       "${DATASET}" \
#     --tag           "${TAG}_frozen" \
#     --pretrain_ckpt "${PRETRAIN_CKPT}" \
#     --freeze_encoder \
#     --hidden        256 \
#     --batch         4 \
#     --epochs        50 \
#     --lr            1e-4 \
#     --gpu           0 \
#     --fold          -1 \
#     --seed          2022 \
#     --logger        ./log

echo "=============================================="
echo "Fine-tuning completed!"
echo "Checkpoint: ./model/${DATASET}_${TAG}/model_-1.ckpt"
echo "=============================================="
