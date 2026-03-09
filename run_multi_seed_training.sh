#!/bin/bash
#SBATCH --job-name=bepi_multi_seed
#SBATCH --output=./log/slurm_multi_seed_%A_%a.out
#SBATCH --error=./log/slurm_multi_seed_%A_%a.err
#SBATCH --array=0-4              # 5 random seeds (0-indexed)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:4
#SBATCH --time=1440:00:00
#SBATCH --partition=8gpu         # modify to your partition name

# ── Environment ──────────────────────────────────────────────────────────────
source ~/.bashrc
conda activate epitope           # using epitope environment as requested

cd /work/home/maben/project/epitope_prediction/GraphBepi

mkdir -p ./log ./model

# ── Random Seeds ─────────────────────────────────────────────────────────────
SEEDS=(2022 2023 2024 2025 2026)

SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

DATASET="BCE_633"
TAG="GraphBepi_seed${SEED}"
MODE="esm_gangxu"

echo "=============================================="
echo "Training OPUS-BEPI with random seed"
echo "TASK_ID   : ${SLURM_ARRAY_TASK_ID}"
echo "Seed      : ${SEED}"
echo "Dataset   : ${DATASET}"
echo "Tag       : ${TAG}"
echo "Mode      : ${MODE}"
echo "=============================================="

# ── Train base OPUS-BEPI model ───────────────────────────────────────────────
python train.py \
    --mode      "${MODE}" \
    --dataset   "${DATASET}" \
    --tag       "${TAG}" \
    --hidden    256 \
    --batch     4 \
    --epochs    100 \
    --gpu       0 \
    --fold      -1 \
    --seed      "${SEED}" \
    --logger    ./log

echo "=============================================="
echo "Training completed for seed ${SEED}!"
echo "Model saved to: ./model/${DATASET}_${TAG}/model_-1.ckpt"
echo "=============================================="
