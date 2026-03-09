#!/bin/bash
#SBATCH --job-name=bepi_ensemble
#SBATCH --output=./log/slurm_ensemble_%j.out
#SBATCH --error=./log/slurm_ensemble_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=120:00:00
#SBATCH --partition=8gpu         # modify to your partition name

# ── Environment ──────────────────────────────────────────────────────────────
source ~/.bashrc
conda activate epitope           # using epitope environment as requested

cd /work/home/maben/project/epitope_prediction/GraphBepi

mkdir -p ./log ./model/ensemble_results

# ── Ensemble Prediction ──────────────────────────────────────────────────────
DATASET="BCE_633"
SEEDS="2022 2023 2024 2025 2026"
TOP_K=3
ENSEMBLE_METHOD="mean"

echo "=============================================="
echo "Ensemble Prediction for OPUS-BEPI"
echo "Dataset        : ${DATASET}"
echo "Seeds          : ${SEEDS}"
echo "Top K models   : ${TOP_K}"
echo "Ensemble method: ${ENSEMBLE_METHOD}"
echo "=============================================="

python ensemble_predict.py \
    --dataset "${DATASET}" \
    --seeds ${SEEDS} \
    --top_k ${TOP_K} \
    --fold -1 \
    --gpu 0 \
    --batch 4 \
    --ensemble_method "${ENSEMBLE_METHOD}" \
    --output_dir ./model/ensemble_results

echo "=============================================="
echo "Ensemble prediction completed!"
echo "Results saved to: ./model/ensemble_results/"
echo "=============================================="
