#!/bin/bash
#SBATCH --job-name=bepi_esm_gangxu
#SBATCH --output=./log/slurm_esm_gangxu_%j.out
#SBATCH --error=./log/slurm_esm_gangxu_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:4
#SBATCH --time=1440:00:00
#SBATCH --partition=8gpu        # modify to your partition name

# ── Environment ──────────────────────────────────────────────────────────────
source ~/.bashrc
conda activate bepi             # modify to your conda env name

cd /work/home/maben/project/epitope_prediction/GraphBepi

mkdir -p ./log ./model

# ── Train ESM GangXu base model ──────────────────────────────────────────────
# This model will be used as the reference checkpoint for DPO fine-tuning

DATASET="BCE_633"
TAG="GraphBepi_gangxu"
MODE="esm_gangxu"

echo "=============================================="
echo "Training ESM GangXu base model"
echo "Dataset   : ${DATASET}"
echo "Tag       : ${TAG}"
echo "Mode      : ${MODE}"
echo "=============================================="

python train.py \
    --mode      "${MODE}" \
    --dataset   "${DATASET}" \
    --tag       "${TAG}" \
    --hidden    256 \
    --batch     4 \
    --epochs    100 \
    --gpu       0 \
    --fold      -1 \
    --seed      2022 \
    --logger    ./log

echo "=============================================="
echo "Training completed!"
echo "Model saved to: ./model/${DATASET}_${TAG}/model_-1.ckpt"
echo "This checkpoint will be used for DPO fine-tuning"
echo "=============================================="
