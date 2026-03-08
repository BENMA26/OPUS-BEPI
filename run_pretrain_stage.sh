#!/bin/bash
#SBATCH --job-name=bepi_pretrain
#SBATCH --output=./log/slurm_pretrain_%j.out
#SBATCH --error=./log/slurm_pretrain_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:4
#SBATCH --time=1440:00:00
#SBATCH --partition=8gpu        # modify to your partition name

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1: PRE-TRAINING ON PROTEIN-PROTEIN BINDING SITES
# ══════════════════════════════════════════════════════════════════════════════
# This stage trains the model on a large-scale protein-protein binding site
# dataset (e.g., Dockground, PDB complexes) to learn general binding patterns.
# The trained checkpoint will be used as initialization for epitope fine-tuning.
#
# Inspired by ScanNet's two-stage training strategy:
# 1. Pre-train on ~20K protein-protein interfaces (Dockground)
# 2. Fine-tune on B-cell epitopes (smaller, specialized dataset)
# ══════════════════════════════════════════════════════════════════════════════

# ── Environment ──────────────────────────────────────────────────────────────
source ~/.bashrc
conda activate bepi             # modify to your conda env name

cd /work/home/maben/project/epitope_prediction/GraphBepi

mkdir -p ./log ./model

# ── Configuration ────────────────────────────────────────────────────────────
# IMPORTANT: You need to prepare a protein-protein binding site dataset
# Dataset options:
#   - Dockground: Large-scale protein-protein docking benchmark
#   - PDB_interfaces: Extracted from PDB complexes
#   - Custom dataset following the same format as BCE_633

DATASET="Dockground_5K"         # modify to your binding site dataset name
TAG="GraphBepi_pretrain"
MODE="esm_gangxu"               # use same feature mode as final model

echo "=============================================="
echo "STAGE 1: PRE-TRAINING ON BINDING SITES"
echo "=============================================="
echo "Dataset   : ${DATASET}"
echo "Tag       : ${TAG}"
echo "Mode      : ${MODE}"
echo "Purpose   : Learn general binding site patterns"
echo "=============================================="

# ── Pre-training ─────────────────────────────────────────────────────────────
python train_pretrain_finetune.py \
    --stage     pretrain \
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
echo "Pre-training completed!"
echo "Checkpoint: ./model/${DATASET}_${TAG}/model_-1.ckpt"
echo ""
echo "Next step: Run fine-tuning stage"
echo "  sbatch run_finetune_stage.sh"
echo "=============================================="
