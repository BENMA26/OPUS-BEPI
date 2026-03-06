#!/bin/bash
#SBATCH --job-name=bepi_loss_cmp
#SBATCH --output=./log/loss_cmp_%A_%a.out
#SBATCH --error=./log/loss_cmp_%A_%a.err
#SBATCH --array=0-4              # 5 loss functions (0=bce, 1=afl, 2=bce_dice, 3=smooth_bce, 4=pu)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu          # modify to your partition name

# ── Environment ───────────────────────────────────────────────────────────────
source ~/.bashrc
conda activate bepi              # modify to your conda env name

cd /work/home/maben/project/epitope_prediction/GraphBepi

mkdir -p ./log ./model

# ── Loss configurations ───────────────────────────────────────────────────────
#
# Index | --loss      | Key hyperparams
# ------+-------------+----------------------------------------------------------
#   0   | bce         | baseline (BCELoss)
#   1   | afl         | gamma_pos=0, gamma_neg=2, clip=0.05
#   2   | bce_dice    | dice_alpha=0.5 (equal BCE + Dice weighting)
#   3   | smooth_bce  | smooth_eps=0.05 (positive-side label smoothing)
#   4   | pu          | pu_prior=0.15 (estimated epitope fraction)
#
LOSS_TYPES=(bce afl bce_dice smooth_bce pu)
LOSS_EXTRA_ARGS=(
    ""
    "--afl_gamma_pos 0.0 --afl_gamma_neg 2.0 --afl_clip 0.05"
    "--dice_alpha 0.5"
    "--smooth_eps 0.05"
    "--pu_prior 0.15"
)

LOSS=${LOSS_TYPES[${SLURM_ARRAY_TASK_ID}]}
EXTRA=${LOSS_EXTRA_ARGS[${SLURM_ARRAY_TASK_ID}]}
TAG="loss_${LOSS}"

echo "=============================================="
echo "TASK_ID   : ${SLURM_ARRAY_TASK_ID}"
echo "loss      : ${LOSS}"
echo "extra     : ${EXTRA}"
echo "tag       : ${TAG}"
echo "=============================================="

# ── Training ──────────────────────────────────────────────────────────────────
python train.py \
    --mode    esm2_3b \
    --dataset BCE_633 \
    --lr      1e-6 \
    --hidden  256 \
    --batch   4 \
    --epochs  300 \
    --gpu     0 \
    --fold    -1 \
    --seed    2022 \
    --tag     "${TAG}" \
    --logger  ./log \
    --loss    "${LOSS}" \
    ${EXTRA}
