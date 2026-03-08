#!/bin/bash
#SBATCH --job-name=bepi_dpo_search
#SBATCH --output=./log/slurm_dpo_%A_%a.out
#SBATCH --error=./log/slurm_dpo_%A_%a.err
#SBATCH --array=0-17           # 18 parameter combinations (0-indexed)
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

# ── Parameter grid ───────────────────────────────────────────────────────────
# DPO-specific hyperparameters
# Total = len(BETAS) * len(LAMBDAS) * len(LRS) * len(DATASETS) = 3 * 3 * 2 * 1 = 18
BETAS=(0.05 0.1 0.2)            # DPO KL penalty
LAMBDAS=(0.5 1.0 2.0)           # Task loss weight
LRS=(1e-5 5e-6)                 # Learning rate for DPO fine-tuning
DATASETS=(BCE_633)              # modify to your actual dataset names

NUM_BETAS=${#BETAS[@]}
NUM_LAMBDAS=${#LAMBDAS[@]}
NUM_LRS=${#LRS[@]}
NUM_DATASETS=${#DATASETS[@]}

# Decode SLURM_ARRAY_TASK_ID -> (beta_idx, lambda_idx, lr_idx, dataset_idx)
TASK_ID=${SLURM_ARRAY_TASK_ID}
BETA_IDX=$((TASK_ID / (NUM_LAMBDAS * NUM_LRS * NUM_DATASETS)))
LAMBDA_IDX=$(( (TASK_ID / (NUM_LRS * NUM_DATASETS)) % NUM_LAMBDAS ))
LR_IDX=$(( (TASK_ID / NUM_DATASETS) % NUM_LRS ))
DATASET_IDX=$((TASK_ID % NUM_DATASETS))

BETA=${BETAS[$BETA_IDX]}
LAMBDA=${LAMBDAS[$LAMBDA_IDX]}
LR=${LRS[$LR_IDX]}
DATASET=${DATASETS[$DATASET_IDX]}

TAG="dpo_gangxu_beta${BETA}_lambda${LAMBDA}_lr${LR}"

# Reference checkpoint path (pre-trained model)
# Modify this to point to your pre-trained esm_gangxu model checkpoint
REF_CKPT="./model/${DATASET}_GraphBepi_gangxu/model_-1.ckpt"

echo "=============================================="
echo "TASK_ID   : ${TASK_ID}"
echo "beta      : ${BETA}"
echo "lambda    : ${LAMBDA}"
echo "lr        : ${LR}"
echo "dataset   : ${DATASET}"
echo "tag       : ${TAG}"
echo "ref_ckpt  : ${REF_CKPT}"
echo "=============================================="

# ── DPO Fine-tuning ──────────────────────────────────────────────────────────
python train_dpo_gangxu.py \
    --ref_ckpt  "${REF_CKPT}" \
    --dataset   "${DATASET}" \
    --tag       "${TAG}" \
    --beta      "${BETA}" \
    --lambda_task "${LAMBDA}" \
    --lr        "${LR}" \
    --feat_dim  2581 \
    --hidden    256 \
    --batch     4 \
    --epochs    50 \
    --gpu       0 \
    --fold      -1 \
    --seed      2022 \
    --logger    ./log
