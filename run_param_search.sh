#!/bin/bash
#SBATCH --job-name=bepi_param_search
#SBATCH --output=./log/slurm_%A_%a.out
#SBATCH --error=./log/slurm_%A_%a.err
#SBATCH --array=0-11           # 12 parameter combinations (0-indexed)
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
# Total = len(LRS) * len(HIDDENS) * len(DATASETS) = 3 * 2 * 2 = 12
LRS=(1e-4 5e-5 1e-5)
HIDDENS=(128 256)
DATASETS=(BCE_633 SARS2)       # modify to your actual dataset names

NUM_LRS=${#LRS[@]}
NUM_HIDDENS=${#HIDDENS[@]}
NUM_DATASETS=${#DATASETS[@]}

# Decode SLURM_ARRAY_TASK_ID -> (lr_idx, hidden_idx, dataset_idx)
TASK_ID=${SLURM_ARRAY_TASK_ID}
LR_IDX=$((TASK_ID / (NUM_HIDDENS * NUM_DATASETS)))
HIDDEN_IDX=$(( (TASK_ID / NUM_DATASETS) % NUM_HIDDENS ))
DATASET_IDX=$((TASK_ID % NUM_DATASETS))

LR=${LRS[$LR_IDX]}
HIDDEN=${HIDDENS[$HIDDEN_IDX]}
DATASET=${DATASETS[$DATASET_IDX]}

TAG="search_lr${LR}_h${HIDDEN}"

echo "=============================================="
echo "TASK_ID   : ${TASK_ID}"
echo "lr        : ${LR}"
echo "hidden    : ${HIDDEN}"
echo "dataset   : ${DATASET}"
echo "tag       : ${TAG}"
echo "=============================================="

# ── Training ─────────────────────────────────────────────────────────────────
python train.py \
    --mode    esm2_3b \
    --dataset "${DATASET}" \
    --lr      "${LR}" \
    --hidden  "${HIDDEN}" \
    --batch   4 \
    --epochs  300 \
    --gpu     0 \
    --fold    -1 \
    --seed    2022 \
    --tag     "${TAG}" \
    --logger  ./log
