# OPUS-BEPI

**B-Cell Epitope Prediction via Graph Neural Networks with Optional DPO Spatial Refinement**

OPUS-BEPI predicts B-cell epitopes (the surface regions of antigens that antibodies bind to) at the residue level. It combines ESM-2 sequence representations, DSSP structural features, and an edge-aware graph attention network (EGAT) with a bidirectional LSTM to capture both local structural context and long-range sequential dependencies.

An optional **DPO (Direct Preference Optimization)** fine-tuning stage further encourages the model to produce spatially coherent predictions — rewarding epitope sets that form contiguous surface patches over scattered, biologically implausible ones.

---

## Table of Contents

- [Architecture](#architecture)
- [File Structure](#file-structure)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Training](#training)
  - [Standard Training](#standard-training)
  - [DPO Fine-tuning](#dpo-fine-tuning)
- [Inference](#inference)
- [Feature Modes](#feature-modes)

---

## Architecture

```
Input protein (PDB / FASTA)
        │
        ▼
  ┌─────────────────────────────────────────────────────────┐
  │  Feature Extraction                                     │
  │   • ESM-2 residue embeddings  (seq-level)               │
  │   • DSSP features (φ/ψ/RSA/SS)  (struct-level)         │
  │   • Optional: FoldSeek tokens, SaProt, ESM-IF, etc.     │
  └──────────────────────────┬──────────────────────────────┘
                             │ concat → feat vector per residue
                             ▼
  ┌─────────────────────────────────────────────────────────┐
  │  GraphBepi                                              │
  │                                                         │
  │   W_v ──► BiLSTM₁ ──────────────────────┐              │
  │                                          ▼              │
  │   W_u ──► BiLSTM₂ ──────────────────── cat ──► MLP ──► p_i  │
  │                                          ▲              │
  │   edge_linear ──► EGAT (graph attn) ────┘              │
  └─────────────────────────────────────────────────────────┘
                             │
                             ▼
              per-residue epitope probability p_i ∈ (0, 1)
```

**EGAT** (Edge-aware Graph Attention) builds a protein contact graph from Cα distances and sequence distance, then propagates structural neighbourhood information across residues.

**DPO stage** (optional):

```
Frozen reference model (π_ref)  ──┐
                                   ├──► DPO loss  ──► fine-tune π_θ
Trainable policy model (π_θ)   ──┘
        │
  y_w = GT labels (spatially coherent surface patch)
  y_l = scrambled labels (same count, spatially dispersed)

  L = -log σ( β·[(log π_θ(y_w|x) - log π_ref(y_w|x))
                - (log π_θ(y_l|x) - log π_ref(y_l|x))] )
    + λ · BCE(p_θ, y_w)
```

---

## File Structure

```
OPUS-BEPI/
│
├── train.py              # Unified training entry point (all feature modes)
├── train_dpo.py          # DPO fine-tuning with spatial coherence preference
├── train_utils.py        # Shared utilities: seed_everything, arg parser, run_training
├── test.py               # Inference on PDB / FASTA inputs
│
├── model.py              # GraphBepi, GraphBepi_att (inherit BaseLightningModel)
├── EGAT.py               # Edge-aware Graph Attention layer + AE encoder
│
├── dataset.py            # BasePDB + 12 dataset variants + PDB_DPO
├── utils.py              # chain class, collate_fn, extract_chain, initial
├── preprocess.py         # PDB parsing, DSSP processing
├── graph_construction.py # Protein contact graph construction
│
├── dpo.py                # DPO loss, sequence_log_prob
├── spatial_utils.py      # Spatial adjacency, coherence metric, y_l construction
│
└── tool.py               # METRICS (AUROC, AUPRC, MCC, F1, BACC, …)
```

---

## Installation

```bash
# Core dependencies
pip install torch pytorch-lightning torchmetrics
pip install fair-esm biopython tqdm pandas numpy

# DSSP binary (place mkdssp in ./mkdssp/)
# Download from: https://github.com/PDB-REDO/dssp
```

---

## Data Preparation

Expected directory layout for each dataset:

```
data/<DATASET_NAME>/
├── total.csv               # columns: index=<PDB_ID>_<CHAIN>, "Epitopes (resi_resn)"
├── PDB/                    # raw PDB files
├── purePDB/                # per-chain PDB files
├── feat/                   # ESM-2 embeddings  (*_esm2.ts)
├── dssp/                   # DSSP features     (*.npy, *_pos.npy)
├── graph/                  # contact graphs    (*.graph)
├── train.pkl               # list of chain objects (training set)
├── test.pkl                # list of chain objects (test set)
└── cross-validation.npy    # sorted indices for 10-fold CV
```

**Build from scratch:**

```bash
# Extracts chains, runs ESM-2, DSSP, and graph construction, then splits by date
python dataset.py --root ./data/BCE_633 --gpu 0
```

---

## Training

### Standard Training

All feature configurations are unified in a single entry point via `--mode`:

```bash
python train.py --mode <MODE> [options]
```

| `--mode` | Feature stack | `feat_dim` | Early stop |
|---|---|---|---|
| `esm2_3b` | ESM-2 (3B) | 2560 | ✗ |
| `esm2_650m` | ESM-2 (650M) | 1280 | ✓ |
| `esm2_3b_es` | ESM-2 (3B) + early stop | 2560 | ✓ |
| `esm_t` | ESM-2 (3B) + ESM-T token | 3200 | ✗ |
| `saport` | SaProt embeddings | 446 | ✗ |
| `foldseek_multi` | ESM-2 + FoldSeek (6 tokens) | 2686 | ✓ |
| `foldseek_single` | ESM-2 + FoldSeek local+global | 2602 | ✗ |
| `foldseek_attn` | ESM-2 + FoldSeek attention | 2581 | ✓ |
| `esm_if` | ESM-2 + ESM-IF | 3072 | ✗ |
| `esm_if_foldseek` | ESM-2 + ESM-IF + FoldSeek | 3094 | ✗ |
| `structure` | Structure diffusion features | 640 | ✗ |
| `esm2_structure` | ESM-2 + structure diffusion | 3200 | ✗ |
| `esm2_gangxu` | ESM-2 (gangxu variant) | 2560 | ✗ |
| `saport_gangxu` | SaProt (gangxu variant) | 446 | ✗ |

**Common options:**

```
--dataset   BCE_633      dataset directory name under ./data/
--tag       GraphBepi    experiment name (checkpoint saved to ./model/<dataset>_<tag>/)
--fold      -1           cross-validation fold; -1 = use full training set
--gpu       0            GPU id; -1 = CPU
--lr        1e-6         learning rate
--hidden    256          hidden dimension
--batch     4            batch size
--epochs    300          max epochs
--logger    ./log        TensorBoard log directory
```

**Examples:**

```bash
# Full training on BCE_633 with ESM-2 3B features
python train.py --mode esm2_3b --dataset BCE_633 --tag GraphBepi --gpu 0

# 10-fold cross-validation (fold 0)
python train.py --mode esm2_3b --dataset BCE_633 --fold 0 --gpu 0

# Structure diffusion features (requires --sub_dir)
python train.py --mode structure --dataset BCE_633 --sub_dir feat_esm --gpu 0
```

---

### DPO Fine-tuning

DPO refines a pre-trained GraphBepi to favour spatially coherent epitope predictions.

**How it works:**
- **y_w** (preferred): ground-truth labels, which naturally form contiguous surface patches.
- **y_l** (dispreferred): labels with the same number of positives but placed randomly on surface-exposed residues (RSA > 0.15), destroying spatial coherence.
- The DPO objective shifts the policy toward y_w without collapsing the model's original discriminative capability (controlled by `--lambda_task`).

Because GraphBepi is encoder-only, the sequence log-probability factorises over independent residues:

$$\log \pi(y \mid x) = \frac{1}{N} \sum_i \left[ y_i \log p_i + (1 - y_i) \log(1 - p_i) \right]$$

This means **one forward pass** per model suffices to evaluate both $\log\pi(y_w|x)$ and $\log\pi(y_l|x)$.

**Usage:**

```bash
# Step 1 — train reference model
python train.py --mode esm2_3b --dataset BCE_633 --tag GraphBepi --fold -1 --gpu 0

# Step 2 — DPO fine-tuning
python train_dpo.py \
    --ref_ckpt  ./model/BCE_633_GraphBepi/model_-1.ckpt \
    --dataset   BCE_633 \
    --tag       GraphBepi_dpo \
    --gpu       0 \
    --fold      -1 \
    --epochs    50 \
    --beta      0.1 \
    --lambda_task 1.0
```

**DPO-specific options:**

| Option | Default | Description |
|---|---|---|
| `--ref_ckpt` | (required) | Checkpoint of the pre-trained reference model |
| `--beta` | `0.1` | KL penalty; higher = stay closer to reference |
| `--lambda_task` | `1.0` | Weight of the BCE task loss; prevents DPO from hurting accuracy |
| `--feat_dim` | `2560` | Must match the checkpoint's input dimension |

**Hyperparameter guidance:**
- `--beta`: start with `0.05–0.2`; lower values allow more deviation from the reference.
- `--lambda_task`: `0.5–2.0`; set higher if validation AUPRC drops during DPO training.
- Spatial distance threshold (in `spatial_utils.py`): default **8 Å**; adjust based on the typical epitope–paratope interface geometry.

---

## Inference

Predict epitopes from a PDB file or FASTA sequences:

```bash
# From a single PDB chain
python test.py -p -i 6OGE_A.pdb -o ./output --gpu 0

# From FASTA (runs ESMfold for structure prediction first)
python test.py -f -i sequences.fasta -o ./output --gpu 0
```

Output: one CSV per chain with columns `resn`, `score`, `is epitope`.

---

## Feature Modes — Technical Notes

**FoldSeek tokens** encode the local structural alphabet of each residue using TM-align-based 3Di states. The `foldseek_attn` mode uses `GraphBepi_att`, which applies a learnable self-attention layer over the 6-neighbourhood token vectors before concatenating with ESM-2 features.

**ESM-IF** provides structure-conditioned inverse-folding embeddings (512-dim per residue), offering complementary information to sequence-based ESM-2 features.

**Structure diffusion** features (640-dim) are extracted from a structure diffusion model, encoding global structural context beyond local DSSP secondary structure.

All variants share the same `GraphBepi` backbone and differ only in the input `feat_dim` — making it straightforward to add new feature types by subclassing `BasePDB` in `dataset.py`.
