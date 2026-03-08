"""
Two-stage training workflow inspired by ScanNet:
Stage 1: Pre-train on protein-protein binding sites (large dataset)
Stage 2: Fine-tune on B-cell epitopes (target task)

This approach follows ScanNet's strategy of learning general binding site patterns
first, then specializing to epitope prediction.

Usage
-----
Stage 1 (Pre-training on binding sites):
    python train_pretrain_finetune.py \
        --stage pretrain \
        --dataset Dockground_5K \
        --mode esm_gangxu \
        --tag GraphBepi_pretrain \
        --epochs 100 \
        --gpu 0

Stage 2 (Fine-tuning on epitopes):
    python train_pretrain_finetune.py \
        --stage finetune \
        --dataset BCE_633 \
        --mode esm_gangxu \
        --tag GraphBepi_finetune \
        --pretrain_ckpt ./model/Dockground_5K_GraphBepi_pretrain/model_-1.ckpt \
        --epochs 50 \
        --gpu 0 \
        --lr 1e-5
"""
import warnings
warnings.simplefilter('ignore')
from tool import METRICS
from model import GraphBepi, GraphBepi_att
from dataset import (
    PDB, PDB_esm, PDB_token_esm, PDB_saport,
    PDB_foldseek, PDB_foldseek_local_golbal, PDB_foldseek_attn,
    PDB_esm_if, PDB_esm_if_foldseek_tokens, PDB_foldseek_tokens,
    PDB_structure, PDB_esm_structure,
    collate_fn, collate_fn_fold_tokens,
)
from train_utils import seed_everything, build_arg_parser, build_loss_fn, run_training
import argparse

# Registry: mode -> (dataset_class, feat_dim, model_class, collate_fn, use_early_stop, needs_sub_dir)
CONFIGS = {
    'esm2_3b':        (PDB,                          2560,        GraphBepi,     collate_fn,             False, False),
    'esm2_650m':      (PDB,                          1280,        GraphBepi,     collate_fn,             True,  False),
    'esm2_3b_es':     (PDB,                          2560,        GraphBepi,     collate_fn,             True,  False),
    'esm_t':          (PDB_token_esm,                2560+640,    GraphBepi,     collate_fn,             False, False),
    'saport':         (PDB_saport,                   446,         GraphBepi,     collate_fn,             False, False),
    'esm2_gangxu':    (PDB_esm,                      2560,        GraphBepi,     collate_fn,             False, False),
    'esm_gangxu':     (PDB_foldseek_tokens,          2581,        GraphBepi,     collate_fn,             False, False),
    'structure':      (PDB_structure,                640,         GraphBepi,     collate_fn,             False, True),
    'esm2_structure': (PDB_esm_structure,            2560+640,    GraphBepi,     collate_fn,             False, True),
    'saport_gangxu':  (PDB_saport,                   446,         GraphBepi,     collate_fn,             False, False),
    'foldseek_multi': (PDB_foldseek,                 2686,        GraphBepi,     collate_fn,             True,  False),
    'foldseek_single':(PDB_foldseek_local_golbal,    2581+21,     GraphBepi,     collate_fn,             False, False),
    'foldseek_attn':  (PDB_foldseek_attn,            2581,        GraphBepi_att, collate_fn_fold_tokens, True,  False),
    'esm_if':         (PDB_esm_if,                   2560+512,    GraphBepi,     collate_fn,             False, False),
    'esm_if_foldseek':(PDB_esm_if_foldseek_tokens,   2560+512+21, GraphBepi,     collate_fn,             False, False),
}

parser = build_arg_parser()
parser.add_argument('--stage', type=str, required=True, choices=['pretrain', 'finetune'],
                    help='Training stage: pretrain (binding sites) or finetune (epitopes)')
parser.add_argument('--mode', type=str, required=True, choices=CONFIGS,
                    help='Feature configuration to use.')
parser.add_argument('--pretrain_ckpt', type=str, default=None,
                    help='Path to pretrained checkpoint (required for finetune stage)')
parser.add_argument('--freeze_encoder', action='store_true',
                    help='Freeze encoder layers during fine-tuning (only train classifier head)')
parser.add_argument('--sub_dir', type=str, default='BCE_633',
                    help='Structure feature sub-directory (used by structure/esm2_structure modes).')
args = parser.parse_args()

# Validation
if args.stage == 'finetune' and args.pretrain_ckpt is None:
    raise ValueError("--pretrain_ckpt is required for finetune stage")

seed_everything(args.seed)
loss_fn = build_loss_fn(args)

dataset_cls, feat_dim, model_cls, cfn, use_es, needs_sub_dir = CONFIGS[args.mode]
device   = 'cpu' if args.gpu == -1 else f'cuda:{args.gpu}'
root     = f'./data/{args.dataset}'
log_name = f'{args.dataset}_{args.tag}'

ds_kwargs = {'sub_dir': args.sub_dir} if needs_sub_dir else {}
trainset = dataset_cls(mode='train', fold=args.fold, root=root, **ds_kwargs)
valset   = dataset_cls(mode='val',   fold=args.fold, root=root, **ds_kwargs)
testset  = dataset_cls(mode='test',  fold=args.fold, root=root, **ds_kwargs)

# ── Stage-specific configuration ──────────────────────────────────────────────
if args.stage == 'pretrain':
    print("=" * 80)
    print("STAGE 1: PRE-TRAINING ON PROTEIN-PROTEIN BINDING SITES")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Mode: {args.mode}")
    print(f"Feature dim: {feat_dim}")
    print(f"Training from scratch...")
    print("=" * 80)

    model = model_cls(
        feat_dim=feat_dim, hidden_dim=args.hidden, exfeat_dim=13, edge_dim=51,
        augment_eps=0.05, dropout=0.2, lr=args.lr,
        metrics=METRICS(device), result_path=f'./model/{log_name}',
        loss_fn=loss_fn,
    )

elif args.stage == 'finetune':
    print("=" * 80)
    print("STAGE 2: FINE-TUNING ON B-CELL EPITOPES")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Mode: {args.mode}")
    print(f"Feature dim: {feat_dim}")
    print(f"Loading pretrained checkpoint: {args.pretrain_ckpt}")
    print(f"Freeze encoder: {args.freeze_encoder}")
    print(f"Fine-tuning learning rate: {args.lr}")
    print("=" * 80)

    # Load pretrained model
    model = model_cls.load_from_checkpoint(
        args.pretrain_ckpt,
        feat_dim=feat_dim, hidden_dim=args.hidden, exfeat_dim=13, edge_dim=51,
        augment_eps=0.05, dropout=0.2, lr=args.lr,
        metrics=METRICS(device), result_path=f'./model/{log_name}',
        loss_fn=loss_fn,
    )

    # Optional: freeze encoder layers (only train classifier head)
    if args.freeze_encoder:
        print("Freezing encoder layers (GAT, LSTM, attention)...")
        for name, param in model.named_parameters():
            # Freeze everything except the final MLP classifier
            if 'mlp' not in name:
                param.requires_grad = False

        # Count trainable parameters
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

run_training(model, trainset, valset, testset, args, cfn, use_early_stop=use_es)

print("=" * 80)
print(f"STAGE {args.stage.upper()} COMPLETED!")
print(f"Model saved to: ./model/{log_name}/model_{args.fold}.ckpt")
if args.stage == 'pretrain':
    print("Next step: Run fine-tuning stage with --stage finetune --pretrain_ckpt <path>")
print("=" * 80)
