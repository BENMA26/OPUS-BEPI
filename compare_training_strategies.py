"""
Compare different training strategies for epitope prediction.

This script evaluates and compares:
1. Training from scratch (baseline)
2. Pre-training + Fine-tuning (ScanNet-inspired)
3. DPO training (spatial coherence)
4. Pre-training + Fine-tuning + DPO (combined)

Usage:
    python compare_training_strategies.py \
        --dataset BCE_633 \
        --mode esm_gangxu \
        --gpu 0

Output:
    - Performance comparison table
    - Training time comparison
    - Model checkpoints for each strategy
"""

import os
import json
import time
import argparse
import pandas as pd
from pathlib import Path
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from tool import METRICS
from model import GraphBepi
from dataset import PDB_foldseek_tokens, collate_fn
from train_utils import seed_everything


class TrainingStrategy:
    """Base class for training strategies."""

    def __init__(self, name, description):
        self.name = name
        self.description = description
        self.results = {}
        self.training_time = 0

    def train(self, args):
        """Train the model and return results."""
        raise NotImplementedError

    def evaluate(self, model, test_loader, device):
        """Evaluate model on test set."""
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in test_loader:
                feats, edges, labels = batch
                feats = [f.to(device) for f in feats]
                edges = [e.to(device) for e in edges]
                labels = labels.to(device)

                preds = model(feats, edges).squeeze(-1)
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        preds = torch.cat(all_preds)
        labels = torch.cat(all_labels)

        metrics = METRICS(device)
        results = metrics(preds.to(device), labels.to(device))

        return {
            'AUROC': results['AUROC'],
            'AUPRC': results['AUPRC'],
            'MCC': results['MCC'],
            'F1': results['F1'],
        }


class FromScratchStrategy(TrainingStrategy):
    """Train from scratch (baseline)."""

    def __init__(self):
        super().__init__(
            name="From Scratch",
            description="Train directly on epitope data without pre-training"
        )

    def train(self, args):
        print(f"\n{'='*80}")
        print(f"Strategy: {self.name}")
        print(f"Description: {self.description}")
        print(f"{'='*80}\n")

        start_time = time.time()

        # Load dataset
        trainset = PDB_foldseek_tokens(mode='train', fold=args.fold, root=f'./data/{args.dataset}')
        valset = PDB_foldseek_tokens(mode='val', fold=args.fold, root=f'./data/{args.dataset}')
        testset = PDB_foldseek_tokens(mode='test', fold=args.fold, root=f'./data/{args.dataset}')

        # Create model
        device = f'cuda:{args.gpu}' if args.gpu != -1 else 'cpu'
        model = GraphBepi(
            feat_dim=2581, hidden_dim=args.hidden, exfeat_dim=13, edge_dim=51,
            augment_eps=0.05, dropout=0.2, lr=args.lr,
            metrics=METRICS(device), result_path=f'./model/compare_scratch',
            loss_fn=torch.nn.BCELoss(),
        )

        # Train
        train_loader = DataLoader(trainset, batch_size=args.batch, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(valset, batch_size=args.batch, collate_fn=collate_fn)
        test_loader = DataLoader(testset, batch_size=args.batch, collate_fn=collate_fn)

        trainer = pl.Trainer(
            accelerator='gpu' if args.gpu != -1 else 'cpu',
            max_epochs=args.epochs,
            enable_checkpointing=False,
            logger=False,
        )
        trainer.fit(model, train_loader, val_loader)

        self.training_time = time.time() - start_time

        # Evaluate
        self.results = self.evaluate(model, test_loader, device)
        self.results['training_time'] = self.training_time

        return self.results


class PretrainFinetuneStrategy(TrainingStrategy):
    """Pre-train on binding sites, then fine-tune on epitopes."""

    def __init__(self):
        super().__init__(
            name="Pre-train + Fine-tune",
            description="ScanNet-inspired: pre-train on binding sites, fine-tune on epitopes"
        )

    def train(self, args):
        print(f"\n{'='*80}")
        print(f"Strategy: {self.name}")
        print(f"Description: {self.description}")
        print(f"{'='*80}\n")

        # Check if pre-trained model exists
        pretrain_ckpt = f'./model/{args.pretrain_dataset}_GraphBepi_pretrain/model_-1.ckpt'
        if not os.path.exists(pretrain_ckpt):
            print(f"Warning: Pre-trained checkpoint not found: {pretrain_ckpt}")
            print("Skipping this strategy. Run pre-training first.")
            return None

        start_time = time.time()

        # Load dataset
        trainset = PDB_foldseek_tokens(mode='train', fold=args.fold, root=f'./data/{args.dataset}')
        valset = PDB_foldseek_tokens(mode='val', fold=args.fold, root=f'./data/{args.dataset}')
        testset = PDB_foldseek_tokens(mode='test', fold=args.fold, root=f'./data/{args.dataset}')

        # Load pre-trained model
        device = f'cuda:{args.gpu}' if args.gpu != -1 else 'cpu'
        model = GraphBepi.load_from_checkpoint(
            pretrain_ckpt,
            feat_dim=2581, hidden_dim=args.hidden, exfeat_dim=13, edge_dim=51,
            augment_eps=0.05, dropout=0.2, lr=args.lr * 0.1,  # Lower LR for fine-tuning
            metrics=METRICS(device), result_path=f'./model/compare_finetune',
            loss_fn=torch.nn.BCELoss(),
        )

        # Fine-tune
        train_loader = DataLoader(trainset, batch_size=args.batch, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(valset, batch_size=args.batch, collate_fn=collate_fn)
        test_loader = DataLoader(testset, batch_size=args.batch, collate_fn=collate_fn)

        trainer = pl.Trainer(
            accelerator='gpu' if args.gpu != -1 else 'cpu',
            max_epochs=args.epochs // 2,  # Fewer epochs for fine-tuning
            enable_checkpointing=False,
            logger=False,
        )
        trainer.fit(model, train_loader, val_loader)

        self.training_time = time.time() - start_time

        # Evaluate
        self.results = self.evaluate(model, test_loader, device)
        self.results['training_time'] = self.training_time

        return self.results


def compare_strategies(args):
    """Compare all training strategies."""

    strategies = [
        FromScratchStrategy(),
        PretrainFinetuneStrategy(),
        # Add more strategies here (DPO, combined, etc.)
    ]

    results = []

    for strategy in strategies:
        try:
            result = strategy.train(args)
            if result is not None:
                results.append({
                    'Strategy': strategy.name,
                    'Description': strategy.description,
                    **result
                })
        except Exception as e:
            print(f"Error in {strategy.name}: {e}")
            continue

    # Create comparison table
    df = pd.DataFrame(results)

    print("\n" + "="*80)
    print("TRAINING STRATEGY COMPARISON")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)

    # Save results
    output_dir = './results/strategy_comparison'
    os.makedirs(output_dir, exist_ok=True)

    df.to_csv(f'{output_dir}/comparison.csv', index=False)
    df.to_json(f'{output_dir}/comparison.json', orient='records', indent=2)

    print(f"\nResults saved to {output_dir}/")

    # Print recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    if len(results) > 1:
        best_auroc = df.loc[df['AUROC'].idxmax()]
        best_auprc = df.loc[df['AUPRC'].idxmax()]

        print(f"Best AUROC: {best_auroc['Strategy']} ({best_auroc['AUROC']:.4f})")
        print(f"Best AUPRC: {best_auprc['Strategy']} ({best_auprc['AUPRC']:.4f})")

        if 'Pre-train' in best_auroc['Strategy']:
            print("\n✓ Pre-training improves performance!")
            print("  Recommendation: Use pre-training + fine-tuning workflow")
        else:
            print("\n✗ Pre-training did not improve performance")
            print("  Possible reasons:")
            print("  - Pre-training dataset too different from epitope data")
            print("  - Need more fine-tuning epochs")
            print("  - Try different learning rates")

    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Compare training strategies")
    parser.add_argument('--dataset', type=str, default='BCE_633',
                       help='Epitope dataset name')
    parser.add_argument('--pretrain_dataset', type=str, default='Dockground_5K',
                       help='Pre-training dataset name')
    parser.add_argument('--mode', type=str, default='esm_gangxu',
                       help='Feature mode')
    parser.add_argument('--hidden', type=int, default=256,
                       help='Hidden dimension')
    parser.add_argument('--batch', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Training epochs (for from-scratch)')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device (-1 for CPU)')
    parser.add_argument('--fold', type=int, default=-1,
                       help='Cross-validation fold (-1 for all data)')
    parser.add_argument('--seed', type=int, default=2022,
                       help='Random seed')

    args = parser.parse_args()
    seed_everything(args.seed)

    compare_strategies(args)


if __name__ == '__main__':
    main()
