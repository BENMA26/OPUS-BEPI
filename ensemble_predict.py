"""
Ensemble prediction script for OPUS-BEPI
Selects top 3 models based on validation AUPRC and performs ensemble prediction on test set
"""
import os
import torch
import pickle as pk
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from model import GraphBepi
from dataset import PDB_foldseek_tokens, collate_fn
from tool import METRICS


def load_model_results(dataset, seed, fold=-1):
    """Load model checkpoint and validation results"""
    tag = f"GraphBepi_seed{seed}"
    model_dir = f'./model/{dataset}_{tag}'
    ckpt_path = f'{model_dir}/model_{fold}.ckpt'
    result_path = f'{model_dir}/result_{fold}.pkl'

    if not os.path.exists(ckpt_path):
        return None, None

    # Load checkpoint to get validation AUPRC
    ckpt = torch.load(ckpt_path, map_location='cpu')

    # Try to get validation AUPRC from checkpoint
    val_auprc = None
    if 'callbacks' in ckpt:
        # Try to extract from ModelCheckpoint callback
        for callback_state in ckpt.get('callbacks', {}).values():
            if 'best_model_score' in callback_state:
                val_auprc = callback_state['best_model_score'].item()
                break

    return ckpt, val_auprc


def select_top_models(dataset, seeds, fold=-1, top_k=3):
    """Select top K models based on validation AUPRC"""
    model_scores = []

    for seed in seeds:
        ckpt, val_auprc = load_model_results(dataset, seed, fold)
        if ckpt is not None and val_auprc is not None:
            model_scores.append((seed, val_auprc, ckpt))
            print(f"Seed {seed}: val_AUPRC = {val_auprc:.4f}")
        else:
            print(f"Seed {seed}: checkpoint or validation score not found")

    if len(model_scores) == 0:
        raise ValueError("No valid models found!")

    # Sort by validation AUPRC (descending)
    model_scores.sort(key=lambda x: x[1], reverse=True)

    # Select top K
    top_models = model_scores[:top_k]
    print(f"\nSelected top {top_k} models:")
    for seed, auprc, _ in top_models:
        print(f"  Seed {seed}: val_AUPRC = {auprc:.4f}")

    return top_models


def ensemble_predict(models, test_loader, device, ensemble_method='mean'):
    """
    Perform ensemble prediction

    Args:
        models: list of (seed, val_auprc, model) tuples
        test_loader: DataLoader for test set
        device: device to run inference on
        ensemble_method: 'mean' or 'vote'
    """
    all_predictions = []
    all_labels = []

    for seed, val_auprc, model_instance in models:
        model_instance.eval()
        model_instance = model_instance.to(device)

        predictions = []
        labels = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Predicting with seed {seed}"):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                # Get predictions
                pred = model_instance(batch)
                predictions.append(pred.cpu())
                labels.append(batch['label'].cpu())

        predictions = torch.cat(predictions, dim=0)
        all_predictions.append(predictions)

        if len(all_labels) == 0:
            all_labels = torch.cat(labels, dim=0)

    # Ensemble predictions
    all_predictions = torch.stack(all_predictions, dim=0)  # (n_models, n_samples)

    if ensemble_method == 'mean':
        ensemble_pred = all_predictions.mean(dim=0)
    elif ensemble_method == 'vote':
        # Majority voting (threshold at 0.5)
        votes = (all_predictions > 0.5).float()
        ensemble_pred = votes.mean(dim=0)
    else:
        raise ValueError(f"Unknown ensemble method: {ensemble_method}")

    return ensemble_pred, all_labels


def main():
    parser = argparse.ArgumentParser(description='Ensemble prediction for OPUS-BEPI')
    parser.add_argument('--dataset', type=str, default='BCE_633', help='dataset name')
    parser.add_argument('--seeds', type=int, nargs='+', default=[2022, 2023, 2024, 2025, 2026],
                       help='random seeds to consider')
    parser.add_argument('--top_k', type=int, default=3, help='number of top models to ensemble')
    parser.add_argument('--fold', type=int, default=-1, help='dataset fold')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device')
    parser.add_argument('--batch', type=int, default=4, help='batch size')
    parser.add_argument('--ensemble_method', type=str, default='mean',
                       choices=['mean', 'vote'], help='ensemble method')
    parser.add_argument('--output_dir', type=str, default='./model/ensemble_results',
                       help='output directory for ensemble results')
    args = parser.parse_args()

    device = 'cpu' if args.gpu == -1 else f'cuda:{args.gpu}'
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*60)
    print("Ensemble Prediction for OPUS-BEPI")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Seeds: {args.seeds}")
    print(f"Top K: {args.top_k}")
    print(f"Ensemble method: {args.ensemble_method}")
    print("="*60)

    # Select top K models
    top_models_info = select_top_models(args.dataset, args.seeds, args.fold, args.top_k)

    # Load models
    models = []
    for seed, val_auprc, ckpt in top_models_info:
        model = GraphBepi(
            feat_dim=2581,
            hidden_dim=256,
            exfeat_dim=13,
            edge_dim=51,
            augment_eps=0.05,
            dropout=0.2,
            metrics=METRICS(device),
            result_path=args.output_dir,
        )
        model.load_state_dict(ckpt['state_dict'])
        models.append((seed, val_auprc, model))

    # Load test dataset
    root = f'./data/{args.dataset}'
    testset = PDB_foldseek_tokens(mode='test', fold=args.fold, root=root)
    test_loader = DataLoader(testset, batch_size=args.batch, shuffle=False,
                            collate_fn=collate_fn)

    print(f"\nTest set size: {len(testset)}")

    # Perform ensemble prediction
    print("\nPerforming ensemble prediction...")
    ensemble_pred, labels = ensemble_predict(models, test_loader, device, args.ensemble_method)

    # Evaluate ensemble performance
    from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

    labels_np = labels.numpy()
    pred_np = ensemble_pred.numpy()

    auroc = roc_auc_score(labels_np, pred_np)
    auprc = average_precision_score(labels_np, pred_np)
    acc = accuracy_score(labels_np, (pred_np > 0.5).astype(int))

    print("\n" + "="*60)
    print("Ensemble Results:")
    print("="*60)
    print(f"AUROC: {auroc:.4f}")
    print(f"AUPRC: {auprc:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print("="*60)

    # Save results
    result = {
        'pred': ensemble_pred,
        'label': labels,
        'auroc': auroc,
        'auprc': auprc,
        'accuracy': acc,
        'top_models': [(seed, val_auprc) for seed, val_auprc, _ in top_models_info],
        'ensemble_method': args.ensemble_method,
    }

    output_path = f'{args.output_dir}/ensemble_result_top{args.top_k}_{args.ensemble_method}.pkl'
    torch.save(result, output_path)
    print(f"\nResults saved to: {output_path}")

    # Save detailed predictions
    df = pd.DataFrame({
        'prediction': pred_np,
        'label': labels_np,
    })
    csv_path = f'{args.output_dir}/ensemble_predictions_top{args.top_k}_{args.ensemble_method}.csv'
    df.to_csv(csv_path, index=False)
    print(f"Predictions saved to: {csv_path}")


if __name__ == '__main__':
    main()
