"""
分析测试结果，为每条链生成详细的 TP/TN/FP/FN 位置信息
"""
import os
import torch
import pickle as pk
import pandas as pd
import argparse
from dataset import PDB


def analyze_chain_predictions(result_path, dataset_root, output_dir, fold=-1):
    """
    分析每条链的预测结果，记录每个位置的分类情况

    Args:
        result_path: 包含 result.pkl 的目录路径
        dataset_root: 数据集根目录
        output_dir: 输出目录
        fold: 数据集折数
    """
    # 加载测试结果
    result = torch.load(f'{result_path}/result.pkl')
    pred = result['pred']
    gt = result['gt']
    threshold = result.get('threshold', 0.5)
    tp = result['tp']
    tn = result['tn']
    fp = result['fp']
    fn = result['fn']

    # 加载测试集
    testset = PDB(mode='test', fold=fold, root=dataset_root)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 为每条链创建索引
    IDX = []
    for i in range(len(testset)):
        IDX += [i] * len(testset.data[i])
    IDX = torch.LongTensor(IDX)

    # 统计信息
    summary_data = []

    # 为每条链生成详细报告
    for i in range(len(testset)):
        chain = testset.data[i]
        idx = IDX == i

        # 提取该链的数据
        chain_pred = pred[idx]
        chain_gt = gt[idx]
        chain_tp = tp[idx]
        chain_tn = tn[idx]
        chain_fp = fp[idx]
        chain_fn = fn[idx]

        # 创建分类标签
        classification = []
        for j in range(len(chain_pred)):
            if chain_tp[j]:
                classification.append('TP')
            elif chain_tn[j]:
                classification.append('TN')
            elif chain_fp[j]:
                classification.append('FP')
            elif chain_fn[j]:
                classification.append('FN')
            else:
                classification.append('Unknown')

        # 创建 DataFrame
        df = pd.DataFrame({
            'position': list(range(1, len(chain.sequence) + 1)),
            'residue': list(chain.sequence),
            'true_label': chain_gt.numpy(),
            'pred_score': chain_pred.numpy(),
            'pred_label': (chain_pred > threshold).long().numpy(),
            'classification': classification,
            'is_TP': chain_tp.numpy(),
            'is_TN': chain_tn.numpy(),
            'is_FP': chain_fp.numpy(),
            'is_FN': chain_fn.numpy()
        })

        # 保存该链的详细结果
        csv_path = f'{output_dir}/{chain.name}_detailed.csv'
        df.to_csv(csv_path, index=False)

        # 统计该链的分类数量
        n_tp = chain_tp.sum().item()
        n_tn = chain_tn.sum().item()
        n_fp = chain_fp.sum().item()
        n_fn = chain_fn.sum().item()

        # 计算该链的指标
        precision = n_tp / (n_tp + n_fp) if (n_tp + n_fp) > 0 else 0
        recall = n_tp / (n_tp + n_fn) if (n_tp + n_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (n_tp + n_tn) / len(chain_pred)

        summary_data.append({
            'chain_name': chain.name,
            'length': len(chain.sequence),
            'n_epitope': chain_gt.sum().item(),
            'TP': n_tp,
            'TN': n_tn,
            'FP': n_fp,
            'FN': n_fn,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy
        })

        print(f"Chain {chain.name}: TP={n_tp}, TN={n_tn}, FP={n_fp}, FN={n_fn}")

    # 保存汇总统计
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(f'{output_dir}/summary.csv', index=False)

    # 打印总体统计
    print("\n" + "="*60)
    print("总体统计:")
    print("="*60)
    print(f"总链数: {len(testset)}")
    print(f"总残基数: {len(pred)}")
    print(f"总 TP: {tp.sum().item()}")
    print(f"总 TN: {tn.sum().item()}")
    print(f"总 FP: {fp.sum().item()}")
    print(f"总 FN: {fn.sum().item()}")
    print(f"阈值: {threshold:.4f}")
    print(f"\n详细结果已保存到: {output_dir}")
    print(f"汇总统计: {output_dir}/summary.csv")
    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='分析测试结果中的 TP/TN/FP/FN')
    parser.add_argument('--result_path', type=str, required=True,
                       help='包含 result.pkl 的目录路径')
    parser.add_argument('--dataset', type=str, default='BCE_633',
                       help='数据集名称')
    parser.add_argument('--output_dir', type=str, default='./analysis_results',
                       help='输出目录')
    parser.add_argument('--fold', type=int, default=-1,
                       help='数据集折数')

    args = parser.parse_args()

    dataset_root = f'./data/{args.dataset}'

    analyze_chain_predictions(
        result_path=args.result_path,
        dataset_root=dataset_root,
        output_dir=args.output_dir,
        fold=args.fold
    )
