"""
从 PDB 复合物中提取蛋白质-蛋白质界面数据，用于预训练。

这个脚本会：
1. 读取 PDB 复合物文件
2. 识别蛋白质-蛋白质界面残基（距离 < 6Å）
3. 为每条链创建 chain 对象
4. 标注界面残基为正样本（label=1）
5. 生成与 BCE_633 相同格式的数据集

使用方法：
    python extract_binding_sites.py \
        --input_dir ./data/pdb_complexes \
        --output_dir ./data/Dockground_5K \
        --distance_cutoff 6.0 \
        --min_interface_size 5

输出：
    ./data/Dockground_5K/
    ├── train.pkl
    ├── test.pkl
    ├── cross-validation.npy
    ├── feat/          # ESM-2 特征（需要后续生成）
    ├── dssp/          # DSSP 特征（需要后续生成）
    └── graph/         # 图结构（需要后续生成）
"""

import os
import sys
import pickle as pk
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# 导入你的工具函数
from utils import chain
from preprocess import judge, get_dssp, DICT


def extract_interface_residues(pdb_file, distance_cutoff=6.0):
    """
    从 PDB 复合物中提取界面残基。

    Args:
        pdb_file: PDB 文件路径
        distance_cutoff: 界面距离阈值（Å）

    Returns:
        dict: {chain_id: set(interface_residue_positions)}
    """
    # 读取所有链的坐标
    chains_coords = defaultdict(dict)  # {chain_id: {res_pos: [x,y,z]}}
    chains_amino = defaultdict(dict)   # {chain_id: {res_pos: amino}}

    with open(pdb_file, 'r') as f:
        for line in f:
            feats = judge(line, 'CA')  # 只读取 CA 原子
            if feats is None:
                continue

            amino, chain_id, site, x, y, z = feats
            if len(amino) > 3:
                amino = amino[-3:]

            chains_coords[chain_id][site] = np.array([x, y, z])
            chains_amino[chain_id][site] = amino

    # 如果只有一条链，跳过
    if len(chains_coords) < 2:
        return None

    # 计算每条链与其他链的界面残基
    interfaces = defaultdict(set)

    chain_ids = list(chains_coords.keys())
    for i, chain_a in enumerate(chain_ids):
        for chain_b in chain_ids[i+1:]:
            # 计算链 A 和链 B 之间的距离
            for pos_a, coord_a in chains_coords[chain_a].items():
                for pos_b, coord_b in chains_coords[chain_b].items():
                    dist = np.linalg.norm(coord_a - coord_b)
                    if dist < distance_cutoff:
                        interfaces[chain_a].add(pos_a)
                        interfaces[chain_b].add(pos_b)

    return interfaces, chains_amino


def create_chain_object(pdb_file, chain_id, interface_residues, root):
    """
    为单条链创建 chain 对象，并标注界面残基。

    Args:
        pdb_file: PDB 文件路径
        chain_id: 链 ID
        interface_residues: 界面残基位置集合
        root: 输出根目录

    Returns:
        chain 对象
    """
    data = chain()

    # 设置名称
    pdb_id = Path(pdb_file).stem
    data.protein_name = pdb_id
    data.chain_name = chain_id
    data.name = f"{pdb_id}_{chain_id}"

    # 读取链的坐标和序列
    with open(pdb_file, 'r') as f:
        for line in f:
            feats = judge(line, 'CA')
            if feats is None:
                continue

            amino, cid, site, x, y, z = feats
            if cid != chain_id:
                continue

            if len(amino) > 3:
                amino = amino[-3:]

            # 添加残基
            data.add(amino, site, [x, y, z])

    # 处理序列
    data.process()

    # 标注界面残基
    for pos in interface_residues:
        idx = data.site.get(pos, None)
        if idx is not None:
            data.label[idx] = 1

    return data


def process_pdb_complex(pdb_file, output_root, distance_cutoff=6.0, min_interface_size=5):
    """
    处理单个 PDB 复合物文件。

    Returns:
        list of chain objects
    """
    try:
        # 提取界面残基
        result = extract_interface_residues(pdb_file, distance_cutoff)
        if result is None:
            return []

        interfaces, chains_amino = result

        # 过滤：界面残基数量太少的链
        valid_chains = []
        for chain_id, interface_res in interfaces.items():
            if len(interface_res) >= min_interface_size:
                valid_chains.append(chain_id)

        if len(valid_chains) < 2:
            return []

        # 为每条有效链创建 chain 对象
        chain_objects = []
        for chain_id in valid_chains:
            try:
                data = create_chain_object(
                    pdb_file, chain_id,
                    interfaces[chain_id],
                    output_root
                )

                # 检查是否有足够的正样本
                if data.label.sum() >= min_interface_size:
                    chain_objects.append(data)

            except Exception as e:
                print(f"  Error processing chain {chain_id}: {e}")
                continue

        return chain_objects

    except Exception as e:
        print(f"Error processing {pdb_file}: {e}")
        return []


def prepare_dataset(input_dir, output_dir, distance_cutoff=6.0,
                   min_interface_size=5, test_ratio=0.1, seed=2022):
    """
    准备预训练数据集。
    """
    np.random.seed(seed)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{output_dir}/feat', exist_ok=True)
    os.makedirs(f'{output_dir}/dssp', exist_ok=True)
    os.makedirs(f'{output_dir}/graph', exist_ok=True)

    # 查找所有 PDB 文件
    pdb_files = list(Path(input_dir).glob("*.pdb"))
    print(f"Found {len(pdb_files)} PDB files")

    if len(pdb_files) == 0:
        print(f"Error: No PDB files found in {input_dir}")
        return

    # 处理每个 PDB 文件
    all_chains = []
    stats = {
        'total_files': len(pdb_files),
        'processed_files': 0,
        'failed_files': 0,
        'total_chains': 0,
        'positive_residues': 0,
        'total_residues': 0,
    }

    for pdb_file in tqdm(pdb_files, desc="Processing PDB files"):
        chains = process_pdb_complex(
            pdb_file, output_dir,
            distance_cutoff, min_interface_size
        )

        if chains:
            all_chains.extend(chains)
            stats['processed_files'] += 1
            stats['total_chains'] += len(chains)

            for c in chains:
                stats['positive_residues'] += c.label.sum().item()
                stats['total_residues'] += len(c)
        else:
            stats['failed_files'] += 1

    print(f"\n{'='*80}")
    print("Processing Statistics:")
    print(f"  Total PDB files: {stats['total_files']}")
    print(f"  Successfully processed: {stats['processed_files']}")
    print(f"  Failed: {stats['failed_files']}")
    print(f"  Total chains extracted: {stats['total_chains']}")
    print(f"  Total residues: {stats['total_residues']}")
    print(f"  Interface residues: {stats['positive_residues']}")
    print(f"  Positive ratio: {stats['positive_residues']/stats['total_residues']:.2%}")
    print(f"{'='*80}\n")

    if len(all_chains) == 0:
        print("Error: No valid chains extracted!")
        return

    # 划分训练集和测试集
    n_test = int(len(all_chains) * test_ratio)
    indices = np.random.permutation(len(all_chains))

    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    trainset = [all_chains[i] for i in train_indices]
    testset = [all_chains[i] for i in test_indices]

    # 保存数据集
    with open(f'{output_dir}/train.pkl', 'wb') as f:
        pk.dump(trainset, f)

    with open(f'{output_dir}/test.pkl', 'wb') as f:
        pk.dump(testset, f)

    # 创建交叉验证索引
    cv_indices = np.random.permutation(len(trainset))
    np.save(f'{output_dir}/cross-validation.npy', cv_indices)

    print(f"Dataset saved to {output_dir}/")
    print(f"  Train: {len(trainset)} chains")
    print(f"  Test:  {len(testset)} chains")
    print(f"  Cross-validation indices: {len(cv_indices)}")

    # 保存统计信息
    with open(f'{output_dir}/stats.txt', 'w') as f:
        f.write("Dataset Statistics\n")
        f.write("="*80 + "\n")
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
        f.write(f"\nTrain set: {len(trainset)} chains\n")
        f.write(f"Test set: {len(testset)} chains\n")

    print("\n" + "="*80)
    print("IMPORTANT: Next steps to complete the dataset preparation:")
    print("="*80)
    print("1. Generate ESM-2 features:")
    print(f"   python generate_esm_features.py --dataset {Path(output_dir).name}")
    print("\n2. Generate DSSP features:")
    print(f"   python generate_dssp_features.py --dataset {Path(output_dir).name}")
    print("\n3. Generate graph structures:")
    print(f"   python generate_graphs.py --dataset {Path(output_dir).name}")
    print("\n4. Start pre-training:")
    print("   sbatch run_pretrain_stage.sh")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Extract protein-protein binding sites from PDB complexes"
    )
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing PDB complex files')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for processed dataset')
    parser.add_argument('--distance_cutoff', type=float, default=6.0,
                       help='Distance cutoff (Å) for interface residues (default: 6.0)')
    parser.add_argument('--min_interface_size', type=int, default=5,
                       help='Minimum number of interface residues (default: 5)')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                       help='Fraction of data for test set (default: 0.1)')
    parser.add_argument('--seed', type=int, default=2022,
                       help='Random seed (default: 2022)')

    args = parser.parse_args()

    print("="*80)
    print("Extracting Protein-Protein Binding Sites")
    print("="*80)
    print(f"Input directory:     {args.input_dir}")
    print(f"Output directory:    {args.output_dir}")
    print(f"Distance cutoff:     {args.distance_cutoff} Å")
    print(f"Min interface size:  {args.min_interface_size} residues")
    print(f"Test ratio:          {args.test_ratio}")
    print(f"Random seed:         {args.seed}")
    print("="*80 + "\n")

    prepare_dataset(
        args.input_dir,
        args.output_dir,
        args.distance_cutoff,
        args.min_interface_size,
        args.test_ratio,
        args.seed
    )


if __name__ == '__main__':
    main()
