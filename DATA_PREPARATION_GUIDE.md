# 蛋白质-蛋白质结合位点数据准备指南

本指南详细说明如何准备预训练数据集（蛋白质-蛋白质结合位点）。

---

## 方案 1: 使用 Dockground 数据集（推荐）

### 1.1 下载 Dockground

```bash
# 创建数据目录
mkdir -p ./data/pdb_complexes

# 下载 DB5.5 数据集（约 230 个复合物）
cd ./data/pdb_complexes
wget https://zlab.umassmed.edu/benchmark/benchmark5.5.tgz
tar -xzf benchmark5.5.tgz

# 或者下载更大的 Dockground 数据集
# 访问 https://dockground.compbio.ku.edu/ 下载
```

### 1.2 提取结合位点

```bash
# 返回项目根目录
cd /Users/benma/Desktop/表位预测/OPUS-BEPI

# 运行提取脚本
python extract_binding_sites.py \
    --input_dir ./data/pdb_complexes/benchmark5.5 \
    --output_dir ./data/Dockground_230 \
    --distance_cutoff 6.0 \
    --min_interface_size 5
```

---

## 方案 2: 从 PDB 批量下载复合物

### 2.1 创建下载脚本

```bash
cat > download_pdb_complexes.py << 'EOF'
"""
从 PDB 批量下载蛋白质复合物
"""
import requests
import json
from pathlib import Path
from tqdm import tqdm
import time

def search_pdb_complexes(min_chains=2, max_resolution=3.0, limit=5000):
    """搜索 PDB 中的蛋白质复合物"""

    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.polymer_entity_count_protein",
                        "operator": "greater_or_equal",
                        "value": min_chains
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.resolution_combined",
                        "operator": "less_or_equal",
                        "value": max_resolution
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "exptl.method",
                        "operator": "exact_match",
                        "value": "X-RAY DIFFRACTION"
                    }
                }
            ]
        },
        "return_type": "entry",
        "request_options": {
            "pager": {
                "start": 0,
                "rows": limit
            }
        }
    }

    url = "https://search.rcsb.org/rcsbsearch/v2/query"
    response = requests.post(url, json=query)

    if response.status_code == 200:
        results = response.json()
        pdb_ids = [item['identifier'] for item in results.get('result_set', [])]
        return pdb_ids
    else:
        print(f"Error: {response.status_code}")
        return []

def download_pdb_file(pdb_id, output_dir="./data/pdb_complexes"):
    """下载 PDB 文件"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    output_path = f"{output_dir}/{pdb_id}.pdb"

    # 如果文件已存在，跳过
    if Path(output_path).exists():
        return True

    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            with open(output_path, 'w') as f:
                f.write(response.text)
            return True
    except Exception as e:
        print(f"Error downloading {pdb_id}: {e}")

    return False

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='./data/pdb_complexes')
    parser.add_argument('--min_chains', type=int, default=2)
    parser.add_argument('--max_resolution', type=float, default=2.5)
    parser.add_argument('--limit', type=int, default=5000)
    parser.add_argument('--download_limit', type=int, default=1000,
                       help='Maximum number of files to download')
    args = parser.parse_args()

    print("Searching PDB for protein complexes...")
    pdb_ids = search_pdb_complexes(
        min_chains=args.min_chains,
        max_resolution=args.max_resolution,
        limit=args.limit
    )
    print(f"Found {len(pdb_ids)} complexes")

    print(f"\nDownloading up to {args.download_limit} PDB files...")
    success = 0
    failed = 0

    for i, pdb_id in enumerate(tqdm(pdb_ids[:args.download_limit]), 1):
        if download_pdb_file(pdb_id, args.output_dir):
            success += 1
        else:
            failed += 1

        # 避免请求过快
        if i % 100 == 0:
            time.sleep(1)

    print(f"\nDownload complete:")
    print(f"  Success: {success}")
    print(f"  Failed: {failed}")
    print(f"  Output: {args.output_dir}")
EOF
```

### 2.2 运行下载

```bash
# 下载 1000 个高质量复合物（分辨率 < 2.5Å）
python download_pdb_complexes.py \
    --output_dir ./data/pdb_complexes \
    --max_resolution 2.5 \
    --download_limit 1000
```

### 2.3 提取结合位点

```bash
python extract_binding_sites.py \
    --input_dir ./data/pdb_complexes \
    --output_dir ./data/PDB_1000 \
    --distance_cutoff 6.0 \
    --min_interface_size 5
```

---

## 方案 3: 使用现有的高质量数据集

### 3.1 SKEMPI 2.0

```bash
# 下载 SKEMPI 数据
mkdir -p ./data/skempi
cd ./data/skempi
wget https://life.bsc.es/pid/skempi2/database/download/skempi_v2.csv

# SKEMPI 包含 PDB ID 和突变信息
# 你需要根据 CSV 文件中的 PDB ID 下载结构
```

### 3.2 Protein-Protein Docking Benchmark 5.5

```bash
# 这是一个精心策划的数据集，包含 230 个复合物
wget https://zlab.umassmed.edu/benchmark/benchmark5.5.tgz
tar -xzf benchmark5.5.tgz -C ./data/pdb_complexes/
```

---

## 完整的数据准备流程

### 步骤 1: 下载 PDB 复合物

选择上述任一方案下载 PDB 文件到 `./data/pdb_complexes/`

### 步骤 2: 提取结合位点

```bash
python extract_binding_sites.py \
    --input_dir ./data/pdb_complexes \
    --output_dir ./data/Dockground_5K \
    --distance_cutoff 6.0 \
    --min_interface_size 5
```

**输出**:
```
./data/Dockground_5K/
├── train.pkl              # 训练集（chain 对象列表）
├── test.pkl               # 测试集
├── cross-validation.npy   # 交叉验证索引
├── stats.txt              # 统计信息
├── feat/                  # ESM-2 特征（待生成）
├── dssp/                  # DSSP 特征（待生成）
└── graph/                 # 图结构（待生成）
```

### 步骤 3: 生成 ESM-2 特征

```bash
# 使用你现有的 ESM-2 特征提取脚本
# 或者创建一个新的脚本

python << 'EOF'
import os
import esm
import torch
import pickle as pk
from tqdm import tqdm

# 加载 ESM-2 模型
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 加载数据集
dataset_name = 'Dockground_5K'
with open(f'./data/{dataset_name}/train.pkl', 'rb') as f:
    trainset = pk.load(f)
with open(f'./data/{dataset_name}/test.pkl', 'rb') as f:
    testset = pk.load(f)

all_chains = trainset + testset

# 生成特征
os.makedirs(f'./data/{dataset_name}/feat', exist_ok=True)

for chain_obj in tqdm(all_chains, desc="Generating ESM-2 features"):
    if len(chain_obj) > 1024:
        print(f"Skipping {chain_obj.name} (too long: {len(chain_obj)})")
        continue

    output_path = f'./data/{dataset_name}/feat/{chain_obj.name}_esm2.ts'
    if os.path.exists(output_path):
        continue

    # 准备输入
    data = [(chain_obj.name, chain_obj.sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)

    # 提取特征
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33])
        feat = results['representations'][33].squeeze(0)[1:-1].cpu()  # 去掉 <cls> 和 <eos>

    torch.save(feat, output_path)

print("ESM-2 features generated!")
EOF
```

### 步骤 4: 生成 DSSP 特征

```bash
# 使用你现有的 DSSP 生成脚本
python << 'EOF'
import os
import pickle as pk
from tqdm import tqdm
from preprocess import get_dssp

dataset_name = 'Dockground_5K'
with open(f'./data/{dataset_name}/train.pkl', 'rb') as f:
    trainset = pk.load(f)
with open(f'./data/{dataset_name}/test.pkl', 'rb') as f:
    testset = pk.load(f)

all_chains = trainset + testset

os.makedirs(f'./data/{dataset_name}/dssp', exist_ok=True)

for chain_obj in tqdm(all_chains, desc="Generating DSSP features"):
    # 这里需要根据你的 get_dssp 函数调整
    # get_dssp 通常需要 PDB 文件
    pass

print("DSSP features generated!")
EOF
```

### 步骤 5: 生成图结构

```bash
python << 'EOF'
import os
import pickle as pk
from tqdm import tqdm

dataset_name = 'Dockground_5K'
with open(f'./data/{dataset_name}/train.pkl', 'rb') as f:
    trainset = pk.load(f)
with open(f'./data/{dataset_name}/test.pkl', 'rb') as f:
    testset = pk.load(f)

all_chains = trainset + testset

os.makedirs(f'./data/{dataset_name}/graph', exist_ok=True)

for chain_obj in tqdm(all_chains, desc="Generating graph structures"):
    output_path = f'./data/{dataset_name}/graph/{chain_obj.name}.graph'
    if os.path.exists(output_path):
        continue

    # 生成图结构
    chain_obj.get_adj(f'./data/{dataset_name}')

print("Graph structures generated!")
EOF
```

### 步骤 6: 开始预训练

```bash
# 修改 run_pretrain_stage.sh 中的数据集名称
# DATASET="Dockground_5K"

sbatch run_pretrain_stage.sh
```

---

## 推荐的数据集规模

| 数据集规模 | 复合物数量 | 链数量（约） | 训练时间（约） | 适用场景 |
|-----------|----------|------------|--------------|---------|
| 小型 | 100-500 | 200-1000 | 6-12 小时 | 快速验证 |
| 中型 | 500-2000 | 1000-4000 | 1-2 天 | 一般使用 |
| 大型 | 2000-5000 | 4000-10000 | 3-5 天 | 最佳性能 |
| 超大型 | 5000+ | 10000+ | 5-10 天 | 研究级别 |

---

## 数据质量检查

### 检查提取的数据

```bash
python << 'EOF'
import pickle as pk
import numpy as np

dataset_name = 'Dockground_5K'

with open(f'./data/{dataset_name}/train.pkl', 'rb') as f:
    trainset = pk.load(f)

print(f"Train set: {len(trainset)} chains")
print(f"\nSample chain info:")
sample = trainset[0]
print(f"  Name: {sample.name}")
print(f"  Length: {len(sample)}")
print(f"  Sequence: {sample.sequence[:50]}...")
print(f"  Positive residues: {sample.label.sum().item()}")
print(f"  Positive ratio: {sample.label.sum().item() / len(sample):.2%}")

# 统计所有链的正样本比例
pos_ratios = [c.label.sum().item() / len(c) for c in trainset]
print(f"\nPositive ratio statistics:")
print(f"  Mean: {np.mean(pos_ratios):.2%}")
print(f"  Median: {np.median(pos_ratios):.2%}")
print(f"  Min: {np.min(pos_ratios):.2%}")
print(f"  Max: {np.max(pos_ratios):.2%}")
EOF
```

---

## 常见问题

### Q1: 下载 PDB 文件很慢怎么办？

使用镜像站点：
- 中国镜像: http://www.rcsb.org.cn/
- 日本镜像: https://pdbj.org/

### Q2: 提取的界面残基太少怎么办？

调整参数：
```bash
--distance_cutoff 8.0      # 增大距离阈值
--min_interface_size 3     # 降低最小界面大小
```

### Q3: 内存不足怎么办？

分批处理：
```bash
# 将 PDB 文件分成多个子目录
# 分别处理后合并结果
```

---

## 下一步

数据准备完成后：

1. 验证数据集质量
2. 运行预训练: `sbatch run_pretrain_stage.sh`
3. 监控训练进度
4. 在表位数据上微调

祝数据准备顺利！
