# 如何准备蛋白质-蛋白质复合物结构数据

## 快速开始（3 步）

### 步骤 1: 下载 PDB 复合物

```bash
# 下载 1000 个高质量复合物（分辨率 < 2.5Å，至少 2 条链）
python download_pdb_complexes.py \
    --output_dir ./data/pdb_complexes \
    --max_resolution 2.5 \
    --download_limit 1000
```

### 步骤 2: 提取结合位点

```bash
# 从复合物中提取蛋白质-蛋白质界面
python extract_binding_sites.py \
    --input_dir ./data/pdb_complexes \
    --output_dir ./data/PDB_1000 \
    --distance_cutoff 6.0 \
    --min_interface_size 5
```

### 步骤 3: 生成特征（需要在服务器上运行）

```bash
# SSH 到服务器
ssh g01n23_maben
cd /work/home/maben/project/epitope_prediction/GraphBepi

# 生成 ESM-2 特征、DSSP 特征和图结构
# 参考 DATA_PREPARATION_GUIDE.md 中的详细步骤
```

---

## 详细说明

### 方案 1: 自动下载（推荐）

**优点**: 简单快速，自动化程度高
**缺点**: 需要网络连接，可能较慢

```bash
# 1. 搜索并下载
python download_pdb_complexes.py \
    --output_dir ./data/pdb_complexes \
    --max_resolution 2.5 \
    --min_chains 2 \
    --download_limit 1000

# 2. 提取界面
python extract_binding_sites.py \
    --input_dir ./data/pdb_complexes \
    --output_dir ./data/PDB_1000
```

### 方案 2: 使用 Dockground 数据集

**优点**: 高质量，已标注
**缺点**: 需要手动下载

```bash
# 1. 下载 DB5.5 数据集
wget https://zlab.umassmed.edu/benchmark/benchmark5.5.tgz
tar -xzf benchmark5.5.tgz -C ./data/pdb_complexes/

# 2. 提取界面
python extract_binding_sites.py \
    --input_dir ./data/pdb_complexes/benchmark5.5 \
    --output_dir ./data/Dockground_230
```

### 方案 3: 从 PDB ID 列表下载

**优点**: 精确控制下载内容
**缺点**: 需要预先准备 ID 列表

```bash
# 1. 创建 PDB ID 列表文件
cat > pdb_ids.txt << EOF
1A2K
1AHW
1BRS
1CGI
EOF

# 2. 从列表下载
python download_pdb_complexes.py \
    --pdb_list pdb_ids.txt \
    --output_dir ./data/pdb_complexes

# 3. 提取界面
python extract_binding_sites.py \
    --input_dir ./data/pdb_complexes \
    --output_dir ./data/Custom_Dataset
```

---

## 参数说明

### download_pdb_complexes.py

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--output_dir` | `./data/pdb_complexes` | PDB 文件输出目录 |
| `--min_chains` | 2 | 最少链数（2=二聚体） |
| `--max_resolution` | 2.5 | 最大分辨率（Å） |
| `--download_limit` | 1000 | 下载文件数量上限 |
| `--search_only` | False | 只搜索不下载 |
| `--pdb_list` | None | 从文件列表下载 |

### extract_binding_sites.py

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input_dir` | 必需 | PDB 文件输入目录 |
| `--output_dir` | 必需 | 数据集输出目录 |
| `--distance_cutoff` | 6.0 | 界面距离阈值（Å） |
| `--min_interface_size` | 5 | 最少界面残基数 |
| `--test_ratio` | 0.1 | 测试集比例 |

---

## 推荐配置

### 快速验证（100-200 个复合物）

```bash
python download_pdb_complexes.py \
    --download_limit 200 \
    --max_resolution 2.0

python extract_binding_sites.py \
    --input_dir ./data/pdb_complexes \
    --output_dir ./data/PDB_200 \
    --distance_cutoff 6.0
```

**预期**: 约 400-600 条链，训练时间 6-12 小时

### 标准配置（500-1000 个复合物）

```bash
python download_pdb_complexes.py \
    --download_limit 1000 \
    --max_resolution 2.5

python extract_binding_sites.py \
    --input_dir ./data/pdb_complexes \
    --output_dir ./data/PDB_1000 \
    --distance_cutoff 6.0
```

**预期**: 约 2000-3000 条链，训练时间 1-2 天

### 大规模配置（2000+ 个复合物）

```bash
python download_pdb_complexes.py \
    --download_limit 3000 \
    --max_resolution 3.0

python extract_binding_sites.py \
    --input_dir ./data/pdb_complexes \
    --output_dir ./data/PDB_3000 \
    --distance_cutoff 6.0
```

**预期**: 约 6000-9000 条链，训练时间 3-5 天

---

## 输出文件结构

```
./data/PDB_1000/
├── train.pkl              # 训练集（chain 对象列表）
├── test.pkl               # 测试集
├── cross-validation.npy   # 10-fold 交叉验证索引
├── stats.txt              # 数据集统计信息
├── feat/                  # ESM-2 特征（需生成）
│   └── 1A2K_A_esm2.ts
├── dssp/                  # DSSP 特征（需生成）
│   ├── 1A2K_A.npy
│   └── 1A2K_A_pos.npy
└── graph/                 # 图结构（需生成）
    └── 1A2K_A.graph
```

---

## 常见问题

### Q: 下载速度很慢怎么办？

**A**: 使用 `--download_limit` 限制数量，或分批下载：

```bash
# 分 5 批，每批 200 个
for i in {1..5}; do
    python download_pdb_complexes.py \
        --download_limit 200 \
        --output_dir ./data/pdb_complexes_batch${i}
done
```

### Q: 提取的界面残基太少怎么办？

**A**: 调整参数：

```bash
python extract_binding_sites.py \
    --distance_cutoff 8.0      # 增大距离阈值
    --min_interface_size 3     # 降低最小界面大小
```

### Q: 如何验证数据质量？

**A**: 运行检查脚本：

```bash
python << 'EOF'
import pickle as pk
import numpy as np

with open('./data/PDB_1000/train.pkl', 'rb') as f:
    trainset = pk.load(f)

print(f"训练集: {len(trainset)} 条链")

# 统计正样本比例
pos_ratios = [c.label.sum().item() / len(c) for c in trainset]
print(f"界面残基比例: {np.mean(pos_ratios):.2%} ± {np.std(pos_ratios):.2%}")

# 统计链长度
lengths = [len(c) for c in trainset]
print(f"链长度: {np.mean(lengths):.0f} ± {np.std(lengths):.0f}")
EOF
```

---

## 完整工作流程

```bash
# 1. 下载 PDB 复合物（本地）
python download_pdb_complexes.py \
    --output_dir ./data/pdb_complexes \
    --download_limit 1000

# 2. 提取结合位点（本地）
python extract_binding_sites.py \
    --input_dir ./data/pdb_complexes \
    --output_dir ./data/PDB_1000

# 3. 同步到服务器
rsync -avz ./data/PDB_1000/ \
    g01n23_maben:/work/home/maben/project/epitope_prediction/GraphBepi/data/PDB_1000/

# 4. 在服务器上生成特征（参考 DATA_PREPARATION_GUIDE.md）

# 5. 开始预训练
ssh g01n23_maben
cd /work/home/maben/project/epitope_prediction/GraphBepi
sbatch run_pretrain_stage.sh
```

---

## 相关文档

- **DATA_PREPARATION_GUIDE.md** - 完整的数据准备指南
- **SCANNET_WORKFLOW.md** - 两阶段训练详细文档
- **QUICKSTART.md** - 快速开始指南

---

## 新增文件

本次创建的文件：

1. **download_pdb_complexes.py** - PDB 下载脚本
2. **extract_binding_sites.py** - 结合位点提取脚本
3. **DATA_PREPARATION_GUIDE.md** - 数据准备详细指南
4. **HOW_TO_PREPARE_COMPLEXES.md** - 本文档

所有文件已保存在项目根目录。
