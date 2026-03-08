# ScanNet-Inspired Workflow - 快速开始指南

## 概述

本指南帮助你快速上手基于 ScanNet 的两阶段训练 workflow。

**核心思想**: 先在大规模蛋白质-蛋白质结合位点数据上预训练，再在表位数据上微调。

---

## 前置准备

### 1. 环境检查

确保你的环境已配置好：

```bash
# 检查 conda 环境
conda activate bepi

# 检查必要的包
python -c "import torch; import pytorch_lightning; print('Environment OK')"
```

### 2. 数据准备

你需要两个数据集：

#### 数据集 A: 蛋白质-蛋白质结合位点（预训练）

**选项 1: 使用现有数据集**
- Dockground: https://dockground.compbio.ku.edu/
- DB5.5: https://zlab.umassmed.edu/benchmark/
- SKEMPI: https://life.bsc.es/pid/skempi2/

**选项 2: 从 PDB 提取**

```bash
# 下载 PDB 复合物文件到 ./data/pdb_complexes/
# 然后运行提取脚本
python prepare_pretrain_dataset.py \
    --input_dir ./data/pdb_complexes \
    --output_dir ./data/Dockground_5K \
    --distance_cutoff 6.0 \
    --min_interface_size 5
```

**注意**: `prepare_pretrain_dataset.py` 是一个模板脚本，你需要根据实际的 `Protein` 类和特征提取流程进行调整。

#### 数据集 B: B 细胞表位（微调）

使用现有的 BCE_633 数据集（已准备好）。

---

## 快速开始（3 步）

### 步骤 1: 预训练

```bash
# 修改 run_pretrain_stage.sh 中的数据集名称
# DATASET="Dockground_5K"  # 改为你的数据集名称

# 提交预训练任务
sbatch run_pretrain_stage.sh
```

**预期时间**: 根据数据集大小，1-2 天

**监控进度**:
```bash
# 查看任务状态
squeue -u maben

# 查看日志
tail -f ./log/slurm_pretrain_*.out
```

### 步骤 2: 微调

等待预训练完成后：

```bash
# 检查预训练模型是否存在
ls -lh ./model/Dockground_5K_GraphBepi_pretrain/model_-1.ckpt

# 提交微调任务
sbatch run_finetune_stage.sh
```

**预期时间**: 4-8 小时

### 步骤 3: 评估

```bash
# 查看结果
python -c "
import torch
result = torch.load('./model/BCE_633_GraphBepi_finetune/result_-1.pkl')
print('Predictions shape:', result['pred'].shape)
print('Ground truth shape:', result['gt'].shape)
"

# 计算指标
python test.py \
    --checkpoint ./model/BCE_633_GraphBepi_finetune/model_-1.ckpt \
    --dataset BCE_633
```

---

## 高级用法

### 超参数搜索

如果你想找到最佳的微调超参数：

```bash
sbatch run_finetune_param_search.sh
```

这会并行运行 12 个不同的超参数组合。

### 比较训练策略

比较从头训练 vs 预训练+微调：

```bash
python compare_training_strategies.py \
    --dataset BCE_633 \
    --pretrain_dataset Dockground_5K \
    --gpu 0
```

### 结合 DPO

在微调后进一步使用 DPO 优化空间一致性：

```bash
python train_dpo_gangxu.py \
    --ref_ckpt ./model/BCE_633_GraphBepi_finetune/model_-1.ckpt \
    --dataset BCE_633 \
    --tag GraphBepi_finetune_dpo \
    --beta 0.1 \
    --lambda_task 1.0 \
    --epochs 30 \
    --lr 5e-6 \
    --gpu 0
```

---

## 常见问题

### Q1: 我没有大规模的蛋白质-蛋白质结合位点数据，怎么办？

**方案 A**: 使用公开数据集
- 下载 Dockground benchmark (推荐)
- 使用 DB5.5 或 SKEMPI (较小但高质量)

**方案 B**: 从 PDB 提取
- 下载包含多条链的 PDB 文件
- 使用 `prepare_pretrain_dataset.py` 提取界面

**方案 C**: 跳过预训练
- 直接使用 `train.py` 从头训练
- 或使用 DPO 方法 (`train_dpo_gangxu.py`)

### Q2: 预训练需要多长时间？

取决于数据集大小：
- 1,000 个蛋白质: ~6-12 小时
- 5,000 个蛋白质: ~1-2 天
- 20,000 个蛋白质: ~3-5 天

### Q3: 微调时应该冻结编码器吗？

**建议**:
- 如果表位数据集 > 500 样本: 不冻结（全模型微调）
- 如果表位数据集 < 500 样本: 冻结编码器（防止过拟合）

使用 `--freeze_encoder` 参数来冻结。

### Q4: 预训练真的有用吗？

根据 ScanNet 的经验：
- 在小样本场景下提升明显（AUROC +2-5%）
- 对新抗原的泛化能力更强
- 训练更稳定，收敛更快

建议运行 `compare_training_strategies.py` 来验证。

### Q5: 如何调整学习率？

**经验法则**:
- 预训练: 1e-4 (标准)
- 微调（全模型）: 1e-5 (预训练的 1/10)
- 微调（冻结编码器）: 1e-4 (与预训练相同)
- DPO: 5e-6 (更小)

---

## 文件结构

```
OPUS-BEPI/
├── train_pretrain_finetune.py          # 两阶段训练主脚本
├── run_pretrain_stage.sh               # 预训练 SLURM 脚本
├── run_finetune_stage.sh               # 微调 SLURM 脚本
├── run_finetune_param_search.sh        # 超参数搜索脚本
├── prepare_pretrain_dataset.py         # 数据准备脚本（模板）
├── compare_training_strategies.py      # 策略比较脚本
├── SCANNET_WORKFLOW.md                 # 详细文档
├── QUICKSTART.md                       # 本文档
└── data/
    ├── BCE_633/                        # 表位数据集
    └── Dockground_5K/                  # 预训练数据集（需准备）
```

---

## 下一步

1. **准备预训练数据**: 下载或提取蛋白质-蛋白质界面数据
2. **运行预训练**: `sbatch run_pretrain_stage.sh`
3. **运行微调**: `sbatch run_finetune_stage.sh`
4. **评估结果**: 比较与从头训练的差异

---

## 获取帮助

- 详细文档: 查看 `SCANNET_WORKFLOW.md`
- ScanNet 论文: https://www.nature.com/articles/s41592-022-01490-7
- GitHub: https://github.com/jertubiana/ScanNet

---

## 检查清单

在开始之前，确保：

- [ ] Conda 环境已激活 (`conda activate bepi`)
- [ ] 预训练数据集已准备 (`./data/Dockground_5K/`)
- [ ] 表位数据集存在 (`./data/BCE_633/`)
- [ ] SLURM 脚本中的路径已更新
- [ ] GPU 资源可用

准备好了？运行：
```bash
sbatch run_pretrain_stage.sh
```

祝训练顺利！🚀
