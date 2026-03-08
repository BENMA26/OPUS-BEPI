# ScanNet-Inspired Two-Stage Training Workflow

## 项目概述

基于 ScanNet (Nature Methods 2022) 的训练策略，为 OPUS-BEPI 构建了两阶段训练 workflow：

1. **阶段 1**: 在蛋白质-蛋白质结合位点数据上预训练
2. **阶段 2**: 在 B 细胞表位数据上微调

---

## 创建的文件清单

### 核心训练脚本

| 文件 | 用途 | 说明 |
|------|------|------|
| `train_pretrain_finetune.py` | 两阶段训练主脚本 | 支持 `--stage pretrain/finetune`，处理预训练和微调逻辑 |
| `prepare_pretrain_dataset.py` | 数据准备脚本 | 从 PDB 复合物提取蛋白质-蛋白质界面（模板，需调整） |
| `compare_training_strategies.py` | 策略比较脚本 | 对比从头训练 vs 预训练+微调的性能 |

### SLURM 提交脚本

| 文件 | 用途 | 说明 |
|------|------|------|
| `run_pretrain_stage.sh` | 预训练任务提交 | 在蛋白质-蛋白质结合位点数据上训练 |
| `run_finetune_stage.sh` | 微调任务提交 | 在表位数据上微调预训练模型 |
| `run_finetune_param_search.sh` | 超参数搜索 | 并行搜索 12 个微调超参数组合 |

### 文档

| 文件 | 用途 | 说明 |
|------|------|------|
| `SCANNET_WORKFLOW.md` | 详细技术文档 | 完整的 workflow 说明、参数配置、故障排除 |
| `QUICKSTART.md` | 快速开始指南 | 3 步快速上手，常见问题解答 |
| `README_SCANNET.md` | 本文档 | 文件清单和使用说明 |

---

## 使用流程

### 方案 A: 完整两阶段训练（推荐）

```bash
# 1. 准备预训练数据（如果还没有）
python prepare_pretrain_dataset.py \
    --input_dir ./data/pdb_complexes \
    --output_dir ./data/Dockground_5K

# 2. 预训练
sbatch run_pretrain_stage.sh

# 3. 微调
sbatch run_finetune_stage.sh

# 4. （可选）超参数搜索
sbatch run_finetune_param_search.sh
```

### 方案 B: 仅微调（如果已有预训练模型）

```bash
# 直接微调
sbatch run_finetune_stage.sh
```

### 方案 C: 比较不同策略

```bash
# 对比从头训练 vs 预训练+微调
python compare_training_strategies.py \
    --dataset BCE_633 \
    --pretrain_dataset Dockground_5K \
    --gpu 0
```

---

## 与现有 Workflow 的关系

### 现有训练方法

1. **单阶段训练** (`train.py`)
   - 直接在表位数据上训练
   - 适合: 数据充足，不需要迁移学习

2. **DPO 训练** (`train_dpo_gangxu.py`)
   - 基础模型 + DPO 空间一致性优化
   - 适合: 改善预测的空间连续性

### 新增方法

3. **两阶段训练** (`train_pretrain_finetune.py`)
   - 预训练 + 微调
   - 适合: 小样本场景，需要迁移学习

4. **三阶段训练** (组合使用)
   - 预训练 + 微调 + DPO
   - 适合: 追求最佳性能

---

## 配置说明

### 预训练配置 (`run_pretrain_stage.sh`)

```bash
DATASET="Dockground_5K"      # 预训练数据集名称
TAG="GraphBepi_pretrain"     # 模型标签
MODE="esm_gangxu"            # 特征模式
EPOCHS=100                   # 训练轮数
LR=1e-4                      # 学习率
```

### 微调配置 (`run_finetune_stage.sh`)

```bash
DATASET="BCE_633"            # 表位数据集
TAG="GraphBepi_finetune"     # 模型标签
PRETRAIN_CKPT="..."          # 预训练模型路径
EPOCHS=50                    # 微调轮数（比预训练少）
LR=1e-5                      # 学习率（比预训练低）
```

### 超参数搜索空间

```bash
LRS=(1e-5 5e-6 1e-6)         # 学习率
FREEZE_OPTIONS=(0 1)         # 是否冻结编码器
EPOCHS_OPTIONS=(30 50)       # 训练轮数
# 总共 3 × 2 × 2 = 12 个组合
```

---

## 输出文件

### 预训练阶段

```
./model/Dockground_5K_GraphBepi_pretrain/
├── model_-1.ckpt            # 预训练模型 checkpoint
└── hparams.yaml             # 超参数配置

./log/
└── slurm_pretrain_*.out     # 训练日志
```

### 微调阶段

```
./model/BCE_633_GraphBepi_finetune/
├── model_-1.ckpt            # 微调后的模型
├── result_-1.pkl            # 测试集预测结果
└── hparams.yaml             # 超参数配置

./log/
└── slurm_finetune_*.out     # 微调日志
```

### 超参数搜索

```
./model/BCE_633_finetune_lr1e-5_full_ep50/
./model/BCE_633_finetune_lr5e-6_full_ep50/
./model/BCE_633_finetune_lr1e-6_full_ep50/
./model/BCE_633_finetune_lr1e-5_frozen_ep50/
...                          # 12 个模型目录
```

---

## 性能预期

基于 ScanNet 的经验：

| 指标 | 从头训练 | 预训练+微调 | 提升 |
|------|---------|------------|------|
| AUROC | 0.75 | 0.77-0.80 | +2-5% |
| AUPRC | 0.45 | 0.47-0.50 | +2-5% |
| 泛化能力 | 中等 | 较强 | ✓ |
| 训练稳定性 | 中等 | 较好 | ✓ |

**注意**: 实际效果取决于：
- 预训练数据集的质量和规模
- 预训练数据与表位数据的相似度
- 微调超参数的选择

---

## 技术细节

### 模型架构

使用与原始 OPUS-BEPI 相同的 `GraphBepi` 模型：
- GAT (Graph Attention Network)
- Bi-LSTM
- MLP 分类器

### 特征

- **ESM-2 embeddings**: 2560 维
- **FoldSeek tokens**: 21 维
- **DSSP**: 13 维
- **总计**: 2581 维 (esm_gangxu mode)

### 迁移学习策略

1. **全模型微调** (默认)
   - 所有层都参与训练
   - 学习率降低 10 倍
   - 适合数据充足的场景

2. **冻结编码器**
   - 只训练 MLP 分类器
   - 学习率保持不变
   - 适合小样本场景

---

## 故障排除

### 问题 1: 预训练 checkpoint 不存在

```bash
ERROR: Pre-trained checkpoint not found
```

**解决**: 先运行预训练
```bash
sbatch run_pretrain_stage.sh
```

### 问题 2: 微调性能不如从头训练

**可能原因**:
1. 预训练数据与表位数据差异太大
2. 学习率不合适
3. 微调轮数不够

**解决**:
```bash
# 尝试超参数搜索
sbatch run_finetune_param_search.sh

# 或手动调整学习率
python train_pretrain_finetune.py \
    --stage finetune \
    --lr 5e-6  # 更小的学习率
```

### 问题 3: 内存不足

**解决**:
```bash
# 减小 batch size
--batch 2

# 或使用梯度累积（需修改代码）
```

---

## 下一步开发

### 可能的改进

1. **数据增强**
   - 在预训练阶段使用更多数据增强
   - 旋转、平移蛋白质结构

2. **多任务学习**
   - 同时预测结合位点和表位
   - 共享编码器，独立分类头

3. **对比学习**
   - 使用对比损失函数
   - 学习更好的表征

4. **集成学习**
   - 训练多个预训练模型
   - 集成预测结果

---

## 参考资料

### ScanNet 论文

- **标题**: ScanNet: an interpretable geometric deep learning model for structure-based protein binding site prediction
- **作者**: Tubiana, J., Schneidman-Duhovny, D., & Wolfson, H. J.
- **期刊**: Nature Methods (2022)
- **链接**: https://www.nature.com/articles/s41592-022-01490-7

### 相关资源

- **ScanNet GitHub**: https://github.com/jertubiana/ScanNet
- **ScanNet Web Server**: http://bioinfo3d.cs.tau.ac.il/ScanNet/
- **Dockground**: https://dockground.compbio.ku.edu/

---

## 联系与支持

如有问题或建议，请：
1. 查看 `SCANNET_WORKFLOW.md` 详细文档
2. 查看 `QUICKSTART.md` 快速开始指南
3. 参考 ScanNet 原始论文和代码

---

## 版本历史

- **v1.0** (2026-03-08): 初始版本
  - 实现两阶段训练 workflow
  - 添加超参数搜索
  - 创建完整文档

---

## 许可

本 workflow 基于 OPUS-BEPI 项目，遵循相同的许可协议。
ScanNet 方法的引用请参考原始论文。
