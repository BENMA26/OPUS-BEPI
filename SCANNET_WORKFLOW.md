# ScanNet-Inspired Two-Stage Training Workflow

## 概述

本 workflow 基于 ScanNet 的训练策略，采用两阶段训练方法：

1. **阶段 1 (Pre-training)**: 在大规模蛋白质-蛋白质结合位点数据集上预训练，学习通用的结合位点模式
2. **阶段 2 (Fine-tuning)**: 在 B 细胞表位数据集上微调，专门化到抗体-抗原表位预测任务

### ScanNet 的启发

ScanNet (Nature Methods 2022) 采用类似策略：
- **预训练**: 在 Dockground 数据库的 ~20,000 个蛋白质-蛋白质界面上训练
- **微调**: 迁移到 B 细胞表位预测任务
- **效果**: 预训练模型学到的通用结合模式（氢键、疏水性补丁等）可以迁移到表位预测

### 与 DPO 方法的区别

| 方法 | 预训练数据 | 微调策略 | 优势 |
|------|-----------|---------|------|
| **Two-stage (本方法)** | 蛋白质-蛋白质结合位点 | 标准监督学习 | 学习通用结合模式，数据利用率高 |
| **DPO** | 表位数据 | 空间一致性偏好学习 | 改善空间连续性，无需额外数据 |
| **Combined** | 蛋白质-蛋白质结合位点 | DPO 微调 | 结合两者优势（可选） |

---

## 数据准备

### 阶段 1: 蛋白质-蛋白质结合位点数据集

你需要准备一个大规模的蛋白质-蛋白质界面数据集。推荐数据源：

1. **Dockground** (推荐)
   - 网站: https://dockground.compbio.ku.edu/
   - 规模: ~20,000 个蛋白质复合物
   - 包含: 蛋白质-蛋白质对接基准数据

2. **PDB 复合物**
   - 从 PDB 提取蛋白质-蛋白质界面
   - 使用工具: PISA, PDBePISA
   - 标注界面残基作为正样本

3. **DB5.5 / SKEMPI**
   - 较小但高质量的蛋白质-蛋白质界面数据集
   - 适合快速验证

**数据格式**: 与 BCE_633 相同的格式
```
./data/Dockground_5K/
├── train.pkl          # 训练集（蛋白质对象列表）
├── test.pkl           # 测试集
├── cross-validation.npy  # 交叉验证索引
└── [其他特征文件]
```

### 阶段 2: B 细胞表位数据集

使用现有的 BCE_633 数据集（已准备好）。

---

## 训练流程

### 阶段 1: 预训练（Binding Site Pre-training）

**目标**: 在大规模蛋白质-蛋白质结合位点数据上训练，学习通用的结合位点识别能力。

#### 单次训练

```bash
sbatch run_pretrain_stage.sh
```

**配置** (`run_pretrain_stage.sh`):
- Dataset: `Dockground_5K` (修改为你的数据集名称)
- Mode: `esm_gangxu` (ESM-2 + FoldSeek tokens)
- Epochs: 100
- Learning rate: 默认 1e-4
- Batch size: 4

#### 输出

- **模型保存路径**: `./model/Dockground_5K_GraphBepi_pretrain/model_-1.ckpt`
- **日志**: `./log/slurm_pretrain_*.out`

---

### 阶段 2: 微调（Epitope Fine-tuning）

**目标**: 在表位数据集上微调预训练模型，专门化到抗体-抗原表位预测。

#### 前置条件

确保阶段 1 的预训练模型已完成：
```bash
ls -lh ./model/Dockground_5K_GraphBepi_pretrain/model_-1.ckpt
```

#### 单次微调

```bash
sbatch run_finetune_stage.sh
```

**配置** (`run_finetune_stage.sh`):
- Dataset: `BCE_633`
- Pretrain checkpoint: `./model/Dockground_5K_GraphBepi_pretrain/model_-1.ckpt`
- Epochs: 50 (比预训练少)
- Learning rate: 1e-5 (比预训练低 10 倍)
- Batch size: 4

**两种微调策略**:

1. **全模型微调** (推荐)
   - 微调所有层
   - 学习率: 1e-5
   - 适合: 表位数据集足够大 (>500 样本)

2. **冻结编码器**
   - 只训练分类头 (MLP)
   - 学习率: 1e-4
   - 适合: 表位数据集较小，防止过拟合
   - 添加 `--freeze_encoder` 参数

#### 超参数搜索

```bash
sbatch run_finetune_param_search.sh
```

**搜索空间** (12 个组合):
- Learning rate: {1e-5, 5e-6, 1e-6}
- Freeze encoder: {True, False}
- Epochs: {30, 50}

#### 输出

- **模型保存路径**: `./model/BCE_633_GraphBepi_finetune/model_-1.ckpt`
- **日志**: `./log/slurm_finetune_*.out`

---

## 监控与评估

### 查看任务状态

```bash
# 查看所有任务
ssh g01n23_maben "squeue -u maben"

# 查看预训练日志
ssh g01n23_maben "tail -f /work/home/maben/project/epitope_prediction/GraphBepi/log/slurm_pretrain_*.out"

# 查看微调日志
ssh g01n23_maben "tail -f /work/home/maben/project/epitope_prediction/GraphBepi/log/slurm_finetune_*.out"
```

### 检查模型文件

```bash
# 预训练模型
ssh g01n23_maben "ls -lh /work/home/maben/project/epitope_prediction/GraphBepi/model/Dockground_5K_GraphBepi_pretrain/"

# 微调模型
ssh g01n23_maben "ls -lh /work/home/maben/project/epitope_prediction/GraphBepi/model/BCE_633_GraphBepi_finetune/"
```

### TensorBoard 可视化

```bash
tensorboard --logdir=./log --port=6006
```

---

## 预期效果

基于 ScanNet 的经验：

1. **预训练的好处**:
   - 学习通用的结合位点特征（氢键、疏水性、静电相互作用）
   - 提高小样本表位数据集的泛化能力
   - 减少过拟合风险

2. **性能提升**:
   - 相比从头训练，AUROC 提升 2-5%
   - 在小样本场景下提升更明显
   - 对新抗原的泛化能力更强

3. **训练时间**:
   - 预训练: 取决于数据集大小（Dockground 5K 约 1-2 天）
   - 微调: 约 4-8 小时（BCE_633）

---

## 高级用法

### 结合 DPO 的三阶段训练

可以将两阶段预训练与 DPO 结合：

```bash
# Stage 1: 预训练（蛋白质-蛋白质结合位点）
sbatch run_pretrain_stage.sh

# Stage 2: 微调（表位数据）
sbatch run_finetune_stage.sh

# Stage 3: DPO 空间一致性优化
python train_dpo_gangxu.py \
    --ref_ckpt ./model/BCE_633_GraphBepi_finetune/model_-1.ckpt \
    --dataset BCE_633 \
    --tag GraphBepi_finetune_dpo \
    --beta 0.1 --lambda_task 1.0 \
    --epochs 30 --lr 5e-6
```

### 跨数据集迁移

预训练模型可以迁移到其他表位数据集：

```bash
# 在 IEDB 数据集上微调
python train_pretrain_finetune.py \
    --stage finetune \
    --dataset IEDB \
    --pretrain_ckpt ./model/Dockground_5K_GraphBepi_pretrain/model_-1.ckpt \
    --tag GraphBepi_finetune_IEDB \
    --epochs 50 --lr 1e-5
```

---

## 故障排除

### 问题 1: 预训练 checkpoint 不存在

```
ERROR: Pre-trained checkpoint not found
```

**解决**: 先运行阶段 1 预训练
```bash
sbatch run_pretrain_stage.sh
```

### 问题 2: 微调性能不佳

**可能原因**:
1. 学习率过高 → 尝试降低到 5e-6 或 1e-6
2. 过拟合 → 使用 `--freeze_encoder` 冻结编码器
3. 预训练数据与表位数据差异大 → 增加微调 epochs

### 问题 3: 内存不足

**解决**:
- 减小 batch size: `--batch 2`
- 使用梯度累积（需修改代码）
- 使用更小的模型: `--hidden 128`

---

## 文件清单

### 新增文件

1. **train_pretrain_finetune.py** - 两阶段训练主脚本
2. **run_pretrain_stage.sh** - 预训练 SLURM 脚本
3. **run_finetune_stage.sh** - 微调 SLURM 脚本
4. **run_finetune_param_search.sh** - 微调超参数搜索脚本
5. **SCANNET_WORKFLOW.md** - 本文档

### 现有文件（保持不变）

- train.py - 原始单阶段训练
- train_dpo_gangxu.py - DPO 训练
- model.py - 模型定义
- dataset.py - 数据集类

---

## 参考文献

1. **ScanNet**: Tubiana, J., Schneidman-Duhovny, D., & Wolfson, H. J. (2022).
   ScanNet: an interpretable geometric deep learning model for structure-based
   protein binding site prediction. *Nature Methods*, 19(6), 730-739.

2. **GitHub**: https://github.com/jertubiana/ScanNet

3. **Web Server**: http://bioinfo3d.cs.tau.ac.il/ScanNet/

---

## 下一步

1. **准备预训练数据集**: 下载并处理 Dockground 或 PDB 复合物数据
2. **运行预训练**: `sbatch run_pretrain_stage.sh`
3. **运行微调**: `sbatch run_finetune_stage.sh`
4. **评估性能**: 对比与从头训练的差异
5. **（可选）结合 DPO**: 在微调后进一步优化空间一致性
