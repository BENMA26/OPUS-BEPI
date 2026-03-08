# ESM GangXu + DPO Fine-tuning 训练流程

## 概述

本流程分为两个阶段：
1. **阶段 1**: 训练 ESM GangXu 基础模型（使用 ESM-2 + FoldSeek tokens）
2. **阶段 2**: 使用 DPO 进行空间一致性微调，并进行参数搜索

## 阶段 1: 训练基础模型

### 特征配置
- **Mode**: `esm_gangxu`
- **特征维度**: 2581 (ESM-2: 2560 + FoldSeek tokens: 21)
- **数据集类**: `PDB_foldseek_tokens`
- **特征来源**: GangXu 的 FoldSeek tokens (`_FOLD_SEEK_GANGXU` 目录)

### 训练命令

```bash
sbatch run_train_esm_gangxu.sh
```

### 训练参数
- Dataset: BCE_633
- Hidden dim: 256
- Batch size: 4
- Epochs: 100
- GPU: 0
- Fold: -1 (使用所有数据)
- Seed: 2022

### 输出
- **模型保存路径**: `./model/BCE_633_GraphBepi_gangxu/model_-1.ckpt`
- **日志**: `./log/slurm_esm_gangxu_*.out`

## 阶段 2: DPO 参数搜索

### 前置条件
确保阶段 1 的基础模型已训练完成，checkpoint 文件存在：
```bash
./model/BCE_633_GraphBepi_gangxu/model_-1.ckpt
```

### 参数搜索配置

总共 18 个参数组合：

| 参数 | 候选值 | 说明 |
|------|--------|------|
| Beta | 0.05, 0.1, 0.2 | DPO KL penalty (控制与参考模型的偏离程度) |
| Lambda | 0.5, 1.0, 2.0 | Task loss weight (平衡 DPO loss 和 BCE loss) |
| Learning rate | 1e-5, 5e-6 | DPO 微调学习率 |

### 训练命令

```bash
sbatch run_dpo_param_search.sh
```

### DPO 训练参数
- Reference checkpoint: `./model/BCE_633_GraphBepi_gangxu/model_-1.ckpt`
- Feature dim: 2581
- Hidden dim: 256
- Batch size: 4
- Epochs: 50
- GPU: 0
- Fold: -1

### 输出
- **模型保存路径**: `./model/BCE_633_dpo_gangxu_beta{beta}_lambda{lambda}_lr{lr}/model_dpo_-1.ckpt`
- **日志**: `./log/slurm_dpo_*_*.out` (每个参数组合一个)

## 当前状态

✅ **阶段 1 进行中**: Job ID 144814 (已提交并运行)
⏳ **阶段 2 待执行**: 等待阶段 1 完成后再提交

## 监控命令

```bash
# 查看任务状态
ssh g01n23_maben "squeue -u maben"

# 查看基础模型训练日志
ssh g01n23_maben "tail -f /work/home/maben/project/epitope_prediction/GraphBepi/log/slurm_esm_gangxu_144814.out"

# 检查模型是否训练完成
ssh g01n23_maben "ls -lh /work/home/maben/project/epitope_prediction/GraphBepi/model/BCE_633_GraphBepi_gangxu/"
```

## 下一步

1. 等待基础模型训练完成（约需数小时到一天）
2. 确认 checkpoint 文件存在
3. 提交 DPO 参数搜索任务：
   ```bash
   ssh g01n23_maben "cd /work/home/maben/project/epitope_prediction/GraphBepi && sbatch run_dpo_param_search.sh"
   ```
