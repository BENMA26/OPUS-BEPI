# TP/TN/FP/FN 位置记录功能说明

## 功能概述

在模型测试阶段，系统会自动记录每条链上每个位置的预测分类结果：
- **TP (True Positive)**: 真阳性 - 正确预测为表位
- **TN (True Negative)**: 真阴性 - 正确预测为非表位
- **FP (False Positive)**: 假阳性 - 错误预测为表位
- **FN (False Negative)**: 假阴性 - 错误预测为非表位

## 修改内容

### 1. model.py 修改
在 `BaseLightningModel.on_test_epoch_end()` 方法中，测试结果文件 `result.pkl` 现在包含：
- `pred`: 预测分数
- `gt`: 真实标签
- `threshold`: 最优阈值
- `tp`: 每个位置的 TP 标记 (0/1)
- `tn`: 每个位置的 TN 标记 (0/1)
- `fp`: 每个位置的 FP 标记 (0/1)
- `fn`: 每个位置的 FN 标记 (0/1)

### 2. test.py 修改
预测脚本现在会在输出的 CSV 文件中包含以下额外列：
- `position`: 残基位置（从1开始）
- `true_label`: 真实标签（如果有）
- `classification`: 分类结果（TP/TN/FP/FN）
- `is_TP`, `is_TN`, `is_FP`, `is_FN`: 布尔标记

### 3. analyze_predictions.py (新增)
专门的分析脚本，用于详细分析测试结果。

## 使用方法

### 方法1: 使用 test_model.py 进行测试

```bash
python test_model.py \
    --dataset BCE_633 \
    --fold 0 \
    --gpu 0 \
    --tag GraphBepi
```

测试完成后，结果保存在 `./model/{dataset}_{tag}/result_{fold}.pkl`，包含完整的 TP/TN/FP/FN 信息。

### 方法2: 使用 test.py 进行预测

```bash
python test.py \
    --gpu 0 \
    --threshold 0.1763 \
    --input your_protein.pdb \
    --output ./output \
    --pdb
```

或使用 FASTA 格式：

```bash
python test.py \
    --gpu 0 \
    --threshold 0.1763 \
    --input sequences.fasta \
    --output ./output \
    --fasta
```

输出的 CSV 文件会包含每个残基的分类信息。

### 方法3: 使用 analyze_predictions.py 分析结果

对已有的测试结果进行详细分析：

```bash
python analyze_predictions.py \
    --result_path ./model/BCE_633_GraphBepi \
    --dataset BCE_633 \
    --output_dir ./analysis_results \
    --fold 0
```

这会生成：
1. **每条链的详细 CSV 文件** (`{chain_name}_detailed.csv`)：
   - 每个残基的位置、氨基酸类型
   - 真实标签和预测分数
   - 分类结果（TP/TN/FP/FN）

2. **汇总统计文件** (`summary.csv`)：
   - 每条链的 TP/TN/FP/FN 数量
   - 每条链的 precision、recall、F1、accuracy

## 输出示例

### 单链详细结果 CSV 示例
```csv
position,residue,true_label,pred_score,pred_label,classification,is_TP,is_TN,is_FP,is_FN
1,M,0,0.0523,0,TN,0,1,0,0
2,K,0,0.1234,0,TN,0,1,0,0
3,L,1,0.8765,1,TP,1,0,0,0
4,V,1,0.1456,0,FN,0,0,0,1
5,A,0,0.6543,1,FP,0,0,1,0
...
```

### 汇总统计 CSV 示例
```csv
chain_name,length,n_epitope,TP,TN,FP,FN,precision,recall,f1,accuracy
1ADQ_A,108,15,12,85,8,3,0.6000,0.8000,0.6857,0.8981
1BJ1_B,217,28,22,175,14,6,0.6111,0.7857,0.6875,0.9078
...
```

## 在代码中读取结果

```python
import torch
import pandas as pd

# 读取测试结果
result = torch.load('./model/BCE_633_GraphBepi/result_0.pkl')
pred = result['pred']
gt = result['gt']
threshold = result['threshold']
tp = result['tp']
tn = result['tn']
fp = result['fp']
fn = result['fn']

# 读取单链详细结果
df = pd.read_csv('./analysis_results/1ADQ_A_detailed.csv')
tp_positions = df[df['classification'] == 'TP']['position'].tolist()
fp_positions = df[df['classification'] == 'FP']['position'].tolist()

# 读取汇总统计
summary = pd.read_csv('./analysis_results/summary.csv')
print(summary[['chain_name', 'TP', 'TN', 'FP', 'FN', 'f1']])
```

## 注意事项

1. **阈值选择**: 系统会自动使用最优 F1 阈值，也可以手动指定
2. **内存占用**: 对于大规模测试集，TP/TN/FP/FN 张量会占用额外内存
3. **兼容性**: 修改后的代码向后兼容，旧的测试脚本仍可正常运行

## 应用场景

1. **错误分析**: 找出模型经常预测错误的位置和模式
2. **可视化**: 在蛋白质结构上标注 TP/FP/FN 位置
3. **模型改进**: 分析 FP 和 FN 的特征，针对性改进模型
4. **结果报告**: 生成详细的预测报告和统计数据
