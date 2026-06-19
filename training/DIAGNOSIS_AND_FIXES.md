# CNN 模型 R² 负数问题诊断与修复

**问题**: 初始训练结果 R² = -0.0632（远低于目标 > 0.80）

## 根本原因分析

### 1. **参数范围不匹配** ✅ 已修复
- **症状**: 模型输出范围 [-0.19, 0.18]，但参数范围 [-100, 100]
- **原因**: 神经网络倾向于输出 [-1, 1]，但参数范围太大
- **修复**: 添加 `param_normalizer.py`，在数据加载时归一化参数到 [-1, 1]，推理时反归一化

### 2. **参数采样分布不均** ✅ 正在改进
- **症状**: 某些参数方差极小或为零
  - HueAdjustmentGreen: std = 0.0（完全没有方差）
  - SplitToningShadowSaturation: std = 0.17
  - 许多 zero_heavy 参数只有 0.5% 非零值
- **原因**: 原始采样分布设置不当
  - skew_neg 参数过度偏向 0
  - zero_heavy 参数零比例太高
- **修复**:
  - 改 skew_neg → uniform（所有 HSL Saturation）
  - zero_heavy 零比例从 70% → 30%，确保 70% 充分非零值
  - 数据生成中统一使用均匀分布

### 3. **数据集规模与质量**
- **初期**: 100 对 → 500 对 → 1372 对 → 现在 2000 对（目标 5000 对）
- **当前**: 5000 对数据生成中...

## 修复步骤

### 步骤 1: 创建参数归一化器 ✅
```python
# 文件: training/param_normalizer.py
- PARAM_RANGES: 22 个参数的实际范围统计
- ParamNormalizer 类：normalize() 和 denormalize() 方法
```

### 步骤 2: 集成到数据管道 ✅
```python
# dataset.py
- 加载参数后立即归一化到 [-1, 1]
- 训练时使用归一化参数

# train.py
- evaluate() 方法使用反归一化参数计算指标
- 指标（MAE, RMSE, R²）在原始参数空间中计算
```

### 步骤 3: 后端集成 ✅
```python
# backend/modules/cnn_predictor.py
- 推理时对模型输出进行反归一化
- 返回原始参数范围的值
```

### 步骤 4: 改进参数采样分布 ⏳
```python
# generate_dataset.py
- SaturationAdjustment*: skew_neg → uniform
- HueAdjustment*: 保持 uniform
- SplitToning: 30% 零 + 70% 均匀分布
```

## 训练进度

| 数据量 | 配置 | R² 均值 | 好的参数 | 坏的参数 | 备注 |
|------|------|--------|--------|--------|------|
| 100对 | 50 ep | -0.06 | - | - | 太小 |
| 500对 | 100 ep | -0.063 | - | - | 参数分布不均 |
| 744对 | 150 ep | -414M | 0/22 | 6/22 | 参数方差极小 |
| 1372对 | 200 ep | -10.1B | 6/22 | 3/22 | **改善！** |
| 2000对 | - | - | - | - | 数据已准备，等待训练 |
| **5000对** | - | - | - | - | **生成中...** |

## 预期结果

采用改进的采样分布和 5000 对数据：
- 所有 22 个参数都应该有充分方差
- 预期 R² > 0.70（大部分参数）
- 关键参数（Exposure, Highlights, Saturation）目标 R² > 0.85

## 下一步

1. ⏳ 等待 5000 对数据生成完成
2. 🚀 使用 5000 对数据训练 200+ epochs
3. 📊 检查 R² 分数分布
4. 🔄 如果仍不理想，考虑：
   - 增加数据到 10000+ 对
   - 调整模型架构（增加容量或层数）
   - 微调学习率和优化器参数

## 技术细节

### 参数范围（从 2000 对数据统计）
```
Exposure: [-1.5, 1.5] (跨度 3.0)
Highlights: [-100, 10] (跨度 110)
Shadows: [-30, 80] (跨度 110)
SplitToningShadowHue: [0, 360] (跨度 360)
SplitToningShadowSaturation: [0, 15] (跨度 15)
...
```

### 归一化公式
```
归一化: value_norm = (value - mid) / span
其中: mid = (lo + hi) / 2, span = (hi - lo) / 2

反归一化: value_orig = value_norm * span + mid
```

### R² 计算（在原始参数空间）
```python
all_preds_denorm = normalizer.denormalize_array(all_preds)
all_targets_denorm = normalizer.denormalize_array(all_targets)
ss_res = np.sum((all_targets_denorm - all_preds_denorm) ** 2)
ss_tot = np.sum((all_targets_denorm - np.mean(all_targets_denorm)) ** 2)
r2 = 1 - (ss_res / ss_tot)
```

## 文件修改清单

- ✅ training/param_normalizer.py (新建)
- ✅ training/dataset.py (添加归一化)
- ✅ training/train.py (添加反归一化用于评估)
- ✅ backend/modules/cnn_predictor.py (添加推理时反归一化)
- ✅ training/generate_dataset.py (改进采样分布)

## 关键学习点

1. **参数范围匹配**: 神经网络输出倾向于 [-1, 1]，需要显式归一化/反归一化
2. **数据质量 > 数据量**: 小数据集中参数方差不均会导致 R² 计算失败
3. **采样策略**: zero_heavy 参数需要特殊处理，不能单纯靠概率避免 0
4. **验证指标**: 始终在原始参数空间计算评估指标，以便与现实对应

