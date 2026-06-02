# LR Preset Generator - CNN 实现完整方案

## 📊 核心架构

```
用户上传 (原图 + 参考图)
    ↓
[分析模块] 快速推理 (现有)
    ↓
[CNN模块] 参数预测 (新增)
    ↓
[融合器] 综合优化结果
    ↓
生成 XMP 预设
```

---

## 🎯 实现步骤

### Phase 1: 数据准备（1-3 天）

#### Step 1.1: 准备源图
```bash
# 收集 100-1000 张不同风格的照片
# 支持 JPG, RAW (CR2, NEF, ARW 等)
mkdir -p training/photos
# 放入照片...
```

#### Step 1.2: 生成训练数据

```bash
cd training

# 小规模测试（100 对，验证管道）
python generate_dataset.py \
  --src-dir ./photos \
  --out-dir ./data_test \
  --n-pairs 100 \
  --n-workers 4

# 正式训练（5000-10000 对）
python generate_dataset.py \
  --src-dir ./photos \
  --out-dir ./data \
  --n-pairs 5000 \
  --n-workers 8 \
  --calib ../backend/data/calibration.json
```

**产出**:
```
training/data/
├── 000000_src.jpg
├── 000000_ref.jpg
├── 000000_params.json
├── 000001_src.jpg
├── ...
└── 004999_params.json
```

---

### Phase 2: 模型训练（2-7 天）

#### Step 2.1: 快速验证（GPU 推荐）

```bash
cd training

# 小数据集快速测试（验证管道正确性）
python train.py \
  --data-dir ./data_test \
  --epochs 20 \
  --batch-size 16 \
  --lr 0.001 \
  --output-dir ./checkpoints_test \
  --device cuda  # 使用 GPU（如果可用）

# 预期：5-10 分钟，R² 可能在 0.5-0.7（小数据集）
```

#### Step 2.2: 正式训练

```bash
python train.py \
  --data-dir ./data \
  --epochs 150 \
  --batch-size 32 \
  --lr 0.001 \
  --weight-decay 0.0001 \
  --backbone resnet18 \
  --output-dir ./checkpoints \
  --device cuda \
  --num-workers 4

# 预期：8-48 小时（取决于数据量和硬件）
# 输出：best_model_epoch*.pt (R² 目标 > 0.80)
```

#### Step 2.3: 监控训练进度（实时）

```bash
# 另开终端，实时查看 TensorBoard
tensorboard --logdir=./checkpoints/logs
# 访问 http://localhost:6006
```

**关键指标**:
- Loss: 应该单调下降
- MAE: 越小越好
- R²: 目标 > 0.80
- 验证 R² 不应低于训练 R²（否则过拟合）

---

### Phase 3: 模型评估（1 天）

#### Step 3.1: 查看测试结果

```bash
cat ./checkpoints/test_results.json | python -m json.tool

# 输出示例：
# {
#   "test_metrics": {
#     "loss": 12.5,
#     "mae": 6.2,
#     "rmse": 9.8,
#     "r2_mean": 0.82,
#     "param_mae": {
#       "Exposure": 0.18,
#       "Highlights": 9.2,
#       "Shadows": 8.1,
#       ...
#     }
#   }
# }
```

**评估标准**:
| 指标 | 目标 | 评价 |
|------|------|------|
| R² | > 0.80 | 优秀（解释 80% 的方差） |
| MAE (Exposure) | < 0.3 | 很好（误差 ±0.3EV） |
| MAE (Highlights/Shadows) | < 15 | 很好 |
| 推理时间 | < 100ms (GPU) | 实时可用 |

#### Step 3.2: 对比分析

```bash
# 抽取 10 个测试样本，比较 CNN vs 现有分析
python evaluate_cnn.py \
  --model ./checkpoints/best_model_epoch050_r2XXXX.pt \
  --data-dir ./data \
  --num-samples 10 \
  --output-dir ./evaluation_report
```

（可选脚本，用于详细诊断）

---

### Phase 4: 后端集成（1-2 天）

#### Step 4.1: 文件整理

```bash
# 复制最佳模型到后端
mkdir -p backend/models
cp training/checkpoints/best_model_epoch050_r2XXXX.pt backend/models/param_predictor.pt

# 验证文件存在
ls -lh backend/models/param_predictor.pt
# 预期：~50-100 MB（ResNet-18）
```

#### Step 4.2: 修改 backend/main.py

在 `backend/main.py` 顶部添加：

```python
# ─── CNN 模块初始化 ──────────────────────────────────────
from modules.cnn_predictor import CNNParameterPredictor
import torch

_cnn_predictor = None

@app.on_event("startup")
async def startup():
    """应用启动时初始化"""
    global _cnn_predictor
    
    # 现有初始化
    load_user_styles()
    load_calibration()
    load_learned()
    
    # CNN 模块初始化
    model_path = os.path.join(os.path.dirname(__file__), 'models/param_predictor.pt')
    if os.path.exists(model_path):
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            _cnn_predictor = CNNParameterPredictor(model_path=model_path, device=device)
            print(f"✓ CNN 参数预测器已加载 (device: {device})")
        except Exception as e:
            print(f"⚠ CNN 加载失败: {e}")
            _cnn_predictor = None
    else:
        print(f"⚠ 模型文件不存在: {model_path}")
# ──────────────────────────────────────────────────────────
```

#### Step 4.3: 在 /analyze 端点使用 CNN

在 `analyze()` 函数中，luminance + color 分析后添加：

```python
# ─── CNN 参数预测（可选增强） ────────────────────────
cnn_params = None
if _cnn_predictor and _cnn_predictor.is_loaded:
    try:
        # CNN 预测
        src_for_cnn = src_data['rgb_float'] if src_data else ref_data['rgb_float']
        cnn_params = _cnn_predictor.predict(src_for_cnn, ref_data['rgb_float'])
        
        # 融合策略：CNN 作为补充
        # blend_weight = 0.2 表示 20% CNN，80% 传统分析
        blend_weight = 0.2
        
        for param_name in ['Highlights', 'Shadows', 'Contrast']:
            if param_name in cnn_params and param_name in luminance_params:
                cnn_val = cnn_params[param_name]
                orig_val = luminance_params[param_name]
                luminance_params[param_name] = (
                    blend_weight * cnn_val + (1 - blend_weight) * orig_val
                )
                
    except Exception as e:
        print(f"⚠ CNN 预测失败: {e}")
        # 继续使用传统分析结果
# ──────────────────────────────────────────────────────────

# 在响应中添加 CNN 信息（可选）
response_data = {
    ...
    "cnn_used": _cnn_predictor is not None and _cnn_predictor.is_loaded,
    "cnn_params_sample": {k: v for k, v in list(cnn_params.items())[:5]} if cnn_params else None,
}
```

#### Step 4.4: 测试集成

```bash
# 终端 1: 启动后端
cd backend
export ANTHROPIC_API_KEY="sk-..."
python main.py
# ✓ CNN 参数预测器已加载 (device: cuda)

# 终端 2: 测试 API
curl -X POST http://localhost:8000/health
# {"status": "ok"}

# 上传测试图像
curl -X POST http://localhost:8000/analyze \
  -F "ref_image=@test_ref.jpg" \
  -F "src_image=@test_src.jpg" \
  -F "preset_name=CNN测试"

# 预期响应包含 "cnn_used": true
```

---

## 📈 预期结果对比

### 训练前（现有系统）
- **精度**: 80-90% (双图模式)
- **推理时间**: 500ms-2s (Claude API + 分析)
- **可扩展性**: 受 Claude API 限制

### 训练后（CNN 增强）
- **精度**: 80-95% (CNN + 传统融合)
- **推理时间**: 50-100ms (GPU) / 200ms (CPU)
- **可扩展性**: 无需外部 API，完全本地

---

## 🔧 故障排除

### 问题 1: 数据生成太慢

```bash
# ❌ 太慢
python generate_dataset.py --src-dir ./photos --out-dir ./data --n-pairs 5000 --n-workers 2

# ✅ 更快
python generate_dataset.py --src-dir ./photos --out-dir ./data --n-pairs 5000 --n-workers 16
# 增加 n-workers（取决于 CPU 核数）
```

### 问题 2: 显存不足

```bash
# 减小批大小
python train.py --data-dir ./data --batch-size 16  # 从 32 改为 16
# 或
python train.py --data-dir ./data --device cpu  # 使用 CPU
```

### 问题 3: 模型精度不理想（R² < 0.70）

```bash
# 检查 1: 数据质量
ls -la data/ | wc -l  # 确保有足够的样本

# 检查 2: 增加训练数据
python generate_dataset.py --out-dir ./data --n-pairs 10000

# 检查 3: 调整超参
python train.py --data-dir ./data --lr 0.0005 --epochs 200 --weight-decay 0.0005

# 检查 4: 更强的模型
python train.py --data-dir ./data --backbone resnet34
```

### 问题 4: 推理时出错

```python
# 检查模型是否加载
curl http://localhost:8000/health

# 查看日志
python main.py  # 查看启动时的错误信息

# 手动测试 CNN
python -c "
from modules.cnn_predictor import CNNParameterPredictor
import numpy as np

predictor = CNNParameterPredictor('./models/param_predictor.pt')
src = np.random.randint(0, 256, (384, 384, 3), dtype=np.uint8)
ref = np.random.randint(0, 256, (384, 384, 3), dtype=np.uint8)
params = predictor.predict(src, ref)
print('✓ CNN 工作正常')
"
```

---

## 📋 检查清单

### 前期准备
- [ ] 安装 PyTorch：`pip install torch torchvision`
- [ ] 安装 Darktable：`sudo apt install darktable`
- [ ] 收集 100+ 张源图

### 数据生成
- [ ] 生成测试数据集（100 对）验证管道
- [ ] 生成正式训练数据（5000+ 对）
- [ ] 检查 `data/` 目录结构正确

### 模型训练
- [ ] 快速测试训练（20 epoch）
- [ ] 正式训练（150 epoch）
- [ ] 监控 TensorBoard 无异常
- [ ] 最终 R² 得分 > 0.80

### 后端集成
- [ ] 复制模型文件到 `backend/models/`
- [ ] 修改 `main.py` 加入 CNN 初始化
- [ ] 测试 `/analyze` 端点
- [ ] 验证 `cnn_used` 字段为 true

### 生产部署
- [ ] 压缩模型（量化或蒸馏，可选）
- [ ] 部署到云（Railway / Docker）
- [ ] 性能监控和日志

---

## 🚀 快速启动命令（一键执行）

```bash
# 全流程自动化脚本（需创建 setup.sh）
#!/bin/bash
set -e

echo "1️⃣ 生成训练数据..."
cd training
python generate_dataset.py --src-dir ./photos --out-dir ./data --n-pairs 1000

echo "2️⃣ 训练 CNN 模型..."
python train.py --data-dir ./data --epochs 100 --output-dir ./checkpoints --device cuda

echo "3️⃣ 集成到后端..."
mkdir -p ../backend/models
cp ./checkpoints/best_model_*.pt ../backend/models/param_predictor.pt

echo "✅ 完成！后端已集成 CNN"
cd ../backend
python main.py
```

---

## 💡 下一步优化（可选）

1. **模型量化**: 减小模型大小 (50MB → 10MB)
2. **参数微调**: 针对特定使用场景调整融合权重
3. **参考库**: 预生成常见风格的参数
4. **A/B 测试**: 对比 CNN vs 传统分析的用户满意度

---

## 参考文档

- 详细训练指南: `training/README_CNN.md`
- 模型架构: `training/cnn_model.py`
- 数据加载: `training/dataset.py`
- 后端集成: `backend/modules/cnn_predictor.py`
- 训练脚本: `training/train.py`

---

**预计总时间**: 3-10 天（取决于数据生成和训练硬件）

**难度等级**: ⭐⭐⭐ (中等)

**预期收益**: 推理速度提升 10x，精度提升 5-10%，无需依赖 Claude API
