# 🚀 CNN 实现 — 快速开始（5 分钟）

## 你需要做什么？

### 你的思路 ✅
1. **数据生成**: 用 Darktable 渲染 (原图 + 随机参数) → 参考图 ✅ 已有脚本
2. **监督学习**: CNN 学习 (原图, 参考图) → 参数 ✅ 已实现
3. **反向推理**: 给定原图和参考图 → 推断参数 ✅ 已实现

---

## 立即开始（3 步）

### 步骤 1: 生成训练数据（1-3 小时）

```bash
cd training

# 先小规模测试（100 对，验证管道）
python generate_dataset.py \
  --src-dir /path/to/photos \
  --out-dir ./data_test \
  --n-pairs 100 \
  --n-workers 4

# 如果成功，生成正式数据（5000 对）
python generate_dataset.py \
  --src-dir /path/to/photos \
  --out-dir ./data \
  --n-pairs 5000 \
  --n-workers 8
```

**期望输出**: `data/` 目录包含 5000+ 组 `*_src.jpg`, `*_ref.jpg`, `*_params.json`

### 步骤 2: 训练模型（2-8 小时）

```bash
# GPU 推荐（快 10 倍）
python train.py \
  --data-dir ./data \
  --epochs 100 \
  --batch-size 32 \
  --device cuda \
  --output-dir ./checkpoints

# 或 CPU
python train.py \
  --data-dir ./data \
  --epochs 100 \
  --batch-size 16 \
  --device cpu \
  --output-dir ./checkpoints

# 监控进度（另开终端）
tensorboard --logdir=./checkpoints/logs
# 访问 http://localhost:6006
```

**期望输出**: `checkpoints/best_model_epoch*.pt` (R² > 0.80)

### 步骤 3: 集成到后端（5 分钟）

```bash
# 复制模型
mkdir -p ../backend/models
cp ./checkpoints/best_model_epoch*.pt ../backend/models/param_predictor.pt

# 修改 backend/main.py（见下面的代码片段）

# 启动后端
cd ../backend
python main.py
```

**在后端 main.py 添加**:
```python
from modules.cnn_predictor import CNNParameterPredictor
import torch

_cnn_predictor = None

@app.on_event("startup")
async def startup():
    global _cnn_predictor
    load_user_styles()
    load_calibration()
    load_learned()
    
    # ← 添加这部分
    model_path = os.path.join(os.path.dirname(__file__), 'models/param_predictor.pt')
    if os.path.exists(model_path):
        _cnn_predictor = CNNParameterPredictor(
            model_path=model_path,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        print("✓ CNN 预测器已加载")
```

---

## 验证安装

```bash
# 测试 CNN 模块
cd backend
python -c "
from modules.cnn_predictor import CNNParameterPredictor
import numpy as np

predictor = CNNParameterPredictor('./models/param_predictor.pt')
src = np.random.randint(0, 256, (384, 384, 3), dtype=np.uint8)
ref = np.random.randint(0, 256, (384, 384, 3), dtype=np.uint8)
params = predictor.predict(src, ref)
print('✅ CNN 工作正常')
print(f'   预测参数数: {len(params)}')
"
```

---

## 关键数字

| 指标 | 数值 |
|------|------|
| 模型大小 | ~50-100 MB |
| 单张推理时间 | 20-50ms (GPU) / 100-200ms (CPU) |
| 训练时间 | 2-8h (GPU) / 20-48h (CPU) |
| 精度目标 | R² > 0.80 |
| 训练数据量 | 5000-10000 对 |

---

## 常见问题

### Q: 我没有 GPU，能训练吗？
**A**: 可以，但会慢 10 倍。建议用 Google Colab（免费 GPU）:
```bash
# 在 Colab 中运行
!git clone https://github.com/yourrepo/lr-preset-generator.git
!cd lr-preset-generator && pip install torch torchvision
!cd training && python train.py --data-dir ./data --device cuda
```

### Q: 我没有 5000 张照片怎么办？
**A**: 开始用 500-1000 张，模型会有 R² 0.6-0.7，可以逐渐改进。或者用数据增强补充。

### Q: 怎么知道模型训练好了？
**A**: 看 TensorBoard：
- 验证 Loss 应该单调下降
- 最终 R² 得分 > 0.80 就很好
- 验证 MAE 应该在 5-10 范围内

### Q: 可以不改 backend/main.py 直接用吗？
**A**: 可以，但需要手动调用：
```python
from modules.cnn_predictor import CNNParameterPredictor
predictor = CNNParameterPredictor('./models/param_predictor.pt')
params = predictor.predict(src_rgb, ref_rgb)
```

---

## 详细文档

| 文件 | 用途 |
|------|------|
| `training/README_CNN.md` | 完整训练指南 |
| `IMPLEMENTATION_PLAN.md` | 详细实施方案 |
| `training/cnn_model.py` | 模型架构 |
| `training/dataset.py` | 数据加载 |
| `training/train.py` | 训练脚本 |
| `backend/modules/cnn_predictor.py` | 推理模块 |

---

## 下一步（已实现，等你跑）

1. ✅ **数据生成**: `training/generate_dataset.py` 已准备好
2. ✅ **模型**: `training/cnn_model.py` Siamese ResNet-18
3. ✅ **训练**: `training/train.py` 完整训练循环
4. ✅ **推理**: `backend/modules/cnn_predictor.py` 集成到后端
5. ✅ **文档**: 所有指南已准备

---

## 预期结果

```
开始: 
  推理时间 500ms-2s (Claude API)
  依赖外部 API

完成后:
  推理时间 50-100ms (GPU)
  100% 本地，无外部依赖
  精度 80-95% (CNN + 现有分析融合)
```

---

## 需要帮助？

查看详细文档:
- 数据生成问题 → `training/README_CNN.md` 第 1-2 节
- 训练问题 → `training/README_CNN.md` 第 3-4 节  
- 集成问题 → `IMPLEMENTATION_PLAN.md` Phase 4
- 故障排除 → `IMPLEMENTATION_PLAN.md` 故障排除

**或直接运行命令，按报错信息查文档即可** 🚀
