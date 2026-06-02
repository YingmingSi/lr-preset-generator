# CNN 参数预测模型 — 完整实施指南

## 核心思路

1. **数据生成**：用 `darktable-cli` 渲染 (原图 + 随机参数) → 参考图
2. **监督学习**：CNN 学习 (原图, 参考图) → 参数 的映射
3. **推理**：给定用户上传的原图和参考图，模型预测最接近的参数

---

## 完整流程

### 第 1 步：安装依赖

```bash
pip install torch torchvision torchaudio
pip install tensorboard  # 可选，用于训练可视化
```

确保 `darktable-cli` 已安装：
```bash
sudo apt install darktable  # Linux
brew install darktable     # macOS
```

### 第 2 步：生成训练数据

使用现有的 `generate_dataset.py` 生成大量 (原图, 参考图, 参数) 三元组：

```bash
cd training

# 生成 1000 对数据（约 3-10 小时，取决于机器配置）
python generate_dataset.py \
  --src-dir /path/to/photos \
  --out-dir ./data \
  --n-pairs 1000 \
  --n-workers 4

# 支持断点续传：重新运行会跳过已生成的样本
# 支持按用户校准分布采样（如果有 calibration.json）
python generate_dataset.py \
  --src-dir /path/to/photos \
  --out-dir ./data \
  --n-pairs 10000 \
  --calib ../backend/data/calibration.json
```

**数据输出格式**：
```
data/
├── 000000_src.jpg        # 原图
├── 000000_ref.jpg        # 参考图（应用参数渲染后）
├── 000000_params.json    # 参数标签 {"params": {...}}
├── 000001_src.jpg
├── 000001_ref.jpg
├── 000001_params.json
└── ...
```

**数据量建议**：
- 小规模测试：100-500 对（1-3 小时）
- 正式训练：5000-50000 对（1-2 天）
- 生产级：100000+ 对（需多机并行）

### 第 3 步：训练 CNN 模型

```bash
# 基础训练
python train.py \
  --data-dir ./data \
  --epochs 100 \
  --batch-size 32 \
  --lr 0.001 \
  --output-dir ./checkpoints

# 使用 GPU（如果可用）
python train.py \
  --data-dir ./data \
  --device cuda \
  --batch-size 64 \
  --epochs 100 \
  --output-dir ./checkpoints

# 使用更大的骨干网络
python train.py \
  --data-dir ./data \
  --backbone resnet34 \
  --batch-size 32 \
  --epochs 150 \
  --output-dir ./checkpoints

# 恢复中断的训练
python train.py \
  --data-dir ./data \
  --resume ./checkpoints/best_model_epoch050_r2XXXX.pt \
  --epochs 200 \
  --output-dir ./checkpoints
```

**训练参数说明**：
- `--lr 0.001`：初始学习率（会通过余弦衰减调整到 1e-5）
- `--batch-size 32`：批大小（GPU 可用时可增加到 64-128）
- `--weight-decay 1e-4`：正则化（防止过拟合）
- `--epochs 100`：最多训练 100 个 epoch（早停会提前终止）

**输出**：
```
checkpoints/
├── logs/
│   └── 20250602_143021/
│       ├── events.out.tfevents.XXX  # TensorBoard 日志
│       └── ...
├── best_model_epoch050_r2XXXX.pt    # 最佳模型
├── best_model_epoch070_r2YYYY.pt    # （如果继续改进）
└── test_results.json                # 最终测试指标

# 查看 TensorBoard（可视化训练过程）
tensorboard --logdir=./checkpoints/logs
# 打开 http://localhost:6006
```

### 第 4 步：评估模型

训练完成后，自动生成 `test_results.json`：

```json
{
  "test_metrics": {
    "loss": 12.34,
    "mae": 5.67,          # 平均绝对误差
    "rmse": 8.90,         # 均方根误差
    "r2_mean": 0.85,      # 平均 R² 得分（0-1，越高越好）
    "param_mae": {        # 每个参数的 MAE
      "Exposure": 0.15,
      "Highlights": 8.2,
      "Shadows": 7.1,
      ...
    }
  },
  "best_val_r2": 0.87
}
```

**指标解读**：
- **R² = 0.85**：模型解释 85% 的参数方差，很好
- **MAE**：平均误差（单位同参数范围）
  - Exposure （范围 -1.5~1.5）的 MAE 0.15 很优秀
  - Highlights （范围 -100~10）的 MAE 8.2 可以接受

### 第 5 步：集成到后端

#### 5a. 复制模型到后端目录

```bash
# 复制最佳模型
cp checkpoints/best_model_epoch050_r2XXXX.pt ../backend/models/param_predictor.pt

# 创建模型目录（如果不存在）
mkdir -p ../backend/models
```

#### 5b. 修改后端 main.py，加入 CNN 预测

编辑 `backend/main.py`：

```python
from modules.cnn_predictor import CNNParameterPredictor, is_predictor_loaded
import os

# 启动时加载 CNN 模型
_cnn_predictor = None

def _init_cnn():
    global _cnn_predictor
    model_path = os.path.join(os.path.dirname(__file__), 'models/param_predictor.pt')
    if os.path.exists(model_path):
        try:
            _cnn_predictor = CNNParameterPredictor(model_path=model_path, device='cuda' if torch.cuda.is_available() else 'cpu')
            print("✓ CNN 参数预测器已加载")
        except Exception as e:
            print(f"⚠ CNN 加载失败: {e}")
    else:
        print(f"⚠ 模型文件不存在: {model_path}")

# 在 app 启动时调用
@app.on_event("startup")
async def startup():
    load_user_styles()
    load_calibration()
    load_learned()
    _init_cnn()
```

#### 5c. 可选：在分析接口中融合 CNN 预测

在 `/analyze` 端点中：

```python
# 在 analyze 函数中，获得 luminance_params 和 color_params 后
luminance_params = analyze_luminance(ref_data, src_data)
luminance_params = apply_luminance_linkage(luminance_params)
color_params = analyze_color(ref_data, src_data)

# ─── CNN 预测（可选，快速路径）─────────────────
cnn_params = {}
if _cnn_predictor and _cnn_predictor.is_loaded:
    try:
        cnn_params = _cnn_predictor.predict(
            src_data['rgb_float'] if src_data else ref_data['rgb_float'],
            ref_data['rgb_float']
        )
        # 可选：融合 CNN 和分析结果
        blend_weight = 0.3  # 30% CNN，70% 分析
        for k in luminance_params:
            if k in cnn_params:
                luminance_params[k] = blend_weight * cnn_params[k] + (1 - blend_weight) * luminance_params[k]
        for k in color_params:
            if k in cnn_params:
                color_params[k] = blend_weight * cnn_params[k] + (1 - blend_weight) * color_params[k]
    except Exception as e:
        print(f"CNN 预测失败: {e}")
# ──────────────────────────────────────────────────
```

### 第 6 步：测试集成

启动后端，测试 CNN 预测：

```bash
cd backend
export ANTHROPIC_API_KEY="your-key"
python main.py
```

在另一个终端测试：

```bash
curl -X POST http://localhost:8000/analyze \
  -F "ref_image=@test_ref.jpg" \
  -F "src_image=@test_src.jpg" \
  -F "preset_name=CNN测试"
```

响应中会包含 CNN 预测和融合结果。

---

## 高级优化

### 1. 数据增强策略

修改 `dataset.py` 中的 `aug_transform`：

```python
self.aug_transform = transforms.Compose([
    # 几何变换
    transforms.RandomAffine(
        degrees=10,          # 旋转 ±10°
        translate=(0.1, 0.1),  # 平移 ±10%
        scale=(0.9, 1.1),    # 缩放 0.9-1.1x
    ),
    # 颜色增强（模拟相机和编辑变化）
    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.15,
        hue=0.08,
    ),
    # 随机擦除（模拟部分遮挡）
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(...),
])
```

### 2. 模型架构选择

尝试不同的骨干网络：

```bash
# ResNet-50（更大，更精准但更慢）
python train.py --data-dir ./data --backbone resnet50 --batch-size 16

# 轻量级模型（EfficientNet）— 需修改 cnn_model.py
# 或使用 MobileNetV3（移动部署）
```

### 3. 损失函数改进

修改 `train.py` 中的损失函数：

```python
# 目前使用 L1Loss，也可尝试：

# Huber Loss（L1 + L2 混合，对异常值更鲁棒）
self.criterion = nn.HuberLoss(delta=10.0)

# 加权损失（某些参数更重要）
weights = torch.tensor([
    1.0,  # Exposure（很重要）
    1.5,  # Highlights（很重要）
    1.5,  # Shadows
    1.0,  # ...
    # ... 其他参数权重
])
self.criterion = nn.L1Loss(reduction='none')

# 在训练循环中：
loss = (self.criterion(pred, target) * weights).mean()
```

### 4. 多任务学习（可选）

同时预测参数 + 图像质量评分：

```python
# 在 ParamPredictor 中添加辅助头
self.quality_head = nn.Sequential(
    nn.Linear(128, 32),
    nn.ReLU(),
    nn.Linear(32, 1),  # 输出单个质量分数
)

# 返回 (params, quality_score)
```

---

## 常见问题

### Q: 模型训练速度太慢？
- A: 
  - 使用 GPU（`--device cuda`）
  - 减小 `--batch-size` 可能加快收敛
  - 使用更小的骨干网络（`--backbone resnet18`）
  - 减少 `--num-workers` 数据加载工作

### Q: 模型过拟合？
- A:
  - 增加数据（生成更多样本）
  - 增强数据增强强度（修改 `dataset.py`）
  - 增加 `--weight-decay`（例如 5e-4）
  - 增加 Dropout（修改 `cnn_model.py`）

### Q: R² 得分很低？
- A:
  - 检查数据质量（确保 darktable 渲染正确）
  - 增加训练数据量（至少 5000 对）
  - 尝试更大的模型（`--backbone resnet34`）
  - 调整学习率（尝试 0.0005-0.002）

### Q: 推理速度？
- A: ResNet-18 在 GPU 上约 20-50 ms/样本，CPU 上约 100-200 ms

---

## 下一步：参考图匹配（可选）

一旦模型训练完毕，可以：

1. **建立参考图库**：为常见风格预生成参数预测
2. **快速风格转移**：用户上传参考图 → CNN 快速预测 → 立即生成 XMP
3. **参数搜索**：构建参数 → 风格的反向索引

---

## 文件清单

```
training/
├── cnn_model.py           # 模型架构
├── dataset.py             # 数据加载器
├── train.py               # 训练脚本
├── generate_dataset.py    # 数据生成（已有）
├── README_CNN.md          # 本文档
├── data/                  # 生成的训练数据
│   ├── 000000_src.jpg
│   ├── 000000_ref.jpg
│   ├── 000000_params.json
│   └── ...
└── checkpoints/           # 模型检查点
    ├── logs/
    ├── best_model_epoch050_r2XXXX.pt
    └── test_results.json

backend/
├── modules/
│   └── cnn_predictor.py   # 推理集成（新增）
├── models/                # 部署模型目录（新增）
│   └── param_predictor.pt # 复制的最佳模型
└── main.py                # 修改集成 CNN（需修改）
```

---

## 总结

| 阶段 | 命令 | 时间 | 产出 |
|------|------|------|------|
| 数据生成 | `generate_dataset.py` | 3-10h | `data/` |
| 模型训练 | `train.py` | 2-24h | `checkpoints/*.pt` |
| 集成部署 | 修改 `main.py` | <1h | CNN-enhanced API |

祝你训练顺利！🚀
