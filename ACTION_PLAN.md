# 🎯 CNN 模型训练完整行动计划

**当前日期**: 2026-06-08  
**项目状态**: 代码完成 ✅ | 环境缺失 ❌ | 数据缺失 ❌ | 集成未做 ❌

---

## 📊 当前状态扫描

### ✅ 已完成
- [x] CNN 模型架构 (`training/cnn_model.py`) — Siamese ResNet-18
- [x] 数据加载器 (`training/dataset.py`) — 自动分割、增强
- [x] 训练脚本 (`training/train.py`) — 完整循环、评估、TensorBoard
- [x] 推理模块 (`backend/modules/cnn_predictor.py`) — 准备就绪
- [x] 饱和度约束 (SplitToning ≤ 15) — 已在 4 处加入
- [x] 完整文档 (`QUICK_START_CNN.md`, `README_CNN.md`, `IMPLEMENTATION_PLAN.md`)

### ❌ 待完成
- [ ] 环境：PyTorch、Darktable、依赖包
- [ ] 数据：源图 (100+ 张) 
- [ ] 数据：生成训练集 (5000+ 对)
- [ ] 训练：快速验证 (20 epoch)
- [ ] 训练：正式训练 (100+ epoch)
- [ ] 集成：后端 CNN 模块加载

---

## 🚀 分阶段行动清单

### 阶段 1️⃣: 环境准备 (30 分钟)

#### 1.1 安装 PyTorch
```bash
# 检查当前状态
python3 -c "import torch; print(torch.__version__)" || echo "❌ PyTorch 未安装"

# 安装（选择适合你的版本）
# CPU 版本（快速安装）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 或 GPU 版本（NVIDIA GPU）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 或 GPU 版本（AMD GPU / 其他）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
```

✅ **验证**: `python3 -c "import torch; print(f'✓ PyTorch {torch.__version__}')"`

#### 1.2 安装其他 Python 依赖
```bash
cd /home/yingmingsi/projects/Lr_Preset/lr-preset-generator

pip install \
  tensorboard \
  pillow \
  numpy \
  scipy \
  scikit-image

# 后端依赖（如果还未安装）
pip install fastapi uvicorn python-multipart anthropic
```

✅ **验证**: `python3 -c "import tensorboard, PIL, numpy; print('✓ 所有依赖就绪')"`

#### 1.3 安装 Darktable
```bash
# Linux (Ubuntu/Debian)
sudo apt update && sudo apt install -y darktable

# macOS
brew install darktable

# Windows
# 从 https://www.darktable.org/install/ 下载安装器

# 验证
which darktable-cli || darktable-cli --version
```

✅ **验证**: `darktable-cli --version`

---

### 阶段 2️⃣: 数据准备 (1-5 小时)

#### 2.1 准备源图
```bash
# 创建目录
mkdir -p /home/yingmingsi/projects/Lr_Preset/lr-preset-generator/training/photos

# 放入 100+ 张照片（JPG 或 RAW）
# 建议：多种场景（风景、人物、静物）、不同亮度、不同色温
# 文件格式支持：JPG, PNG, RAW, CR2, NEF, ARW, DNG, RAF, ORF, PEF

# 验证
ls -1 training/photos | wc -l  # 应该 ≥ 100
```

**建议来源**:
- Unsplash、Pexels 免费照片库
- 自己拍摄的照片
- 现有项目测试图片

**最少准备**: 100 张  
**推荐**: 500-1000 张（更好的泛化能力）

#### 2.2 生成小规模测试数据（验证管道）
```bash
cd training

# 生成 100 对数据（约 5-10 分钟）
python generate_dataset.py \
  --src-dir ./photos \
  --out-dir ./data_test \
  --n-pairs 100 \
  --n-workers 4

# 验证输出
ls -la data_test/ | head -20
# 应该看到: 000000_src.jpg, 000000_ref.jpg, 000000_params.json, ...
```

**期望时间**: 5-10 分钟  
**期望输出**: 300 个文件 (100 对 × 3 文件)

#### 2.3 生成正式训练数据
```bash
# 生成 5000 对数据（约 2-5 小时，根据硬件）
python generate_dataset.py \
  --src-dir ./photos \
  --out-dir ./data \
  --n-pairs 5000 \
  --n-workers 8 \
  --img-size 384

# 监控进度（在另一个终端）
watch -n 10 'ls -1 data/ | wc -l'

# 完成后验证
ls data/ | wc -l  # 应该 ≈ 15000 (5000 × 3)
du -sh data/       # 应该 ≈ 5-10 GB
```

**期望时间**: 2-5 小时（取决于 CPU 数和图片大小）  
**期望输出**: ~15000 个文件（5000 对）  
**磁盘占用**: ~5-10 GB

**加速技巧**:
```bash
# 使用更多工作进程（根据 CPU 核数调整）
--n-workers 16  # 如果你有 16+ 核 CPU

# 或分批生成（先 2500 对，再 2500 对）
python generate_dataset.py --src-dir ./photos --out-dir ./data --n-pairs 2500 --n-workers 8
python generate_dataset.py --src-dir ./photos --out-dir ./data --n-pairs 5000 --n-workers 8  # 自动续传
```

---

### 阶段 3️⃣: 模型训练 (2-24 小时)

#### 3.1 快速验证训练脚本（可选但推荐）
```bash
cd training

# 用小数据集快速测试（20 epoch，约 10-30 分钟）
python train.py \
  --data-dir ./data_test \
  --epochs 20 \
  --batch-size 16 \
  --lr 0.001 \
  --output-dir ./checkpoints_test \
  --device cpu  # 或 cuda（如有 GPU）
  --num-workers 2

# 期望输出：
# - 无错误
# - 最后 Loss < 15, MAE < 10, R² > 0.5
```

**目的**: 验证代码无误，估算完整训练时间  
**期望时间**: 10-30 分钟  
**GPU 时间**: 5-10 分钟

#### 3.2 正式训练（完整模型）

**方案 A: GPU 版本（推荐，快速）**
```bash
cd training

python train.py \
  --data-dir ./data \
  --epochs 150 \
  --batch-size 32 \
  --lr 0.001 \
  --weight-decay 0.0001 \
  --backbone resnet18 \
  --output-dir ./checkpoints \
  --device cuda \
  --num-workers 4 \
  --seed 42

# 在另一个终端监控（实时可视化）
tensorboard --logdir=./checkpoints/logs
# 访问 http://localhost:6006
```

**期望时间**: 2-8 小时（取决于 GPU）  
**GPU 要求**: ≥ 6GB 显存（RTX 3060 及以上）

**方案 B: CPU 版本（无 GPU）**
```bash
cd training

python train.py \
  --data-dir ./data \
  --epochs 150 \
  --batch-size 8  # 减小批大小
  --lr 0.001 \
  --output-dir ./checkpoints \
  --device cpu \
  --num-workers 2  # 减少工作进程

# 预计 20-48 小时
```

**方案 C: 混合方案（推荐）**
```bash
# 用 Google Colab 免费 GPU 训练
# 1. 上传 data.tar.gz 到 Google Drive
# 2. 在 Colab 运行上面的 GPU 命令
# 3. 下载训练好的模型
```

#### 3.3 训练过程中监控

**TensorBoard 实时查看**:
```bash
# 在训练目录开新终端
tensorboard --logdir=./checkpoints/logs

# 打开浏览器访问
# http://localhost:6006
```

**关键指标**:
- **Loss**: 应该单调下降（从 ~20 → ~5-10）
- **MAE**: 应该逐渐减小（从 ~20 → ~5-8）
- **R²**: 应该上升（目标 > 0.80）
- **验证 R² > 训练 R²**: 说明未过拟合

**预警信号** ⚠️:
- Loss 不下降 → 降低学习率
- 验证 Loss > 训练 Loss (大幅) → 增加数据、增加正则化
- 内存溢出 → 减小 batch_size
- NaN 出现 → 降低学习率，检查数据

#### 3.4 训练中断恢复
```bash
# 如果训练被中断，可以恢复
python train.py \
  --data-dir ./data \
  --resume ./checkpoints/best_model_epoch050_r2XXXX.pt \
  --epochs 200 \
  --output-dir ./checkpoints
```

---

### 阶段 4️⃣: 模型评估 (15 分钟)

#### 4.1 查看训练结果
```bash
cd training

# 查看最终测试结果
cat checkpoints/test_results.json | python -m json.tool

# 期望看到：
# {
#   "test_metrics": {
#     "loss": 8-12,
#     "mae": 5-8,
#     "rmse": 8-12,
#     "r2_mean": 0.80-0.90,      ← 关键指标
#     "param_mae": { ... }        ← 每个参数的精度
#   }
# }
```

#### 4.2 精度评估

| 指标 | 目标 | 说明 |
|------|------|------|
| R² | > 0.80 | 优秀（解释 80% 的方差） |
| MAE | 5-8 | 平均绝对误差 |
| Exposure MAE | < 0.2 | 曝光精度 ±0.2EV |
| Highlights MAE | < 10 | 高光精度 ±10 |
| Shadows MAE | < 10 | 阴影精度 ±10 |

**如果 R² 太低** (< 0.70):
```bash
# 检查 1: 增加训练数据
python generate_dataset.py --out-dir ./data --n-pairs 10000

# 检查 2: 调整超参
python train.py --data-dir ./data --lr 0.0005 --epochs 200 --weight-decay 0.0005

# 检查 3: 更强模型
python train.py --data-dir ./data --backbone resnet34

# 检查 4: 数据质量（手工检查 data/ 中的几个样本）
```

#### 4.3 选择最佳模型
```bash
# 找到最佳模型
ls -lt checkpoints/best_model_*.pt | head -1

# 记住文件名，比如: best_model_epoch050_r20.85.pt
```

---

### 阶段 5️⃣: 后端集成 (30 分钟)

#### 5.1 复制模型到后端目录
```bash
# 创建模型目录
mkdir -p backend/models

# 复制最佳模型
cp training/checkpoints/best_model_epoch*.pt backend/models/param_predictor.pt

# 验证
ls -lh backend/models/param_predictor.pt
# 应该看到 ~50-100 MB 的文件
```

#### 5.2 在 backend/main.py 中集成 CNN

**找到启动事件**:
```bash
# 在 backend/main.py 中找到这一行
grep -n "def startup\|def on_event" backend/main.py
```

**添加 CNN 初始化代码** (在 load_learned() 之后):

在 `backend/main.py` 的导入部分添加:
```python
from modules.cnn_predictor import CNNParameterPredictor
import torch
```

然后在全局变量部分添加:
```python
_cnn_predictor = None
```

在 startup 事件中添加 (在 load_learned() 之后):
```python
@app.on_event("startup")
async def startup():
    global _cnn_predictor
    load_user_styles()
    load_calibration()
    load_learned()
    
    # ← 添加以下代码
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
```

#### 5.3 在分析中使用 CNN（可选）

如果想融合 CNN 和传统分析，在 `/analyze` 端点中修改：

**找到位置**:
```bash
grep -n "scene_result = analyze_scene_and_correct" backend/main.py
```

**在那之后添加** (在应用校准和校正前):
```python
# ─── CNN 参数预测（可选增强）────────────────
if _cnn_predictor and _cnn_predictor.is_loaded:
    try:
        src_for_cnn = src_data['rgb_float'] if src_data else ref_data['rgb_float']
        cnn_params = _cnn_predictor.predict(src_for_cnn, ref_data['rgb_float'])
        
        # 融合权重: 20% CNN + 80% 传统分析
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
# ────────────────────────────────────────────
```

#### 5.4 测试集成

```bash
# 启动后端
cd backend
export ANTHROPIC_API_KEY="sk-..."  # 如果需要
python main.py

# 期望看到：
# ✓ CNN 参数预测器已加载 (device: cuda)

# 在另一个终端测试
curl -X POST http://localhost:8000/health

# 测试上传图片
curl -X POST http://localhost:8000/analyze \
  -F "ref_image=@test_ref.jpg" \
  -F "src_image=@test_src.jpg" \
  -F "preset_name=CNN测试"

# 检查响应中是否包含参数
```

---

## 📋 完整命令速查表

### 快速启动（所有步骤合并）
```bash
#!/bin/bash
set -e

cd /home/yingmingsi/projects/Lr_Preset/lr-preset-generator

echo "=== 1️⃣ 生成测试数据 ==="
cd training
python generate_dataset.py --src-dir ./photos --out-dir ./data_test --n-pairs 100 --n-workers 4

echo "=== 2️⃣ 快速验证 ==="
python train.py --data-dir ./data_test --epochs 20 --device cuda --output-dir ./checkpoints_test

echo "=== 3️⃣ 生成正式数据 ==="
python generate_dataset.py --src-dir ./photos --out-dir ./data --n-pairs 5000 --n-workers 8

echo "=== 4️⃣ 正式训练（在后台，可选）==="
# python train.py --data-dir ./data --epochs 150 --device cuda --output-dir ./checkpoints &

echo "✅ 所有准备完成！"
```

### 按需命令

```bash
# 查看 TensorBoard
tensorboard --logdir=./training/checkpoints/logs

# 停止训练
# Ctrl+C 或 kill $(pgrep -f "train.py")

# 查看进度
ls -1 training/data | wc -l
watch -n 5 'ls -1 training/data | wc -l'

# 查看模型大小
du -sh training/checkpoints/best_model_*.pt

# 启动后端
cd backend
python main.py

# 测试后端健康状态
curl http://localhost:8000/health
```

---

## 🎯 关键里程碑检查清单

### 环境准备 ✅/❌
- [ ] PyTorch 安装成功
- [ ] Darktable 安装成功
- [ ] 所有 Python 依赖就绪

### 数据准备 ✅/❌
- [ ] 100+ 张源图在 `training/photos/`
- [ ] 100 对测试数据生成成功
- [ ] 5000+ 对正式数据生成成功

### 模型训练 ✅/❌
- [ ] 快速验证训练完成（20 epoch）
- [ ] 正式训练完成（100+ epoch）
- [ ] 最终 R² > 0.80

### 后端集成 ✅/❌
- [ ] 模型复制到 `backend/models/`
- [ ] CNN 代码添加到 `main.py`
- [ ] 后端启动无错误
- [ ] API 测试成功

---

## 🚨 常见问题与解决

### Q1: PyTorch 安装报错
```bash
# 确认 Python 版本 3.8+
python3 --version

# 清除缓存重试
pip cache purge
pip install torch --force-reinstall
```

### Q2: Darktable 找不到
```bash
# Linux
sudo apt install -y darktable

# 确认安装
which darktable-cli
darktable-cli --version
```

### Q3: 数据生成很慢
```bash
# 增加工作进程
--n-workers 16  # 根据 CPU 核数调整

# 或减小输出大小
--img-size 256  # 默认 384

# 或分批生成
python generate_dataset.py --n-pairs 2500 --n-workers 8
python generate_dataset.py --n-pairs 5000 --n-workers 8  # 自动续传
```

### Q4: 训练太慢/内存溢出
```bash
# 减小批大小
--batch-size 16  # 从 32 改为 16 或 8

# 减少工作进程
--num-workers 2  # 从 4 改为 2

# 使用 CPU
--device cpu
```

### Q5: R² 太低 (< 0.70)
```bash
# 检查 1: 增加数据
python generate_dataset.py --n-pairs 10000

# 检查 2: 调整学习率
python train.py --lr 0.0005 --epochs 200

# 检查 3: 数据质量
# 手工检查 data/ 中的图片是否正常
```

---

## ⏱️ 预计总耗时

| 阶段 | CPU | GPU |
|------|-----|-----|
| 环境准备 | 30m | 30m |
| 数据准备 | 3-5h | 3-5h |
| 快速验证 | 20m | 5m |
| 正式训练 | 24-48h | 2-8h |
| 集成测试 | 15m | 15m |
| **总计** | **28-54h** | **6-18h** |

**建议**: 
- 周末晚上启动，让训练跑一夜
- 或用 Google Colab 节省时间

---

**下一步**: 从「阶段 1️⃣: 环境准备」开始！ 🚀
