# 接下来你要做的事 — 一页纸版本

## 当前状态
✅ **代码**: 完成  
❌ **环境**: 缺失 (PyTorch, Darktable)  
❌ **数据**: 缺失  
❌ **集成**: 未做  

---

## 5 个关键步骤

### 1️⃣ 环境 (30 分钟)
```bash
# PyTorch
pip install torch torchvision  # CPU版
# 或 GPU: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Darktable
sudo apt install darktable  # Linux
brew install darktable      # macOS

# 其他依赖
pip install tensorboard pillow numpy scipy scikit-image
```

### 2️⃣ 数据 - 准备源图 (手动)
```bash
# 放 100-1000 张照片到这里
mkdir -p training/photos
# 放入 JPG/RAW 文件...
```

### 3️⃣ 数据 - 生成训练集 (1-5 小时)
```bash
cd training

# 快速测试 (100对，10分钟)
python generate_dataset.py --src-dir ./photos --out-dir ./data_test --n-pairs 100 --n-workers 4

# 正式数据 (5000对，2-5小时)
python generate_dataset.py --src-dir ./photos --out-dir ./data --n-pairs 5000 --n-workers 8
```

### 4️⃣ 训练模型 (2-24 小时)
```bash
cd training

# GPU 版本（推荐，2-8小时）
python train.py --data-dir ./data --epochs 150 --device cuda --output-dir ./checkpoints

# CPU 版本（24-48小时）
python train.py --data-dir ./data --epochs 150 --device cpu --output-dir ./checkpoints

# 监控进度（另开终端）
tensorboard --logdir=./checkpoints/logs
# 访问 http://localhost:6006
```

### 5️⃣ 集成到后端 (30 分钟)
```bash
# 复制模型
mkdir -p backend/models
cp training/checkpoints/best_model_epoch*.pt backend/models/param_predictor.pt

# 修改 backend/main.py：
# 1. 添加导入:
#    from modules.cnn_predictor import CNNParameterPredictor
#    import torch
#
# 2. 添加全局变量:
#    _cnn_predictor = None
#
# 3. 在 startup() 事件中添加（load_learned() 之后）:
#    model_path = os.path.join(os.path.dirname(__file__), 'models/param_predictor.pt')
#    if os.path.exists(model_path):
#        _cnn_predictor = CNNParameterPredictor(model_path=model_path, device='cuda' if torch.cuda.is_available() else 'cpu')
#        print("✓ CNN 参数预测器已加载")

# 启动后端
cd backend
python main.py

# 测试
curl http://localhost:8000/health
```

---

## ⏱️ 预计时间
- 环境: 30 分钟
- 数据: 3-5 小时
- 训练: 2-8 小时 (GPU) / 24-48 小时 (CPU)
- 集成: 30 分钟
- **总计: 6-18 小时 (GPU) 或 28-54 小时 (CPU)**

**建议**: 
- 有 GPU 立即开始
- 无 GPU 用 Google Colab (免费)

---

## 📋 详细指南
- 完整步骤 → `ACTION_PLAN.md`
- 快速参考 → `QUICK_START_CNN.md`
- 训练细节 → `training/README_CNN.md`
- 实施方案 → `IMPLEMENTATION_PLAN.md`

---

## ✅ 验证清单
- [ ] PyTorch 安装：`python3 -c "import torch; print(torch.__version__)"`
- [ ] Darktable 安装：`darktable-cli --version`
- [ ] 源图 100+：`ls -1 training/photos | wc -l`
- [ ] 测试数据生成：`ls training/data_test | wc -l` ≈ 300
- [ ] 正式数据生成：`ls training/data | wc -l` ≈ 15000
- [ ] 训练成功：查看 `training/checkpoints/test_results.json`，R² > 0.80
- [ ] 模型复制：`ls -lh backend/models/param_predictor.pt`
- [ ] 后端启动：`python backend/main.py` 无错误
- [ ] API 工作：`curl http://localhost:8000/health`

---

**现在就开始吧！** 🚀 从环境准备开始 → 详见 `ACTION_PLAN.md`
