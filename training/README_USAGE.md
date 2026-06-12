# 训练脚本使用指南

## 🚀 快速开始

### 完整训练流程（推荐）
```bash
cd training
./train_full.sh
```

这会自动执行：
1. ✅ 生成 100 对测试数据
2. ✅ 生成 5000 对正式训练数据  
3. ✅ 快速验证训练脚本 (20 epoch)
4. ✅ 正式训练模型 (150 epoch)

**耗时**: 2-10 小时 (GPU) 或 28-58 小时 (CPU)

---

## 📋 高级用法

### 使用 CPU 训练
```bash
./train_full.sh --device cpu
```

### 仅执行特定步骤

#### 步骤 1: 生成测试数据
```bash
./train_full.sh --step 1
# 生成 100 对测试数据，快速验证管道
```

#### 步骤 2: 生成正式数据
```bash
./train_full.sh --step 2
# 生成 5000 对训练数据
```

#### 步骤 3: 快速验证训练
```bash
./train_full.sh --step 3
# 用小数据集快速测试，验证代码无误
```

#### 步骤 4: 正式训练
```bash
./train_full.sh --step 4
# 用完整数据训练最终模型
```

### 自定义参数

```bash
# 增加数据生成工作进程（加快数据生成）
./train_full.sh --workers 16

# 修改批大小（GPU 训练）
./train_full.sh --batch-size 64

# 组合使用
./train_full.sh --device cuda --workers 12 --batch-size 32
```

---

## 📊 预期输出

### 成功的数据生成
```
找到 1024 张源图，目标生成 100 对...
  进度: 100/100  (100.0%)
完成！共生成 100 对数据 → ./data_test
✓ 测试数据生成完成！
   生成了 100 对数据
   磁盘占用: 500 MB
```

### 成功的快速验证
```
使用设备: cuda
加载数据: ./data_test
训练集: 80, 验证集: 10, 测试集: 10

--- Epoch 1/20 ---
训练损失: 18.234567
验证损失: 15.123456, MAE: 12.34, RMSE: 15.67, R²: 0.45

--- Epoch 20/20 ---
训练损失: 8.123456
验证损失: 7.654321, MAE: 6.12, RMSE: 8.34, R²: 0.68
✓ 快速验证完成！
```

### 最终结果
```
📈 训练结果
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
关键指标:
  R² 得分:     0.852 (目标 > 0.80)
  MAE:         6.12
  Loss:        8.23
✓ 很好！R² > 0.80
```

---

## 🔍 监控训练进度

在训练过程中，打开另一个终端：

```bash
# 查看 TensorBoard（实时可视化）
tensorboard --logdir=./checkpoints/logs
# 然后访问 http://localhost:6006

# 或查看日志文件
tail -f checkpoints/logs/events.out.tfevents.*

# 或简单的进度监控
watch -n 30 'ls -1 data/ | wc -l'  # 监控数据生成进度
```

---

## ✅ 检查清单

运行脚本前：
- [ ] 有 100+ 张照片在 `training/photos/`
- [ ] PyTorch 已安装 (`python -c "import torch"`)
- [ ] Darktable 已安装 (`darktable-cli --version`)
- [ ] 磁盘空间充足 (≥ 10 GB)

脚本运行时：
- [ ] 不要关闭终端
- [ ] 可以在另一个终端监控进度
- [ ] 网络需要稳定（尤其是数据生成时）

训练完成后：
- [ ] 查看 `checkpoints/test_results.json`
- [ ] 检查 R² 是否 > 0.80
- [ ] 找到最佳模型文件

---

## 🛠️ 故障排除

### "找不到 Python"
```bash
# 检查 Python 安装
python3 --version

# 如果找不到，安装 Python 3.8+
```

### "找不到 Darktable"
```bash
# Linux
sudo apt install darktable

# macOS
brew install darktable

# 验证
darktable-cli --version
```

### "找不到 PyTorch"
```bash
# 安装 PyTorch (CPU)
pip install torch torchvision

# 或 GPU (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 验证
python -c "import torch; print(torch.__version__)"
```

### 数据生成太慢
```bash
# 增加工作进程（根据 CPU 核数调整）
./train_full.sh --workers 16  # 从 8 改为 16
```

### 训练过程中内存溢出
```bash
# 减小批大小
./train_full.sh --step 4 --device cuda --batch-size 16
```

### 训练中断想继续
```bash
# 从上次中断的地方恢复
cd training
python train.py \
  --data-dir ./data \
  --resume ./checkpoints/best_model_epoch050_r20.85.pt \
  --epochs 200 \
  --device cuda \
  --output-dir ./checkpoints
```

---

## 📈 训练后的步骤

### 1. 检查结果
```bash
cat checkpoints/test_results.json | python -m json.tool
```

### 2. 复制模型到后端
```bash
# 找到最佳模型
ls -lh checkpoints/best_model_*.pt | tail -1

# 复制到后端（创建目录如果不存在）
mkdir -p ../backend/models
cp checkpoints/best_model_epoch*.pt ../backend/models/param_predictor.pt
```

### 3. 集成到后端
见 `../ACTION_PLAN.md` 的"阶段 5️⃣"章节

### 4. 启动后端测试
```bash
cd ../backend
python main.py

# 另一个终端测试
curl http://localhost:8000/health
```

---

## 💡 常见问题

**Q: 脚本需要多长时间？**  
A: 总耗时 6-18 小时 (GPU) 或 28-58 小时 (CPU)

**Q: 中间可以暂停吗？**  
A: 可以，但仅在步骤之间（按 Ctrl+C）。训练中断会放在已生成的文件上。

**Q: 可以多次运行脚本吗？**  
A: 可以，脚本会跳过已存在的数据。

**Q: 可以修改参数吗？**  
A: 可以，使用 `--step 4 --device cuda` 等参数自定义。

**Q: 如何知道训练是否正常？**  
A: 监控 TensorBoard 的 Loss 和 R² 指标。

---

## 🔗 相关文件

- `train_full.sh` - 本脚本
- `cnn_model.py` - CNN 模型定义
- `train.py` - 训练脚本（脚本调用）
- `generate_dataset.py` - 数据生成脚本（脚本调用）
- `../ACTION_PLAN.md` - 详细实施计划
- `../NEXT_STEPS.md` - 一页纸版本

---

**有问题？查看 `../ACTION_PLAN.md` 的故障排除章节！** 🎯
