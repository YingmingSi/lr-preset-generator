# Lightroom Preset Forge

AI驱动的Lightroom XMP预设生成器。上传参考图，自动分析色彩风格，生成可导入Lightroom Classic的XMP文件。

---

## 快速启动

### 1. 安装后端依赖

```bash
pip install fastapi uvicorn python-multipart Pillow numpy scikit-image rawpy anthropic
```

### 2. 配置API Key

```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

### 3. 启动后端

```bash
cd backend
python main.py
# 后端运行在 http://localhost:8000
```

### 4. 启动前端

```bash
cd frontend
npm install
npm run dev
# 前端运行在 http://localhost:5173
```

---

## 使用方式

### 模式A：单张参考图
- 上传一张你想模仿风格的图片
- 点击「生成预设」
- 精度约65-75%，适合快速获得大致方向

### 模式B：参考图 + 原图
- 上传风格参考图 + 你手头的原图（支持RAW格式）
- 算法通过差值分析推导更精确的调整量
- 精度约80-90%

---

## 项目结构

```
lrpreset/
├── backend/
│   ├── main.py                  # FastAPI应用入口
│   ├── modules/
│   │   ├── image_loader.py      # 图像加载（RAW/JPG/PNG）
│   │   ├── luminance_analyzer.py # 亮度分析（直方图/曲线）
│   │   ├── color_analyzer.py    # 色彩分析（HSL/颜色分级）
│   │   ├── scene_analyzer.py    # Claude场景识别+参数校正
│   │   └── xmp_generator.py     # XMP文件生成
│   └── templates/
│       └── preset_template.xmp  # XMP模板
└── frontend/
    └── src/
        └── App.jsx              # React前端
```

---

## 支持的RAW格式

Canon CR2/CR3, Nikon NEF, Sony ARW, Fuji RAF, Adobe DNG, Olympus ORF, Pentax PEF

---

## 参数覆盖范围

- ✅ 基础调整（曝光/对比度/高光/阴影/白色/黑色）
- ✅ 色调曲线（亮度曲线 + RGB分量曲线）
- ✅ HSL（色相/饱和度/明度 × 8色相）
- ✅ 颜色分级（阴影/中间调/高光三向色轮）
- ✅ 清晰度/自然饱和度/颗粒感
- ⚠️ 白平衡（仅给方向，建议手动校准）
- ❌ 局部调整/蒙版（计划后续版本支持）
