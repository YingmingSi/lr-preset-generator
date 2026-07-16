"""
LR 风格移植 - FastAPI 主应用（纯 LUT）

工作流：上传 原图 + 风格参考图 → CNN 预测 72 维颜色变换 → 烘焙成 3D LUT (.cube)
LUT 可在任何软件（LR / PS / DaVinci / 剪辑…）应用，通用且颜色一致。
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import sys
import os
import gc

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from modules.image_loader import load_image
from modules.lut_generator import bake_cube_lut
from modules.lr_image_processor import apply_lr_params
from modules.params_config import (
    GROUP_LUMINANCE, GROUP_CURVE, GROUP_COLORGRADE, GROUP_CALIBRATION, GROUP_HSL,
)
from modules.cnn_predictor import (
    load_predictor as load_cnn, predict_params as cnn_predict,
    is_predictor_loaded as cnn_ready,
)

app = FastAPI(title="LR Style LUT", version="3.0.0")
load_cnn()

app.add_middleware(
    CORSMiddleware,
    # 任意 Vercel 子域名 + 本地开发
    allow_origin_regex=(
        r"https://[a-zA-Z0-9-]+\.vercel\.app"
        r"|http://(localhost|127\.0\.0\.1):\d+"
    ),
    allow_methods=["*"],
    allow_headers=["*"],
)


def _summary(cnn_params: dict) -> dict:
    """72 维参数 → 前端分组摘要（只显示非零）"""
    def pick(keys):
        return {k: cnn_params[k] for k in keys if cnn_params.get(k, 0) != 0}
    return {
        '基础调整': pick(GROUP_LUMINANCE),
        '色调曲线': pick(GROUP_CURVE),
        '颜色分级': pick(GROUP_COLORGRADE),
        '相机校准': pick(GROUP_CALIBRATION),
        'HSL 混色器': pick(GROUP_HSL),
    }


_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _repro_stats(cnn: np.ndarray, ref: np.ndarray, K: int = 8) -> dict:
    """
    还原补偿统计量：
      · 亮度全局仿射：CNN结果 vs 参考的 luma mean/std（对齐整体明暗/对比）
      · 按明暗分档的色度偏移：K 档亮度，每档 参考色度均值 − CNN色度均值
        （让阴影拉向参考阴影色、高光拉向参考高光色 → 还原色调分离）
    cnn/ref: (N,3) float [0,1]
    """
    Lc = cnn @ _LUMA
    Lr = ref @ _LUMA
    # 色度 = RGB − 自身亮度（去亮度，仅保留颜色偏移）
    chroma_c = cnn - Lc[:, None]
    chroma_r = ref - Lr[:, None]
    edges = np.linspace(0.0, 1.0, K + 1)
    centers, delta = [], []
    for b in range(K):
        lo, hi = edges[b], edges[b + 1]
        mc = (Lc >= lo) & (Lc <= hi if b == K - 1 else Lc < hi)
        mr = (Lr >= lo) & (Lr <= hi if b == K - 1 else Lr < hi)
        dc = chroma_c[mc].mean(0) if mc.any() else np.zeros(3, np.float32)
        dr = chroma_r[mr].mean(0) if mr.any() else np.zeros(3, np.float32)
        centers.append(round(float((lo + hi) / 2), 4))
        delta.append((dr - dc).round(5).tolist())
    return {
        "cnn_Lmean": round(float(Lc.mean()), 5),
        "cnn_Lstd":  round(float(Lc.std() + 1e-4), 5),
        "ref_Lmean": round(float(Lr.mean()), 5),
        "ref_Lstd":  round(float(Lr.std() + 1e-4), 5),
        "bins":      centers,       # K 个亮度档中心
        "delta":     delta,         # K×3 每档色度偏移（RGB）
    }


def _align(src_data, ref_data):
    """src 与 ref 尺寸对齐"""
    if src_data['rgb_float'].shape == ref_data['rgb_float'].shape:
        return src_data
    from scipy.ndimage import zoom
    h, w = ref_data['rgb_float'].shape[:2]
    hs, ws = src_data['rgb_float'].shape[:2]
    scale = (h / hs, w / ws)
    src_data['rgb_float'] = np.stack(
        [zoom(src_data['rgb_float'][:, :, c], scale, order=1) for c in range(3)], axis=2)
    return src_data


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "cnn_loaded": cnn_ready(),
        "model": "color_cnn_v9 (61参数 → 3D LUT)" if cnn_ready() else None,
    }


@app.post("/analyze")
async def analyze(
    src_image:   UploadFile = File(...),
    ref_image:   UploadFile = File(...),
    preset_name: str = Form("AI Style"),
):
    src_bytes = await src_image.read()
    ref_bytes = await ref_image.read()
    if not src_bytes or not ref_bytes:
        raise HTTPException(400, "需要同时上传原图和风格参考图")
    if not cnn_ready():
        raise HTTPException(503, "模型未加载")

    try:
        src_data = load_image(src_bytes, src_image.filename or "src.jpg")
        ref_data = load_image(ref_bytes, ref_image.filename or "ref.jpg")
        src_data = _align(src_data, ref_data)

        src_rgb = (src_data['rgb_float'] * 255).clip(0, 255).astype(np.uint8)
        ref_rgb = (ref_data['rgb_float'] * 255).clip(0, 255).astype(np.uint8)
        params = cnn_predict(src_rgb, ref_rgb)

        # CNN 预测的 LR 参数 → 烘焙成 3D LUT
        lut_content = bake_cube_lut(params, size=33, title=preset_name)

        # 还原补偿统计量：亮度全局仿射 + 按明暗分档的色度偏移（还原色调分离）
        cnn_res = apply_lr_params(src_rgb, params, skip_local=True).astype(np.float32) / 255.0
        ref_f = ref_rgb.astype(np.float32) / 255.0
        repro = _repro_stats(cnn_res.reshape(-1, 3), ref_f.reshape(-1, 3))

        response = JSONResponse({
            "success":     True,
            "summary":     _summary(params),
            "lut_content": lut_content,
            "repro":       repro,
        })
        del src_data, ref_data, src_bytes, ref_bytes, params
        gc.collect()
        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"生成失败：{str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
