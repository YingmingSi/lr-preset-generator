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

        # 还原补偿统计量：CNN 结果 vs 参考图的每通道 mean/std（前端全局仿射校正用）
        cnn_res = apply_lr_params(src_rgb, params, skip_local=True).astype(np.float32) / 255.0
        ref_f = ref_rgb.astype(np.float32) / 255.0
        repro = {
            "cnn_mean": cnn_res.reshape(-1, 3).mean(0).round(5).tolist(),
            "cnn_std":  (cnn_res.reshape(-1, 3).std(0) + 1e-4).round(5).tolist(),
            "ref_mean": ref_f.reshape(-1, 3).mean(0).round(5).tolist(),
            "ref_std":  (ref_f.reshape(-1, 3).std(0) + 1e-4).round(5).tolist(),
        }

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
