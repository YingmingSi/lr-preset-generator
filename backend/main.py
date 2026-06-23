"""
LR 预设生成器 - FastAPI 主应用（CNN 纯净版）

工作流：
  上传 ref（必须）+ src（可选） → CNN 预测 22 维参数 → 生成 XMP

如果只有 ref（单图模式），退化到传统色彩分析。
如果有 src + ref（双图模式），CNN 提供权威预测。
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
import sys
import os
import gc

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from modules.image_loader import load_image
from modules.luminance_analyzer import analyze_luminance, apply_luminance_linkage
from modules.color_analyzer import analyze_color
from modules.xmp_generator import generate_xmp, params_summary
from modules.cnn_predictor import (
    load_predictor as load_cnn, predict_params as cnn_predict,
    is_predictor_loaded as cnn_ready,
)


app = FastAPI(title="LR Preset Generator", version="2.0.0")

# 启动时加载 CNN 模型
load_cnn()

app.add_middleware(
    CORSMiddleware,
    # 允许：
    #   - 任意 Vercel 子域名（包括 preview deployments）
    #   - 本地开发（localhost / 127.0.0.1 任意端口）
    allow_origin_regex=(
        r"https://[a-zA-Z0-9-]+\.vercel\.app"
        r"|http://(localhost|127\.0\.0\.1):\d+"
    ),
    allow_methods=["*"],
    allow_headers=["*"],
)


def _inject_cnn_params(luminance_params: dict, color_params: dict, cnn_params: dict):
    """把 CNN 预测的 22 维参数注入到 luminance/color 字典"""
    LUM_KEYS = ('Exposure', 'Highlights', 'Shadows', 'Blacks', 'Whites',
                'Contrast', 'Clarity')
    COLOR_KEYS = ('Saturation', 'Vibrance',
                  'SaturationAdjustmentOrange', 'SaturationAdjustmentAqua',
                  'SaturationAdjustmentGreen', 'SaturationAdjustmentBlue',
                  'HueAdjustmentOrange', 'HueAdjustmentGreen', 'HueAdjustmentAqua',
                  'LuminanceAdjustmentOrange', 'LuminanceAdjustmentBlue',
                  'SplitToningShadowHue', 'SplitToningShadowSaturation',
                  'SplitToningHighlightHue', 'SplitToningHighlightSaturation')

    for k in LUM_KEYS:
        if k in cnn_params:
            luminance_params[k] = cnn_params[k]
    for k in COLOR_KEYS:
        if k in cnn_params:
            color_params[k] = cnn_params[k]


def _align_src_to_ref(src_data, ref_data):
    """如果 src 和 ref 尺寸不一致，把 src resize 到 ref 的尺寸"""
    if src_data is None or src_data['rgb_float'].shape == ref_data['rgb_float'].shape:
        return src_data
    from scipy.ndimage import zoom
    h_ref, w_ref = ref_data['rgb_float'].shape[:2]
    h_src, w_src = src_data['rgb_float'].shape[:2]
    scale = (h_ref / h_src, w_ref / w_src)
    src_data['rgb_float'] = np.stack([
        zoom(src_data['rgb_float'][:, :, c], scale, order=1)
        for c in range(3)
    ], axis=2)
    return src_data


def _predict_with_cnn(src_data, ref_data):
    """运行 CNN 推理；失败时返回 None"""
    if src_data is None or not cnn_ready():
        return None
    try:
        src_rgb = (src_data['rgb_float'] * 255).clip(0, 255).astype(np.uint8)
        ref_rgb = (ref_data['rgb_float'] * 255).clip(0, 255).astype(np.uint8)
        return cnn_predict(src_rgb, ref_rgb)
    except Exception as e:
        print(f"⚠ CNN 预测失败: {e}")
        return None


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "cnn_loaded": cnn_ready(),
        "cnn_model": "color_cnn_v5 (R²=0.73, pixel_RMSE=3.9%)" if cnn_ready() else None,
    }


@app.post("/analyze")
async def analyze(
    ref_image:    UploadFile = File(...),
    src_image:    UploadFile = File(None),
    preset_name:  str = Form("AI生成预设"),
):
    ref_bytes = await ref_image.read()
    src_bytes = await src_image.read() if src_image else None

    if not ref_bytes:
        raise HTTPException(400, "参考图不能为空")

    try:
        ref_data = load_image(ref_bytes, ref_image.filename or "ref.jpg")
        src_data = load_image(src_bytes, src_image.filename or "src.jpg") if src_bytes else None
        src_data = _align_src_to_ref(src_data, ref_data)
        mode     = "B_dual" if src_data else "A_single"

        # CNN 预测（双图模式）
        cnn_params = _predict_with_cnn(src_data, ref_data)

        # 传统色彩/亮度分析（始终运行 — 提供 tone_curve、HSL 基础值等）
        luminance_params = analyze_luminance(ref_data, src_data)
        luminance_params = apply_luminance_linkage(luminance_params)
        color_params     = analyze_color(ref_data, src_data)

        # CNN 覆盖（如果可用，CNN 是 22 维参数的权威源）
        if cnn_params is not None:
            _inject_cnn_params(luminance_params, color_params, cnn_params)

        # 生成 XMP + 参数摘要
        empty_scene = {'params': {}, 'report': {}}
        xmp_content = generate_xmp(luminance_params, color_params, empty_scene,
                                    preset_name=preset_name)
        summary     = params_summary(luminance_params, color_params, empty_scene)
        curve_style = luminance_params.get('_curve_style', '')

        response = JSONResponse({
            "success":              True,
            "mode":                 mode,
            "summary":              summary,
            "xmp_content":          xmp_content,
            "compression_detected": ref_data.get('compression_suspected', False),
            "is_raw_source":        src_data.get('is_raw', False) if src_data else False,
            "curve_style":          curve_style,
            "cnn_used":             cnn_params is not None,
        })
        # 释放大对象 + GC（内存受限部署环境）
        del ref_data, src_data, ref_bytes, src_bytes
        del luminance_params, color_params, cnn_params
        gc.collect()
        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"分析失败：{str(e)}")


@app.post("/download_xmp")
async def download_xmp(
    ref_image:    UploadFile = File(...),
    src_image:    UploadFile = File(None),
    preset_name:  str = Form("AI生成预设"),
):
    ref_bytes = await ref_image.read()
    src_bytes = await src_image.read() if src_image else None

    try:
        ref_data = load_image(ref_bytes, ref_image.filename or "ref.jpg")
        src_data = load_image(src_bytes, src_image.filename or "src.jpg") if src_bytes else None
        src_data = _align_src_to_ref(src_data, ref_data)

        cnn_params = _predict_with_cnn(src_data, ref_data)

        luminance_params = analyze_luminance(ref_data, src_data)
        luminance_params = apply_luminance_linkage(luminance_params)
        color_params     = analyze_color(ref_data, src_data)

        if cnn_params is not None:
            _inject_cnn_params(luminance_params, color_params, cnn_params)

        empty_scene = {'params': {}, 'report': {}}
        xmp_content = generate_xmp(luminance_params, color_params, empty_scene,
                                    preset_name=preset_name)
        safe_name   = preset_name.replace(' ', '_').replace('/', '_')

        return Response(
            content=xmp_content.encode('utf-8'),
            media_type="application/xml",
            headers={"Content-Disposition": f'attachment; filename="{safe_name}.xmp"'},
        )

    except Exception as e:
        raise HTTPException(500, str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
