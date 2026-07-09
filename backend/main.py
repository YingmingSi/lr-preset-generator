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
from modules.xmp_generator import generate_xmp, params_summary, generate_xmp_from_cnn
from modules.params_config import (
    GROUP_LUMINANCE, GROUP_CURVE, GROUP_COLORGRADE, GROUP_CALIBRATION, GROUP_HSL,
)


def _cnn_summary(cnn_params: dict) -> dict:
    """把 72 维 CNN 参数整理成前端展示的分组摘要（只显示非零）"""
    def pick(keys):
        return {k: cnn_params[k] for k in keys if cnn_params.get(k, 0) != 0}
    return {
        '基础调整': pick(GROUP_LUMINANCE),
        '色调曲线': pick(GROUP_CURVE),
        '颜色分级': pick(GROUP_COLORGRADE),
        '相机校准': pick(GROUP_CALIBRATION),
        'HSL 混色器': pick(GROUP_HSL),
    }
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
    """
    把 CNN 预测的 22 维参数注入到 luminance/color 字典。
    同时清零 CNN 没预测的 ColorGrade 参数（避免传统分析的高数值主导颜色）。
    """
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

    # 清零 CNN 没预测但传统分析可能产生的"现代颜色分级"参数。
    # 这些参数（ColorGrade*）不在 CNN 22 维输出中，由 color_analyzer
    # 独立计算，对分布外照片往往产生过高数值（如 MidtoneSat=35），导致
    # 整体颜色被错误主导。清零让颜色完全由 CNN 预测的 HSL + SplitToning 决定。
    COLOR_GRADE_RESET_KEYS = (
        'ColorGradeMidtoneHue', 'ColorGradeMidtoneSat', 'ColorGradeMidtoneLum',
        'ColorGradeShadowLum', 'ColorGradeHighlightLum',
        'ColorGradeGlobalHue', 'ColorGradeGlobalSat', 'ColorGradeGlobalLum',
    )
    for k in COLOR_GRADE_RESET_KEYS:
        color_params[k] = 0


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


def _predict_with_cnn(src_data, ref_data, boldness: float = 1.0):
    """运行 CNN 推理；失败时返回 None"""
    if src_data is None or not cnn_ready():
        return None
    try:
        src_rgb = (src_data['rgb_float'] * 255).clip(0, 255).astype(np.uint8)
        ref_rgb = (ref_data['rgb_float'] * 255).clip(0, 255).astype(np.uint8)
        return cnn_predict(src_rgb, ref_rgb, boldness=boldness)
    except Exception as e:
        print(f"⚠ CNN 预测失败: {e}")
        return None


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "cnn_loaded": cnn_ready(),
        "cnn_model": "color_cnn_v7 (72参数, R²=0.45)" if cnn_ready() else None,
    }


@app.post("/analyze")
async def analyze(
    ref_image:    UploadFile = File(...),
    src_image:    UploadFile = File(None),
    preset_name:  str = Form("AI生成预设"),
    boldness:     float = Form(1.0),
):
    ref_bytes = await ref_image.read()
    src_bytes = await src_image.read() if src_image else None

    if not ref_bytes:
        raise HTTPException(400, "参考图不能为空")

    boldness = max(0.5, min(2.0, boldness))  # 安全区间

    try:
        ref_data = load_image(ref_bytes, ref_image.filename or "ref.jpg")
        src_data = load_image(src_bytes, src_image.filename or "src.jpg") if src_bytes else None
        src_data = _align_src_to_ref(src_data, ref_data)
        mode     = "B_dual" if src_data else "A_single"

        # CNN 预测（双图模式）
        cnn_params = _predict_with_cnn(src_data, ref_data, boldness=boldness)

        if cnn_params is not None:
            # 双图模式：CNN 预测 72 维参数，直接生成 XMP
            xmp_content = generate_xmp_from_cnn(cnn_params, preset_name=preset_name)
            summary = _cnn_summary(cnn_params)
            curve_style = ''
        else:
            # 单图模式：退化到传统分析
            luminance_params = analyze_luminance(ref_data, src_data)
            luminance_params = apply_luminance_linkage(luminance_params)
            color_params     = analyze_color(ref_data, src_data)
            empty_scene = {'params': {}, 'report': {}}
            xmp_content = generate_xmp(luminance_params, color_params, empty_scene,
                                        preset_name=preset_name)
            summary = params_summary(luminance_params, color_params, empty_scene)
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
        del ref_data, src_data, ref_bytes, src_bytes, cnn_params
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
    boldness:     float = Form(1.0),
):
    ref_bytes = await ref_image.read()
    src_bytes = await src_image.read() if src_image else None
    boldness = max(0.5, min(2.0, boldness))

    try:
        ref_data = load_image(ref_bytes, ref_image.filename or "ref.jpg")
        src_data = load_image(src_bytes, src_image.filename or "src.jpg") if src_bytes else None
        src_data = _align_src_to_ref(src_data, ref_data)

        cnn_params = _predict_with_cnn(src_data, ref_data, boldness=boldness)

        if cnn_params is not None:
            xmp_content = generate_xmp_from_cnn(cnn_params, preset_name=preset_name)
        else:
            luminance_params = analyze_luminance(ref_data, src_data)
            luminance_params = apply_luminance_linkage(luminance_params)
            color_params     = analyze_color(ref_data, src_data)
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
