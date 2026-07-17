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
from modules.params_config import (
    GROUP_LUMINANCE, GROUP_CURVE, GROUP_COLORGRADE, GROUP_CALIBRATION, GROUP_HSL,
    HSL_COLORS,
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

# 影调匹配参数（情况B 不该迁移——会把亮原图压暗/抬阴影成浑浊）
_TONAL_PARAMS = {'Exposure', 'Contrast', 'Highlights', 'Shadows', 'Blacks', 'Whites'} | {
    f'LumaCurve{i}' for i in range(5)}


def _tame_colors(params: dict) -> dict:
    """情况B：CNN 常预测过强的'减饱和/压暗颜色'(如红-37、-44)——让颜色发灰、
    红等鲜艳色掉饱和。风格迁移应偏向'色相/暖调'而非抹灰，故强抑制各处的
    '减饱和'(负向饱和只留 15%)，并少压暗颜色明度(负向明度留 40%)。"""
    out = dict(params)
    for c in HSL_COLORS:
        if out.get('SaturationAdjustment' + c, 0) < 0:
            out['SaturationAdjustment' + c] = int(round(out['SaturationAdjustment' + c] * 0.15))
        if out.get('LuminanceAdjustment' + c, 0) < 0:
            out['LuminanceAdjustment' + c] = int(round(out['LuminanceAdjustment' + c] * 0.40))
    for c in ('Red', 'Green', 'Blue'):                       # 相机校准的减饱和
        if out.get(c + 'Saturation', 0) < 0:
            out[c + 'Saturation'] = int(round(out[c + 'Saturation'] * 0.20))
    for k in ('Saturation', 'Vibrance'):                     # 全局减饱和
        if out.get(k, 0) < 0:
            out[k] = int(round(out[k] * 0.30))
    return out


def _repro_stats(cnn: np.ndarray, ref: np.ndarray, K: int = 8, Q: int = 17) -> dict:
    """
    还原补偿统计量：
      · 亮度曲线匹配：CNN结果 与 参考 的 luma 分位数（可还原"压高光/提阴影"等非线性影调）
      · 按明暗分档的色度偏移：K 档亮度，每档 参考色度均值 − CNN色度均值
        （让阴影拉向参考阴影色、高光拉向参考高光色 → 还原色调分离）
    cnn/ref: (N,3) float [0,1]
    """
    Lc = cnn @ _LUMA
    Lr = ref @ _LUMA
    q = np.linspace(0.0, 1.0, Q)
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
        "cnn_Lq": np.quantile(Lc, q).round(5).tolist(),   # 亮度曲线：CNN 分位数（x）
        "ref_Lq": np.quantile(Lr, q).round(5).tolist(),   # 亮度曲线：参考分位数（y）
        "bins":   centers,       # K 个亮度档中心
        "delta":  delta,         # K×3 每档色度偏移（RGB）
    }


def _shrink(rgb: np.ndarray, max_side: int = 256) -> np.ndarray:
    """降采样 uint8 图（还原统计量只需全局分布，小图足够，省内存）"""
    from PIL import Image
    h, w = rgb.shape[:2]
    if max(h, w) <= max_side:
        return rgb
    s = max_side / max(h, w)
    return np.asarray(Image.fromarray(rgb).resize((int(w * s), int(h * s)), Image.BILINEAR))


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

        src_rgb = (src_data['rgb_float'] * 255).clip(0, 255).astype(np.uint8)
        ref_rgb = (ref_data['rgb_float'] * 255).clip(0, 255).astype(np.uint8)
        del src_data, ref_data, src_bytes, ref_bytes
        params = cnn_predict(src_rgb, ref_rgb)
        del src_rgb, ref_rgb
        # 防灰：抑制情况B 常见的过度减饱和/压暗颜色
        params_t = _tame_colors(params)

        # 两个 LUT：完整（含影调匹配）+ 仅颜色（压掉影调，保留原图明暗——情况B 首选）
        lut_content = bake_cube_lut(params_t, size=33, title=preset_name)
        params_color = {k: (0 if k in _TONAL_PARAMS else v) for k, v in params_t.items()}
        lut_color = bake_cube_lut(params_color, size=33, title=preset_name)

        response = JSONResponse({
            "success":     True,
            "summary":     _summary(params),
            "lut_content": lut_content,   # 完整（含影调）
            "lut_color":   lut_color,     # 仅颜色（保留原图明暗）
        })
        del params, params_t, params_color, lut_content, lut_color
        gc.collect()
        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"生成失败：{str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
