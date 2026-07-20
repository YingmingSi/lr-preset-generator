"""
LR 风格移植 - FastAPI 主应用（按色相外观匹配 → 3D LUT）

工作流：上传 原图 + 风格参考图 → 按色相 band 匹配 色相/饱和/亮度（不学数量，
内容无关，适配"不同照片风格移植"）→ 烘焙成 3D LUT (.cube)。
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
from PIL import Image

from modules.image_loader import load_image
from modules.hue_transfer import bake_hue_lut
from modules.correspondence import is_aligned, bake_correspondence_lut

app = FastAPI(title="LR Style LUT", version="4.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=(
        r"https://[a-zA-Z0-9-]+\.vercel\.app"
        r"|http://(localhost|127\.0\.0\.1):\d+"
    ),
    allow_methods=["*"],
    allow_headers=["*"],
)

# 12 个色相 band 的名字（与 hue_transfer._NB 对应，中心 (b+0.5)/12）
_BAND_NAMES = ['红', '橙', '黄', '黄绿', '绿', '青绿', '青', '蓝青', '蓝', '紫', '品红', '洋红']


def _shrink(rgb: np.ndarray, max_side: int = 256) -> np.ndarray:
    h, w = rgb.shape[:2]
    if max(h, w) <= max_side:
        return rgb
    s = max_side / max(h, w)
    return np.asarray(Image.fromarray(rgb).resize((int(w * s), int(h * s)), Image.BILINEAR))


def _summary(deltas) -> dict:
    """每 band 的色相位移 → 前端摘要（只显示有变化的）。"""
    dhue = deltas[0]
    picked = {}
    for b, name in enumerate(_BAND_NAMES):
        if abs(dhue[b]) < 2e-3:
            continue
        picked[name] = f"色相{dhue[b] * 360:+.0f}°"
    return {'色相迁移': picked}


@app.get("/health")
async def health():
    return {"status": "ok", "engine": "hue_transfer (按色相外观匹配 → 3D LUT)"}


@app.post("/analyze")
async def analyze(
    src_image:   UploadFile = File(...),
    ref_image:   UploadFile = File(...),
    preset_name: str = Form("AI Style"),
    mode:        str = Form("auto"),      # auto / A(精确复刻) / B(色相迁移)
    pull:        float = Form(0.5),       # 色相归拢强度（情况B）：0=各色保留，↑=弱色向强色归拢
):
    src_bytes = await src_image.read()
    ref_bytes = await ref_image.read()
    if not src_bytes or not ref_bytes:
        raise HTTPException(400, "需要同时上传原图和风格参考图")

    try:
        src_data = load_image(src_bytes, src_image.filename or "src.jpg")
        ref_data = load_image(ref_bytes, ref_image.filename or "ref.jpg")
        src_rgb = _shrink((src_data['rgb_float'] * 255).clip(0, 255).astype(np.uint8), 256)
        ref_rgb = _shrink((ref_data['rgb_float'] * 255).clip(0, 255).astype(np.uint8), 256)
        del src_data, ref_data, src_bytes, ref_bytes

        use_A = (mode == "A") or (mode == "auto" and is_aligned(src_rgb, ref_rgb))
        if use_A:
            # 情况A：同一张图调色 → 空间对应，精确复刻（含色相旋转）
            if ref_rgb.shape != src_rgb.shape:
                ref_rgb = np.asarray(Image.fromarray(ref_rgb).resize(
                    (src_rgb.shape[1], src_rgb.shape[0]), Image.BILINEAR))
            lut_content = bake_correspondence_lut(src_rgb, ref_rgb, size=33, title=preset_name)
            summary = {'迁移模式': {'情况A · 空间对应': '精确复刻参考调色（含色相旋转）'}}
            mode = "A"
        else:
            # 情况B：不同照片 → 按色相外观匹配（内容无关）
            lut_content, deltas = bake_hue_lut(src_rgb, ref_rgb, size=33, title=preset_name,
                                               pull=float(np.clip(pull, 0, 1)))
            summary = _summary(deltas)
            summary['迁移模式'] = {'情况B · 按色相匹配': '内容无关的颜色迁移'}
            mode = "B"

        response = JSONResponse({
            "success":     True,
            "mode":        mode,
            "summary":     summary,
            "lut_content": lut_content,
        })
        del src_rgb, ref_rgb, lut_content
        gc.collect()
        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"生成失败：{str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
