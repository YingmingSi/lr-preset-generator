"""
Lightroom预设生成器 - FastAPI主应用
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from modules.image_loader import load_image
from modules.luminance_analyzer import analyze_luminance, apply_luminance_linkage
from modules.color_analyzer import analyze_color
from modules.scene_analyzer import analyze_scene_and_correct
from modules.camera_profiles import apply_camera_compensation, get_camera_description
from modules.xmp_generator import generate_xmp, params_summary
from modules.preset_renderer import render_and_validate

app = FastAPI(title="LR Preset Generator", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://lr-preset-generator.vercel.app",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(
    ref_image:      UploadFile = File(...),
    src_image:      UploadFile = File(None),
    preset_name:    str = Form("AI生成预设"),
    ref_scene_type: str = Form("auto"),
    src_scene_type: str = Form("auto"),
    camera_brand:   str = Form(""),
):
    ref_bytes = await ref_image.read()
    src_bytes = await src_image.read() if src_image else None

    if len(ref_bytes) == 0:
        raise HTTPException(400, "参考图不能为空")

    try:
        ref_data = load_image(ref_bytes, ref_image.filename or "ref.jpg")
        src_data = load_image(src_bytes, src_image.filename or "src.jpg") if src_bytes else None
        mode     = "B_dual" if src_data else "A_single"

        luminance_params = analyze_luminance(ref_data, src_data)
        luminance_params = apply_luminance_linkage(luminance_params)
        color_params     = analyze_color(ref_data, src_data)

        camera_note = ""
        if camera_brand:
            color_params = apply_camera_compensation(color_params, camera_brand)
            camera_note  = get_camera_description(camera_brand)

        scene_result = analyze_scene_and_correct(
            ref_bytes,
            {**luminance_params, **color_params},
            src_bytes,
            ref_scene_type=ref_scene_type,
            src_scene_type=src_scene_type,
        )

        # ── 自我验证与自动修正（仅双图模式）─────────────────────────────
        validation   = {}
        preview_b64  = None
        if src_data is not None:
            tone_curve  = luminance_params.get('tone_curve', [(0,0),(64,64),(128,128),(192,192),(255,255)])
            all_params  = {**luminance_params, **color_params, **scene_result.get('params', {})}
            val         = render_and_validate(
                src_data['rgb_float'],
                ref_data['rgb_float'],
                all_params,
                tone_curve,
            )
            # 将自动修正写回 scene_result.params，使 XMP 采用修正后的值
            corrections = {k: v for k, v in val.get('corrections', {}).items()
                           if not k.startswith('_')}
            if corrections:
                scene_result['params'].update(corrections)

            preview_b64 = val.pop('preview_b64', None)
            validation  = {k: v for k, v in val.items() if k != 'corrections'}
            validation['corrections_applied'] = list(corrections.keys())
            color_cast  = val.get('corrections', {}).get('_color_cast_note', '')
            if color_cast:
                validation['color_cast_note'] = color_cast
        # ─────────────────────────────────────────────────────────────────

        xmp_content = generate_xmp(luminance_params, color_params, scene_result, preset_name=preset_name)
        summary     = params_summary(luminance_params, color_params, scene_result)
        curve_style = luminance_params.get('_curve_style', '')

        response_data = {
            "success":              True,
            "mode":                 mode,
            "report":               scene_result.get('report', {}),
            "summary":              summary,
            "xmp_content":          xmp_content,
            "compression_detected": ref_data.get('compression_suspected', False),
            "is_raw_source":        src_data.get('is_raw', False) if src_data else False,
            "camera_note":          camera_note,
            "curve_style":          curve_style,
            "validation":           validation,
        }
        if preview_b64:
            response_data["preview_b64"] = preview_b64

        return JSONResponse(response_data)

    except Exception as e:
        raise HTTPException(500, f"分析失败：{str(e)}")


@app.post("/download_xmp")
async def download_xmp(
    ref_image:      UploadFile = File(...),
    src_image:      UploadFile = File(None),
    preset_name:    str = Form("AI生成预设"),
    ref_scene_type: str = Form("auto"),
    src_scene_type: str = Form("auto"),
    camera_brand:   str = Form(""),
):
    ref_bytes = await ref_image.read()
    src_bytes = await src_image.read() if src_image else None

    try:
        ref_data = load_image(ref_bytes, ref_image.filename or "ref.jpg")
        src_data = load_image(src_bytes, src_image.filename or "src.jpg") if src_bytes else None

        luminance_params = analyze_luminance(ref_data, src_data)
        luminance_params = apply_luminance_linkage(luminance_params)
        color_params     = analyze_color(ref_data, src_data)

        if camera_brand:
            color_params = apply_camera_compensation(color_params, camera_brand)

        scene_result = analyze_scene_and_correct(
            ref_bytes,
            {**luminance_params, **color_params},
            src_bytes,
            ref_scene_type=ref_scene_type,
            src_scene_type=src_scene_type,
        )

        xmp_content = generate_xmp(luminance_params, color_params, scene_result, preset_name=preset_name)
        safe_name   = preset_name.replace(' ', '_').replace('/', '_')

        return Response(
            content=xmp_content.encode('utf-8'),
            media_type="application/xml",
            headers={"Content-Disposition": f'attachment; filename="{safe_name}.xmp"'}
        )

    except Exception as e:
        raise HTTPException(500, str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
