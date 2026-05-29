"""
Lightroom预设生成器 - FastAPI主应用
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
import sys
import os
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))

from typing import List

from modules.image_loader import load_image
from modules.luminance_analyzer import analyze_luminance, apply_luminance_linkage
from modules.color_analyzer import analyze_color
from modules.scene_analyzer import analyze_scene_and_correct
from modules.camera_profiles import apply_camera_compensation, get_camera_description
from modules.xmp_generator import generate_xmp, params_summary
from modules.preset_renderer import render_and_validate
from modules.preset_library import (
    load_user_styles, add_user_preset, add_user_preset_incremental,
    list_styles, parse_xmp_params,
    reset_user_styles, batch_cluster, BATCH_KMEANS_MIN, promote_to_seeded,
)
from modules.calibration import (
    load_calibration, apply_calibration, is_calibrated,
    update_from_params_list, get_summary as calib_summary, reset_calibration,
)
from modules.action_basis import (
    load_learned, decompose as action_decompose, compose as action_compose,
    top_actions, learn_from_uploads, get_action_info, reset_learned,
    derive_user_actions, has_user_actions,
)

# 上传接口开关：默认开启；Railway 生产环境设 DISABLE_PRESET_UPLOAD=true 关闭
UPLOAD_ENABLED = os.getenv("DISABLE_PRESET_UPLOAD", "false").lower() != "true"

app = FastAPI(title="LR Preset Generator", version="1.0.0")

# 启动时加载用户预设库、校准数据、已学习动作
load_user_styles()
load_calibration()
load_learned()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://lr-preset-generator.vercel.app",
    ],
    allow_origin_regex=r"http://localhost:\d+",   # 允许本地任意端口
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

        # ── 校准约束 ──────────────────────────────────────────────────────
        if is_calibrated():
            luminance_params = apply_calibration(luminance_params)
            color_params     = apply_calibration(color_params)
        # ─────────────────────────────────────────────────────────────────

        # ── 动作基底分解 + 合成 ───────────────────────────────────────────
        raw_combined = {**luminance_params, **color_params,
                        **scene_result.get('params', {})}
        action_weights, action_r2 = action_decompose(raw_combined)
        composed = action_compose(action_weights, raw_combined, r2=action_r2)
        # 将合成结果写回（HSL + 亮度均更新）
        for k, v in composed.items():
            if k in luminance_params:
                luminance_params[k] = v
            elif k in color_params or any(k.startswith(p) for p in
                 ('HueAdjustment', 'SaturationAdjustment', 'LuminanceAdjustment',
                  'Saturation', 'Vibrance')):
                color_params[k] = v
        action_top = top_actions(action_weights)
        # ─────────────────────────────────────────────────────────────────

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
            "action_top":           action_top,
            "action_r2":            round(action_r2, 3),
            "calibration":          calib_summary(),
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

        if is_calibrated():
            luminance_params = apply_calibration(luminance_params)
            color_params     = apply_calibration(color_params)

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


@app.post("/upload_presets")
async def upload_presets(preset_files: List[UploadFile] = File(...)):
    """上传 XMP 预设文件（仅本地/管理员环境开放）"""
    if not UPLOAD_ENABLED:
        raise HTTPException(403, "预设上传在此部署环境中已禁用")

    # ── 解析所有文件 ──────────────────────────────────────────────────────────
    parsed = []   # [{filename, params}]
    for f in preset_files:
        if not f.filename.lower().endswith('.xmp'):
            continue
        content     = (await f.read()).decode('utf-8', errors='ignore')
        params      = parse_xmp_params(content)
        if params:
            parsed.append({'filename': f.filename, 'params': params})

    params_batch = [p['params'] for p in parsed]
    filenames    = [p['filename'] for p in parsed]

    # ── 风格聚类 ──────────────────────────────────────────────────────────────
    is_bulk = len(parsed) >= BATCH_KMEANS_MIN

    if is_bulk:
        # 批量首次建库：K-means 全局分组
        cluster_results  = batch_cluster(params_batch, filenames)
        cluster_mode     = f"批量 K-means ({len(parsed)} 个文件 → {len(cluster_results)} 个聚类)"
        preset_summaries = cluster_results
    else:
        # 增量添加：严格阈值 0.95，新风格始终新建不强制合并
        preset_summaries = []
        for p in parsed:
            summary = add_user_preset_incremental(p['params'], p['filename'])
            preset_summaries.append(summary)
        cluster_mode = f"增量模式 ({len(parsed)} 个文件，新风格独立保留)"

    # ── 更新校准范围 ─────────────────────────────────────────────────────────
    calib_report = {}
    if params_batch:
        calib_report = update_from_params_list(params_batch)

    # ── 动作库更新 ────────────────────────────────────────────────────────────
    # 批量首次建库：全量 PCA（直接 PCA），动作量级 = 真实调色幅度
    #   已能覆盖大部分方差，不再叠加残差 PCA（避免冗余）
    # 增量少量上传：残差 PCA（只补充现有基底覆盖不到的新方向）
    learned_new = {}
    if is_bulk:
        derive_user_actions(params_batch)        # 全量 PCA → user_actions.json
    elif len(params_batch) >= 3:
        learned_new = learn_from_uploads(        # 残差 PCA → 仅补充新方向
            params_batch, min_samples=3, var_threshold=0.10
        )

    return JSONResponse({
        "success":        True,
        "imported":       len(parsed),
        "cluster_mode":   cluster_mode,
        "presets":        preset_summaries,
        "library":        list_styles(),
        "calibration":    calib_summary(),
        "calib_updated":  len(calib_report),
        "actions":        get_action_info(),
        "learned_new":    [v.get('_label', k) for k, v in learned_new.items()],
    })


@app.post("/styles/seed")
async def seed_styles():
    """
    固化当前学习状态：
    · seeded_styles.json  — K-means 风格聚类（提交 git 后永久展示）
    · user_actions.json   — PCA 动作基底（提交 git 后作为生成依据）
    · calibration.json    — 参数范围约束（已自动更新）
    """
    if not UPLOAD_ENABLED:
        raise HTTPException(403, "此操作在生产环境中已禁用")
    from modules.action_basis import save_user_actions, has_user_actions
    style_result = promote_to_seeded()
    if has_user_actions():
        save_user_actions()   # 确保 user_actions.json 写入磁盘
    return JSONResponse({
        **style_result,
        "user_actions_saved": has_user_actions(),
        "library": list_styles(),
        "actions": get_action_info(),
    })


@app.delete("/styles/user")
async def delete_user_styles():
    """清空当前 session 上传的聚类（不影响已固化的 seeded_styles）"""
    if not UPLOAD_ENABLED:
        raise HTTPException(403, "此操作在生产环境中已禁用")
    reset_user_styles()
    return JSONResponse({"success": True, "library": list_styles()})


@app.get("/styles")
async def get_styles():
    """获取当前风格模板库列表及校准状态"""
    return JSONResponse({
        "styles":      list_styles(),
        "calibration": calib_summary(),
    })


@app.get("/actions")
async def get_actions():
    """获取当前动作基底列表（内置 + 已学习）"""
    return JSONResponse({"actions": get_action_info()})


@app.delete("/actions/learned")
async def clear_learned_actions():
    """清除所有学习到的动作，恢复仅使用内置动作基底"""
    reset_learned()
    return JSONResponse({"success": True, "actions": get_action_info()})


@app.delete("/calibration")
async def clear_calibration():
    """清除校准数据，恢复使用内置默认范围"""
    reset_calibration()
    return JSONResponse({"success": True, "calibration": calib_summary()})


# ── 数据导出 / 导入（用于 Railway ↔ 本地同步）──────────────────────────────

@app.get("/data/export")
async def export_data():
    """
    导出所有学习数据为单个 JSON（校准参数 + 动作基 + 风格库）。
    可下载后导入到其他环境（本地 ↔ Railway 双向同步）。
    """
    from modules.action_basis    import _learned_actions
    from modules.calibration     import _calib
    from modules.preset_library  import _user_styles

    payload = {
        "version":         "1.0",
        "exported_at":     datetime.now(timezone.utc).isoformat(),
        "calibration":     _calib,
        "learned_actions": _learned_actions,
        "user_styles":     _user_styles,
    }
    return Response(
        content=__import__('json').dumps(payload, ensure_ascii=False, indent=2).encode('utf-8'),
        media_type="application/json",
        headers={"Content-Disposition": 'attachment; filename="lr_preset_data.json"'},
    )


@app.post("/data/import")
async def import_data(payload: dict = Body(...)):
    """导入学习数据（仅本地/管理员环境开放）"""
    if not UPLOAD_ENABLED:
        raise HTTPException(403, "数据导入在此部署环境中已禁用")
    from modules import calibration as _cal, action_basis as _ab, preset_library as _pl

    errors = []

    if "calibration" in payload:
        try:
            _cal._calib = payload["calibration"]
            _cal.save_calibration()
        except Exception as e:
            errors.append(f"calibration: {e}")

    if "learned_actions" in payload:
        try:
            _ab._learned_actions = payload["learned_actions"]
            _ab._invalidate_cache()
            _ab.save_learned()
        except Exception as e:
            errors.append(f"learned_actions: {e}")

    if "user_styles" in payload:
        try:
            _pl._user_styles = payload["user_styles"]
            _pl.save_user_styles()
        except Exception as e:
            errors.append(f"user_styles: {e}")

    return JSONResponse({
        "success":   len(errors) == 0,
        "errors":    errors,
        "summary": {
            "calibration_params": len(_cal._calib),
            "learned_actions":    len(_ab._learned_actions),
            "user_styles":        len(_pl._user_styles),
        },
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
