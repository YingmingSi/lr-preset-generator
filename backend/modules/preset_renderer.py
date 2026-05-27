"""
Lightroom预设近似渲染器
将XMP参数模拟应用到图像，与参考图对比实现自我验证
精度约 75–85%（JPEG源图），不等同于Lightroom实际输出
支持：曝光、高光/阴影/白色/黑色、对比度、色调曲线、HSL、饱和度/自然饱和度
"""

import numpy as np
from PIL import Image
import base64
import io

HUE_BUCKETS = {
    'Red':     (  0, 345,  15),
    'Orange':  ( 30,  15,  45),
    'Yellow':  ( 60,  45,  75),
    'Green':   (120,  75, 165),
    'Aqua':    (180, 165, 210),
    'Blue':    (225, 210, 255),
    'Purple':  (270, 255, 300),
    'Magenta': (315, 300, 345),
}


def render_and_validate(
    src_rgb:    np.ndarray,
    ref_rgb:    np.ndarray,
    params:     dict,
    tone_curve: list,
) -> dict:
    """
    将参数模拟渲染到原图，与参考图比较。

    Returns:
        match_score:         0-1 综合匹配分数
        lum_score:           亮度匹配
        color_score:         色调匹配
        sat_score:           饱和度匹配
        rendered_lum/sat:    渲染结果统计
        target_lum/sat:      参考图统计
        corrections:         自动修正建议 {参数名: 修正后值}
        preview_b64:         渲染结果缩略图 base64 JPEG
    """
    rendered = apply_preset(src_rgb, params, tone_curve)
    scores   = compare_images(rendered, ref_rgb)
    corr     = compute_corrections(rendered, ref_rgb, params)
    preview  = _encode_preview(rendered)

    return {**scores, 'corrections': corr, 'preview_b64': preview}


def apply_preset(rgb: np.ndarray, params: dict, tone_curve: list) -> np.ndarray:
    """将LR参数近似渲染到图像，返回 float32 [0,1]"""
    img = np.clip(rgb.astype(np.float64), 0.0, 1.0)

    # 1. 曝光
    exp = float(params.get('Exposure', 0.0))
    if abs(exp) > 1e-3:
        img = img * (2.0 ** exp)

    # 2. 高光/阴影/白色/黑色（分区权重曲线）
    img = _apply_tonal_regions(img, params)

    # 3. 对比度（以0.5为轴心的线性缩放）
    contrast = float(params.get('Contrast', 0)) / 100.0
    if abs(contrast) > 0.01:
        img = 0.5 + (img - 0.5) * (1.0 + contrast * 0.5)

    # 4. 亮度色调曲线
    if tone_curve and len(tone_curve) >= 2:
        lut = _curve_to_lut(tone_curve)
        idx = np.clip((img * 255).astype(np.int32), 0, 255)
        img = lut[idx]

    # 5. HSL面板
    img = _apply_hsl(img, params)

    # 6. 整体饱和度 / 自然饱和度
    img = _apply_vibrance_saturation(img, params)

    return np.clip(img, 0.0, 1.0).astype(np.float32)


# ── 基础调整 ────────────────────────────────────────────────────────────────────

def _apply_tonal_regions(img: np.ndarray, params: dict) -> np.ndarray:
    hl = float(params.get('Highlights', 0)) / 100.0
    sh = float(params.get('Shadows',    0)) / 100.0
    wh = float(params.get('Whites',     0)) / 100.0
    bk = float(params.get('Blacks',     0)) / 100.0

    if abs(hl) < 0.01 and abs(sh) < 0.01 and abs(wh) < 0.01 and abs(bk) < 0.01:
        return img

    lum = (0.2126 * img[:, :, 0] +
           0.7152 * img[:, :, 1] +
           0.0722 * img[:, :, 2])

    w_hl = np.clip((lum - 0.5) * 2.0, 0, 1) ** 1.5
    w_sh = np.clip((0.5 - lum) * 2.0, 0, 1) ** 1.5
    w_wh = np.clip((lum - 0.80) / 0.20, 0, 1) ** 2
    w_bk = np.clip((0.20 - lum) / 0.20, 0, 1) ** 2

    delta = (hl * 0.25 * w_hl +
             sh * 0.25 * w_sh +
             wh * 0.15 * w_wh +
             bk * 0.10 * w_bk)[..., np.newaxis]

    return img + delta


def _curve_to_lut(curve_points: list) -> np.ndarray:
    pts = sorted(curve_points)
    xs  = [p[0] for p in pts]
    ys  = [p[1] for p in pts]
    lut = np.zeros(256, dtype=np.float64)
    for i in range(256):
        idx = np.searchsorted(xs, i)
        if idx == 0:
            lut[i] = ys[0]
        elif idx >= len(xs):
            lut[i] = ys[-1]
        else:
            x0, y0 = xs[idx - 1], ys[idx - 1]
            x1, y1 = xs[idx],     ys[idx]
            t = (i - x0) / max(x1 - x0, 1)
            lut[i] = y0 + (y1 - y0) * t
    return lut / 255.0


# ── HSL ─────────────────────────────────────────────────────────────────────────

def _apply_hsl(img: np.ndarray, params: dict) -> np.ndarray:
    hsv = _rgb_to_hsv(img)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    for bucket, (_, lo, hi) in HUE_BUCKETS.items():
        if bucket == 'Red':
            mask = ((h >= 345) | (h < 15)) & (s > 0.05)
        else:
            mask = (h >= lo) & (h < hi) & (s > 0.05)

        if not mask.any():
            continue

        hue_adj = float(params.get(f'HueAdjustment{bucket}',        0))
        sat_adj = float(params.get(f'SaturationAdjustment{bucket}', 0)) / 100.0
        lum_adj = float(params.get(f'LuminanceAdjustment{bucket}',  0)) / 100.0

        if abs(hue_adj) > 0.01:
            h[mask] = (h[mask] + hue_adj) % 360.0
        if abs(sat_adj) > 0.001:
            s[mask] = np.clip(s[mask] * (1.0 + sat_adj), 0.0, 1.0)
        if abs(lum_adj) > 0.001:
            v[mask] = np.clip(v[mask] + lum_adj * 0.5, 0.0, 1.0)

    return _hsv_to_rgb(np.stack([h, s, v], axis=2))


def _apply_vibrance_saturation(img: np.ndarray, params: dict) -> np.ndarray:
    sat_val = float(params.get('Saturation', 0)) / 100.0
    vib_val = float(params.get('Vibrance',   0)) / 100.0

    if abs(sat_val) < 0.001 and abs(vib_val) < 0.001:
        return img

    hsv = _rgb_to_hsv(img)
    s   = hsv[:, :, 1]

    if abs(sat_val) > 0.001:
        s = np.clip(s * (1.0 + sat_val), 0.0, 1.0)

    if abs(vib_val) > 0.001:
        protection = 1.0 - s
        s = np.clip(s + vib_val * 0.5 * protection, 0.0, 1.0)

    hsv[:, :, 1] = s
    return _hsv_to_rgb(hsv)


# ── 色彩空间转换 ─────────────────────────────────────────────────────────────────

def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    max_c   = np.maximum(np.maximum(r, g), b)
    min_c   = np.minimum(np.minimum(r, g), b)
    delta   = max_c - min_c
    s       = np.where(max_c > 0, delta / np.maximum(max_c, 1e-9), 0.0)
    h       = np.zeros_like(r)
    eps     = 1e-9
    mr = (delta > eps) & (max_c == r)
    mg = (delta > eps) & (max_c == g)
    mb = (delta > eps) & (max_c == b)
    h[mr]  = (60.0 * ((g[mr] - b[mr]) / delta[mr])) % 360.0
    h[mg]  = (60.0 * ((b[mg] - r[mg]) / delta[mg]) + 120.0) % 360.0
    h[mb]  = (60.0 * ((r[mb] - g[mb]) / delta[mb]) + 240.0) % 360.0
    return np.stack([h, s, max_c], axis=2)


def _hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    h6 = h / 60.0
    i  = np.floor(h6).astype(int) % 6
    f  = h6 - np.floor(h6)
    p  = v * (1 - s)
    q  = v * (1 - f * s)
    t  = v * (1 - (1 - f) * s)
    r  = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [v, q, p, p, t, v])
    g  = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [t, v, v, q, p, p])
    b  = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [p, p, t, v, v, q])
    return np.clip(np.stack([r, g, b], axis=2), 0.0, 1.0)


# ── 对比与修正 ───────────────────────────────────────────────────────────────────

def compare_images(img1: np.ndarray, img2: np.ndarray) -> dict:
    """比较两图色彩统计，返回匹配分数（0–1）"""
    r1 = _resize(img1, 100)
    r2 = _resize(img2, 100)

    lum1      = float(_lum(r1).mean())
    lum2      = float(_lum(r2).mean())
    lum_score = max(0.0, 1.0 - abs(lum1 - lum2) / 0.12)

    color_diff  = float(np.mean(np.abs(r1.mean((0, 1)) - r2.mean((0, 1)))))
    color_score = max(0.0, 1.0 - color_diff / 0.10)

    sat1      = _mean_sat(r1)
    sat2      = _mean_sat(r2)
    sat_score = max(0.0, 1.0 - abs(sat1 - sat2) / 0.12)

    overall = round(lum_score * 0.40 + color_score * 0.35 + sat_score * 0.25, 2)

    return {
        'match_score':  overall,
        'lum_score':    round(lum_score,   2),
        'color_score':  round(color_score, 2),
        'sat_score':    round(sat_score,   2),
        'rendered_lum': round(lum1, 3),
        'target_lum':   round(lum2, 3),
        'rendered_sat': round(sat1, 3),
        'target_sat':   round(sat2, 3),
    }


def compute_corrections(rendered: np.ndarray, reference: np.ndarray, params: dict) -> dict:
    """基于渲染误差计算自动修正量（50%步长，避免过冲）"""
    r1 = _resize(rendered,  200)
    r2 = _resize(reference, 200)
    corrections = {}

    # 亮度误差 → 修正曝光
    lum_err = float(_lum(r1).mean()) - float(_lum(r2).mean())
    if abs(lum_err) > 0.03:
        ev_corr = -lum_err * 2.5 * 0.5
        new_exp = params.get('Exposure', 0.0) + ev_corr
        corrections['Exposure'] = round(max(-2.0, min(2.0, float(new_exp))), 2)

    # 饱和度误差 → 修正全局饱和度
    sat_err = _mean_sat(r1) - _mean_sat(r2)
    if abs(sat_err) > 0.03:
        sat_corr = int(-sat_err * 130 * 0.5)
        new_sat  = int(params.get('Saturation', 0)) + sat_corr
        corrections['Saturation'] = max(-60, min(60, new_sat))

    # 色调偏差（R vs B）→ 记录供报告使用
    r_diff = float(r2[:, :, 0].mean() - r1[:, :, 0].mean())
    b_diff = float(r2[:, :, 2].mean() - r1[:, :, 2].mean())
    if abs(r_diff - b_diff) > 0.05:
        direction = '偏暖（缺红/多蓝）' if b_diff > r_diff else '偏冷（多红/缺蓝）'
        corrections['_color_cast_note'] = direction

    return corrections


# ── 工具 ─────────────────────────────────────────────────────────────────────────

def _lum(rgb: np.ndarray) -> np.ndarray:
    return 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]


def _mean_sat(rgb: np.ndarray) -> float:
    return float(_rgb_to_hsv(rgb)[:, :, 1].mean())


def _resize(rgb: np.ndarray, size: int) -> np.ndarray:
    img = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    img = img.resize((size, size), Image.LANCZOS)
    return np.array(img).astype(np.float32) / 255.0


def _encode_preview(rgb: np.ndarray, max_size: int = 400) -> str:
    h, w   = rgb.shape[:2]
    scale  = min(1.0, max_size / max(h, w))
    nw, nh = int(w * scale), int(h * scale)
    img    = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    img    = img.resize((nw, nh), Image.LANCZOS)
    buf    = io.BytesIO()
    img.save(buf, format='JPEG', quality=75)
    return base64.b64encode(buf.getvalue()).decode('utf-8')
