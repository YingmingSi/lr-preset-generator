"""
色彩分析模块
分析HSL、色调曲线RGB分量、颜色分级（阴影/高光对立色）
"""

import numpy as np
from typing import Optional
import colorsys


# HSL 8个色相桶的中心角度和范围（度）
HUE_BUCKETS = {
    'Red':     (  0, 345, 15),   # 中心0°，范围345-15
    'Orange':  ( 30,  15, 45),
    'Yellow':  ( 60,  45, 75),
    'Green':   (120,  75, 165),
    'Aqua':    (180, 165, 210),
    'Blue':    (225, 210, 255),
    'Purple':  (270, 255, 300),
    'Magenta': (315, 300, 345),
}


def analyze_color(ref_data: dict, src_data: Optional[dict] = None) -> dict:
    """
    分析色彩参数

    Args:
        ref_data: 参考图数据
        src_data: 原图数据（可选）

    Returns:
        色彩相关的LR参数字典
    """
    ref_hsv = _rgb_to_hsv(ref_data['rgb_float'])

    if src_data is not None:
        src_hsv = _rgb_to_hsv(src_data['rgb_float'])
        hsl_params = _diff_hsl(src_hsv, ref_hsv)
        wb_params = _estimate_white_balance_diff(src_data['rgb_float'], ref_data['rgb_float'])
    else:
        hsl_params = _feature_hsl(ref_hsv)
        wb_params = _estimate_white_balance_single(ref_data['rgb_float'])

    # 色调曲线RGB分量
    rgb_curves = _derive_rgb_curves(ref_data['rgb_float'], src_data['rgb_float'] if src_data else None)

    # 颜色分级（阴影/高光对立色）
    color_grading = _analyze_color_grading(ref_data['rgb_float'])

    # 整体饱和度/自然饱和度
    vibrance_sat = _estimate_vibrance_saturation(ref_hsv, src_data['rgb_float'] if src_data else None)

    return {
        **wb_params,
        **hsl_params,
        **rgb_curves,
        **color_grading,
        **vibrance_sat,
    }


def _rgb_to_hsv(rgb_float: np.ndarray) -> np.ndarray:
    """RGB转HSV（向量化），返回同形状数组，H范围0-360"""
    r = rgb_float[:, :, 0]
    g = rgb_float[:, :, 1]
    b = rgb_float[:, :, 2]

    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    delta = max_c - min_c

    s = np.where(max_c > 0, delta / np.maximum(max_c, 1e-9), 0.0)

    h = np.zeros_like(r)
    eps = 1e-9
    mask_r = (delta > eps) & (max_c == r)
    mask_g = (delta > eps) & (max_c == g)
    mask_b = (delta > eps) & (max_c == b)

    h[mask_r] = (60.0 * ((g[mask_r] - b[mask_r]) / delta[mask_r])) % 360.0
    h[mask_g] = (60.0 * ((b[mask_g] - r[mask_g]) / delta[mask_g]) + 120.0) % 360.0
    h[mask_b] = (60.0 * ((r[mask_b] - g[mask_b]) / delta[mask_b]) + 240.0) % 360.0

    return np.stack([h, s, max_c], axis=2)


def _get_hue_mask(hsv: np.ndarray, bucket_name: str) -> np.ndarray:
    """获取某个色相桶的像素掩码"""
    _, lo, hi = HUE_BUCKETS[bucket_name]
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]

    if bucket_name == 'Red':
        mask = (h >= 345) | (h < 15)
    else:
        mask = (h >= lo) & (h < hi)

    # 只考虑有饱和度的像素
    mask = mask & (s > 0.1)
    return mask


def _analyze_bucket(hsv: np.ndarray, bucket_name: str) -> dict:
    """分析单个色相桶的统计特征"""
    mask = _get_hue_mask(hsv, bucket_name)
    count = mask.sum()

    if count < 50:
        return {'hue_mean': None, 'sat_mean': 0, 'val_mean': 0, 'weight': 0}

    h_vals = hsv[:, :, 0][mask]
    s_vals = hsv[:, :, 1][mask]
    v_vals = hsv[:, :, 2][mask]

    center = HUE_BUCKETS[bucket_name][0]

    # 色相均值（处理红色跨0度的特殊情况）
    if bucket_name == 'Red':
        h_adj = np.where(h_vals > 180, h_vals - 360, h_vals)
        hue_mean = float(h_adj.mean())
    else:
        hue_mean = float(h_vals.mean())

    return {
        'hue_mean': hue_mean,
        'hue_center': center,
        'sat_mean': float(s_vals.mean()),
        'val_mean': float(v_vals.mean()),
        'weight': float(count) / (hsv.shape[0] * hsv.shape[1]),
    }


def _diff_hsl(src_hsv: np.ndarray, ref_hsv: np.ndarray) -> dict:
    """模式B：对比原图和参考图的HSL差值"""
    params = {}

    for bucket in HUE_BUCKETS:
        src_stats = _analyze_bucket(src_hsv, bucket)
        ref_stats = _analyze_bucket(ref_hsv, bucket)

        # 色相偏移
        if src_stats['hue_mean'] is not None and ref_stats['hue_mean'] is not None:
            hue_delta = ref_stats['hue_mean'] - src_stats['hue_mean']
            hue_adj = clamp(int(hue_delta * 1.5), -100, 100)
        else:
            hue_adj = 0

        # 饱和度变化
        sat_delta = ref_stats['sat_mean'] - src_stats['sat_mean']
        sat_adj = clamp(int(sat_delta * 160), -80, 80)

        # 明度变化
        val_delta = ref_stats['val_mean'] - src_stats['val_mean']
        lum_adj = clamp(int(val_delta * 150), -100, 100)

        params[f'HueAdjustment{bucket}'] = hue_adj
        params[f'SaturationAdjustment{bucket}'] = sat_adj
        params[f'LuminanceAdjustment{bucket}'] = lum_adj

    return params


def _feature_hsl(ref_hsv: np.ndarray) -> dict:
    """模式A：从参考图提取HSL风格特征"""
    params = {}
    bucket_stats = {b: _analyze_bucket(ref_hsv, b) for b in HUE_BUCKETS}

    for bucket, stats in bucket_stats.items():
        # 该色相在图中几乎不存在 → 无有效像素，不做调整
        if stats['weight'] < 0.01:
            params[f'HueAdjustment{bucket}']        = 0
            params[f'SaturationAdjustment{bucket}'] = 0
            params[f'LuminanceAdjustment{bucket}']  = 0
            continue

        sat_adj = clamp(int((stats['sat_mean'] - 0.40) * 150), -65, 65)

        if stats['hue_mean'] is not None:
            hue_center = HUE_BUCKETS[bucket][0]
            hue_offset = stats['hue_mean'] - hue_center
            # 红色桶跨越0°，修正偏移方向
            if bucket == 'Red' and abs(hue_offset) > 180:
                hue_offset -= 360 if hue_offset > 0 else -360
            hue_adj = clamp(int(hue_offset * 0.8), -40, 40)
        else:
            hue_adj = 0

        lum_adj = clamp(int((stats['val_mean'] - 0.5) * 60), -50, 50)

        params[f'HueAdjustment{bucket}']        = hue_adj
        params[f'SaturationAdjustment{bucket}'] = sat_adj
        params[f'LuminanceAdjustment{bucket}']  = lum_adj

    return params


def _derive_rgb_curves(ref_rgb: np.ndarray, src_rgb: Optional[np.ndarray] = None) -> dict:
    """
    推导RGB分量色调曲线
    Mode A：分析参考图各通道偏离中性的程度（35%衰减，避免与HSL叠加）
    Mode B：计算原图→参考图的通道差值，仅应用50%差值
    """
    ref_gray = 0.2126 * ref_rgb[:, :, 0] + 0.7152 * ref_rgb[:, :, 1] + 0.0722 * ref_rgb[:, :, 2]
    if src_rgb is not None:
        src_gray = 0.2126 * src_rgb[:, :, 0] + 0.7152 * src_rgb[:, :, 1] + 0.0722 * src_rgb[:, :, 2]

    curves = {}
    lum_points = np.linspace(0, 1, 9)

    for ch_idx, ch_name in enumerate(['Red', 'Green', 'Blue']):
        ref_ch = ref_rgb[:, :, ch_idx]
        points = []

        for lum in lum_points:
            lo = max(0.0, lum - 0.08)
            hi = min(1.0, lum + 0.08)

            if src_rgb is not None:
                src_ch   = src_rgb[:, :, ch_idx]
                src_mask = (src_gray >= lo) & (src_gray < hi)
                ref_mask = (ref_gray >= lo) & (ref_gray < hi)
                if src_mask.sum() > 20 and ref_mask.sum() > 20:
                    delta    = float(ref_ch[ref_mask].mean()) - float(src_ch[src_mask].mean())
                    dampened = clamp(lum + delta * 0.5, 0.0, 1.0)
                else:
                    dampened = lum
            else:
                mask = (ref_gray >= lo) & (ref_gray < hi)
                if mask.sum() > 20:
                    ch_mean  = float(ref_ch[mask].mean())
                    dampened = lum + (ch_mean - lum) * 0.35
                else:
                    dampened = lum

            points.append((int(lum * 255), clamp(int(dampened * 255), 0, 255)))

        curves[f'tone_curve_{ch_name.lower()}'] = _enforce_monotone(points)

    return curves


def _enforce_monotone(points: list) -> list:
    """确保曲线控制点输出值非递减（Lightroom要求）"""
    result = list(points)
    for i in range(1, len(result)):
        if result[i][1] < result[i - 1][1]:
            result[i] = (result[i][0], result[i - 1][1])
    return result


def _analyze_color_grading(rgb_float: np.ndarray) -> dict:
    """
    分析颜色分级（阴影和高光的对立色倾向）
    """
    gray = 0.2126 * rgb_float[:, :, 0] + 0.7152 * rgb_float[:, :, 1] + 0.0722 * rgb_float[:, :, 2]

    # 阴影区域（亮度0-33%）
    shadow_mask = gray < 0.33
    # 高光区域（亮度67-100%）
    highlight_mask = gray > 0.67
    # 中间调（33-67%）
    midtone_mask = (gray >= 0.33) & (gray <= 0.67)

    def zone_color(mask):
        if mask.sum() < 100:
            return 0, 0
        r = float(rgb_float[:, :, 0][mask].mean())
        g = float(rgb_float[:, :, 1][mask].mean())
        b = float(rgb_float[:, :, 2][mask].mean())
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        return int(h * 360), int(s * 100)

    shadow_hue, shadow_sat = zone_color(shadow_mask)
    highlight_hue, highlight_sat = zone_color(highlight_mask)
    midtone_hue, midtone_sat = zone_color(midtone_mask)

    # 验证是否构成对立色关系
    hue_diff = abs(shadow_hue - highlight_hue)
    is_complementary = 150 < hue_diff < 210

    # 如果不是对立色，降低颜色分级强度
    sat_scale = 1.0 if is_complementary else 0.5

    return {
        'SplitToningShadowHue':          shadow_hue,
        'SplitToningShadowSaturation':   int(shadow_sat * 0.4 * sat_scale),
        'SplitToningHighlightHue':       highlight_hue,
        'SplitToningHighlightSaturation': int(highlight_sat * 0.4 * sat_scale),
        'SplitToningBalance':            0,
        'ColorGradeMidtoneHue':          midtone_hue,
        'ColorGradeMidtoneSat':          int(midtone_sat * 0.2),
        'ColorGradeShadowLum':           0,
        'ColorGradeHighlightLum':        0,
        'is_complementary_grading':      is_complementary,  # 供报告使用
    }


def _estimate_vibrance_saturation(ref_hsv: np.ndarray, src_rgb: Optional[np.ndarray]) -> dict:
    """估算整体饱和度和自然饱和度
    Mode B：计算原图→参考图的饱和度差值（只统计有色像素）
    Mode A：相对于经验中性基准估算（只统计有色像素，基准0.42）
    """
    ref_sat_vals  = ref_hsv[:, :, 1]
    ref_color_mask = ref_sat_vals > 0.12
    # 只在有实际颜色的像素上计算均值，避免大量中性像素压低结果
    if ref_color_mask.sum() > 500:
        ref_mean_sat = float(ref_sat_vals[ref_color_mask].mean())
    else:
        ref_mean_sat = float(ref_sat_vals.mean())

    if src_rgb is not None:
        src_hsv       = _rgb_to_hsv(src_rgb)
        src_sat_vals  = src_hsv[:, :, 1]
        src_color_mask = src_sat_vals > 0.12
        if src_color_mask.sum() > 500:
            src_mean_sat = float(src_sat_vals[src_color_mask].mean())
        else:
            src_mean_sat = float(src_sat_vals.mean())
        delta      = ref_mean_sat - src_mean_sat
        saturation = clamp(int(delta * 200), -65, 65)
        vibrance   = clamp(int(delta * 170), -65, 65)
    else:
        # 基准 0.42：有色像素的中性参考饱和度（LR默认开发的典型值）
        saturation = clamp(int((ref_mean_sat - 0.42) * 160), -60, 60)
        vibrance   = clamp(int(saturation * 0.8), -60, 60)

    return {
        'Vibrance':   vibrance,
        'Saturation': saturation,
    }


def _estimate_white_balance_diff(src_rgb: np.ndarray, ref_rgb: np.ndarray) -> dict:
    """
    双图模式：白平衡保持原图设置（As Shot）
    不输出任何色温调整值，由LR根据RAW原图自行判断
    """
    return {
        'wb_confidence': 'as_shot',
    }


def _estimate_white_balance_single(rgb_float: np.ndarray) -> dict:
    """单图模式：白平衡同样保持As Shot，不做推算"""
    return {
        'wb_confidence': 'as_shot',
    }


def clamp(value, lo, hi):
    return max(lo, min(hi, value))