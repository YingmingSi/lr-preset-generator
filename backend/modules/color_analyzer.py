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
    """RGB转HSV，返回同形状数组，H范围0-360"""
    h, w, _ = rgb_float.shape
    flat = rgb_float.reshape(-1, 3)

    hsv = np.zeros_like(flat)
    for i, (r, g, b) in enumerate(flat):
        h_val, s_val, v_val = colorsys.rgb_to_hsv(
            float(r), float(g), float(b)
        )
        hsv[i] = [h_val * 360, s_val, v_val]

    return hsv.reshape(h, w, 3)


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
        sat_adj = clamp(int(sat_delta * 280), -100, 100)

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

    # 计算各色相的权重总和
    total_weight = sum(s['weight'] for s in bucket_stats.values())
    if total_weight < 1e-6:
        total_weight = 1.0

    # 计算"期望的"色相分布（均匀分布作为中性基准）
    neutral_weight = 1.0 / len(HUE_BUCKETS)

    for bucket, stats in bucket_stats.items():
        weight_ratio = stats['weight'] / max(total_weight * neutral_weight, 1e-6)

        # 该色相在图中的比重决定是否需要强化或压制
        if stats['weight'] < 0.01:
            # 几乎没有这个颜色 → 可能需要压制
            sat_adj = clamp(int((stats['sat_mean'] - 0.5) * -60), -50, 10)
        else:
            # 存在这个颜色 → 分析其特征
            sat_adj = clamp(int((stats['sat_mean'] - 0.45) * 130), -60, 60)

        # 色相偏移：分析色相均值与桶中心的偏差
        if stats['hue_mean'] is not None:
            hue_offset = stats['hue_mean'] - stats['hue_center'] if 'hue_center' in stats else 0
            hue_adj = clamp(int(hue_offset * 0.8), -40, 40)
        else:
            hue_adj = 0

        lum_adj = clamp(int((stats['val_mean'] - 0.5) * 60), -50, 50)

        params[f'HueAdjustment{bucket}'] = hue_adj
        params[f'SaturationAdjustment{bucket}'] = sat_adj
        params[f'LuminanceAdjustment{bucket}'] = lum_adj

    return params


def _derive_rgb_curves(ref_rgb: np.ndarray, src_rgb: Optional[np.ndarray] = None) -> dict:
    """
    推导RGB分量色调曲线
    分析各通道在不同亮度区间的色彩偏向
    """
    curves = {}

    for ch_idx, ch_name in enumerate(['Red', 'Green', 'Blue']):
        channel = ref_rgb[:, :, ch_idx]
        gray = 0.2126 * ref_rgb[:, :, 0] + 0.7152 * ref_rgb[:, :, 1] + 0.0722 * ref_rgb[:, :, 2]

        # 在8个亮度区间采样通道值
        points = []
        lum_points = np.linspace(0, 1, 9)

        for lum in lum_points:
            lo = max(0, lum - 0.08)
            hi = min(1, lum + 0.08)
            mask = (gray >= lo) & (gray < hi)

            if mask.sum() > 20:
                ch_mean = float(channel[mask].mean())
            else:
                ch_mean = float(lum)  # 无数据时使用中性值

            inp = int(lum * 255)
            out = int(ch_mean * 255)
            points.append((inp, out))

        curves[f'tone_curve_{ch_name.lower()}'] = points

    return curves


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
    """估算整体饱和度和自然饱和度"""
    sat_vals = ref_hsv[:, :, 1].flatten()
    mean_sat = float(sat_vals.mean())
    sat_std = float(sat_vals.std())

    # 饱和度分布集中 → 风格化强（高饱和或低饱和）
    # 饱和度分布分散 → 自然感强

    # 整体饱和度：相对于中性值0.45的偏差
    saturation = clamp(int((mean_sat - 0.45) * 130), -60, 60)

    # 自然饱和度：sat_std小说明高度风格化，用vibrance而非saturation
    if sat_std < 0.15:
        vibrance = saturation
        saturation = int(saturation * 0.4)
    else:
        vibrance = int(saturation * 0.5)

    return {
        'Vibrance':   clamp(vibrance, -60, 60),
        'Saturation': clamp(saturation, -60, 60),
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