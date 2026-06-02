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

WARM_BUCKETS = {'Red', 'Orange', 'Yellow'}


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
    src_rgb = src_data['rgb_float'] if src_data else None

    if src_data is not None:
        src_hsv = _rgb_to_hsv(src_rgb)
        hsl_params = _diff_hsl(src_hsv, ref_hsv)
        wb_params = _estimate_white_balance_diff(src_rgb, ref_data['rgb_float'])
    else:
        hsl_params = _feature_hsl(ref_hsv)
        wb_params = _estimate_white_balance_single(ref_data['rgb_float'])

    # 色调曲线RGB分量
    rgb_curves = _derive_rgb_curves(ref_data['rgb_float'], src_rgb)

    # 颜色分级：双图模式用差值（ref 相对 src 的色调偏移），单图用参考图绝对值
    color_grading = _analyze_color_grading_diff(ref_data['rgb_float'], src_rgb)

    # 整体饱和度/自然饱和度
    vibrance_sat = _estimate_vibrance_saturation(ref_hsv, src_rgb)

    # 相机校准面板
    calibration = _compute_calibration(ref_data['rgb_float'], src_rgb)

    return {
        **wb_params,
        **hsl_params,
        **rgb_curves,
        **color_grading,
        **vibrance_sat,
        **calibration,
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
            hue_adj = clamp(int(hue_delta * 2.0), -100, 100)
        else:
            hue_adj = 0

        # 饱和度变化：暖色用更大系数（肤色/橙黄差值往往绝对值小但视觉感知强）
        sat_delta = ref_stats['sat_mean'] - src_stats['sat_mean']
        sat_mult  = 360 if bucket in WARM_BUCKETS else 300
        sat_adj   = clamp(int(sat_delta * sat_mult), -100, 100)

        # 明度变化
        val_delta = ref_stats['val_mean'] - src_stats['val_mean']
        lum_adj = clamp(int(val_delta * 240), -100, 100)

        params[f'HueAdjustment{bucket}'] = hue_adj
        params[f'SaturationAdjustment{bucket}'] = sat_adj
        params[f'LuminanceAdjustment{bucket}'] = lum_adj

    return _apply_directional_focus(params)


def _feature_hsl(ref_hsv: np.ndarray) -> dict:
    """模式A：从参考图提取HSL风格特征"""
    params = {}
    bucket_stats = {b: _analyze_bucket(ref_hsv, b) for b in HUE_BUCKETS}

    for bucket, stats in bucket_stats.items():
        # 该色相在图中几乎不存在 → 无有效像素，不做调整
        if stats['weight'] < 0.005:
            params[f'HueAdjustment{bucket}']        = 0
            params[f'SaturationAdjustment{bucket}'] = 0
            params[f'LuminanceAdjustment{bucket}']  = 0
            continue

        # 暖色（红/橙/黄）基线更低：肤色/暖调的 HSV 饱和度通常只有 0.18-0.32
        # 冷色（绿/青/蓝/紫/品红）基线 0.26
        if bucket in WARM_BUCKETS:
            sat_adj = clamp(int((stats['sat_mean'] - 0.18) * 320), -80, 80)
        else:
            sat_adj = clamp(int((stats['sat_mean'] - 0.26) * 285), -80, 80)

        if stats['hue_mean'] is not None:
            hue_center = HUE_BUCKETS[bucket][0]
            hue_offset = stats['hue_mean'] - hue_center
            # 红色桶跨越0°，修正偏移方向
            if bucket == 'Red' and abs(hue_offset) > 180:
                hue_offset -= 360 if hue_offset > 0 else -360
            hue_adj = clamp(int(hue_offset * 1.5), -60, 60)
        else:
            hue_adj = 0

        lum_adj = clamp(int((stats['val_mean'] - 0.5) * 170), -65, 65)

        params[f'HueAdjustment{bucket}']        = hue_adj
        params[f'SaturationAdjustment{bucket}'] = sat_adj
        params[f'LuminanceAdjustment{bucket}']  = lum_adj

    return _apply_directional_focus(params)


def _derive_rgb_curves(ref_rgb: np.ndarray, src_rgb: Optional[np.ndarray] = None) -> dict:
    """
    RGB分量色调曲线：返回中性曲线。
    颜色信息已由HSL面板完整捕捉；RGB分量曲线会与HSL叠加产生色偏和过曝，
    因此统一输出线性恒等曲线，避免干扰。
    """
    identity = [(0, 0), (64, 64), (128, 128), (192, 192), (255, 255)]
    return {
        'tone_curve_red':   identity,
        'tone_curve_green': identity,
        'tone_curve_blue':  identity,
    }


def _enforce_monotone(points: list) -> list:
    """确保曲线控制点输出值非递减（Lightroom要求）"""
    result = list(points)
    for i in range(1, len(result)):
        if result[i][1] < result[i - 1][1]:
            result[i] = (result[i][0], result[i - 1][1])
    return result


def _analyze_color_grading(rgb_float: np.ndarray) -> dict:
    """单图模式：从参考图本身估算颜色分级（精度较低，备用）"""
    return _analyze_color_grading_diff(rgb_float, None)


def _analyze_color_grading_diff(ref_rgb: np.ndarray,
                                  src_rgb: Optional[np.ndarray]) -> dict:
    """
    颜色分级分析。

    双图模式（src_rgb 不为 None）：
        计算 ref 与 src 各区域的色调差值 → 真正需要 SplitToning 施加的量。
        公式：对每个区域取 ref_mean_rgb - src_mean_rgb，
              将差值向量转为 Hue + Saturation。

    单图模式：从 ref 绝对色调估算（精度较低）。
    """
    def _gray(img):
        return 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]

    ref_gray = _gray(ref_rgb)
    shadow_mask    = ref_gray < 0.33
    highlight_mask = ref_gray > 0.67
    midtone_mask   = (ref_gray >= 0.33) & (ref_gray <= 0.67)

    def _diff_hue_sat(mask, scale=1.8):
        """计算 ref-src 在区域内的颜色偏移，映射到 (hue, saturation)"""
        if mask.sum() < 100:
            return 0, 0
        ref_mean = np.array([ref_rgb[:, :, c][mask].mean() for c in range(3)])
        if src_rgb is not None:
            # 对齐 src_rgb 尺寸到 ref_rgb（如果用户上传的两张图大小不同）
            if src_rgb.shape[:2] != ref_rgb.shape[:2]:
                from scipy.ndimage import zoom
                scale_h = ref_rgb.shape[0] / src_rgb.shape[0]
                scale_w = ref_rgb.shape[1] / src_rgb.shape[1]
                src_rgb_aligned = np.array([
                    zoom(src_rgb[:, :, c], (scale_h, scale_w), order=1)
                    for c in range(3)
                ]).transpose(1, 2, 0)
            else:
                src_rgb_aligned = src_rgb
            src_mean = np.array([src_rgb_aligned[:, :, c][mask].mean() for c in range(3)])
            delta    = ref_mean - src_mean          # 需要"加入"多少色彩
        else:
            delta = ref_mean - ref_mean.mean()      # 单图：相对中性基准

        # delta → hue + saturation（笛卡尔 → 极坐标近似）
        r_d, g_d, b_d = float(delta[0]), float(delta[1]), float(delta[2])
        # 将 delta 解释为带符号的 RGB 色调偏移
        # 橙色: R↑ G↑ B↓, 蓝色: R↓ G↓ B↑, 绿色: G↑ R↓ B↓
        warm_score = r_d * 0.6 + g_d * 0.2 - b_d * 0.8   # 正 = 偏暖
        cool_score = b_d * 0.8 + g_d * 0.1 - r_d * 0.7   # 正 = 偏冷

        if abs(warm_score) < 0.005 and abs(cool_score) < 0.005:
            return 0, 0

        # 选主方向，映射到 Lightroom SplitToning 典型色相角
        if warm_score > cool_score and warm_score > 0:
            # 偏橙暖（约 30-50°）
            green_component = max(0.0, g_d - r_d * 0.5)
            hue = int(38 + green_component * 600)   # 纯橙38°，偏黄则角度↑
            hue = max(15, min(65, hue))
            strength = warm_score
        elif cool_score > warm_score and cool_score > 0:
            # 偏蓝青冷（约 185-230°）
            hue = int(210 - (g_d - b_d * 0.3) * 500)   # 纯蓝210°，偏青则角度↓
            hue = max(170, min(255, hue))
            strength = cool_score
        elif r_d > 0.01 and g_d < -0.005:
            hue = 0    # 红调
            strength = abs(r_d)
        elif g_d > 0.01 and r_d < 0 and b_d < 0:
            hue = 120  # 绿调
            strength = abs(g_d)
        else:
            return 0, 0

        sat = min(35, int(strength * scale * 600))
        return hue, sat

    shadow_hue,    shadow_sat    = _diff_hue_sat(shadow_mask,    scale=1.8)
    highlight_hue, highlight_sat = _diff_hue_sat(highlight_mask, scale=1.5)
    midtone_hue,   midtone_sat   = _diff_hue_sat(midtone_mask,   scale=0.8)

    hue_diff        = abs(shadow_hue - highlight_hue) if shadow_sat > 0 and highlight_sat > 0 else 0
    is_complementary = 100 < hue_diff < 250

    return {
        'SplitToningShadowHue':           shadow_hue,
        'SplitToningShadowSaturation':    shadow_sat,
        'SplitToningHighlightHue':        highlight_hue,
        'SplitToningHighlightSaturation': highlight_sat,
        'SplitToningBalance':             0,
        'ColorGradeMidtoneHue':           midtone_hue,
        'ColorGradeMidtoneSat':           midtone_sat,
        'ColorGradeShadowLum':            0,
        'ColorGradeHighlightLum':         0,
        'is_complementary_grading':       is_complementary,
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
        saturation = clamp(int(delta * 280), -80, 80)
        vibrance   = clamp(int(delta * 300) + 20, -80, 100)
    else:
        # 基准 0.38：有色像素的中性参考饱和度
        saturation = clamp(int((ref_mean_sat - 0.38) * 260), -75, 75)
        vibrance   = clamp(int(saturation * 1.3) + 20, -75, 100)

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


def _apply_directional_focus(params: dict) -> dict:
    """
    对 HSL 调整结果进行方向性聚焦，避免"每个通道都动一点"的分散感。

    规则：
    · 饱和度：主导桶（|adj| ≥ 最大值 × 40%）→ ×1.7 放大
              非主导正向小调整 → ×0.20 压缩（几乎清零）
              非主导负向（降饱和） → ×0.40 保留（不完全抹去）
    · 色相偏移：主导桶（|adj| ≥ 最大值 × 40%）→ ×1.5 放大
               非主导 → ×0.30 压缩
    · 明度：不做聚焦（亮度调整方向性弱，保留原值）
    """
    result = dict(params)

    # ── 饱和度聚焦 ──────────────────────────────────────────────────────────
    sat_vals = {b: params.get(f'SaturationAdjustment{b}', 0) for b in HUE_BUCKETS}
    max_sat  = max((abs(v) for v in sat_vals.values()), default=0)
    if max_sat >= 8:
        threshold = max_sat * 0.40
        for bucket, v in sat_vals.items():
            key = f'SaturationAdjustment{bucket}'
            if abs(v) >= threshold:
                result[key] = clamp(int(v * 1.7), -100, 100)
            elif v > 0:
                result[key] = int(v * 0.20)   # 正向小值 → 近似清零
            else:
                result[key] = int(v * 0.40)   # 负向（降饱和）→ 保留部分

    # ── 色相偏移聚焦 ─────────────────────────────────────────────────────────
    hue_vals = {b: params.get(f'HueAdjustment{b}', 0) for b in HUE_BUCKETS}
    max_hue  = max((abs(v) for v in hue_vals.values()), default=0)
    if max_hue >= 5:
        threshold = max_hue * 0.40
        for bucket, v in hue_vals.items():
            key = f'HueAdjustment{bucket}'
            if abs(v) >= threshold:
                result[key] = clamp(int(v * 1.5), -100, 100)
            else:
                result[key] = int(v * 0.30)

    return result


def _compute_calibration(ref_rgb: np.ndarray, src_rgb: Optional[np.ndarray]) -> dict:
    """
    推算相机校准面板参数（RedSaturation / GreenSaturation / BlueSaturation / ShadowTint）
    Mode B：比较原图与参考图的 RGB 通道比值差异
    Mode A：以参考图的通道偏差估算（中性图 R≈G≈B）
    """
    ref_r = float(ref_rgb[:, :, 0].mean())
    ref_g = float(ref_rgb[:, :, 1].mean())
    ref_b = float(ref_rgb[:, :, 2].mean())
    ref_g = max(ref_g, 0.01)

    if src_rgb is not None:
        src_r = float(src_rgb[:, :, 0].mean())
        src_g = float(src_rgb[:, :, 1].mean())
        src_b = float(src_rgb[:, :, 2].mean())
        src_g = max(src_g, 0.01)

        # 各通道相对绿通道的比值差 → 校准饱和度
        red_sat   = clamp(int((ref_r / ref_g - src_r / src_g) * 160), -40, 55)
        blue_sat  = clamp(int((ref_b / ref_g - src_b / src_g) * 130), -35, 45)
        green_sat = 0  # 绿通道作参考，不调整
        # 阴影色调：R-B 差值的变化
        shadow_tint = clamp(int((src_b / src_g - ref_b / ref_g) * 100), -20, 20)
    else:
        # Mode A：偏离中性（R=G=B）的程度
        red_sat   = clamp(int((ref_r / ref_g - 1.0) * 130), -30, 45)
        blue_sat  = clamp(int((ref_b / ref_g - 1.0) * 110), -25, 35)
        green_sat = 0
        shadow_tint = 0

    return {
        'ShadowTint':       shadow_tint,
        'RedHue':           0,
        'RedSaturation':    red_sat,
        'GreenHue':         0,
        'GreenSaturation':  green_sat,
        'BlueHue':          0,
        'BlueSaturation':   blue_sat,
    }


def clamp(value, lo, hi):
    return max(lo, min(hi, value))