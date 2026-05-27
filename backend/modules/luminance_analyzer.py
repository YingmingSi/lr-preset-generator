"""
亮度分析模块
分析直方图和色调曲线，生成LR亮度类参数
优先级：先亮度后色彩，符合专业修图顺序
"""

import numpy as np
from typing import Optional


# 曲线形状模板（仅用于报告识别，不强制替换算法结果）
CURVE_TEMPLATES = {
    'standard_s':     [(0,0),(64,50),(128,128),(192,210),(255,255)],
    'lifted_black_s': [(0,20),(64,60),(128,135),(192,210),(255,255)],
    'reverse_s':      [(0,0),(64,75),(128,128),(192,185),(255,255)],
    'filmic':         [(0,20),(64,70),(128,128),(192,190),(255,235)],
}

CURVE_TEMPLATE_LABELS = {
    'standard_s':     '标准S曲线',
    'lifted_black_s': '压黑S曲线（电影感）',
    'reverse_s':      '反S曲线（日系低对比）',
    'filmic':         '胶片曲线（无死黑死白）',
}


def analyze_luminance(ref_data: dict, src_data: Optional[dict] = None) -> dict:
    """
    分析亮度参数

    Args:
        ref_data: 参考图数据（目标风格）
        src_data: 原图数据（可选，有则做差值分析）

    Returns:
        亮度相关的LR参数字典
    """
    ref_gray = _to_grayscale(ref_data['rgb_float'])
    ref_hist = _compute_histogram(ref_gray)

    if src_data is not None:
        src_gray = _to_grayscale(src_data['rgb_float'])
        src_hist = _compute_histogram(src_gray)
        params = _diff_analysis(src_hist, ref_hist)
    else:
        params = _feature_analysis(ref_hist)

    # 推导色调曲线
    tone_curve = _derive_tone_curve(ref_gray)
    params['tone_curve'] = tone_curve

    # 识别曲线形状（仅用于报告）
    params['_curve_style'] = _match_curve_template(tone_curve)

    params['Clarity'] = _estimate_clarity(ref_data['rgb_float'])
    params['Dehaze']  = 0

    return params


def apply_luminance_linkage(params: dict) -> dict:
    """
    亮度参数联动校验
    基于专业修图经验的搭配规律
    """
    h       = params.get('Highlights', 0)
    s       = params.get('Shadows', 0)
    w       = params.get('Whites', 0)
    b       = params.get('Blacks', 0)
    contrast = params.get('Contrast', 0)
    clarity  = params.get('Clarity', 0)

    # 规律1：压高光时适当提白色，充分利用亮度空间且不死白
    if h < -40:
        params['Whites'] = max(w, int(-h * 0.25))

    # 规律2：提阴影时适当降黑色，防止阴影浮灰
    if s > 40:
        params['Blacks'] = min(b, -int(s * 0.25))

    # 规律3：高对比配合降清晰度，防止过于生硬
    if contrast > 40 and clarity > 20:
        params['Clarity'] = int(clarity * 0.6)

    # 规律4：低对比配合提清晰度，找回细节
    if contrast < -30 and clarity < 0:
        params['Clarity'] = max(clarity, 10)

    # 规律5：防止高光和白色同时大幅压低
    if h < -60 and w < -40:
        params['Whites'] = max(w, -20)

    # 规律6：防止阴影和黑色同时大幅提高导致浮灰
    if s > 50 and b > 20:
        params['Blacks'] = min(b, 10)

    return params


def _to_grayscale(rgb_float: np.ndarray) -> np.ndarray:
    """转换为感知亮度（Rec.709加权）"""
    return (0.2126 * rgb_float[:, :, 0] +
            0.7152 * rgb_float[:, :, 1] +
            0.0722 * rgb_float[:, :, 2])


def _compute_histogram(gray: np.ndarray, bins: int = 256) -> np.ndarray:
    """计算归一化直方图"""
    hist, _ = np.histogram(gray, bins=bins, range=(0, 1))
    return hist.astype(np.float32) / hist.sum()


def _compute_zone_stats(hist: np.ndarray) -> dict:
    """将直方图分为5个区间"""
    zones = {
        'blacks':     hist[:26],
        'shadows':    hist[26:77],
        'midtones':   hist[77:179],
        'highlights': hist[179:230],
        'whites':     hist[230:],
    }
    return {k: float(v.sum()) for k, v in zones.items()}


def _diff_analysis(src_hist: np.ndarray, ref_hist: np.ndarray) -> dict:
    """模式B：通过直方图差值推导LR参数"""
    src_zones = _compute_zone_stats(src_hist)
    ref_zones = _compute_zone_stats(ref_hist)

    def zone_delta(zone):
        src = src_zones[zone]
        ref = ref_zones[zone]
        if src < 1e-6:
            return 0
        return (ref - src) / max(src, 0.01)

    src_mean = sum(i * src_hist[i] for i in range(256)) / 255.0
    ref_mean = sum(i * ref_hist[i] for i in range(256)) / 255.0
    exposure_shift = (ref_mean - src_mean) * 2.5

    src_std  = _hist_std(src_hist)
    ref_std  = _hist_std(ref_hist)
    contrast = clamp((ref_std - src_std) / max(src_std, 0.01) * 150, -60, 60)

    highlights = clamp(-zone_delta('highlights') * 100, -80, 20)
    shadows    = clamp( zone_delta('shadows')    * 100, -20, 60)
    whites     = clamp(-zone_delta('whites')     *  80, -50, 30)
    blacks     = clamp( zone_delta('blacks')     *  80, -30, 30)

    return {
        'Exposure':   round(clamp(exposure_shift, -1.2, 1.2), 2),
        'Contrast':   int(contrast),
        'Highlights': int(highlights),
        'Shadows':    int(shadows),
        'Whites':     int(whites),
        'Blacks':     int(blacks),
    }


def _feature_analysis(ref_hist: np.ndarray) -> dict:
    """模式A：从单张参考图提取亮度风格特征"""
    zones = _compute_zone_stats(ref_hist)
    mean  = sum(i * ref_hist[i] for i in range(256)) / 255.0
    std   = _hist_std(ref_hist)

    exposure   = clamp((mean - 0.45) * 1.5, -1.0, 1.0)
    contrast   = clamp((std - 0.25) / 0.25 * 60, -50, 50)
    highlights = clamp((0.15 - zones['highlights'] - zones['whites']) * 300, -70, 20)
    shadows    = clamp((zones['blacks'] + zones['shadows'] - 0.2) * 200, -20, 50)
    whites     = clamp((0.05 - zones['whites']) * 300, -50, 30)
    blacks     = clamp((zones['blacks'] - 0.03) * 300, -30, 30)

    return {
        'Exposure':   round(exposure, 2),
        'Contrast':   int(contrast),
        'Highlights': int(highlights),
        'Shadows':    int(shadows),
        'Whites':     int(whites),
        'Blacks':     int(blacks),
    }


def _derive_tone_curve(gray: np.ndarray, num_points: int = 5) -> list:
    """
    从参考图推导色调曲线控制点
    使用分段亮度均值对比，避免CDF拉伸失真
    """
    input_points = np.linspace(0, 255, num_points, dtype=int)
    curve_points = []

    for inp in input_points:
        inp_norm = inp / 255.0
        half_win = 0.06  # 采样窗口±6%亮度范围
        lo = max(0.0, inp_norm - half_win)
        hi = min(1.0, inp_norm + half_win)
        mask = (gray >= lo) & (gray < hi)

        if mask.sum() > 50:
            zone_mean = float(gray[mask].mean())
            # 图像数据40%权重，输入点60%权重，防止曲线震荡
            out_norm = inp_norm * 0.6 + zone_mean * 0.4
        else:
            out_norm = inp_norm

        out = clamp(int(out_norm * 255), 0, 255)
        curve_points.append((int(inp), out))

    # 确保端点合理
    curve_points[0]  = (0,   clamp(curve_points[0][1],  0,  25))
    curve_points[-1] = (255, clamp(curve_points[-1][1], 230, 255))

    # 强制单调递增（LR要求）
    for i in range(1, len(curve_points)):
        if curve_points[i][1] < curve_points[i - 1][1]:
            curve_points[i] = (curve_points[i][0], curve_points[i - 1][1])

    return curve_points


def _match_curve_template(curve_points: list) -> str:
    """
    将推导出的曲线与模板做相似度匹配
    仅用于报告中告知用户识别到的曲线风格
    """
    def curve_to_array(pts):
        arr = np.zeros(256)
        pts = sorted(pts)
        for i in range(len(pts) - 1):
            x0, y0 = pts[i]
            x1, y1 = pts[i + 1]
            if x1 > x0:
                for x in range(x0, x1 + 1):
                    arr[x] = y0 + (y1 - y0) * (x - x0) / (x1 - x0)
        return arr

    input_arr = curve_to_array(curve_points)
    best_label = 'standard_s'
    best_dist  = float('inf')

    for key, template_pts in CURVE_TEMPLATES.items():
        tmpl_arr = curve_to_array(template_pts)
        dist = float(np.mean((input_arr - tmpl_arr) ** 2))
        if dist < best_dist:
            best_dist  = dist
            best_label = key

    return CURVE_TEMPLATE_LABELS.get(best_label, '标准S曲线')


def _hist_std(hist: np.ndarray) -> float:
    """计算直方图标准差（反映对比度）"""
    indices  = np.arange(len(hist))
    mean     = float(np.sum(indices * hist)) / len(hist)
    variance = float(np.sum(((indices - mean * len(hist)) ** 2) * hist))
    return np.sqrt(variance) / len(hist)


def _estimate_clarity(rgb_float: np.ndarray) -> int:
    """通过局部对比度估算Clarity值"""
    from PIL import Image, ImageFilter

    gray = (0.299 * rgb_float[:, :, 0] +
            0.587 * rgb_float[:, :, 1] +
            0.114 * rgb_float[:, :, 2])

    img     = Image.fromarray((gray * 255).astype(np.uint8))
    blurred = np.array(img.filter(ImageFilter.GaussianBlur(radius=8))) / 255.0
    mid_freq     = np.abs(gray - blurred)
    mid_contrast = float(mid_freq.mean())

    return clamp(int((mid_contrast - 0.05) * 500), -20, 35)


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def format_tone_curve_xml(curve_points: list) -> str:
    """将曲线控制点格式化为XMP的rdf:li格式"""
    lines = []
    for inp, out in curve_points:
        lines.append(f'          <rdf:li>{inp}, {out}</rdf:li>')
    return '\n'.join(lines)