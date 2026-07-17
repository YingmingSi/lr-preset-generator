"""
Lightroom 风格图像处理器（72 维参数版，numpy）

把 72 个 LR 参数应用到 RGB 图像，用于生成训练数据。
与 lr_image_processor_torch.py（可微版）保持像素级一致。

处理顺序（近似 LR 内部管线）：
  1. 校准（RGB 原色 H/S 偏移）
  2. 曝光 → 对比度 → 高光/阴影/白/黑
  3. 色调曲线（Luma + RGB 四条）
  4. Texture / Clarity / Dehaze（局部对比）
  5. 饱和度 / 鲜艳度
  6. HSL 混色器（8 色 × H/S/L）
  7. 颜色分级（阴影/中间调/高光 3 区）
"""

import numpy as np
import colorsys
# 注：scipy 仅用于局部对比（Texture/Clarity/Dehaze）。后端始终 skip_local，
# 故惰性导入，避免启动时加载 scipy 占内存。

from modules.params_config import HSL_COLORS, HSL_COLOR_HUE


# ─── HSV 向量化转换 ───────────────────────────────────────────────────────

def rgb_to_hsv_vectorized(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    v = maxc
    delta = maxc - minc
    s = np.where(maxc > 0, delta / (maxc + 1e-10), 0)
    rc = np.where(delta > 0, (maxc - r) / (delta + 1e-10), 0)
    gc = np.where(delta > 0, (maxc - g) / (delta + 1e-10), 0)
    bc = np.where(delta > 0, (maxc - b) / (delta + 1e-10), 0)
    h = np.where(r == maxc, bc - gc,
        np.where(g == maxc, 2.0 + rc - bc, 4.0 + gc - rc))
    h = (h / 6.0) % 1.0
    return np.stack([h, s, v], axis=-1)


def hsv_to_rgb_vectorized(hsv: np.ndarray) -> np.ndarray:
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    i = np.floor(h * 6).astype(int)
    f = h * 6 - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    i = i % 6
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=-1)


def hsv_to_single(h, s, v):
    return np.array(colorsys.hsv_to_rgb(h % 1.0, s, v))


def _color_mask(h: np.ndarray, color: str) -> np.ndarray:
    """某 HSL 颜色的软掩码（高斯权重，处理 Red 跨 0 边界）"""
    h_lo, h_hi = HSL_COLOR_HUE[color]
    center = (h_lo + h_hi) / 2
    sigma = max((h_hi - h_lo) / 2, 1e-3)
    if color == 'Red':
        d = np.minimum(np.abs(h - 0.0), np.abs(h - 1.0))
    else:
        d = h - center
    return np.exp(-(d ** 2) / (2 * sigma ** 2))


# ─── 主入口 ───────────────────────────────────────────────────────────────

def apply_lr_params(img_rgb: np.ndarray, params: dict, skip_local: bool = False) -> np.ndarray:
    """
    应用 72 个 LR 参数。

    Args:
        img_rgb: (H, W, 3) RGB，uint8 [0,255] 或 float [0,1]
        params:  参数字典（缺失的按 0 处理）

    Returns:
        (H, W, 3) uint8
    """
    if img_rgb.dtype == np.uint8:
        img = img_rgb.astype(np.float32) / 255.0
    else:
        img = img_rgb.astype(np.float32).copy()

    img = _apply_calibration(img, params)
    img = _apply_basic_tone(img, params)
    img = _apply_tone_curves(img, params)
    if not skip_local:
        # 空间操作（纹理/清晰度/去朦胧）——LUT 烘焙时跳过（逐像素网格上无意义）
        img = _apply_local_contrast(img, params)
    img = _apply_sat_vibrance(img, params)
    img = _apply_hsl(img, params)
    img = _apply_color_grading(img, params)

    return (img * 255).clip(0, 255).astype(np.uint8)


# ─── 1. 校准（RGB 原色 H/S 偏移）──────────────────────────────────────────

def _apply_calibration(img: np.ndarray, p: dict) -> np.ndarray:
    primaries = {'Red': 0.0, 'Green': 1/3, 'Blue': 2/3}
    has = any(p.get(f'{c}Hue', 0) or p.get(f'{c}Saturation', 0) for c in primaries)
    if not has:
        return img
    hsv = rgb_to_hsv_vectorized(img)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    sigma = 0.18  # 原色影响较宽
    for c, center in primaries.items():
        hue_shift = p.get(f'{c}Hue', 0) / 100.0 * 0.1        # ±10% 色相
        sat_shift = p.get(f'{c}Saturation', 0) / 100.0
        if hue_shift == 0 and sat_shift == 0:
            continue
        d = np.minimum(np.abs(h - center), 1.0 - np.abs(h - center))
        mask = np.exp(-(d ** 2) / (2 * sigma ** 2))
        if hue_shift:
            h = (h + hue_shift * mask) % 1.0
        if sat_shift:
            s = np.clip(s + sat_shift * mask * 0.5, 0, 1)
    return np.clip(hsv_to_rgb_vectorized(np.stack([h, s, v], axis=-1)), 0, 1)


# ─── 2. 基础影调 ─────────────────────────────────────────────────────────

def _apply_basic_tone(img: np.ndarray, p: dict) -> np.ndarray:
    exp = p.get('Exposure', 0)
    if exp:
        img = np.clip(img * (2.0 ** exp), 0, 1)

    contrast = p.get('Contrast', 0) / 100.0
    if contrast:
        img = np.clip(0.5 + (img - 0.5) * (1 + contrast), 0, 1)

    highlights = p.get('Highlights', 0) / 100.0
    shadows    = p.get('Shadows', 0) / 100.0
    blacks     = p.get('Blacks', 0) / 100.0
    whites     = p.get('Whites', 0) / 100.0
    if any([highlights, shadows, blacks, whites]):
        lum = img.mean(axis=2, keepdims=True)
        if highlights:
            img = img + highlights * 0.7 * (np.clip((lum - 0.4) / 0.6, 0, 1) ** 2)
        if shadows:
            img = img + shadows * 0.7 * (np.clip((0.6 - lum) / 0.6, 0, 1) ** 2)
        if blacks:
            img = img + blacks * 0.5 * (np.clip((0.3 - lum) / 0.3, 0, 1) ** 2)
        if whites:
            img = img + whites * 0.5 * (np.clip((lum - 0.7) / 0.3, 0, 1) ** 2)
        img = np.clip(img, 0, 1)
    return img


# ─── 3. 色调曲线（Luma + RGB）────────────────────────────────────────────

_CURVE_X = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)


def _build_curve_lut(offsets, n=256):
    """5 点偏移 → 256 级 LUT。offsets 单位是 0-255 空间的 y 偏移。"""
    ctrl_y = np.clip(_CURVE_X + np.array(offsets, dtype=np.float32) / 255.0, 0, 1)
    xs = np.linspace(0, 1, n)
    return np.interp(xs, _CURVE_X, ctrl_y).astype(np.float32)


def _apply_curve_channel(channel: np.ndarray, offsets) -> np.ndarray:
    if not any(offsets):
        return channel
    lut = _build_curve_lut(offsets)
    idx = np.clip((channel * 255).astype(int), 0, 255)
    return lut[idx]


def _apply_tone_curves(img: np.ndarray, p: dict) -> np.ndarray:
    luma_off = [p.get(f'LumaCurve{i}', 0) for i in range(5)]
    if any(luma_off):
        lum = img.mean(axis=2)
        new_lum = _apply_curve_channel(lum, luma_off)
        ratio = (new_lum / (lum + 1e-6))[..., None]
        img = np.clip(img * ratio, 0, 1)

    for ci, cn in enumerate(['Red', 'Green', 'Blue']):
        off = [p.get(f'{cn}Curve{i}', 0) for i in range(5)]
        if any(off):
            img[..., ci] = _apply_curve_channel(img[..., ci], off)
    return np.clip(img, 0, 1)


# ─── 4. 局部对比（Texture / Clarity / Dehaze）────────────────────────────

def _apply_local_contrast(img: np.ndarray, p: dict) -> np.ndarray:
    texture = p.get('Texture', 0) / 100.0
    clarity = p.get('Clarity', 0) / 100.0
    dehaze  = p.get('Dehaze', 0) / 100.0

    from scipy.ndimage import gaussian_filter  # 惰性导入（仅此函数用到）

    lum = img.mean(axis=2)

    # Texture：中频细节（小半径 unsharp）
    if texture:
        blur = gaussian_filter(lum, sigma=2.0)
        detail = (lum - blur)[..., None]
        img = np.clip(img + texture * 1.2 * detail, 0, 1)

    # Clarity：中间调局部对比（大半径 unsharp，中间调加权）
    if clarity:
        lum2 = img.mean(axis=2)
        blur = gaussian_filter(lum2, sigma=8.0)
        detail = (lum2 - blur)[..., None]
        mid_w = (1 - np.abs(lum2 - 0.5) * 2)[..., None]
        img = np.clip(img + clarity * 1.0 * detail * mid_w, 0, 1)

    # Dehaze：增对比 + 增饱和（简化全局版）
    if dehaze:
        img = np.clip(0.5 + (img - 0.5) * (1 + dehaze * 0.5), 0, 1)
        hsv = rgb_to_hsv_vectorized(img)
        hsv[..., 1] = np.clip(hsv[..., 1] * (1 + dehaze * 0.4), 0, 1)
        img = np.clip(hsv_to_rgb_vectorized(hsv), 0, 1)
    return img


# ─── 5. 饱和度 / 鲜艳度 ──────────────────────────────────────────────────

def _apply_sat_vibrance(img: np.ndarray, p: dict) -> np.ndarray:
    saturation = p.get('Saturation', 0) / 100.0
    vibrance   = p.get('Vibrance', 0) / 100.0
    if saturation == 0 and vibrance == 0:
        return img
    hsv = rgb_to_hsv_vectorized(img)
    s = hsv[..., 1]
    if saturation:
        s = s * (1 + saturation)
    if vibrance:
        s = s + vibrance * 0.5 * (1 - s) ** 2
    hsv[..., 1] = np.clip(s, 0, 1)
    return np.clip(hsv_to_rgb_vectorized(hsv), 0, 1)


# ─── 6. HSL 混色器（8 色 × H/S/L）────────────────────────────────────────

def _apply_hsl(img: np.ndarray, p: dict) -> np.ndarray:
    has = any(p.get(f'{t}Adjustment{c}', 0)
              for t in ['Hue', 'Saturation', 'Luminance'] for c in HSL_COLORS)
    if not has:
        return img
    hsv = rgb_to_hsv_vectorized(img)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    for color in HSL_COLORS:
        hue_shift = p.get(f'HueAdjustment{color}', 0) / 100.0 * 0.18
        sat_shift = p.get(f'SaturationAdjustment{color}', 0) / 100.0
        lum_shift = p.get(f'LuminanceAdjustment{color}', 0) / 100.0
        if hue_shift == 0 and sat_shift == 0 and lum_shift == 0:
            continue
        mask = _color_mask(h, color)
        if hue_shift:
            h = (h + hue_shift * mask) % 1.0
        if sat_shift:
            s = np.clip(s + sat_shift * mask * 0.9, 0, 1)
        if lum_shift:
            v = np.clip(v + lum_shift * mask * 0.6, 0, 1)
    return np.clip(hsv_to_rgb_vectorized(np.stack([h, s, v], axis=-1)), 0, 1)


# ─── 7. 颜色分级（阴影/中间调/高光 3 区）─────────────────────────────────

def _apply_color_grading(img: np.ndarray, p: dict) -> np.ndarray:
    balance  = p.get('ColorGradeBalance', 0) / 100.0    # -1..1
    blending = p.get('ColorGradeBlending', 50) / 100.0  # 0..1

    # 任一区有 Sat>0 或 Lum≠0 才需要处理（Lum 独立于 Sat 生效）
    active = any(
        p.get(f'ColorGrade{z}Sat', 0) > 0 or p.get(f'ColorGrade{z}Lum', 0) != 0
        for z in ['Shadow', 'Midtone', 'Highlight']
    )
    if not active:
        return img

    lum = img.mean(axis=2, keepdims=True)
    # balance 偏移阴影/高光区间分界
    sh_edge = 0.5 + balance * 0.2
    hi_edge = 0.5 + balance * 0.2

    for zone in ['Shadow', 'Midtone', 'Highlight']:
        sat = p.get(f'ColorGrade{zone}Sat', 0) / 100.0
        hue = p.get(f'ColorGrade{zone}Hue', 0)
        lum_adj = p.get(f'ColorGrade{zone}Lum', 0) / 100.0
        if sat <= 0 and lum_adj == 0:
            continue

        if zone == 'Shadow':
            mask = np.clip((sh_edge - lum) / max(sh_edge, 1e-3), 0, 1)
        elif zone == 'Highlight':
            mask = np.clip((lum - hi_edge) / max(1 - hi_edge, 1e-3), 0, 1)
        else:  # Midtone
            mask = 1 - np.abs(lum - 0.5) * 2
        mask = np.clip(mask, 0, 1) * (0.3 + 0.7 * blending)

        if sat > 0:
            tint = hsv_to_single(hue / 360.0, 1.0, 1.0)
            tint_mean = (tint[0] + tint[1] + tint[2]) / 3.0
            # 系数 1.5：sat≤10 范围内也能产生可见 tint（sat=10 → ~15%）
            blend = mask[..., 0] * 1.5 * sat
            # 只移色相、保亮度：向 (像素亮度 + tint 色度) 混合，避免把阴影抬成亮色
            for c in range(3):
                target = lum[..., 0] + (tint[c] - tint_mean)
                img[..., c] = img[..., c] * (1 - blend) + target * blend
        if lum_adj:
            img = img + lum_adj * 0.3 * mask
    return np.clip(img, 0, 1)


if __name__ == '__main__':
    import sys
    from PIL import Image
    img = np.array(Image.open('./data/000005_src.jpg').convert('RGB')) \
        if __import__('os').path.exists('./data/000005_src.jpg') \
        else np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)

    test = {
        'Exposure': 0.8, 'Contrast': 20, 'Highlights': -40, 'Shadows': 30,
        'Texture': 20, 'Clarity': 15, 'Dehaze': 10, 'Vibrance': 30, 'Saturation': 15,
        'LumaCurve2': 8, 'RedCurve4': 5,
        'HueAdjustmentOrange': 30, 'SaturationAdjustmentBlue': -40,
        'LuminanceAdjustmentRed': 20,
        'ColorGradeShadowHue': 220, 'ColorGradeShadowSat': 8,
        'ColorGradeHighlightHue': 40, 'ColorGradeHighlightSat': 6,
        'RedHue': 20, 'BlueSaturation': 30,
    }
    out = apply_lr_params(img, test)
    diff = np.abs(img.astype(float) - out.astype(float)).mean()
    print(f"测试渲染像素差异: {diff:.2f} / 255")
    print(f"输出 shape: {out.shape}, dtype: {out.dtype}")
