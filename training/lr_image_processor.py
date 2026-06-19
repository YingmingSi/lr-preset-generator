"""
Lightroom 风格图像处理器（纯 Python 实现）

绕过 darktable-cli 不支持 Adobe XMP 格式的问题，
直接在 numpy/PIL 中实现 LR 基础调整。

支持的参数（22 个，与 CNN 训练目标对应）：
  曝光控制: Exposure, Highlights, Shadows, Blacks, Whites, Contrast
  色彩控制: Saturation, Vibrance, Clarity
  HSL: Saturation/Hue/Luminance Adjustments (Orange/Aqua/Green/Blue)
  Split Toning: Shadow/Highlight Hue & Saturation
"""

import numpy as np
from PIL import Image
import colorsys


def apply_lr_params(img_rgb: np.ndarray, params: dict) -> np.ndarray:
    """
    将 LR 参数应用到 RGB 图像。

    Args:
        img_rgb: (H, W, 3) RGB 图像，值 [0, 255] uint8 或 [0, 1] float
        params: LR 参数字典

    Returns:
        调整后的 RGB 图像 (H, W, 3) uint8
    """
    # 转 [0, 1] float
    if img_rgb.dtype == np.uint8:
        img = img_rgb.astype(np.float32) / 255.0
    else:
        img = img_rgb.astype(np.float32)

    # 1. 曝光（Exposure stops，-3 到 +3）
    exp = params.get('Exposure', 0)
    if exp != 0:
        img = img * (2.0 ** exp)
        img = np.clip(img, 0, 1)

    # 2. 对比度（-100 到 +100，S 曲线）
    contrast = params.get('Contrast', 0)
    if contrast != 0:
        # 简单 S 曲线：以 0.5 为中心
        amount = contrast / 100.0  # -1 到 1
        img = 0.5 + (img - 0.5) * (1 + amount)
        img = np.clip(img, 0, 1)

    # 3. 高光/阴影/黑色/白色（基于亮度分段调整）
    highlights = params.get('Highlights', 0) / 100.0  # -1 到 1
    shadows = params.get('Shadows', 0) / 100.0
    blacks = params.get('Blacks', 0) / 100.0
    whites = params.get('Whites', 0) / 100.0

    if any([highlights, shadows, blacks, whites]):
        # 计算亮度（用于权重）
        lum = img.mean(axis=2, keepdims=True)  # (H, W, 1)

        # 高光区域：增强系数 0.3 → 0.7，效果更明显
        if highlights != 0:
            # 软掩码：高光区域（lum > 0.55，超过的指数衰减）
            mask = np.clip((lum - 0.4) / 0.6, 0, 1) ** 2
            img = img + highlights * 0.7 * mask
        # 阴影区域：增强系数 0.3 → 0.7
        if shadows != 0:
            mask = np.clip((0.6 - lum) / 0.6, 0, 1) ** 2
            img = img + shadows * 0.7 * mask
        # 黑色（最暗 30%）：增强 0.2 → 0.5
        if blacks != 0:
            mask = np.clip((0.3 - lum) / 0.3, 0, 1) ** 2
            img = img + blacks * 0.5 * mask
        # 白色（最亮 30%）：增强 0.2 → 0.5
        if whites != 0:
            mask = np.clip((lum - 0.7) / 0.3, 0, 1) ** 2
            img = img + whites * 0.5 * mask

        img = np.clip(img, 0, 1)

    # 4. 清晰度（Clarity，中间调对比）：增强 0.15 → 0.4
    clarity = params.get('Clarity', 0) / 100.0
    if clarity != 0:
        mid_mask = 1 - np.abs(img - 0.5) * 2  # 中间调最强
        img = img + clarity * 0.4 * mid_mask * np.sign(img - 0.5)
        img = np.clip(img, 0, 1)

    # 5. 饱和度 & 活力（在 HSV 空间调整）
    saturation = params.get('Saturation', 0) / 100.0
    vibrance = params.get('Vibrance', 0) / 100.0

    if saturation != 0 or vibrance != 0:
        hsv = rgb_to_hsv_vectorized(img)
        s = hsv[..., 1]
        if saturation != 0:
            s = s * (1 + saturation)
        if vibrance != 0:
            # 活力：低饱和区域加得多，高饱和区域加得少
            weight = (1 - s) ** 2
            s = s + vibrance * 0.5 * weight
        hsv[..., 1] = np.clip(s, 0, 1)
        img = hsv_to_rgb_vectorized(hsv)
        img = np.clip(img, 0, 1)

    # 6. HSL 调整（按颜色区间）
    img = apply_hsl_adjustments(img, params)

    # 7. Split Toning（阴影和高光的色彩着色）
    img = apply_split_toning(img, params)

    return (img * 255).clip(0, 255).astype(np.uint8)


def rgb_to_hsv_vectorized(rgb: np.ndarray) -> np.ndarray:
    """向量化 RGB → HSV 转换。输入 [0,1]，输出 H[0,1], S[0,1], V[0,1]"""
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
    """向量化 HSV → RGB 转换"""
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


def apply_hsl_adjustments(img: np.ndarray, params: dict) -> np.ndarray:
    """
    HSL 色彩调整：按颜色区间（Orange/Aqua/Green/Blue）调整 H/S/L
    """
    color_ranges = {
        'Orange': (0.04, 0.11),   # 橙色: hue ≈ 0.05-0.10
        'Green':  (0.22, 0.42),   # 绿色: hue ≈ 0.25-0.4
        'Aqua':   (0.42, 0.55),   # 青色: hue ≈ 0.45-0.55
        'Blue':   (0.55, 0.72),   # 蓝色: hue ≈ 0.55-0.70
    }

    hsv = rgb_to_hsv_vectorized(img)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    for color, (h_lo, h_hi) in color_ranges.items():
        # 该颜色区间的软掩码（高斯权重）
        center = (h_lo + h_hi) / 2
        sigma = (h_hi - h_lo) / 2
        mask = np.exp(-((h - center) ** 2) / (2 * sigma ** 2))

        # 调整 H：增强 5% → 18%（更明显的色相偏移）
        hue_shift = params.get(f'HueAdjustment{color}', 0) / 100.0 * 0.18
        if hue_shift != 0:
            h = (h + hue_shift * mask) % 1.0

        # 调整 S：增强 0.5 → 0.9（接近完全饱和度反转）
        sat_shift = params.get(f'SaturationAdjustment{color}', 0) / 100.0
        if sat_shift != 0:
            s = np.clip(s + sat_shift * mask * 0.9, 0, 1)

        # 调整 L（亮度）：增强 0.3 → 0.6
        lum_shift = params.get(f'LuminanceAdjustment{color}', 0) / 100.0
        if lum_shift != 0:
            v = np.clip(v + lum_shift * mask * 0.6, 0, 1)

    hsv = np.stack([h, s, v], axis=-1)
    return hsv_to_rgb_vectorized(hsv)


def apply_split_toning(img: np.ndarray, params: dict) -> np.ndarray:
    """Split Toning：阴影和高光区域分别添加色调"""
    shadow_hue = params.get('SplitToningShadowHue', 0)
    shadow_sat = params.get('SplitToningShadowSaturation', 0) / 100.0
    highlight_hue = params.get('SplitToningHighlightHue', 0)
    highlight_sat = params.get('SplitToningHighlightSaturation', 0) / 100.0

    if shadow_sat == 0 and highlight_sat == 0:
        return img

    lum = img.mean(axis=2, keepdims=True)

    # 阴影上色：增强强度 0.3 → 0.6
    if shadow_sat > 0:
        sh_color = hsv_to_single(shadow_hue / 360.0, shadow_sat, 1.0)
        mask = np.clip((0.5 - lum) * 2, 0, 1)
        for c in range(3):
            img[..., c] = img[..., c] * (1 - mask[..., 0] * 0.6) + sh_color[c] * mask[..., 0] * 0.6

    # 高光上色：增强 0.3 → 0.6
    if highlight_sat > 0:
        hi_color = hsv_to_single(highlight_hue / 360.0, highlight_sat, 1.0)
        mask = np.clip((lum - 0.5) * 2, 0, 1)
        for c in range(3):
            img[..., c] = img[..., c] * (1 - mask[..., 0] * 0.6) + hi_color[c] * mask[..., 0] * 0.6

    return np.clip(img, 0, 1)


def hsv_to_single(h, s, v):
    """单像素 HSV → RGB"""
    return np.array(colorsys.hsv_to_rgb(h, s, v))


if __name__ == '__main__':
    # 测试
    from PIL import Image
    import os

    # 加载测试图
    test_path = './photos'
    photos = [f for f in os.listdir(test_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not photos:
        # 用 CR3 转出的图
        photos = [f for f in os.listdir('./data') if f.endswith('_src.jpg')]
        if photos:
            img = np.array(Image.open(f'./data/{photos[0]}').convert('RGB'))
        else:
            print("找不到测试图")
            exit()
    else:
        img = np.array(Image.open(f'{test_path}/{photos[0]}').convert('RGB'))

    print(f"测试图: {photos[0]}, shape: {img.shape}")

    # 应用极端参数
    extreme_params = {
        'Exposure': 1.5,
        'Contrast': 50,
        'Highlights': -80,
        'Shadows': 60,
        'Saturation': 50,
        'SaturationAdjustmentOrange': 80,
    }

    result = apply_lr_params(img, extreme_params)

    # 计算差异
    diff = np.abs(img.astype(np.float32) - result.astype(np.float32)).mean()
    print(f"\n参数: {extreme_params}")
    print(f"像素差异: {diff:.2f} / 255 = {diff/255*100:.1f}%")

    # 保存对比
    Image.fromarray(result).save('/tmp/lr_result.jpg')
    Image.fromarray(img).save('/tmp/lr_input.jpg')
    print(f"\n✓ 已保存:\n  原图: /tmp/lr_input.jpg\n  处理: /tmp/lr_result.jpg")
