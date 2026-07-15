"""
Lightroom 风格图像处理器（72 维，PyTorch 可微版）

与 lr_image_processor.py（numpy）保持像素级一致，用于训练时的像素重构 loss。

输入:
    src:         (B, 3, H, W) [0, 1]
    params_norm: (B, 72) 归一化 [-1, 1]
输出:
    (B, 3, H, W) [0, 1]
"""

import torch
import torch.nn.functional as F

from params_config import PARAM_ORDER, PARAM_RANGES, HSL_COLORS, HSL_COLOR_HUE


# ─── 参数反归一化 ─────────────────────────────────────────────────────────

def _range_tensors(device, dtype):
    mids = torch.tensor(
        [(PARAM_RANGES[p][0] + PARAM_RANGES[p][1]) / 2 for p in PARAM_ORDER],
        device=device, dtype=dtype)
    spans = torch.tensor(
        [(PARAM_RANGES[p][1] - PARAM_RANGES[p][0]) / 2 for p in PARAM_ORDER],
        device=device, dtype=dtype)
    return mids, spans


_IDX = {name: i for i, name in enumerate(PARAM_ORDER)}


# ─── HSV 转换（可微）──────────────────────────────────────────────────────

def rgb_to_hsv(rgb):
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    maxc, _ = rgb.max(dim=1, keepdim=True)
    minc, _ = rgb.min(dim=1, keepdim=True)
    v = maxc
    delta = maxc - minc
    s = torch.where(maxc > 1e-8, delta / (maxc + 1e-10), torch.zeros_like(maxc))
    rc = (maxc - r) / (delta + 1e-10)
    gc = (maxc - g) / (delta + 1e-10)
    bc = (maxc - b) / (delta + 1e-10)
    h = torch.where(r >= maxc, bc - gc,
        torch.where(g >= maxc, 2.0 + rc - bc, 4.0 + gc - rc))
    h = (h / 6.0) % 1.0
    h = torch.where(delta < 1e-8, torch.zeros_like(h), h)
    return torch.cat([h, s, v], dim=1)


def hsv_to_rgb(hsv):
    h, s, v = hsv[:, 0:1], hsv[:, 1:2], hsv[:, 2:3]
    i = (h * 6.0).floor()
    f = h * 6.0 - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    im = (i.long() % 6)
    r = torch.where(im == 0, v, torch.where(im == 1, q, torch.where(im == 2, p,
        torch.where(im == 3, p, torch.where(im == 4, t, v)))))
    g = torch.where(im == 0, t, torch.where(im == 1, v, torch.where(im == 2, v,
        torch.where(im == 3, q, torch.where(im == 4, p, p)))))
    b = torch.where(im == 0, p, torch.where(im == 1, p, torch.where(im == 2, t,
        torch.where(im == 3, v, torch.where(im == 4, v, q)))))
    return torch.cat([r, g, b], dim=1)


def _bc(x):
    """(B,) → (B,1,1,1)"""
    return x.view(-1, 1, 1, 1)


def _color_mask(h, color):
    h_lo, h_hi = HSL_COLOR_HUE[color]
    center = (h_lo + h_hi) / 2
    sigma = max((h_hi - h_lo) / 2, 1e-3)
    if color == 'Red':
        d = torch.minimum((h - 0.0).abs(), (h - 1.0).abs())
    else:
        d = h - center
    return torch.exp(-(d ** 2) / (2 * sigma ** 2))


# ─── 可微高斯模糊（separable conv）────────────────────────────────────────

_GAUSS_CACHE = {}


def _gaussian_kernel(sigma, device, dtype):
    key = (round(sigma, 2), device, dtype)
    if key not in _GAUSS_CACHE:
        radius = max(int(4 * sigma + 0.5), 1)
        x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
        k = torch.exp(-(x ** 2) / (2 * sigma ** 2))
        k = k / k.sum()
        _GAUSS_CACHE[key] = k
    return _GAUSS_CACHE[key]


def _gaussian_blur(x, sigma):
    """x: (B,1,H,W) → 模糊后同形状（separable）"""
    k = _gaussian_kernel(sigma, x.device, x.dtype)
    r = (k.numel() - 1) // 2
    kx = k.view(1, 1, 1, -1)
    ky = k.view(1, 1, -1, 1)
    x = F.pad(x, (r, r, 0, 0), mode='reflect')
    x = F.conv2d(x, kx)
    x = F.pad(x, (0, 0, r, r), mode='reflect')
    x = F.conv2d(x, ky)
    return x


# ─── 可微色调曲线（5 点分段线性）─────────────────────────────────────────

_CURVE_X = [0.0, 0.25, 0.5, 0.75, 1.0]


def _apply_curve(channel, offsets):
    """
    channel: (B,1,H,W) [0,1]；offsets: (B,5) 在 0-255 空间
    分段线性插值，可微 wrt offsets。
    """
    x = channel.clamp(0, 1)
    Y = (torch.tensor(_CURVE_X, device=channel.device, dtype=channel.dtype)
         .unsqueeze(0) + offsets / 255.0).clamp(0, 1)  # (B,5)
    out = torch.zeros_like(x)
    for s in range(4):
        x0, x1 = _CURVE_X[s], _CURVE_X[s + 1]
        y0 = _bc(Y[:, s])
        y1 = _bc(Y[:, s + 1])
        local = ((x - x0) / (x1 - x0)).clamp(0, 1)
        seg_val = y0 + (y1 - y0) * local
        in_seg = ((x >= x0) & (x < x1)).float() if s < 3 else (x >= x0).float()
        out = out + seg_val * in_seg
    return out


# ─── 主入口 ───────────────────────────────────────────────────────────────

def apply_lr_params_torch(src, params_norm):
    mids, spans = _range_tensors(src.device, src.dtype)
    params = params_norm * spans + mids   # (B, 72) 原始范围

    _zero = torch.zeros(params.shape[0], device=params.device, dtype=params.dtype)

    def P(name):
        # 缺失参数（61 维裁剪掉的）返回 0，与 numpy 处理器 .get(key,0) 一致
        i = _IDX.get(name)
        return params[:, i] if i is not None else _zero

    img = src.clamp(0, 1)

    img = _calibration(img, P)
    img = _basic_tone(img, P)
    img = _tone_curves(img, P)
    img = _local_contrast(img, P)
    img = _sat_vibrance(img, P)
    img = _hsl(img, P)
    img = _color_grading(img, P)
    return img.clamp(0, 1)


def _calibration(img, P):
    primaries = {'Red': 0.0, 'Green': 1/3, 'Blue': 2/3}
    hsv = rgb_to_hsv(img)
    h, s, v = hsv[:, 0:1], hsv[:, 1:2], hsv[:, 2:3]
    sigma = 0.18
    changed = False
    for c, center in primaries.items():
        hue_shift = P(f'{c}Hue') / 100.0 * 0.1
        sat_shift = P(f'{c}Saturation') / 100.0
        d = torch.minimum((h - center).abs(), 1.0 - (h - center).abs())
        mask = torch.exp(-(d ** 2) / (2 * sigma ** 2))
        h = (h + _bc(hue_shift) * mask) % 1.0
        s = (s + _bc(sat_shift) * mask * 0.5).clamp(0, 1)
        changed = True
    if not changed:
        return img
    return hsv_to_rgb(torch.cat([h, s, v], dim=1)).clamp(0, 1)


def _basic_tone(img, P):
    img = (img * _bc(2.0 ** P('Exposure'))).clamp(0, 1)
    contrast = P('Contrast') / 100.0
    img = (0.5 + (img - 0.5) * _bc(1 + contrast)).clamp(0, 1)

    hi = P('Highlights') / 100.0
    sh = P('Shadows') / 100.0
    bk = P('Blacks') / 100.0
    wh = P('Whites') / 100.0
    lum = img.mean(dim=1, keepdim=True)
    img = img + _bc(hi * 0.7) * (((lum - 0.4) / 0.6).clamp(0, 1) ** 2)
    img = img + _bc(sh * 0.7) * (((0.6 - lum) / 0.6).clamp(0, 1) ** 2)
    img = img + _bc(bk * 0.5) * (((0.3 - lum) / 0.3).clamp(0, 1) ** 2)
    img = img + _bc(wh * 0.5) * (((lum - 0.7) / 0.3).clamp(0, 1) ** 2)
    return img.clamp(0, 1)


def _tone_curves(img, P):
    # 缺失曲线点（RGB 首尾锚点已裁剪）经 P 返回 0，等价固定为 identity 锚点
    luma_off = torch.stack([P(f'LumaCurve{i}') for i in range(5)], dim=1)
    lum = img.mean(dim=1, keepdim=True)
    new_lum = _apply_curve(lum, luma_off)
    ratio = new_lum / (lum + 1e-6)
    img = (img * ratio).clamp(0, 1)

    for ci, cn in enumerate(['Red', 'Green', 'Blue']):
        off = torch.stack([P(f'{cn}Curve{i}') for i in range(5)], dim=1)
        ch = _apply_curve(img[:, ci:ci+1], off)
        img = torch.cat([img[:, :ci], ch, img[:, ci+1:]], dim=1)
    return img.clamp(0, 1)


def _local_contrast(img, P):
    texture = P('Texture') / 100.0
    clarity = P('Clarity') / 100.0
    dehaze  = P('Dehaze') / 100.0

    lum = img.mean(dim=1, keepdim=True)
    blur2 = _gaussian_blur(lum, 2.0)
    img = (img + _bc(texture * 1.2) * (lum - blur2)).clamp(0, 1)

    lum2 = img.mean(dim=1, keepdim=True)
    blur8 = _gaussian_blur(lum2, 8.0)
    mid_w = 1 - (lum2 - 0.5).abs() * 2
    img = (img + _bc(clarity * 1.0) * (lum2 - blur8) * mid_w).clamp(0, 1)

    img = (0.5 + (img - 0.5) * _bc(1 + dehaze * 0.5)).clamp(0, 1)
    hsv = rgb_to_hsv(img)
    hsv = torch.cat([hsv[:, 0:1],
                     (hsv[:, 1:2] * _bc(1 + dehaze * 0.4)).clamp(0, 1),
                     hsv[:, 2:3]], dim=1)
    img = hsv_to_rgb(hsv).clamp(0, 1)
    return img


def _sat_vibrance(img, P):
    saturation = P('Saturation') / 100.0
    vibrance   = P('Vibrance') / 100.0
    hsv = rgb_to_hsv(img)
    s = hsv[:, 1:2]
    s = s * _bc(1 + saturation)
    s = s + _bc(vibrance * 0.5) * (1 - s) ** 2
    s = s.clamp(0, 1)
    hsv = torch.cat([hsv[:, 0:1], s, hsv[:, 2:3]], dim=1)
    return hsv_to_rgb(hsv).clamp(0, 1)


def _hsl(img, P):
    hsv = rgb_to_hsv(img)
    h, s, v = hsv[:, 0:1], hsv[:, 1:2], hsv[:, 2:3]
    for color in HSL_COLORS:
        hue_shift = P(f'HueAdjustment{color}') / 100.0 * 0.18
        sat_shift = P(f'SaturationAdjustment{color}') / 100.0
        lum_shift = P(f'LuminanceAdjustment{color}') / 100.0
        mask = _color_mask(h, color)
        h = (h + _bc(hue_shift) * mask) % 1.0
        s = (s + _bc(sat_shift) * mask * 0.9).clamp(0, 1)
        v = (v + _bc(lum_shift) * mask * 0.6).clamp(0, 1)
    return hsv_to_rgb(torch.cat([h, s, v], dim=1)).clamp(0, 1)


def _color_grading(img, P):
    balance  = P('ColorGradeBalance') / 100.0
    blending = P('ColorGradeBlending') / 100.0
    lum = img.mean(dim=1, keepdim=True)
    sh_edge = _bc(0.5 + balance * 0.2)
    hi_edge = _bc(0.5 + balance * 0.2)

    for zone in ['Shadow', 'Midtone', 'Highlight']:
        sat = (P(f'ColorGrade{zone}Sat') / 100.0).clamp(min=0)
        hue = P(f'ColorGrade{zone}Hue')
        lum_adj = P(f'ColorGrade{zone}Lum') / 100.0

        if zone == 'Shadow':
            mask = ((sh_edge - lum) / sh_edge.clamp(min=1e-3)).clamp(0, 1)
        elif zone == 'Highlight':
            mask = ((lum - hi_edge) / (1 - hi_edge).clamp(min=1e-3)).clamp(0, 1)
        else:
            mask = 1 - (lum - 0.5).abs() * 2
        mask = mask.clamp(0, 1) * _bc(0.3 + 0.7 * blending)

        # tint 纯色（系数 1.5：与 numpy 一致，sat≤10 也可见）
        tint_hsv = torch.stack([hue / 360.0, torch.ones_like(hue), torch.ones_like(hue)], dim=-1)
        tint_rgb = hsv_to_rgb(tint_hsv.view(-1, 3, 1, 1))
        blend = mask * _bc(1.5 * sat)
        img = img * (1 - blend) + tint_rgb * blend
        img = img + _bc(lum_adj * 0.3) * mask
    return img.clamp(0, 1)


if __name__ == '__main__':
    import numpy as np
    from PIL import Image
    import os
    from lr_image_processor import apply_lr_params
    import params_config as pc

    img = np.array(Image.open('./data/000005_src.jpg').convert('RGB')) \
        if os.path.exists('./data/000005_src.jpg') \
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

    np_out = apply_lr_params(img, test)

    norm = np.array([pc.normalize(test.get(p, 0), p) for p in pc.PARAM_ORDER], dtype=np.float32)
    src_t = torch.from_numpy(img.astype(np.float32) / 255).permute(2, 0, 1).unsqueeze(0)
    p_t = torch.from_numpy(norm).unsqueeze(0)
    with torch.no_grad():
        t_out = apply_lr_params_torch(src_t, p_t)
    t_np = (t_out[0].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)

    diff = np.abs(np_out.astype(float) - t_np.astype(float))
    print(f"=== numpy vs torch 一致性 ===")
    print(f"  平均差异: {diff.mean():.2f} / 255")
    print(f"  最大差异: {diff.max():.2f} / 255")
    print("  ✅ 一致" if diff.mean() < 3 else "  ⚠ 需检查")
