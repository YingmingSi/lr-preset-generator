"""
Lightroom 风格图像处理器 — PyTorch 可微分版

与 lr_image_processor.py 行为一致，但全部用 torch tensor 操作，
允许通过反向传播让模型学习"参数 → 真实像素"映射。

用法（训练时）:
    rendered = apply_lr_params_torch(src_normalized, params_normalized)
    pixel_loss = F.mse_loss(rendered, ref_normalized)

输入约定：
    src_normalized: (B, 3, H, W) 已归一化到 [-1, 1]（CNN 输入空间）
    params_normalized: (B, 22) 已归一化到 [-1, 1]

返回：
    rendered: (B, 3, H, W) [-1, 1] 范围，与 src 同维度
"""

import torch
import torch.nn.functional as F


# 与 dataset/cnn_predictor 保持一致的参数顺序
PARAM_ORDER = (
    'Exposure', 'Highlights', 'Shadows', 'Blacks', 'Whites', 'Contrast',
    'Saturation', 'Vibrance', 'Clarity',
    'SaturationAdjustmentOrange', 'SaturationAdjustmentAqua',
    'SaturationAdjustmentGreen', 'SaturationAdjustmentBlue',
    'HueAdjustmentOrange', 'HueAdjustmentGreen', 'HueAdjustmentAqua',
    'LuminanceAdjustmentOrange', 'LuminanceAdjustmentBlue',
    'SplitToningShadowHue', 'SplitToningShadowSaturation',
    'SplitToningHighlightHue', 'SplitToningHighlightSaturation',
)

# 参数范围（与 param_normalizer.py 一致）
PARAM_RANGES = {
    'Exposure':     (-3.0, 3.0),
    'Highlights':   (-100, 100),
    'Shadows':      (-100, 100),
    'Blacks':       (-100, 100),
    'Whites':       (-100, 100),
    'Contrast':     (-100, 100),
    'Saturation':   (-100, 100),
    'Vibrance':     (-100, 100),
    'Clarity':      (-100, 100),
    'SaturationAdjustmentOrange': (-100, 100),
    'SaturationAdjustmentAqua':   (-100, 100),
    'SaturationAdjustmentGreen':  (-100, 100),
    'SaturationAdjustmentBlue':   (-100, 100),
    'HueAdjustmentOrange': (-100, 100),
    'HueAdjustmentGreen':  (-100, 100),
    'HueAdjustmentAqua':   (-100, 100),
    'LuminanceAdjustmentOrange': (-100, 100),
    'LuminanceAdjustmentBlue':   (-100, 100),
    'SplitToningShadowHue':           (0, 360),
    'SplitToningShadowSaturation':    (0, 100),
    'SplitToningHighlightHue':        (0, 360),
    'SplitToningHighlightSaturation': (0, 100),
}


def denormalize_params(params_norm: torch.Tensor) -> torch.Tensor:
    """
    把归一化的参数 [-1, 1] 还原到原始范围。

    Args:
        params_norm: (B, 22)

    Returns:
        params: (B, 22) 原始范围
    """
    device = params_norm.device
    mids = torch.tensor(
        [(lo + hi) / 2 for lo, hi in (PARAM_RANGES[k] for k in PARAM_ORDER)],
        device=device, dtype=params_norm.dtype,
    )
    spans = torch.tensor(
        [(hi - lo) / 2 for lo, hi in (PARAM_RANGES[k] for k in PARAM_ORDER)],
        device=device, dtype=params_norm.dtype,
    )
    return params_norm * spans + mids  # (B, 22)


def rgb_to_hsv(rgb: torch.Tensor) -> torch.Tensor:
    """
    RGB → HSV（可微）
    输入: (B, 3, H, W), 值 [0, 1]
    输出: (B, 3, H, W), H/S/V 都在 [0, 1]
    """
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    maxc, _ = rgb.max(dim=1, keepdim=True)
    minc, _ = rgb.min(dim=1, keepdim=True)
    v = maxc
    delta = maxc - minc

    s = torch.where(maxc > 1e-8, delta / (maxc + 1e-10), torch.zeros_like(maxc))

    rc = (maxc - r) / (delta + 1e-10)
    gc = (maxc - g) / (delta + 1e-10)
    bc = (maxc - b) / (delta + 1e-10)

    h_r = bc - gc
    h_g = 2.0 + rc - bc
    h_b = 4.0 + gc - rc

    h = torch.where(r >= maxc, h_r,
        torch.where(g >= maxc, h_g, h_b))
    h = (h / 6.0) % 1.0
    h = torch.where(delta < 1e-8, torch.zeros_like(h), h)
    return torch.cat([h, s, v], dim=1)


def hsv_to_rgb(hsv: torch.Tensor) -> torch.Tensor:
    """
    HSV → RGB（可微）
    输入: (B, 3, H, W), H/S/V 都在 [0, 1]
    输出: (B, 3, H, W), 值 [0, 1]
    """
    h, s, v = hsv[:, 0:1], hsv[:, 1:2], hsv[:, 2:3]

    i = (h * 6.0).floor()  # 不可微，但只用于分支选择
    f = h * 6.0 - i

    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)

    i_mod = (i.long() % 6)

    # 用 where 选择对应区间的 RGB
    r = torch.where(i_mod == 0, v,
        torch.where(i_mod == 1, q,
        torch.where(i_mod == 2, p,
        torch.where(i_mod == 3, p,
        torch.where(i_mod == 4, t, v)))))
    g = torch.where(i_mod == 0, t,
        torch.where(i_mod == 1, v,
        torch.where(i_mod == 2, v,
        torch.where(i_mod == 3, q,
        torch.where(i_mod == 4, p, p)))))
    b = torch.where(i_mod == 0, p,
        torch.where(i_mod == 1, p,
        torch.where(i_mod == 2, t,
        torch.where(i_mod == 3, v,
        torch.where(i_mod == 4, v, q)))))
    return torch.cat([r, g, b], dim=1)


def _bcast(scalar: torch.Tensor) -> torch.Tensor:
    """(B,) → (B, 1, 1, 1) 以便与图像广播"""
    return scalar.view(-1, 1, 1, 1)


def apply_lr_params_torch(src: torch.Tensor, params_norm: torch.Tensor) -> torch.Tensor:
    """
    PyTorch 可微版 LR 参数应用。

    Args:
        src: (B, 3, H, W) [0, 1] RGB 图像
        params_norm: (B, 22) 归一化的参数 [-1, 1]

    Returns:
        rendered: (B, 3, H, W) [0, 1] 应用参数后的图像
    """
    # 反归一化到原始参数范围
    params = denormalize_params(params_norm)  # (B, 22)

    # 提取每个参数 (B,)
    def p(name):
        return params[:, PARAM_ORDER.index(name)]

    img = src.clamp(0, 1)

    # 1. 曝光（stops）
    exp = p('Exposure')
    img = img * _bcast(torch.pow(torch.tensor(2.0, device=src.device), exp))
    img = img.clamp(0, 1)

    # 2. 对比度（S 曲线）
    contrast = p('Contrast') / 100.0
    img = 0.5 + (img - 0.5) * _bcast(1.0 + contrast)
    img = img.clamp(0, 1)

    # 3. 高光/阴影/黑/白
    highlights = p('Highlights') / 100.0
    shadows    = p('Shadows') / 100.0
    blacks     = p('Blacks') / 100.0
    whites     = p('Whites') / 100.0

    lum = img.mean(dim=1, keepdim=True)  # (B, 1, H, W)

    # 高光区域
    hi_mask = ((lum - 0.4) / 0.6).clamp(0, 1) ** 2
    img = img + _bcast(highlights * 0.7) * hi_mask
    # 阴影区域
    sh_mask = ((0.6 - lum) / 0.6).clamp(0, 1) ** 2
    img = img + _bcast(shadows * 0.7) * sh_mask
    # 黑色（最暗 30%）
    bk_mask = ((0.3 - lum) / 0.3).clamp(0, 1) ** 2
    img = img + _bcast(blacks * 0.5) * bk_mask
    # 白色（最亮 30%）
    wh_mask = ((lum - 0.7) / 0.3).clamp(0, 1) ** 2
    img = img + _bcast(whites * 0.5) * wh_mask
    img = img.clamp(0, 1)

    # 4. 清晰度（中间调对比）
    clarity = p('Clarity') / 100.0
    mid_mask = 1.0 - (img - 0.5).abs() * 2.0
    img = img + _bcast(clarity * 0.4) * mid_mask * (img - 0.5).sign()
    img = img.clamp(0, 1)

    # 5. 饱和度 + 活力（HSV 空间）
    saturation = p('Saturation') / 100.0
    vibrance   = p('Vibrance') / 100.0

    hsv = rgb_to_hsv(img)
    h_ch, s_ch, v_ch = hsv[:, 0:1], hsv[:, 1:2], hsv[:, 2:3]

    s_ch = s_ch * _bcast(1.0 + saturation)
    weight = (1.0 - s_ch) ** 2
    s_ch = s_ch + _bcast(vibrance * 0.5) * weight
    s_ch = s_ch.clamp(0, 1)
    img = hsv_to_rgb(torch.cat([h_ch, s_ch, v_ch], dim=1))
    img = img.clamp(0, 1)

    # 6. HSL 调整（Orange/Aqua/Green/Blue 各自调 H/S/L）
    color_ranges = {
        'Orange': (0.04, 0.11),
        'Green':  (0.22, 0.42),
        'Aqua':   (0.42, 0.55),
        'Blue':   (0.55, 0.72),
    }
    hsv = rgb_to_hsv(img)
    h_ch, s_ch, v_ch = hsv[:, 0:1], hsv[:, 1:2], hsv[:, 2:3]

    for color, (h_lo, h_hi) in color_ranges.items():
        center = (h_lo + h_hi) / 2
        sigma  = (h_hi - h_lo) / 2
        mask = torch.exp(-((h_ch - center) ** 2) / (2 * sigma ** 2))

        hue_shift = p(f'HueAdjustment{color}') / 100.0 * 0.18 if f'HueAdjustment{color}' in PARAM_ORDER else None
        sat_shift = p(f'SaturationAdjustment{color}') / 100.0 if f'SaturationAdjustment{color}' in PARAM_ORDER else None
        lum_shift = p(f'LuminanceAdjustment{color}') / 100.0 if f'LuminanceAdjustment{color}' in PARAM_ORDER else None

        if hue_shift is not None:
            h_ch = (h_ch + _bcast(hue_shift) * mask) % 1.0
        if sat_shift is not None:
            s_ch = (s_ch + _bcast(sat_shift * 0.9) * mask).clamp(0, 1)
        if lum_shift is not None:
            v_ch = (v_ch + _bcast(lum_shift * 0.6) * mask).clamp(0, 1)

    img = hsv_to_rgb(torch.cat([h_ch, s_ch, v_ch], dim=1))
    img = img.clamp(0, 1)

    # 7. Split Toning（线性化版：tint 用纯色，blend = mask * 0.6 * sat）
    sh_hue = p('SplitToningShadowHue') / 360.0
    sh_sat = (p('SplitToningShadowSaturation') / 100.0).clamp(min=0)
    hi_hue = p('SplitToningHighlightHue') / 360.0
    hi_sat = (p('SplitToningHighlightSaturation') / 100.0).clamp(min=0)

    lum = img.mean(dim=1, keepdim=True)

    # 阴影上色：强度 0.3（v3 调整，原 0.6）
    sh_tint_hsv = torch.stack(
        [sh_hue, torch.ones_like(sh_hue), torch.ones_like(sh_hue)], dim=-1
    ).view(-1, 3, 1, 1)
    sh_tint_rgb = hsv_to_rgb(sh_tint_hsv)
    sh_mask = ((0.5 - lum) * 2.0).clamp(0, 1)
    sh_factor = sh_mask * _bcast(0.3 * sh_sat)
    img = img * (1.0 - sh_factor) + sh_tint_rgb * sh_factor

    # 高光上色（强度 0.3）
    hi_tint_hsv = torch.stack(
        [hi_hue, torch.ones_like(hi_hue), torch.ones_like(hi_hue)], dim=-1
    ).view(-1, 3, 1, 1)
    hi_tint_rgb = hsv_to_rgb(hi_tint_hsv)
    hi_mask = ((lum - 0.5) * 2.0).clamp(0, 1)
    hi_factor = hi_mask * _bcast(0.3 * hi_sat)
    img = img * (1.0 - hi_factor) + hi_tint_rgb * hi_factor

    return img.clamp(0, 1)


if __name__ == '__main__':
    # 测试：与 numpy 版本对比
    import sys
    sys.path.insert(0, '.')
    import numpy as np
    from PIL import Image
    from lr_image_processor import apply_lr_params
    from param_normalizer import ParamNormalizer

    # 构造测试数据
    img = np.array(Image.open('./data/000005_src.jpg').convert('RGB'))
    test_params = {
        'Exposure': 1.5, 'Contrast': 50, 'Highlights': -80, 'Shadows': 60,
        'Blacks': -20, 'Whites': 30, 'Saturation': 30, 'Vibrance': 20,
        'Clarity': 40,
        'SaturationAdjustmentOrange': 50, 'SaturationAdjustmentAqua': -30,
        'SaturationAdjustmentGreen': 0, 'SaturationAdjustmentBlue': 20,
        'HueAdjustmentOrange': 10, 'HueAdjustmentGreen': -15, 'HueAdjustmentAqua': 5,
        'LuminanceAdjustmentOrange': 20, 'LuminanceAdjustmentBlue': -10,
        'SplitToningShadowHue': 220, 'SplitToningShadowSaturation': 15,
        'SplitToningHighlightHue': 38, 'SplitToningHighlightSaturation': 10,
    }

    # numpy 版本结果
    result_np = apply_lr_params(img, test_params)

    # PyTorch 版本结果
    normalizer = ParamNormalizer()
    normalized = normalizer.normalize(test_params)
    params_arr = np.array([normalized.get(k, 0) for k in PARAM_ORDER], dtype=np.float32)

    src_t = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    params_t = torch.from_numpy(params_arr).unsqueeze(0)

    with torch.no_grad():
        result_t = apply_lr_params_torch(src_t, params_t)
    result_t_np = (result_t[0].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)

    # 比较
    diff = np.abs(result_np.astype(np.float32) - result_t_np.astype(np.float32))
    print(f"=== 一致性检查 ===")
    print(f"  numpy 版与 torch 版差异:")
    print(f"    平均像素差: {diff.mean():.2f} / 255")
    print(f"    最大像素差: {diff.max():.2f} / 255")
    print(f"  目标：< 5（小差异由 HSV 转换精度差异造成）")

    # 保存对比
    Image.fromarray(result_np).save('/tmp/numpy_result.jpg')
    Image.fromarray(result_t_np).save('/tmp/torch_result.jpg')
    print(f"\n  已保存对比:\n    /tmp/numpy_result.jpg\n    /tmp/torch_result.jpg")
