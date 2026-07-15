"""
解析色彩/影调匹配 — 直接从 src/ref 两张图解析真实的影调+色彩差异，烘进 3D LUT。

动机：CNN 预测系统性偏保守（回归均值），需拉到 ~150% 强度才接近参考图。
但推理时我们手上就有 src 和 ref，可直接解析二者差异，无需依赖偏弱的 CNN：

  · 影调（L 通道）：直方图/分位数匹配 src→ref，精确对齐亮度分布（曝光/对比/影调形态）
  · 色彩（a/b 通道）：均值+方差匹配（Reinhard），对齐白平衡与饱和度

在 LAB 空间完成（纯 numpy 实现，无额外依赖）。最终与 CNN 的 LR 风格 LUT 混合。
"""

import numpy as np
from modules.lr_image_processor import apply_lr_params


# ─── sRGB ↔ LAB（D65，纯 numpy）───────────────────────────────────────────

_M_RGB2XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
], dtype=np.float64)
_M_XYZ2RGB = np.linalg.inv(_M_RGB2XYZ)
_WHITE = np.array([0.95047, 1.0, 1.08883])  # D65
_EPS = 216 / 24389        # (6/29)^3
_KAPPA = 24389 / 27       # (29/3)^3


def _srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * (c ** (1 / 2.4)) - 0.055)


def rgb_to_lab(rgb):
    """rgb: (...,3) in [0,1] → lab: L∈[0,100], a/b∈~[-128,127]"""
    lin = _srgb_to_linear(rgb.astype(np.float64))
    xyz = lin @ _M_RGB2XYZ.T
    xyz = xyz / _WHITE
    f = np.where(xyz > _EPS, np.cbrt(xyz), (_KAPPA * xyz + 16) / 116)
    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return np.stack([L, a, b], axis=-1)


def lab_to_rgb(lab):
    """lab → rgb in [0,1]"""
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16) / 116
    fx = fy + a / 500
    fz = fy - b / 200
    fx3, fy3, fz3 = fx ** 3, fy ** 3, fz ** 3
    xr = np.where(fx3 > _EPS, fx3, (116 * fx - 16) / _KAPPA)
    yr = np.where(L > _KAPPA * _EPS, fy3, L / _KAPPA)
    zr = np.where(fz3 > _EPS, fz3, (116 * fz - 16) / _KAPPA)
    xyz = np.stack([xr, yr, zr], axis=-1) * _WHITE
    lin = xyz @ _M_XYZ2RGB.T
    return np.clip(_linear_to_srgb(lin), 0, 1)


# ─── 解析匹配 ─────────────────────────────────────────────────────────────

def _to01(img):
    img = np.asarray(img)
    return img.astype(np.float64) / 255.0 if img.dtype == np.uint8 or img.max() > 1.5 else img.astype(np.float64)


def compute_match(src_rgb, ref_rgb, color_strength: float = 0.85):
    """
    从 src/ref 解析 LAB 匹配变换，返回一个作用于任意 RGB 网格的函数。

    color_strength: a/b 色彩迁移强度（<1 抑制主体色溢出，1=完全对齐参考图均值）
    """
    src_lab = rgb_to_lab(_to01(src_rgb)).reshape(-1, 3)
    ref_lab = rgb_to_lab(_to01(ref_rgb)).reshape(-1, 3)

    # 影调：L 通道分位数匹配（256 个分位点，稳健且单调）
    q = np.linspace(0, 1, 256)
    src_Lq = np.quantile(src_lab[:, 0], q)
    ref_Lq = np.quantile(ref_lab[:, 0], q)

    # 色彩：a/b 均值+方差匹配（Reinhard），方差比裁剪防过冲
    src_m, ref_m = src_lab.mean(0), ref_lab.mean(0)
    src_s, ref_s = src_lab.std(0) + 1e-6, ref_lab.std(0) + 1e-6
    ab_scale = np.clip(ref_s[1:] / src_s[1:], 0.5, 2.0)
    # 目标均值按 color_strength 从 src 均值向 ref 均值插值（抑制溢色）
    ab_target_m = src_m[1:] + color_strength * (ref_m[1:] - src_m[1:])

    def transform(rgb01):
        lab = rgb_to_lab(rgb01)
        L = np.interp(lab[..., 0], src_Lq, ref_Lq)
        a = (lab[..., 1] - src_m[1]) * ab_scale[0] + ab_target_m[0]
        b = (lab[..., 2] - src_m[2]) * ab_scale[1] + ab_target_m[1]
        return lab_to_rgb(np.stack([L, a, b], axis=-1))

    return transform


def bake_match_lut(src_rgb, ref_rgb, cnn_params: dict, size: int = 33,
                   match_weight: float = 0.7, color_strength: float = 0.85,
                   title: str = "AI Style LUT") -> str:
    """
    烘焙「解析匹配 ⊕ CNN 风格」混合 3D LUT。

    match_weight: 解析匹配占比（0=纯 CNN，1=纯解析匹配）。默认 0.7 以还原为主。
    """
    N = size
    idx = np.arange(N ** 3)
    r = idx % N
    g = (idx // N) % N
    b = idx // (N * N)
    grid01 = np.stack([r, g, b], axis=1).astype(np.float64) / (N - 1)  # (N^3,3)

    # 解析匹配分支
    A = compute_match(src_rgb, ref_rgb, color_strength)(grid01)        # (N^3,3) [0,1]

    # CNN 风格分支（逐像素颜色操作，跳过空间操作）
    grid_img = (grid01.reshape(N, N * N, 3) * 255).astype(np.uint8)
    C = apply_lr_params(grid_img, cnn_params, skip_local=True).astype(np.float64) / 255.0
    C = C.reshape(N ** 3, 3)

    out = np.clip(match_weight * A + (1 - match_weight) * C, 0, 1)

    lines = [
        f'TITLE "{title}"',
        f'LUT_3D_SIZE {N}',
        'DOMAIN_MIN 0.0 0.0 0.0',
        'DOMAIN_MAX 1.0 1.0 1.0',
        '',
    ]
    lines.extend(f'{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}' for v in out)
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    # 自测：LAB 往返精度 + 匹配 LUT 非平凡
    rng = np.random.default_rng(0)
    rgb = rng.random((1000, 3))
    back = lab_to_rgb(rgb_to_lab(rgb))
    print(f"LAB 往返最大误差: {np.abs(rgb - back).max():.2e}")  # 应 ~1e-6

    src = (rng.random((200, 200, 3)) * 180 + 20).astype(np.uint8)          # 偏暗
    ref = (rng.random((200, 200, 3)) * 100 + 140).clip(0, 255).astype(np.uint8)  # 偏亮
    cube = bake_match_lut(src, ref, {'Exposure': 0.2}, size=17)
    n = sum(1 for ln in cube.splitlines() if ln[:1].isdigit())
    print(f"✓ 混合 LUT 生成，{n} 个数据点")
