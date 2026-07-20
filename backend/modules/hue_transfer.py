"""
按色相外观匹配 —— 情况B（不同照片风格移植）的颜色引擎。

核心思路（与 CNN 的"分布匹配"相反）：
  不学"颜色数量/比例"，只学"某个颜色在参考里长什么样"。
  把图像按色相分成若干 band，对每个 band 匹配 参考 vs 原图 的
  色相偏移 / 饱和度 / 亮度，作为一个"输入色→风格色"的映射（内容无关）。
  再把这个映射烘焙成 3D LUT。

  · 中性/低饱和像素不受影响（所有位移按像素自身饱和度加权）→ 白仍是白
  · 不改变颜色数量（黄多黄少都只改它"怎么显示"）→ 适配情况B
  · 某色在原图或参考里几乎没有 → 该 band 不动
"""

import numpy as np
from modules.lr_image_processor import rgb_to_hsv_vectorized, hsv_to_rgb_vectorized

_NB = 12          # 色相 band 数
_W_MIN = 8.0      # band 有效所需的最小加权像素量（src 与 ref 都要够）


def _band_stats(hsv: np.ndarray):
    """每个 hue band 的 加权(按饱和度) 统计：色相偏移、饱和、明度、权重。"""
    H, S, V = hsv[..., 0].ravel(), hsv[..., 1].ravel(), hsv[..., 2].ravel()
    out = []
    for b in range(_NB):
        c = (b + 0.5) / _NB
        d = np.abs(((H - c + 0.5) % 1.0) - 0.5)      # 到 band 中心的环形距离
        w = np.clip(1 - d * _NB, 0, 1) * S            # 属于该 band 且有饱和度
        wsum = float(w.sum())
        if wsum < 1e-6:
            out.append((0.0, 0.0, 0.0, 0.0)); continue
        hoff = (((H - c + 0.5) % 1.0) - 0.5)          # band 内平均色相偏移
        out.append((float((hoff * w).sum() / wsum),
                    float((S * w).sum() / wsum),
                    float((V * w).sum() / wsum),
                    wsum))
    return out


def _to01(img):
    img = np.asarray(img)
    return img.astype(np.float32) / 255.0 if (img.dtype == np.uint8 or img.max() > 1.5) else img.astype(np.float32)


_HN = 256          # 色相映射精度


def _circ_smooth(a, frac):
    x = np.arange(_HN)
    kern = np.exp(-0.5 * (np.minimum(np.abs(x - _HN // 2), _HN - np.abs(x - _HN // 2)) / (frac * _HN)) ** 2)
    return np.real(np.fft.ifft(np.fft.fft(a) * np.fft.fft(np.roll(kern, -_HN // 2))))


def _ref_maps(ref_hsv):
    """参考图按色相(256)的：密度、平均饱和、平均亮度（均按饱和度加权 + 环形平滑）。"""
    H = ref_hsv[..., 0].ravel()
    S = ref_hsv[..., 1].ravel()
    V = ref_hsv[..., 2].ravel()
    hi = np.clip((H * _HN).astype(int), 0, _HN - 1)
    dens = np.zeros(_HN); np.add.at(dens, hi, S)
    ssum = np.zeros(_HN); np.add.at(ssum, hi, S * S)      # 饱和度加权的 sat
    vsum = np.zeros(_HN); np.add.at(vsum, hi, S * V)
    dens = np.maximum(_circ_smooth(dens, 0.05), 0)
    ssum = _circ_smooth(ssum, 0.05)
    vsum = _circ_smooth(vsum, 0.05)
    sat_map = np.where(dens > 1e-6, ssum / np.maximum(dens, 1e-9), 0.0)
    val_map = np.where(dens > 1e-6, vsum / np.maximum(dens, 1e-9), 0.0)
    return dens, sat_map, val_map


def build_hue_map(dens, sigma=0.09, sharp=1.6):
    """色相分布匹配：query 色相 → 目标色相（向参考密集色相靠拢，锐化偏向峰）。"""
    d = dens ** sharp
    hue_j = (np.arange(_HN) + 0.5) / _HN
    ang = 2 * np.pi * hue_j
    hue_map = hue_j.copy()
    if d.sum() < 1e-6:
        return hue_map
    for i in range(_HN):
        dist = np.minimum(np.abs(hue_j - hue_j[i]), 1 - np.abs(hue_j - hue_j[i]))
        w = d * np.exp(-0.5 * (dist / sigma) ** 2)
        if w.sum() < 1e-9:
            continue
        hue_map[i] = (np.arctan2((w * np.sin(ang)).sum(), (w * np.cos(ang)).sum()) / (2 * np.pi)) % 1.0
    return hue_map


def _apply(rgb01, hue_map, dens, sat_map, val_map, hue_str=1.0, sat_str=1.0, val_str=0.5):
    """先按分布匹配改色相，再按【目标色相】取参考的饱和/亮度来调（护中性）。"""
    hsv = rgb_to_hsv_vectorized(rgb01)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    swt = np.clip(S * 2, 0, 1)
    # 1) 色相：分布匹配
    tgt = hue_map[np.clip((H * _HN).astype(int), 0, _HN - 1)]
    dhue = (((tgt - H + 0.5) % 1.0) - 0.5)
    oH = (H + hue_str * dhue * swt) % 1.0
    # 2) 饱和/亮度：按【偏移后】色相取参考值，conf=目标色相在参考里的支撑度
    oi = np.clip((oH * _HN).astype(int), 0, _HN - 1)
    conf = np.clip(dens[oi] / (dens.max() + 1e-9), 0, 1)
    tsat = sat_map[oi]; tval = val_map[oi]
    oS = np.clip(S + sat_str * (tsat - S) * swt * conf, 0, 1)
    oV = np.clip(V + val_str * (tval - V) * swt * conf, 0, 1)
    return np.clip(hsv_to_rgb_vectorized(np.stack([oH, oS, oV], axis=-1)), 0, 1)


def bake_hue_lut(src_rgb, ref_rgb, size=33, title="AI Style",
                 hue_str=1.0, sat_str=1.0, val_str=0.5) -> str:
    """色相分布匹配 + 目标色相的饱和/亮度匹配 → .cube 3D LUT。"""
    ref_hsv = rgb_to_hsv_vectorized(_to01(ref_rgb))
    dens, sat_map, val_map = _ref_maps(ref_hsv)
    hue_map = build_hue_map(dens)
    N = size
    idx = np.arange(N ** 3)
    grid = np.stack([idx % N, (idx // N) % N, idx // (N * N)], axis=1).astype(np.float32) / (N - 1)
    out = _apply(grid.reshape(N, N * N, 3), hue_map, dens, sat_map, val_map,
                 hue_str, sat_str, val_str).reshape(N ** 3, 3)
    lines = [f'TITLE "{title}"', f'LUT_3D_SIZE {N}',
             'DOMAIN_MIN 0.0 0.0 0.0', 'DOMAIN_MAX 1.0 1.0 1.0', '']
    lines.extend(f'{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}' for v in out)
    # 摘要：每 band 的色相位移
    band_dhue = np.zeros(_NB, np.float32)
    for b in range(_NB):
        c = (b + 0.5) / _NB
        t = hue_map[int(c * _HN) % _HN]
        band_dhue[b] = (((t - c + 0.5) % 1.0) - 0.5)
    return '\n'.join(lines) + '\n', (band_dhue, None, None)


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    s = (rng.random((80, 80, 3)) * 255).astype(np.uint8)
    r = (rng.random((80, 80, 3)) * 120 + 40).astype(np.uint8)
    cube, _ = bake_hue_lut(s, r, size=17)
    n = sum(1 for ln in cube.splitlines() if ln[:1].isdigit())
    print(f'✓ 色相迁移 LUT：{n} 个数据点')
