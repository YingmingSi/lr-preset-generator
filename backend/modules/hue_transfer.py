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
    """参考图按色相(256)的：密度(轻平滑,判'有没有该色')、平均饱和、平均亮度。"""
    H = ref_hsv[..., 0].ravel()
    S = ref_hsv[..., 1].ravel()
    V = ref_hsv[..., 2].ravel()
    hi = np.clip((H * _HN).astype(int), 0, _HN - 1)
    dens = np.zeros(_HN); np.add.at(dens, hi, S)
    ssum = np.zeros(_HN); np.add.at(ssum, hi, S * S)
    vsum = np.zeros(_HN); np.add.at(vsum, hi, S * V)
    dens_light = np.maximum(_circ_smooth(dens, 0.02), 0)  # 轻平滑：真实有像素才算"有"
    dens_heavy = np.maximum(_circ_smooth(dens, 0.05), 0)  # 较宽：稳定 sat/val 归一化
    ssum = _circ_smooth(ssum, 0.05)
    vsum = _circ_smooth(vsum, 0.05)
    sat_map = np.where(dens_heavy > 1e-6, ssum / np.maximum(dens_heavy, 1e-9), 0.0)
    val_map = np.where(dens_heavy > 1e-6, vsum / np.maximum(dens_heavy, 1e-9), 0.0)
    return dens_light, sat_map, val_map


def build_hue_map(dens, thr=0.08, rng=0.16):
    """色相映射：参考里"有"的色相 → 保持原样（identity，不塌不并）；
    参考里"没有"的色相 → 吸附到最近的 present 色相。软阈值平滑过渡。
    （关键：present 色相绝不被大峰吞并——避免橙/绿都塌成脏黄的 bug）"""
    hue_j = (np.arange(_HN) + 0.5) / _HN
    dmax = float(dens.max())
    if dmax < 1e-9:
        return hue_j.copy()

    def cdist(a, b):
        d = np.abs(a - b); return np.minimum(d, 1 - d)

    # 参考的色相"峰"（局部极大 + 够高）
    is_peak = (dens >= np.roll(dens, 1)) & (dens >= np.roll(dens, -1)) & (dens > thr * dmax)
    pk = np.where(is_peak)[0]
    if pk.size == 0:
        pk = np.array([int(np.argmax(dens))])
    ph, pd = hue_j[pk], dens[pk]                     # 峰的色相 / 密度

    hue_map = hue_j.copy()
    for i in range(_HN):
        if is_peak[i]:                       # 独立色峰 → 保持（各自单独调整，不塌不并）
            continue
        score = pd * np.exp(-0.5 * (cdist(ph, hue_j[i]) / rng) ** 2)  # 又强又近的峰
        j = int(np.argmax(score))
        weight = np.clip(1 - dens[i] / (pd[j] + 1e-9), 0, 1)         # 仅 谷/缺失色 → 吸附最近强峰
        d = (((ph[j] - hue_j[i] + 0.5) % 1.0) - 0.5)
        hue_map[i] = (hue_j[i] + weight * d) % 1.0
    return hue_map


def _apply(rgb01, hue_map, dens, s_sat, s_val, r_sat, r_val,
           hue_str=1.0, sat_str=1.0, val_str=0.5):
    """色相分布匹配；饱和/亮度按【参考均值 − 原图均值】整体平移
    （保留原图同色相内的明暗/饱和对比，不抹平），护中性。"""
    hsv = rgb_to_hsv_vectorized(rgb01)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    # 只护"近中性"(S<~0.14)，淡色(如天空 S~0.2)也充分调整——
    # 色相在低饱和像素上本就看不见，削弱它会让淡蓝够不到青、饱和也提不起来
    t = np.clip((S - 0.03) / (0.14 - 0.03), 0, 1)
    swt = t * t * (3 - 2 * t)
    hi = np.clip((H * _HN).astype(int), 0, _HN - 1)
    # 1) 色相：分布匹配
    dhue = (((hue_map[hi] - H + 0.5) % 1.0) - 0.5)
    oH = (H + hue_str * dhue * swt) % 1.0
    # 2) 饱和/亮度：整体平移 = 参考(目标色相均值) − 原图(原色相均值)，保留个体偏差
    oi = np.clip((oH * _HN).astype(int), 0, _HN - 1)
    conf = np.clip(dens[oi] / (dens.max() + 1e-9), 0, 1)
    dS = (r_sat[oi] - s_sat[hi]) * swt * conf
    dV = (r_val[oi] - s_val[hi]) * swt * conf
    oS = np.clip(S + sat_str * dS, 0, 1)
    oV = np.clip(V + val_str * dV, 0, 1)
    return np.clip(hsv_to_rgb_vectorized(np.stack([oH, oS, oV], axis=-1)), 0, 1)


def bake_hue_lut(src_rgb, ref_rgb, size=33, title="AI Style",
                 hue_str=1.0, sat_str=1.0, val_str=1.0) -> str:
    """色相匹配（独立色各自保持）+ 饱和/亮度按均值平移（保对比）→ .cube 3D LUT。"""
    _, s_sat, s_val = _ref_maps(rgb_to_hsv_vectorized(_to01(src_rgb)))   # 原图每色相均值
    dens, r_sat, r_val = _ref_maps(rgb_to_hsv_vectorized(_to01(ref_rgb)))
    hue_map = build_hue_map(dens)
    N = size
    idx = np.arange(N ** 3)
    grid = np.stack([idx % N, (idx // N) % N, idx // (N * N)], axis=1).astype(np.float32) / (N - 1)
    out = _apply(grid.reshape(N, N * N, 3), hue_map, dens, s_sat, s_val, r_sat, r_val,
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
