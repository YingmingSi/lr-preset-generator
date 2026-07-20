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


# 非均匀色相 band：(中心, 半宽, 最大偏移) 单位=色相比例(×360=度)
# 暖色(红橙黄绿)细分、小幅调整、防塌；青蓝紫合并成一个【宽 band】、允许大幅靠拢(蓝→青)——
# 一张图通常只出现青/蓝/紫之一，合并不会互相吃，且用均值平移保留其间对比。
_BANDS = [
    (0.00, 0.055, 0.05),   # 红
    (0.09, 0.05,  0.05),   # 橙
    (0.16, 0.05,  0.05),   # 黄
    (0.25, 0.06,  0.06),   # 黄绿
    (0.35, 0.09,  0.06),   # 绿
    (0.65, 0.22,  0.15),   # 青蓝紫（宽，最大偏移 ~54°）
    (0.90, 0.06,  0.06),   # 品红
]


def _band_offsets(hsv):
    """每个 band 的 加权(饱和度) 平均色相偏移(相对 band 中心) 与 权重。"""
    H = hsv[..., 0].ravel(); S = hsv[..., 1].ravel()
    n = len(_BANDS)
    offs = np.zeros(n); ws = np.zeros(n)
    for bi, (c, hw, _) in enumerate(_BANDS):
        d = np.minimum(np.abs(H - c), 1 - np.abs(H - c))
        m = np.clip(1 - d / hw, 0, 1) * S
        wsum = float(m.sum())
        if wsum < _W_MIN:
            continue
        hoff = (((H - c + 0.5) % 1.0) - 0.5)
        offs[bi] = float((hoff * m).sum() / wsum); ws[bi] = wsum
    return offs, ws


def build_hue_map(src_hsv, ref_hsv):
    """双图 per-band 色相靠拢：源每个 band 的平均色相 → 向参考同 band 靠拢（均值平移，
    保留 band 内色相对比），偏移限在各 band 的 max_shift 内；参考缺该 band 则不动(不吃色)。
    青蓝紫为宽 band，允许蓝大幅靠向青。"""
    so, sw = _band_offsets(src_hsv)
    ro, rw = _band_offsets(ref_hsv)
    dhue_band = np.zeros(len(_BANDS))
    for bi, (c, hw, ms) in enumerate(_BANDS):
        if sw[bi] < _W_MIN or rw[bi] < _W_MIN:
            continue
        dhue_band[bi] = np.clip(ro[bi] - so[bi], -ms, ms)

    hue_j = (np.arange(_HN) + 0.5) / _HN
    centers = np.array([c for c, _, _ in _BANDS])
    hws = np.array([hw for _, hw, _ in _BANDS])
    hue_map = hue_j.copy()
    for i in range(_HN):
        d = np.minimum(np.abs(centers - hue_j[i]), 1 - np.abs(centers - hue_j[i]))
        m = np.clip(1 - d / hws, 0, 1)                # 到各 band 的软隶属（各自半宽）
        tot = m.sum()
        if tot > 1e-6:
            hue_map[i] = (hue_j[i] + (m * dhue_band).sum() / tot) % 1.0
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
    # 过曝/死黑保护：接近极值时减弱该方向的推动（亮部别再提亮、暗部别再压暗、别过饱和）
    def sstep(a, b, x):
        t = np.clip((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t)
    gV = np.where(dV > 0, 1 - sstep(0.82, 1.0, V), sstep(0.0, 0.18, V))
    gS = np.where(dS > 0, 1 - sstep(0.85, 1.0, S), 1.0)
    oS = np.clip(S + sat_str * dS * gS, 0, 1)
    oV = np.clip(V + val_str * dV * gV, 0, 1)
    return np.clip(hsv_to_rgb_vectorized(np.stack([oH, oS, oV], axis=-1)), 0, 1)


_LUMA3 = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _zone_weights(L):
    """按亮度分 阴影/中间调/高光 三区的软权重（归一化）。"""
    sh = np.clip(1 - L / 0.5, 0, 1)          # 1@黑 → 0@中
    hi = np.clip((L - 0.5) / 0.5, 0, 1)      # 0@中 → 1@白
    mid = np.clip(1 - np.abs(L - 0.5) * 2, 0, 1)
    s = sh + mid + hi + 1e-9
    return sh / s, mid / s, hi / s


def _grade_shifts(src01, ref01):
    """颜色分级：参考 vs 原图 在 阴影/中间/高光 的色度均值之差。
    偏重【近中性】像素——色调分离的真实色偏体现在灰面上，而非饱和内容色；
    这样估出的才是'grade'本身，避免内容色造成全图色偏。"""
    def zone_chroma(img):
        L = img @ _LUMA3
        chroma = (img - L[..., None]).reshape(-1, 3)
        cmag = np.sqrt((chroma ** 2).sum(1))              # 每像素色度大小 ≈ 饱和度
        neutral = np.clip((0.16 - cmag) / (0.16 - 0.04), 0, 1)  # 只取近中性灰面(饱和内容排除)
        sh, mid, hi = (w.reshape(-1) * neutral for w in _zone_weights(L))
        return tuple((chroma * w[:, None]).sum(0) / (w.sum() + 1e-9) for w in (sh, mid, hi))
    s_sh, s_mid, s_hi = zone_chroma(src01)
    r_sh, r_mid, r_hi = zone_chroma(ref01)
    grade = np.stack([r_sh - s_sh, r_mid - s_mid, r_hi - s_hi])   # (3,3)
    # 阴影/高光色相分离：削弱两者共有的色偏（同向=假分离/全局色偏），让二者更互补
    shared = 0.5 * (grade[0] + grade[2])
    grade[0] -= 0.45 * shared
    grade[2] -= 0.45 * shared
    return grade


def bake_hue_lut(src_rgb, ref_rgb, size=33, title="AI Style",
                 hue_str=1.0, sat_str=0.85, val_str=0.5, grade_str=1.0) -> str:
    """双图 per-band 色相有界靠拢 + 饱和/亮度按每色相均值平移 +
    颜色分级(阴影/中间/高光三区色度匹配参考，保亮度) → .cube 3D LUT。"""
    src01, ref01 = _to01(src_rgb), _to01(ref_rgb)
    src_hsv = rgb_to_hsv_vectorized(src01)
    ref_hsv = rgb_to_hsv_vectorized(ref01)
    _, s_sat, s_val = _ref_maps(src_hsv)                # 原图每色相均值
    dens, r_sat, r_val = _ref_maps(ref_hsv)
    hue_map = build_hue_map(src_hsv, ref_hsv)
    gsh = _grade_shifts(src01, ref01)                  # (3,3) 三区色度差
    N = size
    idx = np.arange(N ** 3)
    grid = np.stack([idx % N, (idx // N) % N, idx // (N * N)], axis=1).astype(np.float32) / (N - 1)
    out = _apply(grid.reshape(N, N * N, 3), hue_map, dens, s_sat, s_val, r_sat, r_val,
                 hue_str, sat_str, val_str).reshape(N ** 3, 3)
    # 颜色分级：按每个输出色的亮度，叠加三区色度差（保亮度）
    if grade_str > 0:
        Lg = out @ _LUMA3
        sw, mw, hw = _zone_weights(Lg)
        grade = sw[:, None] * gsh[0] + mw[:, None] * gsh[1] + hw[:, None] * gsh[2]
        # 极值保护：近纯黑/纯白处减弱，防死黑被抬、纯白染色
        def sstep(a, b, x):
            t = np.clip((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t)
        ext = (sstep(0.0, 0.06, Lg) * (1 - sstep(0.94, 1.0, Lg)))[:, None]
        out = np.clip(out + grade_str * grade * ext, 0, 1)
    lines = [f'TITLE "{title}"', f'LUT_3D_SIZE {N}',
             'DOMAIN_MIN 0.0 0.0 0.0', 'DOMAIN_MAX 1.0 1.0 1.0', '']
    lines.extend(f'{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}' for v in out)
    # 摘要：每 band 的色相位移
    band_dhue = np.zeros(len(_BANDS), np.float32)
    for bi, (c, _, _) in enumerate(_BANDS):
        t = hue_map[int(c * _HN) % _HN]
        band_dhue[bi] = (((t - c + 0.5) % 1.0) - 0.5)
    return '\n'.join(lines) + '\n', (band_dhue, None, None)


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    s = (rng.random((80, 80, 3)) * 255).astype(np.uint8)
    r = (rng.random((80, 80, 3)) * 120 + 40).astype(np.uint8)
    cube, _ = bake_hue_lut(s, r, size=17)
    n = sum(1 for ln in cube.splitlines() if ln[:1].isdigit())
    print(f'✓ 色相迁移 LUT：{n} 个数据点')
