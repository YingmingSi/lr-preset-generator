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


def build_deltas(src_rgb, ref_rgb, hue_str=1.0, sat_str=1.0, val_str=0.4):
    """从 src/ref 解析每个 band 的 (色相位移, 饱和比, 亮度位移)。"""
    ss = _band_stats(rgb_to_hsv_vectorized(_to01(src_rgb)))
    rs = _band_stats(rgb_to_hsv_vectorized(_to01(ref_rgb)))
    dhue = np.zeros(_NB, np.float32)
    srat = np.ones(_NB, np.float32)
    dval = np.zeros(_NB, np.float32)
    for b in range(_NB):
        sh, ssat, sval, sw = ss[b]
        rh, rsat, rval, rw = rs[b]
        if sw < _W_MIN or rw < _W_MIN:               # 某色两边不都有 → 不动
            continue
        dhue[b] = (rh - sh) * hue_str
        srat[b] = np.clip(rsat / (ssat + 1e-3), 0.5, 2.0)
        dval[b] = (rval - sval) * val_str
    return dhue, srat, dval


def _apply(rgb01, dhue, srat, dval, sat_str=1.0):
    """把 per-band 变换施加到 RGB（所有位移按像素饱和度加权，护中性）。"""
    hsv = rgb_to_hsv_vectorized(rgb01)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    oH, oS, oV = H.copy(), S.copy(), V.copy()
    for b in range(_NB):
        if dhue[b] == 0 and srat[b] == 1 and dval[b] == 0:
            continue
        c = (b + 0.5) / _NB
        d = np.abs(((H - c + 0.5) % 1.0) - 0.5)
        m = np.clip(1 - d * _NB, 0, 1) * S            # 软属于该 band × 像素饱和度
        oH = (oH + dhue[b] * m) % 1.0
        oS = np.clip(oS * (1 + (srat[b] - 1) * sat_str * m), 0, 1)
        oV = np.clip(oV + dval[b] * m, 0, 1)
    return np.clip(hsv_to_rgb_vectorized(np.stack([oH, oS, oV], axis=-1)), 0, 1)


def bake_hue_lut(src_rgb, ref_rgb, size=33, title="AI Style",
                 hue_str=1.0, sat_str=1.0, val_str=0.4) -> str:
    """按色相外观匹配 → .cube 3D LUT 字符串。"""
    dhue, srat, dval = build_deltas(src_rgb, ref_rgb, hue_str, sat_str, val_str)
    N = size
    idx = np.arange(N ** 3)
    grid = np.stack([idx % N, (idx // N) % N, idx // (N * N)], axis=1).astype(np.float32) / (N - 1)
    out = _apply(grid.reshape(N, N * N, 3), dhue, srat, dval, sat_str).reshape(N ** 3, 3)
    lines = [f'TITLE "{title}"', f'LUT_3D_SIZE {N}',
             'DOMAIN_MIN 0.0 0.0 0.0', 'DOMAIN_MAX 1.0 1.0 1.0', '']
    lines.extend(f'{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}' for v in out)
    return '\n'.join(lines) + '\n', (dhue, srat, dval)


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    s = (rng.random((80, 80, 3)) * 255).astype(np.uint8)
    r = (rng.random((80, 80, 3)) * 120 + 40).astype(np.uint8)
    cube, _ = bake_hue_lut(s, r, size=17)
    n = sum(1 for ln in cube.splitlines() if ln[:1].isdigit())
    print(f'✓ 色相迁移 LUT：{n} 个数据点')
