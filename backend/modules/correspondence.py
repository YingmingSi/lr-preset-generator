"""
空间对应色彩迁移 —— 情况A（参考图=原图调色后，同一张图）的精确引擎。

原图与参考图逐像素对应，直接学"原图这个颜色 → 参考里变成什么颜色"的映射，
散点 trilinear 拟合成 3D LUT。能精确捕捉任意色相旋转/饱和/亮度变化（不保守）。
对稀疏格点向 identity 正则化，避免噪声。

`is_aligned` 判别两图是否同一内容（结构对齐）→ 决定用本引擎还是色相匹配。
"""

import numpy as np
from PIL import Image


def is_aligned(src_rgb: np.ndarray, ref_rgb: np.ndarray, thresh: float = 0.45) -> bool:
    """两图是否同一内容（情况A）：比较边缘/梯度结构的相关性（对调色不敏感）。"""
    def edges(a):
        g = np.asarray(Image.fromarray(a).convert('L').resize((128, 128)), np.float32)
        gy, gx = np.gradient(g)
        m = np.hypot(gx, gy)
        return (m - m.mean()) / (m.std() + 1e-6)
    e1, e2 = edges(src_rgb), edges(ref_rgb)
    return float((e1 * e2).mean()) >= thresh


def bake_correspondence_lut(src_rgb, ref_rgb, size: int = 33, reg: float = 6.0,
                            title: str = "AI Style") -> str:
    """从逐像素对应 (src→ref) 拟合 3D LUT，输出 .cube 字符串。"""
    N = size
    s = (np.asarray(src_rgb, np.float32) / 255.0).reshape(-1, 3)
    r = (np.asarray(ref_rgb, np.float32) / 255.0).reshape(-1, 3)
    acc = np.zeros((N, N, N, 3), np.float64)
    wsum = np.zeros((N, N, N), np.float64)
    f = s * (N - 1)
    i0 = np.clip(np.floor(f).astype(int), 0, N - 2)
    fr = f - i0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = ((fr[:, 0] if dx else 1 - fr[:, 0]) *
                     (fr[:, 1] if dy else 1 - fr[:, 1]) *
                     (fr[:, 2] if dz else 1 - fr[:, 2]))
                np.add.at(acc, (i0[:, 0] + dx, i0[:, 1] + dy, i0[:, 2] + dz), r * w[:, None])
                np.add.at(wsum, (i0[:, 0] + dx, i0[:, 1] + dy, i0[:, 2] + dz), w)
    rr, gg, bb = np.meshgrid(np.arange(N), np.arange(N), np.arange(N), indexing='ij')
    idg = np.stack([rr, gg, bb], -1).astype(np.float64) / (N - 1)   # identity（[r,g,b] 索引）
    ws = wsum[..., None]
    fitted = np.where(ws > 0, acc / np.maximum(ws, 1e-9), idg)
    conf = ws / (ws + reg)                                          # 稀疏格点向 identity 正则
    grid = np.clip(idg * (1 - conf) + fitted * conf, 0, 1)          # (N,N,N,3) 索引[r,g,b]

    lines = [f'TITLE "{title}"', f'LUT_3D_SIZE {N}',
             'DOMAIN_MIN 0.0 0.0 0.0', 'DOMAIN_MAX 1.0 1.0 1.0', '']
    # .cube 顺序：red 最快
    idx = np.arange(N ** 3)
    rc, gc, bc = idx % N, (idx // N) % N, idx // (N * N)
    flat = grid[rc, gc, bc]
    lines.extend(f'{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}' for v in flat)
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    a = (rng.random((100, 100, 3)) * 255).astype(np.uint8)
    b = np.clip(a * 0.8 + 20, 0, 255).astype(np.uint8)             # 同内容、调色
    print('is_aligned(同图):', is_aligned(a, b))
    c = (rng.random((100, 100, 3)) * 255).astype(np.uint8)
    print('is_aligned(不同图):', is_aligned(a, c))
    cube = bake_correspondence_lut(a, b, size=17)
    print('LUT 数据点:', sum(1 for ln in cube.splitlines() if ln[:1].isdigit()))
