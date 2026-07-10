"""
3D LUT 生成器 — 把 CNN 预测的 72 维参数烘焙成 .cube 文件

原理：
  用我们的处理器对一个中性 RGB 网格（identity）应用参数，
  输出网格即为 3D LUT。这样"我们的处理器"就是最终渲染器，
  LUT 到任何软件（LR/PS/DaVinci/剪辑…）都产生一致的颜色变换，
  彻底绕过"处理器 vs 真实 LR"的差异。

仅烘焙颜色/影调（跳过纹理/清晰度/去朦胧等空间操作——LUT 无法表达）。
"""

import numpy as np
from modules.lr_image_processor import apply_lr_params


def bake_cube_lut(params: dict, size: int = 33, title: str = "AI Style LUT") -> str:
    """
    把参数烘焙成 .cube 格式字符串。

    Args:
        params: 72 维 LR 参数字典
        size:   LUT 分辨率（每维格点数，33 是行业标准）
        title:  LUT 标题

    Returns:
        .cube 文件内容（字符串）
    """
    N = size
    # 构造 identity 网格。.cube 顺序：red 变化最快，然后 green，再 blue
    idx = np.arange(N ** 3)
    r = idx % N
    g = (idx // N) % N
    b = idx // (N * N)
    grid = np.stack([r, g, b], axis=1).astype(np.float32) / (N - 1)  # (N^3, 3) ∈ [0,1]

    # reshape 成 2D "图像" 供处理器逐像素处理（颜色操作与空间布局无关）
    img = (grid.reshape(N, N * N, 3) * 255).astype(np.uint8)

    # 只应用颜色/影调（跳过空间操作）
    out = apply_lr_params(img, params, skip_local=True).astype(np.float32) / 255.0
    out_flat = np.clip(out.reshape(N ** 3, 3), 0, 1)

    # 写 .cube
    lines = [
        f'TITLE "{title}"',
        f'LUT_3D_SIZE {N}',
        'DOMAIN_MIN 0.0 0.0 0.0',
        'DOMAIN_MAX 1.0 1.0 1.0',
        '',
    ]
    # 每行一个 RGB 输出（保留 6 位小数）
    lines.extend(f'{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}' for v in out_flat)
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    # 自测：极端参数烘焙 LUT，检查非平凡
    test = {
        'Exposure': 0.8, 'Contrast': 25, 'Shadows': 40, 'Highlights': -40,
        'Saturation': 30, 'Vibrance': 20,
        'HueAdjustmentOrange': 30, 'SaturationAdjustmentBlue': -50,
        'ColorGradeShadowHue': 220, 'ColorGradeShadowSat': 8,
        'ColorGradeHighlightHue': 40, 'ColorGradeHighlightSat': 6,
        'RedHue': 20, 'BlueSaturation': 30,
        'LumaCurve2': 8,
    }
    cube = bake_cube_lut(test, size=17)
    n_data = sum(1 for ln in cube.splitlines() if ln and ln[0].isdigit() or ln.startswith('0.') or ln.startswith('1.'))
    print(f"✓ 生成 .cube，共 {len(cube.splitlines())} 行")
    print("前 8 行:")
    for ln in cube.splitlines()[:8]:
        print("  ", ln)
    # 检查中间点（0.5,0.5,0.5）是否被改变
    print("...")
