"""
darktable-cli 驱动的迭代优化模块

流程：
  1. rough_analysis()  → 初始参数猜测（毫秒级，减少后续迭代次数）
  2. optimize()        → CMA-ES 无导数优化（30 次 darktable 渲染，约 30-60 秒）
  3. 输出 darktable XMP sidecar 或 LR XMP

darktable 参数空间（仅优化最关键的 12 个）：
  Exposure, Contrast, Highlights, Shadows, Blacks, Whites,
  Saturation, Vibrance,
  ShadowColor (hue, sat), HighlightColor (hue, sat)
"""

import os
import json
import math
import shutil
import tempfile
import subprocess
import numpy as np
from typing import Optional

# ─── darktable 参数定义 ──────────────────────────────────────────────────────
# 每个参数: (dt_module, dt_param, lr_param, scale, offset, min, max)
# scale/offset: lr_value = dt_value * scale + offset
DT_PARAM_MAP = {
    'Exposure':   ('exposure', 'exposure',    1.0,   0.0, -5.0,  5.0),
    'Contrast':   ('brightness-contrast', 'contrast', 0.01, 0.0, -1.0, 1.0),
    'Highlights': ('highlights', 'highlights', 0.01,  0.0, -1.0, 0.0),
    'Shadows':    ('shadows-highlights', 'shadows',   0.01, 0.0,  0.0, 1.0),
    'Blacks':     ('levels', 'black',     1.0,   0.0,  0.0, 0.5),
    'Saturation': ('colorbalancergb', 'global-saturation', 0.01, 1.0, 0.0, 2.0),
}

# 优化的参数向量（子集，减少搜索空间）
OPTIM_PARAMS = [
    # (lr_param_name, init_scale, search_range)
    ('Exposure',          0.0,  (-2.0,  2.0)),
    ('Highlights',        0.0,  (-100,   10)),
    ('Shadows',           0.0,  (-30,    80)),
    ('Blacks',            0.0,  (-60,    40)),
    ('Whites',            0.0,  (-60,    40)),
    ('Contrast',          0.0,  (-60,    80)),
    ('Saturation',        0.0,  (-50,    50)),
    ('SplitToningShadowHue',        210, (0, 360)),
    ('SplitToningShadowSaturation',   8, (0,  15)),
    ('SplitToningHighlightHue',      38, (0, 360)),
    ('SplitToningHighlightSaturation', 6, (0,  15)),
    ('SaturationAdjustmentOrange',    0, (-50,  50)),
    ('SaturationAdjustmentAqua',      0, (-50,  50)),
    ('SaturationAdjustmentGreen',     0, (-80,  30)),
]

PARAM_NAMES  = [p[0] for p in OPTIM_PARAMS]
PARAM_INITS  = [p[1] for p in OPTIM_PARAMS]
PARAM_BOUNDS = [p[2] for p in OPTIM_PARAMS]


# ─── darktable XMP sidecar 生成 ─────────────────────────────────────────────

def params_to_dt_xmp(params: dict, src_path: str) -> str:
    """
    将 LR 风格参数转为 darktable sidecar XMP 内容。
    注：这是一个近似映射，darktable 的参数名和含义与 LR 有差异。
    实际部署时需要根据 darktable 版本精调映射关系。
    """
    exp   = float(params.get('Exposure',   0))
    hl    = float(params.get('Highlights', 0)) / 100.0
    sh    = float(params.get('Shadows',    0)) / 100.0
    bk    = float(params.get('Blacks',     0)) / 100.0
    ct    = float(params.get('Contrast',   0)) / 100.0
    sat   = 1.0 + float(params.get('Saturation', 0)) / 100.0

    sh_hue = float(params.get('SplitToningShadowHue',         0))
    sh_sat = float(params.get('SplitToningShadowSaturation',  0)) / 100.0
    hl_hue = float(params.get('SplitToningHighlightHue',      0))
    hl_sat = float(params.get('SplitToningHighlightSaturation', 0)) / 100.0

    # darktable XMP sidecar 骨架（process_version="4" = darktable 4.x）
    xmp = f"""<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description rdf:about=""
      xmlns:darktable="http://darktable.sf.net/image/raw/"
      xmlns:xmp="http://ns.adobe.com/xap/1.0/"
      darktable:import_timestamp="-1"
      darktable:change_timestamp="-1"
      darktable:export_timestamp="-1"
      darktable:print_timestamp="-1"
      darktable:xmp_version="4"
      darktable:raw_params="0"
      darktable:auto_presets_applied="0"
      darktable:iop_order_version="2"
      darktable:history_end="-1">
      <darktable:history>
        <rdf:Seq>
          <rdf:li darktable:operation="exposure"
            darktable:enabled="1"
            darktable:modversion="7"
            darktable:params="{json.dumps({'exposure': exp, 'black': max(0.0, -bk), 'deflicker_percentile': 50.0, 'mode': 0})}"
            darktable:blendop_version="11"
            darktable:blendop_params="gz12eJxjYGBg"/>
          <rdf:li darktable:operation="highlights"
            darktable:enabled="1"
            darktable:modversion="3"
            darktable:params="{json.dumps({'mode': 0, 'blendif': 0, 'radius': 0.1, 'strength': max(0, -hl)})}"
            darktable:blendop_version="11"
            darktable:blendop_params="gz12eJxjYGBg"/>
          <rdf:li darktable:operation="shadhi"
            darktable:enabled="1"
            darktable:modversion="7"
            darktable:params="{json.dumps({'shadows': sh * 100, 'highlights': 0, 'whitepoint': max(0, -bk * 0.5) * 100})}"
            darktable:blendop_version="11"
            darktable:blendop_params="gz12eJxjYGBg"/>
          <rdf:li darktable:operation="colorbalancergb"
            darktable:enabled="1"
            darktable:modversion="7"
            darktable:params="{json.dumps({
                'global_saturation': sat,
                'shadows_H': sh_hue / 360.0,
                'shadows_C': sh_sat,
                'highlights_H': hl_hue / 360.0,
                'highlights_C': hl_sat,
                'midtones_H': 0.0,
                'midtones_C': 0.0,
                'global_H': 0.0,
                'global_C': 0.0,
                'contrast': ct,
            })}"
            darktable:blendop_version="11"
            darktable:blendop_params="gz12eJxjYGBg"/>
        </rdf:Seq>
      </darktable:history>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>"""
    return xmp


# ─── darktable-cli 调用 ──────────────────────────────────────────────────────

def is_darktable_available() -> bool:
    return shutil.which('darktable-cli') is not None


def render_with_darktable(src_path: str, xmp_path: str,
                           width: int = 512) -> Optional[np.ndarray]:
    """
    调用 darktable-cli 渲染，返回 RGB float 数组。
    失败时返回 None。
    """
    if not is_darktable_available():
        return None

    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        out_path = f.name

    try:
        result = subprocess.run(
            ['darktable-cli', src_path, xmp_path, out_path,
             '--width', str(width), '--height', str(width),
             '--hq', '0'],
            capture_output=True, timeout=30
        )
        if result.returncode != 0:
            return None

        import imageio
        img = imageio.imread(out_path).astype(np.float32) / 255.0
        return img

    except Exception:
        return None
    finally:
        if os.path.exists(out_path):
            os.remove(out_path)


# ─── 损失函数 ────────────────────────────────────────────────────────────────

def image_loss(rendered: np.ndarray, ref: np.ndarray) -> float:
    """
    多尺度图像损失：
    - L2 像素差（主要）
    - 直方图差（颜色分布）
    """
    # 缩放到相同大小
    h = min(rendered.shape[0], ref.shape[0])
    w = min(rendered.shape[1], ref.shape[1])
    r = rendered[:h, :w]
    t = ref[:h, :w]

    pixel_loss = float(np.mean((r - t) ** 2))

    # 各通道直方图差
    hist_loss = 0.0
    for c in range(3):
        hr, _ = np.histogram(r[:, :, c], bins=64, range=(0, 1))
        ht, _ = np.histogram(t[:, :, c], bins=64, range=(0, 1))
        hist_loss += float(np.mean((hr / hr.sum() - ht / ht.sum()) ** 2))

    return pixel_loss * 0.7 + hist_loss * 0.3


# ─── 主优化接口 ──────────────────────────────────────────────────────────────

def optimize(src_path: str,
             ref_rgb: np.ndarray,
             initial_params: dict,
             n_iter: int = 30,
             n_workers: int = 4,
             progress_cb=None) -> dict:
    """
    用 Nelder-Mead 无导数优化在 darktable 参数空间中搜索最优 XMP。

    Args:
        src_path:       原图路径（darktable-cli 需要文件路径）
        ref_rgb:        参考图 RGB float 数组
        initial_params: 初始 LR 参数猜测（来自图像分析）
        n_iter:         最大函数评估次数（每次 ~1-2 秒）
        n_workers:      并行 darktable 实例数
        progress_cb:    进度回调 (iteration, loss) → None

    Returns:
        优化后的参数字典（LR 格式键名）
    """
    if not is_darktable_available():
        return initial_params   # darktable 未安装，返回分析结果

    # 初始向量（从初始参数提取）
    x0 = np.array([
        float(initial_params.get(name, init))
        for name, init, _ in OPTIM_PARAMS
    ], dtype=np.float64)

    # 参数归一化到 [-1, 1]（方便优化器）
    lo = np.array([b[0] for b in PARAM_BOUNDS], dtype=np.float64)
    hi = np.array([b[1] for b in PARAM_BOUNDS], dtype=np.float64)

    def normalize(x):
        return 2.0 * (x - lo) / (hi - lo + 1e-8) - 1.0

    def denormalize(z):
        return lo + (z + 1.0) * 0.5 * (hi - lo)

    z0 = normalize(x0)
    call_count = [0]
    best_loss  = [float('inf')]

    def objective(z):
        call_count[0] += 1
        params = dict(zip(PARAM_NAMES, denormalize(z)))
        params = {k: (round(v, 2) if k == 'Exposure' else int(round(v)))
                  for k, v in params.items()}

        # 写临时 XMP
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xmp',
                                          delete=False) as f:
            xmp_path = f.name
            f.write(params_to_dt_xmp(params, src_path))

        try:
            rendered = render_with_darktable(src_path, xmp_path)
            if rendered is None:
                return 1e6
            loss = image_loss(rendered, ref_rgb)
            if loss < best_loss[0]:
                best_loss[0] = loss
            if progress_cb:
                progress_cb(call_count[0], loss)
            return loss
        finally:
            if os.path.exists(xmp_path):
                os.remove(xmp_path)

    from scipy.optimize import minimize
    result = minimize(
        objective, z0,
        method='Nelder-Mead',
        options={
            'maxiter':  n_iter,
            'xatol':    0.02,
            'fatol':    1e-4,
            'adaptive': True,
        }
    )

    best_z = result.x
    best_params = dict(zip(PARAM_NAMES, denormalize(best_z)))
    best_params = {k: (round(v, 2) if k == 'Exposure' else int(round(v)))
                   for k, v in best_params.items()}

    # 合并回初始参数（其余不优化的参数保持原值）
    merged = dict(initial_params)
    merged.update(best_params)
    return merged
