"""
训练数据生成脚本

使用 darktable-cli 将源图 + 随机 XMP 参数渲染为参考图，
生成 (src, ref, params) 三元组作为 CNN 训练数据。

运行前提：
  1. 安装 darktable:  sudo apt install darktable
  2. 准备源图目录:    任意 JPG/RAW 照片
  3. 运行:           python generate_dataset.py --src-dir ./photos --out-dir ./data

生成速度约 1-3 张/分钟（取决于图片大小和机器配置）。
1000 对数据约需 3-10 小时，10 万对建议用多台机器并行。
"""

import os
import json
import random
import argparse
import subprocess
import tempfile
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ─── 参数采样分布 ─────────────────────────────────────────────────────────────
# 从 calibration.json 加载后，这些范围会被用户真实预设覆盖
# 默认范围覆盖常见调色操作
PARAM_DISTRIBUTIONS = {
    'Exposure':    {'type': 'uniform', 'lo': -1.5,  'hi': 1.5},
    'Highlights':  {'type': 'uniform', 'lo': -100,  'hi': 10},
    'Shadows':     {'type': 'uniform', 'lo': -30,   'hi': 80},
    'Blacks':      {'type': 'uniform', 'lo': -60,   'hi': 30},
    'Whites':      {'type': 'uniform', 'lo': -50,   'hi': 40},
    'Contrast':    {'type': 'uniform', 'lo': -50,   'hi': 70},
    'Saturation':  {'type': 'uniform', 'lo': -30,   'hi': 30},
    'Vibrance':    {'type': 'uniform', 'lo': -20,   'hi': 40},
    'Clarity':     {'type': 'uniform', 'lo': -20,   'hi': 40},
    # HSL 饱和度（注意：很多真实预设以负值为主）
    'SaturationAdjustmentRed':    {'type': 'skew_neg', 'lo': -60, 'hi': 40},
    'SaturationAdjustmentOrange': {'type': 'skew_neg', 'lo': -60, 'hi': 50},
    'SaturationAdjustmentYellow': {'type': 'skew_neg', 'lo': -60, 'hi': 40},
    'SaturationAdjustmentGreen':  {'type': 'skew_neg', 'lo': -90, 'hi': 30},
    'SaturationAdjustmentAqua':   {'type': 'skew_neg', 'lo': -60, 'hi': 40},
    'SaturationAdjustmentBlue':   {'type': 'skew_neg', 'lo': -70, 'hi': 40},
    'SaturationAdjustmentPurple': {'type': 'skew_neg', 'lo': -60, 'hi': 30},
    'SaturationAdjustmentMagenta':{'type': 'skew_neg', 'lo': -60, 'hi': 30},
    # HSL 色相
    'HueAdjustmentOrange': {'type': 'uniform', 'lo': -30, 'hi': 30},
    'HueAdjustmentYellow': {'type': 'uniform', 'lo': -40, 'hi': 20},
    'HueAdjustmentGreen':  {'type': 'uniform', 'lo': -40, 'hi': 20},
    'HueAdjustmentAqua':   {'type': 'uniform', 'lo': -40, 'hi': 20},
    'HueAdjustmentBlue':   {'type': 'uniform', 'lo': -20, 'hi': 20},
    # HSL 亮度
    'LuminanceAdjustmentOrange': {'type': 'uniform', 'lo': -30, 'hi': 20},
    'LuminanceAdjustmentGreen':  {'type': 'uniform', 'lo': -40, 'hi': 20},
    'LuminanceAdjustmentBlue':   {'type': 'uniform', 'lo': -30, 'hi': 20},
    # SplitToning
    'SplitToningShadowHue':           {'type': 'circular_cluster',
                                        'clusters': [210, 38, 0],  # 蓝/橙/无色
                                        'spread': 30},
    'SplitToningShadowSaturation':    {'type': 'zero_heavy', 'lo': 0, 'hi': 30},
    'SplitToningHighlightHue':        {'type': 'circular_cluster',
                                        'clusters': [38, 210, 0],
                                        'spread': 25},
    'SplitToningHighlightSaturation': {'type': 'zero_heavy', 'lo': 0, 'hi': 25},
}

# 最终 CNN 预测的参数列表（与模型输出维度对应）
OUTPUT_PARAMS = [
    'Exposure', 'Highlights', 'Shadows', 'Blacks', 'Whites', 'Contrast',
    'Saturation', 'Vibrance', 'Clarity',
    'SaturationAdjustmentOrange', 'SaturationAdjustmentAqua',
    'SaturationAdjustmentGreen',  'SaturationAdjustmentBlue',
    'HueAdjustmentOrange', 'HueAdjustmentGreen', 'HueAdjustmentAqua',
    'LuminanceAdjustmentOrange', 'LuminanceAdjustmentBlue',
    'SplitToningShadowHue', 'SplitToningShadowSaturation',
    'SplitToningHighlightHue', 'SplitToningHighlightSaturation',
]


def sample_params(calibration: dict = None) -> dict:
    """
    采样一组随机 XMP 参数。
    如果提供了 calibration.json 的内容，则按用户真实分布采样
    （这样模型学到的是"用户风格"的参数范围）。
    """
    params = {}
    for name, dist in PARAM_DISTRIBUTIONS.items():
        if calibration and name in calibration:
            # 用用户校准范围覆盖默认分布
            cal = calibration[name]
            lo, hi = cal['lo'], cal['hi']
            # 在 P5-P95 范围内均匀采样，并允许 30% 超出（覆盖更多情况）
            span  = hi - lo
            lo_ext, hi_ext = lo - span * 0.3, hi + span * 0.3
            val = random.uniform(lo_ext, hi_ext)
        else:
            lo, hi = dist['lo'], dist['hi']
            if dist['type'] == 'uniform':
                val = random.uniform(lo, hi)
            elif dist['type'] == 'skew_neg':
                # 偏负值（真实预设的典型分布）
                val = random.triangular(lo, hi, lo + (hi - lo) * 0.2)
            elif dist['type'] == 'zero_heavy':
                # 70% 概率为 0（许多预设不调这个参数）
                val = 0 if random.random() < 0.7 else random.uniform(lo, hi)
            elif dist['type'] == 'circular_cluster':
                # SplitToning Hue：集中在几个常用角度附近
                cluster = random.choice(dist['clusters'])
                val = (cluster + random.gauss(0, dist['spread'])) % 360
            else:
                val = random.uniform(lo, hi)

        # 整数化
        if name == 'Exposure':
            params[name] = round(val, 2)
        else:
            params[name] = int(round(val))

    return params


def params_to_lr_xmp(params: dict, preset_name: str = 'train') -> str:
    """生成 LR 格式 XMP（darktable-cli 能读取 crs: 命名空间）"""
    lines = [
        '<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>',
        '<x:xmpmeta xmlns:x="adobe:ns:meta/">',
        '  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">',
        '    <rdf:Description rdf:about=""',
        '      xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"',
        f'      crs:PresetName="{preset_name}"',
        '      crs:ProcessVersion="11.0"',
        '      crs:WhiteBalance="As Shot"',
    ]
    for k, v in params.items():
        key = k + ('2012' if k in ('Exposure', 'Contrast', 'Highlights',
                                    'Shadows', 'Whites', 'Blacks') else '')
        lines.append(f'      crs:{key}="{v}"')
    lines += ['    />', '  </rdf:RDF>', '</x:xmpmeta>', '<?xpacket end="w"?>']
    return '\n'.join(lines)


def render_pair(src_path: str, params: dict,
                out_dir: str, idx: int,
                img_size: int = 384) -> dict | None:
    """
    渲染一对 (src, ref)，返回保存路径和参数字典。
    """
    out_src = os.path.join(out_dir, f'{idx:06d}_src.jpg')
    out_ref = os.path.join(out_dir, f'{idx:06d}_ref.jpg')
    out_par = os.path.join(out_dir, f'{idx:06d}_params.json')

    # 如果已存在，跳过（支持断点续传）
    if os.path.exists(out_par):
        return json.load(open(out_par))

    # 写临时 XMP
    xmp_content = params_to_lr_xmp(params)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xmp', delete=False) as f:
        xmp_path = f.name
        f.write(xmp_content)

    try:
        # 渲染 ref（src + XMP 参数）
        result = subprocess.run(
            ['darktable-cli', src_path, xmp_path, out_ref,
             '--width', str(img_size), '--height', str(img_size),
             '--hq', '0', '--apply-custom-presets', '0'],
            capture_output=True, timeout=60
        )
        if result.returncode != 0:
            return None

        # 复制/缩放 src（无 XMP 的版本）
        subprocess.run(
            ['darktable-cli', src_path, out_src,
             '--width', str(img_size), '--height', str(img_size),
             '--hq', '0', '--apply-custom-presets', '0'],
            capture_output=True, timeout=60
        )

        # 提取 OUTPUT_PARAMS 子集
        output = {k: params.get(k, 0) for k in OUTPUT_PARAMS}
        record = {
            'src': out_src, 'ref': out_ref,
            'params': output, 'idx': idx,
        }
        json.dump(record, open(out_par, 'w'))
        return record

    except Exception as e:
        print(f'  [错误] idx={idx}: {e}')
        return None
    finally:
        if os.path.exists(xmp_path):
            os.remove(xmp_path)


def generate(src_dir: str, out_dir: str,
             n_pairs: int = 10000,
             img_size: int = 384,
             n_workers: int = 4,
             calib_path: str = None):
    """
    主生成函数。

    Args:
        src_dir:    源图目录（JPG/RAW）
        out_dir:    输出目录
        n_pairs:    生成对数
        img_size:   输出图像大小（正方形）
        n_workers:  并行 darktable 进程数
        calib_path: calibration.json 路径（按用户分布采样）
    """
    os.makedirs(out_dir, exist_ok=True)

    # 加载校准数据
    calibration = None
    if calib_path and os.path.exists(calib_path):
        calibration = json.load(open(calib_path))
        print(f'使用用户校准分布: {calib_path}')

    # 收集源图
    exts = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.raw', '.cr2',
            '.nef', '.arw', '.dng', '.raf'}
    src_files = [str(p) for p in Path(src_dir).rglob('*')
                 if p.suffix.lower() in exts]
    if not src_files:
        print(f'在 {src_dir} 中没有找到图像文件')
        return

    print(f'找到 {len(src_files)} 张源图，目标生成 {n_pairs} 对...')

    # 构造任务列表
    tasks = []
    for i in range(n_pairs):
        src = src_files[i % len(src_files)]
        params = sample_params(calibration)
        tasks.append((src, params, i))

    # 并行渲染
    done = 0
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(render_pair, src, params, out_dir, idx, img_size): idx
            for src, params, idx in tasks
        }
        for fut in as_completed(futures):
            result = fut.result()
            done += 1
            if done % 100 == 0:
                print(f'  进度: {done}/{n_pairs}  ({done/n_pairs*100:.1f}%)')

    print(f'完成！共生成 {done} 对数据 → {out_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='生成 CNN 训练数据')
    parser.add_argument('--src-dir',   required=True,  help='源图目录')
    parser.add_argument('--out-dir',   required=True,  help='输出目录')
    parser.add_argument('--n-pairs',   type=int, default=10000)
    parser.add_argument('--img-size',  type=int, default=384)
    parser.add_argument('--n-workers', type=int, default=4)
    parser.add_argument('--calib',     default=None,   help='calibration.json 路径')
    args = parser.parse_args()

    generate(args.src_dir, args.out_dir, args.n_pairs,
             args.img_size, args.n_workers, args.calib)
