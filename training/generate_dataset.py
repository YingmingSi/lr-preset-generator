"""
训练数据生成 v3 — 跳过 darktable，纯 Python 处理

核心改进：
  1. 输入 JPG（不需要 darktable 解码 RAW）
  2. 使用所有源照片（无人为限制）
  3. 课程学习变体策略：
     - 22 个单变量变体（每个只改一个参数）
     - 8 个组合变体（随机 2-5 个参数）
  4. 每张照片 30 变体 = 总计 N_photos × 30 对
"""

import os
import json
import random
import argparse
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from tqdm import tqdm

from lr_image_processor import apply_lr_params

# ─── 参数定义 ─────────────────────────────────────────────────────────────

# 22 个输出参数（与 CNN 输出对应）
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

# 各参数的采样范围（与 param_normalizer.py 保持一致）
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


def sample_strong_value(param: str) -> float:
    """采样一个"强"参数值（避开接近 0，确保视觉效果明显）"""
    lo, hi = PARAM_RANGES[param]

    if lo < 0 < hi:
        # 对称范围：正负方向随机，取 [40%, 100%] 极值
        sign = random.choice([-1, 1])
        magnitude = random.uniform(abs(lo) * 0.4 if sign < 0 else hi * 0.4,
                                    abs(lo) if sign < 0 else hi)
        val = sign * magnitude
    else:
        # 单边范围（如 0-360, 0-100）：取 [30%, 100%]
        span = hi - lo
        val = lo + span * random.uniform(0.3, 1.0)

    return round(val, 2) if param == 'Exposure' else int(round(val))


def sample_random_value(param: str) -> float:
    """采样一个随机参数值（任意大小，包括接近 0）"""
    lo, hi = PARAM_RANGES[param]
    val = random.uniform(lo, hi)
    return round(val, 2) if param == 'Exposure' else int(round(val))


def generate_single_variant(param_idx: int) -> dict:
    """
    生成单变量变体：只改第 param_idx 个参数，其他全为 0。
    """
    params = {p: 0 for p in OUTPUT_PARAMS}
    param = OUTPUT_PARAMS[param_idx]
    params[param] = sample_strong_value(param)
    return params


def generate_combination_variant(n_active: int = None) -> dict:
    """
    生成组合变体：随机 2-5 个参数同时改变。
    """
    if n_active is None:
        n_active = random.randint(2, 5)

    params = {p: 0 for p in OUTPUT_PARAMS}
    active = random.sample(OUTPUT_PARAMS, n_active)
    for param in active:
        params[param] = sample_random_value(param)
    return params


def generate_variants_for_photo(photo_idx: int) -> list:
    """
    为单张照片生成 30 个变体：
      - 22 个单变量（每个参数一个）
      - 8 个组合（随机 2-5 参数）
    """
    variants = []

    # 22 个单变量
    for i in range(len(OUTPUT_PARAMS)):
        variants.append({
            'type': 'single',
            'active_param': OUTPUT_PARAMS[i],
            'params': generate_single_variant(i),
        })

    # 8 个组合
    for _ in range(8):
        variants.append({
            'type': 'combo',
            'active_param': None,
            'params': generate_combination_variant(),
        })

    return variants


def render_one_pair(src_path: str, params: dict, variant_meta: dict,
                    out_dir: str, idx: int,
                    img_size: int = 384) -> dict | None:
    """渲染单个 (src, ref) 对"""
    out_src = os.path.join(out_dir, f'{idx:06d}_src.jpg')
    out_ref = os.path.join(out_dir, f'{idx:06d}_ref.jpg')
    out_par = os.path.join(out_dir, f'{idx:06d}_params.json')

    if os.path.exists(out_par):
        return json.load(open(out_par))

    try:
        # 加载源 JPG 并 resize（一次即可，多变体共用）
        img = Image.open(src_path).convert('RGB')
        img = img.resize((img_size, img_size), Image.BILINEAR)
        src_arr = np.array(img)

        # 保存 src
        Image.fromarray(src_arr).save(out_src, quality=95)

        # 应用参数 → ref
        ref_arr = apply_lr_params(src_arr, params)
        Image.fromarray(ref_arr).save(out_ref, quality=95)

        # 保存元数据（包含原照片路径，用于按照片划分）
        record = {
            'src':            out_src,
            'ref':            out_ref,
            'params':         {k: params.get(k, 0) for k in OUTPUT_PARAMS},
            'idx':            idx,
            'source_photo':   Path(src_path).stem,
            'variant_type':   variant_meta['type'],
            'active_param':   variant_meta.get('active_param'),
        }
        json.dump(record, open(out_par, 'w'))
        return record

    except Exception as e:
        print(f'  [错误] idx={idx}: {e}')
        return None


def generate(src_dir: str, out_dir: str,
             variants_per_photo: int = 30,
             img_size: int = 384,
             n_workers: int = 8):
    """
    主生成函数。

    每张照片 30 变体（22 单变量 + 8 组合）。
    """
    os.makedirs(out_dir, exist_ok=True)

    # 收集源 JPG 文件
    src_files = sorted([
        str(p) for p in Path(src_dir).glob('*')
        if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}
    ])
    if not src_files:
        print(f'❌ 在 {src_dir} 中没有找到 JPG/PNG')
        return

    n_photos = len(src_files)
    total = n_photos * variants_per_photo
    print(f'📷 源照片: {n_photos} 张')
    print(f'📊 每张 {variants_per_photo} 变体（22 单变量 + 8 组合）')
    print(f'📦 总计: {total} 对训练数据\n')

    # 构造所有任务
    tasks = []
    idx = 0
    for photo_path in src_files:
        variants = generate_variants_for_photo(0)  # 22 single + 8 combo
        for v in variants:
            tasks.append((photo_path, v['params'], v, idx))
            idx += 1

    # 并行渲染
    n_done = 0
    n_fail = 0
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(render_one_pair, src, params, v_meta, out_dir, i, img_size): i
            for src, params, v_meta, i in tasks
        }

        for fut in tqdm(as_completed(futures), total=len(futures), desc='渲染中'):
            if fut.result():
                n_done += 1
            else:
                n_fail += 1

    print(f'\n✓ 完成: {n_done} 成功, {n_fail} 失败')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src-dir', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--variants-per-photo', type=int, default=30)
    parser.add_argument('--img-size', type=int, default=384)
    parser.add_argument('--n-workers', type=int, default=8)
    args = parser.parse_args()

    generate(args.src_dir, args.out_dir, args.variants_per_photo,
             args.img_size, args.n_workers)
