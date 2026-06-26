"""
训练数据生成 v7 — 72 维参数，纯 Python 处理器

参数定义统一来自 params_config.py（单一数据源）。

变体策略（每张照片 N 个，默认 50）：
  - 单变量变体：只改 1 个参数（学习孤立的因果关系）
    72 个参数按照片索引轮转，确保整个数据集每个参数都有充分的孤立样本
  - 组合变体：随机 2-6 个参数（学习参数交互）
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
from params_config import PARAM_ORDER, PARAM_RANGES, FLOAT_PARAMS

OUTPUT_PARAMS = PARAM_ORDER  # 72 维


def _quantize(val: float, param: str) -> float:
    return round(val, 2) if param in FLOAT_PARAMS else int(round(val))


def sample_strong_value(param: str) -> float:
    """采样一个"强"参数值（避开接近 0，确保视觉效果明显）"""
    lo, hi = PARAM_RANGES[param]
    if lo < 0 < hi:
        sign = random.choice([-1, 1])
        hi_mag = hi if sign > 0 else abs(lo)
        magnitude = random.uniform(hi_mag * 0.4, hi_mag)
        val = sign * magnitude
    else:
        span = hi - lo
        val = lo + span * random.uniform(0.3, 1.0)
    return _quantize(val, param)


def sample_random_value(param: str) -> float:
    """采样一个随机参数值（任意大小，包括接近 0）"""
    lo, hi = PARAM_RANGES[param]
    return _quantize(random.uniform(lo, hi), param)


# 颜色分级修饰符：单独无可见效果，需配合同区 Sat+Hue 才能被模型学习
_CG_ZONES = ['Shadow', 'Midtone', 'Highlight']


def generate_single_variant(param: str) -> dict:
    """
    单变量变体：只改一个参数。

    特殊处理颜色分级修饰符（它们单独不产生可见效果）：
      - ColorGrade*Hue：同时设同区 Sat（让色调可见）
      - ColorGradeBalance/Blending：激活一个随机区的 Hue+Sat（让修饰有作用对象）
    """
    params = {p: 0 for p in OUTPUT_PARAMS}
    params[param] = sample_strong_value(param)

    if param.startswith('ColorGrade') and param.endswith('Hue'):
        zone = param[len('ColorGrade'):-len('Hue')]
        params[f'ColorGrade{zone}Sat'] = sample_strong_value(f'ColorGrade{zone}Sat')
    elif param in ('ColorGradeBalance', 'ColorGradeBlending'):
        zone = random.choice(_CG_ZONES)
        params[f'ColorGrade{zone}Hue'] = sample_random_value(f'ColorGrade{zone}Hue')
        params[f'ColorGrade{zone}Sat'] = sample_strong_value(f'ColorGrade{zone}Sat')

    return params


def generate_combination_variant() -> dict:
    """组合变体：随机 2-6 个参数同时改变"""
    params = {p: 0 for p in OUTPUT_PARAMS}
    for param in random.sample(OUTPUT_PARAMS, random.randint(2, 6)):
        params[param] = sample_random_value(param)
    return params


def generate_variants_for_photo(photo_idx: int, n_variants: int,
                                 n_single: int) -> list:
    """
    为单张照片生成 n_variants 个变体。

    - 前 n_single 个：单变量，参数按 photo_idx 轮转覆盖全部 72 个
    - 其余：组合变体
    """
    variants = []
    n_params = len(OUTPUT_PARAMS)
    for i in range(n_single):
        param = OUTPUT_PARAMS[(photo_idx * n_single + i) % n_params]
        variants.append({
            'type': 'single', 'active_param': param,
            'params': generate_single_variant(param),
        })
    for _ in range(n_variants - n_single):
        variants.append({
            'type': 'combo', 'active_param': None,
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
             variants_per_photo: int = 50,
             n_single: int = 35,
             img_size: int = 256,
             n_workers: int = 8):
    """
    主生成函数。

    每张照片 variants_per_photo 个变体（n_single 单变量 + 其余组合）。
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
    print(f'📊 每张 {variants_per_photo} 变体（{n_single} 单变量 + '
          f'{variants_per_photo - n_single} 组合）')
    print(f'🎯 参数维度: {len(OUTPUT_PARAMS)}')
    print(f'📦 总计: {total} 对训练数据\n')

    # 构造所有任务
    tasks = []
    idx = 0
    for photo_idx, photo_path in enumerate(src_files):
        variants = generate_variants_for_photo(photo_idx, variants_per_photo, n_single)
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
    parser.add_argument('--variants-per-photo', type=int, default=50)
    parser.add_argument('--n-single', type=int, default=35)
    parser.add_argument('--img-size', type=int, default=256)
    parser.add_argument('--n-workers', type=int, default=8)
    args = parser.parse_args()

    generate(args.src_dir, args.out_dir, args.variants_per_photo,
             args.n_single, args.img_size, args.n_workers)
