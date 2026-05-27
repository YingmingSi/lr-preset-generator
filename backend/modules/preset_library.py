"""
预设风格模板库
职责：
  1. 解析 XMP 文件 → 提取数值参数
  2. 内置 6 个经典风格模板
  3. 从 data/presets/ 目录自动加载用户上传的 XMP 文件
  4. 将图像分析结果匹配到最近风格（余弦相似度）
  5. 将分析结果与风格模板混合，提供方向性先验
"""

import re
import json
import os
import numpy as np
from typing import Optional

# ─── 常量 ────────────────────────────────────────────────────────────────────

BUCKETS   = ['Red', 'Orange', 'Yellow', 'Green', 'Aqua', 'Blue', 'Purple', 'Magenta']
SAT_KEYS  = [f'SaturationAdjustment{b}' for b in BUCKETS]
HUE_KEYS  = [f'HueAdjustment{b}' for b in BUCKETS]
LUM_KEYS  = [f'LuminanceAdjustment{b}' for b in BUCKETS]

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'presets')

# ─── XMP 解析 ─────────────────────────────────────────────────────────────────

def parse_xmp_params(xmp_content: str) -> dict:
    """
    从 XMP 文本提取数值参数。
    处理 crs:Key="val" 和 crs:Key2012="val" 两种格式。
    """
    params = {}
    for m in re.finditer(r'crs:(\w+?)(?:2012)?="(-?[\d.]+)"', xmp_content):
        key, val_str = m.group(1), m.group(2)
        try:
            params[key] = float(val_str) if '.' in val_str else int(val_str)
        except ValueError:
            pass
    return params


def auto_name_preset(params: dict, filename: str = '') -> str:
    """
    根据 HSL 饱和度特征自动推断风格名称。
    优先从文件名提取中文/英文关键词。
    """
    name_hints = {
        '青橙': 'teal_orange', 'tealoran': 'teal_orange', 'teal': 'teal_orange',
        '黑金': 'black_gold',  'blackgold': 'black_gold',
        '日系': 'japanese',    'japan': 'japanese', 'vsco': 'japanese', '小清新': 'japanese',
        '胶片': 'film_vintage', 'film': 'film_vintage', '复古': 'film_vintage', 'vintage': 'film_vintage',
        '暗调': 'moody_dark',  'moody': 'moody_dark', 'dark': 'moody_dark',
        '自然': 'fresh_nature', 'nature': 'fresh_nature', 'green': 'fresh_nature',
    }
    low = filename.lower().replace(' ', '').replace('_', '').replace('-', '')
    for kw, style in name_hints.items():
        if kw in low:
            return style

    # 按 HSL 特征推断
    orange = params.get('SaturationAdjustmentOrange', 0)
    aqua   = params.get('SaturationAdjustmentAqua',   0)
    green  = params.get('SaturationAdjustmentGreen',  0)
    blue   = params.get('SaturationAdjustmentBlue',   0)
    sat_mean = np.mean([params.get(k, 0) for k in SAT_KEYS])

    if orange > 40 and aqua > 30:
        return 'teal_orange'
    if orange > 40 and green < -50 and blue < -50:
        return 'black_gold'
    if sat_mean < -10:
        return 'japanese'
    if green > 30 and blue > 20:
        return 'fresh_nature'
    if params.get('Blacks', 0) < -30 and params.get('Contrast', 0) > 40:
        return 'moody_dark'
    return 'film_vintage'


# ─── 内置经典风格模板 ─────────────────────────────────────────────────────────
# 每个模板描述"当该风格被应用到中性源图时"的典型 HSL/亮度参数值
# 注意：这里不是"与中性图的差值"，而是"最终预设应有的参数"

BUILTIN_STYLES: dict = {
    'teal_orange': {
        '_name': '青橙',
        '_desc': '暖色偏橙，冷色偏青，橙青饱和大幅提高，绿/紫/品红全压低',
        'SaturationAdjustmentRed':    -25,
        'SaturationAdjustmentOrange':  85,
        'SaturationAdjustmentYellow': -15,
        'SaturationAdjustmentGreen':  -55,
        'SaturationAdjustmentAqua':    78,
        'SaturationAdjustmentBlue':   -25,
        'SaturationAdjustmentPurple': -40,
        'SaturationAdjustmentMagenta':-30,
        'HueAdjustmentRed':    -8,
        'HueAdjustmentOrange': 10,
        'HueAdjustmentYellow':  5,
        'HueAdjustmentGreen':  15,
        'HueAdjustmentAqua':  -10,
        'HueAdjustmentBlue':  -18,
        'HueAdjustmentPurple':  0,
        'HueAdjustmentMagenta': 0,
        'LuminanceAdjustmentOrange':  8,
        'LuminanceAdjustmentAqua':   -8,
        'Shadows': 10, 'Highlights': -20,
    },
    'black_gold': {
        '_name': '黑金',
        '_desc': '深黑高对比，仅橙金色彩保留，其余几乎完全去饱和',
        'SaturationAdjustmentRed':    -65,
        'SaturationAdjustmentOrange':  70,
        'SaturationAdjustmentYellow':  55,
        'SaturationAdjustmentGreen':  -95,
        'SaturationAdjustmentAqua':   -95,
        'SaturationAdjustmentBlue':   -95,
        'SaturationAdjustmentPurple': -95,
        'SaturationAdjustmentMagenta':-75,
        'HueAdjustmentRed':    15,
        'HueAdjustmentOrange':  8,
        'HueAdjustmentYellow': -8,
        'HueAdjustmentGreen':   0,
        'HueAdjustmentAqua':    0,
        'HueAdjustmentBlue':    0,
        'HueAdjustmentPurple':  0,
        'HueAdjustmentMagenta': 0,
        'LuminanceAdjustmentOrange': 18,
        'LuminanceAdjustmentYellow': 22,
        'LuminanceAdjustmentGreen': -20,
        'LuminanceAdjustmentBlue':  -25,
        'Blacks': -50, 'Shadows': -30, 'Highlights': -20, 'Contrast': 65, 'Saturation': -25,
    },
    'japanese': {
        '_name': '日系小清新',
        '_desc': '低对比提阴影，整体偏亮偏粉，绿偏黄，整体饱和度低',
        'SaturationAdjustmentRed':     12,
        'SaturationAdjustmentOrange':  20,
        'SaturationAdjustmentYellow':  -5,
        'SaturationAdjustmentGreen':  -38,
        'SaturationAdjustmentAqua':   -28,
        'SaturationAdjustmentBlue':   -42,
        'SaturationAdjustmentPurple': -22,
        'SaturationAdjustmentMagenta': 10,
        'HueAdjustmentRed':    5,
        'HueAdjustmentOrange': 5,
        'HueAdjustmentYellow': 5,
        'HueAdjustmentGreen':  28,
        'HueAdjustmentAqua':   18,
        'HueAdjustmentBlue':   18,
        'HueAdjustmentPurple': 10,
        'HueAdjustmentMagenta': 5,
        'LuminanceAdjustmentRed':    12,
        'LuminanceAdjustmentOrange':  8,
        'LuminanceAdjustmentGreen': -10,
        'LuminanceAdjustmentBlue':  -12,
        'Shadows': 55, 'Blacks': 28, 'Highlights': -18, 'Whites': 12,
        'Contrast': -38, 'Saturation': -18, 'Vibrance': 12,
    },
    'film_vintage': {
        '_name': '胶片复古',
        '_desc': '提黑褪色感，暖色调，蓝绿压低，Red/Orange偏暖',
        'SaturationAdjustmentRed':    22,
        'SaturationAdjustmentOrange': 28,
        'SaturationAdjustmentYellow':  8,
        'SaturationAdjustmentGreen':  -32,
        'SaturationAdjustmentAqua':   -38,
        'SaturationAdjustmentBlue':   -42,
        'SaturationAdjustmentPurple': -28,
        'SaturationAdjustmentMagenta':  8,
        'HueAdjustmentRed':    10,
        'HueAdjustmentOrange':  8,
        'HueAdjustmentYellow':  5,
        'HueAdjustmentGreen':  22,
        'HueAdjustmentAqua':   15,
        'HueAdjustmentBlue':   18,
        'HueAdjustmentPurple':  5,
        'HueAdjustmentMagenta': 0,
        'LuminanceAdjustmentRed':     8,
        'LuminanceAdjustmentOrange': 12,
        'LuminanceAdjustmentBlue':  -10,
        'Blacks': 22, 'Shadows': 18, 'Highlights': -12,
        'Contrast': -18, 'Saturation': -10, 'Vibrance': 10,
    },
    'moody_dark': {
        '_name': '暗调浓郁',
        '_desc': '高对比深黑，压高光，蓝绿偏冷，饱和度选择性',
        'SaturationAdjustmentRed':    28,
        'SaturationAdjustmentOrange': 38,
        'SaturationAdjustmentYellow': 12,
        'SaturationAdjustmentGreen':  -22,
        'SaturationAdjustmentAqua':   -18,
        'SaturationAdjustmentBlue':    28,
        'SaturationAdjustmentPurple':  18,
        'SaturationAdjustmentMagenta':-18,
        'HueAdjustmentRed':    0,
        'HueAdjustmentOrange': 0,
        'HueAdjustmentYellow': 0,
        'HueAdjustmentGreen':  0,
        'HueAdjustmentAqua':  -12,
        'HueAdjustmentBlue':  -15,
        'HueAdjustmentPurple': 0,
        'HueAdjustmentMagenta':0,
        'LuminanceAdjustmentBlue':   -18,
        'LuminanceAdjustmentGreen':  -12,
        'LuminanceAdjustmentOrange':   5,
        'Blacks': -48, 'Shadows': -28, 'Highlights': -32,
        'Contrast': 58, 'Vibrance': 18,
    },
    'fresh_nature': {
        '_name': '清新自然',
        '_desc': '户外自然感，绿色鲜亮，天蓝清透，整体明亮',
        'SaturationAdjustmentRed':     5,
        'SaturationAdjustmentOrange':  18,
        'SaturationAdjustmentYellow':  28,
        'SaturationAdjustmentGreen':   58,
        'SaturationAdjustmentAqua':    38,
        'SaturationAdjustmentBlue':    38,
        'SaturationAdjustmentPurple': -18,
        'SaturationAdjustmentMagenta':-12,
        'HueAdjustmentRed':    0,
        'HueAdjustmentOrange': 0,
        'HueAdjustmentYellow': -5,
        'HueAdjustmentGreen': -15,
        'HueAdjustmentAqua':   -5,
        'HueAdjustmentBlue':    8,
        'HueAdjustmentPurple': 0,
        'HueAdjustmentMagenta':0,
        'LuminanceAdjustmentGreen': 12,
        'LuminanceAdjustmentAqua':   8,
        'LuminanceAdjustmentBlue':   5,
        'Shadows': 18, 'Highlights': -18,
        'Vibrance': 28, 'Saturation': 15,
    },
}

# 运行时用户模板（从磁盘加载 + 本次 session 上传的）
_user_styles: dict = {}


# ─── 持久化 ───────────────────────────────────────────────────────────────────

def _user_styles_path() -> str:
    return os.path.join(_DATA_DIR, '_user_styles.json')


def load_user_styles() -> None:
    """启动时从 data/presets/ 目录加载用户预设（JSON + XMP 文件）"""
    global _user_styles
    # 加载上次 session 保存的 JSON
    json_path = _user_styles_path()
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            _user_styles.update(json.load(f))

    # 自动扫描目录下的 .xmp 文件（新增或更新）
    if os.path.isdir(_DATA_DIR):
        for fname in os.listdir(_DATA_DIR):
            if not fname.lower().endswith('.xmp'):
                continue
            style_id = fname[:-4]
            if style_id in _user_styles:
                continue  # 已加载，跳过
            fpath = os.path.join(_DATA_DIR, fname)
            with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                params = parse_xmp_params(f.read())
            style_name = auto_name_preset(params, fname)
            _user_styles[style_id] = {**params, '_name': fname, '_source': style_name}


def save_user_styles() -> None:
    os.makedirs(_DATA_DIR, exist_ok=True)
    with open(_user_styles_path(), 'w', encoding='utf-8') as f:
        json.dump(_user_styles, f, ensure_ascii=False, indent=2)


def add_user_preset(xmp_content: str, filename: str) -> dict:
    """解析并添加用户上传的 XMP 预设，返回摘要"""
    params     = parse_xmp_params(xmp_content)
    style_id   = os.path.splitext(filename)[0].replace(' ', '_')
    style_cat  = auto_name_preset(params, filename)
    _user_styles[style_id] = {**params, '_name': filename, '_source': style_cat}
    save_user_styles()
    return {
        'id':       style_id,
        'filename': filename,
        'category': style_cat,
        'category_name': all_styles().get(style_cat, {}).get('_name', style_cat),
        'key_params': {k: params[k] for k in SAT_KEYS if k in params},
    }


def all_styles() -> dict:
    """返回内置 + 用户模板的合并字典"""
    merged = dict(BUILTIN_STYLES)
    # 按 _source 类别聚合用户模板（同类取均值追加到内置模板上）
    category_groups: dict[str, list] = {}
    for sid, sp in _user_styles.items():
        cat = sp.get('_source', 'film_vintage')
        category_groups.setdefault(cat, []).append(sp)

    for cat, presets in category_groups.items():
        if cat not in merged:
            merged[cat] = dict(presets[0])
            continue
        base = dict(merged[cat])
        for key in SAT_KEYS + HUE_KEYS + LUM_KEYS:
            vals = [p[key] for p in presets if key in p]
            if vals:
                # 用户均值以 40% 权重混入内置模板
                user_mean = float(np.mean(vals))
                base[key] = int(round(base.get(key, 0) * 0.6 + user_mean * 0.4))
        merged[cat] = base

    return merged


# ─── 风格匹配 ─────────────────────────────────────────────────────────────────

def _style_vector(params: dict) -> np.ndarray:
    """提取风格特征向量（饱和度8维 + 亮度5维，量纲归一化）"""
    sat = np.array([params.get(k, 0) for k in SAT_KEYS],  dtype=np.float32)
    hue = np.array([params.get(k, 0) for k in HUE_KEYS],  dtype=np.float32) * 0.3
    lum_extra = np.array([
        params.get('Contrast',   0) * 0.6,
        params.get('Shadows',    0) * 0.4,
        params.get('Highlights', 0) * 0.4,
        params.get('Blacks',     0) * 0.4,
        params.get('Saturation', 0) * 0.5,
    ], dtype=np.float32)
    return np.concatenate([sat, hue, lum_extra])


def match_style(params: dict) -> tuple:
    """
    在全部模板中找最近风格。
    返回 (style_key, cosine_similarity, style_name)。
    similarity < 0.25 时返回 (None, 0, '')，表示不做混合。
    """
    query = _style_vector(params)
    qnorm = np.linalg.norm(query)
    if qnorm < 1e-6:
        return None, 0.0, ''

    best_key, best_sim = None, -1.0
    for key, style in all_styles().items():
        if key.startswith('_'):
            continue
        svec  = _style_vector(style)
        snorm = np.linalg.norm(svec)
        if snorm < 1e-6:
            continue
        sim = float(np.dot(query, svec) / (qnorm * snorm))
        if sim > best_sim:
            best_sim, best_key = sim, key

    if best_sim < 0.25:
        return None, 0.0, ''

    name = all_styles().get(best_key, {}).get('_name', best_key)
    return best_key, round(best_sim, 3), name


def blend_with_style(
    analysis_params: dict,
    style_key:       str,
    similarity:      float = 0.5,
) -> dict:
    """
    将图像分析参数与风格模板混合。
    blend_ratio 随 similarity 自适应：
      sim ≥ 0.70 → ratio 0.45
      sim ≥ 0.50 → ratio 0.35
      sim ≥ 0.30 → ratio 0.25
    """
    if similarity >= 0.70:
        ratio = 0.45
    elif similarity >= 0.50:
        ratio = 0.35
    else:
        ratio = 0.25

    style   = all_styles().get(style_key, {})
    result  = dict(analysis_params)
    mix_keys = SAT_KEYS + HUE_KEYS + LUM_KEYS

    for key in mix_keys:
        a_val = float(analysis_params.get(key, 0))
        t_val = float(style.get(key, 0))
        result[key] = int(round(a_val * (1 - ratio) + t_val * ratio))

    # 亮度参数做更保守的混合（ratio * 0.4），保留图像本身的曝光特征
    for key in ['Contrast', 'Shadows', 'Highlights', 'Blacks', 'Whites']:
        a_val = float(analysis_params.get(key, 0))
        t_val = float(style.get(key, 0))
        lum_ratio = ratio * 0.4
        result[key] = int(round(a_val * (1 - lum_ratio) + t_val * lum_ratio))

    return result


def list_styles() -> list:
    """返回所有可用风格的摘要列表（供前端展示）"""
    out = []
    for key, s in all_styles().items():
        if key.startswith('_'):
            continue
        out.append({
            'key':    key,
            'name':   s.get('_name', key),
            'desc':   s.get('_desc', ''),
            'source': 'builtin' if key in BUILTIN_STYLES else 'user',
        })
    return out
