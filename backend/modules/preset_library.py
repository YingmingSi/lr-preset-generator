"""
预设风格模板库
职责：
  1. 解析 XMP 文件 → 提取数值参数
  2. 内置 6 个经典风格模板
  3. 用户上传的 XMP 以聚类原型存储：
       - 相似度 ≥ MERGE_THRESHOLD → 合并到最近原型（加权平均）
       - 相似度  < MERGE_THRESHOLD 且原型数 < MAX_USER_CLUSTERS → 新建原型
       - 原型数已满 → 强制并入最近原型
  4. 每个原型自动生成语义标签和名称
  5. 将图像分析结果匹配到最近风格（余弦相似度），并混合
"""

import re
import json
import os
import numpy as np

# ─── 常量 ────────────────────────────────────────────────────────────────────

BUCKETS  = ['Red', 'Orange', 'Yellow', 'Green', 'Aqua', 'Blue', 'Purple', 'Magenta']
SAT_KEYS = [f'SaturationAdjustment{b}' for b in BUCKETS]
HUE_KEYS = [f'HueAdjustment{b}'        for b in BUCKETS]
LUM_KEYS = [f'LuminanceAdjustment{b}'  for b in BUCKETS]

MAX_USER_CLUSTERS = 12    # 用户聚类上限
MERGE_THRESHOLD   = 0.78  # 余弦相似度超过此值时合并而非新建

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'presets')

# ─── XMP 解析 ─────────────────────────────────────────────────────────────────

def parse_xmp_params(xmp_content: str) -> dict:
    """从 XMP 文本提取数值参数，兼容 crs:Key 和 crs:Key2012 两种格式"""
    params = {}
    for m in re.finditer(r'crs:(\w+?)(?:2012)?="(-?[\d.]+)"', xmp_content):
        key, val_str = m.group(1), m.group(2)
        try:
            params[key] = float(val_str) if '.' in val_str else int(val_str)
        except ValueError:
            pass
    return params


# ─── 自动分析与命名 ────────────────────────────────────────────────────────────

def _auto_tags(params: dict) -> list:
    """从参数特征生成语义标签列表"""
    warm_sat   = float(np.mean([params.get(f'SaturationAdjustment{b}', 0) for b in ('Red', 'Orange', 'Yellow')]))
    cool_sat   = float(np.mean([params.get(f'SaturationAdjustment{b}', 0) for b in ('Aqua', 'Blue', 'Purple')]))
    orange_sat = float(params.get('SaturationAdjustmentOrange', 0))
    aqua_sat   = float(params.get('SaturationAdjustmentAqua',   0))
    green_sat  = float(params.get('SaturationAdjustmentGreen',  0))
    overall    = float(params.get('Saturation',  0))
    blacks     = float(params.get('Blacks',       0))
    shadows    = float(params.get('Shadows',      0))
    contrast   = float(params.get('Contrast',     0))
    highlights = float(params.get('Highlights',   0))

    tags = []

    # 色调方向：先检查橙+青的签名性特征，再看整体均值
    if orange_sat > 40 and aqua_sat > 25:
        tags.append('青橙')
    elif warm_sat > 18 and cool_sat > 12:
        tags.append('青橙')
    elif warm_sat > 15 or orange_sat > 25:
        tags.append('暖色')
    elif cool_sat > 15 or aqua_sat > 25:
        tags.append('冷调')
    if green_sat > 30:
        tags.append('自然绿')

    # 影调风格
    if blacks < -30 and contrast > 30:
        tags.append('电影感')
    elif blacks > 18 or (shadows > 30 and contrast < -10):
        tags.append('胶片感')
    elif shadows > 40 and contrast < -20:
        tags.append('日系')

    # 饱和度
    all_sat_neg = warm_sat < 0 and cool_sat < 0
    if overall < -15 or (all_sat_neg and abs(warm_sat) + abs(cool_sat) > 30):
        tags.append('低饱和')
    elif overall > 25 or warm_sat > 40 or cool_sat > 40:
        tags.append('高饱和')

    # 反差
    if contrast > 45:
        tags.append('高对比')
    elif highlights < -20 and shadows > 15:
        tags.append('低反差')

    if not tags:
        tags.append('中性')

    return tags


def _auto_name(params: dict) -> str:
    tags = _auto_tags(params)
    if '青橙'   in tags:                            return '青橙'
    if '电影感' in tags and '暖色' in tags:          return '暖调电影'
    if '电影感' in tags:                             return '暗调电影'
    if '日系'   in tags:                            return '日系清新'
    if '胶片感' in tags and '暖色' in tags:          return '暖调胶片'
    if '胶片感' in tags:                             return '胶片复古'
    if '低饱和' in tags and '暖色' in tags:          return '日系暖调'
    if '低饱和' in tags:                             return '低饱和淡雅'
    if '自然绿' in tags:                             return '清新自然'
    if '高饱和' in tags and '暖色' in tags:          return '暖色鲜艳'
    if '冷调'   in tags:                            return '冷调风格'
    if '暖色'   in tags:                            return '暖调风格'
    return '自然调色'


# ─── 内置经典风格模板 ─────────────────────────────────────────────────────────

BUILTIN_STYLES: dict = {
    'teal_orange': {
        '_name': '青橙',
        '_desc': '暖色偏橙，冷色偏青，橙青饱和大幅提高，绿/紫/品红全压低',
        '_tags': ['青橙', '高饱和'],
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
        '_tags': ['暖色', '电影感', '高对比'],
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
        '_tags': ['日系', '低饱和', '低反差'],
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
        '_tags': ['胶片感', '暖色'],
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
        '_tags': ['电影感', '高对比'],
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
        '_tags': ['自然绿', '高饱和'],
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

# ─── 运行时用户聚类存储 ───────────────────────────────────────────────────────
# 格式：{ cluster_key: { name, desc, tags, count, source, params } }
_user_styles: dict = {}


# ─── 持久化 ───────────────────────────────────────────────────────────────────

def _user_styles_path() -> str:
    return os.path.join(_DATA_DIR, '_user_styles.json')


def load_user_styles() -> None:
    """启动时从磁盘加载用户聚类 JSON"""
    global _user_styles
    json_path = _user_styles_path()
    if not os.path.exists(json_path):
        return
    with open(json_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    # 兼容旧格式（params 直接平铺在顶层）
    for key, val in raw.items():
        if 'params' in val:
            _user_styles[key] = val
        else:
            # 旧格式迁移：提取真实参数，重建为新格式
            params = {k: v for k, v in val.items() if not k.startswith('_')}
            if params:
                _user_styles[key] = _make_cluster(params, source='user')


def save_user_styles() -> None:
    os.makedirs(_DATA_DIR, exist_ok=True)
    with open(_user_styles_path(), 'w', encoding='utf-8') as f:
        json.dump(_user_styles, f, ensure_ascii=False, indent=2)


# ─── 聚类核心逻辑 ────────────────────────────────────────────────────────────

def _make_cluster(params: dict, source: str = 'user') -> dict:
    """从参数字典创建一个新聚类原型"""
    tags = _auto_tags(params)
    return {
        'name':   _auto_name(params),
        'desc':   ' · '.join(tags),
        'tags':   tags,
        'count':  1,
        'source': source,
        'params': {k: params[k] for k in (SAT_KEYS + HUE_KEYS + LUM_KEYS +
                   ['Contrast','Shadows','Highlights','Blacks','Whites','Saturation','Vibrance'])
                   if k in params},
    }


def _merge_cluster(key: str, new_params: dict) -> None:
    """将新参数加权合并到已有聚类（运行加权平均）"""
    cluster = _user_styles[key]
    old_params = cluster['params']
    n = cluster['count']
    all_keys = set(old_params) | set(new_params)
    merged = {}
    for k in all_keys:
        old_v = float(old_params.get(k, 0))
        new_v = float(new_params.get(k, 0))
        merged[k] = round((old_v * n + new_v) / (n + 1), 1)
        # 存整数（LR 参数均为整数）
        if isinstance(old_params.get(k, 0), int) and isinstance(new_params.get(k, 0), int):
            merged[k] = int(round(merged[k]))
    cluster['params'] = merged
    cluster['count']  = n + 1
    # 重新计算语义标签（合并后可能偏移）
    tags             = _auto_tags(merged)
    cluster['tags']  = tags
    cluster['desc']  = ' · '.join(tags)
    cluster['name']  = _auto_name(merged)


def _find_closest_user(vec: np.ndarray) -> tuple:
    """在用户聚类中找余弦最近邻，返回 (key, similarity)"""
    qnorm = np.linalg.norm(vec)
    if qnorm < 1e-6 or not _user_styles:
        return None, 0.0
    best_key, best_sim = None, -1.0
    for key, cluster in _user_styles.items():
        svec  = _style_vector(cluster['params'])
        snorm = np.linalg.norm(svec)
        if snorm < 1e-6:
            continue
        sim = float(np.dot(vec, svec) / (qnorm * snorm))
        if sim > best_sim:
            best_sim, best_key = sim, key
    return best_key, best_sim


def _next_cluster_key() -> str:
    existing = [k for k in _user_styles if k.startswith('u_')]
    idx = len(existing)
    while f'u_{idx}' in _user_styles:
        idx += 1
    return f'u_{idx}'


def add_user_preset(xmp_content: str, filename: str) -> dict:
    """
    解析并添加用户上传的 XMP 预设。
    自动执行聚类合并：
      - 相似度 ≥ MERGE_THRESHOLD → 合并到最近原型
      - 原型数 < MAX_USER_CLUSTERS → 新建原型
      - 原型数已满 → 强制并入最近原型
    返回摘要信息。
    """
    params = parse_xmp_params(xmp_content)
    if not params:
        return {'filename': filename, 'error': '无法解析 XMP 参数'}

    vec = _style_vector(params)
    closest_key, closest_sim = _find_closest_user(vec)

    if closest_key is not None and closest_sim >= MERGE_THRESHOLD:
        # 足够相似，合并
        _merge_cluster(closest_key, params)
        action = 'merged'
        target_key = closest_key
    elif len(_user_styles) < MAX_USER_CLUSTERS:
        # 有空位，新建原型
        target_key = _next_cluster_key()
        _user_styles[target_key] = _make_cluster(params)
        action = 'added'
        closest_sim = 0.0
    else:
        # 库已满，强制并入最近原型
        _merge_cluster(closest_key, params)
        action = 'merged_full'
        target_key = closest_key

    save_user_styles()

    cluster = _user_styles[target_key]
    return {
        'filename': filename,
        'action':   action,
        'name':     cluster['name'],
        'desc':     cluster['desc'],
        'tags':     cluster['tags'],
        'count':    cluster['count'],
        'similarity': round(closest_sim, 3) if action != 'added' else None,
    }


# ─── 风格汇总 ─────────────────────────────────────────────────────────────────

def all_styles() -> dict:
    """
    返回内置模板 + 用户聚类的合并字典。
    用户聚类的 params 被展开为顶层键，与内置模板格式一致，
    供 _style_vector / blend_with_style 直接使用。
    """
    merged = dict(BUILTIN_STYLES)
    for key, cluster in _user_styles.items():
        flat = {**cluster['params'],
                '_name':  cluster['name'],
                '_desc':  cluster['desc'],
                '_tags':  cluster['tags'],
                '_count': cluster['count']}
        merged[key] = flat
    return merged


# ─── 风格匹配 ─────────────────────────────────────────────────────────────────

def _style_vector(params: dict) -> np.ndarray:
    """提取风格特征向量（饱和度8维 + 色相8维×0.3 + 亮度5维）"""
    sat = np.array([params.get(k, 0) for k in SAT_KEYS], dtype=np.float32)
    hue = np.array([params.get(k, 0) for k in HUE_KEYS], dtype=np.float32) * 0.3
    lum = np.array([
        params.get('Contrast',   0) * 0.6,
        params.get('Shadows',    0) * 0.4,
        params.get('Highlights', 0) * 0.4,
        params.get('Blacks',     0) * 0.4,
        params.get('Saturation', 0) * 0.5,
    ], dtype=np.float32)
    return np.concatenate([sat, hue, lum])


def match_style(params: dict) -> tuple:
    """
    在全部模板中找最近风格。
    返回 (style_key, cosine_similarity, style_name)。
    similarity < 0.25 时返回 (None, 0, '')，不做混合。
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
    亮度参数额外衰减（ratio × 0.4），保留原图曝光特征。
    """
    if similarity >= 0.70:
        ratio = 0.45
    elif similarity >= 0.50:
        ratio = 0.35
    else:
        ratio = 0.25

    style  = all_styles().get(style_key, {})
    result = dict(analysis_params)

    for key in SAT_KEYS + HUE_KEYS + LUM_KEYS:
        a = float(analysis_params.get(key, 0))
        t = float(style.get(key, 0))
        result[key] = int(round(a * (1 - ratio) + t * ratio))

    lum_ratio = ratio * 0.4
    for key in ['Contrast', 'Shadows', 'Highlights', 'Blacks', 'Whites']:
        a = float(analysis_params.get(key, 0))
        t = float(style.get(key, 0))
        result[key] = int(round(a * (1 - lum_ratio) + t * lum_ratio))

    return result


def list_styles() -> list:
    """返回所有可用风格的摘要列表（供前端展示）"""
    out = []
    for key, s in all_styles().items():
        if key.startswith('_'):
            continue
        count = s.get('_count', None)
        out.append({
            'key':    key,
            'name':   s.get('_name', key),
            'desc':   s.get('_desc', ''),
            'tags':   s.get('_tags', []),
            'source': 'builtin' if key in BUILTIN_STYLES else 'user',
            'count':  count,
        })
    return out
