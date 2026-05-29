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

MAX_USER_CLUSTERS      = 25   # K-means 批量模式上限
MERGE_THRESHOLD        = 0.90  # 批量顺序模式合并阈值
INCREMENTAL_THRESHOLD  = 0.95  # 增量模式合并阈值（更严格，保护新风格）
BATCH_KMEANS_MIN       = 15   # ≥此数量用 K-means，否则走增量模式

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
    """
    从参数特征生成语义标签列表。

    设计原则：
    · 色调方向用「各方向最大正向值」而非均值，避免被零值通道稀释
      （Orange=25, Red=0, Yellow=0 → warm_max=25，而非均值 8.3）
    · 青橙判断：Orange 和 Aqua 同时为正，且都不低于全局最大绝对值的 20%
    · 阈值相对化，适配真实用户预设（幅度 10–40）而非内置样本（70–90）
    """
    sat = {b: float(params.get(f'SaturationAdjustment{b}', 0)) for b in BUCKETS}

    # 各方向最大正向提升值（负值归零，不污染方向判断）
    warm_max  = max(sat['Red'],  sat['Orange'], sat['Yellow'], 0)
    cool_max  = max(sat['Aqua'], sat['Blue'],   sat['Purple'], 0)
    green_max = max(sat['Green'], 0)

    # 全局最大绝对值（衡量整体调整强度）
    all_max    = max((abs(v) for v in sat.values()), default=0)
    orange_sat = sat['Orange']
    aqua_sat   = sat['Aqua']
    overall    = float(params.get('Saturation',  0))
    blacks     = float(params.get('Blacks',       0))
    shadows    = float(params.get('Shadows',      0))
    contrast   = float(params.get('Contrast',     0))
    highlights = float(params.get('Highlights',   0))

    tags = []

    # ── 色调方向 ──────────────────────────────────────────────────────────
    # 最低有效阈值 8：低于此视为噪声，不作方向判断
    MIN_SAT = 8
    if warm_max >= MIN_SAT or cool_max >= MIN_SAT or green_max >= MIN_SAT:
        dominant = max(warm_max, cool_max, green_max)
        sig = dominant * 0.20   # 次方向至少达到主方向 20% 才算"共同存在"

        # 青橙：橙色和青色都是正向，且都足够显著
        if orange_sat >= max(sig, MIN_SAT) and aqua_sat >= max(sig, MIN_SAT):
            tags.append('青橙')
        elif warm_max >= cool_max and warm_max >= green_max:
            tags.append('暖色')
        elif cool_max > warm_max and cool_max >= green_max:
            tags.append('冷调')

        # 绿色独立主导时打标
        if green_max >= MIN_SAT and green_max > warm_max * 1.2 and green_max > cool_max * 1.2:
            tags.append('自然绿')

    # ── 整体去饱和 ────────────────────────────────────────────────────────
    neg_count = sum(1 for v in sat.values() if v < -8)
    if overall < -10 or (neg_count >= 5 and all_max < 20):
        tags.append('低饱和')
    elif overall > 15 or all_max >= 35:
        tags.append('高饱和')

    # ── 影调风格（独立判断，不互斥；_auto_name 中按优先级选主标签）────────
    if blacks < -22 and contrast > 22:
        tags.append('电影感')
    if blacks > 12 or (shadows > 22 and contrast < -8):
        tags.append('胶片感')
    if shadows > 32 and contrast < -15:
        tags.append('日系')

    # ── 反差 ──────────────────────────────────────────────────────────────
    if contrast > 38:
        tags.append('高对比')
    elif highlights < -15 and shadows > 10:
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
    # ── 1. 青橙 ──────────────────────────────────────────────────────────────
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
    # ── 7. 暖阳落日 ───────────────────────────────────────────────────────────
    'sunset_warm': {
        '_name': '暖阳落日',
        '_desc': '橙黄暖光主导，冷色大幅压低，适合日出日落',
        '_tags': ['暖色', '高饱和'],
        'SaturationAdjustmentRed':    38,
        'SaturationAdjustmentOrange': 68,
        'SaturationAdjustmentYellow': 48,
        'SaturationAdjustmentGreen':  -28,
        'SaturationAdjustmentAqua':   -38,
        'SaturationAdjustmentBlue':   -45,
        'SaturationAdjustmentPurple': -22,
        'SaturationAdjustmentMagenta': 15,
        'HueAdjustmentRed':    8,
        'HueAdjustmentOrange': 10,
        'HueAdjustmentYellow':  5,
        'HueAdjustmentGreen':  15,
        'HueAdjustmentAqua':   -8,
        'HueAdjustmentBlue':  -12,
        'HueAdjustmentPurple':  0,
        'HueAdjustmentMagenta': 5,
        'LuminanceAdjustmentOrange': 18,
        'LuminanceAdjustmentYellow': 12,
        'LuminanceAdjustmentBlue':  -12,
        'Shadows': 18, 'Highlights': -28, 'Whites': 15,
        'Contrast': 20, 'Saturation': 8,
    },
    # ── 8. 冷调都市 ───────────────────────────────────────────────────────────
    'cold_city': {
        '_name': '冷调都市',
        '_desc': '蓝青主导，压暖色，高对比，适合建筑街拍',
        '_tags': ['冷调', '高对比', '电影感'],
        'SaturationAdjustmentRed':    -28,
        'SaturationAdjustmentOrange': -30,
        'SaturationAdjustmentYellow': -18,
        'SaturationAdjustmentGreen':  -15,
        'SaturationAdjustmentAqua':    42,
        'SaturationAdjustmentBlue':    48,
        'SaturationAdjustmentPurple':  18,
        'SaturationAdjustmentMagenta':-18,
        'HueAdjustmentRed':    0,
        'HueAdjustmentOrange': 0,
        'HueAdjustmentYellow': 0,
        'HueAdjustmentGreen':  0,
        'HueAdjustmentAqua':  -10,
        'HueAdjustmentBlue':  -18,
        'HueAdjustmentPurple': 0,
        'HueAdjustmentMagenta':0,
        'LuminanceAdjustmentBlue':  -12,
        'LuminanceAdjustmentAqua':   -8,
        'Blacks': -38, 'Shadows': -22, 'Highlights': -38,
        'Contrast': 48, 'Saturation': -12,
    },
    # ── 9. 梦幻柔光 ───────────────────────────────────────────────────────────
    'soft_dream': {
        '_name': '梦幻柔光',
        '_desc': '提黑柔和，品红紫调，低对比，少女感',
        '_tags': ['胶片感', '低反差', '低饱和'],
        'SaturationAdjustmentRed':    12,
        'SaturationAdjustmentOrange': 18,
        'SaturationAdjustmentYellow': -8,
        'SaturationAdjustmentGreen':  -25,
        'SaturationAdjustmentAqua':   -18,
        'SaturationAdjustmentBlue':   -22,
        'SaturationAdjustmentPurple':  15,
        'SaturationAdjustmentMagenta': 22,
        'HueAdjustmentRed':    8,
        'HueAdjustmentOrange': 5,
        'HueAdjustmentYellow': 0,
        'HueAdjustmentGreen':  20,
        'HueAdjustmentAqua':   12,
        'HueAdjustmentBlue':   15,
        'HueAdjustmentPurple': -8,
        'HueAdjustmentMagenta':-5,
        'LuminanceAdjustmentRed':    15,
        'LuminanceAdjustmentOrange': 10,
        'LuminanceAdjustmentMagenta': 8,
        'Blacks': 28, 'Shadows': 32, 'Highlights': -12, 'Whites': 18,
        'Contrast': -28, 'Saturation': -15, 'Vibrance': 10,
    },
    # ── 10. 翡翠森林 ──────────────────────────────────────────────────────────
    'emerald_forest': {
        '_name': '翡翠森林',
        '_desc': '深饱和绿色，青调辅助，暗调森林感',
        '_tags': ['自然绿', '高饱和'],
        'SaturationAdjustmentRed':    -22,
        'SaturationAdjustmentOrange': -18,
        'SaturationAdjustmentYellow':  28,
        'SaturationAdjustmentGreen':   78,
        'SaturationAdjustmentAqua':    38,
        'SaturationAdjustmentBlue':    18,
        'SaturationAdjustmentPurple': -20,
        'SaturationAdjustmentMagenta':-18,
        'HueAdjustmentRed':    0,
        'HueAdjustmentOrange': 0,
        'HueAdjustmentYellow':-10,
        'HueAdjustmentGreen': -12,
        'HueAdjustmentAqua':   -5,
        'HueAdjustmentBlue':    8,
        'HueAdjustmentPurple':  0,
        'HueAdjustmentMagenta': 0,
        'LuminanceAdjustmentGreen': 10,
        'LuminanceAdjustmentAqua':   5,
        'LuminanceAdjustmentBlue':  -8,
        'Blacks': -22, 'Shadows': -12, 'Highlights': -20,
        'Contrast': 28, 'Vibrance': 22,
    },
    # ── 11. 人像肤色 ──────────────────────────────────────────────────────────
    'portrait_skin': {
        '_name': '人像肤色',
        '_desc': '橙色肤色强化，冷色轻度压低，高光柔和，自然感',
        '_tags': ['暖色', '低反差'],
        'SaturationAdjustmentRed':    28,
        'SaturationAdjustmentOrange': 42,
        'SaturationAdjustmentYellow': 18,
        'SaturationAdjustmentGreen':  -22,
        'SaturationAdjustmentAqua':   -28,
        'SaturationAdjustmentBlue':   -18,
        'SaturationAdjustmentPurple': -12,
        'SaturationAdjustmentMagenta': 10,
        'HueAdjustmentRed':    5,
        'HueAdjustmentOrange': 6,
        'HueAdjustmentYellow': 3,
        'HueAdjustmentGreen':  0,
        'HueAdjustmentAqua':   0,
        'HueAdjustmentBlue':   0,
        'HueAdjustmentPurple': 0,
        'HueAdjustmentMagenta':0,
        'LuminanceAdjustmentRed':    8,
        'LuminanceAdjustmentOrange': 15,
        'LuminanceAdjustmentYellow': 5,
        'Blacks': 12, 'Shadows': 28, 'Highlights': -15,
        'Contrast': -15, 'Saturation': -5, 'Vibrance': 18,
    },
    # ── 12. 落日金光 ──────────────────────────────────────────────────────────
    'golden_hour': {
        '_name': '落日金光',
        '_desc': '极暖橙黄，强烈金色质感，大幅压蓝绿',
        '_tags': ['暖色', '高饱和', '高对比'],
        'SaturationAdjustmentRed':    48,
        'SaturationAdjustmentOrange': 75,
        'SaturationAdjustmentYellow': 60,
        'SaturationAdjustmentGreen':  -30,
        'SaturationAdjustmentAqua':   -50,
        'SaturationAdjustmentBlue':   -58,
        'SaturationAdjustmentPurple': -28,
        'SaturationAdjustmentMagenta': 18,
        'HueAdjustmentRed':    8,
        'HueAdjustmentOrange': 12,
        'HueAdjustmentYellow':  5,
        'HueAdjustmentGreen':  18,
        'HueAdjustmentAqua':  -10,
        'HueAdjustmentBlue':  -15,
        'HueAdjustmentPurple':  0,
        'HueAdjustmentMagenta': 8,
        'LuminanceAdjustmentOrange': 22,
        'LuminanceAdjustmentYellow': 16,
        'LuminanceAdjustmentBlue':  -15,
        'Shadows': 15, 'Highlights': -32, 'Whites': 22,
        'Contrast': 22, 'Saturation': 12,
    },
    # ── 13. 霓虹赛博 ──────────────────────────────────────────────────────────
    'neon_cyber': {
        '_name': '霓虹赛博',
        '_desc': '青色+品红霓虹，压暖色，深黑高对比',
        '_tags': ['冷调', '电影感', '高对比', '高饱和'],
        'SaturationAdjustmentRed':    -35,
        'SaturationAdjustmentOrange': -42,
        'SaturationAdjustmentYellow': -48,
        'SaturationAdjustmentGreen':  -38,
        'SaturationAdjustmentAqua':    88,
        'SaturationAdjustmentBlue':    38,
        'SaturationAdjustmentPurple':  48,
        'SaturationAdjustmentMagenta': 68,
        'HueAdjustmentRed':    0,
        'HueAdjustmentOrange': 0,
        'HueAdjustmentYellow': 0,
        'HueAdjustmentGreen':  0,
        'HueAdjustmentAqua':  -18,
        'HueAdjustmentBlue':  -12,
        'HueAdjustmentPurple':-10,
        'HueAdjustmentMagenta':12,
        'LuminanceAdjustmentAqua':   -10,
        'LuminanceAdjustmentBlue':   -12,
        'LuminanceAdjustmentMagenta': 5,
        'Blacks': -48, 'Shadows': -22, 'Highlights': -28,
        'Contrast': 58, 'Saturation': -15,
    },
    # ── 14. 哑光褪色 ──────────────────────────────────────────────────────────
    'matte_fade': {
        '_name': '哑光褪色',
        '_desc': '提黑褪色，全通道轻度去饱和，低对比哑光质感',
        '_tags': ['胶片感', '低饱和', '低反差'],
        'SaturationAdjustmentRed':    -18,
        'SaturationAdjustmentOrange': -15,
        'SaturationAdjustmentYellow': -12,
        'SaturationAdjustmentGreen':  -22,
        'SaturationAdjustmentAqua':   -18,
        'SaturationAdjustmentBlue':   -22,
        'SaturationAdjustmentPurple': -15,
        'SaturationAdjustmentMagenta':-12,
        'HueAdjustmentRed':    5,
        'HueAdjustmentOrange': 5,
        'HueAdjustmentYellow': 3,
        'HueAdjustmentGreen':  8,
        'HueAdjustmentAqua':   5,
        'HueAdjustmentBlue':   8,
        'HueAdjustmentPurple': 5,
        'HueAdjustmentMagenta':3,
        'Blacks': 28, 'Shadows': 22, 'Highlights': -12,
        'Contrast': -28, 'Saturation': -22, 'Vibrance': -8,
    },
    # ── 15. 高调清透 ──────────────────────────────────────────────────────────
    'high_key': {
        '_name': '高调清透',
        '_desc': '整体提亮，低对比，干净清爽，适合白底产品/人像',
        '_tags': ['低反差', '低饱和'],
        'SaturationAdjustmentRed':    10,
        'SaturationAdjustmentOrange': 12,
        'SaturationAdjustmentYellow': -5,
        'SaturationAdjustmentGreen':  -10,
        'SaturationAdjustmentAqua':   -12,
        'SaturationAdjustmentBlue':   -15,
        'SaturationAdjustmentPurple': -10,
        'SaturationAdjustmentMagenta':  5,
        'HueAdjustmentRed':    3,
        'HueAdjustmentOrange': 3,
        'HueAdjustmentYellow': 0,
        'HueAdjustmentGreen':  5,
        'HueAdjustmentAqua':   5,
        'HueAdjustmentBlue':   8,
        'HueAdjustmentPurple': 3,
        'HueAdjustmentMagenta':0,
        'LuminanceAdjustmentRed':    18,
        'LuminanceAdjustmentOrange': 12,
        'LuminanceAdjustmentYellow': 10,
        'LuminanceAdjustmentGreen':   5,
        'Blacks': 22, 'Shadows': 38, 'Highlights': -10, 'Whites': 25,
        'Contrast': -22, 'Saturation': -12, 'Vibrance': 8,
    },
    # ── 16. 戏剧人像 ──────────────────────────────────────────────────────────
    'dramatic_portrait': {
        '_name': '戏剧人像',
        '_desc': '深黑高对比，强化橙色肤色，冷调背景形成分离感',
        '_tags': ['暖色', '电影感', '高对比'],
        'SaturationAdjustmentRed':    32,
        'SaturationAdjustmentOrange': 58,
        'SaturationAdjustmentYellow': 22,
        'SaturationAdjustmentGreen':  -42,
        'SaturationAdjustmentAqua':   -35,
        'SaturationAdjustmentBlue':   -28,
        'SaturationAdjustmentPurple': -20,
        'SaturationAdjustmentMagenta':-15,
        'HueAdjustmentRed':    5,
        'HueAdjustmentOrange': 8,
        'HueAdjustmentYellow': 3,
        'HueAdjustmentGreen':  0,
        'HueAdjustmentAqua':  -8,
        'HueAdjustmentBlue':  -12,
        'HueAdjustmentPurple': 0,
        'HueAdjustmentMagenta':0,
        'LuminanceAdjustmentOrange':  -8,
        'LuminanceAdjustmentBlue':   -20,
        'LuminanceAdjustmentGreen':  -15,
        'Blacks': -58, 'Shadows': -38, 'Highlights': -30,
        'Contrast': 68, 'Vibrance': 15,
    },
    # ── 17. 旅行鲜艳 ──────────────────────────────────────────────────────────
    'vivid_travel': {
        '_name': '旅行鲜艳',
        '_desc': '全通道高饱和，鲜亮活力，适合旅行风光',
        '_tags': ['高饱和', '自然绿'],
        'SaturationAdjustmentRed':    25,
        'SaturationAdjustmentOrange': 38,
        'SaturationAdjustmentYellow': 42,
        'SaturationAdjustmentGreen':  58,
        'SaturationAdjustmentAqua':   45,
        'SaturationAdjustmentBlue':   50,
        'SaturationAdjustmentPurple': 22,
        'SaturationAdjustmentMagenta': 18,
        'HueAdjustmentRed':    0,
        'HueAdjustmentOrange': 0,
        'HueAdjustmentYellow': -5,
        'HueAdjustmentGreen': -10,
        'HueAdjustmentAqua':   -5,
        'HueAdjustmentBlue':    5,
        'HueAdjustmentPurple':  0,
        'HueAdjustmentMagenta': 0,
        'LuminanceAdjustmentGreen':  8,
        'LuminanceAdjustmentAqua':   5,
        'LuminanceAdjustmentBlue':  -5,
        'Shadows': 12, 'Highlights': -18,
        'Contrast': 15, 'Vibrance': 45, 'Saturation': 22,
    },
    # ── 18. 北欧冬日 ──────────────────────────────────────────────────────────
    'nordic_winter': {
        '_name': '北欧冬日',
        '_desc': '冷蓝白调，暖色去饱和，低对比高明度，北欧简约',
        '_tags': ['冷调', '低饱和', '低反差'],
        'SaturationAdjustmentRed':    -28,
        'SaturationAdjustmentOrange': -32,
        'SaturationAdjustmentYellow': -22,
        'SaturationAdjustmentGreen':  -18,
        'SaturationAdjustmentAqua':    28,
        'SaturationAdjustmentBlue':    38,
        'SaturationAdjustmentPurple':  10,
        'SaturationAdjustmentMagenta':-12,
        'HueAdjustmentRed':    0,
        'HueAdjustmentOrange': 0,
        'HueAdjustmentYellow': 0,
        'HueAdjustmentGreen':  8,
        'HueAdjustmentAqua':   -5,
        'HueAdjustmentBlue':  -12,
        'HueAdjustmentPurple':  0,
        'HueAdjustmentMagenta': 0,
        'LuminanceAdjustmentBlue':   10,
        'LuminanceAdjustmentAqua':    8,
        'Blacks': 18, 'Shadows': 28, 'Highlights': -18, 'Whites': 12,
        'Contrast': -18, 'Saturation': -18, 'Vibrance': -5,
    },
    # ── 19. 秋叶暖色 ──────────────────────────────────────────────────────────
    'autumn_leaves': {
        '_name': '秋叶暖色',
        '_desc': '橙红黄浓郁，绿色偏黄褪去，蓝天轻压，秋日氛围',
        '_tags': ['暖色', '高饱和'],
        'SaturationAdjustmentRed':    55,
        'SaturationAdjustmentOrange': 68,
        'SaturationAdjustmentYellow': 50,
        'SaturationAdjustmentGreen':  -42,
        'SaturationAdjustmentAqua':   -30,
        'SaturationAdjustmentBlue':   -32,
        'SaturationAdjustmentPurple': -18,
        'SaturationAdjustmentMagenta': 15,
        'HueAdjustmentRed':    5,
        'HueAdjustmentOrange': 8,
        'HueAdjustmentYellow': -8,
        'HueAdjustmentGreen':  22,
        'HueAdjustmentAqua':  -5,
        'HueAdjustmentBlue':   -8,
        'HueAdjustmentPurple':  0,
        'HueAdjustmentMagenta': 5,
        'LuminanceAdjustmentOrange': 15,
        'LuminanceAdjustmentYellow': 10,
        'LuminanceAdjustmentGreen':  -8,
        'Shadows': 12, 'Highlights': -20,
        'Contrast': 22, 'Saturation': 12,
    },
    # ── 20. 都市青调 ──────────────────────────────────────────────────────────
    'urban_teal': {
        '_name': '都市青调',
        '_desc': '青色主导，暖色适度压低，中对比，都市建筑感',
        '_tags': ['青橙', '冷调'],
        'SaturationAdjustmentRed':    -22,
        'SaturationAdjustmentOrange': -28,
        'SaturationAdjustmentYellow': -15,
        'SaturationAdjustmentGreen':  -18,
        'SaturationAdjustmentAqua':    68,
        'SaturationAdjustmentBlue':    45,
        'SaturationAdjustmentPurple':  15,
        'SaturationAdjustmentMagenta':-20,
        'HueAdjustmentRed':    0,
        'HueAdjustmentOrange': 0,
        'HueAdjustmentYellow': 0,
        'HueAdjustmentGreen': -5,
        'HueAdjustmentAqua':  -10,
        'HueAdjustmentBlue':  -15,
        'HueAdjustmentPurple': 0,
        'HueAdjustmentMagenta':0,
        'LuminanceAdjustmentAqua':  -10,
        'LuminanceAdjustmentBlue':  -12,
        'Blacks': -30, 'Shadows': -18, 'Highlights': -25,
        'Contrast': 38, 'Saturation': -10,
    },
}

# ─── 运行时存储 ───────────────────────────────────────────────────────────────
# _seeded_styles : 已固化的风格（committed to git，永久可用）
# _user_styles   : 当前 session 上传的实验聚类（可随时 Reset）
_seeded_styles: dict = {}
_user_styles:   dict = {}

_DATA_DIR_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # repo root / backend
_SEEDED_PATH   = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'seeded_styles.json')


# ─── 持久化 ───────────────────────────────────────────────────────────────────

def _user_styles_path() -> str:
    return os.path.join(_DATA_DIR, '_user_styles.json')


def _load_cluster_dict(raw: dict) -> dict:
    """将 JSON dict 加载为规范 cluster 格式（兼容旧格式迁移）"""
    result = {}
    for key, val in raw.items():
        if 'params' in val:
            result[key] = val
        else:
            params = {k: v for k, v in val.items() if not k.startswith('_')}
            if params:
                result[key] = _make_cluster(params, source='seeded')
    return result


def load_user_styles() -> None:
    """启动时加载 seeded_styles.json（固化库）和 _user_styles.json（会话库）"""
    global _seeded_styles, _user_styles

    # 固化库：每次部署都有，不会被 Reset 清除
    if os.path.exists(_SEEDED_PATH):
        with open(_SEEDED_PATH, 'r', encoding='utf-8') as f:
            _seeded_styles = _load_cluster_dict(json.load(f))

    # 会话库：当前 session 上传的实验聚类
    session_path = _user_styles_path()
    if os.path.exists(session_path):
        with open(session_path, 'r', encoding='utf-8') as f:
            _user_styles = _load_cluster_dict(json.load(f))


def save_user_styles() -> None:
    os.makedirs(_DATA_DIR, exist_ok=True)
    with open(_user_styles_path(), 'w', encoding='utf-8') as f:
        json.dump(_user_styles, f, ensure_ascii=False, indent=2)


def promote_to_seeded() -> dict:
    """
    将当前 session 的 K-means 聚类固化为基础库（写入 seeded_styles.json）。
    固化后的风格在任何环境部署后都自动可用，无需重新上传 XMP。
    返回固化摘要。
    """
    global _seeded_styles

    if not _user_styles:
        return {'promoted': 0, 'message': '当前会话没有可固化的聚类'}

    # 将 session 聚类合并进 seeded，source 标记为 'seeded'
    new_count = 0
    for key, cluster in _user_styles.items():
        seeded_cluster = dict(cluster)
        seeded_cluster['source'] = 'seeded'
        _seeded_styles[key] = seeded_cluster
        new_count += 1

    os.makedirs(os.path.dirname(_SEEDED_PATH), exist_ok=True)
    with open(_SEEDED_PATH, 'w', encoding='utf-8') as f:
        json.dump(_seeded_styles, f, ensure_ascii=False, indent=2)

    return {
        'promoted':    new_count,
        'total_seeded': len(_seeded_styles),
        'path':        _SEEDED_PATH,
        'message':     f'已固化 {new_count} 个聚类到 seeded_styles.json，提交到 git 后永久生效',
    }


def reset_user_styles() -> None:
    """清空当前 session 的聚类（不影响已固化的 seeded_styles）"""
    global _user_styles
    _user_styles = {}
    path = _user_styles_path()
    if os.path.exists(path):
        os.remove(path)


# ─── K-means 批量聚类 ─────────────────────────────────────────────────────────

def _kmeans(X: np.ndarray, k: int, n_iter: int = 80) -> tuple:
    """
    简单 K-means++（纯 numpy，无外部依赖）。
    返回 (labels, centers)，labels[i] 是第 i 个样本所属聚类。
    """
    n = X.shape[0]
    k = min(k, n)

    # K-means++ 初始化
    rng = np.random.default_rng(42)
    center_idx = [int(rng.integers(n))]
    for _ in range(k - 1):
        dists = np.array([
            min(float(np.linalg.norm(X[i] - X[c])) for c in center_idx)
            for i in range(n)
        ])
        probs = dists ** 2
        total = probs.sum()
        if total < 1e-12:
            break
        center_idx.append(int(rng.choice(n, p=probs / total)))
    centers = X[center_idx].copy().astype(np.float64)

    labels = np.zeros(n, dtype=int)
    for _ in range(n_iter):
        # 分配：计算每个样本到各中心的距离
        diffs   = X[:, None, :] - centers[None, :, :]   # n × k × d
        dists   = np.linalg.norm(diffs, axis=2)          # n × k
        new_lbl = np.argmin(dists, axis=1)
        if np.all(new_lbl == labels):
            break
        labels = new_lbl
        # 更新中心
        for j in range(k):
            mask = labels == j
            if mask.any():
                centers[j] = X[mask].mean(axis=0)

    return labels, centers


def batch_cluster(params_list: list, filenames: list) -> list:
    """
    对大批量 XMP 参数使用 K-means 聚类，替换现有用户聚类。
    k = min(n // 4, MAX_USER_CLUSTERS)，每个聚类至少 3 个文件。

    返回每个新建聚类的摘要列表。
    """
    global _user_styles

    n = len(params_list)
    if n == 0:
        return []

    k = max(3, min(n // 4, MAX_USER_CLUSTERS))
    vecs = np.array([_style_vector(p) for p in params_list], dtype=np.float64)

    labels, _ = _kmeans(vecs, k)

    # 清空旧的用户聚类
    _user_styles = {ky: v for ky, v in _user_styles.items()
                    if not ky.startswith('u_')}  # 保留非 u_ 开头的（如果有）

    results = []
    for cluster_idx in range(k):
        mask = labels == cluster_idx
        count = int(mask.sum())
        if count == 0:
            continue

        # 聚类内参数取均值
        cluster_params_list = [params_list[i] for i in range(n) if labels[i] == cluster_idx]
        all_keys = set().union(*(p.keys() for p in cluster_params_list))
        avg: dict = {}
        for key in all_keys:
            vals = [float(p[key]) for p in cluster_params_list if key in p]
            if vals:
                v = float(np.mean(vals))
                avg[key] = round(v, 2) if key == 'Exposure' else int(round(v))

        cluster_key            = _next_cluster_key()
        cluster                = _make_cluster(avg, source='user')
        cluster['count']       = count
        cluster['source_files'] = [filenames[i] for i in range(n) if labels[i] == cluster_idx]
        _user_styles[cluster_key] = cluster
        results.append({
            'cluster_key': cluster_key,
            'name':        cluster['name'],
            'tags':        cluster['tags'],
            'desc':        cluster['desc'],
            'count':       count,
            'action':      'kmeans',
        })

    save_user_styles()
    return results


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
    """
    在用户聚类中找余弦最近邻，返回 (key, similarity)。
    当新向量本身幅度很小（<4.0）时跳过合并判断，防止近零向量之间
    产生虚假高相似度（两个不同的"微调"预设都接近零，余弦却可达 0.9+）。
    """
    qnorm = np.linalg.norm(vec)
    if qnorm < 4.0 or not _user_styles:   # 幅度太小 → 不参与合并
        return None, 0.0
    best_key, best_sim = None, -1.0
    for key, cluster in _user_styles.items():
        svec  = _style_vector(cluster['params'])
        snorm = np.linalg.norm(svec)
        if snorm < 4.0:   # 原型幅度也太小 → 跳过，避免噪声干扰
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


def add_user_preset_incremental(params: dict, filename: str) -> dict:
    """
    增量添加单个预设（新风格保护模式）。

    与 add_user_preset 的关键区别：
    · 合并阈值更高（INCREMENTAL_THRESHOLD = 0.95），只合并几乎相同的预设
    · 库满时不强制合并——新风格始终新建聚类，库容量允许动态扩展
    · 这样后续上传的少量新风格不会被已有聚类"吞没"
    """
    vec = _style_vector(params)
    closest_key, closest_sim = _find_closest_user(vec)

    if closest_key is not None and closest_sim >= INCREMENTAL_THRESHOLD:
        _merge_cluster(closest_key, params)
        action     = 'merged'
        target_key = closest_key
    else:
        # 相似度不够高 → 始终新建，不管库是否"已满"
        target_key = _next_cluster_key()
        _user_styles[target_key] = _make_cluster(params, source='user')
        action    = 'added'
        closest_sim = 0.0

    save_user_styles()
    cluster = _user_styles[target_key]
    return {
        'filename':   filename,
        'action':     action,
        'name':       cluster['name'],
        'desc':       cluster['desc'],
        'tags':       cluster['tags'],
        'count':      cluster['count'],
        'similarity': round(closest_sim, 3) if action != 'added' else None,
    }


# ─── 风格汇总 ─────────────────────────────────────────────────────────────────

def all_styles() -> dict:
    """
    返回内置模板 + 固化风格 + 会话聚类的合并字典。
    优先级：builtin < seeded < user（后者可覆盖同名 key）
    """
    merged = dict(BUILTIN_STYLES)

    def _flatten(cluster: dict, source_tag: str) -> dict:
        return {**cluster['params'],
                '_name':   cluster['name'],
                '_desc':   cluster['desc'],
                '_tags':   cluster['tags'],
                '_count':  cluster['count'],
                '_source': source_tag}

    for key, cluster in _seeded_styles.items():
        merged[key] = _flatten(cluster, 'seeded')
    for key, cluster in _user_styles.items():
        merged[key] = _flatten(cluster, 'user')
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


def decompose_seeded_styles() -> dict:
    """
    把所有 seeded_styles 分解为动作权重，写回 JSON。
    这样每个风格就成为 PCA 空间中的一个点，可被 mix_by_style_prior() 使用。
    """
    from modules.action_basis import decompose

    report = {}
    for key, cluster in _seeded_styles.items():
        params = cluster.get('params', {})
        if not params:
            continue
        weights, r2 = decompose(params)
        _seeded_styles[key]['action_weights'] = weights
        _seeded_styles[key]['action_r2'] = round(r2, 3)
        report[key] = {'weights': len(weights), 'r2': r2}

    save_user_styles()  # seeded 也存在这个文件
    return report


def get_style_action_weights(style_key: str) -> tuple:
    """
    获取某个风格的动作权重。
    返回 (weights_dict, r2)，若无权重则返回 ({}, 0.0)
    """
    style = _seeded_styles.get(style_key, {})
    weights = style.get('action_weights', {})
    r2 = style.get('action_r2', 0.0)
    return weights, r2


def find_closest_seeded_style(params: dict) -> tuple:
    """
    在 seeded_styles 中找最相近的风格。
    返回 (style_key, similarity_score, style_info)
    """
    if not _seeded_styles:
        return None, 0.0, {}

    vec = _style_vector(params)
    best_key = None
    best_sim = -1.0
    best_style = {}

    for key, style in _seeded_styles.items():
        style_params = style.get('params', {})
        if not style_params:
            continue
        style_vec = _style_vector(style_params)
        norm_v = float(np.linalg.norm(vec))
        norm_s = float(np.linalg.norm(style_vec))
        if norm_v < 0.1 or norm_s < 0.1:
            continue
        sim = float(np.dot(vec, style_vec) / (norm_v * norm_s + 1e-8))
        if sim > best_sim:
            best_sim = sim
            best_key = key
            best_style = style

    return best_key, best_sim, best_style


def list_styles() -> list:
    """返回所有可用风格的摘要列表（供前端展示）"""
    out = []
    for key, s in all_styles().items():
        if key.startswith('_'):
            continue
        raw_source = s.get('_source', '')
        if key in BUILTIN_STYLES:
            source = 'builtin'
        elif raw_source == 'seeded':
            source = 'seeded'
        else:
            source = 'user'
        out.append({
            'key':    key,
            'name':   s.get('_name', key),
            'desc':   s.get('_desc', ''),
            'tags':   s.get('_tags', []),
            'source': source,
            'count':  s.get('_count'),
        })
    return out
