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

MAX_USER_CLUSTERS      = 8    # K-means 聚类上限（只保留核心风格）
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
    同时识别正向（提升）和负向（压制）的参数特征。
    适配真实调色习惯：压饱和、压高光等负向操作同样产生有意义的标签。
    """
    sat = {b: float(params.get(f'SaturationAdjustment{b}', 0)) for b in BUCKETS}
    highlights  = float(params.get('Highlights', 0))
    shadows     = float(params.get('Shadows',    0))
    blacks      = float(params.get('Blacks',     0))
    contrast    = float(params.get('Contrast',   0))
    overall_sat = float(params.get('Saturation', 0))

    # 各通道的正向/负向强度
    warm_up   = max(sat['Orange'], sat['Red'],  sat['Yellow'], 0)
    cool_up   = max(sat['Aqua'],   sat['Blue'], sat['Purple'], 0)
    warm_down = max(-sat['Orange'],-sat['Red'], -sat['Yellow'], 0)
    cool_down = max(-sat['Aqua'],  -sat['Blue'], 0)
    green_down= max(-sat['Green'], 0)

    tags = []

    # ── 色调方向（HSL 和 SplitToning 合并判断）──────────────────────────
    sh_sat = float(params.get('SplitToningShadowSaturation',    0))
    sh_hue = float(params.get('SplitToningShadowHue',           0))
    hl_sat = float(params.get('SplitToningHighlightSaturation', 0))
    hl_hue = float(params.get('SplitToningHighlightHue',        0))

    def _is_orange(h): return 15 <= h <= 65
    def _is_blue(h):   return 165 <= h <= 260
    def _is_warm(h):   return 15 <= h <= 65 or 320 <= h <= 360
    def _is_cool(h):   return 155 <= h <= 265

    st_shadow_cool  = _is_cool(sh_hue)  and sh_sat >= 4
    st_shadow_warm  = _is_warm(sh_hue)  and sh_sat >= 4
    st_hl_warm      = _is_warm(hl_hue)  and hl_sat >= 3
    st_hl_cool      = _is_cool(hl_hue)  and hl_sat >= 3
    st_hl_orange    = _is_orange(hl_hue) and hl_sat >= 3

    # 青橙：冷色阴影 + 暖/橙色高光（最经典的 SplitToning 配色）
    if st_shadow_cool and (st_hl_warm or st_hl_orange):
        tags.append('青橙')
    # HSL 饱和度正向的青橙
    elif sat['Orange'] > 10 and sat['Aqua'] > 8:
        tags.append('青橙')
    # 单方向 SplitToning
    elif st_shadow_cool or st_hl_cool:
        tags.append('冷调')
    elif st_shadow_warm or st_hl_warm:
        tags.append('暖色')
    # 纯 HSL 方向判断（SplitToning 为零时）
    elif warm_up > 10:
        tags.append('暖色')
    elif cool_up > 8:
        tags.append('冷调')
    elif warm_down > cool_down + 12 and warm_down > 15:
        tags.append('冷感')
    elif cool_down > warm_down + 12 and cool_down > 15:
        tags.append('暖感')

    # ── 色彩抑制特征 ────────────────────────────────────────────────────
    if green_down > 35:
        tags.append('压绿')
    if sat['Yellow'] < -30 and sat['Orange'] < -20:
        tags.append('压黄橙')
    if sat['Blue'] < -25 and sat['Aqua'] < -18:
        tags.append('压蓝青')
    if sat['Red'] < -20:
        tags.append('压红')

    # ── 影调 ────────────────────────────────────────────────────────────
    if highlights < -55:
        tags.append('强压高光')
    elif highlights < -25:
        tags.append('压高光')

    if blacks < -25 and contrast > 20:
        tags.append('电影感')
    elif blacks > 12 or (shadows > 20 and contrast < -10):
        tags.append('胶片感')

    if shadows > 30 and contrast < -15:
        tags.append('日系')

    # ── 反差 ────────────────────────────────────────────────────────────
    if contrast > 35:
        tags.append('高对比')
    elif contrast < -20 or (highlights < -30 and shadows > 5):
        tags.append('低反差')

    # ── 全局饱和度 ────────────────────────────────────────────────────
    neg_heavy = sum(1 for v in sat.values() if v < -18)
    pos_heavy = sum(1 for v in sat.values() if v > 15)
    if overall_sat < -8 or neg_heavy >= 4:
        tags.append('低饱和')
    elif overall_sat > 12 or pos_heavy >= 3:
        tags.append('高饱和')

    if not tags:
        tags.append('中性')

    return tags


def _auto_name(params: dict) -> str:
    """
    根据参数特征生成描述性名称，覆盖正向和负向调整。
    生成的名称不保证全局唯一——batch_cluster 负责在结果中去重。
    """
    tags = _auto_tags(params)
    hl    = float(params.get('Highlights', 0))
    ct    = float(params.get('Contrast',   0))
    bk    = float(params.get('Blacks',     0))
    sh    = float(params.get('Shadows',    0))
    sat_o = float(params.get('SaturationAdjustmentOrange', 0))
    sat_g = float(params.get('SaturationAdjustmentGreen',  0))
    sat_b = float(params.get('SaturationAdjustmentBlue',   0))
    sat_y = float(params.get('SaturationAdjustmentYellow', 0))
    sat_a = float(params.get('SaturationAdjustmentAqua',   0))

    # ── 色调方向明确时直接命名 ─────────────────────────────────────────
    if '青橙'   in tags:
        # 区分"纯提橙"与"压蓝绿+提橙"
        if sat_o > 15 and sat_g < -20:  return '青橙去绿'
        if sat_o > 15:                  return '青橙'
        if sat_g < -50 and sat_b < -40: return '青橙极压冷色'
        return '青橙'
    if '暗调电影' in tags or ('电影感' in tags and '暖色' not in tags):
        if bk < -35: return '深黑电影'
        return '暗调电影'
    if '暖调电影' in tags or ('电影感' in tags and '暖色' in tags):
        return '暖调电影'
    if '日系'   in tags:                return '日系清新'
    if '胶片感' in tags and '暖色' in tags: return '暖调胶片'
    if '胶片感' in tags:                return '胶片复古'
    if '冷调'   in tags:
        if sat_b < -40: return '冷调压蓝'
        if sat_g < -30: return '冷调去绿'
        return '冷调'
    if '暖色'   in tags and '高饱和' in tags: return '暖色鲜艳'
    if '暖色'   in tags:                return '暖调'

    # ── 从最显著的参数组合生成名称 ──────────────────────────────────
    # 影调词
    tone = ''
    if hl < -75:    tone = '极压高光'
    elif hl < -55:  tone = '强压高光'
    elif hl < -30:  tone = '压高光'
    elif ct > 30 and bk < -25: tone = '高对比'
    elif ct < -25 or (sh > 30 and bk > 10): tone = '低反差'

    # 色彩词：通道幅度优先，没有显著通道才用冷感/暖感
    max_chan = max(abs(sat_g), abs(sat_b), abs(sat_o), abs(sat_y), abs(sat_a))
    color_scores = [
        ('极压绿蓝', sat_g < -60 and sat_b < -60),
        ('压绿蓝',   sat_g < -35 and sat_b < -25),
        ('主压绿',   sat_g < -50 and abs(sat_g) > abs(sat_b) * 1.5),
        ('主压蓝',   sat_b < -40 and abs(sat_b) > abs(sat_g) * 1.5),
        ('压橙黄',   sat_o < -30 and sat_y < -20),
        ('压橙',     sat_o < -35),
        # 当没有显著单通道时，才用冷暖感描述
        ('暖感',     '暖感' in tags and max_chan < 30),
        ('冷感',     '冷感' in tags and max_chan < 30),
    ]
    color = next((k for k, v in color_scores if v), '')

    # 深黑：允许补充极压高光（BK 极深时区分两个极压高光簇）
    depth = ''
    if bk <= -35 and tone == '极压高光' and not color:
        color = '深黑'   # 极压高光·深黑
    elif bk <= -30 and tone not in ('强压高光', '高对比') and not color:
        depth = '深黑'   # 压高光·深黑 / 深黑（主词）

    parts = [p for p in [tone, color or depth] if p]
    if len(parts) == 2:
        return '·'.join(parts)
    if len(parts) == 1:
        return parts[0]
    return '中性平调'


# ─── 内置经典风格模板（已移除，改由用户 XMP 上传生成 seeded_styles）────────
# 保留空字典占位，供 action_basis 兜底时引用

BUILTIN_STYLES: dict = {}  # 已移除内置风格

# ─── 20个命名风格签名（方向特征，用于聚类命名匹配）──────────────────────────
# 每条只列出最能区分该风格的几个关键参数；_style_vector 会把它们投影到统一特征空间
# SplitToning 参数用真实 XMP 键名，_style_vector 内部做 cos/sin 编码
STYLE_SIGNATURES: dict = {
    '青橙':     {'SplitToningShadowHue': 210, 'SplitToningShadowSaturation': 15,
                 'SplitToningHighlightHue': 38, 'SplitToningHighlightSaturation': 10,
                 'SaturationAdjustmentGreen': -25, 'SaturationAdjustmentPurple': -30},
    '黑金':     {'Blacks': -45, 'Contrast': 60,
                 'SaturationAdjustmentGreen': -80, 'SaturationAdjustmentAqua': -75,
                 'SaturationAdjustmentBlue': -60, 'SaturationAdjustmentPurple': -70,
                 'SaturationAdjustmentMagenta': -65, 'SaturationAdjustmentOrange': 20},
    '日系小清新': {'Shadows': 45, 'Blacks': 18, 'Contrast': -40,
                  'Saturation': -15, 'SaturationAdjustmentGreen': 8},
    '胶片复古':  {'Blacks': 22, 'SaturationAdjustmentOrange': 15,
                  'SaturationAdjustmentGreen': -20, 'SaturationAdjustmentBlue': -18,
                  'Contrast': 12},
    '暗调浓郁':  {'Blacks': -50, 'Contrast': 65, 'Highlights': -40, 'Shadows': -20,
                  'Saturation': -10},
    '清新自然':  {'SaturationAdjustmentGreen': 28, 'SaturationAdjustmentAqua': 18,
                  'Shadows': 15, 'Highlights': -12},
    '暖阳落日':  {'SaturationAdjustmentOrange': 35, 'SaturationAdjustmentYellow': 25,
                  'SaturationAdjustmentBlue': -38, 'SaturationAdjustmentAqua': -28},
    '冷调都市':  {'SaturationAdjustmentBlue': 32, 'SaturationAdjustmentAqua': 20,
                  'SaturationAdjustmentOrange': -22, 'Contrast': 35, 'Blacks': -18},
    '梦幻柔光':  {'Blacks': 20, 'SaturationAdjustmentPurple': 22,
                  'SaturationAdjustmentMagenta': 18, 'Contrast': -35, 'Shadows': 20},
    '翡翠森林':  {'SaturationAdjustmentGreen': 35, 'SaturationAdjustmentAqua': 22,
                  'Blacks': -22, 'Contrast': 18},
    '人像肤色':  {'SaturationAdjustmentOrange': 22, 'SaturationAdjustmentRed': 12,
                  'SaturationAdjustmentBlue': -12, 'Highlights': -18, 'Shadows': 10},
    '落日金光':  {'SaturationAdjustmentOrange': 45, 'SaturationAdjustmentYellow': 35,
                  'SaturationAdjustmentBlue': -50, 'SaturationAdjustmentAqua': -40,
                  'SaturationAdjustmentGreen': -30},
    '霓虹赛博':  {'SaturationAdjustmentPurple': 35, 'SaturationAdjustmentMagenta': 28,
                  'SaturationAdjustmentBlue': 22, 'Blacks': -38, 'Contrast': 45},
    '哑光褪色':  {'Blacks': 28, 'Contrast': -32, 'Saturation': -22, 'Clarity': -18,
                  'SaturationAdjustmentGreen': -15},
    '高调清透':  {'Whites': 30, 'Shadows': 22, 'Contrast': -38,
                  'Saturation': -18, 'Highlights': -8},
    '戏剧人像':  {'SaturationAdjustmentOrange': 22, 'Blacks': -42, 'Contrast': 55,
                  'SaturationAdjustmentBlue': -22, 'SaturationAdjustmentGreen': -25},
    '旅行鲜艳':  {'Saturation': 22, 'Vibrance': 18,
                  'SaturationAdjustmentGreen': 22, 'SaturationAdjustmentOrange': 18,
                  'SaturationAdjustmentBlue': 15},
    '北欧冬日':  {'SaturationAdjustmentBlue': 18, 'SaturationAdjustmentAqua': 12,
                  'SaturationAdjustmentOrange': -25, 'Contrast': -28,
                  'Saturation': -12, 'Whites': 15},
    '秋叶暖色':  {'SaturationAdjustmentOrange': 28, 'SaturationAdjustmentRed': 22,
                  'SaturationAdjustmentYellow': 18, 'SaturationAdjustmentGreen': -32,
                  'SaturationAdjustmentBlue': -15},
    '都市青调':  {'SaturationAdjustmentAqua': 28, 'SaturationAdjustmentBlue': 18,
                  'SaturationAdjustmentOrange': -18, 'Contrast': 22,
                  'SplitToningShadowHue': 185, 'SplitToningShadowSaturation': 8},
}


def _match_named_style(params: dict) -> str:
    """
    将聚类质心参数与 STYLE_SIGNATURES 做余弦匹配，返回最近的风格名称。
    始终返回一个名称（无阈值拒绝），保证聚类都有人类可读标签。
    """
    vec = _style_vector(params)
    norm_v = float(np.linalg.norm(vec))

    best_name = '自然调色'
    best_sim  = -1.0
    for name, sig in STYLE_SIGNATURES.items():
        sig_vec = _style_vector(sig)
        norm_s  = float(np.linalg.norm(sig_vec))
        if norm_v < 0.5 or norm_s < 0.5:
            continue
        sim = float(np.dot(vec, sig_vec) / (norm_v * norm_s + 1e-8))
        if sim > best_sim:
            best_sim  = sim
            best_name = name

    return best_name


# ─── 运行时存储 ───────────────────────────────────────────────────────────────
# _seeded_styles : 已固化的风格（committed to git，永久可用）
# _user_styles   : 当前 session 上传的实验聚类（可随时 Reset）
_seeded_styles: dict = {}
_user_styles:   dict = {}

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


def save_seeded_styles() -> None:
    """将 _seeded_styles（含 action_weights）写入 seeded_styles.json"""
    os.makedirs(os.path.dirname(_SEEDED_PATH), exist_ok=True)
    with open(_SEEDED_PATH, 'w', encoding='utf-8') as f:
        json.dump(_seeded_styles, f, ensure_ascii=False, indent=2)


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

    # k ≈ sqrt(n/2)，保证每组样本量充足、风格鲜明，上限 MAX_USER_CLUSTERS
    k = max(3, min(round((n / 2) ** 0.5), MAX_USER_CLUSTERS))
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

        cluster_key             = _next_cluster_key()
        cluster                 = _make_cluster(avg, source='user')
        cluster['count']        = count
        cluster['source_files'] = [filenames[i] for i in range(n) if labels[i] == cluster_idx]
        _user_styles[cluster_key] = cluster
        results.append({
            'cluster_key': cluster_key,
            'params':      avg,
            'count':       count,
        })

    # ── 聚类名称去重（同名加数字后缀）────────────────────────────────────
    name_count: dict = {}
    for r in results:
        key = r['cluster_key']
        raw_name = _user_styles[key]['name']
        name_count[raw_name] = name_count.get(raw_name, 0) + 1

    name_seen: dict = {}
    for r in results:
        key = r['cluster_key']
        raw_name = _user_styles[key]['name']
        if name_count[raw_name] > 1:
            name_seen[raw_name] = name_seen.get(raw_name, 0) + 1
            unique_name = f'{raw_name}{name_seen[raw_name]}'
            _user_styles[key]['name'] = unique_name
            # desc 也同步
            _user_styles[key]['desc'] = _user_styles[key]['desc'] or unique_name

        r['name'] = _user_styles[key]['name']
        r['tags'] = _user_styles[key]['tags']
        r['desc'] = _user_styles[key]['desc']
        r['action'] = 'kmeans'

    save_user_styles()
    return results


# ─── 聚类核心逻辑 ────────────────────────────────────────────────────────────

def _make_cluster(params: dict, source: str = 'user') -> dict:
    """从参数字典创建一个新聚类原型"""
    tags = _auto_tags(params)
    return {
        'name':   _match_named_style(params),   # 匹配到最近的已知风格名称
        'desc':   ' · '.join(tags),
        'tags':   tags,
        'count':  1,
        'source': source,
        'params': {k: params[k] for k in (SAT_KEYS + HUE_KEYS + LUM_KEYS +
                   ['Contrast','Shadows','Highlights','Blacks','Whites','Saturation','Vibrance',
                    'SplitToningShadowHue','SplitToningShadowSaturation',
                    'SplitToningHighlightHue','SplitToningHighlightSaturation',
                    'SplitToningBalance'])
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
    cluster['name']  = _match_named_style(merged)


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
    # 同时避开 _user_styles 和 _seeded_styles 中的已有 key
    all_keys = set(_user_styles.keys()) | set(_seeded_styles.keys())
    idx = 0
    while f'u_{idx}' in all_keys:
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
    返回固化风格 + 会话聚类的合并字典（不含手工内置风格）。
    内置风格仅保留在 BUILTIN_STYLES 常量供动作分解兜底使用，不展示给用户。
    """
    def _flatten(cluster: dict, source_tag: str) -> dict:
        return {**cluster.get('params', {}),
                '_name':   cluster['name'],
                '_desc':   cluster.get('desc', ''),
                '_tags':   cluster.get('tags', []),
                '_count':  cluster.get('count', 1),
                '_source': source_tag}

    merged = {}
    for key, cluster in _seeded_styles.items():
        merged[key] = _flatten(cluster, 'seeded')
    for key, cluster in _user_styles.items():
        merged[key] = _flatten(cluster, 'user')
    return merged


# ─── 风格匹配 ─────────────────────────────────────────────────────────────────

def _style_vector(params: dict) -> np.ndarray:
    """
    提取风格特征向量（8+8+5 维 HSL/影调 + 4 维 SplitToning 笛卡尔分量）。
    SplitToning 对颜色方向贡献最大，给予较高权重（×3），
    使 K-means 主要按颜色特征分组。
    """
    rad = np.pi / 180.0
    sat = np.array([params.get(k, 0) for k in SAT_KEYS], dtype=np.float32)
    hue = np.array([params.get(k, 0) for k in HUE_KEYS], dtype=np.float32) * 0.3
    lum = np.array([
        params.get('Contrast',   0) * 0.6,
        params.get('Shadows',    0) * 0.4,
        params.get('Highlights', 0) * 0.4,
        params.get('Blacks',     0) * 0.4,
        params.get('Saturation', 0) * 0.5,
    ], dtype=np.float32)
    # SplitToning 色调分离（笛卡尔编码，量级 ≈ HSL 参数，权重 ×3 以主导色彩分组）
    sh_sat = float(params.get('SplitToningShadowSaturation',    0))
    sh_hue = float(params.get('SplitToningShadowHue',           0))
    hl_sat = float(params.get('SplitToningHighlightSaturation', 0))
    hl_hue = float(params.get('SplitToningHighlightHue',        0))
    st = np.array([
        sh_sat * np.cos(sh_hue * rad) * 3,
        sh_sat * np.sin(sh_hue * rad) * 3,
        hl_sat * np.cos(hl_hue * rad) * 3,
        hl_sat * np.sin(hl_hue * rad) * 3,
    ], dtype=np.float32)
    return np.concatenate([sat, hue, lum, st])


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


def rename_seeded_styles() -> dict:
    """
    用最新的 _auto_name / _auto_tags 对所有 seeded_styles 重新命名，
    并在结果中去重（同名加数字后缀），然后写入磁盘。
    返回 {key: new_name} 的映射。
    """
    if not _seeded_styles:
        return {}

    # 第一遍：生成每个风格的新名称和标签
    proposals: dict[str, str] = {}
    for key, cluster in _seeded_styles.items():
        params = cluster.get('params', {})
        name   = _match_named_style(params)   # 匹配到已知风格名称
        tags   = _auto_tags(params)
        desc   = ' · '.join(tags)
        proposals[key] = name
        _seeded_styles[key]['name'] = name
        _seeded_styles[key]['tags'] = tags
        _seeded_styles[key]['desc'] = desc

    # 第二遍：去重（同名加数字后缀）
    name_count: dict[str, int] = {}
    for name in proposals.values():
        name_count[name] = name_count.get(name, 0) + 1

    name_seen: dict[str, int] = {}
    result: dict[str, str] = {}
    for key, name in proposals.items():
        if name_count[name] > 1:
            name_seen[name] = name_seen.get(name, 0) + 1
            unique = f'{name}{name_seen[name]}'
            _seeded_styles[key]['name'] = unique
            result[key] = unique
        else:
            result[key] = name

    save_seeded_styles()
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

    save_seeded_styles()   # action_weights 写入 seeded_styles.json，重启后仍有效
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
        source = 'seeded' if raw_source == 'seeded' else 'user'
        out.append({
            'key':    key,
            'name':   s.get('_name', key),
            'desc':   s.get('_desc', ''),
            'tags':   s.get('_tags', []),
            'source': source,
            'count':  s.get('_count'),
        })
    return out
