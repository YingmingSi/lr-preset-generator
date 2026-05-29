"""
动作基底模块
将 LR 参数空间分解为一组语义化的原子"动作"，通过非负最小二乘（NNLS）
的线性组合还原任意风格。

核心接口：
  decompose(params)               → (weights_dict, r2_score)
  compose(weights, raw_params, r2)→ blended_params_dict
  top_actions(weights, n)         → 前端展示用的动作列表
  learn_from_uploads(params_list) → 从上传 XMP 残差学习新动作（PCA）
  get_action_info()               → 当前全部动作列表（内置+学习）
"""

import json
import os
import numpy as np

# ─── 参数空间定义 ─────────────────────────────────────────────────────────────

BUCKETS   = ['Red', 'Orange', 'Yellow', 'Green', 'Aqua', 'Blue', 'Purple', 'Magenta']
SAT_KEYS  = [f'SaturationAdjustment{b}' for b in BUCKETS]
HUE_KEYS  = [f'HueAdjustment{b}'        for b in BUCKETS]
LUM_KEYS  = [f'LuminanceAdjustment{b}'  for b in BUCKETS]
TONE_KEYS = ['Exposure', 'Contrast', 'Highlights', 'Shadows', 'Whites', 'Blacks',
             'Saturation', 'Vibrance', 'Clarity']

# 参与分解的全部参数（顺序固定 = 向量维度）共 41 维
PARAM_KEYS = SAT_KEYS + HUE_KEYS + LUM_KEYS + TONE_KEYS

_DATA_DIR          = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
_LEARNED_PATH      = os.path.join(_DATA_DIR, 'learned_actions.json')
_USER_ACTIONS_PATH = os.path.join(_DATA_DIR, 'user_actions.json')

# ─── 内置手工动作基底（12个，近似正交设计）────────────────────────────────────
# 每个动作描述"以强度 1.0 应用时"的典型参数变化
# '_label' 为前端展示名，其余字段为 LR 参数增量

BUILTIN_ACTIONS: dict = {
    'lift_shadows': {
        '_label': '提阴影',
        'Shadows': 60, 'Blacks': 28, 'Highlights': -8,
    },
    'pull_highlights': {
        '_label': '压高光',
        'Highlights': -65, 'Whites': -28, 'Shadows': 8,
    },
    'high_contrast': {
        '_label': '增对比',
        'Contrast': 60, 'Blacks': -25, 'Highlights': -18, 'Shadows': -12,
    },
    'film_fade': {
        '_label': '胶片褪色',
        'Blacks': 28, 'Contrast': -32, 'Saturation': -22, 'Vibrance': -10,
    },
    'teal_orange': {
        '_label': '青橙互补',
        'SaturationAdjustmentOrange': 75, 'SaturationAdjustmentAqua': 68,
        'SaturationAdjustmentRed':   -25, 'SaturationAdjustmentGreen': -55,
        'SaturationAdjustmentBlue':  -28, 'SaturationAdjustmentYellow': -12,
        'HueAdjustmentOrange': 10,        'HueAdjustmentAqua': -10,
    },
    'warm_shift': {
        '_label': '暖色偏移',
        'SaturationAdjustmentOrange': 48, 'SaturationAdjustmentRed':    32,
        'SaturationAdjustmentYellow': 22,
        'SaturationAdjustmentBlue':  -38, 'SaturationAdjustmentAqua': -30,
        'HueAdjustmentOrange': 8,         'HueAdjustmentRed': 5,
    },
    'cool_shift': {
        '_label': '冷色偏移',
        'SaturationAdjustmentBlue':    48, 'SaturationAdjustmentAqua':    42,
        'SaturationAdjustmentOrange': -32, 'SaturationAdjustmentRed':    -24,
        'HueAdjustmentBlue': -15,          'HueAdjustmentAqua': -8,
    },
    'portrait_skin': {
        '_label': '肤色强化',
        'SaturationAdjustmentOrange': 48, 'SaturationAdjustmentRed': 24,
        'LuminanceAdjustmentOrange':  20,
        'SaturationAdjustmentGreen': -20, 'SaturationAdjustmentAqua': -24,
        'Shadows': 24,
    },
    'vivid_nature': {
        '_label': '自然鲜亮',
        'SaturationAdjustmentGreen': 62, 'SaturationAdjustmentYellow': 40,
        'SaturationAdjustmentAqua':  34, 'SaturationAdjustmentBlue':   26,
        'Vibrance': 32,
    },
    'desaturate_global': {
        '_label': '整体去饱和',
        'Saturation': -38, 'Vibrance': -20,
        'SaturationAdjustmentRed':    -12, 'SaturationAdjustmentOrange': -12,
        'SaturationAdjustmentBlue':   -12, 'SaturationAdjustmentGreen':  -12,
    },
    'dark_moody': {
        '_label': '暗调浓郁',
        'Blacks': -52, 'Shadows': -34, 'Contrast': 62, 'Highlights': -32,
    },
    'japanese_soft': {
        '_label': '日系柔光',
        'Shadows': 60, 'Blacks': 28, 'Contrast': -45, 'Saturation': -24,
        'SaturationAdjustmentBlue':  -28, 'SaturationAdjustmentAqua':  -20,
        'SaturationAdjustmentGreen': -18, 'Highlights': -14,
    },
    # ── 13–20：弥补初始基底盲区 ─────────────────────────────────────────────
    'sky_drama': {
        '_label': '天空强化',
        # 覆盖：SatAqua/Blue ↑, LumAqua/Blue ↓, HueAqua/Blue
        'SaturationAdjustmentAqua':  55, 'SaturationAdjustmentBlue':  45,
        'LuminanceAdjustmentAqua':  -22, 'LuminanceAdjustmentBlue':  -28,
        'HueAdjustmentAqua': -8,         'HueAdjustmentBlue': -15,
        'Highlights': -15,
    },
    'add_clarity': {
        '_label': '清晰质感',
        # 覆盖：Clarity（唯一未被覆盖的 Tone 参数）
        'Clarity': 55,
    },
    'hue_warm_push': {
        '_label': '暖色色相偏橙',
        # 覆盖：HueOrange/Red/Yellow 纯色相旋转，不改变饱和度
        'HueAdjustmentOrange': +20, 'HueAdjustmentRed': +14,
        'HueAdjustmentYellow': -18,
    },
    'lum_sculpt': {
        '_label': '明度通道塑形',
        # 覆盖：所有 LUM 通道（暖色亮 / 冷色暗，增强色彩深度感）
        'LuminanceAdjustmentRed':     22, 'LuminanceAdjustmentOrange': 28,
        'LuminanceAdjustmentYellow':  20,
        'LuminanceAdjustmentGreen':  -22, 'LuminanceAdjustmentAqua':  -20,
        'LuminanceAdjustmentBlue':   -28, 'LuminanceAdjustmentPurple': -15,
        'LuminanceAdjustmentMagenta': 10,
    },
    'foliage_green': {
        '_label': '植被翠绿',
        # 覆盖：HueGreen 纯色相 + Green/Yellow 特化组合
        'SaturationAdjustmentGreen':  58, 'SaturationAdjustmentYellow': 30,
        'HueAdjustmentGreen': -18,        'HueAdjustmentYellow': -10,
        'LuminanceAdjustmentGreen':   14,
        'Vibrance': 15,
    },
    'selective_warm_pop': {
        '_label': '暖色选择保留',
        # 覆盖：暖色饱和 ↑，冷色/绿/紫 饱和大幅 ↓（选择性色彩）
        'SaturationAdjustmentOrange': +42, 'SaturationAdjustmentRed':     +32,
        'SaturationAdjustmentGreen':  -52, 'SaturationAdjustmentBlue':    -48,
        'SaturationAdjustmentAqua':   -45, 'SaturationAdjustmentPurple':  -38,
        'SaturationAdjustmentMagenta':-30,
    },
    'magenta_purple': {
        '_label': '品红紫调',
        # 覆盖：SatMagenta/Purple（之前完全为 0），HueMagenta/Purple
        'SaturationAdjustmentMagenta': 62, 'SaturationAdjustmentPurple': 50,
        'LuminanceAdjustmentMagenta':  16, 'LuminanceAdjustmentPurple':  12,
        'HueAdjustmentMagenta': -12,       'HueAdjustmentPurple': -8,
        'SaturationAdjustmentOrange': -25, 'SaturationAdjustmentYellow': -30,
    },
    'lift_whites': {
        '_label': '白色提亮',
        # 覆盖：Whites 专项（与 pull_highlights 互补；提亮高端）
        'Whites': 45, 'Highlights': 20, 'Blacks': 10,
    },
}

# 三层动作存储：
#   _user_actions   : 从上传 XMP 通过 PCA 推导，反映真实调色习惯（主力）
#   _learned_actions: 残差 PCA 发现的补充动作（辅助）
#   BUILTIN_ACTIONS : 无用户数据时的数学兜底
_user_actions:    dict = {}
_learned_actions: dict = {}

# 动作矩阵缓存 (PARAM_KEYS维度 × n_actions)
_cache_A    = None
_cache_keys = None


# ─── 内部工具 ─────────────────────────────────────────────────────────────────

def _to_vec(params: dict) -> np.ndarray:
    return np.array([float(params.get(k, 0)) for k in PARAM_KEYS], dtype=np.float64)


def _all_actions() -> dict:
    """
    动作选取优先级：
    - 有 user_actions（用户 XMP 推导）→ user_actions + learned_actions
    - 无 user_actions → BUILTIN_ACTIONS + learned_actions（兜底模式）
    """
    if _user_actions:
        return {**_user_actions, **_learned_actions}
    return {**BUILTIN_ACTIONS, **_learned_actions}


def _get_matrix() -> tuple:
    """返回 (A矩阵 [n_params × n_actions], action_keys 列表)；使用缓存。"""
    global _cache_A, _cache_keys
    actions = _all_actions()
    keys    = list(actions.keys())
    if _cache_keys == keys and _cache_A is not None:
        return _cache_A, keys
    cols = [_to_vec({k: v for k, v in actions[ak].items() if not k.startswith('_')})
            for ak in keys]
    A = np.column_stack(cols) if cols else np.zeros((len(PARAM_KEYS), 0))
    _cache_A, _cache_keys = A, keys
    return A, keys


def _invalidate_cache():
    global _cache_A, _cache_keys
    _cache_A = _cache_keys = None


# ─── NNLS（优先 scipy；退而用投影梯度）───────────────────────────────────────

def _nnls(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    try:
        from scipy.optimize import nnls as _sp
        w, _ = _sp(A, b)
        return w
    except ImportError:
        pass
    # Projected gradient descent fallback
    AtA  = A.T @ A
    Atb  = A.T @ b
    lr   = 1.0 / (np.linalg.norm(AtA) + 1e-8)
    w    = np.zeros(A.shape[1])
    for _ in range(800):
        w = np.maximum(0.0, w - lr * (AtA @ w - Atb))
    return w


# ─── 核心接口：分解 / 合成 ────────────────────────────────────────────────────

def decompose(params: dict) -> tuple:
    """
    将 LR 参数字典 NNLS 分解为非负动作权重。

    Returns:
        weights (dict): {action_key: float weight}
        r2      (float): 分解质量 0-1（越高越接近原始向量）
    """
    A, keys = _get_matrix()
    if A.shape[1] == 0:
        return {}, 0.0

    b    = _to_vec(params)
    w    = _nnls(A, b)
    b_hat = A @ w

    ss_res = float(np.sum((b - b_hat) ** 2))
    ss_tot = float(np.sum((b - b.mean()) ** 2))
    r2     = float(np.clip(1.0 - ss_res / (ss_tot + 1e-8), 0.0, 1.0))

    weights = {keys[i]: float(w[i]) for i in range(len(keys)) if w[i] > 1e-4}
    return weights, r2


def compose(weights: dict, raw_params: dict, r2: float = 0.5) -> dict:
    """
    按动作权重重建 LR 参数，并与原始分析结果自适应混合。

    混合权重（raw_alpha）基于分解质量 r2：
      r2 > 0.70 → raw 25%，动作重建 75%（动作覆盖好）
      r2 > 0.40 → 各 50%
      r2 ≤ 0.40 → raw 75%（动作覆盖弱，保留原始分析）
    """
    A, keys  = _get_matrix()
    w_vec    = np.array([weights.get(k, 0.0) for k in keys], dtype=np.float64)
    recon    = A @ w_vec
    raw_vec  = _to_vec(raw_params)

    raw_alpha = 0.25 if r2 > 0.70 else (0.50 if r2 > 0.40 else 0.75)
    blended  = raw_vec * raw_alpha + recon * (1.0 - raw_alpha)

    result = dict(raw_params)
    for i, k in enumerate(PARAM_KEYS):
        v = float(blended[i])
        if k == 'Exposure':
            result[k] = round(v, 2)
        elif abs(v) > 0.5 or k in raw_params:
            result[k] = int(round(v))
    return result


def top_actions(weights: dict, n: int = 6, min_ratio: float = 0.04) -> list:
    """
    返回权重最高的 n 个动作，含归一化占比，供前端展示。
    """
    if not weights:
        return []
    total   = sum(weights.values()) or 1.0
    actions = _all_actions()
    ranked  = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    result  = []
    for key, w in ranked[:n]:
        if w / total < min_ratio:
            break
        label = actions.get(key, {}).get('_label', key)
        result.append({
            'key':   key,
            'label': label,
            'weight': round(float(w), 3),
            'ratio':  round(float(w) / total, 3),
        })
    return result


def mix_weights_with_style_prior(analysis_weights: dict,
                                  style_weights: dict,
                                  style_alpha: float = 0.3) -> dict:
    """
    混合图像分析权重和风格先验权重。

    final_w[k] = analysis_w[k] × (1-α) + style_w[k] × α

    这样可以生成不属于 seeded_styles 本身、但在其权重空间附近的新风格。
    """
    all_keys = set(analysis_weights.keys()) | set(style_weights.keys())
    result = {}
    for key in all_keys:
        a_w = analysis_weights.get(key, 0.0)
        s_w = style_weights.get(key, 0.0)
        mixed = a_w * (1.0 - style_alpha) + s_w * style_alpha
        if abs(mixed) > 1e-4:
            result[key] = float(mixed)
    return result


# ─── 从上传 XMP 推导用户动作基底（主 PCA）────────────────────────────────────

def derive_user_actions(params_list: list,
                         n_components: int = 16,
                         min_var_ratio: float = 0.015) -> dict:
    """
    从上传的 XMP 参数中通过全量 PCA 推导用户动作基底。

    与 learn_from_uploads 的区别：
    · 这里做的是「全量 PCA」，直接用数据的主成分取代手工内置动作
    · 每个成分的量级 = 数据在该方向的标准差（真实调色习惯的幅度，非夸张值）
    · 每个成分也包含其负方向（让 NNLS 能表达反向调整）

    生成的 user_actions.json 提交到 git 后，无需重新上传即可复用。
    """
    global _user_actions

    if len(params_list) < 5:
        return {}

    X = np.array([_to_vec(p) for p in params_list], dtype=np.float64)
    n, d = X.shape
    k = min(n_components, n - 1, d)

    X_mean    = X.mean(axis=0)
    X_c       = X - X_mean
    cov       = X_c.T @ X_c / max(n - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order     = np.argsort(eigvals)[::-1]
    eigvals   = eigvals[order]
    eigvecs   = eigvecs[:, order]
    total_var = eigvals.sum()

    user_acts: dict = {}
    kept = 0
    for i in range(k):
        var_ratio = float(eigvals[i] / (total_var + 1e-12))
        if var_ratio < min_var_ratio:
            break

        comp = eigvecs[:, i].copy()

        # 量级 = 该方向的投影标准差（反映真实调整幅度）
        proj_std = float(np.std(X_c @ comp))
        if proj_std < 0.5:
            continue
        comp_scaled = comp * proj_std

        # 主导参数方向
        dom = int(np.argmax(np.abs(comp_scaled)))
        if comp_scaled[dom] < 0:
            comp_scaled = -comp_scaled

        def _make_entry(vec, suffix_label):
            entry = {}
            for j, key in enumerate(PARAM_KEYS):
                v = float(vec[j])
                if abs(v) > 0.3:
                    entry[key] = round(v, 2) if key == 'Exposure' else int(round(v))
            entry['_label']     = f'{_describe_component(vec)}({suffix_label})'
            entry['_var_ratio'] = round(var_ratio, 4)
            entry['_from_xmp']  = True
            return entry

        # 正方向
        user_acts[f'u_pca_{i}p'] = _make_entry(comp_scaled,  '+')
        # 负方向（让 NNLS 能表达反向调整）
        user_acts[f'u_pca_{i}n'] = _make_entry(-comp_scaled, '-')
        kept += 1

    if user_acts:
        _user_actions = user_acts
        _invalidate_cache()
        save_user_actions()

    return user_acts


def save_user_actions() -> None:
    os.makedirs(_DATA_DIR, exist_ok=True)
    with open(_USER_ACTIONS_PATH, 'w', encoding='utf-8') as f:
        json.dump(_user_actions, f, ensure_ascii=False, indent=2)


def load_user_actions() -> None:
    global _user_actions
    if os.path.exists(_USER_ACTIONS_PATH):
        with open(_USER_ACTIONS_PATH, 'r', encoding='utf-8') as f:
            _user_actions = json.load(f)
        _invalidate_cache()


def reset_user_actions() -> None:
    global _user_actions
    _user_actions = {}
    _invalidate_cache()
    if os.path.exists(_USER_ACTIONS_PATH):
        os.remove(_USER_ACTIONS_PATH)


# ─── 从上传 XMP 学习新动作（PCA 残差） ────────────────────────────────────────

def learn_from_uploads(params_list: list,
                        min_samples:   int   = 8,
                        max_new:       int   = 3,
                        var_threshold: float = 0.08) -> dict:
    """
    从用户上传预设的分解残差中自动发现新的正交动作。

    算法：
      1. 对每个预设做 NNLS 分解 → 计算重建残差
      2. 对残差矩阵做 PCA（协方差特征分解，无外部依赖）
      3. 方差贡献 > var_threshold 且与现有动作余弦相似度 < 0.65 的主成分
         → 自动命名后追加为学习动作
    """
    global _learned_actions

    if len(params_list) < min_samples:
        return {}

    A, _ = _get_matrix()
    vecs = np.array([_to_vec(p) for p in params_list], dtype=np.float64)

    # 1. 分解残差
    residuals = []
    for vec in vecs:
        w     = _nnls(A, vec)
        residuals.append(vec - A @ w)
    R = np.array(residuals)

    # 2. PCA（不用 sklearn，只用 numpy）
    R_c = R - R.mean(axis=0)
    cov = R_c.T @ R_c / max(len(R_c) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order   = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    total_var = eigvals.sum()
    if total_var < 1e-6:
        return {}

    # 现有动作列归一化（用于相似度检查）
    A_normed = A / (np.linalg.norm(A, axis=0, keepdims=True) + 1e-8)

    new_found   = {}
    learned_idx = 0

    for i in range(min(max_new * 3, len(eigvals))):
        if learned_idx >= max_new:
            break

        var_ratio = float(eigvals[i] / total_var)
        if var_ratio < var_threshold:
            break

        comp      = eigvecs[:, i].copy()
        comp_norm = np.linalg.norm(comp)
        if comp_norm < 1e-8:
            continue

        # 确保方向与残差均值一致（PCA 特征向量符号任意，需要对齐）
        mean_residual = R.mean(axis=0)
        if np.dot(comp, mean_residual) < 0:
            comp = -comp
        comp_norm = np.linalg.norm(comp)

        comp_n = comp / comp_norm
        # 与现有动作的最大余弦相似度
        if A_normed.shape[1] > 0 and np.abs(A_normed.T @ comp_n).max() > 0.65:
            continue   # 与已有动作重合，跳过

        # 缩放：使向量 RMS ≈ 25（与内置动作量级一致）
        rms    = float(np.sqrt(np.mean(comp ** 2))) or 1.0
        scaled = comp * (25.0 / rms)

        key   = f'learned_{learned_idx}'
        label = _describe_component(scaled)
        entry = {k: round(float(v), 1)
                 for k, v in zip(PARAM_KEYS, scaled) if abs(v) > 1.0}
        entry['_label']     = f'学习动作·{label}'
        entry['_var_ratio'] = round(var_ratio, 3)
        new_found[key]      = entry
        learned_idx += 1

    if new_found:
        _learned_actions.update(new_found)
        _invalidate_cache()
        save_learned()

    return new_found


def _describe_component(vec: np.ndarray) -> str:
    """根据主导参数自动描述学习到的动作"""
    name_map = {
        'Shadows': '阴影',   'Highlights': '高光', 'Blacks': '黑色',   'Whites': '白色',
        'Contrast': '对比度', 'Saturation': '饱和度', 'Vibrance': '自然饱和', 'Exposure': '曝光',
        'SaturationAdjustmentOrange': '橙饱和', 'SaturationAdjustmentAqua':  '青饱和',
        'SaturationAdjustmentBlue':   '蓝饱和', 'SaturationAdjustmentGreen': '绿饱和',
        'SaturationAdjustmentRed':    '红饱和', 'SaturationAdjustmentYellow':'黄饱和',
        'HueAdjustmentOrange': '橙色相',        'HueAdjustmentAqua':   '青色相',
    }
    params = dict(zip(PARAM_KEYS, vec))
    top    = sorted(params.items(), key=lambda x: abs(x[1]), reverse=True)
    parts  = []
    for k, v in top[:3]:
        if abs(v) < 3:
            break
        cn = name_map.get(k, k.replace('SaturationAdjustment', '').replace('HueAdjustment', ''))
        parts.append(('提' if v > 0 else '压') + cn)
    return '+'.join(parts[:2]) or '混合'


# ─── 持久化 ───────────────────────────────────────────────────────────────────

def load_learned() -> None:
    """加载残差 PCA 补充动作 + 用户 PCA 主动作"""
    global _learned_actions
    if os.path.exists(_LEARNED_PATH):
        with open(_LEARNED_PATH, 'r', encoding='utf-8') as f:
            _learned_actions = json.load(f)
        _invalidate_cache()
    load_user_actions()


def save_learned() -> None:
    os.makedirs(_DATA_DIR, exist_ok=True)
    with open(_LEARNED_PATH, 'w', encoding='utf-8') as f:
        json.dump(_learned_actions, f, ensure_ascii=False, indent=2)


def reset_learned() -> None:
    global _learned_actions
    _learned_actions = {}
    _invalidate_cache()
    if os.path.exists(_LEARNED_PATH):
        os.remove(_LEARNED_PATH)


def get_action_info() -> list:
    """返回所有动作的概要信息（三层），供前端展示"""
    actions = _all_actions()
    result  = []
    for k, a in actions.items():
        if _user_actions and k in _user_actions:
            tier = 'user'    # XMP 推导
        elif k in BUILTIN_ACTIONS:
            tier = 'builtin'
        else:
            tier = 'learned'
        result.append({
            'key':       k,
            'label':     a.get('_label', k),
            'tier':      tier,
            'var_ratio': a.get('_var_ratio'),
            'from_xmp':  bool(a.get('_from_xmp')),
        })
    return result


def has_user_actions() -> bool:
    return bool(_user_actions)
