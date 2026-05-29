"""
参数校准模块
从用户上传的 XMP 文件中学习参数分布，
对未来生成的参数值做"软限制"（超出 P5-P95 区间时向边界收缩 60%）。

使用流程：
  1. 用户上传自己的 XMP 预设集合 → update_from_params_list()
  2. 后续 /analyze 调用 apply_calibration() 对生成值做约束
  3. 校准数据持久化到 data/calibration.json
"""

import json
import os
import numpy as np

_DATA_DIR   = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
_CALIB_PATH = os.path.join(_DATA_DIR, 'calibration.json')

# 运行时校准状态：{param: {lo, hi, mean, count}}
_calib: dict = {}

# 各参数的内置默认范围（无用户校准时作为兜底）
_DEFAULTS: dict = {
    'Exposure':           (-1.0,  1.5),
    'Contrast':           (-60,   70),
    'Highlights':         (-100,  30),
    'Shadows':            (-30,   70),
    'Whites':             (-55,   40),
    'Blacks':             (-50,   40),
    'Saturation':         (-65,   65),
    'Vibrance':           (-50,   85),
    'Clarity':            (-20,   50),
    # HSL sat
    **{f'SaturationAdjustment{b}': (-95, 95)
       for b in ('Red','Orange','Yellow','Green','Aqua','Blue','Purple','Magenta')},
    # HSL hue
    **{f'HueAdjustment{b}':        (-45, 45)
       for b in ('Red','Orange','Yellow','Green','Aqua','Blue','Purple','Magenta')},
    # HSL lum
    **{f'LuminanceAdjustment{b}':  (-65, 65)
       for b in ('Red','Orange','Yellow','Green','Aqua','Blue','Purple','Magenta')},
    # 校准面板
    'RedSaturation':   (-50,  60),
    'BlueSaturation':  (-40,  55),
    'GreenSaturation': (-30,  30),
    'ShadowTint':      (-25,  25),
}

# 内置默认幅度参考（用于前端展示"幅度调节"效果）
_DEFAULT_MULTIPLIERS: dict = {
    'Exposure':   1.0,
    'Contrast':   1.0,
    'Highlights': 1.0,
    'Shadows':    1.0,
    'Whites':     1.0,
    'Blacks':     1.0,
    'Saturation': 1.0,
    'Vibrance':   1.0,
}


# ─── 持久化 ───────────────────────────────────────────────────────────────────

def load_calibration() -> None:
    global _calib
    if os.path.exists(_CALIB_PATH):
        with open(_CALIB_PATH, 'r', encoding='utf-8') as f:
            _calib = json.load(f)


def save_calibration() -> None:
    os.makedirs(_DATA_DIR, exist_ok=True)
    with open(_CALIB_PATH, 'w', encoding='utf-8') as f:
        json.dump(_calib, f, ensure_ascii=False, indent=2)


def is_calibrated() -> bool:
    return len(_calib) > 0


# ─── 从 XMP 参数列表更新校准 ──────────────────────────────────────────────────

def update_from_params_list(params_list: list) -> dict:
    """
    从已解析的 XMP params dict 列表中学习参数分布。
    使用 P5/P95 作为合理范围边界；已有校准时增量合并（按样本数加权）。

    Returns: 受影响参数的统计摘要（供前端展示）
    """
    global _calib

    # 按 key 收集所有值
    buckets: dict = {}
    for params in params_list:
        for key, val in params.items():
            if not isinstance(val, (int, float)):
                continue
            buckets.setdefault(key, []).append(float(val))

    report = {}
    for key, new_vals in buckets.items():
        if len(new_vals) < 2:
            continue

        arr = np.array(new_vals)
        new_lo, new_p50, new_hi = (float(v) for v in np.percentile(arr, [5, 50, 95]))
        n_new = len(new_vals)

        if key in _calib:
            # 增量合并：加权平均 lo/hi/mean
            old  = _calib[key]
            n_old = old.get('count', 1)
            total = n_old + n_new
            merged_lo  = (old['lo']   * n_old + new_lo  * n_new) / total
            merged_hi  = (old['hi']   * n_old + new_hi  * n_new) / total
            merged_mean= (old['mean'] * n_old + new_p50 * n_new) / total
            _calib[key] = {
                'lo':    round(float(merged_lo),   2),
                'hi':    round(float(merged_hi),   2),
                'mean':  round(float(merged_mean), 2),
                'count': int(total),
            }
        else:
            _calib[key] = {
                'lo':    round(new_lo,  2),
                'hi':    round(new_hi,  2),
                'mean':  round(new_p50, 2),
                'count': n_new,
            }

        report[key] = _calib[key]

    save_calibration()
    return report


# ─── 应用校准到生成参数 ───────────────────────────────────────────────────────

def _get_range(key: str) -> tuple:
    """返回参数的 (lo, hi) 校准范围；无校准时用默认值"""
    if key in _calib:
        return _calib[key]['lo'], _calib[key]['hi']
    if key in _DEFAULTS:
        return _DEFAULTS[key]
    return None, None


def apply_calibration(params: dict) -> dict:
    """
    对生成的参数做软限制：
    · 在校准范围 [lo, hi] 内 → 不变
    · 超出范围 → 向边界收缩 60%（保留 40% 的超量）
      例：hi=85, val=110 → result = 85 + (110-85)*0.4 = 95
    · 整数参数保持整数
    """
    if not _calib and not _DEFAULTS:
        return params

    result = dict(params)
    for key, val in params.items():
        if not isinstance(val, (int, float)):
            continue
        lo, hi = _get_range(key)
        if lo is None:
            continue

        fval = float(val)
        if fval < lo:
            fval = lo + (fval - lo) * 0.4   # 向 lo 收缩，保留 40% 超量
        elif fval > hi:
            fval = hi + (fval - hi) * 0.4   # 向 hi 收缩，保留 40% 超量

        result[key] = int(round(fval)) if isinstance(val, int) else round(fval, 3)

    return result


# ─── 校准状态摘要 ─────────────────────────────────────────────────────────────

def get_summary() -> dict:
    """返回当前校准状态摘要，供前端展示"""
    if not _calib:
        return {
            'calibrated':   False,
            'xmp_count':    0,
            'param_count':  0,
            'key_ranges':   {},
        }

    xmp_count = max((v.get('count', 0) for v in _calib.values()), default=0)

    # 展示几个最具代表性的参数范围
    key_params = [
        'Exposure', 'Contrast', 'Highlights', 'Shadows',
        'Saturation', 'Vibrance',
        'SaturationAdjustmentOrange', 'SaturationAdjustmentAqua',
        'SaturationAdjustmentBlue',   'SaturationAdjustmentGreen',
    ]
    key_ranges = {}
    for k in key_params:
        if k in _calib:
            key_ranges[k] = {'lo': _calib[k]['lo'], 'hi': _calib[k]['hi']}

    return {
        'calibrated':   True,
        'xmp_count':    xmp_count,
        'param_count':  len(_calib),
        'key_ranges':   key_ranges,
    }


def reset_calibration() -> None:
    """清除所有校准数据（保留默认范围作为兜底）"""
    global _calib
    _calib = {}
    if os.path.exists(_CALIB_PATH):
        os.remove(_CALIB_PATH)
