"""
参数配置 — 全项目单一数据源（Single Source of Truth）

所有模块（生成器、归一化器、模型、后端）都从这里导入参数定义，
避免 72 维参数在多个文件中各自维护导致不一致。

72 维参数分为 5 组（与课程学习阶段对应）：
  S1 亮度     (11): Exposure ... Saturation
  S2 色调曲线 (20): Luma/Red/Green/Blue × 5 点
  S3 颜色分级 (11): 阴影/中间调/高光 × H/S/L + Balance + Blending
  S4 校准      (6): 红/绿/蓝原色 × 色相/饱和度
  S5 HSL      (24): 色相/饱和度/明度 × 8 色
"""

# ─── 8 个 LR HSL 颜色 ─────────────────────────────────────────────────────
HSL_COLORS = ['Red', 'Orange', 'Yellow', 'Green', 'Aqua', 'Blue', 'Purple', 'Magenta']

# 各颜色在 HSV hue 轴上的中心和半宽（用于处理器的软掩码）
# hue 归一化到 [0, 1]
HSL_COLOR_HUE = {
    'Red':     (0.97, 1.03),   # 跨 0 边界，特殊处理
    'Orange':  (0.04, 0.11),
    'Yellow':  (0.12, 0.20),
    'Green':   (0.22, 0.42),
    'Aqua':    (0.42, 0.55),
    'Blue':    (0.55, 0.72),
    'Purple':  (0.72, 0.85),
    'Magenta': (0.85, 0.95),
}

# ─── 参数分组（顺序即 CNN 输出顺序，也是课程阶段顺序）───────────────────────

GROUP_LUMINANCE = [
    'Exposure', 'Contrast', 'Highlights', 'Shadows', 'Blacks', 'Whites',
    'Texture', 'Clarity', 'Dehaze', 'Vibrance', 'Saturation',
]

GROUP_CURVE = (
    [f'LumaCurve{i}'  for i in range(5)] +
    [f'RedCurve{i}'   for i in range(5)] +
    [f'GreenCurve{i}' for i in range(5)] +
    [f'BlueCurve{i}'  for i in range(5)]
)

GROUP_COLORGRADE = [
    'ColorGradeShadowHue',    'ColorGradeShadowSat',    'ColorGradeShadowLum',
    'ColorGradeMidtoneHue',   'ColorGradeMidtoneSat',   'ColorGradeMidtoneLum',
    'ColorGradeHighlightHue', 'ColorGradeHighlightSat', 'ColorGradeHighlightLum',
    'ColorGradeBalance', 'ColorGradeBlending',
]

GROUP_CALIBRATION = [
    'RedHue', 'RedSaturation',
    'GreenHue', 'GreenSaturation',
    'BlueHue', 'BlueSaturation',
]

GROUP_HSL = (
    [f'HueAdjustment{c}'        for c in HSL_COLORS] +
    [f'SaturationAdjustment{c}' for c in HSL_COLORS] +
    [f'LuminanceAdjustment{c}'  for c in HSL_COLORS]
)

# 完整 72 维参数顺序
PARAM_ORDER = (
    GROUP_LUMINANCE +    # 0-10   (11)
    GROUP_CURVE +        # 11-30  (20)
    GROUP_COLORGRADE +   # 31-41  (11)
    GROUP_CALIBRATION +  # 42-47  (6)
    GROUP_HSL            # 48-71  (24)
)

assert len(PARAM_ORDER) == 72, f"参数数应为 72，实际 {len(PARAM_ORDER)}"

# ─── 课程学习阶段（累积索引）─────────────────────────────────────────────
# 每个 stage 训练时，loss 只在累积到该 stage 的参数上计算
CURRICULUM_STAGES = [
    ('S1_luminance',   GROUP_LUMINANCE),
    ('S2_curve',       GROUP_LUMINANCE + GROUP_CURVE),
    ('S3_colorgrade',  GROUP_LUMINANCE + GROUP_CURVE + GROUP_COLORGRADE),
    ('S4_calibration', GROUP_LUMINANCE + GROUP_CURVE + GROUP_COLORGRADE + GROUP_CALIBRATION),
    ('S5_hsl',         PARAM_ORDER),  # 全部 72
]


def stage_mask(stage_idx: int):
    """返回某 stage 的 0/1 掩码（长度 72），1 = 该参数参与 loss"""
    active = set(CURRICULUM_STAGES[stage_idx][1])
    return [1.0 if p in active else 0.0 for p in PARAM_ORDER]


# ─── 采样范围（基于用户实际经验）─────────────────────────────────────────
# (lo, hi)；色相类是 0-360 角度，其余多为对称区间
PARAM_RANGES = {
    # 亮度 — 基础调整幅度较小
    'Exposure':   (-1.5, 1.5),
    'Contrast':   (-30, 30),
    'Highlights': (-100, 100),
    'Shadows':    (-100, 100),
    'Blacks':     (-100, 100),
    'Whites':     (-100, 100),
    'Texture':    (-30, 30),
    'Clarity':    (-20, 20),
    'Dehaze':     (-20, 20),
    'Vibrance':   (-60, 60),
    'Saturation': (-60, 60),
}
# 色调曲线 5 点 y 偏移：±10
for _p in GROUP_CURVE:
    PARAM_RANGES[_p] = (-10, 10)
# 颜色分级
for _zone in ['Shadow', 'Midtone', 'Highlight']:
    PARAM_RANGES[f'ColorGrade{_zone}Hue'] = (0, 360)
    PARAM_RANGES[f'ColorGrade{_zone}Sat'] = (0, 10)     # 饱和度 ≤ 10
    PARAM_RANGES[f'ColorGrade{_zone}Lum'] = (-50, 50)
PARAM_RANGES['ColorGradeBalance']  = (-100, 100)
PARAM_RANGES['ColorGradeBlending'] = (0, 100)
# 校准 — 幅度较大
for _c in ['Red', 'Green', 'Blue']:
    PARAM_RANGES[f'{_c}Hue']        = (-100, 100)
    PARAM_RANGES[f'{_c}Saturation'] = (-100, 100)
# HSL — 幅度较大
for _c in HSL_COLORS:
    PARAM_RANGES[f'HueAdjustment{_c}']        = (-100, 100)
    PARAM_RANGES[f'SaturationAdjustment{_c}'] = (-100, 100)
    PARAM_RANGES[f'LuminanceAdjustment{_c}']  = (-100, 100)

# 验证所有参数都有范围
_missing = [p for p in PARAM_ORDER if p not in PARAM_RANGES]
assert not _missing, f"缺少范围定义: {_missing}"

# 色相类参数（采样时角度环绕，归一化时也特殊处理）
HUE_PARAMS = {
    'ColorGradeShadowHue', 'ColorGradeMidtoneHue', 'ColorGradeHighlightHue',
}

# 浮点参数（其余整数）
FLOAT_PARAMS = {'Exposure'}


def normalize(value: float, name: str) -> float:
    """单值归一化到 [-1, 1]"""
    lo, hi = PARAM_RANGES[name]
    mid = (lo + hi) / 2
    span = (hi - lo) / 2
    return (value - mid) / span if span > 0 else 0.0


def denormalize(norm: float, name: str) -> float:
    """单值从 [-1, 1] 反归一化"""
    lo, hi = PARAM_RANGES[name]
    mid = (lo + hi) / 2
    span = (hi - lo) / 2
    val = norm * span + mid
    return round(val, 2) if name in FLOAT_PARAMS else int(round(val))


if __name__ == '__main__':
    print(f"总参数数: {len(PARAM_ORDER)}")
    print(f"\n分组:")
    print(f"  S1 亮度:     {len(GROUP_LUMINANCE)}")
    print(f"  S2 色调曲线: {len(GROUP_CURVE)}")
    print(f"  S3 颜色分级: {len(GROUP_COLORGRADE)}")
    print(f"  S4 校准:     {len(GROUP_CALIBRATION)}")
    print(f"  S5 HSL:      {len(GROUP_HSL)}")
    print(f"\n课程阶段累积维度:")
    for i, (name, params) in enumerate(CURRICULUM_STAGES):
        print(f"  Stage {i+1} {name}: {len(params)} 维")
