"""
相机品牌补偿模块
基于各品牌在Lightroom Adobe Standard配置文件下的系统性偏差
"""


CAMERA_COMPENSATION = {
    'canon': {
        # Canon偏暖，皮肤色好，整体饱和度略高
        'Temperature':                -100,
        'SaturationAdjustmentRed':    -5,
        'SaturationAdjustmentOrange': -5,
        'description': 'Canon Adobe Standard偏暖，略微修正色温和暖色饱和度',
    },
    'nikon': {
        # Nikon偏冷，细节丰富，绿色略重
        'Temperature':               +150,
        'SaturationAdjustmentGreen': -8,
        'HueAdjustmentGreen':        +5,
        'description': 'Nikon Adobe Standard偏冷偏绿，修正色温和绿色色相',
    },
    'sony': {
        # Sony偏绿，需要色相和色调校正
        'HueAdjustmentGreen':        +8,
        'SaturationAdjustmentGreen': -10,
        'Tint':                      +5,
        'description': 'Sony Adobe Standard偏绿，修正绿色色相和品红色调',
    },
    'fuji': {
        # Fuji胶片模拟本身已有强风格，全局系数降低
        'global_coefficient': 0.7,
        'description': 'Fuji胶片模拟已有强风格，降低全局调整幅度至70%',
    },
    'panasonic': {
        'description': '接近中性，无需特殊补偿',
    },
    'olympus': {
        # Olympus色彩略饱和
        'Vibrance':   -5,
        'Saturation': -5,
        'description': 'Olympus色彩略饱和，轻微降低饱和度',
    },
    'leica': {
        # Leica色彩中性准确
        'description': 'Leica色彩中性，无需特殊补偿',
    },
}


def apply_camera_compensation(params: dict, brand: str) -> dict:
    """
    将相机品牌补偿叠加到现有参数上

    Args:
        params: 当前参数字典
        brand:  相机品牌字符串（小写）

    Returns:
        补偿后的参数字典
    """
    comp = CAMERA_COMPENSATION.get(brand.lower(), {})
    if not comp:
        return params

    global_coeff = comp.get('global_coefficient', 1.0)

    for key, value in comp.items():
        if key in ('description', 'global_coefficient'):
            continue
        if key in params:
            params[key] = _clamp(
                int(params[key] + value * global_coeff),
                -100, 100
            )

    # Fuji特殊处理：全局系数缩放所有HSL和曲线参数
    if global_coeff != 1.0:
        scalable = [k for k in params if
                    k.startswith('HueAdjustment') or
                    k.startswith('SaturationAdjustment') or
                    k.startswith('LuminanceAdjustment') or
                    k in ('Contrast', 'Clarity', 'Vibrance')]
        for key in scalable:
            params[key] = int(params[key] * global_coeff)

    return params


def get_camera_description(brand: str) -> str:
    comp = CAMERA_COMPENSATION.get(brand.lower(), {})
    return comp.get('description', '')


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))
