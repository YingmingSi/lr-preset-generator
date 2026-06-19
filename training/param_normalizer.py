"""
参数归一化器

CNN 训练时需要将 22 维参数归一化到 [-1, 1] 范围，这样神经网络能更容易学习。
推理时再反归一化回原始范围。
"""

import json
from pathlib import Path

# 参数范围（匹配新的扩大数据生成范围，与 generate_dataset.py 保持一致）
PARAM_RANGES = {
    'Exposure': (-3.0, 3.0),
    'Highlights': (-100, 100),
    'Shadows': (-100, 100),
    'Blacks': (-100, 100),
    'Whites': (-100, 100),
    'Contrast': (-100, 100),
    'Saturation': (-100, 100),
    'Vibrance': (-100, 100),
    'Clarity': (-100, 100),
    'SaturationAdjustmentOrange': (-100, 100),
    'SaturationAdjustmentAqua': (-100, 100),
    'SaturationAdjustmentGreen': (-100, 100),
    'SaturationAdjustmentBlue': (-100, 100),
    'HueAdjustmentOrange': (-100, 100),
    'HueAdjustmentGreen': (-100, 100),
    'HueAdjustmentAqua': (-100, 100),
    'LuminanceAdjustmentOrange': (-100, 100),
    'LuminanceAdjustmentBlue': (-100, 100),
    'SplitToningShadowHue': (0, 360),
    'SplitToningShadowSaturation': (0, 100),
    'SplitToningHighlightHue': (0, 360),
    'SplitToningHighlightSaturation': (0, 100),
}

PARAM_ORDER = [
    'Exposure', 'Highlights', 'Shadows', 'Blacks', 'Whites', 'Contrast',
    'Saturation', 'Vibrance', 'Clarity',
    'SaturationAdjustmentOrange', 'SaturationAdjustmentAqua',
    'SaturationAdjustmentGreen', 'SaturationAdjustmentBlue',
    'HueAdjustmentOrange', 'HueAdjustmentGreen', 'HueAdjustmentAqua',
    'LuminanceAdjustmentOrange', 'LuminanceAdjustmentBlue',
    'SplitToningShadowHue', 'SplitToningShadowSaturation',
    'SplitToningHighlightHue', 'SplitToningHighlightSaturation',
]


class ParamNormalizer:
    """参数归一化/反归一化工具"""

    def __init__(self, ranges=None):
        """
        Args:
            ranges: 参数范围字典，默认使用 PARAM_RANGES
        """
        self.ranges = ranges or PARAM_RANGES.copy()

    def normalize(self, param_dict):
        """
        将参数字典中的值归一化到 [-1, 1]

        Args:
            param_dict: {param_name: value}

        Returns:
            {param_name: normalized_value}
        """
        normalized = {}
        for name, val in param_dict.items():
            if name not in self.ranges:
                normalized[name] = val
                continue

            lo, hi = self.ranges[name]
            mid = (lo + hi) / 2
            span = (hi - lo) / 2

            # 线性映射到 [-1, 1]
            normalized[name] = (val - mid) / span

        return normalized

    def denormalize(self, param_dict):
        """
        将归一化的参数反归一化回原始范围

        Args:
            param_dict: {param_name: normalized_value} where values in [-1, 1]

        Returns:
            {param_name: original_value}
        """
        denormalized = {}
        for name, norm_val in param_dict.items():
            if name not in self.ranges:
                denormalized[name] = norm_val
                continue

            lo, hi = self.ranges[name]
            mid = (lo + hi) / 2
            span = (hi - lo) / 2

            # 从 [-1, 1] 反映射到原始范围
            val = norm_val * span + mid

            # 类型转换（与原数据一致）
            if name == 'Exposure':
                denormalized[name] = round(val, 2)
            else:
                denormalized[name] = int(round(val))

        return denormalized

    def _get_mid_span_arrays(self):
        """获取参数中点和跨度数组（形状为 (22,)）"""
        import numpy as np
        mids = np.zeros(len(PARAM_ORDER), dtype=np.float32)
        spans = np.ones(len(PARAM_ORDER), dtype=np.float32)
        for i, name in enumerate(PARAM_ORDER):
            if name in self.ranges:
                lo, hi = self.ranges[name]
                mids[i] = (lo + hi) / 2
                spans[i] = (hi - lo) / 2
        return mids, spans

    def normalize_array(self, params_array):
        """
        归一化 numpy 数组到 [-1, 1]

        Args:
            params_array: shape (22,) 或 (batch, 22)

        Returns:
            归一化后的数组，形状相同
        """
        import numpy as np
        mids, spans = self._get_mid_span_arrays()
        # 向量化操作（自动广播，支持 1D 和 2D）
        return ((params_array - mids) / spans).astype(np.float32)

    def denormalize_array(self, params_array):
        """
        反归一化 numpy 数组从 [-1, 1] 回到原始范围

        Args:
            params_array: shape (22,) 或 (batch, 22)

        Returns:
            反归一化后的数组，形状相同
        """
        import numpy as np
        mids, spans = self._get_mid_span_arrays()
        # 向量化操作（自动广播）
        return (params_array * spans + mids).astype(np.float32)


if __name__ == '__main__':
    # 测试
    normalizer = ParamNormalizer()

    # 测试字典归一化
    test_params = {
        'Exposure': 0.5,
        'Highlights': -50,
        'SplitToningShadowHue': 180,
    }
    print("原始参数:", test_params)

    normalized = normalizer.normalize(test_params)
    print("归一化:", normalized)

    denormalized = normalizer.denormalize(normalized)
    print("反归一化:", denormalized)

    # 测试数组归一化
    import numpy as np
    arr = np.array([0.5, -50, 180, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 180, 5, 180, 10], dtype=np.float32)
    print("\n原始数组范围:", arr.min(), "-", arr.max())

    norm_arr = normalizer.normalize_array(arr)
    print("归一化后范围:", norm_arr.min(), "-", norm_arr.max())

    denorm_arr = normalizer.denormalize_array(norm_arr)
    print("反归一化后范围:", denorm_arr.min(), "-", denorm_arr.max())
    print("恢复误差:", np.abs(arr - denorm_arr).max())
