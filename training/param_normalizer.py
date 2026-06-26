"""
参数归一化器 — 委托给 params_config（单一数据源）

保留 ParamNormalizer 类 API 以兼容旧代码，但参数定义统一来自 params_config。
"""

import numpy as np
from params_config import PARAM_ORDER, PARAM_RANGES, FLOAT_PARAMS, normalize, denormalize


class ParamNormalizer:
    """参数归一化/反归一化（72 维）"""

    def __init__(self, ranges=None):
        self.ranges = ranges or PARAM_RANGES

    def normalize(self, param_dict: dict) -> dict:
        return {k: (normalize(v, k) if k in self.ranges else v)
                for k, v in param_dict.items()}

    def denormalize(self, param_dict: dict) -> dict:
        return {k: (denormalize(v, k) if k in self.ranges else v)
                for k, v in param_dict.items()}

    def _mid_span(self):
        mids = np.array([(self.ranges[p][0] + self.ranges[p][1]) / 2 for p in PARAM_ORDER],
                        dtype=np.float32)
        spans = np.array([(self.ranges[p][1] - self.ranges[p][0]) / 2 for p in PARAM_ORDER],
                         dtype=np.float32)
        return mids, spans

    def normalize_array(self, arr: np.ndarray) -> np.ndarray:
        mids, spans = self._mid_span()
        return ((arr - mids) / spans).astype(np.float32)

    def denormalize_array(self, arr: np.ndarray) -> np.ndarray:
        mids, spans = self._mid_span()
        return (arr * spans + mids).astype(np.float32)


if __name__ == '__main__':
    n = ParamNormalizer()
    test = {'Exposure': 1.5, 'Highlights': -50, 'ColorGradeShadowHue': 180}
    norm = n.normalize(test)
    print("归一化:", norm)
    print("反归一化:", n.denormalize(norm))
