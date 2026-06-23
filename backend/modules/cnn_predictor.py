"""
CNN 参数预测器 — ONNX Runtime 推理版（生产部署用）

用 onnxruntime 替代 PyTorch 推理：
  - 内存占用 ~50MB（vs torch ~250MB）
  - 推理速度相当或更快
  - 不需要 torch 依赖
"""

import os
import gc
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Dict, Optional

import onnxruntime as ort


# 参数顺序（与训练时一致）
PARAM_NAMES = (
    'Exposure', 'Highlights', 'Shadows', 'Blacks', 'Whites', 'Contrast',
    'Saturation', 'Vibrance', 'Clarity',
    'SaturationAdjustmentOrange', 'SaturationAdjustmentAqua',
    'SaturationAdjustmentGreen', 'SaturationAdjustmentBlue',
    'HueAdjustmentOrange', 'HueAdjustmentGreen', 'HueAdjustmentAqua',
    'LuminanceAdjustmentOrange', 'LuminanceAdjustmentBlue',
    'SplitToningShadowHue', 'SplitToningShadowSaturation',
    'SplitToningHighlightHue', 'SplitToningHighlightSaturation',
)

# 参数范围（与训练时一致）
PARAM_RANGES = {
    'Exposure':     (-3.0, 3.0),
    'Highlights':   (-100, 100),
    'Shadows':      (-100, 100),
    'Blacks':       (-100, 100),
    'Whites':       (-100, 100),
    'Contrast':     (-100, 100),
    'Saturation':   (-100, 100),
    'Vibrance':     (-100, 100),
    'Clarity':      (-100, 100),
    'SaturationAdjustmentOrange': (-100, 100),
    'SaturationAdjustmentAqua':   (-100, 100),
    'SaturationAdjustmentGreen':  (-100, 100),
    'SaturationAdjustmentBlue':   (-100, 100),
    'HueAdjustmentOrange': (-100, 100),
    'HueAdjustmentGreen':  (-100, 100),
    'HueAdjustmentAqua':   (-100, 100),
    'LuminanceAdjustmentOrange': (-100, 100),
    'LuminanceAdjustmentBlue':   (-100, 100),
    'SplitToningShadowHue':           (0, 360),
    'SplitToningShadowSaturation':    (0, 100),
    'SplitToningHighlightHue':        (0, 360),
    'SplitToningHighlightSaturation': (0, 100),
}


class CNNParameterPredictor:
    """LR 参数预测器（ONNX 推理）"""

    def __init__(self, model_path: Optional[str] = None, img_size: int = 384):
        self.img_size = img_size
        self.session: Optional[ort.InferenceSession] = None
        self.is_loaded = False

        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str):
        """加载 ONNX 模型"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        # 单线程 CPU 推理（避免内存峰值）
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=['CPUExecutionProvider'],
        )
        self.is_loaded = True
        print(f"✓ ONNX CNN 模型已加载: {model_path}")

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """RGB → (1, 3, H, W) float32 [0, 1]"""
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        elif img.max() > 1.5:
            img = img.astype(np.float32) / 255.0
        else:
            img = img.astype(np.float32)

        # Resize 到训练尺寸
        if img.shape[0] != self.img_size or img.shape[1] != self.img_size:
            pil = Image.fromarray((img * 255).clip(0, 255).astype(np.uint8))
            pil = pil.resize((self.img_size, self.img_size), Image.BILINEAR)
            img = np.array(pil).astype(np.float32) / 255.0

        # (H, W, 3) → (1, 3, H, W)
        return np.transpose(img, (2, 0, 1))[np.newaxis, ...]

    def _denormalize(self, normalized: np.ndarray) -> dict:
        """从 [-1, 1] 反归一化到原始参数范围"""
        out = {}
        for i, name in enumerate(PARAM_NAMES):
            lo, hi = PARAM_RANGES[name]
            mid = (lo + hi) / 2
            span = (hi - lo) / 2
            val = float(normalized[i]) * span + mid

            if name == 'Exposure':
                out[name] = round(val, 2)
            else:
                out[name] = int(round(val))
        return out

    def predict(self, src_rgb: np.ndarray, ref_rgb: np.ndarray) -> Dict[str, float]:
        """预测 LR 参数"""
        if not self.is_loaded:
            raise RuntimeError("模型未加载")

        src_arr = self._preprocess(src_rgb)
        ref_arr = self._preprocess(ref_rgb)

        # ONNX 推理
        pred = self.session.run(
            None,
            {'src': src_arr, 'ref': ref_arr},
        )[0]  # (1, 22)

        normalized = pred[0]
        params = self._denormalize(normalized)

        # 释放中间数组
        del src_arr, ref_arr, pred
        gc.collect()

        # 安全约束（颜色分级饱和度限制 15）
        params['SplitToningShadowSaturation'] = min(
            params['SplitToningShadowSaturation'], 15
        )
        params['SplitToningHighlightSaturation'] = min(
            params['SplitToningHighlightSaturation'], 15
        )
        return params


# ─── 全局单例 ─────────────────────────────────────────────────────────────

_predictor: Optional[CNNParameterPredictor] = None


def load_predictor(model_path: Optional[str] = None) -> Optional[CNNParameterPredictor]:
    """加载全局 CNN 预测器（应用启动时调用一次）"""
    global _predictor

    if model_path is None:
        backend_dir = Path(__file__).resolve().parent.parent
        model_path = str(backend_dir / 'models' / 'param_predictor.onnx')

    if not os.path.exists(model_path):
        print(f"⚠ ONNX 模型不存在: {model_path}")
        return None

    _predictor = CNNParameterPredictor(model_path=model_path)
    return _predictor


def predict_params(src_rgb: np.ndarray, ref_rgb: np.ndarray) -> Optional[Dict[str, float]]:
    if _predictor is None:
        return None
    return _predictor.predict(src_rgb, ref_rgb)


def is_predictor_loaded() -> bool:
    return _predictor is not None and _predictor.is_loaded


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        backend_dir = Path(__file__).resolve().parent.parent
        model_path = str(backend_dir / 'models' / 'param_predictor.onnx')

    predictor = CNNParameterPredictor(model_path)
    np.random.seed(42)
    src = np.random.randint(0, 256, (384, 384, 3), dtype=np.uint8)
    ref = np.random.randint(0, 256, (384, 384, 3), dtype=np.uint8)

    params = predictor.predict(src, ref)
    print("\n预测参数:")
    for name, val in params.items():
        print(f"  {name:<40}: {val}")
