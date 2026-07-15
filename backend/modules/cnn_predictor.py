"""
CNN 参数预测器 — ONNX Runtime 推理（61 维 v9）

预测 61 个 LR 参数：亮度(8) + 曲线(14) + 颜色分级(9) + 校准(6) + HSL(24)
仅保留对 3D LUT 有实测效果的参数。参数定义来自 params_config（与训练一致）。
"""

import os
import gc
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Dict, Optional

import onnxruntime as ort

from modules.params_config import PARAM_ORDER, PARAM_RANGES, FLOAT_PARAMS

# 弱参数归零集合。v9 用稠密数据重训、意在让色相类也学起来，故暂清空。
# 训练完按新 R² 重新评估，若仍有 R²≈0 的参数再加回这里。
WEAK_PARAMS = set()

# 颜色分级饱和度安全上限（与训练范围一致：0-10）
CG_SAT_PARAMS = {'ColorGradeShadowSat', 'ColorGradeMidtoneSat', 'ColorGradeHighlightSat'}


class CNNParameterPredictor:
    """LR 参数预测器（ONNX，72 维）"""

    def __init__(self, model_path: Optional[str] = None, img_size: int = 384):
        self.img_size = img_size
        self.session: Optional[ort.InferenceSession] = None
        self.is_loaded = False
        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            model_path, sess_options=opts, providers=['CPUExecutionProvider'])
        self.is_loaded = True
        print(f"✓ ONNX CNN 模型已加载（72 维）: {model_path}")

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        elif img.max() > 1.5:
            img = img.astype(np.float32) / 255.0
        else:
            img = img.astype(np.float32)
        if img.shape[0] != self.img_size or img.shape[1] != self.img_size:
            pil = Image.fromarray((img * 255).clip(0, 255).astype(np.uint8))
            pil = pil.resize((self.img_size, self.img_size), Image.BILINEAR)
            img = np.array(pil).astype(np.float32) / 255.0
        return np.transpose(img, (2, 0, 1))[np.newaxis, ...]

    def _denormalize(self, norm: np.ndarray) -> dict:
        out = {}
        for i, name in enumerate(PARAM_ORDER):
            lo, hi = PARAM_RANGES[name]
            mid = (lo + hi) / 2
            span = (hi - lo) / 2
            val = float(norm[i]) * span + mid
            out[name] = round(val, 2) if name in FLOAT_PARAMS else int(round(val))
        return out

    def predict(self, src_rgb: np.ndarray, ref_rgb: np.ndarray) -> Dict[str, float]:
        if not self.is_loaded:
            raise RuntimeError("模型未加载")
        src_arr = self._preprocess(src_rgb)
        ref_arr = self._preprocess(ref_rgb)
        pred = self.session.run(None, {'src': src_arr, 'ref': ref_arr})[0]
        params = self._denormalize(pred[0])
        del src_arr, ref_arr, pred
        gc.collect()

        # 弱参数归零
        for k in WEAK_PARAMS:
            params[k] = 0
        # 颜色分级饱和度上限
        for k in CG_SAT_PARAMS:
            params[k] = max(0, min(params[k], 10))
        return params


# ─── 全局单例 ─────────────────────────────────────────────────────────────

_predictor: Optional[CNNParameterPredictor] = None


def load_predictor(model_path: Optional[str] = None) -> Optional[CNNParameterPredictor]:
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
    backend_dir = Path(__file__).resolve().parent.parent
    p = CNNParameterPredictor(str(backend_dir / 'models' / 'param_predictor.onnx'))
    np.random.seed(0)
    src = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    ref = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    out = p.predict(src, ref)
    print(f"预测 {len(out)} 参数，非零: {sum(1 for v in out.values() if v != 0)}")
