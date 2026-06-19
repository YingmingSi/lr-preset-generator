"""
CNN 参数预测器（后端集成版）

加载 v4 颜色专用 CNN 模型，从 (src_image, ref_image) 预测 22 维 LR 参数。
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Dict, Optional


# ─── 模型架构（与 training/cnn_model.py 一致）────────────────────────────────

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


def rgb_to_hsv(rgb: torch.Tensor) -> torch.Tensor:
    """向量化 RGB → HSV"""
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    maxc, _ = rgb.max(dim=1, keepdim=True)
    minc, _ = rgb.min(dim=1, keepdim=True)
    delta = maxc - minc
    v = maxc
    s = torch.where(maxc > 0, delta / (maxc + 1e-10), torch.zeros_like(maxc))
    rc = (maxc - r) / (delta + 1e-10)
    gc = (maxc - g) / (delta + 1e-10)
    bc = (maxc - b) / (delta + 1e-10)
    h_r = bc - gc
    h_g = 2.0 + rc - bc
    h_b = 4.0 + gc - rc
    h = torch.where(r == maxc, h_r,
        torch.where(g == maxc, h_g, h_b))
    h = (h / 6.0) % 1.0
    h = torch.where(delta == 0, torch.zeros_like(h), h)
    return torch.cat([h, s, v], dim=1)


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1, pool=False):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.GELU()
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()

    def forward(self, x):
        return self.pool(self.act(self.bn(self.conv(x))))


class ParamPredictorModel(nn.Module):
    """色彩映射 CNN（v4 架构）"""

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(12, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        self.stage1 = ConvBlock(64, 96, pool=True)
        self.stage2 = ConvBlock(96, 128, pool=True)
        self.stage3 = ConvBlock(128, 192, pool=True)
        self.stage4 = ConvBlock(192, 256, pool=True)

        self.head = nn.Sequential(
            nn.Linear(256 * 3, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 22),
            nn.Tanh(),
        )

    def forward(self, src, ref):
        src_hsv = rgb_to_hsv(src)
        ref_hsv = rgb_to_hsv(ref)
        x = torch.cat([src, src_hsv, ref, ref_hsv], dim=1)

        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        avg = x.mean(dim=[2, 3])
        mx, _ = x.flatten(2).max(dim=2)
        std = x.flatten(2).std(dim=2)
        return self.head(torch.cat([avg, mx, std], dim=1))


# ─── 推理接口 ─────────────────────────────────────────────────────────────

class CNNParameterPredictor:
    """LR 参数预测器（CNN 推理接口）"""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        img_size: int = 384,
    ):
        self.device = torch.device(device)
        self.img_size = img_size
        self.model = None
        self.is_loaded = False

        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str):
        """加载模型权重"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        self.model = ParamPredictorModel().to(self.device)
        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)

        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            self.model.load_state_dict(ckpt['model_state_dict'])
        else:
            self.model.load_state_dict(ckpt)

        self.model.eval()
        self.is_loaded = True
        print(f"✓ CNN 模型已加载: {model_path}")

    def _preprocess_image(self, img: np.ndarray) -> torch.Tensor:
        """RGB numpy [0-255] 或 [0-1] → tensor (1, 3, H, W) [0-1]"""
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

        # (H, W, 3) → (3, H, W) → (1, 3, H, W)
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    def _denormalize(self, normalized: np.ndarray) -> dict:
        """从 [-1, 1] 反归一化到原始参数范围"""
        out = {}
        for i, name in enumerate(PARAM_NAMES):
            lo, hi = PARAM_RANGES[name]
            mid = (lo + hi) / 2
            span = (hi - lo) / 2
            val = float(normalized[i]) * span + mid

            # 整数化（除 Exposure）
            if name == 'Exposure':
                out[name] = round(val, 2)
            else:
                out[name] = int(round(val))
        return out

    def predict(self, src_rgb: np.ndarray, ref_rgb: np.ndarray) -> Dict[str, float]:
        """
        预测 LR 参数

        Args:
            src_rgb: 原图 RGB (H, W, 3)
            ref_rgb: 参考图 RGB (H, W, 3)

        Returns:
            {param_name: value} 字典
        """
        if not self.is_loaded:
            raise RuntimeError("模型未加载，请先调用 load_model()")

        src_t = self._preprocess_image(src_rgb)
        ref_t = self._preprocess_image(ref_rgb)

        with torch.no_grad():
            pred = self.model(src_t, ref_t)  # (1, 22), 范围 [-1, 1]

        normalized = pred.squeeze(0).cpu().numpy()
        params = self._denormalize(normalized)

        # 安全约束（颜色分级饱和度限制 15，避免过饱和）
        params['SplitToningShadowSaturation'] = min(
            params['SplitToningShadowSaturation'], 15
        )
        params['SplitToningHighlightSaturation'] = min(
            params['SplitToningHighlightSaturation'], 15
        )

        return params


# ─── 全局单例 ─────────────────────────────────────────────────────────────

_predictor: Optional[CNNParameterPredictor] = None


def load_predictor(model_path: Optional[str] = None) -> CNNParameterPredictor:
    """加载全局 CNN 预测器（应用启动时调用一次）"""
    global _predictor

    if model_path is None:
        # 默认路径：backend/models/param_predictor.pt
        backend_dir = Path(__file__).resolve().parent.parent
        model_path = str(backend_dir / 'models' / 'param_predictor.pt')

    if not os.path.exists(model_path):
        print(f"⚠ CNN 模型不存在: {model_path}（将退化到传统分析）")
        return None

    _predictor = CNNParameterPredictor(model_path=model_path)
    return _predictor


def predict_params(src_rgb: np.ndarray, ref_rgb: np.ndarray) -> Optional[Dict[str, float]]:
    """便捷接口：预测参数（如未加载返回 None）"""
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
        model_path = str(backend_dir / 'models' / 'param_predictor.pt')

    predictor = CNNParameterPredictor(model_path)

    # 测试：随机图
    np.random.seed(42)
    src = np.random.randint(0, 256, (384, 384, 3), dtype=np.uint8)
    ref = np.random.randint(0, 256, (384, 384, 3), dtype=np.uint8)

    params = predictor.predict(src, ref)
    print("\n预测参数:")
    for name, val in params.items():
        print(f"  {name:<40}: {val}")
