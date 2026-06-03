"""
CNN 参数预测器 — 集成到后端

推理接口：
  predict_params(src_rgb, ref_rgb) → params_dict
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple

# 动态导入模型定义（支持训练和推理环境）
try:
    from cnn_model import ParamPredictor
except ImportError:
    # 如果在后端目录，从训练目录导入
    import sys
    training_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'training')
    if os.path.exists(training_dir):
        sys.path.insert(0, training_dir)
        from cnn_model import ParamPredictor


class CNNParameterPredictor:
    """CNN 参数预测器"""

    # 参数名称顺序（必须与训练时一致）
    PARAM_NAMES = [
        'Exposure', 'Highlights', 'Shadows', 'Blacks', 'Whites', 'Contrast',
        'Saturation', 'Vibrance', 'Clarity',
        'SaturationAdjustmentOrange', 'SaturationAdjustmentAqua',
        'SaturationAdjustmentGreen', 'SaturationAdjustmentBlue',
        'HueAdjustmentOrange', 'HueAdjustmentGreen', 'HueAdjustmentAqua',
        'LuminanceAdjustmentOrange', 'LuminanceAdjustmentBlue',
        'SplitToningShadowHue', 'SplitToningShadowSaturation',
        'SplitToningHighlightHue', 'SplitToningHighlightSaturation',
    ]

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        """
        初始化预测器

        Args:
            model_path: 模型检查点路径（如为 None，则不加载）
            device: 'cuda' 或 'cpu'
        """
        self.device = torch.device(device)
        self.model = None
        self.is_loaded = False

        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str):
        """加载模型"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        # 创建模型
        self.model = ParamPredictor(backbone='resnet18', pretrained=False)
        self.model = self.model.to(self.device)

        # 加载权重
        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.eval()
        self.is_loaded = True
        print(f"✓ 模型已加载: {model_path}")

    @staticmethod
    def _normalize_image(img_rgb: np.ndarray) -> torch.Tensor:
        """
        将 RGB 图像标准化为 PyTorch 张量

        Args:
            img_rgb: RGB 图像数组 (H, W, 3)，值范围 [0, 255] 或 [0, 1]

        Returns:
            标准化后的张量 (3, H, W)
        """
        # 转换为 [0, 1] 范围
        if img_rgb.max() > 1.0:
            img_rgb = img_rgb.astype(np.float32) / 255.0
        else:
            img_rgb = img_rgb.astype(np.float32)

        # 调整维度：(H, W, 3) → (3, H, W)
        img_rgb = np.transpose(img_rgb, (2, 0, 1))

        # ImageNet 标准化
        img_tensor = torch.from_numpy(img_rgb)
        img_tensor = (img_tensor - torch.tensor(
            [0.485, 0.456, 0.406], dtype=torch.float32
        ).view(3, 1, 1)) / torch.tensor(
            [0.229, 0.224, 0.225], dtype=torch.float32
        ).view(3, 1, 1)

        return img_tensor

    @staticmethod
    def _resize_image(tensor: torch.Tensor, size: int = 384) -> torch.Tensor:
        """
        调整图像大小

        Args:
            tensor: 输入张量 (3, H, W)
            size: 目标大小（正方形）

        Returns:
            调整后的张量 (3, size, size)
        """
        import torch.nn.functional as F
        if tensor.shape[1] != size or tensor.shape[2] != size:
            tensor = F.interpolate(
                tensor.unsqueeze(0),
                size=(size, size),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        return tensor

    def predict(
        self,
        src_rgb: np.ndarray,
        ref_rgb: np.ndarray,
    ) -> Dict[str, float]:
        """
        预测参数

        Args:
            src_rgb: 原图 (H, W, 3)，值范围 [0, 255] 或 [0, 1]
            ref_rgb: 参考图 (H, W, 3)，值范围 [0, 255] 或 [0, 1]

        Returns:
            参数字典 {param_name: value}
        """
        if not self.is_loaded:
            raise RuntimeError("模型未加载，请先调用 load_model()")

        # 预处理
        src_tensor = self._normalize_image(src_rgb)
        ref_tensor = self._normalize_image(ref_rgb)

        src_tensor = self._resize_image(src_tensor, 384)
        ref_tensor = self._resize_image(ref_tensor, 384)

        # 添加 batch 维度
        src_tensor = src_tensor.unsqueeze(0).to(self.device)
        ref_tensor = ref_tensor.unsqueeze(0).to(self.device)

        # 推理
        with torch.no_grad():
            pred_params = self.model(src_tensor, ref_tensor)  # (1, 22)

        # 转换为字典
        pred_np = pred_params.squeeze(0).cpu().numpy()
        params_dict = {
            name: float(value)
            for name, value in zip(self.PARAM_NAMES, pred_np)
        }

        # 约束：SplitToning 饱和度最大 15
        params_dict['SplitToningShadowSaturation'] = min(params_dict['SplitToningShadowSaturation'], 15.0)
        params_dict['SplitToningHighlightSaturation'] = min(params_dict['SplitToningHighlightSaturation'], 15.0)

        return params_dict

    def predict_batch(
        self,
        src_batch: np.ndarray,
        ref_batch: np.ndarray,
    ) -> list:
        """
        批量预测

        Args:
            src_batch: 原图批 (B, H, W, 3)
            ref_batch: 参考图批 (B, H, W, 3)

        Returns:
            参数字典列表 [params_dict, ...]
        """
        if not self.is_loaded:
            raise RuntimeError("模型未加载")

        results = []
        for src, ref in zip(src_batch, ref_batch):
            params = self.predict(src, ref)
            results.append(params)
        return results

    def predict_with_blend(
        self,
        src_rgb: np.ndarray,
        ref_rgb: np.ndarray,
        analysis_params: Dict[str, float],
        blend_weight: float = 0.5,
    ) -> Dict[str, float]:
        """
        融合 CNN 预测和传统分析结果

        Args:
            src_rgb: 原图
            ref_rgb: 参考图
            analysis_params: 传统分析得到的参数
            blend_weight: 融合权重（0 = 纯分析，1 = 纯 CNN）

        Returns:
            融合后的参数字典
        """
        if not self.is_loaded:
            return analysis_params

        cnn_params = self.predict(src_rgb, ref_rgb)

        # 按参数融合
        blended = {}
        for name in self.PARAM_NAMES:
            cnn_val = cnn_params.get(name, 0)
            analysis_val = analysis_params.get(name, 0)

            # 简单线性融合
            blended[name] = blend_weight * cnn_val + (1 - blend_weight) * analysis_val

        return blended


# 全局预测器实例（延迟加载）
_predictor = None


def load_predictor(model_path: str):
    """加载全局预测器实例"""
    global _predictor
    _predictor = CNNParameterPredictor(model_path=model_path)


def predict_params(src_rgb: np.ndarray, ref_rgb: np.ndarray) -> Dict[str, float]:
    """便捷接口：预测参数"""
    if _predictor is None:
        raise RuntimeError("CNN 预测器未初始化，请先调用 load_predictor()")
    return _predictor.predict(src_rgb, ref_rgb)


def is_predictor_loaded() -> bool:
    """检查预测器是否已加载"""
    global _predictor
    return _predictor is not None and _predictor.is_loaded


if __name__ == '__main__':
    # 测试（需要模型文件）
    import sys
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        predictor = CNNParameterPredictor(model_path)

        # 伪数据
        src = np.random.randint(0, 256, (384, 384, 3), dtype=np.uint8)
        ref = np.random.randint(0, 256, (384, 384, 3), dtype=np.uint8)

        params = predictor.predict(src, ref)
        print("预测参数:")
        for k, v in list(params.items())[:5]:
            print(f"  {k}: {v:.2f}")
    else:
        print("用法: python cnn_predictor.py <model_path>")
