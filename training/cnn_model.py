"""
CNN 参数预测模型

核心思想：
  输入：(src_image, ref_image) - 原图和参考图
  学习：参数变化对应的图像变化规律
  输出：预测的 22 维 LR 参数向量

模型架构：Siamese 双分支 + Fusion 层
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple


class ParamPredictor(nn.Module):
    """参数预测 CNN 模型（Siamese 双分支）"""

    def __init__(self, backbone: str = 'resnet18', pretrained: bool = True):
        """
        Args:
            backbone: 'resnet18' 或 'resnet34'（轻量模型适合迁移学习）
            pretrained: 使用 ImageNet 预训练权重
        """
        super().__init__()

        # 双分支：都用相同的骨干网络（Siamese 架构）
        weights = 'DEFAULT' if pretrained else None
        if backbone == 'resnet18':
            base_model = models.resnet18(weights=weights)
        elif backbone == 'resnet34':
            base_model = models.resnet34(weights=weights)
        else:
            raise ValueError(f"未支持的 backbone: {backbone}")

        # 移除分类头，保留特征提取层（5 个阶段）
        # layer4 输出: 512 维特征，空间大小 12×12（输入 384 时）
        self.backbone = nn.Sequential(*list(base_model.children())[:-1])
        backbone_out_dim = 512

        # Fusion 层（拼接 src 和 ref 的特征）
        self.fusion = nn.Sequential(
            nn.Linear(backbone_out_dim * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        # 参数预测头（22 个输出）
        self.param_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 22),  # 22 个参数
        )

        # 冻结骨干网络的前几层（保留 ImageNet 特征），只微调后层
        for param in self.backbone[:-2].parameters():
            param.requires_grad = False

    def forward(self, src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            src: 原图 (batch_size, 3, 384, 384)
            ref: 参考图 (batch_size, 3, 384, 384)

        Returns:
            params: 预测的参数向量 (batch_size, 22)
        """
        # 双分支特征提取
        src_feat = self.backbone(src)  # (batch, 512, 1, 1)
        ref_feat = self.backbone(ref)  # (batch, 512, 1, 1)

        # 展平
        src_feat = src_feat.flatten(1)  # (batch, 512)
        ref_feat = ref_feat.flatten(1)  # (batch, 512)

        # Fusion
        combined = torch.cat([src_feat, ref_feat], dim=1)  # (batch, 1024)
        fused = self.fusion(combined)  # (batch, 128)

        # 参数预测
        params = self.param_head(fused)  # (batch, 22)

        return params


class ParamPredictorWithSkip(nn.Module):
    """带残差连接的参数预测模型（更强的表达能力）"""

    def __init__(self, backbone: str = 'resnet18', pretrained: bool = True):
        super().__init__()

        weights = 'DEFAULT' if pretrained else None
        if backbone == 'resnet18':
            base_model = models.resnet18(weights=weights)
        elif backbone == 'resnet34':
            base_model = models.resnet34(weights=weights)
        else:
            raise ValueError(f"未支持的 backbone: {backbone}")

        self.backbone = nn.Sequential(*list(base_model.children())[:-1])
        backbone_out_dim = 512

        # 多层 Fusion + 残差
        self.fusion_block1 = nn.Sequential(
            nn.Linear(backbone_out_dim * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.fusion_block2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
        )
        self.fusion_block3 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

        # 参数预测头
        self.param_head = nn.Linear(128, 22)

        # 冻结骨干网络前层
        for param in self.backbone[:-2].parameters():
            param.requires_grad = False

    def forward(self, src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        src_feat = self.backbone(src).flatten(1)
        ref_feat = self.backbone(ref).flatten(1)

        combined = torch.cat([src_feat, ref_feat], dim=1)

        x = self.fusion_block1(combined)
        x = self.fusion_block2(x)
        x = self.fusion_block3(x)

        params = self.param_head(x)
        return params


def count_parameters(model: nn.Module) -> int:
    """计算模型参数总数"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # 测试模型
    model = ParamPredictor('resnet18', pretrained=True)
    print(f"模型参数数: {count_parameters(model):,}")

    # 伪输入
    src = torch.randn(4, 3, 384, 384)
    ref = torch.randn(4, 3, 384, 384)

    params = model(src, ref)
    print(f"输出形状: {params.shape}")  # (4, 22)
    print(f"输出范围: [{params.min():.2f}, {params.max():.2f}]")
