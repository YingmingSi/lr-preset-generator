"""
颜色映射 CNN v4 — 极简色彩专用

设计原则:
  1. 不用 ImageNet 预训练（对色彩鲁棒反而是缺点）
  2. 输入 RGB + HSV（直接给模型色彩信息）
  3. 小型 CNN，从头训练
  4. 保留空间特征（不 pool 到 1×1）
  5. 单一回归 head（不要多任务头）

输入: src (B, 3, H, W) + ref (B, 3, H, W)
内部: 计算 HSV，拼接 RGB+HSV (12 通道) → CNN → 72 维参数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


from params_config import PARAM_ORDER

PARAM_NAMES = PARAM_ORDER  # 72 维
N_PARAMS = len(PARAM_ORDER)


def rgb_to_hsv(rgb: torch.Tensor) -> torch.Tensor:
    """
    向量化 RGB → HSV（PyTorch）

    输入: (B, 3, H, W)，值在 [0, 1]
    输出: (B, 3, H, W)，HSV 通道，值都在 [0, 1]
    """
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
    """卷积块：Conv → BN → GELU → (optional) MaxPool"""
    def __init__(self, in_c: int, out_c: int, stride: int = 1, pool: bool = False):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride,
                              padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.GELU()
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()

    def forward(self, x):
        return self.pool(self.act(self.bn(self.conv(x))))


class ParamPredictor(nn.Module):
    """
    色彩映射 CNN

    输入: src (B, 3, H, W), ref (B, 3, H, W) 已经归一化到 [0, 1]
    输出: (B, 72) 参数预测（Tanh 限制到 [-1, 1]）

    架构:
      1. 把 src 和 ref 都计算 RGB+HSV (6 通道每张)
      2. 在通道维度堆叠: 12 通道
      3. 简单 CNN 提取空间特征 (384→12)
      4. 全局池化 + MLP → 72
    """

    def __init__(self, backbone: str = 'simple_color', pretrained: bool = False):
        super().__init__()

        # 输入 12 通道：src(RGB+HSV) + ref(RGB+HSV)
        self.stem = nn.Sequential(
            nn.Conv2d(12, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )  # 384 → 192

        # 主干（5 个 stage）
        self.stage1 = ConvBlock(64, 96, pool=True)    # 192 → 96
        self.stage2 = ConvBlock(96, 128, pool=True)   # 96 → 48
        self.stage3 = ConvBlock(128, 192, pool=True)  # 48 → 24
        self.stage4 = ConvBlock(192, 256, pool=True)  # 24 → 12
        # 不再 pool，保留 12×12 空间特征

        # 全局池化 + 全局统计
        # 全局: avg, max, std → 3 × 256 = 768
        self.head = nn.Sequential(
            nn.Linear(256 * 3, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, N_PARAMS),
            nn.Tanh(),  # 限制到 [-1, 1] 匹配归一化目标
        )

    def _prepare_input(self, src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """拼接 src(RGB+HSV) + ref(RGB+HSV) = 12 通道"""
        src_hsv = rgb_to_hsv(src)  # (B, 3, H, W)
        ref_hsv = rgb_to_hsv(ref)
        # 12 通道: [src_R, src_G, src_B, src_H, src_S, src_V,
        #          ref_R, ref_G, ref_B, ref_H, ref_S, ref_V]
        return torch.cat([src, src_hsv, ref, ref_hsv], dim=1)

    def forward(self, src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        # 准备 12 通道输入
        x = self._prepare_input(src, ref)  # (B, 12, H, W)

        # CNN 主干
        x = self.stem(x)        # (B, 64, 192, 192)
        x = self.stage1(x)      # (B, 96, 96, 96)
        x = self.stage2(x)      # (B, 128, 48, 48)
        x = self.stage3(x)      # (B, 192, 24, 24)
        x = self.stage4(x)      # (B, 256, 12, 12)

        # 全局统计
        avg_pool = x.mean(dim=[2, 3])  # (B, 256)
        max_pool, _ = x.flatten(2).max(dim=2)  # (B, 256)
        std_pool = x.flatten(2).std(dim=2)  # (B, 256)

        global_feat = torch.cat([avg_pool, max_pool, std_pool], dim=1)  # (B, 768)

        # 回归头
        params = self.head(global_feat)  # (B, 72)
        return params


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# 兼容旧代码
ParamPredictorWithSkip = ParamPredictor


if __name__ == '__main__':
    model = ParamPredictor()
    print(f"参数量: {count_parameters(model):,}")

    src = torch.rand(4, 3, 384, 384)
    ref = torch.rand(4, 3, 384, 384)
    with torch.no_grad():
        out = model(src, ref)
    print(f"输出 shape: {out.shape}")
    print(f"输出范围: [{out.min().item():.3f}, {out.max().item():.3f}]")
