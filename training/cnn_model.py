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
    色彩映射 CNN v8 — 空间感知 + 差异输入

    改进（针对"颜色调整保守"）:
      1. 差异输入：显式 ref-src（局部变化检测）
      2. 空间注意力池化：保留"颜色在哪、变多少"的位置信息
      3. 多尺度：融合 stage3(细) + stage4(粗) 特征
      4. 线性输出（去 Tanh）：减少向 0 收缩，敢于输出大值

    输入: src (B,3,H,W), ref (B,3,H,W) ∈ [0,1]
    输出: (B, 72)
    """

    def __init__(self, backbone: str = 'simple_color', pretrained: bool = False):
        super().__init__()

        # 输入 15 通道：src(RGB+HSV 6) + ref(RGB+HSV 6) + diff(RGB 3)
        self.stem = nn.Sequential(
            nn.Conv2d(15, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        self.stage1 = ConvBlock(64, 96, pool=True)
        self.stage2 = ConvBlock(96, 128, pool=True)
        self.stage3 = ConvBlock(128, 192, pool=True)   # 多尺度：细
        self.stage4 = ConvBlock(192, 256, pool=True)   # 粗

        # 空间注意力：学习"关注哪些区域"（捕捉局部色彩变化）
        self.attn = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(64, 1, kernel_size=1),
        )

        # 特征维度: stage4 avg+max+std+attn (4×256) + stage3 avg+std (2×192) = 1408
        feat_dim = 256 * 4 + 192 * 2
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 768),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(768, 384),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(384, N_PARAMS),
            nn.Tanh(),  # 恢复 Tanh：训练稳定（去 Tanh 会震荡）。"敢调色"靠强采样数据
        )

    def _prepare_input(self, src, ref):
        src_hsv = rgb_to_hsv(src)
        ref_hsv = rgb_to_hsv(ref)
        diff = ref - src   # 显式差异
        return torch.cat([src, src_hsv, ref, ref_hsv, diff], dim=1)  # 15 ch

    def forward(self, src, ref):
        x = self._prepare_input(src, ref)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        s3 = self.stage3(x)    # (B, 192, H/16, W/16)
        s4 = self.stage4(s3)   # (B, 256, H/32, W/32)

        # stage4 全局统计
        avg = s4.mean(dim=[2, 3])
        mx, _ = s4.flatten(2).max(dim=2)
        std = s4.flatten(2).std(dim=2)

        # 空间注意力池化
        a = self.attn(s4)                          # (B,1,h,w)
        a = torch.softmax(a.flatten(2), dim=2)     # (B,1,hw)
        attn_pool = (s4.flatten(2) * a).sum(dim=2) # (B,256)

        # stage3 多尺度统计
        s3_avg = s3.mean(dim=[2, 3])
        s3_std = s3.flatten(2).std(dim=2)

        feat = torch.cat([avg, mx, std, attn_pool, s3_avg, s3_std], dim=1)
        return self.head(feat)


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
