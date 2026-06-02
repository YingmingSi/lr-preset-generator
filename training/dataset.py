"""
CNN 训练数据集管道

从 generate_dataset.py 生成的数据加载：
  - 原图: {idx:06d}_src.jpg
  - 参考图（应用参数渲染后）: {idx:06d}_ref.jpg
  - 参数标签: {idx:06d}_params.json
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class PresetDataset(Dataset):
    """LR 参数预测数据集"""

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        img_size: int = 384,
        augment: bool = False,
        seed: int = 42,
    ):
        """
        Args:
            data_dir: 数据目录（包含 XXXXXX_src.jpg, XXXXXX_ref.jpg, XXXXXX_params.json）
            split: 'train', 'val', 或 'test'
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            img_size: 输入图像大小（正方形）
            augment: 是否应用数据增强
            seed: 随机种子（确保不同进程分割一致）
        """
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.augment = augment
        self.split = split

        # 收集所有 params.json 文件（作为数据索引）
        param_files = sorted(self.data_dir.glob('*_params.json'))
        if not param_files:
            raise ValueError(f"在 {data_dir} 中未找到 *_params.json 文件")

        # 按 split 划分（确定性划分）
        np.random.seed(seed)
        indices = np.random.permutation(len(param_files))

        n_train = int(len(indices) * train_ratio)
        n_val = int(len(indices) * val_ratio)

        if split == 'train':
            split_indices = indices[:n_train]
        elif split == 'val':
            split_indices = indices[n_train:n_train + n_val]
        elif split == 'test':
            split_indices = indices[n_train + n_val:]
        else:
            raise ValueError(f"Unknown split: {split}")

        self.param_files = [param_files[i] for i in split_indices]

        # 图像预处理（无数据增强）
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet 均值
                std=[0.229, 0.224, 0.225],   # ImageNet 标准差
            ),
        ])

        # 数据增强（仅训练集）
        self.aug_transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=Image.BILINEAR),
            transforms.RandomAffine(
                degrees=5,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
            ),
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
                hue=0.05,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self) -> int:
        return len(self.param_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        返回一个样本

        Returns:
            {
                'src': tensor (3, 384, 384) - 原图
                'ref': tensor (3, 384, 384) - 参考图
                'params': tensor (22,) - 参数标签
                'idx': int - 样本索引（用于调试）
            }
        """
        param_file = self.param_files[idx]
        base_name = param_file.stem.replace('_params', '')

        # 加载参数
        with open(param_file, 'r') as f:
            data = json.load(f)

        # 参数向量（22 个参数，顺序固定）
        param_names = [
            'Exposure', 'Highlights', 'Shadows', 'Blacks', 'Whites', 'Contrast',
            'Saturation', 'Vibrance', 'Clarity',
            'SaturationAdjustmentOrange', 'SaturationAdjustmentAqua',
            'SaturationAdjustmentGreen', 'SaturationAdjustmentBlue',
            'HueAdjustmentOrange', 'HueAdjustmentGreen', 'HueAdjustmentAqua',
            'LuminanceAdjustmentOrange', 'LuminanceAdjustmentBlue',
            'SplitToningShadowHue', 'SplitToningShadowSaturation',
            'SplitToningHighlightHue', 'SplitToningHighlightSaturation',
        ]

        params = data['params']
        param_vector = torch.tensor(
            [params.get(name, 0) for name in param_names],
            dtype=torch.float32
        )

        # 加载图像
        src_path = self.data_dir / f'{base_name}_src.jpg'
        ref_path = self.data_dir / f'{base_name}_ref.jpg'

        try:
            src_img = Image.open(src_path).convert('RGB')
            ref_img = Image.open(ref_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"加载图像失败 {base_name}: {e}")

        # 应用变换
        if self.augment and self.split == 'train':
            src_tensor = self.aug_transform(src_img)
            ref_tensor = self.aug_transform(ref_img)
        else:
            src_tensor = self.transform(src_img)
            ref_tensor = self.transform(ref_img)

        return {
            'src': src_tensor,
            'ref': ref_tensor,
            'params': param_vector,
            'idx': int(base_name),
        }


def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    img_size: int = 384,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建训练/验证/测试 DataLoader

    Args:
        data_dir: 数据目录
        batch_size: 批大小
        num_workers: 数据加载工作进程数
        img_size: 输入图像大小
        train_ratio: 训练集比例
        val_ratio: 验证集比例

    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_dataset = PresetDataset(
        data_dir, split='train', img_size=img_size,
        augment=True, train_ratio=train_ratio, val_ratio=val_ratio
    )
    val_dataset = PresetDataset(
        data_dir, split='val', img_size=img_size,
        augment=False, train_ratio=train_ratio, val_ratio=val_ratio
    )
    test_dataset = PresetDataset(
        data_dir, split='test', img_size=img_size,
        augment=False, train_ratio=train_ratio, val_ratio=val_ratio
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # 测试数据加载
    loader = DataLoader(
        PresetDataset('./data', split='train', augment=True),
        batch_size=4
    )

    batch = next(iter(loader))
    print(f"src shape: {batch['src'].shape}")  # (4, 3, 384, 384)
    print(f"ref shape: {batch['ref'].shape}")  # (4, 3, 384, 384)
    print(f"params shape: {batch['params'].shape}")  # (4, 22)
    print(f"params range: [{batch['params'].min():.2f}, {batch['params'].max():.2f}]")
