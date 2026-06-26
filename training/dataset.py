"""
训练数据集 v3

关键改进:
  1. 按"原照片"划分 train/val/test（无数据泄漏）
  2. 不做几何/色彩增强（避免污染信号）
  3. 仅做 resize + 归一化
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Tuple, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from param_normalizer import ParamNormalizer
from params_config import PARAM_ORDER

PARAM_NAMES = PARAM_ORDER  # 72 维


class PresetDataset(Dataset):
    """LR 参数预测数据集（按照片划分，无增强）"""

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        img_size: int = 256,
        seed: int = 42,
    ):
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.split = split
        self.normalizer = ParamNormalizer()

        # 加载所有元数据
        param_files = sorted(self.data_dir.glob('*_params.json'))
        if not param_files:
            raise ValueError(f"在 {data_dir} 中未找到 *_params.json")

        # 按 source_photo 分组
        photos_to_samples = {}
        for pf in param_files:
            with open(pf) as f:
                data = json.load(f)
            photo = data.get('source_photo', 'unknown')
            if photo not in photos_to_samples:
                photos_to_samples[photo] = []
            photos_to_samples[photo].append(pf)

        # 按照片划分（确保不泄漏）
        all_photos = sorted(photos_to_samples.keys())
        rng = np.random.RandomState(seed)
        rng.shuffle(all_photos)

        n_photos = len(all_photos)
        n_train_photos = int(n_photos * train_ratio)
        n_val_photos = int(n_photos * val_ratio)

        if split == 'train':
            selected_photos = all_photos[:n_train_photos]
        elif split == 'val':
            selected_photos = all_photos[n_train_photos:n_train_photos + n_val_photos]
        elif split == 'test':
            selected_photos = all_photos[n_train_photos + n_val_photos:]
        else:
            raise ValueError(f"Unknown split: {split}")

        # 收集这些照片的所有样本
        self.param_files = []
        for photo in selected_photos:
            self.param_files.extend(photos_to_samples[photo])

        self.n_photos = len(selected_photos)
        self.n_samples = len(self.param_files)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        param_file = self.param_files[idx]
        with open(param_file) as f:
            data = json.load(f)

        # 加载图像（直接 PIL → tensor，无几何/色彩增强）
        src_img = Image.open(data['src']).convert('RGB')
        ref_img = Image.open(data['ref']).convert('RGB')

        # Resize
        if src_img.size != (self.img_size, self.img_size):
            src_img = src_img.resize((self.img_size, self.img_size), Image.BILINEAR)
            ref_img = ref_img.resize((self.img_size, self.img_size), Image.BILINEAR)

        # → tensor，归一化到 [0, 1]
        src_tensor = torch.from_numpy(np.array(src_img)).permute(2, 0, 1).float() / 255.0
        ref_tensor = torch.from_numpy(np.array(ref_img)).permute(2, 0, 1).float() / 255.0

        # 参数归一化到 [-1, 1]
        params = data['params']
        normalized = self.normalizer.normalize(params)
        param_vector = torch.tensor(
            [normalized.get(name, 0) for name in PARAM_NAMES],
            dtype=torch.float32,
        )

        return {
            'src': src_tensor,
            'ref': ref_tensor,
            'params': param_vector,
            'idx': data.get('idx', idx),
        }


def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    img_size: int = 256,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """创建训练/验证/测试 DataLoader（按照片划分）"""
    train_ds = PresetDataset(data_dir, 'train', train_ratio, val_ratio, img_size)
    val_ds = PresetDataset(data_dir, 'val', train_ratio, val_ratio, img_size)
    test_ds = PresetDataset(data_dir, 'test', train_ratio, val_ratio, img_size)

    print(f"📊 数据集划分（按照片）:")
    print(f"  Train: {train_ds.n_photos} 照片 → {train_ds.n_samples} 样本")
    print(f"  Val:   {val_ds.n_photos} 照片 → {val_ds.n_samples} 样本")
    print(f"  Test:  {test_ds.n_photos} 照片 → {test_ds.n_samples} 样本")

    common = dict(num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **common)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **common)

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    train_loader, val_loader, test_loader = create_dataloaders('./data', batch_size=4)
    batch = next(iter(train_loader))
    print(f"\nbatch shapes:")
    print(f"  src:    {batch['src'].shape}")
    print(f"  ref:    {batch['ref'].shape}")
    print(f"  params: {batch['params'].shape}")
    print(f"  src 范围: [{batch['src'].min():.3f}, {batch['src'].max():.3f}]")
    print(f"  params 范围: [{batch['params'].min():.3f}, {batch['params'].max():.3f}]")
