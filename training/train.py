"""
CNN 训练脚本

用法:
  python train.py --data-dir ./data --epochs 100 --batch-size 32 --lr 0.001
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from cnn_model import ParamPredictor, ParamPredictorWithSkip, count_parameters
from dataset import create_dataloaders

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class Trainer:
    """CNN 训练器"""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        self.model = model.to(device)
        self.device = device

        # 损失函数：L1 Loss（对异常值鲁棒）
        self.criterion = nn.L1Loss()

        # 优化器：AdamW
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        # 学习率调度：余弦衰减
        self.scheduler = None

    def set_scheduler(self, total_epochs: int):
        """设置学习率调度器"""
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_epochs,
            eta_min=1e-5,
        )

    def train_epoch(self, train_loader) -> float:
        """训练一个 epoch，返回平均损失"""
        self.model.train()
        total_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            src = batch['src'].to(self.device)
            ref = batch['ref'].to(self.device)
            params = batch['params'].to(self.device)

            # 前向传播
            pred_params = self.model(src, ref)

            # 计算损失
            loss = self.criterion(pred_params, params)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

            if (batch_idx + 1) % 50 == 0:
                logger.info(
                    f"  Batch {batch_idx + 1}/{len(train_loader)}, "
                    f"Loss: {loss.item():.6f}"
                )

        avg_loss = total_loss / len(train_loader)
        return avg_loss

    @torch.no_grad()
    def evaluate(self, val_loader) -> dict:
        """评估模型，返回指标字典"""
        self.model.eval()

        total_loss = 0.0
        all_preds = []
        all_targets = []

        for batch in val_loader:
            src = batch['src'].to(self.device)
            ref = batch['ref'].to(self.device)
            params = batch['params'].to(self.device)

            pred_params = self.model(src, ref)
            loss = self.criterion(pred_params, params)

            total_loss += loss.item()
            all_preds.append(pred_params.cpu().numpy())
            all_targets.append(params.cpu().numpy())

        all_preds = np.concatenate(all_preds, axis=0)  # (N, 22)
        all_targets = np.concatenate(all_targets, axis=0)  # (N, 22)

        avg_loss = total_loss / len(val_loader)

        # 计算 MAE、RMSE、R² 等指标
        mae = np.mean(np.abs(all_preds - all_targets))
        rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))

        # R² 分数（针对每个参数）
        ss_res = np.sum((all_targets - all_preds) ** 2, axis=0)
        ss_tot = np.sum((all_targets - np.mean(all_targets, axis=0)) ** 2, axis=0)
        r2_scores = 1 - (ss_res / (ss_tot + 1e-10))
        r2_mean = np.mean(r2_scores)

        # 按参数分组的 MAE（便于诊断）
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
        param_mae = {
            name: float(np.mean(np.abs(all_preds[:, i] - all_targets[:, i])))
            for i, name in enumerate(param_names)
        }

        return {
            'loss': float(avg_loss),
            'mae': float(mae),
            'rmse': float(rmse),
            'r2_mean': float(r2_mean),
            'r2_scores': [float(x) for x in r2_scores],
            'param_mae': param_mae,
        }

    def save_checkpoint(self, path: str, epoch: int, metrics: dict):
        """保存模型检查点"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
        }, path)
        logger.info(f"模型已保存到 {path}")

    def load_checkpoint(self, path: str):
        """加载模型检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint.get('epoch', 0)


def main():
    parser = argparse.ArgumentParser(description='CNN 训练脚本')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='训练数据目录')
    parser.add_argument('--output-dir', type=str, default='./checkpoints',
                        help='模型保存目录')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='批大小')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='初始学习率')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='权重衰减')
    parser.add_argument('--backbone', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34'],
                        help='骨干网络')
    parser.add_argument('--no-pretrain', action='store_true',
                        help='不使用 ImageNet 预训练权重')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='数据加载工作进程数')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='计算设备 (cuda/cpu)')
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的检查点路径')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')

    args = parser.parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    logger.info(f"使用设备: {device}")

    # 创建数据加载器
    logger.info(f"加载数据: {args.data_dir}")
    train_loader, val_loader, test_loader = create_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    logger.info(
        f"训练集: {len(train_loader.dataset)}, "
        f"验证集: {len(val_loader.dataset)}, "
        f"测试集: {len(test_loader.dataset)}"
    )

    # 创建模型
    logger.info(f"初始化模型: {args.backbone} (pretrain={not args.no_pretrain})")
    model = ParamPredictor(
        backbone=args.backbone,
        pretrained=not args.no_pretrain,
    )
    logger.info(f"模型参数数: {count_parameters(model):,}")

    # 创建训练器
    trainer = Trainer(model, device, lr=args.lr, weight_decay=args.weight_decay)
    trainer.set_scheduler(args.epochs)

    # TensorBoard 日志
    log_dir = os.path.join(args.output_dir, 'logs',
                          datetime.now().strftime('%Y%m%d_%H%M%S'))
    writer = SummaryWriter(log_dir)
    logger.info(f"日志保存到: {log_dir}")

    # 恢复训练（如果指定）
    start_epoch = 0
    if args.resume:
        logger.info(f"从检查点恢复: {args.resume}")
        start_epoch = trainer.load_checkpoint(args.resume)

    # 训练循环
    best_val_r2 = -np.inf
    patience = 20
    patience_counter = 0

    for epoch in range(start_epoch, args.epochs):
        logger.info(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")

        # 训练
        train_loss = trainer.train_epoch(train_loader)
        logger.info(f"训练损失: {train_loss:.6f}")

        # 验证
        val_metrics = trainer.evaluate(val_loader)
        logger.info(
            f"验证损失: {val_metrics['loss']:.6f}, "
            f"MAE: {val_metrics['mae']:.4f}, "
            f"RMSE: {val_metrics['rmse']:.4f}, "
            f"R²: {val_metrics['r2_mean']:.4f}"
        )

        # 按参数输出 MAE
        for param_name, mae in list(val_metrics['param_mae'].items())[:5]:
            logger.info(f"  {param_name}: MAE={mae:.4f}")
        if len(val_metrics['param_mae']) > 5:
            logger.info(f"  ... ({len(val_metrics['param_mae'])} 个参数总计)")

        # TensorBoard 日志
        writer.add_scalar('loss/train', train_loss, epoch)
        writer.add_scalar('loss/val', val_metrics['loss'], epoch)
        writer.add_scalar('metrics/val_mae', val_metrics['mae'], epoch)
        writer.add_scalar('metrics/val_rmse', val_metrics['rmse'], epoch)
        writer.add_scalar('metrics/val_r2', val_metrics['r2_mean'], epoch)

        # 学习率调度
        if trainer.scheduler:
            trainer.scheduler.step()
            current_lr = trainer.optimizer.param_groups[0]['lr']
            logger.info(f"学习率: {current_lr:.2e}")

        # 保存最佳模型
        if val_metrics['r2_mean'] > best_val_r2:
            best_val_r2 = val_metrics['r2_mean']
            patience_counter = 0
            ckpt_path = os.path.join(
                args.output_dir,
                f'best_model_epoch{epoch:03d}_r2{best_val_r2:.4f}.pt'
            )
            trainer.save_checkpoint(ckpt_path, epoch, val_metrics)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"早停（无进展 {patience} 个 epoch）")
                break

        writer.flush()

    # 测试集评估
    logger.info("\n=== 测试集评估 ===")
    test_metrics = trainer.evaluate(test_loader)
    logger.info(
        f"测试损失: {test_metrics['loss']:.6f}, "
        f"MAE: {test_metrics['mae']:.4f}, "
        f"RMSE: {test_metrics['rmse']:.4f}, "
        f"R²: {test_metrics['r2_mean']:.4f}"
    )

    # 保存测试结果
    results_file = os.path.join(args.output_dir, 'test_results.json')
    with open(results_file, 'w') as f:
        json.dump({
            'test_metrics': test_metrics,
            'best_val_r2': float(best_val_r2),
            'args': vars(args),
        }, f, indent=2)
    logger.info(f"结果已保存到 {results_file}")

    writer.close()
    logger.info("训练完成！")


if __name__ == '__main__':
    main()
