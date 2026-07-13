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
from param_normalizer import ParamNormalizer
from lr_image_processor_torch import apply_lr_params_torch

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
        pixel_loss_weight: float = 1.0,
        param_loss_weight: float = 1.0,
    ):
        self.model = model.to(device)
        self.device = device
        self.normalizer = ParamNormalizer()

        # 双损失：参数空间 MSE + 像素空间 MSE
        # 像素 loss 让模型学到"预测参数应用后真正还原图像"
        self.criterion = nn.MSELoss()
        self.pixel_w = pixel_loss_weight
        self.param_w = param_loss_weight

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

    def set_stage(self, mask, pixel_w: float):
        """设置课程阶段：参数掩码 + 像素 loss 权重 + 每参数权重"""
        self.param_mask = torch.tensor(mask, device=self.device, dtype=torch.float32)
        self.pixel_w = pixel_w
        # 每参数 loss 权重：亮度 1.0，其余创作类（曲线/颜色分级/校准/HSL）1.6
        from params_config import PARAM_ORDER, GROUP_LUMINANCE
        w = [1.0 if p in GROUP_LUMINANCE else 1.6 for p in PARAM_ORDER]
        self.param_weight = torch.tensor(w, device=self.device, dtype=torch.float32)

    def _masked_param_loss(self, pred, target):
        """只在激活参数上计算 MSE（带每参数权重）"""
        if not hasattr(self, 'param_mask'):
            return self.criterion(pred, target)
        w = self.param_mask * getattr(self, 'param_weight', 1.0)
        se = (pred - target) ** 2 * w
        return se.sum() / (w.sum() * pred.shape[0] + 1e-8)

    def train_epoch(self, train_loader) -> dict:
        """训练一个 epoch，返回各损失分量"""
        self.model.train()
        total_loss = total_param = total_pixel = 0.0

        for batch_idx, batch in enumerate(train_loader):
            src = batch['src'].to(self.device)
            ref = batch['ref'].to(self.device)
            params = batch['params'].to(self.device)

            # 前向传播
            pred_params = self.model(src, ref)

            # 参数空间损失（掩码到当前 stage 激活参数）
            loss_param = self._masked_param_loss(pred_params, params)

            # 像素空间损失：用预测参数渲染 src，对比 ref
            if self.pixel_w > 0:
                rendered = apply_lr_params_torch(src, pred_params)
                loss_pixel = self.criterion(rendered, ref)
            else:
                loss_pixel = torch.zeros(1, device=self.device)

            loss = self.param_w * loss_param + self.pixel_w * loss_pixel

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss  += loss.item()
            total_param += loss_param.item()
            total_pixel += loss_pixel.item()

            if (batch_idx + 1) % 50 == 0:
                logger.info(
                    f"  Batch {batch_idx + 1}/{len(train_loader)}, "
                    f"Loss: {loss.item():.6f} "
                    f"(param: {loss_param.item():.4f}, pixel: {loss_pixel.item():.4f})"
                )

        n = len(train_loader)
        return {
            'loss':       total_loss / n,
            'param_loss': total_param / n,
            'pixel_loss': total_pixel / n,
        }

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

        # 反归一化到原始参数空间，用于指标计算
        all_preds_denorm = self.normalizer.denormalize_array(all_preds)
        all_targets_denorm = self.normalizer.denormalize_array(all_targets)

        # 计算 MAE、RMSE、R² 等指标（在原始参数空间）
        mae = np.mean(np.abs(all_preds_denorm - all_targets_denorm))
        rmse = np.sqrt(np.mean((all_preds_denorm - all_targets_denorm) ** 2))

        # R² 分数（针对每个参数）
        ss_res = np.sum((all_targets_denorm - all_preds_denorm) ** 2, axis=0)
        ss_tot = np.sum((all_targets_denorm - np.mean(all_targets_denorm, axis=0)) ** 2, axis=0)
        r2_scores = 1 - (ss_res / (ss_tot + 1e-10))

        # R² 均值只在当前 stage 激活参数上计算（更有意义）
        from params_config import PARAM_ORDER
        param_names = list(PARAM_ORDER)
        if hasattr(self, 'param_mask'):
            mask_np = self.param_mask.cpu().numpy().astype(bool)
            r2_mean = float(np.mean(r2_scores[mask_np])) if mask_np.any() else 0.0
        else:
            r2_mean = float(np.mean(r2_scores))

        # 按参数 MAE（使用反归一化值，与全局 MAE 一致）
        param_mae = {
            name: float(np.mean(np.abs(all_preds_denorm[:, i] - all_targets_denorm[:, i])))
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
    parser.add_argument('--backbone', type=str, default='simple_color',
                        choices=['simple_color', 'resnet18', 'resnet34',
                                 'convnext_tiny', 'convnext_small'],
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
    parser.add_argument('--pixel-loss-weight', type=float, default=0.3,
                        help='像素重构损失权重（仅最后 stage 生效）')
    parser.add_argument('--param-loss-weight', type=float, default=1.0,
                        help='参数 MSE 损失权重')
    parser.add_argument('--stage-epochs', type=str, default='50,50,50,30,30',
                        help='5 个课程阶段各自的 epoch 数（逗号分隔）')
    parser.add_argument('--joint', action='store_true',
                        help='联合训练模式：全 72 维一起训 --epochs 轮（不用课程掩码）')

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
    trainer = Trainer(
        model, device,
        lr=args.lr, weight_decay=args.weight_decay,
        pixel_loss_weight=args.pixel_loss_weight,
        param_loss_weight=args.param_loss_weight,
    )
    logger.info(f"损失权重: param={args.param_loss_weight}, pixel={args.pixel_loss_weight}")
    trainer.set_scheduler(args.epochs)

    # TensorBoard 日志
    log_dir = os.path.join(args.output_dir, 'logs',
                          datetime.now().strftime('%Y%m%d_%H%M%S'))
    writer = SummaryWriter(log_dir)
    logger.info(f"日志保存到: {log_dir}")

    # 恢复训练（如果指定）
    if args.resume:
        logger.info(f"从检查点恢复: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # ─── 课程学习 / 联合训练 ─────────────────────────────────────────────
    from params_config import CURRICULUM_STAGES as _STAGES, stage_mask as _smask, PARAM_ORDER

    if args.joint:
        # 联合训练：单阶段全 72 维 + 像素 loss（避免课程掩码饿死后期参数）
        CURRICULUM_STAGES = [('joint_all72', PARAM_ORDER)]
        stage_epochs = [args.epochs]
        def stage_mask(i):
            return [1.0] * len(PARAM_ORDER)
        logger.info(f"🔀 联合训练模式：全 {len(PARAM_ORDER)} 维，{args.epochs} epoch")
    else:
        CURRICULUM_STAGES = _STAGES
        stage_mask = _smask
        stage_epochs = [int(x) for x in args.stage_epochs.split(',')]
        assert len(stage_epochs) == len(CURRICULUM_STAGES), \
            f"stage_epochs 需 {len(CURRICULUM_STAGES)} 个值"

    patience = 15
    global_epoch = 0
    best_overall_r2 = -np.inf
    best_ckpt_path = None

    for si, (stage_name, stage_params) in enumerate(CURRICULUM_STAGES):
        mask = stage_mask(si)
        is_last = (si == len(CURRICULUM_STAGES) - 1)
        pixel_w = args.pixel_loss_weight if is_last else 0.0
        trainer.set_stage(mask, pixel_w)
        trainer.set_scheduler(stage_epochs[si])

        logger.info(f"\n{'='*60}")
        logger.info(f"📚 Stage {si+1}/{len(CURRICULUM_STAGES)}: {stage_name}")
        logger.info(f"   激活参数: {len(stage_params)} 维  |  像素 loss: {pixel_w}")
        logger.info(f"{'='*60}")

        best_stage_r2 = -np.inf
        patience_counter = 0

        for e in range(stage_epochs[si]):
            logger.info(f"\n--- [{stage_name}] Epoch {e+1}/{stage_epochs[si]} "
                        f"(全局 {global_epoch+1}) ---")
            train_losses = trainer.train_epoch(train_loader)
            logger.info(
                f"训练损失: {train_losses['loss']:.6f} "
                f"(param: {train_losses['param_loss']:.4f}, "
                f"pixel: {train_losses['pixel_loss']:.4f})"
            )
            val_metrics = trainer.evaluate(val_loader)
            logger.info(
                f"验证 MAE: {val_metrics['mae']:.4f}, "
                f"R²(激活): {val_metrics['r2_mean']:.4f}"
            )

            writer.add_scalar(f'{stage_name}/train_param', train_losses['param_loss'], e)
            writer.add_scalar(f'{stage_name}/train_pixel', train_losses['pixel_loss'], e)
            writer.add_scalar(f'{stage_name}/val_r2', val_metrics['r2_mean'], e)

            if trainer.scheduler:
                trainer.scheduler.step()

            # stage 内早停 + 保存全局最佳
            r2 = val_metrics['r2_mean']
            if r2 > best_stage_r2:
                best_stage_r2 = r2
                patience_counter = 0
            else:
                patience_counter += 1

            if r2 > best_overall_r2 or is_last:
                # 最后 stage 总是保存（覆盖为全 72 维训练的权重）
                if r2 > best_overall_r2:
                    best_overall_r2 = r2
                ckpt_path = os.path.join(
                    args.output_dir,
                    f'best_model_{stage_name}_r2{r2:.4f}.pt')
                trainer.save_checkpoint(ckpt_path, global_epoch, val_metrics)
                if is_last:
                    best_ckpt_path = ckpt_path

            global_epoch += 1
            writer.flush()

            if patience_counter >= patience:
                logger.info(f"  [{stage_name}] 早停（{patience} epoch 无进展）")
                break

        logger.info(f"✓ Stage {si+1} 完成，最佳 R²(激活)={best_stage_r2:.4f}")

    # ─── 测试集评估（全 72 维）────────────────────────────────────────
    logger.info("\n=== 测试集评估（全部 72 参数）===")
    trainer.set_stage([1.0] * len(PARAM_ORDER), 0.0)
    test_metrics = trainer.evaluate(test_loader)
    best_val_r2 = best_overall_r2
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
