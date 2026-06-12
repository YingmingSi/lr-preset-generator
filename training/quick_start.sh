#!/bin/bash

#################################################################
# LR Preset Generator - 快速启动脚本
#
# 这是最简单的启动方式，自动执行完整训练流程
# 使用: ./quick_start.sh
#
#################################################################

cd "$(dirname "$0")"

echo "╔════════════════════════════════════════════╗"
echo "║   LR Preset - 快速启动训练                 ║"
echo "╚════════════════════════════════════════════╝"
echo ""

# 检查照片
PHOTO_COUNT=$(find ./photos -type f 2>/dev/null | wc -l)
if [ $PHOTO_COUNT -lt 100 ]; then
    echo "❌ 错误: 照片不足 ($PHOTO_COUNT 张，需要 ≥ 100 张)"
    echo ""
    echo "请先将照片放入 training/photos/ 目录"
    exit 1
fi

echo "✓ 找到 $PHOTO_COUNT 张照片"
echo ""

# 询问用户选择
echo "选择一个选项:"
echo ""
echo "  1️⃣  完整流程 (4 步，2-10h GPU / 28-58h CPU)"
echo "  2️⃣  仅生成测试数据 (5 分钟)"
echo "  3️⃣  仅生成正式数据 (1-5 小时)"
echo "  4️⃣  仅验证训练 (5-30 分钟)"
echo "  5️⃣  仅正式训练 (2-24 小时)"
echo ""
echo "  U) 使用 GPU (默认)"
echo "  C) 使用 CPU"
echo ""

read -p "选择 (默认 1): " choice
choice=${choice:-1}

# 默认使用 GPU
DEVICE="cuda"

case $choice in
    1)
        echo ""
        echo "开始完整训练流程..."
        echo "设备: $DEVICE"
        ./train_full.sh --device $DEVICE
        ;;
    2)
        echo ""
        echo "生成测试数据..."
        ./train_full.sh --device $DEVICE --step 1
        ;;
    3)
        echo ""
        echo "生成正式数据..."
        ./train_full.sh --device $DEVICE --step 2
        ;;
    4)
        echo ""
        echo "验证训练脚本..."
        ./train_full.sh --device $DEVICE --step 3
        ;;
    5)
        echo ""
        echo "正式训练..."
        ./train_full.sh --device $DEVICE --step 4
        ;;
    U)
        echo "使用 GPU"
        ./quick_start.sh
        ;;
    C)
        echo "使用 CPU"
        DEVICE="cpu"
        ./quick_start.sh
        ;;
    *)
        echo "❌ 无效选择"
        exit 1
        ;;
esac
